# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool implementations for the cuOpt MCP server.

Every solve is asynchronous: submitting returns a ``job_id`` and nothing
blocks. A blocking call inside a single ``tools/call`` would exceed the
client timeout on any realistic MILP, and would make cancellation
impossible.

No per-job state is kept here. Column names needed to label a solution are
supplied per call via ``names_from``, so any process — a second editor
window, or this one after a restart — can retrieve a named result for a job
it did not submit.
"""

import json
import os
import tempfile
from pathlib import Path

from .client import (
    CuOptMCPError,
    describe_connection_error,
    endpoint,
    get_client,
    reset_client,
    tls_enabled,
)
from .schema import known_parameters, settings_schema, validate_settings

# Above this many variables a solution is written to a file instead of
# returned inline. The binding limit is the model's context window, not the
# transport: ~200 values is already a large tool result, and cuOpt problems
# routinely have millions.
INLINE_SOLUTION_LIMIT = 200

# Magnitude below which a solution value is treated as zero by nonzero_only.
ZERO_TOL = 1e-9

# At or beyond this magnitude a caller-supplied bound means infinity.
INFINITY_SENTINEL = 1e30

# A job id no server can have issued. The gRPC service exposes no health or
# version RPC, so reachability is probed with the cheapest call that still
# requires a server to answer: a status lookup that must come back NOT_FOUND.
PROBE_JOB_ID = "00000000-0000-0000-0000-000000000000"


def _solution_dir() -> Path:
    path = Path(
        os.environ.get(
            "CUOPT_MCP_SOLUTION_DIR", Path(tempfile.gettempdir()) / "cuopt-mcp"
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_problem(path: str):
    from cuopt.linear_programming import Read

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise CuOptMCPError(f"problem file not found: {resolved}")
    try:
        return Read(str(resolved))
    except Exception as exc:
        raise CuOptMCPError(f"failed to parse {resolved}: {exc}") from exc


def _require(problem: dict, key: str):
    if key not in problem:
        raise CuOptMCPError(f"problem is missing required key {key!r}")
    return problem[key]


def _merge_duplicate_cells(rows, cols, values):
    """Sum COO entries that name the same cell.

    Building a row incrementally naturally emits a cell twice (``2*x`` after
    collecting ``x`` from two terms). Passing both through would leave the
    row's meaning dependent on how the backend treats repeated indices, so
    they are summed here where the intent is unambiguous.
    """
    import numpy as np

    if len(rows) == 0:
        return rows, cols, values
    starts = np.empty(len(rows), dtype=bool)
    starts[0] = True
    starts[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    if starts.all():
        return rows, cols, values
    group = np.cumsum(starts) - 1
    merged = np.zeros(int(group[-1]) + 1, dtype=np.float64)
    np.add.at(merged, group, values)
    return rows[starts], cols[starts], merged


def _to_csr(matrix: dict, n_vars: int, n_cons: int | None = None):
    """Accept either CSR or COO triplets and return CSR arrays.

    COO is what a caller naturally builds when emitting a model row by row,
    so taking it directly removes the most error-prone step of the handoff.

    n_cons pins the row count. Without it the count is inferred from the
    largest row index present, which silently loses a trailing row whose
    coefficients are all zero.
    """
    import numpy as np

    if "offsets" in matrix:
        offsets = np.asarray(matrix["offsets"], dtype=np.int32)
        indices = np.asarray(matrix["indices"], dtype=np.int32)
        values = np.asarray(matrix["values"], dtype=np.float64)
        if len(indices) != len(values):
            raise CuOptMCPError(
                f"constraint_matrix indices ({len(indices)}) and values "
                f"({len(values)}) must have equal length"
            )
        if n_cons is not None and len(offsets) - 1 != n_cons:
            raise CuOptMCPError(
                f"constraint_matrix has {len(offsets) - 1} rows but "
                f"{n_cons} constraint bounds were given"
            )
        return offsets, indices, values

    rows = np.asarray(matrix.get("rows", []), dtype=np.int64)
    cols = np.asarray(matrix.get("cols", []), dtype=np.int64)
    values = np.asarray(matrix.get("values", []), dtype=np.float64)
    if not (len(rows) == len(cols) == len(values)):
        raise CuOptMCPError(
            f"constraint_matrix rows/cols/values must have equal length, got "
            f"{len(rows)}/{len(cols)}/{len(values)}"
        )
    if len(cols) and int(cols.max()) >= n_vars:
        raise CuOptMCPError(
            f"constraint_matrix references column {int(cols.max())} but the "
            f"objective declares only {n_vars} variables"
        )
    inferred = int(rows.max()) + 1 if len(rows) else 0
    if n_cons is None:
        n_cons = inferred
    elif inferred > n_cons:
        raise CuOptMCPError(
            f"constraint_matrix references row {inferred - 1} but only "
            f"{n_cons} constraint bounds were given"
        )
    order = np.lexsort((cols, rows))
    rows, cols, values = _merge_duplicate_cells(
        rows[order], cols[order], values[order]
    )
    counts = np.bincount(rows, minlength=n_cons).astype(np.int32)
    offsets = np.zeros(n_cons + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    return offsets, cols.astype(np.int32), values


def _build_model_from_json(problem: dict):
    """Build a DataModel from plain arrays, with no file in the loop.

    Integrality is declared as a type vector rather than MPS INTORG/INTEND
    markers, so integer columns keep the bounds given here instead of
    silently defaulting to [0, 1].
    """
    import numpy as np

    from cuopt.linear_programming import DataModel

    if not isinstance(problem, dict):
        raise CuOptMCPError("problem must be an object")

    objective = np.asarray(_require(problem, "objective"), dtype=np.float64)
    n_vars = len(objective)

    # Prefer a row count the caller stated over one guessed from the largest
    # row index, so a trailing all-zero row is not silently dropped and a
    # genuine mismatch is reported against the matrix rather than the bounds.
    lengths = {
        key: len(problem[key])
        for key in ("constraint_lower_bounds", "constraint_upper_bounds")
        if problem.get(key) is not None
    }
    if len(set(lengths.values())) > 1:
        raise CuOptMCPError(
            "constraint_lower_bounds and constraint_upper_bounds must have "
            f"the same length, got {lengths}"
        )
    declared = problem.get("n_constraints")
    if declared is None and lengths:
        declared = next(iter(lengths.values()))

    offsets, indices, values = _to_csr(
        _require(problem, "constraint_matrix"), n_vars, declared
    )
    n_cons = max(len(offsets) - 1, 0)

    def vec(key, default, size, dtype=np.float64):
        raw = problem.get(key)
        if raw is None:
            return np.full(size, default, dtype=dtype)
        # JSON has no infinity literal, so null means "unbounded on this
        # side" and is the only way a caller can express a one-sided row.
        arr = np.asarray(
            [default if x is None else x for x in raw], dtype=dtype
        )
        # Callers routinely spell infinity as a large sentinel (1e30 is the
        # MPS-era convention). Left finite, such a bound is not merely loose
        # — cuOpt can return a constraint-violating point reported as
        # Optimal — so normalise it to a true infinity.
        if dtype is np.float64:
            arr = np.where(arr >= INFINITY_SENTINEL, np.inf, arr)
            arr = np.where(arr <= -INFINITY_SENTINEL, -np.inf, arr)
        if len(arr) != size:
            raise CuOptMCPError(
                f"{key} has length {len(arr)}, expected {size}"
            )
        return arr

    model = DataModel()
    model.set_csr_constraint_matrix(values, indices, offsets)
    model.set_objective_coefficients(objective)
    model.set_constraint_lower_bounds(
        vec("constraint_lower_bounds", -np.inf, n_cons)
    )
    model.set_constraint_upper_bounds(
        vec("constraint_upper_bounds", np.inf, n_cons)
    )
    model.set_variable_lower_bounds(vec("variable_lower_bounds", 0.0, n_vars))
    model.set_variable_upper_bounds(
        vec("variable_upper_bounds", np.inf, n_vars)
    )
    model.set_maximize(bool(problem.get("maximize", False)))
    if problem.get("objective_offset"):
        model.set_objective_offset(float(problem["objective_offset"]))
    if problem.get("problem_name"):
        model.set_problem_name(str(problem["problem_name"]))

    types = problem.get("variable_types")
    if types is not None:
        if len(types) != n_vars:
            raise CuOptMCPError(
                f"variable_types has length {len(types)}, expected {n_vars}"
            )
        allowed = {"C", "I"}
        bad = sorted({str(t).upper() for t in types} - allowed)
        if bad:
            raise CuOptMCPError(
                f"variable_types entries must be 'C' or 'I', got {bad}"
            )
        model.set_variable_types(
            np.asarray([str(t).upper() for t in types], dtype="<U1")
        )

    names = problem.get("variable_names")
    if names is not None:
        if len(names) != n_vars:
            raise CuOptMCPError(
                f"variable_names has length {len(names)}, expected {n_vars}"
            )
        model.set_variable_names(np.asarray([str(v) for v in names]))
    return model


def _build_settings(kind: str, settings: dict | None):
    from cuopt.linear_programming import SolverSettings

    validate_settings(kind, settings or {})
    properties = settings_schema(kind)["properties"]
    solver_settings = SolverSettings()
    for name, value in (settings or {}).items():
        # Enum settings are exposed to callers by name ("Barrier") because a
        # bare integer is meaningless to an agent, but cuOpt's string
        # parameter interface takes the integer. The mapping is generated
        # from the field registry alongside the enum itself.
        prop = properties[name]
        mapping = prop.get("x-enum-values")
        if mapping is not None:
            value = mapping[value]
        # The proto field name is not always the CUOPT_* parameter name.
        solver_settings.set_parameter(
            prop.get("x-parameter-name", name), value
        )
    return solver_settings


def _variable_names(names_from: str | None):
    if not names_from:
        return None
    # A JSON submission has no problem file to re-parse, so its names are
    # kept in a sidecar written at submit time. Same contract either way:
    # pass back the "source" the solve returned.
    resolved = Path(names_from).expanduser()
    if resolved.suffix == ".json" and resolved.is_file():
        return list(json.loads(resolved.read_text()))
    model = _read_problem(names_from)
    names = model.get_variable_names()
    return list(names) if names is not None else None


def _write_names_file(job_id: str, names) -> str:
    path = _solution_dir() / f"{job_id}.names.json"
    path.write_text(json.dumps([str(v) for v in names]))
    return str(path)


def health() -> dict:
    """Report where this server is pointed and whether that target answers.

    Every other tool needs a model or a job_id, so without this there is no
    way to check the connection except by submitting work and reading the
    failure — by which point a caller has already built a model, and may
    conclude from the error that no server is running anywhere.
    """
    host, port = endpoint()
    info = {"host": host, "port": port, "tls": tls_enabled()}
    try:
        get_client().status(PROBE_JOB_ID)
    except Exception as exc:
        # The cached channel is process-wide and survives the failure, so a
        # dead one would keep failing every later call. Drop it here and the
        # next call redials.
        reset_client()
        return {
            **info,
            "reachable": False,
            "error": str(describe_connection_error(exc)),
        }
    return {
        **info,
        "reachable": True,
        "note": "This server does not start or stop cuopt_grpc_server; it "
        "only holds a channel to one.",
    }


def submit(
    kind: str,
    problem_path: str | None = None,
    problem: dict | None = None,
    settings: dict | None = None,
) -> dict:
    """Submit a model given either as a file path or as plain JSON arrays."""
    if (problem_path is None) == (problem is None):
        raise CuOptMCPError(
            "pass exactly one of problem_path (an MPS/QPS/LP file) or "
            "problem (a JSON model object)"
        )
    model = (
        _read_problem(problem_path)
        if problem_path is not None
        else _build_model_from_json(problem)
    )
    solver_settings = _build_settings(kind, settings)
    try:
        job_id = get_client().submit(model, solver_settings)
    except Exception as exc:
        raise describe_connection_error(exc) from exc

    # DataModel exposes no public size accessors, so derive both from the
    # arrays it does expose: one lower bound per column, and CSR row offsets
    # numbering rows + 1.
    offsets = model.get_constraint_matrix_offsets()
    if problem_path is not None:
        source = str(Path(problem_path).expanduser())
    else:
        names = problem.get("variable_names")
        source = _write_names_file(job_id, names) if names else None
    return {
        "job_id": job_id,
        "source": source,
        "num_variables": int(len(model.get_variable_lower_bounds())),
        "num_constraints": int(max(len(offsets) - 1, 0)),
        "next": (
            "Poll cuopt_status(job_id). When it reports COMPLETED, call "
            "cuopt_result(job_id, names_from=source) for a named solution."
            if source
            else "Poll cuopt_status(job_id). When it reports COMPLETED, call "
            "cuopt_result(job_id). Values will be keyed by column index; "
            "pass variable_names in the problem to label them."
        ),
    }


def status(job_id: str) -> dict:
    try:
        state = get_client().status(job_id)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    return {
        "job_id": job_id,
        "status": state.name,
        "terminal": state.name
        in ("COMPLETED", "FAILED", "CANCELLED", "NOT_FOUND"),
    }


def _write_solution_file(job_id: str, vars_by_name: dict) -> str:
    path = _solution_dir() / f"{job_id}.json"
    path.write_text(json.dumps(vars_by_name, indent=1))
    return str(path)


def result(
    job_id: str,
    names_from: str | None = None,
    variables: list | None = None,
    nonzero_only: bool = False,
    limit: int = INLINE_SOLUTION_LIMIT,
) -> dict:
    """Fetch a completed solution, shaped to stay within a usable size."""
    try:
        solution = get_client().result(job_id, _variable_names(names_from))
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    if solution is None:
        return {
            "job_id": job_id,
            "ready": False,
            "hint": "Job has not finished. Poll cuopt_status(job_id).",
        }

    primal = solution.get_primal_solution()
    # The status is an IntEnum, so str() would yield the bare number ("1").
    # get_termination_reason() is its .name, which is what a caller can act
    # on. Note LPTerminationStatus numbers Optimal=1 while the wire enum
    # pdlp_termination_status numbers it 2 — never map between them.
    status_enum = solution.get_termination_status()
    summary = {
        "job_id": job_id,
        "ready": True,
        "termination_status": getattr(status_enum, "name", str(status_enum)),
        "termination_status_code": int(status_enum),
        "primal_objective": float(solution.get_primal_objective()),
        "solve_time_s": float(solution.get_solve_time()),
        "num_variables": int(len(primal)),
    }

    vars_by_name = solution.get_vars()
    if not vars_by_name:
        vars_by_name = {str(i): float(v) for i, v in enumerate(primal)}
        if not names_from:
            summary["names"] = (
                "Values are keyed by column index. Pass names_from=<problem "
                "path> to key them by variable name."
            )

    if variables:
        missing = [v for v in variables if v not in vars_by_name]
        summary["variables"] = {
            v: float(vars_by_name[v]) for v in variables if v in vars_by_name
        }
        if missing:
            summary["missing_variables"] = missing
        return summary

    selected = vars_by_name
    if nonzero_only:
        # PDLP is first-order, so an exact != 0 test lets numerical dust
        # (values around 1e-13, sometimes negative on a variable bounded
        # below by 0) through as if it were signal.
        selected = {k: v for k, v in vars_by_name.items() if abs(v) > ZERO_TOL}
        summary["num_nonzero"] = len(selected)

    if len(selected) <= limit:
        summary["variables"] = {k: float(v) for k, v in selected.items()}
    else:
        summary["variables_truncated"] = True
        summary["variables_shown"] = limit
        summary["variables"] = {
            k: float(v) for k, v in list(selected.items())[:limit]
        }
        summary["solution_path"] = _write_solution_file(job_id, vars_by_name)
        summary["hint"] = (
            f"{len(selected)} values exceed the inline limit of {limit}. The "
            "full solution is at solution_path; use variables=[...] or "
            "nonzero_only=true to narrow the result."
        )
    return summary


def cancel(job_id: str) -> dict:
    try:
        get_client().cancel(job_id)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    return {"job_id": job_id, "cancelled": True}


def incumbents(job_id: str, from_index: int = 0) -> dict:
    """Return the MILP incumbent trajectory so far.

    Lets a caller watch the objective improve and stop a run that has
    plateaued, rather than waiting out the full time limit.
    """
    try:
        found = get_client().incumbents(job_id, from_index)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    objectives = [
        {"index": from_index + i, "objective": float(obj)}
        for i, (obj, _) in enumerate(found or [])
    ]
    return {
        "job_id": job_id,
        "count": len(objectives),
        "next_index": from_index + len(objectives),
        "incumbents": objectives,
    }


def logs(job_id: str, from_byte: int = 0, tail_lines: int = 100) -> dict:
    try:
        text = get_client().logs(job_id, from_byte)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    lines = (text or "").splitlines()
    truncated = len(lines) > tail_lines
    return {
        "job_id": job_id,
        "truncated": truncated,
        "lines": lines[-tail_lines:],
        "next_byte": from_byte + len(text or ""),
    }


def list_settings(kind: str, name: str | None = None) -> dict:
    """Describe available solver settings, from the generated schema."""
    if kind not in ("pdlp_settings", "mip_settings"):
        raise CuOptMCPError(
            "kind must be 'pdlp_settings' (LP) or 'mip_settings' (MILP)"
        )
    schema = settings_schema(kind)
    if name:
        if name not in schema["properties"]:
            raise CuOptMCPError(
                f"unknown {kind} parameter {name!r}. "
                f"Known: {sorted(known_parameters(kind))}"
            )
        return {"kind": kind, "name": name, **schema["properties"][name]}
    return {"kind": kind, "parameters": sorted(schema["properties"])}
