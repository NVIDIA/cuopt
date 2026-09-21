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
import re
import stat
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
# returned inline -- the binding limit is the model's context window.
INLINE_SOLUTION_LIMIT = 200

# Upper bound on logs()'s tail_lines, same context-window reasoning.
MAX_TAIL_LINES = 2000

# Magnitude below which a solution value is treated as zero by nonzero_only.
ZERO_TOL = 1e-9

# At or beyond this magnitude a caller-supplied bound means infinity.
INFINITY_SENTINEL = 1e30

# Largest row/column count _to_csr will allocate for -- n_constraints is a
# tiny-payload scalar that sizes an allocation directly.
MAX_PROBLEM_DIMENSION = 10_000_000

# No health RPC exists, so reachability is probed via a status lookup for
# a job id no server can have issued -- must come back NOT_FOUND.
PROBE_JOB_ID = "00000000-0000-0000-0000-000000000000"


def _check_non_negative_int(name: str, value) -> None:
    """Reject a bool (an int subclass in Python) or a negative value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CuOptMCPError(f"{name} must be a non-negative integer")


def _check_dimension(name: str, value) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_PROBLEM_DIMENSION
    ):
        raise CuOptMCPError(
            f"{name} ({value!r}) must be an integer between 0 and "
            f"{MAX_PROBLEM_DIMENSION}"
        )


def _solution_dir() -> Path:
    """Return the directory solution/name files are written to, private to
    this user.

    Defaults under the shared system temp dir. ``mkdir``'s ``mode`` is
    filtered by umask, so it alone can't guarantee 0700 -- ``chmod`` after
    creating makes that unconditional. If the directory already exists
    (e.g. from a prior run, or planted by another local user first), it's
    only trusted when it's already private to this user; otherwise another
    local user could read or tamper with solution files (CWE-377). The
    check uses ``lstat`` and rejects a symlink outright: ``stat`` follows
    the link and would report the *target's* ownership/mode, so a symlink
    planted here pointing at some other 0700 directory this user happens to
    own elsewhere would pass a ``stat``-based check (CWE-59) and redirect
    solution writes there. Also rejects an existing non-directory (e.g. a
    stray file at this path): silently accepting one here would surface
    later as an unguarded ``NotADirectoryError`` from
    :func:`_write_solution_file`.
    """
    path = Path(
        os.environ.get(
            "CUOPT_MCP_SOLUTION_DIR", Path(tempfile.gettempdir()) / "cuopt-mcp"
        )
    )
    try:
        path.mkdir(parents=True, mode=0o700)
    except FileExistsError:
        st = path.lstat()
        if (
            not stat.S_ISDIR(st.st_mode)
            or st.st_uid != os.getuid()
            or stat.S_IMODE(st.st_mode) & 0o077
        ):
            raise CuOptMCPError(
                f"{path} exists but is not a private directory owned by "
                f"this user (mode {oct(stat.S_IMODE(st.st_mode))}, owner "
                f"uid {st.st_uid}) -- refusing to write solution files "
                "there. Remove it or set CUOPT_MCP_SOLUTION_DIR to a "
                "private location."
            ) from None
    else:
        os.chmod(path, 0o700)
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

    _check_dimension("n_vars", n_vars)
    if n_cons is not None:
        _check_dimension("n_constraints", n_cons)

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
        _check_dimension("constraint_matrix row count", inferred)
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

    # Prefer a caller-stated row count: inferring it from the largest row
    # index would silently drop a trailing all-zero row.
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
        # 1e30 (MPS-era convention) means infinity too; left finite, such a
        # bound can make cuOpt report a constraint-violating point Optimal.
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
    if kind not in ("pdlp_settings", "mip_settings"):
        raise CuOptMCPError(
            "kind must be 'pdlp_settings' (LP) or 'mip_settings' (MILP)"
        )
    try:
        validate_settings(kind, settings or {})
    except ValueError as exc:
        raise CuOptMCPError(str(exc)) from exc
    properties = settings_schema(kind)["properties"]

    from cuopt.linear_programming import SolverSettings

    solver_settings = SolverSettings()
    for name, value in (settings or {}).items():
        # Enums are exposed by name ("Barrier"); set_parameter takes the int.
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
    # A JSON submission has no file to re-parse; names live in a sidecar
    # written at submit time instead. Pass back the "source" it returned.
    resolved = Path(names_from).expanduser()
    if resolved.suffix == ".json" and resolved.is_file():
        return list(json.loads(resolved.read_text()))
    model = _read_problem(names_from)
    names = model.get_variable_names()
    return list(names) if names is not None else None


def _write_names_file(job_id: str, names) -> str:
    # job_id is the backend's response, not literal caller input -- nothing
    # guarantees it's a well-formed UUID, so validate before it's a path.
    path = _solution_file_path(job_id).with_suffix(".names.json")
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
        # A dead cached channel would keep failing every later call.
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
    track_incumbents: bool = False,
) -> dict:
    """Submit a model, given either as a file path or plain JSON arrays, for
    an asynchronous solve.

    Args:
        kind: "pdlp_settings" for LP or "mip_settings" for MILP; selects
            which settings schema ``settings`` is validated against.
        problem_path: Path to an MPS/QPS/LP file readable by this process.
            Exactly one of ``problem_path``/``problem`` must be given.
        problem: The model as plain JSON arrays, with no file in the loop.
        settings: Solver settings by name, or ``None`` to use cuOpt
            defaults for all of them.
        track_incumbents: For a MIP job, collect incumbents server-side so
            :func:`incumbents` has something to poll. Off by default:
            collection downloads each incumbent's full variable vector
            server-side even though only its objective is kept, which adds
            up for a large model with many incumbents.

    Returns
    -------
        A dict with ``job_id``, ``source`` (the resolved problem path, for
        ``cuopt_result``'s ``names_from``), and the problem's size.

    Raises
    ------
        CuOptMCPError: ``kind`` is invalid, neither or both of
            ``problem_path``/``problem`` were given, the input doesn't
            exist/parse/validate, or the backend is unreachable.
    """
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
        job_id = get_client().submit(
            model,
            solver_settings,
            enable_incumbents=(kind == "mip_settings" and track_incumbents),
        )
    except Exception as exc:
        raise describe_connection_error(exc) from exc

    # DataModel exposes no public size accessors; derive from CSR offsets.
    offsets = model.get_constraint_matrix_offsets()
    if problem_path is not None:
        source = str(Path(problem_path).expanduser().resolve())
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
    """Report whether a submitted job is queued, running, or finished.

    Args:
        job_id: A job handle previously returned by :func:`submit`.

    Returns
    -------
        A dict with ``status`` (the raw state name) and ``terminal``
        (whether ``status`` is one that will never change again).

    Raises
    ------
        CuOptMCPError: The backend is unreachable.
    """
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


_JOB_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _solution_file_path(job_id: str) -> Path:
    """Return the on-disk path for a job's solution file, after checking
    ``job_id`` looks like a server-issued UUID.

    ``job_id`` is caller-supplied and gets joined onto
    ``CUOPT_MCP_SOLUTION_DIR`` as a filename; without this check, a value
    like ``"../../etc/cron.d/x"`` would let a caller write or unlink an
    arbitrary ``.json``-suffixed path outside that directory.
    """
    if not _JOB_ID_RE.match(job_id):
        raise CuOptMCPError(f"{job_id!r} is not a valid job_id")
    return _solution_dir() / f"{job_id}.json"


def _write_solution_file(job_id: str, vars_by_name: dict) -> str:
    path = _solution_file_path(job_id)
    path.write_text(json.dumps(vars_by_name, indent=1))
    return str(path)


def result(
    job_id: str,
    names_from: str | None = None,
    variables: list | None = None,
    nonzero_only: bool = False,
    limit: int = INLINE_SOLUTION_LIMIT,
) -> dict:
    """Fetch a completed solution, shaped to stay within a usable size.

    Args:
        job_id: A job handle previously returned by :func:`submit`.
        names_from: Path to the problem file, to key ``variables`` by name
            instead of column index — pass back ``submit``'s ``source``.
        variables: Return only these named/indexed variables, skipping the
            inline-size shaping below. An empty list returns no variables
            (distinct from omitting the argument).
        nonzero_only: Drop values with magnitude at most :data:`ZERO_TOL`
            (1e-9) before applying ``limit`` -- a dropped value isn't
            necessarily exact zero, just within solver tolerance of it.
        limit: Maximum variables returned inline; must be between 0 and
            :data:`INLINE_SOLUTION_LIMIT`. Beyond this, the selected values
            (post ``nonzero_only`` filtering) are written to a file and
            ``solution_path`` returned instead.

    Returns
    -------
        ``{"ready": False, ...}`` if the job hasn't finished yet, otherwise
        the termination status, objective, solve time, and the (possibly
        truncated) variable values.

    Raises
    ------
        CuOptMCPError: ``limit`` is out of range, or the backend is
            unreachable.
    """
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 0 <= limit <= INLINE_SOLUTION_LIMIT
    ):
        raise CuOptMCPError(
            f"limit must be an integer between 0 and {INLINE_SOLUTION_LIMIT}"
        )
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
    # IntEnum: use .name, not str(). Its numbering differs from the wire
    # enum pdlp_termination_status -- never map between them.
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

    if variables is not None:
        missing = [v for v in variables if v not in vars_by_name]
        summary["variables"] = {
            v: float(vars_by_name[v]) for v in variables if v in vars_by_name
        }
        if missing:
            summary["missing_variables"] = missing
        return summary

    selected = vars_by_name
    if nonzero_only:
        # Exact != 0 would let PDLP's numerical dust (~1e-13) through as signal.
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
        summary["solution_path"] = _write_solution_file(job_id, selected)
        summary["hint"] = (
            f"{len(selected)} values exceed the inline limit of {limit}. "
            "The values shown above are at solution_path (all of them, "
            "not just the nonzero ones, if nonzero_only wasn't set); use "
            "variables=[...] or nonzero_only=true to narrow the result."
        )
    return summary


def cancel(job_id: str) -> dict:
    """Stop a running job.

    Args:
        job_id: A job handle previously returned by :func:`submit`.

    Returns
    -------
        ``{"job_id": ..., "cancelled": True}`` on success.

    Raises
    ------
        CuOptMCPError: The job has already reached COMPLETED or FAILED (the
            backend rejects cancelling a terminal job), or the backend is
            unreachable.
    """
    try:
        get_client().cancel(job_id)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    return {"job_id": job_id, "cancelled": True}


def delete(job_id: str) -> dict:
    """Release a job's server-side state (solution, logs, incumbents), and
    its local solution file if :func:`result` wrote one.

    cuopt_grpc_server keeps this until deleted, so a caller done with a
    job's result should call this rather than letting it accumulate.
    Cancels first if the job is still running.

    Args:
        job_id: A job handle previously returned by :func:`submit`.

    Returns
    -------
        ``{"job_id": ..., "deleted": True}`` on success.

    Raises
    ------
        CuOptMCPError: ``job_id`` isn't a valid job id, or the backend is
            unreachable.
    """
    if not _JOB_ID_RE.match(job_id):
        raise CuOptMCPError(f"{job_id!r} is not a valid job_id")
    try:
        get_client().delete(job_id)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    try:
        path = _solution_file_path(job_id)
        path.unlink(missing_ok=True)
        path.with_suffix(".names.json").unlink(missing_ok=True)
    except CuOptMCPError:
        pass  # nothing to clean up if the directory itself is unusable
    return {"job_id": job_id, "deleted": True}


def incumbents(job_id: str, from_index: int = 0) -> dict:
    """Return the MILP incumbent trajectory so far.

    Lets a caller watch the objective improve and stop a run that has
    plateaued, rather than waiting out the full time limit.

    Args:
        job_id: A job handle previously returned by :func:`submit`.
        from_index: Skip incumbents before this index — pass back a prior
            call's ``next_index`` to fetch only what's new.

    Returns
    -------
        A dict with the incumbent objectives found since ``from_index`` and
        ``next_index`` for the following call.

    Raises
    ------
        CuOptMCPError: ``from_index`` is negative, or the backend is
            unreachable.
    """
    _check_non_negative_int("from_index", from_index)
    try:
        # Each entry also has "assignment" (full variable vector); omitted
        # here to stay within a usable tool-result size.
        found = get_client().incumbents(job_id, from_index)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    objectives = [
        {"index": entry["index"], "objective": float(entry["objective"])}
        for entry in found or []
    ]
    next_index = objectives[-1]["index"] + 1 if objectives else from_index
    return {
        "job_id": job_id,
        "count": len(objectives),
        "next_index": next_index,
        "incumbents": objectives,
    }


def logs(job_id: str, from_byte: int = 0, tail_lines: int = 100) -> dict:
    """Fetch a job's log lines starting at ``from_byte``, tailed to the last
    ``tail_lines`` lines.

    Solver output is captured to a log for every job, from submission —
    there's nothing to opt into. This tool only returns it once the job
    has finished though: it fetches a snapshot via :meth:`Client.logs`,
    which raises while a job is still queued or running; poll
    ``cuopt_status`` first. A live tail while running is possible on the
    wire (``Client.start_log_stream``) but not exposed by this tool yet.

    For a CANCELLED job specifically, expect an empty or missing log: the
    server deletes a job's log file as part of handling cancellation
    (unlike COMPLETED/FAILED, which keep it until :func:`delete`).

    Args:
        job_id: The job to fetch logs for.
        from_byte: Byte offset to resume from (e.g. ``next_byte`` from a
            prior call), so repeated polling doesn't re-fetch the whole log.
        tail_lines: Keep only the last this many lines of the fetched text;
            must be between 1 and :data:`MAX_TAIL_LINES`. Bounds the
            response, not the fetch: Client.logs() has no server-side tail
            operation, so a multi-GB log is still downloaded and held in
            memory here before being sliced. Use ``from_byte`` to fetch
            incrementally rather than relying on ``tail_lines`` alone for a
            log that large.

    Returns
    -------
        ``{"ready": False, ...}`` if the job hasn't finished yet, otherwise
        a dict with ``lines`` (the tailed text), ``truncated`` (whether more
        preceded them), and ``next_byte`` to pass on the next call.

    Raises
    ------
        CuOptMCPError: ``from_byte`` is negative, ``tail_lines`` is out of
            range, or the backend is unreachable.
    """
    _check_non_negative_int("from_byte", from_byte)
    if (
        isinstance(tail_lines, bool)
        or not isinstance(tail_lines, int)
        or not 1 <= tail_lines <= MAX_TAIL_LINES
    ):
        raise CuOptMCPError(
            f"tail_lines must be an integer between 1 and {MAX_TAIL_LINES}"
        )
    try:
        lines = get_client().logs(job_id, from_byte)
    except Exception as exc:
        # Matched by name, not isinstance, to avoid an eager cuopt import.
        if type(exc).__name__ == "JobNotReadyError":
            return {
                "job_id": job_id,
                "ready": False,
                "hint": "Job has not finished. Poll cuopt_status(job_id).",
            }
        raise describe_connection_error(exc) from exc
    lines = lines or []
    truncated = len(lines) > tail_lines
    # Client.logs() doesn't return the server's byte offset; approximate it
    # from encoded line lengths plus the '\n' each one was split on.
    next_byte = from_byte + sum(
        len(line.encode("utf-8")) + 1 for line in lines
    )
    return {
        "job_id": job_id,
        "ready": True,
        "truncated": truncated,
        "lines": lines[-tail_lines:],
        "next_byte": next_byte,
    }


def list_settings(kind: str, name: str | None = None) -> dict:
    """Describe available solver settings, from the generated schema.

    Args:
        kind: "pdlp_settings" for LP or "mip_settings" for MILP.
        name: A single parameter to describe in full (type, description,
            default). Omit to list all parameter names for ``kind``.

    Returns
    -------
        ``{"kind": ..., "parameters": [...]}`` when listing, or
        ``{"kind": ..., "name": ..., **the property schema}`` for a single
        parameter.

    Raises
    ------
        CuOptMCPError: ``kind`` isn't recognized, or ``name`` isn't a known
            parameter for it.
    """
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
