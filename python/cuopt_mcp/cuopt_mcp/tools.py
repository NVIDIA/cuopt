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

from .client import CuOptMCPError, describe_connection_error, get_client
from .schema import known_parameters, settings_schema, validate_settings

# Above this many variables a solution is written to a file instead of
# returned inline. The binding limit is the model's context window, not the
# transport: ~200 values is already a large tool result, and cuOpt problems
# routinely have millions.
INLINE_SOLUTION_LIMIT = 200


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
    model = _read_problem(names_from)
    names = model.get_variable_names()
    return list(names) if names is not None else None


def submit(problem_path: str, kind: str, settings: dict | None = None) -> dict:
    """Parse a problem file and submit it; return the job handle."""
    model = _read_problem(problem_path)
    solver_settings = _build_settings(kind, settings)
    try:
        job_id = get_client().submit(model, solver_settings)
    except Exception as exc:
        raise describe_connection_error(exc) from exc

    # DataModel exposes no public size accessors, so derive both from the
    # arrays it does expose: one lower bound per column, and CSR row offsets
    # numbering rows + 1.
    offsets = model.get_constraint_matrix_offsets()
    return {
        "job_id": job_id,
        "source": str(Path(problem_path).expanduser()),
        "num_variables": int(len(model.get_variable_lower_bounds())),
        "num_constraints": int(max(len(offsets) - 1, 0)),
        "next": (
            "Poll cuopt_status(job_id). When it reports COMPLETED, call "
            "cuopt_result(job_id, names_from=source) for a named solution."
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
        selected = {k: v for k, v in vars_by_name.items() if v != 0}
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
