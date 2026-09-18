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

from .client import CuOptMCPError, describe_connection_error, get_client
from .schema import known_parameters, settings_schema, validate_settings

# Above this many variables a solution is written to a file instead of
# returned inline -- the binding limit is the model's context window.
INLINE_SOLUTION_LIMIT = 200

# Upper bound on logs()'s tail_lines, same context-window reasoning.
MAX_TAIL_LINES = 2000


def _check_non_negative_int(name: str, value) -> None:
    """Reject a bool (an int subclass in Python) or a negative value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CuOptMCPError(f"{name} must be a non-negative integer")


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
    model = _read_problem(names_from)
    names = model.get_variable_names()
    return list(names) if names is not None else None


def submit(
    problem_path: str,
    kind: str,
    settings: dict | None = None,
    track_incumbents: bool = False,
) -> dict:
    """Parse a problem file and submit it for an asynchronous solve.

    Args:
        problem_path: Path to an MPS/QPS/LP file readable by this process.
        kind: "pdlp_settings" for LP or "mip_settings" for MILP; selects
            which settings schema ``settings`` is validated against.
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
        CuOptMCPError: ``kind`` is invalid, the file doesn't exist, fails to
            parse, carries an unknown setting, or the backend is
            unreachable.
    """
    model = _read_problem(problem_path)
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
    return {
        "job_id": job_id,
        "source": str(Path(problem_path).expanduser().resolve()),
        "num_variables": int(len(model.get_variable_lower_bounds())),
        "num_constraints": int(max(len(offsets) - 1, 0)),
        "next": (
            "Poll cuopt_status(job_id). When it reports COMPLETED, call "
            "cuopt_result(job_id, names_from=source) for a named solution."
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
        nonzero_only: Drop exactly-zero values before applying ``limit``.
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
        _solution_file_path(job_id).unlink(missing_ok=True)
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
