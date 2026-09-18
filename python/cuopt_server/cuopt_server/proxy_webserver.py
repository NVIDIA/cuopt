# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI app for the gRPC-backed HTTP proxy (LP/MILP/VRP)."""

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import Any, List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import ValidationError

from cuopt.utilities import (
    InputRuntimeError,
    InputValidationError,
    OutOfMemoryError,
)

import cuopt_server.utils.settings as settings
from cuopt_server._version import __version__
from cuopt_server.utils.client_version import check_client_version
from cuopt_server.utils.data_definition import (
    DeleteRequestModel,
    DeleteResponse,
    HealthResponse,
    IdModel,
    IdResponse,
    IncumbentSolution,
    IncumbentSolutionResponse,
    LogResponse,
    LogResponseModel,
    RequestResponse,
    RequestStatusModel,
    SolutionResponse,
    ValidationErrorResponse,
    lp_example_data,
    lp_msgpack_example_data,
    lp_zlib_example_data,
    lpschema,
)
from cuopt_server.utils.exceptions import (
    exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from cuopt_server.utils.http_codec import (
    deserialize,
    encode,
    encode_bytes,
    get_data,
    get_format,
    mime_json,
    mime_msgpack,
    mime_pickle,
    mime_wild,
    mime_zlib,
)
from cuopt_server.utils.http_envelope import make_response
from cuopt_server.utils.linear_programming.conversion import (
    create_data_model,
    create_solver,
    extract_pdlpwarmstart_data,
    solution_to_http,
)
from cuopt_server.utils.linear_programming.data_definition import (
    LPData,
    WarmStartData,
)
from cuopt_server.utils.linear_programming.data_transformation import (
    transform_lp_data,
)
from cuopt_server.utils.linear_programming.data_validation import (
    validate_LP_data,
)
from cuopt_server.utils.local_files import (
    file_result_message,
    get_output_name,
    load_optimization_file,
    result_meets_threshold,
    validate_file_path,
    write_result_file,
)
from cuopt_server.utils.routing.conversion import (
    create_data_model as create_routing_data_model,
    create_solver as create_routing_solver,
    populate_optimization_data,
    prep_optimization_data,
    solution_to_http as routing_solution_to_http,
)
from cuopt_server.utils.routing.data_definition import (
    OptimizedRoutingData,
    SolverSettingsConfig,
)
from cuopt_server.utils.routing.initial_solution import add_initial_sol

app = FastAPI(
    title="NVIDIA cuOpt HTTP proxy",
    version=__version__,
    docs_url="/cuopt/docs",
    redoc_url="/cuopt/redoc",
    openapi_url="/cuopt/openapi.json",
)

_grpc_client = None
_routing_client = None
_jobs = {}
_warmstarts = {}
_jobs_lock = threading.Lock()
_incumbent_locks = {}
_max_request_size = 1024 * 1024 * 1024

_ROUTING_KEYS = {
    "cost_matrix_data",
    "task_data",
    "fleet_data",
    "cost_waypoint_graph_data",
    "travel_time_waypoint_graph_data",
    "travel_time_matrix_data",
    "solver_config",
}

_NOT_IMPLEMENTED = "This feature is not implemented on the gRPC HTTP proxy."


def set_grpc_client(client: Any) -> None:
    """Set the gRPC client used by proxy endpoints."""
    global _grpc_client
    _grpc_client = client


def get_grpc_client() -> Any:
    """Return the configured LP/MILP gRPC client or raise HTTP 503."""
    if _grpc_client is None:
        raise HTTPException(
            status_code=503, detail="gRPC client is not connected"
        )
    return _grpc_client


def set_grpc_routing_client(client: Any) -> None:
    """Set the VRP gRPC client used by proxy endpoints."""
    global _routing_client
    _routing_client = client


def get_grpc_routing_client() -> Any:
    """Return the configured VRP gRPC client or raise HTTP 503."""
    if _routing_client is None:
        raise HTTPException(
            status_code=503, detail="gRPC routing client is not connected"
        )
    return _routing_client


def set_max_request_size(size: int) -> None:
    """Set the maximum accepted HTTP request-body size in bytes."""
    if size < 0:
        raise ValueError("max request size must be non-negative")
    global _max_request_size
    _max_request_size = size


def reset_proxy_state() -> None:
    """Clear process-local proxy state used by tests."""
    global _grpc_client, _routing_client
    with _jobs_lock:
        _jobs.clear()
        _warmstarts.clear()
        _incumbent_locks.clear()
    _grpc_client = None
    _routing_client = None


def _store_job(job_id, meta):
    with _jobs_lock:
        _jobs[job_id] = meta


def _get_job(job_id):
    with _jobs_lock:
        return _jobs.get(job_id)


def _pop_job(job_id):
    with _jobs_lock:
        _incumbent_locks.pop(job_id, None)
        _warmstarts.pop(job_id, None)
        return _jobs.pop(job_id, None)


def _incumbent_lock(job_id):
    with _jobs_lock:
        lock = _incumbent_locks.get(job_id)
        if lock is None:
            lock = threading.Lock()
            _incumbent_locks[job_id] = lock
        return lock


def _update_job(job_id, **fields):
    with _jobs_lock:
        meta = _jobs.get(job_id)
        if meta is not None:
            meta.update(fields)
        return meta


def _warmstart_dict_from_sol(sol):
    try:
        return extract_pdlpwarmstart_data(sol.get_pdlp_warm_start_data())
    except AttributeError:
        return None


def _store_warmstart(job_id, blob):
    if not blob:
        return
    with _jobs_lock:
        # Deletion pops the job first; refuse to resurrect a blob after that.
        if job_id in _jobs:
            _warmstarts[job_id] = blob


def _cached_warmstart(job_id):
    with _jobs_lock:
        return _warmstarts.get(job_id)


def _load_warmstart_blob(job_id):
    with _jobs_lock:
        cached = _warmstarts.get(job_id)
        if cached is not None:
            return cached
        meta = _jobs.get(job_id)
        if meta is not None and meta.get("validation_only"):
            return None
    client = get_grpc_client()
    status = client.status(job_id)
    if _is_status(status, "NOT_FOUND"):
        raise HTTPException(status_code=404, detail=f"id {job_id} not found")
    if _is_status(status, "QUEUED", "PROCESSING"):
        return None
    if _is_status(status, "FAILED", "CANCELLED"):
        return None
    names = None if meta is None else meta.get("variable_names")
    sol = client.result(job_id, variable_names=names)
    if sol is None:
        return None
    blob = _warmstart_dict_from_sol(sol)
    _store_warmstart(job_id, blob)
    return blob


def _warmstart_for_submit(warmstart_id):
    _require_uuid(warmstart_id)
    try:
        blob = _load_warmstart_blob(warmstart_id)
    except HTTPException:
        raise
    if not blob:
        raise HTTPException(
            status_code=404,
            detail=f"Warmstart data for id '{warmstart_id}' not found",
        )
    try:
        return WarmStartData.parse_obj(blob)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="unable to validate warmstart data, %s" % str(e),
        )


@app.exception_handler(Exception)
async def request_exception_handler(request, exc):
    return exception_handler(exc)


def _not_implemented(feature):
    raise HTTPException(
        status_code=501,
        detail=f"{feature} is not supported. {_NOT_IMPLEMENTED}",
    )


def _require_uuid(id):
    try:
        uuid.UUID(id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid request id format"
        )


def _resolve_accept(accept, fallback=mime_json):
    if not accept:
        return fallback
    if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported Accept value {accept}, "
            f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
        )
    if accept in mime_wild:
        return fallback
    return accept


def _status_name(status):
    return getattr(status, "name", str(status))


def _is_status(status, *names):
    return _status_name(status) in names


def _map_status(grpc_status):
    name = _status_name(grpc_status)
    mapping = {
        "QUEUED": RequestStatusModel.queued,
        "PROCESSING": RequestStatusModel.running,
        "COMPLETED": RequestStatusModel.completed,
        "FAILED": RequestStatusModel.aborted,
        "CANCELLED": RequestStatusModel.aborted,
    }
    if name == "NOT_FOUND":
        return None
    return mapping.get(name, RequestStatusModel.aborted)


def _as_id_list(series):
    if series is None:
        return None
    to_arrow = getattr(series, "to_arrow", None)
    if to_arrow is not None:
        return [str(x) for x in to_arrow().to_pylist()]
    if hasattr(series, "tolist"):
        return [str(x) for x in series.tolist()]
    return [str(x) for x in list(series)]


def _result_for_job(job_id, meta, kind):
    """Return a typed GetResult payload (one RPC, two Python parsers)."""
    names = None if meta is None else meta.get("variable_names")
    if kind == "vrp":
        return "vrp", get_grpc_routing_client().result(job_id)
    if kind == "lp":
        return "lp", get_grpc_client().result(job_id, variable_names=names)
    try:
        sol = get_grpc_client().result(job_id, variable_names=names)
        if sol is not None:
            return "lp", sol
    except Exception:
        logging.debug(
            "LP GetResult parser failed for %s; trying routing",
            job_id,
            exc_info=True,
        )
    return "vrp", get_grpc_routing_client().result(job_id)


def _is_mip(lp_data):
    types = getattr(lp_data, "variable_types", None)
    if types is None:
        return False
    return any(str(t).upper() in ("I", "B") for t in types)


def _looks_like_routing(data):
    if not isinstance(data, dict):
        return False
    if "csr_constraint_matrix" in data:
        return False
    return bool((_ROUTING_KEYS - {"solver_config"}) & set(data.keys()))


def _prepare_lp(data, warnings, warmstart_data=None):
    if isinstance(data, list):
        _not_implemented("Batch LP (a JSON list of LP problems)")
    try:
        if isinstance(data, dict):
            transform_lp_data(data)
            lp_data = LPData.parse_obj(data)
        else:
            lp_data = data
            transform_lp_data(lp_data)
        validate_LP_data(lp_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="unable to validate optimization data stream, %s"
            % (str(e)),
        )
    dm_warnings, data_model = create_data_model(lp_data)
    warnings.extend(dm_warnings)
    sw_warnings, solver_settings = create_solver(lp_data, warmstart_data)
    warnings.extend(sw_warnings)
    return lp_data, data_model, solver_settings


def _has_waypoint_graph(parsed):
    for key in (
        "cost_waypoint_graph_data",
        "travel_time_waypoint_graph_data",
    ):
        block = parsed.get(key)
        if block is not None and getattr(block, "waypoint_graph", None):
            return True
    return False


def _ensure_rmm_pool():
    """Waypoint densification still uses WaypointMatrix on the device."""
    import rmm

    mr = rmm.mr.get_current_device_resource()
    if isinstance(mr, rmm.mr.PoolMemoryResource) or isinstance(
        mr, rmm.mr.StatisticsResourceAdaptor
    ):
        return
    pool_gigs = int(os.environ.get("CUOPT_GIGABYTES_PER_PROC", 1))
    pool = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(), initial_pool_size=2**30 * pool_gigs
    )
    rmm.mr.set_current_device_resource(pool)


def _prepare_vrp(data, warnings, initial_envelopes=None):
    try:
        data = dict(OptimizedRoutingData.parse_obj(data))
        if initial_envelopes:
            add_initial_sol(data, initial_envelopes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="unable to validate optimization data stream, %s"
            % (str(e)),
        )

    if data.get("solver_config") is None:
        logging.debug("Creating default solver config")
        data["solver_config"] = SolverSettingsConfig()

    try:
        if _has_waypoint_graph(data):
            _ensure_rmm_pool()
        optimization_data = populate_optimization_data(
            **data, warnings=warnings
        )
        (
            optimization_data,
            cost_matrix,
            travel_time_matrix,
            _,
        ) = prep_optimization_data(optimization_data)
        dm_warnings, data_model = create_routing_data_model(
            optimization_data,
            cost_matrix=cost_matrix,
            travel_time_matrix=travel_time_matrix,
        )
        warnings.extend(dm_warnings)
        sw_warnings, solver_settings = create_routing_solver(optimization_data)
        warnings.extend(sw_warnings)
        vehicle_ids = _as_id_list(
            optimization_data.fleet_data.get("vehicle_ids")
        )
        task_ids = _as_id_list(optimization_data.task_data.get("task_ids"))
        return data_model, solver_settings, vehicle_ids, task_ids
    except HTTPException:
        raise
    except (InputValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (InputRuntimeError, OutOfMemoryError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def _collect_vrp_initials(initial_ids):
    if not initial_ids:
        return None
    envelopes = []
    routing = get_grpc_routing_client()
    for iid in initial_ids:
        _require_uuid(iid)
        meta = _get_job(iid)
        if meta is not None and meta.get("kind") == "lp":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"initialId {iid} is an LP/MILP job, not a VRP solution"
                ),
            )
        if meta is not None and meta.get("validation_only"):
            raise HTTPException(
                status_code=400,
                detail=f"initialId {iid} has no VRP solution",
            )
        status = get_grpc_client().status(iid)
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {iid} not found")
        if _is_status(status, "QUEUED", "PROCESSING"):
            raise HTTPException(
                status_code=409,
                detail=f"initialId {iid} is not completed",
            )
        if _is_status(status, "FAILED", "CANCELLED"):
            raise HTTPException(
                status_code=409,
                detail=f"job {iid} {_status_name(status).lower()}",
            )
        raw = routing.result(iid)
        if raw is None:
            raise HTTPException(
                status_code=409,
                detail=f"initialId {iid} is not completed",
            )
        inner = routing_solution_to_http(
            raw,
            vehicle_ids=None if meta is None else meta.get("vehicle_ids"),
            task_ids=None if meta is None else meta.get("task_ids"),
        )
        envelopes.append({"response": {"solver_response": inner}})
    return envelopes


def _deserialize_convert_submit(
    ctype,
    buf,
    file_path,
    warnings,
    validation_only,
    incumbent_solutions,
    solver_logs,
    accept,
    result_file,
    warmstart_id,
    initial_ids,
):
    """CPU + blocking gRPC work for POST /cuopt/request (run off the event loop)."""
    if file_path:
        data = load_optimization_file(file_path, warnings)
    else:
        data = deserialize(ctype, buf)
    if _looks_like_routing(data):
        initials = _collect_vrp_initials(initial_ids)
        data_model, solver_settings, vehicle_ids, task_ids = _prepare_vrp(
            data, warnings, initials
        )
        if validation_only:
            job_id = str(uuid.uuid4())
            envelope = make_response(
                {
                    "solver_response": {
                        "status": 0,
                        "vehicle_data": {},
                    }
                },
                warnings=warnings,
                notes=["Input is valid"],
                reqId=job_id,
            )
            _store_job(
                job_id,
                {
                    "kind": "vrp",
                    "accept": accept,
                    "warnings": warnings,
                    "vehicle_ids": vehicle_ids,
                    "task_ids": task_ids,
                    "result_file": result_file,
                    "solver_logs": False,
                    "incumbents_enabled": False,
                    "incumbent_next_index": 0,
                    "validation_only": True,
                    "validation_result": envelope,
                },
            )
            return job_id
        job_id = get_grpc_routing_client().submit(data_model, solver_settings)
        _store_job(
            job_id,
            {
                "kind": "vrp",
                "accept": accept,
                "warnings": warnings,
                "vehicle_ids": vehicle_ids,
                "task_ids": task_ids,
                "result_file": result_file,
                "solver_logs": False,
                "incumbents_enabled": False,
                "incumbent_next_index": 0,
                "validation_only": False,
            },
        )
        return job_id
    if initial_ids:
        _not_implemented("Query parameter initialId")
    pdlp = None
    if warmstart_id:
        pdlp = _warmstart_for_submit(warmstart_id)
    lp_data, data_model, solver_settings = _prepare_lp(data, warnings, pdlp)
    variable_names = lp_data.variable_names
    if validation_only:
        job_id = str(uuid.uuid4())
        envelope = make_response(
            {"solver_response": {"status": 0, "solution": {}}},
            warnings=warnings,
            notes=["Input is valid"],
            reqId=job_id,
        )
        _store_job(
            job_id,
            {
                "kind": "lp",
                "accept": accept,
                "warnings": warnings,
                "variable_names": variable_names,
                "result_file": result_file,
                "solver_logs": False,
                "incumbents_enabled": False,
                "incumbent_next_index": 0,
                "validation_only": True,
                "validation_result": envelope,
            },
        )
        return job_id
    client = get_grpc_client()
    incumbents_enabled = bool(incumbent_solutions) and _is_mip(lp_data)
    job_id = client.submit(
        data_model,
        solver_settings,
        enable_incumbents=incumbents_enabled,
    )
    _store_job(
        job_id,
        {
            "kind": "lp",
            "accept": accept,
            "warnings": warnings,
            "variable_names": variable_names,
            "result_file": result_file,
            "solver_logs": bool(solver_logs),
            "incumbents_enabled": incumbents_enabled,
            "incumbent_next_index": 0,
            "validation_only": False,
        },
    )
    return job_id


@app.get("/", responses=HealthResponse)
@app.get("/cuopt/health", responses=HealthResponse)
@app.get("/v2/health/ready", responses=HealthResponse)
@app.get("/v2/health/live", responses=HealthResponse)
def health():
    try:
        _require_grpc_healthy()
        return {"status": "RUNNING", "version": app.version}
    except HTTPException as e:
        return http_exception_handler(e)


def _grpc_backend_ok():
    """Return (True, '') if cuopt_grpc_server answered a short probe."""
    client = _grpc_client
    if client is None:
        return False, "gRPC client is not connected"
    try:
        ping = getattr(client, "ping", None)
        if callable(ping):
            ping(5)
        else:
            client.status("__connection_probe__")
        return True, ""
    except Exception as e:
        text = str(e).strip() or type(e).__name__
        return False, text


def _require_grpc_healthy():
    ok, msg = _grpc_backend_ok()
    if not ok:
        logging.error("ERROR : %s", msg)
        raise HTTPException(
            status_code=500,
            detail=(
                "Status : Broken\n"
                "The gRPC backend is unavailable. "
                "This process cannot serve solves until "
                "cuopt_grpc_server is healthy.\n"
                "ERROR : gRPC backend unavailable"
            ),
        )


@app.get(
    "/cuopt/log/{id}",
    response_model=LogResponseModel,
    responses=LogResponse,
)
def getsolverlogs(
    id: str,
    accept: str = Header(default="application/json"),
    frombyte: Optional[int] = Query(default=0),
):
    try:
        accept = _resolve_accept(accept)
        _require_uuid(id)
        if frombyte < 0:
            raise HTTPException(
                status_code=422, detail="frombyte must be >= 0"
            )
        meta = _get_job(id)
        if meta is not None and (
            meta.get("kind") == "vrp"
            or meta.get("validation_only")
            or not meta.get("solver_logs")
        ):
            raise HTTPException(
                status_code=404, detail=f"log not found for request {id}"
            )
        client = get_grpc_client()
        try:
            from cuopt.grpc.linear_programming import JobNotReadyError

            lines = client.logs(id, frombyte)
        except JobNotReadyError:
            return encode({"log": [""], "nbytes": frombyte}, accept)
        except Exception as e:
            if "not found" in str(e).lower() or "NOT_FOUND" in str(e):
                raise HTTPException(
                    status_code=404, detail=f"log not found for request {id}"
                )
            raise
        payload = "\n".join(lines)
        return encode(
            {"log": lines, "nbytes": frombyte + len(payload.encode())},
            accept,
        )
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.delete("/cuopt/log/{id}", responses=DeleteResponse)
def deletesolverlogs(
    id: str,
    accept: str = Header(default="application/json"),
):
    try:
        accept = _resolve_accept(accept)
        _require_uuid(id)
        # gRPC deletes logs with the job. HTTP DELETE log is a no-op 200.
        return Response(status_code=200)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.get(
    "/cuopt/solution/{id}/incumbents",
    response_model=List[IncumbentSolution],
    responses=IncumbentSolutionResponse,
)
def getincumbent(
    id: str,
    accept: str = Header(default="application/json"),
):
    try:
        accept = _resolve_accept(accept)
        _require_uuid(id)
        meta = _get_job(id)
        if meta is not None and meta.get("kind") == "vrp":
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        if meta is not None and meta.get("validation_only"):
            return encode(
                [{"solution": [], "cost": None, "bound": None}], accept
            )
        client = get_grpc_client()
        status = client.status(id)
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        with _incumbent_lock(id):
            meta = _get_job(id)
            from_index = (
                0 if meta is None else meta.get("incumbent_next_index", 0)
            )
            entries = client.incumbents(id, from_index=from_index)
            result = [
                {
                    "solution": e.get("assignment", []),
                    "cost": e.get("objective"),
                    "bound": None,
                }
                for e in entries
            ]
            if entries:
                next_index = max(e["index"] for e in entries) + 1
                _update_job(id, incumbent_next_index=next_index)
        terminal = not _is_status(status, "QUEUED", "PROCESSING")
        if not result and terminal:
            result = [{"solution": [], "cost": None, "bound": None}]
        return encode(result, accept)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.post(
    "/cuopt/solution",
    response_model=IdModel,
    responses=IdResponse,
)
async def postsolution():
    _not_implemented("POST /cuopt/solution (uploaded solutions)")


@app.delete("/cuopt/solution/{id}", responses=DeleteResponse)
def deletesolution(
    id: str = Path(...),
    accept: str = Header(default="application/json"),
):
    try:
        accept = _resolve_accept(accept)
        _require_uuid(id)
        meta = _get_job(id)
        if meta is not None and meta.get("validation_only"):
            _pop_job(id)
            return Response(status_code=200)
        status = get_grpc_client().status(id)
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        try:
            get_grpc_client().delete(id)
        except Exception as e:
            if "not found" in str(e).lower() or "NOT_FOUND" in str(e):
                raise HTTPException(
                    status_code=404, detail=f"id {id} not found"
                )
            raise
        _pop_job(id)
        return Response(status_code=200)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.delete(
    "/cuopt/request/{id}",
    response_model=DeleteRequestModel,
    responses=ValidationErrorResponse,
)
def deleterequest(
    id: str = Path(...),
    accept: str = Header(default="application/json"),
    running: Optional[bool] = Query(default=None),
    queued: Optional[bool] = Query(default=None),
    cached: Optional[bool] = Query(default=None),
):
    try:
        accept = _resolve_accept(accept)
        if id == "*":
            _not_implemented("Wildcard DELETE /cuopt/request/*")
        if running is not None or queued is not None or cached is not None:
            _not_implemented(
                "DELETE /cuopt/request flags running/queued/cached"
            )
        _require_uuid(id)
        meta = _get_job(id)
        counts = {"queued": 0, "running": 0, "cached": 0}
        if meta is not None and meta.get("validation_only"):
            return encode(counts, accept)
        status = get_grpc_client().status(id)
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        if _is_status(status, "QUEUED"):
            counts["queued"] = 1
        elif _is_status(status, "PROCESSING"):
            counts["running"] = 1
        # gRPC cancel rejects completed jobs; HTTP abort of a finished
        # request is a no-op 200 (solution remains until DELETE solution).
        if _is_status(status, "QUEUED", "PROCESSING"):
            get_grpc_client().cancel(id)
        return encode(counts, accept)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.get(
    "/cuopt/solution/{id}/warmstart",
    include_in_schema=False,
)
def getwarmstart(id: str):
    try:
        _require_uuid(id)
        meta = _get_job(id)
        if meta is not None and meta.get("validation_only"):
            return encode({"reqId": id}, mime_msgpack)
        blob = _load_warmstart_blob(id)
        if not blob:
            return encode({"reqId": id}, mime_msgpack)
        return Response(
            content=encode_bytes(blob, mime_msgpack),
            media_type=mime_msgpack,
        )
    except HTTPException as e:
        return encode(http_exception_handler(e), mime_msgpack)
    except Exception as e:
        return encode(exception_handler(e), mime_msgpack)


@app.get(
    "/cuopt/solution/{id}",
    responses=SolutionResponse,
)
def getsolution(
    id: str,
    accept: str = Header(default="application/json"),
):
    try:
        fallback = mime_json
        meta = _get_job(id)
        if meta is not None:
            fallback = meta.get("accept", mime_json)
        accept = _resolve_accept(accept, fallback)
        _require_uuid(id)
        if meta is not None and meta.get("validation_only"):
            return encode(meta["validation_result"], accept, job_result=True)
        status = get_grpc_client().status(id)
        kind = None if meta is None else meta.get("kind")
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        if _is_status(status, "QUEUED", "PROCESSING"):
            return encode({"reqId": id}, accept)
        if _is_status(status, "FAILED", "CANCELLED"):
            raise HTTPException(
                status_code=409,
                detail=f"job {id} {_status_name(status).lower()}",
            )
        result_kind, sol = _result_for_job(id, meta, kind)
        if sol is None:
            return encode({"reqId": id}, accept)
        notes = []
        warnings = [] if meta is None else list(meta.get("warnings") or [])
        if result_kind == "vrp":
            inner = routing_solution_to_http(
                sol,
                vehicle_ids=None if meta is None else meta.get("vehicle_ids"),
                task_ids=None if meta is None else meta.get("task_ids"),
            )
            if inner.get("status") == 1:
                notes.append(sol.get("status_message") or "")
            solve_time = 0
            envelope = make_response(
                {"solver_response": inner},
                warnings=warnings,
                notes=[n for n in notes if n],
                reqId=id,
                total_solve_time=solve_time,
            )
        else:
            _store_warmstart(id, _warmstart_dict_from_sol(sol))
            inner = solution_to_http(sol, include_warmstart=False)
            try:
                notes.append(sol.get_termination_reason())
            except Exception:
                pass
            solve_time = 0
            if inner.get("solution"):
                solve_time = inner["solution"].get("solver_time") or 0
            envelope = make_response(
                {"solver_response": inner},
                warnings=warnings,
                notes=notes,
                reqId=id,
                total_solve_time=solve_time,
            )
        resultdir, maxresult, mode = settings.get_result_dir()
        result_file = "" if meta is None else meta.get("result_file") or ""
        if result_file and resultdir:
            raw = encode_bytes(envelope, accept)
            if result_meets_threshold(
                result_file, resultdir, len(raw), maxresult
            ):
                write_result_file(resultdir, result_file, raw, mode)
                file_msg = file_result_message(
                    result_file, warnings=warnings, notes=notes
                )
                file_msg["format"] = get_format(accept)
                return encode(file_msg, accept, job_result=True)
        return encode(envelope, accept, job_result=True)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.get(
    "/cuopt/request/{id}",
    response_model=RequestStatusModel,
    responses=RequestResponse,
)
def getrequest(
    id: str,
    accept: str = Header(default="application/json"),
):
    try:
        accept = _resolve_accept(accept)
        _require_uuid(id)
        meta = _get_job(id)
        if meta is not None and meta.get("validation_only"):
            return encode(RequestStatusModel.completed.value, accept)
        status = get_grpc_client().status(id)
        mapped = _map_status(status)
        if mapped is None or _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        return encode(mapped.value, accept)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.post(
    "/cuopt/cuopt",
)
async def post_cuopt_sync():
    _not_implemented("POST /cuopt/cuopt")


@app.post(
    "/cuopt/request",
    response_model=IdModel,
    responses=IdResponse,
    summary="Solve an LP/MILP/VRP problem via gRPC (self-hosted proxy)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": lpschema,
                    "examples": {
                        "LP request": {"value": lp_example_data},
                    },
                },
                "application/vnd.msgpack": {
                    "schema": {"type": "string", "format": "byte"},
                    "examples": {
                        "LP request compressed with msgpack": {
                            "value": lp_msgpack_example_data
                        },
                    },
                },
                "application/zlib": {
                    "schema": {"type": "string", "format": "byte"},
                    "examples": {
                        "LP request compressed with zlib": {
                            "value": lp_zlib_example_data
                        },
                    },
                },
            },
            "required": True,
        }
    },
)
async def postrequest(
    request: Request,
    cache: Optional[bool] = Query(default=False),
    reqId: Optional[str] = Query(default=None),
    initialId: Optional[List[str]] = Query(default=None),
    warmstartId: Optional[str] = Query(default=None),
    validation_only: Optional[bool] = Query(default=False),
    incumbent_solutions: Optional[bool] = Query(default=False),
    incumbent_set_solutions: Optional[bool] = Query(default=False),
    solver_logs: Optional[bool] = Query(default=False),
    cuopt_data_file: str = Header(default=None),
    cuopt_result_file: str = Header(default=None),
    client_version: str = Header(default=None),
    accept: str = Header(default="application/json"),
    content_type: str = Header(default="application/json"),
    content_length: int = Header(default=0),
):
    ctype = content_type
    sz = content_length
    if cuopt_data_file is None:
        cuopt_data_file = ""
    if cuopt_result_file is None:
        cuopt_result_file = ""
    if client_version is None:
        client_version = ""

    warnings = check_client_version(client_version)

    try:
        await asyncio.to_thread(_require_grpc_healthy)
        accept = _resolve_accept(
            accept, ctype if ctype != mime_pickle else mime_json
        )
        if cache:
            _not_implemented("Query parameter cache")
        if reqId:
            _not_implemented("Query parameter reqId (cached-body solve)")
        if incumbent_set_solutions:
            _not_implemented("Query parameter incumbent_set_solutions")

        sz = int(sz)
        if sz < 0:
            raise HTTPException(
                status_code=422, detail="Content-Length must be non-negative"
            )
        if sz > _max_request_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Content-Length exceeds maximum of "
                    f"{_max_request_size} bytes"
                ),
            )
        if sz == 0 and not cuopt_data_file:
            raise HTTPException(
                status_code=422, detail="Data length is zero and reqId not set"
            )

        if ctype not in [mime_json, mime_msgpack, mime_zlib, mime_pickle]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Content-Type value {ctype}, "
                f"supported values are "
                f"{[mime_json, mime_msgpack, mime_zlib, mime_pickle]}",
            )

        file_path = ""
        if cuopt_data_file:
            file_path = validate_file_path(cuopt_data_file)

        buf = None
        if not file_path:
            now = time.time()
            buf = bytearray(sz)
            await get_data(buf, request)
            logging.debug(f"time to receive data {time.time() - now}")

        resultdir, _, _ = settings.get_result_dir()
        result_file = get_output_name(
            resultdir, cuopt_data_file, cuopt_result_file
        )
        job_id = await asyncio.to_thread(
            _deserialize_convert_submit,
            ctype,
            buf,
            file_path,
            warnings,
            validation_only,
            incumbent_solutions,
            solver_logs,
            accept,
            result_file,
            warmstartId,
            initialId,
        )
        return encode({"reqId": job_id}, accept)

    except (RequestValidationError, ValidationError) as e:
        return encode(validation_exception_handler(e), accept)

    except HTTPException as e:
        return encode(http_exception_handler(e), accept)

    except Exception as e:
        return encode(exception_handler(e), accept)


def run_server(ip, port, log_level, ssl_certfile, ssl_keyfile):
    uvi_config = uvicorn.config.LOGGING_CONFIG
    for uvilog in uvi_config["handlers"].keys():
        uvi_config["handlers"][uvilog] = {
            "class": "logging.NullHandler",
        }
    for uvilog in uvi_config["loggers"].keys():
        uvi_config["loggers"][uvilog]["propagate"] = True

    if len(ssl_certfile) > 0 and len(ssl_keyfile) > 0:
        if not os.path.exists(ssl_certfile):
            raise ValueError(f"File path '{ssl_certfile}' doesn't exist")
        if not os.path.exists(ssl_keyfile):
            raise ValueError(f"File path '{ssl_keyfile}' doesn't exist")
    elif len(ssl_certfile) > 0 or len(ssl_keyfile) > 0:
        raise ValueError(
            "Need to provide both certfile and keyfile to enable SSL"
        )
    else:
        ssl_certfile = None
        ssl_keyfile = None

    uvicorn.run(
        "cuopt_server.proxy_webserver:app",
        host=ip,
        port=port,
        log_config=uvi_config,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
    logging.info("proxy webserver finished")
