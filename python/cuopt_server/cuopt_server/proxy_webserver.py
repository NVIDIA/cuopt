# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI app for the gRPC-backed HTTP proxy (LP/MILP, C10)."""

import logging
import os
import threading
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import ValidationError

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
    solution_to_legacy_http,
)
from cuopt_server.utils.linear_programming.data_definition import LPData
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

app = FastAPI(
    title="NVIDIA cuOpt HTTP proxy",
    version=__version__,
    docs_url="/cuopt/docs",
    redoc_url="/cuopt/redoc",
    openapi_url="/cuopt/openapi.json",
)

_grpc_client = None
_jobs = {}
_jobs_lock = threading.Lock()

_ROUTING_KEYS = {
    "cost_matrix_data",
    "task_data",
    "fleet_data",
    "cost_waypoint_graph_data",
    "travel_time_waypoint_graph_data",
    "travel_time_matrix_data",
    "solver_config",
}

_NOT_IMPLEMENTED = (
    "This feature is not implemented on the gRPC HTTP proxy. "
    "Use the legacy HTTP server (python -m cuopt_server.cuopt_service) "
    "or a later proxy PR."
)


def set_grpc_client(client):
    global _grpc_client
    _grpc_client = client


def get_grpc_client():
    if _grpc_client is None:
        raise HTTPException(
            status_code=503, detail="gRPC client is not connected"
        )
    return _grpc_client


def reset_proxy_state():
    global _grpc_client
    with _jobs_lock:
        _jobs.clear()
    _grpc_client = None


def _store_job(job_id, meta):
    with _jobs_lock:
        _jobs[job_id] = meta


def _get_job(job_id):
    with _jobs_lock:
        return _jobs.get(job_id)


def _pop_job(job_id):
    with _jobs_lock:
        return _jobs.pop(job_id, None)


def _update_job(job_id, **fields):
    with _jobs_lock:
        meta = _jobs.get(job_id)
        if meta is not None:
            meta.update(fields)
        return meta


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


def _looks_like_routing(data):
    if not isinstance(data, dict):
        return False
    if "csr_constraint_matrix" in data:
        return False
    return bool((_ROUTING_KEYS - {"solver_config"}) & set(data.keys()))


def _prepare_lp(data, warnings):
    if isinstance(data, list):
        _not_implemented("Batch LP (a JSON list of LP problems)")
    if _looks_like_routing(data):
        _not_implemented("Vehicle routing (VRP)")
    try:
        if isinstance(data, dict):
            transform_lp_data(data)
            lp_data = LPData.parse_obj(data)
        else:
            lp_data = data
            transform_lp_data(lp_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="unable to validate optimization data stream, %s"
            % (str(e)),
        )
    validate_LP_data(lp_data)
    dm_warnings, data_model = create_data_model(lp_data)
    warnings.extend(dm_warnings)
    sw_warnings, solver_settings = create_solver(lp_data, None)
    warnings.extend(sw_warnings)
    return lp_data, data_model, solver_settings


@app.get("/", responses=HealthResponse)
@app.get("/cuopt/health", responses=HealthResponse)
@app.get("/v2/health/ready", responses=HealthResponse)
@app.get("/v2/health/live", responses=HealthResponse)
def health():
    return {"status": "RUNNING", "version": app.version}


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
        if meta is not None and meta.get("validation_only"):
            raise HTTPException(
                status_code=404, detail=f"log not found for request {id}"
            )
        if meta is not None and not meta.get("solver_logs"):
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
        return exception_handler(e)


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
        if meta is not None and meta.get("validation_only"):
            return encode(
                [{"solution": [], "cost": None, "bound": None}], accept
            )
        client = get_grpc_client()
        status = client.status(id)
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        from_index = 0 if meta is None else meta.get("incumbent_next_index", 0)
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
        meta = _pop_job(id)
        if meta is not None and meta.get("validation_only"):
            return Response(status_code=200)
        client = get_grpc_client()
        try:
            client.delete(id)
        except Exception as e:
            if "not found" in str(e).lower() or "NOT_FOUND" in str(e):
                raise HTTPException(
                    status_code=404, detail=f"id {id} not found"
                )
            raise
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
        client = get_grpc_client()
        status = client.status(id)
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        if _is_status(status, "QUEUED"):
            counts["queued"] = 1
        elif _is_status(status, "PROCESSING"):
            counts["running"] = 1
        # gRPC cancel rejects completed jobs; HTTP abort of a finished
        # request is a no-op 200 (solution remains until DELETE solution).
        if _is_status(status, "QUEUED", "PROCESSING"):
            client.cancel(id)
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
    _not_implemented("GET /cuopt/solution/{id}/warmstart (C13)")


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
        client = get_grpc_client()
        status = client.status(id)
        if _is_status(status, "NOT_FOUND"):
            raise HTTPException(status_code=404, detail=f"id {id} not found")
        if _is_status(status, "QUEUED", "PROCESSING"):
            return encode({"reqId": id}, accept)
        sol = client.result(
            id,
            variable_names=(
                None if meta is None else meta.get("variable_names")
            ),
        )
        if sol is None:
            return encode({"reqId": id}, accept)
        inner = solution_to_legacy_http(sol, include_warmstart=False)
        notes = []
        try:
            notes.append(sol.get_termination_reason())
        except Exception:
            pass
        warnings = [] if meta is None else list(meta.get("warnings") or [])
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
        client = get_grpc_client()
        status = client.status(id)
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
    _not_implemented("POST /cuopt/cuopt (C12)")


@app.post(
    "/cuopt/request",
    response_model=IdModel,
    responses=IdResponse,
    summary="Solve an LP/MILP problem via gRPC (self-hosted proxy)",
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
        accept = _resolve_accept(
            accept, ctype if ctype != mime_pickle else mime_json
        )
        if cache:
            _not_implemented("Query parameter cache")
        if reqId:
            _not_implemented("Query parameter reqId (cached-body solve)")
        if initialId:
            _not_implemented("Query parameter initialId (C11)")
        if warmstartId:
            _not_implemented("Query parameter warmstartId (C13)")
        if incumbent_set_solutions:
            _not_implemented("Query parameter incumbent_set_solutions")

        sz = int(sz)
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

        if file_path:
            data = load_optimization_file(file_path, warnings)
        else:
            now = time.time()
            buf = bytearray(sz)
            await get_data(buf, request)
            logging.debug(f"time to receive data {time.time() - now}")
            data = deserialize(ctype, buf)

        lp_data, data_model, solver_settings = _prepare_lp(data, warnings)
        variable_names = lp_data.variable_names
        resultdir, _, _ = settings.get_result_dir()
        result_file = get_output_name(
            resultdir, cuopt_data_file, cuopt_result_file
        )

        if validation_only:
            job_id = str(uuid.uuid4())
            notes = ["Input is valid"]
            envelope = make_response(
                {"solver_response": {"status": 0, "solution": {}}},
                warnings=warnings,
                notes=notes,
                reqId=job_id,
            )
            _store_job(
                job_id,
                {
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
            return encode({"reqId": job_id}, accept)

        client = get_grpc_client()
        job_id = client.submit(
            data_model,
            solver_settings,
            enable_incumbents=bool(incumbent_solutions),
        )
        _store_job(
            job_id,
            {
                "accept": accept,
                "warnings": warnings,
                "variable_names": variable_names,
                "result_file": result_file,
                "solver_logs": bool(solver_logs),
                "incumbents_enabled": bool(incumbent_solutions),
                "incumbent_next_index": 0,
                "validation_only": False,
            },
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
