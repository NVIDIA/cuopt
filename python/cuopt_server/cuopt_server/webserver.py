# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import queue
import threading
import time
import uuid
import zlib
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
from typing import List, Optional, Union

import msgpack
import msgpack_numpy
import psutil
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

import cuopt_server.utils.health_check as health_check
import cuopt_server.utils.settings as settings
from cuopt_server._version import __version__
from cuopt_server.utils.data_definition import (
    DeleteRequestModel,
    DeleteResponse,
    EmptyResponseModel,
    HealthResponse,
    IdModel,
    IdResponse,
    IncumbentSolution,
    IncumbentSolutionResponse,
    LogResponse,
    LogResponseModel,
    ManagedRequestResponse,
    RequestResponse,
    RequestStatusModel,
    SolutionModel,
    SolutionModelInFile,
    SolutionModelWithId,
    SolutionResponse,
    ValidationErrorResponse,
    cuoptdataschema,
    lp_example_data,
    lp_msgpack_example_data,
    lp_zlib_example_data,
    lpschema,
    managed_lp_example_data,
    managed_vrp_example_data,
    solutionschema,
    vrp_example_data,
    vrp_msgpack_example_data,
    vrpschema,
)
from cuopt_server.utils.exceptions import (
    exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from cuopt_server.utils.job_queue import (
    BaseResult,
    BinaryJobResult,
    NVCFJobResult,
    Shutdown,
    SolverBinaryJob,
    SolverBinaryJobPath,
    abort_all,
    abort_by_id,
    add_cache_entry,
    check_client_version,
    delete_cache_entry,
    get_cache_content_type,
    get_incumbents_for_id,
    get_solution_for_id,
    get_warmstart_data_for_id,
    mime_json,
    mime_msgpack,
    mime_pickle,
    mime_wild,
    mime_zlib,
    status_by_id,
    update_cache_entry,
)
from cuopt_server.utils.logutil import message, set_ncaid, set_requestid

msgpack_numpy.patch()


def get_cuopt_version():
    return __version__[:5]


app = FastAPI(
    title="cuOpt Server",
    summary="OpenAPI Specification for cuOpt",
    version=get_cuopt_version(),
    docs_url="/cuopt/docs",
    redoc_url="/cuopt/redoc",
    openapi_url="/cuopt.yaml",
)


# This is a blanket handler to turn any HTTPException into
# a response in our desired format.
@app.exception_handler(HTTPException)
async def request_http_exception_handler(request, exc):
    return http_exception_handler(exc)


# This is a blanket handler to turn any ValidationError into
# a reponse in our desired format.
# Since the fastapi RequestValidationError is derived from
# pydantic.ValidationError, it works for both.
@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
async def request_validation_exception_handler(request, exc):
    return validation_exception_handler(exc)


# Register a catchall for unhandled exceptions
# Use the same handler we call explicitly to log an
# abbreviated callstack and create an error return
@app.exception_handler(Exception)
async def request_exception_handler(request, exc):
    return exception_handler(exc)


@app.get(
    "/",
    description="To ping if server is running",
    responses=HealthResponse,
)
@app.get(
    "/cuopt/health",
    description="To ping if server is running",
    responses=HealthResponse,
)
@app.get(
    "/v2/health/ready",
    description="To check readiness of the server",
    responses=HealthResponse,
)
@app.get(
    "/v2/health/live",
    description="To check liveness of the server",
    responses=HealthResponse,
)
def health():
    status, msg = health_check.health()
    if status == 0:
        return {"status": "RUNNING", "version": app.version}
    else:
        msg = (
            """
            Status : Broken
            The server will be restarted and will be available in 15 mins !!!
            """
            + "ERROR : "
            + msg
        )
        raise HTTPException(status_code=500, detail=f"{msg}")


# Get name for file that stores the result of Solve
def get_output_name(resultdir, CUOPT_DATA_FILE, CUOPT_RESULT_FILE):
    # Prevent absolute paths, or navigating with ../..
    if CUOPT_RESULT_FILE.startswith("/") or ".." in CUOPT_RESULT_FILE:
        CUOPT_RESULT_FILE = ""
    if not resultdir:
        res = ""
    elif CUOPT_RESULT_FILE:
        res = CUOPT_RESULT_FILE
    elif CUOPT_DATA_FILE:
        res = os.path.basename(CUOPT_DATA_FILE) + ".result"
    else:
        res = str(uuid.uuid4())
    return res


# Validate if given data file and file path exists
def validate_file_path(CUOPT_DATA_FILE):
    ddir = settings.get_data_dir()
    try:
        file_path = os.path.join(ddir, CUOPT_DATA_FILE)
        if not ddir:
            logging.error("cuopt data directory not set!")
            # If no datadir was set but the path is relative,
            # this can't work
            if not CUOPT_DATA_FILE.startswith("/"):
                raise ValueError(
                    f"cuopt server was started without data directory "
                    f"defined but local path {CUOPT_DATA_FILE} "
                    "was specified"
                )
        if not os.path.exists(file_path):
            logging.error(f"File path '{file_path}' doesn't exist")
            msg = f"Specified path '{file_path}' does not exist"
            if CUOPT_DATA_FILE.startswith("/"):
                dir = os.path.dirname(CUOPT_DATA_FILE)
                if not os.path.isdir(dir):
                    msg += f". Absolute path '{dir}' does not exist"
                msg += ". Perhaps you did not intend to "
                "specify an absolute path?"
            raise ValueError(msg)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="unable to read "
            "optimization data from file %s, %s" % (file_path, str(e)),
        )
    return file_path


app_exit = None
job_queue = None
abort_queue = None
abort_list = None


def set_queues_and_flags(app_ex, job_que, abort_que, abort_lst):
    global app_exit
    global job_queue
    global abort_queue
    global abort_list
    app_exit = app_ex
    job_queue = job_que
    abort_queue = abort_que
    abort_list = abort_lst


def wait_for_job(result, job, timeout=None):
    global job_queue
    logging.info(f"waiting for job {job.id} with timeout {timeout}")
    job_queue.put(job)

    return result.wait(timeout)


# This class is to wrap an exception response generated
# in the solver process in a local exception so that the
# web server can raise it for an early exit
class SolverException(Exception):
    def __init__(self, response):
        self.response = response


def encode(result, accept, job_result=False):

    if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
        accept = mime_json

    # This is an exception packaged up elsewhere
    if isinstance(result, JSONResponse):
        status_code = result.status_code
        result = json.loads(result.body)
        result["error_result"] = job_result
        if accept == mime_json:
            return JSONResponse(result, status_code)
    else:
        status_code = 200

    # Expect a dictionary at this point
    if accept == mime_json:
        logging.debug("job_result returning json")
        r = result
    elif accept == mime_zlib:
        logging.debug("job_result returning zlib")
        d = bytes(json.dumps(result), encoding="utf-8")
        r = Response(
            content=zlib.compress(d, zlib.Z_BEST_SPEED),
            media_type=mime_zlib,
            status_code=status_code,
        )
    else:
        logging.debug("job_result returning msgpack")
        r = Response(
            content=msgpack.dumps(result),
            media_type=mime_msgpack,
            status_code=status_code,
        )
    return r


def get_format(mime_type):

    f = {
        mime_json: "json",
        mime_zlib: "zlib",
        mime_msgpack: "msgpack",
        mime_pickle: "pickle",
    }
    return f[mime_type]


@app.get(
    "/cuopt/log/{id}",
    description="Note: This is for self-hosted. "
    "Query solver log. The 'id' is the uuid returned when the request "
    "was made.",
    summary="Query solver logs by id self-hosted",
    response_model=LogResponseModel,
    responses=LogResponse,
)
def getsolverlogs(
    id: str,
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. ",
    ),
    frombyte: Optional[int] = Query(
        default=0, description="Indicates the position to start log read"
    ),
):
    try:
        if not accept:
            accept = mime_json

        if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )
        # TODO: Create a log dir
        log_dir, _, _ = settings.get_result_dir()
        log_fname = "log_" + id
        log_file = os.path.join(log_dir, log_fname)
        logging.info(f"Extracting logs from {log_file}")
        log_response = {"log": None, "nbytes": None}

        with open(log_file, "r") as out:
            out.seek(frombyte)
            log_data = out.read()
            nbytes = out.tell()
            log_response["log"] = log_data.split("\n")
            log_response["nbytes"] = nbytes
        return encode(log_response, accept)

    except FileNotFoundError:
        return http_exception_handler(
            HTTPException(
                status_code=404, detail=f"log not found for request {id}"
            )
        )


@app.get(
    "/cuopt/solution/{id}/incumbents",
    description="Note: for use with self-hosted cuOpt instances. "
    "Return incumbent solutions from the MIP solver produced for "
    "this id since the last GET. Result will be a list of the form "
    "[{'solution': [1.0, 1.0], 'cost': 2.0}] where each item "
    "contains the fields 'solution' (a list of floats) and "
    "'cost' (a float). "
    "An empty list indicates that there are no current incumbent solutions "
    "at this time. A sentinel value of [{'solution': [], 'cost': None}] "
    "indicates that no future incumbent values will be produced. "
    "The 'id' is the reqId value returned from a POST to /cuopt/request",
    summary="Get incumbent solutions for MIP (self-hosted)",
    response_model=List[IncumbentSolution],
    responses=IncumbentSolutionResponse,
)
def getincumbent(
    id: str,
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. "
        "If a wildcard is used, the accept mime_type will be set "
        "to the content_type mime_type of the original request.",
    ),
):
    try:
        if not accept:
            accept = mime_json

        if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )

        result = get_incumbents_for_id(id)
        if accept in mime_wild:
            accept = mime_json

        return encode(result, accept)

    except (RequestValidationError, ValidationError) as e:
        return encode(validation_exception_handler(e), accept)

    except HTTPException as e:
        return encode(http_exception_handler(e), accept)

    except Exception as e:
        return encode(exception_handler(e), accept)


@app.delete(
    "/cuopt/log/{id}",
    description="Note: This is for self-hosted. "
    "Delete accumulated logs for a particular request.",
    summary="Delete solver logs by id (self-hosted)",
    responses=DeleteResponse,
)
def deletesolverlogs(
    id: str,
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. "
        "This applies to exception messages returned by this request.",
    ),
):
    try:
        if not accept:
            accept = mime_json

        if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )

        # Delete the log for the request if the request is not done
        log_dir, _, _ = settings.get_result_dir()
        log_fname = "log_" + id
        log_file = os.path.join(log_dir, log_fname)
        os.unlink(log_file)
    except FileNotFoundError:
        return http_exception_handler(
            HTTPException(
                status_code=404, detail=f"log not found for request {id}"
            )
        )
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return exception_handler(e)


@app.post(
    "/cuopt/solution",
    description=(
        "Note: for use with self-hosted cuOpt instances. "
        "Accepts a cuOpt solution and returns a reqId. "
        "This is useful for uploading a VRP solution to use as an initial solution for another request. "  # noqa
    ),
    # This is for validation by Pydantic
    response_model=IdModel,
    # This is for response examples and schema
    responses=IdResponse,
    summary="Upload a routing solution only(self-hosted)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": solutionschema,
                }
            },
            "required": True,
        }
    },
)
async def postsolution(
    request: Request,
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. "
        "If a wildcard is used, the accept mime_type will be set "
        "to the content_type mime_type",
    ),
    content_type: str = Header(
        default="application/json",
        description="Supported content mime_types are 'application/json', "
        "'application/vnd.msgpack', and 'application/zlib'",
    ),
    content_length: int = Header(),
):

    # TODO validate the solution
    # This will have to be call to a thread to read the id, validate,
    # and then update the validation status on the solution.
    # The solution id can be passed through a validation queue to a
    # validation thread in the webserver to do this asynchronously

    # just rename these for compat with existing code
    ctype = content_type
    sz = content_length

    try:
        # Skips the job if health status is bad
        health()

        sz = int(sz)
        if sz == 0:
            raise HTTPException(status_code=422, detail="Data length is zero")

        if ctype not in [mime_json, mime_msgpack, mime_zlib, mime_pickle]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Content-Type value {ctype}, "
                f"supported values are "
                f"{[mime_json, mime_msgpack, mime_zlib, mime_pickle]}",
            )

        if (
            accept
            and accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild
        ):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )

        if accept in mime_wild:
            accept = ctype if ctype != mime_pickle else mime_json

        shm_enabled = os.environ.get("CUOPT_SERVER_SHM", False) in [
            "True",
            "true",
            True,
        ]

        r = BaseResult(rtype=ctype)
        id = r.register_result()

        # Stream the input data.
        # Write to shared memory if enabled, otherwise write to a bytearray
        s = None
        solution = None
        if shm_enabled:
            logging.debug("writing input data to shared memory")
            # Stream data to shared memory
            solution = id + "result"
            s = shared_memory.SharedMemory(create=True, size=sz, name=solution)

            # This unregister prevents Python from managing the shm segment
            # when s goes out of scope.
            unregister(s._name, "shared_memory")
            buf = s.buf
        else:
            buf = bytearray(sz)
            solution = buf

        await get_data(buf, request)
        if s:
            s.close()
        if isinstance(solution, bytearray):
            solution = bytes(solution)

        r.set_result(solution, quiet=True)

        return encode({"reqId": id}, accept)

    except (RequestValidationError, ValidationError) as e:
        return encode(validation_exception_handler(e), accept)

    except HTTPException as e:
        return encode(http_exception_handler(e), accept)

    except Exception as e:
        return encode(exception_handler(e), accept)


@app.delete(
    "/cuopt/solution/{id}",
    description="Note: for use with self-hosted cuOpt instances.  "
    "Delete a solution by id. The 'id' is the reqId value returned "
    "from a POST to /cuopt/request or /cuopt/solution.",
    summary="Delete a solution by id (self-hosted)",
    responses=DeleteResponse,
)
def deletesolution(
    id: str = Path(
        ...,
        description="ID of the solution to delete. ",
    ),
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. "
        "This applies to exception messages returned by this request.",
    ),
):
    try:
        if not accept:
            accept = mime_json

        if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )
        get_solution_for_id(id, delete=True)
        return Response(status_code=200)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.delete(
    "/cuopt/request/{id}",
    description="Note: for use with self hosted cuOpt instances. "
    "Delete a request (either a cached request or a request to be solved). "
    "The 'id' is the reqId value returned from a POST to /cuopt/request.",
    summary="Delete a request by id (self-hosted)",
    response_model=DeleteRequestModel,
    responses=ValidationErrorResponse,
)
def deleterequest(
    id: str = Path(
        ...,
        description="ID of the request to delete. "
        "The wildcard ID '*' will match all requests.",
    ),
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. ",
    ),
    running: Optional[bool] = Query(
        default=None,
        description="If set to True, the request will be aborted if it is "
        "currently running. Defaults to True if a specific request id is "
        "given and running, queued, and cached are all unspecified, "
        "otherwise defaults to False.",
    ),
    queued: Optional[bool] = Query(
        default=None,
        description="If set to True, the request will be aborted if it is "
        "queued. Defaults to True if a specific request id is given and "
        "running, queued, and cached are all unspecified, otherwise "
        "defaults to False.",
    ),
    cached: Optional[bool] = Query(
        default=None,
        description="If set to True, the request will be aborted if it is "
        "cached. Defaults to True if a specific request id is given and "
        "running, queued, and cached are all unspecified, otherwise "
        "defaults to False.",
    ),
):
    logging.info(f"Deleting id {id}")

    try:
        if not accept:
            accept = mime_json

        if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )

        # If id is '*' or any flag is explicitly set,
        # require all to be explicitly set
        if id == "*" or (
            queued is not None or running is not None or cached is not None
        ):
            running = running if running is not None else False
            queued = queued if queued is not None else False
            cached = cached if cached is not None else False

        else:
            # Specific id given and no flags, set all to True
            queued = running = cached = True

        if id == "*" and not (queued or running or cached):
            raise HTTPException(
                status_code=400,
                detail="Request id is wildcard but no flags set",
            )

        result = {"cached": 0, "queued": 0, "running": 0}

        # Request might be a cache id, or it might be a job,
        # but it can't be both
        if cached:
            result["cached"] = delete_cache_entry(id)
            if result["cached"] != 0 and id != "*":
                return encode(result, accept)

        # If neither flag is set there is nothing to do
        if running or queued:
            if id == "*":
                result["queued"], result["running"] = abort_all(
                    abort_queue, abort_list, running, queued
                )
            else:
                result["queued"], result["running"] = abort_by_id(
                    id, abort_queue, abort_list, running, queued
                )
        return encode(result, accept)

    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


def getsolutionbody(id, accept, delete, warmstart=False):
    try:
        if not accept:
            accept = mime_json

        if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )

        result, file_result, mime_type = None, None, None
        if warmstart:
            result, mime_type = get_warmstart_data_for_id(id)
        else:
            result, file_result, mime_type = get_solution_for_id(
                id, delete=delete
            )
        if accept in mime_wild:
            accept = mime_type if mime_type else mime_json

        if file_result:
            result = file_result
            result["format"] = get_format(mime_type)

        elif not result:
            # job is not done yet
            result = {"reqId": id}
            mime_type = None

        r = None
        if mime_type and (
            isinstance(result, str) or isinstance(result, bytes)
        ):
            s = None
            if isinstance(result, str):
                s = shared_memory.SharedMemory(name=result)
                d = bytes(s.buf)
            else:
                d = result
            try:
                if accept in [mime_type] + mime_wild:
                    logging.debug(f"job_result returning {mime_type}")
                    r = Response(content=d, media_type=mime_type)
                elif mime_type == mime_json:
                    logging.debug("job_result decode result as json")
                    result = json.loads(d)
                elif mime_type == mime_zlib:
                    logging.debug(
                        "job_result decode result as zlib compressed json"
                    )
                    result = json.loads(zlib.decompress(d))
                else:
                    logging.debug("job_result decode result as msgpack")
                    result = msgpack.loads(d, strict_map_key=False)
            finally:
                if delete and s:
                    s.unlink()
        if not r:
            # A JSONResponse holding an error will be returned here
            r = encode(result, accept, job_result=True)
        return r

    except HTTPException as e:
        return encode(http_exception_handler(e), accept)
    except Exception as e:
        return encode(exception_handler(e), accept)


@app.get(
    "/cuopt/solution/{id}",
    description="Note: for use with cuOpt self-hosted instances. "
    "Get a solution by id. The 'id' is the reqId value returned from "
    "a POST to /cuopt/request or /cuopt/solution. If the solution "
    "is generated by a POST to /cuopt/request and the request has "
    "not yet completed, the reqId value will be returned and "
    "can be used to continue polling.",
    summary="Get a solution by id (self-hosted)",
    response_model=Union[
        SolutionModelWithId,
        SolutionModelInFile,
        IdModel,
    ],
    responses=SolutionResponse,
)
def getsolution(
    id: str,
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. "
        "If a wildcard is used, the accept mime_type will be set "
        "to the content_type mime_type of the original request.",
    ),
):
    return getsolutionbody(id, accept, False)


@app.get(
    "/cuopt/solution/{id}/warmstart",
    description="Note: for use with cuOpt self-hosted instances. "
    "Get pdlp warmstart data from solution by id. The 'id' is the reqId "
    "value returned from a POST to /cuopt/request. ",
    summary="Get pdlp warmstart data by id (self-hosted)",
    include_in_schema=False,
)
def getwarmstart(
    id: str,
):
    return getsolutionbody(id, mime_msgpack, False, True)


@app.get(
    "/cuopt/request/{id}",
    description="Note: for use with self-hosted cuOpt instances. "
    "Check the status of a request. ",
    summary="Check the status of a request by id (self-hosted)",
    response_model=RequestStatusModel,
    responses=RequestResponse,
)
def getrequest(
    id: str,
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. "
        "If a wildcard is used, the accept mime_type will be set "
        "to the content_type mime_type of the original request.",
    ),
):
    try:
        return encode(status_by_id(id, abort_list), accept)
    except HTTPException as e:
        return encode(http_exception_handler(e), accept)


@app.post(
    "/cuopt/request",
    description=(
        "Note: This endpoint is for use with self-hosted cuOpt instances. "
        "Takes VRP/LP/MILP data and options at once, submits any type of cuOpt problem and returns the request id."  # noqa
    ),
    # This is for validation by Pydantic
    response_model=IdModel,
    # This is for response examples and schema
    responses=IdResponse,
    summary="Solve a cuOpt problem (self-hosted)",
    # This form is necessary to allow multiple schemas for the
    # possible inputs, with multiple examples
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"oneOf": [vrpschema, lpschema]},
                    "examples": {
                        "VRP request": {"value": vrp_example_data},
                        "LP request": {"value": lp_example_data},
                    },
                },
                "application/vnd.msgpack": {
                    "schema": {"type": "string", "format": "byte"},
                    "examples": {
                        "VRP request compressed with msgpack": {
                            "value": vrp_msgpack_example_data
                        },
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
    cache: Optional[bool] = Query(
        default=False,
        description="If set to True, the data for the request "
        "will be cached and a request id returned.",
    ),
    reqId: Optional[str] = Query(
        default=None,
        description="If set, this request will use the cached data "
        "from the request identified by reqId.",
    ),
    initialId: Optional[List[str]] = Query(
        default=None,
        description="Note: Only applicable to routing. "
        "If set, the solutions identified by id "
        "will be used by the solver as initial solutions for this request",
    ),
    warmstartId: Optional[str] = Query(
        default=None,
        description="If set, the warmstart data in solution identified by id "
        "will be used by the solver as warmstart data for this request. "
        "Enabled for single LP problem, not enabled for Batch LP",
    ),
    validation_only: Optional[bool] = Query(
        default=False,
        description="If set to True, input will be validated, if input is valid, returns a successful message, else returns an error.",  # noqa
    ),
    incumbent_solutions: Optional[bool] = Query(
        default=False,
        description="If set to True, MIP problems will produce incumbent solutions that can be retrieved from /cuopt/solution/{id}/incumbents",  # noqa
    ),
    solver_logs: Optional[bool] = Query(
        default=False,
        description="If set to True, MIP problems will produce detailed solver logs that can be retrieved from /cuopt/log/{id}",  # noqa
    ),
    cuopt_data_file: str = Header(
        default=None,
        description="Name of data file to process in the "
        "server's CUOPT_DATA_DIR when using the local file feature",
    ),
    cuopt_result_file: str = Header(
        default=None,
        description="Result file name if output dir is enabled "
        "and size >= maxresult",
    ),
    client_version: str = Header(
        default=None,
        description="cuOpt client version. "
        "Set to 'custom' to skip version check",
    ),
    accept: str = Header(
        default="application/json",
        description="Supported result mime_types are 'application/json', "
        "'application/vnd.msgpack', 'application/zlib', and "
        "standard mime_type wildcards. "
        "If a wildcard is used, the accept mime_type will be set "
        "to the content_type mime_type",
    ),
    content_type: str = Header(
        default="application/json",
        description="Supported content mime_types are 'application/json', "
        "'application/vnd.msgpack', and 'application/zlib'",
    ),
    content_length: int = Header(),
):
    if app_exit.is_set():
        raise HTTPException(status_code=500, detail="cuOpt is shutting down")

    # just rename these for compat with existing code
    ctype = content_type
    sz = content_length

    if cuopt_data_file is None:
        cuopt_data_file = ""
    if cuopt_result_file is None:
        cuopt_result_file = ""
    if client_version is None:
        client_version = ""

    # TODO should we keep the stuff that limits
    # simultaneous connections for sync?
    # What is the purpose really?

    warnings = check_client_version(client_version)

    try:
        # Skips the job if health status is bad
        health()

        if cuopt_data_file and (cache or reqId):
            raise HTTPException(
                status_code=422,
                detail="Query parameters 'cache' and 'reqId' "
                "are mutually exclusive with header value 'cuopt-data-file'",
            )

        if cache and (reqId or initialId):
            raise HTTPException(
                status_code=422,
                detail="Query parameters 'cache' "
                "and 'reqId' or 'initialId' are mutually exclusive",
            )

        sz = int(sz)
        if sz == 0 and not reqId and not cuopt_data_file:
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

        if (
            accept
            and accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild
        ):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported values are {[mime_json, mime_msgpack, mime_zlib]}",
            )

        if accept in mime_wild:
            accept = ctype if ctype != mime_pickle else mime_json

        file_path = ""
        if cuopt_data_file:
            logging.info("Received file data")
            logging.debug(f"cuopt_data_file: '{cuopt_data_file}'")
            file_path = validate_file_path(cuopt_data_file)

        shm_enabled = os.environ.get("CUOPT_SERVER_SHM", False) in [
            "True",
            "true",
            True,
        ]

        # ... otherwise we need a BinaryJobResult
        resultdir, maxresult, mode = settings.get_result_dir()
        result_file = get_output_name(
            resultdir, cuopt_data_file, cuopt_result_file
        )
        r = BinaryJobResult(
            reqId, accept, result_file, resultdir, maxresult, mode
        )
        id = r.register_result()
        if cache:
            # If cache is set we're going to store the job but we
            # need to validate it as well. Use the same id ...
            add_cache_entry(id, ctype)
            validation_only = True

        # TODO do we need a maxsize on allowed requests?
        # do we need to track outstanding requests and memory consumption?
        # should this be governed by an optional env var?
        now = time.time()

        # Get the content_type and data from the cache if reqId is set
        # If shared memory is enabled, reqId will name the shared memory
        # segment holding the data and data_bytes will be None
        if reqId:
            ctype, data_bytes = get_cache_content_type(reqId)
            if not ctype:
                r.unregister_result()
                raise HTTPException(
                    status_code=422,
                    detail=f"Requtest id '{reqId}' does not exist",
                )

        # if reqId is not set and we're not using file_path, stream in the data
        elif not file_path:

            # Stream the input data.
            # Write to shared memory if enabled, otherwise write to a bytearray
            s = None
            data_bytes = None
            if shm_enabled:
                logging.debug("writing input data to shared memory")
                # Stream data to shared memory.
                # Data will be accessed through named shm segment "id"
                s = shared_memory.SharedMemory(create=True, size=sz, name=id)

                # This unregister prevents Python from managing the shm segment
                # when s goes out of scope. The segment has to remain so that
                # the solver can read the data.
                unregister(s._name, "shared_memory")
                buf = s.buf
            else:
                buf = bytearray(sz)
                data_bytes = buf  # save this reference for later

            await get_data(buf, request)
            if s:
                s.close()
            elif cache and data_bytes:
                # If shared memory is not enabled, save the byte array
                update_cache_entry(id, data_bytes)

        # Go find a previous solution with this id and use it as
        # as an initial solution for the solver
        init_sols = []
        warmstart_data = None
        if initialId:
            logging.debug(f"initialId {initialId}")
            for initid in initialId:
                res, _, mime_type = get_solution_for_id(initid, delete=False)

                # Add the initial id to the job
                init_sols.append((mime_type, res))

        if warmstartId:
            logging.debug(f"warmstartId {warmstartId}")
            warmstart_data = get_warmstart_data_for_id(warmstartId)
            if warmstart_data[0] is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Warmstart data for id '{warmstartId}' not found",
                )

        logging.debug(f"time to receive data {time.time() - now}")

        if file_path:
            job = SolverBinaryJobPath(
                id,
                warnings,
                accept,
                file_path,
                validator_enabled=validation_only,
                init_sols=init_sols,
                incumbents=incumbent_solutions,
                solver_logs=solver_logs,
                warmstart_data=warmstart_data,
            )

        else:
            job = SolverBinaryJob(
                id,
                warnings,
                accept,
                ctype,
                shm_reference=reqId,
                validator_enabled=validation_only,
                data_bytes=data_bytes,
                init_sols=init_sols,
                warmstart_data=warmstart_data,
                incumbents=incumbent_solutions,
                solver_logs=solver_logs,
            )

        result, file_result = wait_for_job(r, job, 0)

        if file_result is not None:
            result = file_result
            result["format"] = get_format(accept)

        # Result equal to None means job is still running
        if result is None:
            result = {"reqId": id}

        # TODO do we need a better indicator of a shm result?
        r = None
        if isinstance(result, str):
            s = shared_memory.SharedMemory(name=result)
            r = Response(content=bytes(s.buf), media_type=accept)
        elif isinstance(result, bytes):
            r = Response(content=result, media_type=accept)
        if r:
            return r

        # Exception from the solver, or a request id as a dictionary
        return encode(result, accept, job_result=True)

    except (RequestValidationError, ValidationError) as e:
        return encode(validation_exception_handler(e), accept)

    except HTTPException as e:
        return encode(http_exception_handler(e), accept)

    except Exception as e:
        return encode(exception_handler(e), accept)


async def get_data(buf, request):
    pos = 0
    try:
        async for chunk in request.stream():
            buf[pos : pos + len(chunk)] = chunk
            pos = pos + len(chunk)
    except Exception:
        print("exception in get_data")


async def get_body(request: Request):
    return await request.body()


# Sync CUOPT SERVICE ENDPOINT
@app.post(
    "/cuopt/cuopt",
    description=(
        "Note: This is for the managed service, and users will never call "
        "this API directly. "
        "The description is here to illustrate the format of a cuOpt request "
        "to be sent through the managed service client, and the format of a "
        "response. If you are self-hosting cuOpt, do not use this endpoint, "
        "use /cuopt/request instead. Takes all the data and options at once, "
        "solves any type of cuOpt problem and returns result."
    ),
    include_in_schema=False,
    summary="Managed Service Endpoint",
    response_model=Union[EmptyResponseModel, SolutionModel],
    responses=ManagedRequestResponse,
    # include_in_schema=False,
    # This form is necessary to allow multiple literal examples to
    # be added to the Swaggger and redoc UIs. Noe the schema is
    # taken by fastapi from the cuoptData parameter.
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": cuoptdataschema,
                    "examples": {
                        "VRP request": {"value": managed_vrp_example_data},
                        "LP request": {"value": managed_lp_example_data},
                    },
                },
            },
            "required": True,
        }
    },
)
def cuopt(request: Request, data_bytes: bytes = Depends(get_body)):

    headers = dict(request.headers)

    accept = headers.get("accept", mime_json)
    content_type = headers.get("content-type", mime_json)
    NVCF_ASSET_DIR = headers.get("nvcf-asset-dir", "")
    NVCF_FUNCTION_ASSET_IDS = headers.get("nvcf-function-asset-ids", "")
    NVCF_LARGE_OUTPUT_DIR = headers.get("nvcf-large-output-dir", "")
    NVCF_NCAID = headers.get("nvcf-ncaid", "")
    NVCF_REQID = headers.get("nvcf-reqid", "")
    NVCF_MAX_RESPONSE_SIZE_BYTES = headers.get(
        "nvcf-max-response-size-bytes", ""
    )

    begin_time = time.time()
    try:
        if app_exit.is_set():
            raise HTTPException(
                status_code=500, detail="cuOpt is shutting down"
            )

        health()

        # We allow msgpack for local testing with direct calls
        # but it is not usable on NVCF
        if content_type not in [mime_json, mime_msgpack]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Content-Type value {content_type}, "
                f"supported values is {mime_json}",
            )

        if accept in mime_wild:
            accept = mime_json

        # Same, allow msgpack returns for local testing
        if accept not in [mime_json, mime_msgpack]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Accept value {accept}, "
                f"supported value is {mime_json}",
            )

        warnings = []
        set_ncaid(NVCF_NCAID)
        set_requestid(NVCF_REQID)

        logging.debug(
            message(
                f"NVCF_ASSET_DIR: '{NVCF_ASSET_DIR}', "
                f"NVCF_FUNCTION_ASSET_IDS: '{NVCF_FUNCTION_ASSET_IDS}', "
                f"NVCF_MAX_RESPONSE_BYTES: {NVCF_MAX_RESPONSE_SIZE_BYTES}"
                f"NVCF_LARGE_OUTPUT_DIR: {NVCF_LARGE_OUTPUT_DIR}"
            )
        )

        # Create a NVCFJobResult to hold the solution
        if os.environ.get("CUOPT_SERVER_TEST_LARGE_RESULT", False):
            maxresult = 0
        else:
            try:
                maxresult = int(NVCF_MAX_RESPONSE_SIZE_BYTES) / 1000
            except Exception:
                _, maxresult, _ = settings.get_result_dir()
        r = NVCFJobResult(NVCF_LARGE_OUTPUT_DIR, maxresult, accept)
        id = r.register_result()

        if NVCF_FUNCTION_ASSET_IDS:
            file_path = os.path.join(
                NVCF_ASSET_DIR, NVCF_FUNCTION_ASSET_IDS.split(",")[0]
            )
            job = SolverBinaryJobPath(
                id,
                warnings,
                accept,
                file_path,
                request_filter=True,
                response_id=False,
                wrapper_data={
                    "content_type": content_type,
                    "data": data_bytes,
                },
            )
        else:
            job = SolverBinaryJob(
                id,
                warnings,
                accept,
                content_type,
                request_filter=True,
                response_id=False,
                data_bytes=data_bytes,
            )

        end_time = time.time()
        # In the case of large results, the result processor needs to
        # include perf times when it writes the file. So we need to
        # set it in the job now.
        job.set_initial_etl_time(end_time - begin_time)
        job.set_nvcf_ids(NVCF_NCAID, NVCF_REQID)

        # Get the result.
        res, file_result = wait_for_job(r, job)

        # Leave this here. For managed, we cannot cache jobs
        # or results because the NVCF worker only makes a
        # single synchronous call to cuopt. Additionally, workers
        # may be spun down and redeployed and so the API needs to
        # be stateless
        r.unregister_result()

        if isinstance(res, JSONResponse):
            raise SolverException(res)

        # In the NVCF case, file_result will be {} and the result
        # is passed back in a header
        if file_result is not None:
            return file_result

        # TODO do we need a better indicator of a shm result?
        r = None
        if isinstance(res, str):
            s = shared_memory.SharedMemory(name=res)
            r = Response(content=bytes(s.buf), media_type=accept)

            # Similar to unregister_result above, this unlink
            # of shm must remain because for managed we do not cache
            # jobs or results
            s.unlink()
        elif isinstance(res, bytes):
            r = Response(content=res, media_type=accept)

        if r:
            return r

        # Exception from the solver, or a request id as a dictionary
        return encode(res, accept, job_result=True)

    except SolverException as e:
        return encode(e.response, accept, job_result=True)

    except (RequestValidationError, ValidationError) as e:
        return encode(validation_exception_handler(e), accept)

    except HTTPException as e:
        return encode(http_exception_handler(e), accept)

    except Exception as e:
        return encode(exception_handler(e), accept)


def terminate_pid(pid):
    try:
        p = psutil.Process(pid)
        p.terminate()
    except Exception:
        pass


def heartbeat(parent, results_thread, app_exit, jobs_marked_done):
    while True:
        time.sleep(0.5)
        if not results_thread.is_alive():
            logging.info(
                "heartbeat thread detected receive_results has exited"
            )
            app_exit.set()
            try:
                Shutdown().process()
            except Exception:
                pass
            jobs_marked_done.set()
            break
    logging.info("heartbeat thread terminating webserver")
    terminate_pid(parent)
    logging.info("heartbeat thread finished")


def receive_results(
    parent, app_exit, results_queue, abort_list, jobs_marked_done
):
    try:
        while True:
            try:
                obj = results_queue.get(timeout=1)
            except queue.Empty:
                if app_exit.is_set():
                    obj = Shutdown()
                else:
                    continue

            ncaid, reqid = obj.get_nvcf_ids()
            set_ncaid(ncaid)
            set_requestid(reqid)
            try:
                obj.process(abort_list)
            except Exception:
                pass
            if isinstance(obj, Shutdown):
                jobs_marked_done.set()
                logging.info("receive_results terminating webserver")
                terminate_pid(parent)
                break

    except Exception:
        logging.info("receive_results got an exception")
    logging.info("results thread finished")


def run_server(
    app_exit,
    results_queue,
    job_queue,
    abort_queue,
    abort_list,
    jobs_marked_done,
    ip="0.0.0.0",
    port=5000,
    log_level="debug",
    log_file="",
    log_max=0,
    log_backup=0,
    ssl_certfile="",
    ssl_keyfile="",
    **kwargs,
):
    """
    Initialize and run the server

    Parameters
    ----------
    ip : IP-address
        IP address on which the server should run, by default it would be
        0.0.0.0
    port : Integer
        Port number on which the service would be listening, default is 5000
    log_level : String
        Log level to be set for the server, default is "debug".
        Options are critical, error, warning, info, debug, trace.

    Note: You can also pass additional parameters through kwargs which are
    supported by uvicorn.run.
    """
    import signal

    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    set_queues_and_flags(app_exit, job_queue, abort_queue, abort_list)

    t = threading.Thread(
        target=receive_results,
        args=(
            os.getpid(),
            app_exit,
            results_queue,
            abort_list,
            jobs_marked_done,
        ),
    )
    t.start()

    # Since the results processing thread could conceivably spend a lot of time
    # writing results or may get an exception, monitor it from another thread
    h = threading.Thread(
        target=heartbeat, args=(os.getpid(), t, app_exit, jobs_marked_done)
    )
    h.start()

    # Set up health check sema
    health_check.health_init()

    # Replace the uvilog handlers with null handlers and allow it to
    # propagate messages to our root handler
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
        "cuopt_server.webserver:app",
        host=ip,
        port=port,
        log_config=uvi_config,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        **kwargs,
    )
    logging.info("webserver finished")
    results_queue.close()
    job_queue.close()
    abort_queue.close()
    t.join(timeout=0.2)
    h.join(timeout=0.2)
    jobs_marked_done.set()

    # Check for leftover threads. Sometimes queue feeder threads do not exit
    thread_names = {
        t.name for t in threading.enumerate() if t.name != "MainThread"
    }
    if len(thread_names) > 0:
        logging.info("not all threads exited in webserver, killing")
        psutil.Process(os.getpid()).kill()
