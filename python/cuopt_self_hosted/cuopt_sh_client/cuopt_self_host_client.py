# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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
import pickle
import threading
import time
import zlib
from enum import Enum
from uuid import UUID

import cuopt_mps_parser
import msgpack
import msgpack_numpy
import numpy as np
import requests
import solution

from . import _version
from .thin_client_solver_settings import ThinClientSolverSettings

msgpack_numpy.patch()


def get_version():
    """
    Return client version.
    """
    return f"{_version.__version__} (local)"


# pickle_first will cause pickle to be tried first for serialization.
# If it fails, msgpack will be tried.
# The default order is to try msgpack first, and then pickle if it
# fails.
# msgpack has array size limits that may prevent serialization
# from working, which is why we fail over to pickle. However,
# since msgpack may not fail fast, if a user knows that a dataset
# is too big for msgpack, we allow them to change the order and
# use pickle first
pickle_first = os.getenv("CUOPT_PREFER_PICKLE", False) in [
    "True",
    "true",
    True,
]

# use_zlib causes zlib to be used to compress JSON, and msgpack/pickle
# are ignored. use_zlib takes precedence.
use_zlib = os.getenv("CUOPT_USE_ZLIB", False) in ["True", "true", True]

if use_zlib:

    def do_serialize(data):
        log.debug("using zlib")
        return (
            zlib.compress(
                bytes(json.dumps(data), encoding="utf-8"), zlib.Z_BEST_SPEED
            ),
            "application/zlib",
        )

elif pickle_first:

    def do_serialize(data):
        try:
            log.debug("trying pickle")
            return pickle.dumps(data), "application/octet-stream"
        except Exception:
            pass
        log.debug("pickle failed, using msgpack")
        return msgpack.dumps(data), "application/vnd.msgpack"

else:

    def do_serialize(data):
        try:
            log.debug("trying msgpack")
            return msgpack.dumps(data), "application/vnd.msgpack"
        except Exception:
            pass
        log.debug("msgpack failed, using pickle")
        return pickle.dumps(data), "application/octet-stream"


log_fmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)


class mime_type(Enum):
    JSON = "application/json"
    ZLIB = "application/zlib"
    MSGPACK = "application/vnd.msgpack"
    WILDCARD = "application/*"


def set_log_level(level):
    log.setLevel(level)


def load_data(file_path, lp=False):
    with open(file_path, "rb") as f:
        raw_data = f.read()

        # Check the file extension to determine the content type
        ext = file_path.split(".")[-1] if "." in file_path else ""
        if ext == "zlib":
            content_type = "application/zlib"
        elif ext == "msgpack":
            content_type = "application/vnd.msgpack"
        elif ext in ["json", ""]:
            if not ext:
                log.info("No file extension given, assuming JSON")
            content_type = "application/json"
        elif ext == "pickle":
            content_type = "application/octet-stream"
        else:
            raise ValueError(
                f"File extension {ext} is unsupported. "
                "Supported file extensions are "
                ".json, .zlib, .msgpack, or .pickle"
            )
        return raw_data, content_type


def is_uuid(cuopt_problem_data):
    try:
        _ = UUID(cuopt_problem_data, version=4)
        return True
    except Exception:
        return False


def _mps_parse(LP_problem_data, solver_config):

    if isinstance(LP_problem_data, cuopt_mps_parser.parser_wrapper.DataModel):
        model = LP_problem_data
        log.debug("Received Mps parser DataModel object")
    else:
        t0 = time.time()
        model = cuopt_mps_parser.ParseMps(LP_problem_data)
        parse_time = time.time() - t0
        log.debug(f"mps_parsing time was {parse_time}")
    problem_data = cuopt_mps_parser.toDict(model, json=use_zlib)

    if type(solver_config) is dict:
        problem_data["solver_config"] = solver_config
    else:
        problem_data["solver_config"] = solver_config.toDict()

    return problem_data


def create_lp_response(response_dict):
    def create_solution_obj(solver_response):
        sol = solver_response["solution"]
        status = solver_response["status"]
        problem_category = sol["problem_category"]

        # MILP
        if problem_category in [1, 2]:
            solution_obj = solution.Solution(
                problem_category,
                sol["vars"],
                solve_time=sol["solver_time"],
                termination_status=status,
                primal_solution=np.array(sol["primal_solution"]),
                reduced_cost=np.array(sol["reduced_cost"]),
                primal_objective=sol["primal_objective"],
                mip_gap=sol["milp_statistics"]["mip_gap"],
                solution_bound=sol["milp_statistics"]["solution_bound"],
                presolve_time=sol["milp_statistics"]["presolve_time"],
                max_constraint_violation=sol["milp_statistics"][
                    "max_constraint_violation"
                ],
                max_int_violation=sol["milp_statistics"]["max_int_violation"],
                max_variable_bound_violation=sol["milp_statistics"][
                    "max_variable_bound_violation"
                ],
                num_nodes=sol["milp_statistics"]["num_nodes"],
                num_simplex_iterations=sol["milp_statistics"][
                    "num_simplex_iterations"
                ],
            )
        else:
            solution_obj = solution.Solution(
                problem_category,
                sol["vars"],
                solve_time=sol["solver_time"],
                termination_status=status,
                primal_solution=np.array(sol["primal_solution"]),
                dual_solution=np.array(sol["dual_solution"]),
                reduced_cost=np.array(sol["reduced_cost"]),
                primal_residual=sol["lp_statistics"]["primal_residual"],
                dual_residual=sol["lp_statistics"]["dual_residual"],
                gap=sol["lp_statistics"]["gap"],
                nb_iterations=sol["lp_statistics"]["nb_iterations"],
                primal_objective=sol["primal_objective"],
                dual_objective=sol["dual_objective"],
                solved_by_pdlp=sol["solved_by_pdlp"],
            )
        return status, solution_obj

    try:
        solver_responses = response_dict["response"]["solver_response"]
        if type(solver_responses) is list:
            status_list = []
            solution_obj_list = []
            for solver_response in solver_responses:
                status, solution_obj = create_solution_obj(solver_response)
                status_list.append(status)
                solution_obj_list.append(solution_obj)
            final_response = {
                "status": status_list,
                "solution": solution_obj_list,
            }
            response_dict["response"]["solver_response"] = final_response
        else:
            status, solution_obj = create_solution_obj(solver_responses)
            response_dict["response"]["solver_response"]["status"] = status
            response_dict["response"]["solver_response"][
                "solution"
            ] = solution_obj
        return response_dict
    except Exception:
        return response_dict


class CuOptServiceSelfHostClient:
    """
    This version of the CuOptServiceClient is an interface
    with a self hosted version of the cuOpt core service.
    This client allows users to make calls to a self hosted
    instance of the cuOpt service at a specific ip and port.

    It closely emulates the interface of the managed service client,
    however it does not implement most of the managed service-specific
    features required for interaction with NVIDIA Cloud Functions.

    Parameters
    ----------
    ip : str
        The IP address of the cuOpt service. Defaults to 0.0.0.0
    port : str
        The port of the cuOpt service. Set to empty string or None
        to omit the port from cupot urls (this is useful with exported
        services in kubernetes for example). Defaults to 5000.
    use_https : boolean
        Use HTTPS to communicate with server in secured way.
    self_signed_cert : str
            A complete path to
            self signed certificate. If it's a standard certificate,
            then no need to provide anything.
    polling_interval : int
            The duration in seconds between
            consecutive polling attempts. Defaults to 1.
    request_excess_timeout : int
            Note: Deprecated, Use polling_timeout instead
    only_validate : boolean
            Only validates input. Defaults to False.
    polling_timeout : int
            The time in seconds that the client will poll for a result
            before exiting and returning a request id. The request id may be
            polled again in a call to repoll(). If set to None, the client
            will never timeout and will poll indefinitely. Defaults to 600.
    timeout_exception : boolean
            If True, the client returns a TimeoutError exception
            if polling_timeout seconds passes before a solution is returned.
            The value of the exception contains JSON giving the request id
            so that repoll() may be called to poll again for a result.
            If False, the client returns a dictionary containing the
            repoll information with no exception. Defaults to True.
    result_type : enum
            Supported result mime_types are mime_type.JSON,
            mime_type.MSGPACK, mime_type.ZLIB and mime_type.WILDCARD. If a
            wildcard is used, the result mime_type will be set to the
            content_type mime_type of the original request.
            If not provided, result_type defaults to mime_type.MSGPACK
    """

    # Initialize the CuOptServiceSelfHostClient
    def __init__(
        self,
        ip: str = "0.0.0.0",
        port: str = "5000",
        use_https: bool = False,
        self_signed_cert="",
        polling_interval=1,
        request_excess_timeout=None,
        only_validate=False,
        polling_timeout=600,
        timeout_exception=True,
        result_type=mime_type.MSGPACK,
    ):
        self.timeout_exception = timeout_exception
        self.ip = ip
        self.port = port
        self.only_validate = only_validate
        self.accept_type = result_type

        self.protocol = "https" if use_https else "http"
        self.verify = False
        if use_https is True:
            if len(self_signed_cert) > 0:
                self.verify = self_signed_cert
            else:
                self.verify = True
        self.loggedbytes = 0

        # Allow port to be passed as None or "" for cases where the service
        # name is exported from kubernetes (for example) and the port is
        # inherent in the hostname
        if self.port:
            self.request_url = f"{self.protocol}://{self.ip}:{self.port}/cuopt/request"  # noqa
            self.log_url = f"{self.protocol}://{self.ip}:{self.port}/cuopt/log"
            self.solution_url = f"{self.protocol}://{self.ip}:{self.port}/cuopt/solution"  # noqa
        else:
            self.request_url = (
                f"{self.protocol}://{self.ip}/cuopt/request"  # noqa
            )
            self.log_url = f"{self.protocol}://{self.ip}/cuopt/log"
            self.solution_url = (
                f"{self.protocol}://{self.ip}/cuopt/solution"  # noqa
            )

        self.polling_interval = polling_interval
        self.timeout = (
            request_excess_timeout
            if request_excess_timeout is not None
            else polling_timeout
        )

    def _get_response(self, response):
        if response.headers["content-type"] == mime_type.JSON.value:
            log.debug("reading response as json")
            response = response.json()
        elif response.headers["content-type"] == mime_type.ZLIB.value:
            log.debug("reading response as zlib")
            response = json.loads(zlib.decompress(response.content))
        else:
            log.debug("reading response as msgpack")
            response = msgpack.loads(response.content, raw=False)
        return response

    def _handle_request_exception(self, response, reqId=None):
        r = self._get_response(response)
        complete = "error_result" in r and r["error_result"] is True
        msg = r.get("error", r)
        err = f"cuOpt Error: {response.reason} - {response.status_code}: {msg}"
        if reqId:
            err += f"\nreqId: {reqId}"
        return err, complete

    def _get_logs(self, reqId, logging_callback):
        if logging_callback is None or not callable(logging_callback):
            return
        try:
            headers = {"Accept": self.accept_type.value}
            params = {"frombyte": self.loggedbytes}
            response = requests.get(
                self.log_url + f"/{reqId}",
                verify=self.verify,
                headers=headers,
                params=params,
                timeout=30,
            )

            # File has not been created yet
            if response.status_code == 404:
                return

            response = self._get_response(response)
            if "error" in response:
                log.info(response["error"])

            elif response["log"]:
                logging_callback(response["log"])
                self.loggedbytes = response["nbytes"]

        except requests.exceptions.HTTPError as e:
            log.debug(str(e))

    def _get_incumbents(self, reqId, incumbent_callback):
        if incumbent_callback is None or not callable(incumbent_callback):
            return
        try:
            headers = {"Accept": self.accept_type.value}
            response = requests.get(
                self.solution_url + f"/{reqId}/incumbents",
                verify=self.verify,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            response = self._get_response(response)
            for ic in response:
                # check for sentinel marking finished
                if not ic["solution"] and ic["cost"] is None:
                    log.debug("saw sentinel incumbent")
                    return True
                incumbent_callback(ic["solution"], ic["cost"])
            return False
        except requests.exceptions.HTTPError:
            # We should return True if we get a
            # no such job error
            r = self._get_response(response)

            # This should never happen unless someone deletes the
            # job underneath us
            if "error" in r and "does not exist" in r["error"]:
                return True
            return False

    def _poll_request(
        self, response, delete, incumbent_callback=None, logging_callback=None
    ):

        log_t = None
        inc_t = None
        complete = False
        reqId = None
        done = threading.Event()

        def poll_for_logs(reqId, logging_callback, done):
            while not done.is_set():
                self._get_logs(reqId, logging_callback)
                time.sleep(1)
            # Issue one more call in case logs showed up
            # during sleep
            self._get_logs(reqId, logging_callback)

        def start_log_thread(reqId):
            t = threading.Thread(
                target=poll_for_logs, args=(reqId, logging_callback, done)
            )
            t.start()
            return t

        def poll_for_incumbents(reqId, incumbent_callback, done):
            while True:
                inc_done = self._get_incumbents(reqId, incumbent_callback)
                if inc_done or done.is_set():
                    break
                time.sleep(1)

        def start_inc_thread(reqId):
            t = threading.Thread(
                target=poll_for_incumbents,
                args=(reqId, incumbent_callback, done),
            )
            t.start()
            return t

        def stop_threads(log_t, inc_t, done):
            done.set()
            if log_t:
                log_t.join()
            if inc_t:
                inc_t.join()

        response = self._get_response(response)
        if "reqId" in response:
            reqId = response["reqId"]
            if logging_callback is not None:
                log_t = start_log_thread(reqId)
            if incumbent_callback is not None:
                inc_t = start_inc_thread(reqId)

        poll_start = time.time()
        try:
            while True:
                # just a reqId means the request is still pending
                if not (len(response) == 1 and "reqId" in response):
                    complete = (
                        "response" in response
                        or "result_file" in response
                        or (
                            "error_result" in response
                            and response["error_result"] is True
                        )
                    )
                    break
                if (
                    self.timeout is not None
                    and time.time() - poll_start > self.timeout
                ):
                    if self.timeout_exception:
                        raise TimeoutError(json.dumps(response))
                    else:
                        break

                time.sleep(self.polling_interval)
                reqId = response["reqId"]
                try:
                    log.debug(f"GET {self.solution_url}/{reqId}")
                    headers = {"Accept": self.accept_type.value}
                    response = requests.get(
                        self.solution_url + f"/{reqId}",
                        verify=self.verify,
                        headers=headers,
                        timeout=30,
                    )
                    response.raise_for_status()
                    response = self._get_response(response)
                except requests.exceptions.HTTPError as e:
                    log.debug(str(e))
                    err, complete = self._handle_request_exception(
                        response, reqId
                    )
                    raise ValueError(err)
            return response

        finally:
            stop_threads(log_t, inc_t, done)
            if complete and delete and reqId is not None:
                self._delete(reqId)

    # Send the request to the local cuOpt core service
    def _send_request(
        self,
        cuopt_problem_data,
        filepath,
        cache,
        output,
        initial_ids=[],
        warmstart_id=None,
        delete=True,
        incumbent_callback=None,
        logging_callback=None,
    ):
        def serialize(cuopt_problem_data):
            if isinstance(cuopt_problem_data, dict):
                data, content_type = do_serialize(cuopt_problem_data)
            elif isinstance(cuopt_problem_data, list):
                if all(isinstance(d, str) for d in cuopt_problem_data):
                    # Make this a list of tuples of content_type and
                    # a byte stream, and serialize the whole thing
                    # with mspagck
                    final_list = []
                    for d in cuopt_problem_data:
                        data, content_type = load_data(d)
                        final_list.append((content_type, data))
                    data, content_type = do_serialize(final_list)
                else:
                    data, content_type = do_serialize(cuopt_problem_data)
            else:
                data, content_type = load_data(cuopt_problem_data)
            return data, content_type

        try:
            log.debug(f"POST {self.request_url}")
            headers = {}
            params = {"validation_only": self.only_validate}
            params["cache"] = cache
            params["incumbent_solutions"] = incumbent_callback is not None
            params["initialId"] = initial_ids
            params["solver_logs"] = logging_callback is not None
            params["warmstartId"] = warmstart_id

            if is_uuid(cuopt_problem_data):
                data = {}
                params["reqId"] = cuopt_problem_data
                content_type = "application/json"
            elif filepath:
                headers["CUOPT-DATA-FILE"] = cuopt_problem_data
                data = {}
                content_type = "application/json"
            else:
                data, content_type = serialize(cuopt_problem_data)
            headers["CLIENT-VERSION"] = _version.__version__
            # Immediately return and then poll on the id
            if output:
                headers["CUOPT-RESULT-FILE"] = output
            headers["Content-Type"] = content_type
            headers["Accept"] = self.accept_type.value
            response = requests.post(
                self.request_url,
                params=params,
                data=data,
                headers=headers,
                verify=self.verify,
                timeout=30,
            )
            response.raise_for_status()
            log.debug(response.status_code)
        except requests.exceptions.HTTPError as e:
            log.debug(str(e))
            err, _ = self._handle_request_exception(response)
            raise ValueError(err)
        if cache:
            return self._get_response(response)
        return self._poll_request(
            response, delete, incumbent_callback, logging_callback
        )

    def _cleanup_response(self, res):
        if "warnings" in res:
            for w in res["warnings"]:
                log.warning(w)
            del res["warnings"]
        if "notes" in res:
            for n in res["notes"]:
                log.info(n)
            del res["notes"]
        return res

    # Get optimized routes for the given cuOpt problem instance
    def get_optimized_routes(
        self,
        cuopt_problem_json_data,
        filepath=False,
        cache=False,
        output="",
        delete_solution=True,
        initial_ids=[],
    ):
        """
        Get optimized routing solution for a given problem.

        Parameters
        ----------
        cuopt_problem_json_data : dict or str
            This is either the problem data as a dictionary or the
            path of a file containing the problem data as JSON or
            the reqId of a cached cuopt problem data.
            Please refer to the server doc for the
            structure of this dictionary.
        filepath : boolean
            Indicates that cuopt_problem_json_data
            is the relative path of a cuopt data file under the server's
            data directory. The data directory is specified when the server
            is started (see the server documentation for more detail).
            Defaults to False.
        output : str
            Optional name of the result file.
            If the server has been configured to write results to files and
            the size of the result is greater than the configured
            limit, the server will write the result to a file with
            this name under the server's result directory (see the
            server documentation for more detail). Defaults to a
            name based on the path if 'filepath' is True,
            or a uuid if 'filepath' is False.
        delete_solution: boolean
            Delete the solution when it is returned. Defaults to True.
        """
        if filepath and cuopt_problem_json_data.startswith("/"):
            log.warn(
                "Path of the data file on the server was specified, "
                "but an absolute path was given. "
                "Best practice is to specify the relative path of a "
                "data file under the CUOPT_DATA_DIR directory "
                "which was configured when the cuopt server was started."
            )

        res = self._send_request(
            cuopt_problem_json_data,
            filepath,
            cache,
            output,
            initial_ids,
            delete=delete_solution,
        )
        return self._cleanup_response(res)

    def get_LP_solve(
        self,
        cuopt_data_models,
        solver_config=ThinClientSolverSettings(),
        cache=False,
        response_type="obj",
        filepath=False,
        output="",
        delete_solution=True,
        warmstart_id=None,
        incumbent_callback=None,
        logging_callback=None,
    ):
        """
        Get linear programming solution for a given problem.

        Parameters
        ----------
        cuopt_data_models :
            Note - Batch mode is only supported in LP and not in MILP

            File path to mps or json/dict/DataModel returned by
            cuopt_mps_parser/list[mps file paths]/list[dict]/list[DataModel].

            For single problem, input should be either a path to mps/json file,
            /DataModel returned by cuopt_mps_parser/ path to json file/
            dictionary.

            For batch problem, input should be either a list of paths to mps
            files/ a list of DataModel returned by cuopt_mps_parser/ a
            list of dictionaries.

            To use a cached cuopt problem data, input should be a uuid
            identifying the reqId of the cached data.
        solver_config : SolverSettings object or Dict
            Contains solver settings including tolerance values.
            See the LP documentation for details on solver settings.
        response_type : str
            Choose "dict" if response should be returned as a dictionary or
            "obj" for Solution object. Defaults to "obj"
        filepath : boolean
            Indicates that cuopt_problem_json_data
            is the relative path of a cuopt data file under the server's
            data directory. The data directory is specified when the server
            is started (see the server documentation for more detail).
            Defaults to False.
        output : str
            Optional name of the result file.
            If the server has been configured to write results to files and
            the size of the result is greater than the configured
            limit, the server will write the result to a file with
            this name under the server's result directory (see the
            server documentation for more detail). Defaults to a
            name based on the path if 'filepath' is True,
            or a uuid if 'filepath' is False.
        delete_solution: boolean
            Delete the solution when it is returned. Defaults to True.
        incumbent_callback : callable
            # Note : Only applicable to MIP

            A callback that will be invoked as incumbent_callback(solution, cost) to # noqa
            receive incumbent solutions from the MIP solver where solution is
            a list of floats and cost is a float. The callback will be invoked
            each time the solver produces an incumbent solution. The LP solver
            will not return any incumbent solutions. Default is None.
        logging_callback: callable
            # Note : Only applicable to MIP

            A callback that will be invoked as logging_callback(solution) to
            receive log lines from the MIP solver. Solution will be
            a list of strings. The LP solver will not return any
            incumbent solutions. Default is None.

        Returns: dict or Solution object.
        """

        if incumbent_callback is not None and not callable(incumbent_callback):
            raise ValueError(
                "incumbent_callback must be callable as "
                "incumbent_callback(solution=[], cost=0.0)"
            )

        if solver_config is None:
            solver_config = ThinClientSolverSettings()

        def read_cuopt_problem_data(cuopt_data_model, filepath):
            if isinstance(cuopt_data_model, dict):
                mps = False
                filepath = False
            else:
                mps = (
                    isinstance(cuopt_data_model, str)
                    and cuopt_data_model.endswith(".mps")
                ) or not isinstance(cuopt_data_model, str)

            if mps:
                if filepath:
                    raise ValueError(
                        "Cannot use local file mode with MPS data. "
                        "Resubmit with filepath=False."
                    )

                cuopt_data_model = _mps_parse(cuopt_data_model, solver_config)

            elif filepath and cuopt_data_model.startswith("/"):
                log.warn(
                    "Path of the data file on the server was specified, "
                    "but an absolute path was given. "
                    "Best practice is to specify the relative path of a "
                    "data file under the CUOPT_DATA_DIR directory "
                    "which was configured when the cuopt server was started."
                )
            return cuopt_data_model

        cuopt_problem_data = None
        if type(cuopt_data_models) is list:
            cuopt_problem_data = []
            for cuopt_data_model in cuopt_data_models:
                problem_data = read_cuopt_problem_data(
                    cuopt_data_model, filepath
                )
                cuopt_problem_data.append(problem_data)
        else:
            cuopt_problem_data = read_cuopt_problem_data(
                cuopt_data_models, filepath
            )

        res = self._send_request(
            cuopt_problem_data,
            filepath,
            cache,
            output,
            delete=delete_solution,
            warmstart_id=warmstart_id,
            incumbent_callback=incumbent_callback,
            logging_callback=logging_callback,
        )

        if response_type == "obj":
            return create_lp_response(self._cleanup_response(res))
        else:
            return self._cleanup_response(res)

    def delete(self, id, running=None, queued=None, cached=None):
        """
        Delete a cached entry by id or abort a job by id.

        Parameters
        ----------
        id : str
            A uuid identifying the cached entry or job to be deleted. The
            wildcard id '*' will match all uuids (filtered by 'running',
            'queued', and 'cached').
        running : bool
            If set to True, the request will be aborted if 'id' is a currently
            running job. Defaults to True if 'id' is a specific uuid and both
            'queued' and 'cached' are unspecified, otherwise False.
        queued : bool
            If set to True, the request will be aborted if 'id' is a currently
            queued job. Defaults to True if 'id' is a specific uuid and both
            'running' and 'cached' are unspecified, otherwise False.
        cached: bool
            If set to True, the request will be aborted if 'id' is a cached
            data entry. Defaults to True if 'id' is a specific uuid and both
            'running' and 'queued' are unspecified, otherwise False.
        """
        try:
            response = requests.delete(
                self.request_url + f"/{id}",
                headers={"Accept": self.accept_type.value},
                params={
                    "running": running,
                    "queued": queued,
                    "cached": cached,
                },
                verify=self.verify,
                timeout=30,
            )
            response.raise_for_status()
            log.debug(response.status_code)
            return self._get_response(response)

        except requests.exceptions.HTTPError as e:
            log.debug(str(e))
            err, _ = self._handle_request_exception(response)
            raise ValueError(err)

    def _delete(self, id):
        try:
            self.delete_solution(id)
        except Exception:
            pass

    def delete_solution(self, id):
        """
        Delete a solution by id.

        Parameters
        ----------
        id : str
            A uuid identifying the solution to be deleted.
        """
        if isinstance(id, dict):
            id = id["reqId"]
        try:
            headers = {"Accept": self.accept_type.value}
            response = requests.delete(
                self.solution_url + f"/{id}",
                headers=headers,
                verify=self.verify,
                timeout=30,
            )
            response.raise_for_status()
            log.debug(response.status_code)

            # Get rid of a log if it exists.
            # It may not so just squash exceptions.
            try:
                response = requests.delete(
                    self.log_url + f"/{id}",
                    verify=self.verify,
                    timeout=30,
                )
            except Exception:
                pass

        except requests.exceptions.HTTPError as e:
            log.warn(f"Deletion of solution {id} failed")
            log.debug(str(e))
            err, _ = self._handle_request_exception(response)
            raise ValueError(err)

    def repoll(self, data, response_type="obj", delete_solution=True):
        """
        Poll for a result when a previous command resulted in
        a timeout. The request id is returned as JSON
        in the result of the original call.

        Parameters
        ----------
        data : str
            A uuid identifying the original request.
            For backward compatibility, data may also be a dictionary
            containing the key 'reqId' where the value is the uuid.
        response_type: str
            For LP problem choose "dict" if response should be returned
            as a dictionary or "obj" for Solution object.
            Defaults to "obj".
            For VRP problem, response_type is ignored and always
            returns a dict.
        delete_solution: boolean
            Delete the solution when it is returned. Defaults to True.
        """
        if isinstance(data, dict):
            data = data["reqId"]
        headers = {"Accept": self.accept_type.value}
        try:
            response = requests.get(
                self.solution_url + f"/{data}",
                verify=self.verify,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.debug(str(e))
            err, complete = self._handle_request_exception(
                response, reqId=data
            )
            if complete and delete_solution:
                self._delete(data)
            raise ValueError(err)

        if response_type == "dict":
            return self._cleanup_response(
                self._poll_request(response, delete_solution)
            )
        else:
            return create_lp_response(
                self._cleanup_response(
                    self._poll_request(response, delete_solution)
                )
            )

    def status(self, id):
        """
        Return the status of a cuOpt server request.

        id : str
            A uuid identifying the solution to be deleted.
        """
        if isinstance(id, dict):
            id = id["reqId"]
        headers = {"Accept": self.accept_type.value}
        try:
            response = requests.get(
                self.request_url + f"/{id}?status",
                verify=self.verify,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.debug(str(e))
            err, _ = self._handle_request_exception(response)
            raise ValueError(err)

        return self._cleanup_response(self._get_response(response))

    def upload_solution(self, solution):
        """
        Store a solution on the server and return a request id.
        This can be used to upload a solution to use as an initial
        solution for a VRP problem (referenced by reqId).

        Parameters
        ----------
        solution:
            A solution in the cuOpt result format. May be a dictionary
            or a file.
        """

        if isinstance(solution, dict):
            data, content_type = do_serialize(solution)
        else:
            data, content_type = load_data(solution)

        headers = {
            "Accept": self.accept_type.value,
            "Content-Type": content_type,
        }
        try:
            response = requests.post(
                self.solution_url,
                verify=self.verify,
                data=data,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.debug(str(e))
            err, _ = self._handle_request_exception(response)
            raise ValueError(err)

        return self._cleanup_response(self._get_response(response))
