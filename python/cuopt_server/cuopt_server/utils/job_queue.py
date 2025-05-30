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

import io
import json
import logging
import multiprocessing
import os
import pickle
import time
import uuid
import zlib
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
from threading import Event, Lock

import msgpack
import msgpack_numpy
import numpy
import numpy.core.multiarray
from fastapi import HTTPException
from fastapi.responses import JSONResponse

import cuopt_server.utils.health_check as health_check
import cuopt_server.utils.request_filter as request_filter
from cuopt_server._version import __version__
from cuopt_server.utils.data_definition import (
    LPData,
    LPTupleData,
    OptimizedRoutingData,
    SolverSettingsConfig,
    WarmStartData,
    cuoptDataInternal,
    get_valid_actions,
)
from cuopt_server.utils.exceptions import (
    exception_handler,
    http_exception_handler,
)
from cuopt_server.utils.logutil import message
from cuopt_server.utils.routing.initial_solution import add_initial_sol


class PickleForbidden(Exception):
    pass


msgpack_numpy.patch()


def lp_datamodel_compat(data):
    """
    Maintain backward compat for some parameters
    that change names in 25.05. Replace the
    old parameters with the new names
    """

    sc = {
        "solver_mode": "pdlp_solver_mode",
        "heuristics_only": "mip_heuristics_only",
    }

    tol = {
        "integrality_tolerance": "mip_integrality_tolerance",
        "absolute_mip_gap": "mip_absolute_gap",
        "relative_mip_gap": "mip_relative_gap",
    }

    replace = []
    if "solver_config" in data:
        s = data["solver_config"]
        for k, v in sc.items():
            if k in s:
                replace.append([k, v, s[k]])

        for r in replace:
            data["solver_config"][r[1]] = r[2]
            del data["solver_config"][r[0]]

        replace = []
        if "tolerances" in s:
            t = s["tolerances"]
            for k, v in tol.items():
                if k in t:
                    replace.append([k, v, t[k]])

            for r in replace:
                data["solver_config"]["tolerances"][r[1]] = r[2]
                del data["solver_config"]["tolerances"][r[0]]


def check_client_version(client_vers):
    logging.debug(f"client_vers is {client_vers} in check")
    if os.environ.get("CUOPT_CHECK_CLIENT", True) in ["True", True]:
        major, minor, *rest = __version__.split(".")
        matches = False
        if client_vers == "custom":
            return []
        cv = client_vers.split(".")
        if len(cv) != 3:
            logging.warn("Client version missing or bad format")
            return [
                f"Client version missing or not the current format. "
                f"Please upgrade your cuOpt client to '{major}.{minor}', "
                "or set the client version to 'custom' "
                "if this is a custom client."
            ]
        else:
            cmajor, cminor, _ = cv
            matches = (cmajor, cminor) == (major, minor)
        if not matches:
            logging.warn(f"Client version {cmajor}.{cminor} does not match")
            return [
                f"Client version is '{cmajor}.{cminor}' but server "
                f"version is '{major}.{minor}'. Please use a matching client."
            ]
    return []


def get_solver_response(response):
    if "solver_response" in response:
        return response["solver_response"]
    return response["solver_infeasible_response"]


class SafeUnpickler(pickle.Unpickler):
    def __init__(self, file, kind, allowed={}):
        self.allowed = allowed
        self.kind = kind
        super().__init__(file)

    def find_class(self, module, name):
        if (
            module not in self.allowed
            or name not in self.allowed[module]["names"]
        ):
            raise PickleForbidden(
                f"{module}.{name} is forbidden "
                f"in a cuopt {self.kind}pickle file"
            )
        else:
            return getattr(self.allowed[module]["mod"], name)


# LP pickle allow is superset of VRP, so allow the kind
# to be set to "" for messaging and this routine to be
# used when we don't pre-know the problem type
def cuopt_pickle_load(s, kind="LP "):

    allowed_LP = {
        "numpy.core.multiarray": {
            "names": ["_reconstruct"],
            "mod": numpy.core.multiarray,
        },
        "numpy": {"names": ["ndarray", "dtype"], "mod": numpy},
    }

    return SafeUnpickler(io.BytesIO(s), kind, allowed_LP).load()


def cuopt_pickle_load_VRP(s):
    return SafeUnpickler(io.BytesIO(s), "VRP ").load()


all_jobs_marked_done = multiprocessing.Event()

# storage for job results keyed by id
saved_results = {}

# records shared memory segments that we have cached
# keyed by id
cache_list = {}


# Should be used only for:
# * insertion/deletion into saved_results
# * iteration over saved_results
# * presence of key in saved_results
# * setting/checking all_jobs_marked_done
# * operations on cache_list
results_lock = Lock()


mime_json = "application/json"
mime_msgpack = "application/vnd.msgpack"
mime_zlib = "application/zlib"
mime_pickle = "application/octet-stream"
mime_wild = ["application/*", "*/*"]


def add_cache_entry(id, content_type):
    if all_jobs_marked_done.is_set():
        return None

    with results_lock:
        cache_list[id] = {"ctype": content_type, "data": None}
    return id


def update_cache_entry(id, data):
    with results_lock:
        cache_list[id]["data"] = data


def get_cache_content_type(id):
    with results_lock:
        c = cache_list.get(id, None)
        if c:
            return c["ctype"], c["data"]
    return None, None


def delete_cache_entry(id):
    global cache_list

    def delete_shm(c, id):
        if c["data"] is None:
            try:
                s = shared_memory.SharedMemory(name=id)
                s.unlink()
            except FileNotFoundError:
                pass

    with results_lock:
        if id == "*":
            sz = len(cache_list)
            for id in cache_list:
                delete_shm(cache_list[id], id)
            cache_list = {}
            return sz

        elif id in cache_list:
            delete_shm(cache_list[id], id)
            del cache_list[id]
            return 1
    return 0


def mark_all_jobs_done(status_code, msg):
    if all_jobs_marked_done.is_set():
        return
    with results_lock:
        for _, value in saved_results.items():
            if not value.is_done():
                value.set_termination(status_code, msg)
    all_jobs_marked_done.set()
    logging.info("All jobs marked complete")


def get_solution_for_id(id, delete=True):
    with results_lock:
        if id in saved_results:
            # If we have a binary result and the caller can't handle
            # binary, leave the result here and return
            mime_type = saved_results[id].get_mime_type()
            if saved_results[id].is_done():
                s = saved_results[id]
                if delete:
                    del saved_results[id]
                return s.wait() + (mime_type,)
            else:
                return None, None, mime_type
        raise HTTPException(status_code=404, detail=f"job {id} does not exist")


def get_warmstart_data_for_id(id):
    with results_lock:
        if id in saved_results:
            # Warmstart data can be huge and is not returned with solution.
            # Always save as msgpack
            mime_type = mime_msgpack
            if saved_results[id].is_done():
                s = saved_results[id]
                return s.warmstart_data, mime_type
            else:
                return None, mime_type
        raise HTTPException(status_code=404, detail=f"job {id} does not exist")


def get_incumbents_for_id(id):
    with results_lock:
        if id in saved_results:
            return saved_results[id].get_current_incumbents()
        raise HTTPException(status_code=404, detail=f"job {id} does not exist")


class AbortList:
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.jobs = self.manager.dict()
        self.lock = multiprocessing.Lock()

    def add_id_or_return(self, id, proc_id=False):
        with self.lock:
            if id not in self.jobs:
                self.jobs[id] = proc_id
            else:
                return self.jobs[id]
        return None

    def get(self, id):
        with self.lock:
            if id in self.jobs:
                return self.jobs[id]
        return None

    def update(self, id):
        with self.lock:
            if id in self.jobs:
                self.jobs[id] = True

    def delete(self, id):
        with self.lock:
            if id in self.jobs:
                del self.jobs[id]

    def find_job_by_pid(self, pid):
        with self.lock:
            for k, v in self.jobs.items():
                if v == pid:
                    return k
        return None


def create_abort_list():
    return AbortList()


def abort_by_pid(pid, abort_list, results_queue):
    id = abort_list.find_job_by_pid(pid)
    abort_list.delete(id)
    results_queue.put(
        SolverBinaryResponse(
            id, JSONResponse({"error": f"{id} was aborted"}, 500)
        )
    )


def status_by_id(id, abort_list):
    with results_lock:
        if id not in saved_results:
            raise HTTPException(
                status_code=404, detail=f"job {id} does not exist"
            )

        if saved_results[id].aborted:
            return "aborted"
        elif saved_results[id].is_done():
            return "completed"

        # Two possibilities left, queued or running
        # If it's got a proc id or True set in the abort
        # list then it is "running" or just completed
        # If not, then it has to be queued
        s = abort_list.get(id)
        if s is True:
            return "completed"
        elif s is None:
            return "queued"

        # this should have already been caught above
        # from saved results, but we'll check anyway
        elif s is False:
            return "aborted"
        else:
            return "running"


def abort_by_id(id, abort_queue, abort_list, running, queued):

    if not running and not queued:
        return 0, 0

    with results_lock:
        if id in saved_results and not saved_results[id].is_done():
            # Check job status
            if queued:
                value = abort_list.add_id_or_return(id)
            else:
                value = abort_list.get(id)

            if value in [True, False]:
                # completed by solver, but result in transit
                # or queued and already aborted
                return 0, 0

            if (queued and value is None) or (running and value is not None):
                # We want a current waiter to be woken up if blocking.
                # We want the first get to receive an aborted result otherwise.
                # In both of these cases the result will be unregistered.
                saved_results[id].set_aborted()

                # If the value is None, it's on the queue
                # but will be skipped
                if value is not None:
                    # It's currently running so we should stop the process
                    abort_queue.put(value)
                    abort_list.delete(id)
                    return 0, 1
                else:
                    return 1, 0
        return 0, 0


def abort_all(abort_queue, abort_list, running, queued):

    if not running and not queued:
        return

    q_cnt = 0
    r_cnt = 0

    # Save the process shutdowns til the end, because we don't want to restart
    # tasks with new jobs that we are just going to mark aborted anyway
    to_stop = []
    with results_lock:
        for id, res in saved_results.items():
            if res.is_done():
                continue

            if queued:
                value = abort_list.add_id_or_return(id)
            else:
                value = abort_list.get(id)

            if value in [True, False]:
                continue

            elif (queued and value is None) or (running and value is not None):
                # We want a current waiter to be woken up if blocking.
                # We want the first get to receive an aborted result otherwise.
                # In both of these cases the result will be unregistered.
                saved_results[id].set_aborted()

                if value is not None:
                    r_cnt += 1
                    to_stop.append((value, id))
                else:
                    q_cnt += 1

    for s in to_stop:
        abort_queue.put(s[0])
        abort_list.delete(s[1])
    return q_cnt, r_cnt


class BaseResult:
    def __init__(self, resultfile="", rtype=mime_json):

        self.id = str(uuid.uuid4())
        self.etl = None
        self.slv = None
        self.sku = None
        self.done = False

        # If result is returned through a file,
        # then the response sent from the server will be modified
        # to file_result
        self.result = None
        self.file_result = None
        self.warmstart_data = None
        self.resultfile = resultfile
        self.status = None
        self.action = ""
        self.validator_enabled = False
        self.aborted = False
        self.rtype = rtype
        self.incumbents = []

    def get_mime_type(self):
        return self.rtype

    def get_id(self):
        return self.id

    def set_done(self):
        self.done = True

    def set_aborted(self):
        self.aborted = True
        self.set_result(JSONResponse({"error": f"{self.id} was aborted"}, 500))

    def set_result(self, result, file_result=None, quiet=False):
        self.result = result
        self.file_result = file_result
        if not quiet:
            logging.info(f"set done for {self.id}")
        # Log cuopt_complete here so that we get it for
        # all jobs including async.
        if not quiet:
            logging.info({"cuopt_complete": self.status})
        self.set_done()

    def set_warmstart_data(self, warmstart_data):
        self.warmstart_data = warmstart_data

    def set_stats(self, etl, slv, sku, status):
        self.etl = etl
        self.slv = slv
        self.sku = sku
        self.status = status

    def get_stats(self):
        return self.etl, self.slv, self.sku, self.status

    def set_action(self, action):
        self.action = action

    def set_validator_enabled(self, ve):
        self.validator_enabled = ve

    def get_action(self):
        return self.action

    def get_validator_enabled(self):
        return self.validator_enabled

    def get_result(self):
        return self.result, self.file_result

    def is_done(self):
        return self.done

    def _wait(self, timeout=None):
        return True

    def wait(self, timeout=None):
        # If we snuck in during a shutdown and missed being marked
        # complete, just finish
        if not self.is_done() and all_jobs_marked_done.is_set():
            # We just want to log the exception and produce the result,
            # we can skip the raise here
            return (
                http_exception_handler(
                    HTTPException(
                        status_code=500, detail="cuOpt is shutting down"
                    )
                ),
                None,
            )
        if self._wait(timeout):
            return self.result, self.file_result
        return None, None

    def register_result(self):
        with results_lock:
            saved_results[self.id] = self
        return self.id

    def unregister_result(self):
        with results_lock:
            if self.id in saved_results:
                del saved_results[self.id]

    def add_incumbent(self, sol):
        # Mark when we have seen an incumbent, so that
        # we know when the list has reached empty again
        # we can send the sentinel value
        if self.is_done():
            logging.warn("Incumbent added after job marked done!")
        sol["solution"] = sol["solution"].tolist()
        self.incumbents.append(sol)

    def get_current_incumbents(self):
        if self.is_done() and self.incumbents == []:
            logging.debug("returning incumbent sentinel")
            return [{"solution": [], "cost": None}]
        res = self.incumbents
        self.incumbents = []
        return res


def del_shm(name):
    if os.environ.get("CUOPT_SERVER_SHM", False) in ["True", "true", True]:
        try:
            s = shared_memory.SharedMemory(name=name)
            s.unlink()
        except Exception:
            pass


class BinaryJobResult(BaseResult):
    def __init__(
        self,
        shm_reference,
        rtype,
        resultfile="",
        resultdir="",
        maxresult=0,
        mode=644,
    ):

        super().__init__(resultfile=resultfile, rtype=rtype)

        # Note that this is a threading.Event, not
        # a multiprocessing.Event. That's because results
        # are handled wholly within the webserver process.
        self.done = Event()

        # We don't need the shm_reference, we just need to know there
        # is one
        self.uses_cached_shm = shm_reference is not None

        self.resultdir = resultdir
        self.maxresult = maxresult
        self.mode = mode
        self.warnings = None
        self.notes = None

    def set_done(self):
        self.done.set()

    def set_notes_and_warnings(self, notes, warnings):
        # Preserve notes and warnings
        # so that they can be returned in the message
        # to the client in the case of a file result.
        # Notes and warnings have already been serialized
        # in the result itself.
        self.notes = notes
        self.warnings = warnings

    def set_data_size_and_type(self, size, rtype):
        self.data_size = size

        # might as well make sure these match
        if rtype != self.rtype:
            logging.warn(
                "in set_data_size_and_type result mime_type "
                f"does not match, updating {rtype} {self.rtype}"
            )
        self.rtype = rtype

    def set_termination(self, status_code, error):
        # The job associated with this id will never run
        # and it has shared memory associated, try to delete it
        if not self.uses_cached_shm:
            del_shm(self.id)
        if self.result and isinstance(self.result, str):
            del_shm(self.result)
        logging.info(f"set done for {self.id} on termination")
        # We want the exception message logged even for jobs
        # which are not being actively waited on, so do it here
        self.result, self.file_result = (
            http_exception_handler(
                HTTPException(status_code=status_code, detail=error)
            ),
            None,
        )
        self.set_done()

    def get_file_result(self, result):
        r = {"result_file": self.resultfile}
        if self.warnings:
            logging.debug("adding warnings to file result")
            r["warnings"] = self.warnings
        if self.notes:
            logging.debug("adding notes to file result")
            r["notes"] = self.notes
        return r

    def set_result(self, result):
        # We expect normal cuOpt responses for a binary job to be
        # the name of a shared memory segment or bytes.
        # Otherwise it should be a JSONResponse holding
        # an exception. Just pass the result as is to super.
        file_result = None
        if isinstance(result, str) or isinstance(result, bytes):
            try:
                if (
                    self.resultfile
                    and self.resultdir
                    and self.data_size >= self.maxresult * 1000
                ):
                    # TODO: set a file extension based on rtype?
                    op = os.path.join(self.resultdir, self.resultfile)
                    logging.debug(f"Writing large result to disk {op}")
                    file_result = self.get_file_result(result)

                    s = None
                    if isinstance(result, str):
                        # Write the contents of shared memory to disk
                        s = shared_memory.SharedMemory(name=result)
                        buf = s.buf[:]
                    else:
                        buf = result

                    # Write the contents of shared memory to disk
                    # Make sure to eliminate shm slice ref in buf
                    # and unlink in all cases
                    try:
                        with open(op, "wb") as out:
                            out.write(buf)
                        if self.mode:
                            os.chmod(op, self.mode)
                    finally:
                        buf = None
                        if s:
                            s.unlink()
                    self.result = None

            # In case something goes wrong in this routine, we must
            # mark the job complete anyway to keep the service functioning
            except Exception as e:
                result = exception_handler(e)
                file_result = None

        super().set_result(result, file_result)

    def set_warmstart_data(self, warmstart_data):
        super().set_warmstart_data(warmstart_data)

    def is_done(self):
        return self.done.is_set()

    def _wait(self, timeout=None):
        return self.done.wait(timeout)


class NVCFJobResult(BinaryJobResult):
    def __init__(self, resultdir, maxresult, accept=mime_json):
        super().__init__(
            None, accept, "large_result", resultdir, maxresult, mode=None
        )

    def get_file_result(self, result):
        # In the NVCF case we don't send anything back because NVCF
        # takes care of handling the file response
        return {}


# Job objects sent to the solver to be executed #
class SolverBaseJob:
    def __init__(
        self,
        id,
        warnings,
        request_filter,
        validator_enabled,
        response_id=True,
        incumbents=False,
        solver_logs=False,
    ):
        self.id = id
        self.warnings = warnings
        self.request_filter = request_filter
        self.validator_enabled = validator_enabled
        self.response_id = response_id
        self.incumbents = incumbents

        self.initial_etl_time = 0
        self.ncaid = ""
        self.reqid = ""
        self.solver_logs = solver_logs

        # This will be updated in derived types
        if validator_enabled:
            self.action = "cuOpt_Validator"
        else:
            self.action = "cuOpt_Solver"

    def get_action(self):
        return self.action

    def set_nvcf_ids(self, ncaid, reqid):
        self.ncaid = ncaid
        self.reqid = reqid

    def get_nvcf_ids(self):
        return self.ncaid, self.reqid

    def get_data(self):
        return None

    def get_sku(self):
        return 0

    def is_validator_enabled(self):
        return self.validator_enabled

    def get_id_for_response(self):
        if self.response_id:
            return self.id
        return ""

    def set_initial_etl_time(self, t):
        self.initial_etl_time = t

    def solve(self, intermediate_sender):
        return None, self.initial_etl_time, 0

    def get_result_mime_type(self):
        return mime_json

    def return_incumbents(self):
        return self.incumbents

    def delete_data(self):
        pass

    def return_solver_logs(self):
        return self.solver_logs


class SolverJob(SolverBaseJob):
    def __init__(
        self,
        id,
        optimization_data,
        warnings,
        request_filter=False,
        validator_enabled=False,
        response_id=True,
    ):
        super().__init__(
            id,
            warnings,
            request_filter,
            validator_enabled,
            response_id,
        )
        self.optimization_data = optimization_data

        if self.validator_enabled:
            self.action = "cuOpt_RoutingValidator"
        else:
            self.action = "cuOpt_OptimizedRouting"

    def get_data(self):
        return self.optimization_data

    def _check_for_warnings(self):
        for key, value in self.optimization_data.items():
            if hasattr(value, "warnings"):
                self.warnings.extend(value.warnings)

    def _load_data(self):
        # Add an explicit empty solver config here so the
        # calculated time limit will be filled in
        if self.optimization_data["solver_config"] is None:
            logging.debug("Creating default solver config")
            self.optimization_data["solver_config"] = SolverSettingsConfig()
        if self.request_filter:
            status, msg = request_filter.check_restrictions(
                request_filter.get_features(),
                self.optimization_data,
                self.optimization_data,
            )
            if not status:
                logging.error(
                    message(
                        "feature check failed for tier '%s'"
                        % request_filter.get_tier()
                    )
                )
                raise HTTPException(
                    status_code=400,
                    detail="feature check failed for tier '%s', "
                    % request_filter.get_tier()
                    + msg,
                )
            logging.debug(
                message(
                    "feature check succeeeded for tier '%s', "
                    % request_filter.get_tier()
                )
            )
        self._check_for_warnings()

    def get_sku(self):
        if self.optimization_data:
            return len(self.optimization_data["task_data"].task_locations)
        return 0

    def solve(self, intermediate_sender):
        from cuopt_server.utils.solver import solve_optimized_routes_sync

        self._load_data()
        ans, etl, slv = solve_optimized_routes_sync(
            **self.get_data(),
            warnings=self.warnings,
            validation_only=self.is_validator_enabled(),
            reqId=self.get_id_for_response(),
        )

        logging.debug(f"etl_time {etl}, solve_time {slv}")
        return ans, self.initial_etl_time + etl, slv


class SolverLPJob(SolverBaseJob):
    def __init__(
        self,
        id,
        LP_data,
        warmstart_data,
        warnings,
        request_filter=False,
        validator_enabled=False,
        response_id=True,
        transformed=False,
        incumbents=False,
        solver_logs=False,
    ):
        super().__init__(
            id,
            warnings,
            request_filter,
            validator_enabled,
            response_id,
            incumbents=incumbents,
            solver_logs=solver_logs,
        )
        self.LP_data = LP_data
        self.warmstart_data = warmstart_data
        self.transformed = transformed
        if self.validator_enabled:
            self.action = "cuOpt_LPValidator"
        else:
            self.action = "cuOpt_LP"

    def get_data(self):
        return self.LP_data

    def _transform(self, data):
        np = numpy
        tmap = {
            "csr_constraint_matrix": {
                "offsets": (True, np.int32),
                "indices": (True, np.int32),
                "values": (True, np.float64),
            },
            "constraint_bounds": {
                "bounds": (True, np.float64),
                "upper_bounds": (True, np.float64),
                "lower_bounds": (True, np.float64),
                "types": (True, "U1"),
            },
            "initial_solution": {
                "primal": (True, np.float64),
                "dual": (True, np.float64),
            },
            "objective_data": {
                "coefficients": (True, np.float64),
            },
            "variable_bounds": {
                "upper_bounds": (True, np.float64),
                "lower_bounds": (True, np.float64),
            },
            "variable_types": (True, "U1"),
        }

        def modify(value, key, dtype=None, indent=""):
            if isinstance(value, list):
                if "inf" in value or "ninf" in value:
                    value = [
                        np.inf if x == "inf" else -np.inf if x == "ninf" else x
                        for x in value
                    ]
                if dtype is None:
                    return np.array(value)
                return np.array(value, dtype)
            return value

        def apply(data, tmap, indent=""):
            for key, value in data.items():
                try:
                    if isinstance(value, dict) and key in tmap:
                        apply(value, tmap[key], indent + " ")
                    elif key in tmap and tmap[key][0]:
                        data[key] = modify(value, key, tmap[key][1], indent)
                except Exception as e:
                    logging.debug(e)
                    logging.debug(
                        f"{indent}exception key is {key} value is {value}"
                    )
                    raise

        def apply_LPData(data):
            data.csr_constraint_matrix.indices = modify(
                data.csr_constraint_matrix.indices,
                "csr_constraint_matrix.indices",
                np.int32,
            )
            data.csr_constraint_matrix.offsets = modify(
                data.csr_constraint_matrix.offsets,
                "csr_constraint_matrix.offsets",
                np.int32,
            )
            data.csr_constraint_matrix.values = modify(
                data.csr_constraint_matrix.values,
                "csr_constraint_matrix.values",
                np.float64,
            )

            data.constraint_bounds.bounds = modify(
                data.constraint_bounds.bounds,
                "constraint_bounds.bounds",
                np.float64,
            )
            data.constraint_bounds.upper_bounds = modify(
                data.constraint_bounds.upper_bounds,
                "constraint_bounds.upper_bounds",
                np.float64,
            )
            data.constraint_bounds.lower_bounds = modify(
                data.constraint_bounds.lower_bounds,
                "constraint_bounds.lower_bounds",
                np.float64,
            )
            data.constraint_bounds.types = modify(
                data.constraint_bounds.types,
                "constraint_bounds.types",
            )

            data.initial_solution.primal = modify(
                data.initial_solution.primal,
                "initial_solution.primal",
                np.float64,
            )

            data.initial_solution.dual = modify(
                data.initial_solution.dual, "initial_solution.dual", np.float64
            )

            data.objective_data.coefficients = modify(
                data.objective_data.coefficients,
                "objective_data.coefficients",
                np.float64,
            )

            data.variable_bounds.upper_bounds = modify(
                data.variable_bounds.upper_bounds,
                "variable_bounds.upper_bounds",
                np.float64,
            )
            data.variable_bounds.lower_bounds = modify(
                data.variable_bounds.lower_bounds,
                "variable_bounds.lower_bounds",
                np.float64,
            )
            data.variable_types = modify(
                data.variable_types,
                "variable_types",
            )

        then = time.time()
        if isinstance(data, LPData):
            apply_LPData(data)
        else:
            apply(data, tmap)
        logging.info(f"transform time {time.time() - then}")

    def _load_data(self):
        if not self.transformed:
            self._transform(self.LP_data)
            self.transformed = True

        # Add request filtering

    def get_sku(self):
        return 0  # len(self.LP_data["csr_constraint_matrix"].offsets)-1

    def solve(self, intermediate_sender):

        from cuopt_server.utils.solver import solve_LP_sync

        self._load_data()
        ans, etl, slv = solve_LP_sync(
            self.get_data(),
            self.warmstart_data,
            warnings=self.warnings,
            validation_only=self.is_validator_enabled(),
            reqId=self.get_id_for_response(),
            intermediate_sender=intermediate_sender
            if self.return_incumbents()
            else None,
            solver_logging=self.return_solver_logs(),
        )
        logging.debug(f"etl_time {etl}, solve_time {slv}")
        return ans, self.initial_etl_time + etl, slv


def deserialize(ctype, buf):
    try:
        if ctype == mime_json:
            logging.debug("decode as json")
            data = json.loads(buf)
        elif ctype == mime_zlib:
            logging.debug("decode as zlib compressed json")
            data = json.loads(zlib.decompress(buf))
        elif ctype == mime_pickle:
            logging.debug("decode as pickle")
            data = cuopt_pickle_load(buf, kind="")
        else:
            logging.debug("decode as msgpack")
            data = msgpack.loads(buf, strict_map_key=False)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="unable to load " "optimization data stream, %s" % (str(e)),
        )
    return data


def wrapper_fields(data, do_raise=True):
    action = data.get("action", "cuOpt_Solver")
    if action not in get_valid_actions():
        if do_raise:
            raise HTTPException(
                status_code=400,
                detail="Unrecognized action value {action}",  # noqa
            )
        action = "Invalid event/action"
    validator = action in [
        "cuOpt_RoutingValidator",
        "cuOpt_LPValidator",
        "cuOpt_Validator",
    ]
    client_version = data.get("client_version", "")
    return action, validator, client_version


def read_wrapper_data(ctype, buf):
    try:
        d = deserialize(ctype, buf)
        return wrapper_fields(d, do_raise=False)
    except Exception:
        return "unknown", False, ""


class SolverBinaryJob:
    def __init__(
        self,
        id,
        warnings,
        rtype,
        ctype,
        shm_reference=None,
        validator_enabled=False,
        request_filter=False,
        response_id=True,
        data_bytes=None,
        init_sols=[],
        warmstart_data=None,
        incumbents=False,
        solver_logs=False,
    ):

        # This class is a wrapper object around a real job. The actual
        # job type and contents are not determined until the job reaches
        # the solver and the data is loaded.  Here, we just have an unread
        # collection of bytes that will become the job data.
        # This class maintains an internal self.resolved_job object and
        # defers most calls to self.resolved_job.  It implements the
        # interface from SolverBaseJob but does not inherit from it.
        # (TODO: fix this up to use real interface classes, etc)

        # ID is used as to reference shared memory or job results,
        # so we store a copy of the id in this wrapper object for
        # use in data handling.
        # The same id is set in self.resolved_job, which
        # becomes the "real" job
        self.id = id

        self.ctype = ctype  # content mime_type
        self.rtype = rtype  # result mime_type

        # Start life off as a SolverBaseJob so calls on this object work
        # Make routines in the interface forward to the resolved_job
        # Later, change the type of resolved_job once we know the data type
        self.resolved_job = SolverBaseJob(
            id,
            warnings,
            request_filter,
            validator_enabled,
            response_id,
            incumbents=incumbents,
            solver_logs=solver_logs,
        )

        # This will be a reference to self.resolved_job.warnings as well
        self.warnings = warnings

        self.data_bytes = data_bytes
        if self.data_bytes:
            self.shm_reference = None
        else:
            self.shm_reference = shm_reference

        self.init_sols = init_sols
        self.warmstart_data = warmstart_data

    def delete_data(self):
        # This is for cases where we skip a job on cancelation
        # In this case for shared memory use, we need
        # to unlink the shared memory if it is not a cache
        # reference because we will never call _resolve_job
        if not self.data_bytes and not self.shm_reference:
            try:
                s = shared_memory.SharedMemory(name=self.id)
                s.unlink()
            except Exception:
                pass

    def _resolve_job(self):
        buf = None
        s = None
        if self.data_bytes:
            # We have passed a byte array through the queue
            buf = self.data_bytes
        else:
            # We have used shared memory
            if self.shm_reference:
                shmid = self.shm_reference
            else:
                shmid = self.id
            try:
                s = shared_memory.SharedMemory(name=shmid)
                buf = bytes(s.buf)
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not load job {shmid} from shared memory",
                )

        data = deserialize(self.ctype, buf)
        initial_solutions = []
        for init_sol in self.init_sols:
            # tuples of (mime_type, data_bytes) since initial solutions
            # can come from different sources and therefore the
            # mime_type can be different
            initial_solutions.append(deserialize(init_sol[0], init_sol[1]))

        # If we have a shared memory cache reference,
        # we want to leave it in place instead of destroying it.
        if s:
            if self.shm_reference:
                unregister(s._name, "shared_memory")
                s.close()
            else:
                s.unlink()

        # In this case the job came from the /cuopt/cuopt endpoint
        # which has certain things encoded in the job rather than in
        # header values
        if isinstance(data, dict) and "data" in data:
            # do this just for simple validation
            _ = cuoptDataInternal.parse_obj(data)
            self._read_wrapper_data(data)
            data = data["data"]

        warmstart_data = None
        if self.warmstart_data:
            warmstart_data = deserialize(
                self.warmstart_data[1], self.warmstart_data[0]
            )
            warmstart_data = WarmStartData.parse_obj(warmstart_data)

        if isinstance(data, list) or "csr_constraint_matrix" in data:
            try:
                if isinstance(data, list):
                    lpdata = []
                    # See if this is a list of mine_type/bytes tuples
                    try:
                        d = LPTupleData(data_list=data)
                        tuples = True
                        data = d.data_list
                    except Exception:
                        tuples = False

                    for i_data in data:
                        if tuples:
                            i_data = deserialize(i_data[0], i_data[1])

                        # Call transform to change lists to numpy arrays
                        # before validation. Needs a better interface but
                        # just use a throwaway job object for now
                        t = SolverLPJob(0, i_data, None, None)
                        t._transform(t.LP_data)
                        i_data = t.get_data()
                        lp_datamodel_compat(i_data)
                        lpdata.append(LPData.parse_obj(i_data))
                    data = lpdata
                else:
                    # Call transform to change lists to numpy arrays
                    # before validation. Needs a better interface but
                    # just use a throwaway job object for now
                    t = SolverLPJob(0, data, None, None)
                    t._transform(t.LP_data)
                    data = t.get_data()
                    lp_datamodel_compat(data)
                    data = LPData.parse_obj(data)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail="unable to validate "
                    "optimization data stream, %s" % (str(e)),
                )

            # TODO add a getter for self.id and make webserver use it
            self.resolved_job = SolverLPJob(
                self.resolved_job.id,
                data,
                warmstart_data,
                self.resolved_job.warnings,
                self.resolved_job.request_filter,
                self.resolved_job.validator_enabled,
                self.resolved_job.response_id,
                transformed=True,
                incumbents=self.resolved_job.incumbents,
                solver_logs=self.resolved_job.return_solver_logs(),
            )

        else:
            # TODO track down why this must be a dict here versus LP data
            # which is just the result of parse_obj. Historically we added
            # Pydantic later, and so the use of obj vs dict is muddled.
            # We should clean it up and make it work like LP
            try:
                data = dict(OptimizedRoutingData.parse_obj(data))
                if initial_solutions:
                    add_initial_sol(data, initial_solutions)

            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail="unable to validate "
                    "optimization data stream, %s" % (str(e)),
                )

            self.resolved_job = SolverJob(
                self.resolved_job.id,
                data,
                self.resolved_job.warnings,
                self.resolved_job.request_filter,
                self.resolved_job.validator_enabled,
                self.resolved_job.response_id,
            )

    def _read_wrapper_data(self, data):
        action, validator, client_version = wrapper_fields(data)
        self.resolved_job.action = action
        self.resolved_job.validator_enabled = validator
        self.resolved_job.warnings.extend(check_client_version(client_version))

    def get_action(self):
        return self.resolved_job.get_action()

    def set_nvcf_ids(self, ncaid, reqid):
        self.resolved_job.set_nvcf_ids(ncaid, reqid)

    def get_nvcf_ids(self):
        return self.resolved_job.get_nvcf_ids()

    def is_validator_enabled(self):
        return self.resolved_job.is_validator_enabled()

    def get_id_for_response(self):
        return self.resolved_job.get_id_for_response()

    def set_initial_etl_time(self, t):
        self.resolved_job.set_initial_etl_time(t)

    def get_data(self):
        return self.resolved_job.get_data()

    def get_sku(self):
        return self.resolved_job.get_sku()

    def solve(self, intermediate_sender):
        self._resolve_job()
        return self.resolved_job.solve(intermediate_sender)

    def get_result_mime_type(self):
        return self.rtype

    def return_incumbents(self):
        return self.resolved_job.return_incumbents()


class SolverBinaryJobPath(SolverBinaryJob):
    def __init__(
        self,
        id,
        warnings,
        rtype,
        file_path,
        validator_enabled=False,
        request_filter=False,
        response_id=True,
        wrapper_data=None,
        init_sols=[],
        warmstart_data=None,
        incumbents=False,
        solver_logs=False,
    ):

        self.file_path = file_path
        super().__init__(
            id,
            warnings,
            rtype,
            ctype=None,
            validator_enabled=validator_enabled,
            request_filter=request_filter,
            response_id=response_id,
            init_sols=init_sols,
            warmstart_data=warmstart_data,
            incumbents=incumbents,
            solver_logs=solver_logs,
        )

        if wrapper_data:
            data = deserialize(
                wrapper_data["content_type"], wrapper_data["data"]
            )
            # do this just for simple validation
            _ = cuoptDataInternal.parse_obj(data)
            self._read_wrapper_data(data)

    def _try_extension(self, ext, raw_data):
        if ext == "zlib":
            data = json.loads(zlib.decompress(raw_data))
            logging.debug("zlib data")
        elif ext == "msgpack":
            data = msgpack.loads(raw_data, strict_map_key=False)
            logging.debug("msgpack serialized data")
        elif ext == "json":
            data = json.loads(raw_data)
            logging.debug("uncompressed data")
        elif ext == "pickle":
            data = cuopt_pickle_load(raw_data, kind="")
            self.warnings.append(
                "Pickle data format is deprecated. "
                "Use zlib, msgpack, or plain JSON"
            )
            logging.warn("pickle data is deprecated")
            logging.debug("pickle data")
        else:
            raise ValueError(
                f"File extension {ext} is unsupported. "
                "Supported file extensions are "
                ".json, .zlib, .msgpack, or .pickle"
            )
        return data

    def _resolve_job(self):

        # read the data from the file
        # if we have an extension, use it otherwise try everything
        try:
            ext = (
                self.file_path.split(".")[-1] if "." in self.file_path else ""
            )
            read_begin = time.time()
            with open(self.file_path, "rb") as f:
                raw_data = f.read()
                if ext:
                    data = self._try_extension(ext, raw_data)
                else:
                    for e in ["msgpack", "json", "zlib", "pickle"]:
                        try:
                            data = self._try_extension(e, raw_data)
                            break
                        except PickleForbidden:
                            # In this case we know it loaded as pickle but
                            # it failed the class restrictions, no reason
                            # to try anything else
                            raise

                        except Exception:
                            pass
                    else:
                        raise HTTPException(
                            status_code=422,
                            detail="unable to read "
                            "optimization data file, "
                            "no file extension present and failed to load "
                            "as any supported format",
                        )
                logging.debug(
                    f"Total file load time {time.time() - read_begin}"
                )

        except HTTPException:
            raise

        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail="unable to read "
                "optimization data file, %s" % (str(e)),
            )

        initial_solutions = []
        for init_sol in self.init_sols:
            # tuples of (mime_type, data_bytes) since initial solutions
            # can come from different sources and therefore the
            # mime_type can be different
            initial_solutions.append(deserialize(init_sol[0], init_sol[1]))

        warmstart_data = None
        if self.warmstart_data:
            warmstart_data = deserialize(
                self.warmstart_data[1], self.warmstart_data[0]
            )
            warmstart_data = WarmStartData.parse_obj(warmstart_data)

        if isinstance(data, list) or "csr_constraint_matrix" in data:
            try:
                if isinstance(data, list):
                    lpdata = []
                    # See if this is a list of mine_type/bytes tuples
                    try:
                        d = LPTupleData(data_list=data)
                        tuples = True
                        data = d.data_list
                    except Exception:
                        tuples = False

                    for i_data in data:
                        if tuples:
                            i_data = deserialize(i_data[0], i_data[1])

                        # Call transform to change lists to numpy arrays
                        # before validation. Needs a better interface but
                        # just use a throwaway job object for now
                        t = SolverLPJob(0, i_data, None, None)
                        t._transform(t.LP_data)
                        i_data = t.get_data()
                        lp_datamodel_compat(i_data)
                        lpdata.append(LPData.parse_obj(i_data))
                    data = lpdata
                else:
                    # Call transform to change lists to numpy arrays
                    # before validation. Needs a better interface but
                    # just use a throwaway job object for now
                    t = SolverLPJob(0, data, None, None)
                    t._transform(t.LP_data)
                    data = t.get_data()
                    lp_datamodel_compat(data)
                    data = LPData.parse_obj(data)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail="unable to validate "
                    "optimization data file %s, %s" % (self.file_path, str(e)),
                )

            # TODO add a getter for self.id and make webserver use it
            self.resolved_job = SolverLPJob(
                self.resolved_job.id,
                data,
                warmstart_data,
                self.resolved_job.warnings,
                self.resolved_job.request_filter,
                self.resolved_job.validator_enabled,
                self.resolved_job.response_id,
                transformed=True,
                incumbents=self.resolved_job.incumbents,
                solver_logs=self.resolved_job.return_solver_logs(),
            )

        else:
            # TODO track down why this must be a dict here versus LP data
            # which is just the result of parse_obj. Historically we added
            # Pydantic later, and so the use of obj vs dict is muddled.
            # We should clean it up and make it work like LP
            try:
                data = dict(OptimizedRoutingData.parse_obj(data))
                if initial_solutions:
                    add_initial_sol(data, initial_solutions)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail="unable to validate "
                    "optimization data file %s, %s" % (self.file_path, str(e)),
                )

            self.resolved_job = SolverJob(
                self.resolved_job.id,
                data,
                self.resolved_job.warnings,
                self.resolved_job.request_filter,
                self.resolved_job.validator_enabled,
                self.resolved_job.response_id,
            )


# Responses from the solver #
class SolverIntermediateResponse:
    def __init__(self, id, ans):
        self.id = id
        self.ans = ans

    def get_nvcf_ids(self):
        return "", ""

    def process(self, _):
        with results_lock:
            if self.id in saved_results:
                saved_results[self.id].add_incumbent(self.ans)


class SolverBinaryResponse:
    def __init__(
        self,
        id,
        ans=None,
        result_mime_type=mime_json,
        etl=0,
        slv=0,
        sku=0,
        ncaid="",
        reqid="",
        action="",
        validator_enabled=False,
    ):
        self.id = id
        self.ans = ans
        self.pdlpwarmstart_data = None
        self.etl = etl
        self.slv = slv
        self.sku = sku
        self.ncaid = ncaid
        self.reqid = reqid
        self.notes = None
        self.warnings = None
        self.rtype = result_mime_type
        self.size = 0
        self.status = -1
        self.action = action
        self.validator_enabled = validator_enabled

        def data_to_byte(data, result_mime_type):
            # Write data to a byte array based on result mime type
            # Note that notes and warnings are serialized here before
            # they are popped, so the answer still has them
            now = time.time()
            if result_mime_type in [mime_json, mime_zlib]:
                d = bytes(json.dumps(data), encoding="utf-8")
                if result_mime_type == mime_zlib:
                    now = time.time()
                    d = zlib.compress(d, zlib.Z_BEST_SPEED)
                    logging.debug(
                        "Time for zlib compression of "
                        f"result {time.time() - now}"
                    )
            else:
                d = msgpack.dumps(data)
            self.size = len(d)
            return d

        pdlpwarmstart_data = None
        if isinstance(self.ans, dict):
            # Handle additional items we need for stat etc
            if "response" in self.ans:
                if os.environ.get("CUOPT_RETURN_PERF_TIMES", False):
                    self.ans["response"]["perf_times"] = {
                        "etl_time": self.etl,
                        "solver_run_time": self.slv,
                    }
                res = get_solver_response(self.ans["response"])
                if isinstance(res, list):
                    s = []
                    for r in res:
                        s.append(r["status"])
                        if (
                            isinstance(r["solution"], dict)
                            and "pdlpwarmstart_data" in r["solution"]
                        ):
                            del r["solution"]["pdlpwarmstart_data"]
                    self.status = s
                else:
                    self.status = res["status"]
                if self.action == "cuOpt_LP" and not isinstance(res, list):
                    if "pdlpwarmstart_data" in res["solution"]:
                        pdlpwarmstart_data = res["solution"].pop(
                            "pdlpwarmstart_data"
                        )
            else:
                res = {}

            d = data_to_byte(self.ans, result_mime_type)
            if pdlpwarmstart_data:
                self.pdlpwarmstart_data = data_to_byte(
                    pdlpwarmstart_data, mime_msgpack
                )

            # Save the warnings and notes so we can return them
            # in the case of a file result because we do not
            # send the full response directly and we want the
            # client to list the warnings.
            if "warnings" in self.ans:
                logging.debug("popping warnings")
                self.warnings = self.ans.pop("warnings")
            if "notes" in self.ans:
                logging.debug("popping notes")
                self.notes = self.ans.pop("notes")

            if os.environ.get("CUOPT_SERVER_SHM", False) in [
                "True",
                "true",
                True,
            ]:
                now = time.time()
                shm_name = id + "result"
                s = shared_memory.SharedMemory(
                    create=True, size=len(d), name=shm_name
                )
                unregister(s._name, "shared_memory")
                s.buf[:] = d
                s.close()
                self.ans = shm_name
                logging.debug(
                    "Time for shared memory write of "
                    f"result {time.time() - now}"
                )
            else:
                self.ans = d

    def get_nvcf_ids(self):
        return self.ncaid, self.reqid

    def process(self, abort_list):
        try:
            with results_lock:
                # If we have a result for something that is no longer in
                # saved results, it's likely that it was aborted.
                if self.id not in saved_results and isinstance(self.ans, str):
                    del_shm(self.ans)
                res = (
                    saved_results[self.id]
                    if (
                        self.id in saved_results
                        and not saved_results[self.id].is_done()
                    )
                    else None
                )
            if res:
                res.set_stats(self.etl, self.slv, self.sku, self.status)
                res.set_notes_and_warnings(self.notes, self.warnings)
                res.set_data_size_and_type(self.size, self.rtype)
                res.set_action(self.action)
                res.set_validator_enabled(self.validator_enabled)
                # This happens last because it triggers
                # anyone waiting on the result
                res.set_result(self.ans)
                res.set_warmstart_data(self.pdlpwarmstart_data)
            abort_list.delete(self.id)
        except Exception as e:
            res.set_result(exception_handler(e))


# TOOD: ExitJob is meant for the solver, Shutdown and
# CudaUnhealty are meant for the result thread
# Probably should be in different class hierarchies.
# The latter two probably ought to be SolveResponses
class ExitJob:
    def get_nvcf_ids(self):
        return "", ""


class Shutdown(ExitJob):
    def __init__(self, signame=None):
        self.status_code = 500
        self.msg = "cuOpt is shutting down"
        if signame is not None:
            self.msg += f" on signal {signame}"

    def process(self, *args):
        values = []
        with results_lock:
            if all_jobs_marked_done.is_set():
                return
            for _, value in saved_results.items():
                if not value.is_done():
                    values.append(value)
            all_jobs_marked_done.set()
        for v in values:
            logging.info(f"set done for {v.id} on termination")
            v.set_result(
                http_exception_handler(
                    HTTPException(
                        status_code=self.status_code, detail=self.msg
                    )
                )
            )
        delete_cache_entry("*")
        logging.info("All jobs marked complete")


class CudaUnhealthy(Shutdown):
    def __init__(self):
        self.status_code = 500
        self.msg = "CUDA errors detected, system will be rebooted"

    def process(self, *args):
        super().process()
        health_check.set_unhealthy(self.msg)
