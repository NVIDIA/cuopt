# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuopt.grpc.linear_programming.grpc_client cimport (
    grpc_incumbents_result_t,
    grpc_job_status_t,
    grpc_logs_result_t,
    grpc_log_line_callback_t,
    grpc_python_client_t,
    grpc_result_outcome_t,
    grpc_status_result_t,
    grpc_submit_result_t,
)
from cuopt.linear_programming.data_model.data_model_wrapper cimport DataModel
from cuopt.linear_programming.solver.solver cimport solver_ret_t
from cuopt.linear_programming.solver.solver_wrapper cimport (
    build_solution_from_unique_ptr,
)
from cuopt.linear_programming.solver.solver_wrapper import (
    prepare_solver_settings,
    type_cast,
)
from cuopt.linear_programming.solver_settings.solver_settings cimport (
    SolverSettings,
)

from enum import IntEnum
import math
from libc.stdint cimport int64_t
from libc.stddef cimport size_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
import threading
import time

import numpy as np


class JobStatus(IntEnum):
    QUEUED = <int>grpc_job_status_t.QUEUED
    PROCESSING = <int>grpc_job_status_t.PROCESSING
    COMPLETED = <int>grpc_job_status_t.COMPLETED
    FAILED = <int>grpc_job_status_t.FAILED
    CANCELLED = <int>grpc_job_status_t.CANCELLED
    NOT_FOUND = <int>grpc_job_status_t.NOT_FOUND


class GrpcError(RuntimeError):
    pass


class JobNotReadyError(GrpcError):
    pass


cdef int _invoke_log_callback(
    const char* line,
    size_t line_len,
    int job_complete,
    void* userdata,
) noexcept nogil:
    with gil:
        try:
            callback = <object>userdata
            text = line[:line_len].decode("utf-8") if line_len > 0 else ""
            # Only an explicit False stops the stream. print()/append() return
            # None and must not be treated as a stop signal.
            if _call_log_callback(callback, text, bool(job_complete)) is False:
                return 0
            return 1
        except Exception as exc:
            cb = <object>userdata
            state = getattr(cb, "state", None)
            if state is not None:
                state["error"] = exc
            return 0


def _call_log_callback(callback, line, job_complete):
    """Invoke a log callback; accept both ``(line,)`` and ``(line, done)`` forms."""
    try:
        return callback(line, job_complete)
    except TypeError:
        return callback(line)


class _LogStreamHandler:
    """Bridge user callback with stream state for C log streaming."""

    __slots__ = ("state", "callback")

    def __init__(self, state, callback):
        self.state = state
        self.callback = callback

    def __call__(self, line, job_complete):
        self.state["lines"].append(line)
        self.state["live_lines"] += 1
        try:
            return _call_log_callback(self.callback, line, job_complete)
        except Exception as exc:
            self.state["error"] = exc
            raise


def _call_incumbent_callback(callback, index, objective, assignment, job_complete):
    try:
        return callback(index, objective, assignment, job_complete)
    except TypeError:
        return callback(index, objective, assignment)


def _forward_incumbent_to_settings(settings, index, objective, assignment, job_complete):
    from cuopt.linear_programming.internals import GetSolutionCallback

    if job_complete:
        return True
    for mip_callback in settings.get_mip_callbacks():
        if mip_callback is None:
            continue
        if isinstance(mip_callback, GetSolutionCallback):
            solution = np.asarray(assignment, dtype=np.float64)
            cost = np.array([objective], dtype=np.float64)
            bound = np.array([math.nan], dtype=np.float64)
            mip_callback.get_solution(
                solution, cost, bound, mip_callback.user_data
            )
    return True


cdef class Client:
    cdef unique_ptr[grpc_python_client_t] _client
    cdef dict _job_is_mip
    cdef dict _log_threads
    cdef dict _log_thread_errors
    cdef dict _log_stream_state
    cdef dict _incumbent_threads
    cdef dict _incumbent_thread_errors
    cdef str _host
    cdef int _port

    def __init__(self, str host, int port):
        cdef string host_cpp = host.encode("utf-8")
        self._client.reset(new grpc_python_client_t(host_cpp, port))
        self._job_is_mip = {}
        self._log_threads = {}
        self._log_thread_errors = {}
        self._log_stream_state = {}
        self._incumbent_threads = {}
        self._incumbent_thread_errors = {}
        self._host = host
        self._port = port
        cdef string error_out
        if not self._client.get().connect(error_out):
            raise GrpcError(error_out.decode("utf-8"))

    def submit(self, problem, SolverSettings settings not None):
        cdef DataModel data_model
        cdef grpc_submit_result_t submit_result
        cdef string job_id
        cdef bint mip

        data_model = self._as_data_model(problem)
        data_model.variable_types = type_cast(
            data_model.variable_types, "S1", "variable_types"
        )
        mip = _is_mip(data_model.get_variable_types())
        prepare_solver_settings(settings, data_model, mip)
        data_model.set_data_model_view()
        cdef bint enable_incumbents = False
        if mip and settings.get_mip_callbacks():
            enable_incumbents = True
        submit_result = self._client.get().submit(
            data_model.c_data_model_view.get(),
            settings.c_solver_settings.get(),
            enable_incumbents,
        )
        if not submit_result.success:
            raise GrpcError(submit_result.error_message.decode("utf-8"))
        job_id = submit_result.job_id
        self._job_is_mip[job_id.decode("utf-8")] = bool(submit_result.is_mip)
        return job_id.decode("utf-8")

    def status(self, str job_id):
        cdef grpc_status_result_t status_result = self._client.get().status(
            job_id.encode("utf-8")
        )
        if not status_result.success:
            raise GrpcError(status_result.error_message.decode("utf-8"))
        return JobStatus(<int>status_result.status)

    def wait(self, str job_id, timeout=None):
        cdef int timeout_seconds = 0 if timeout is None else int(timeout)
        cdef grpc_status_result_t wait_result = self._client.get().wait(
            job_id.encode("utf-8"), timeout_seconds
        )
        if not wait_result.success:
            raise GrpcError(wait_result.error_message.decode("utf-8"))
        return JobStatus(<int>wait_result.status)

    def cancel(self, str job_id):
        cdef string error_out
        if not self._client.get().cancel(job_id.encode("utf-8"), error_out):
            raise GrpcError(error_out.decode("utf-8"))

    def delete(self, str job_id):
        if job_id in self._incumbent_threads:
            self.join_incumbent_stream(job_id)
        cdef string error_out
        if not self._client.get().delete_job(job_id.encode("utf-8"), error_out):
            raise GrpcError(error_out.decode("utf-8"))
        self._job_is_mip.pop(job_id, None)

    def result(self, str job_id, variable_names=None, is_mip=None):
        cdef grpc_result_outcome_t outcome
        cdef bint fetch_mip
        cdef unique_ptr[solver_ret_t] sol_ret

        if is_mip is not None:
            fetch_mip = bool(is_mip)
        else:
            fetch_mip = self._job_is_mip.get(job_id, False)

        outcome = self._client.get().result(job_id.encode("utf-8"), fetch_mip)
        if outcome.not_ready:
            return None
        if not outcome.success:
            raise GrpcError(outcome.error_message.decode("utf-8"))
        sol_ret = move(outcome.solution)
        return build_solution_from_unique_ptr(move(sol_ret), variable_names)

    def logs(self, str job_id, from_byte=0):
        """
        Return all solver log lines for a job that has finished.

        Raises :class:`JobNotReadyError` if the job is still queued or
        running. For live output during the solve, use
        :meth:`start_log_stream`.
        """
        status = self.status(job_id)
        if status in (JobStatus.QUEUED, JobStatus.PROCESSING):
            raise JobNotReadyError(
                f"job {job_id} is not complete ({status.name})"
            )

        cdef grpc_logs_result_t outcome = self._client.get().fetch_logs(
            job_id.encode("utf-8"), from_byte
        )
        if not outcome.success:
            msg = outcome.error_message.decode("utf-8")
            if not msg:
                msg = self._client.get().last_error().decode("utf-8")
            raise GrpcError(msg or "failed to fetch logs")
        return [line.decode("utf-8") for line in outcome.lines]

    def start_log_stream(self, str job_id, callback=print, from_byte=0):
        """
        Stream solver logs on a background thread until the job completes.

        ``callback`` is invoked as ``callback(line, job_complete)`` for each
        line. Return ``False`` explicitly to stop early; other return values
        (including ``None`` from ``print``) keep the stream open.

        Call :meth:`join_log_stream` before :meth:`delete` to ensure all log
        lines were received. To collect lines in memory::

            lines = []
            client.start_log_stream(job_id, lines.append)
        """
        if job_id in self._log_threads:
            raise GrpcError(f"log stream already running for job {job_id}")

        state = {
            "lines": [],
            "callback": callback,
            "from_byte": from_byte,
            "live_lines": 0,
            "backfilled": False,
            "error": None,
        }
        self._log_stream_state[job_id] = state

        handler = _LogStreamHandler(state, callback)

        # Use a dedicated connection so StreamLogs can run concurrently with
        # status/result polling on this client.
        log_client = Client(self._host, self._port)
        thread = threading.Thread(
            target=self._run_log_stream,
            args=(log_client, job_id, handler, from_byte),
            daemon=True,
        )
        self._log_threads[job_id] = thread
        thread.start()
        return thread

    def join_log_stream(self, str job_id, timeout=None):
        """Wait for a background log stream started by :meth:`start_log_stream`.

        Returns a dict with stream stats (``live_lines``, ``lines``, ``backfilled``)
        when a stream was started for ``job_id``, else ``None``.
        """
        thread = self._log_threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                exc = self._log_thread_errors.get(job_id)
                if exc is not None:
                    raise exc
                return self._log_stream_state.get(job_id)
            self._log_threads.pop(job_id, None)

        exc = self._log_thread_errors.pop(job_id, None)
        if exc is not None:
            raise exc

        state = self._log_stream_state.pop(job_id, None)
        if state is None:
            return None

        if state.get("error") is not None:
            raise state["error"]

        if state["live_lines"] == 0:
            self._backfill_log_stream(job_id, state)
        return state

    def _backfill_log_stream(self, str job_id, state):
        """Fetch logs after live streaming missed output (status/file races)."""
        cdef int attempt
        for attempt in range(6):
            try:
                bulk = self.logs(job_id, state["from_byte"])
            except JobNotReadyError:
                if attempt == 0:
                    self.wait(job_id, timeout=120)
                    continue
                time.sleep(0.2)
                continue
            if bulk:
                for line in bulk:
                    state["lines"].append(line)
                    _call_log_callback(state["callback"], line, True)
                state["backfilled"] = True
                return
            if attempt < 5:
                time.sleep(0.2)

    def _run_log_stream(self, log_client, str job_id, callback, from_byte=0):
        try:
            log_client._stream_logs(job_id, callback, from_byte)
        except Exception as exc:
            self._log_thread_errors[job_id] = exc

    def _stream_logs(self, str job_id, callback, from_byte=0):
        cdef bint ok = self._client.get().stream_logs(
            job_id.encode("utf-8"),
            from_byte,
            _invoke_log_callback,
            <void*>callback,
        )
        if not ok:
            msg = self._client.get().last_error().decode("utf-8")
            raise GrpcError(msg or "log stream failed")

    def incumbents(self, str job_id, from_index=0):
        """
        Return incumbent solutions collected so far (or all remaining).

        Works while the job is running or after it completes. Each entry is a
        dict with ``index``, ``objective``, and ``assignment`` (list of floats).
        """
        cdef grpc_incumbents_result_t outcome = self._client.get().fetch_incumbents(
            job_id.encode("utf-8"), from_index, 0
        )
        if not outcome.success:
            msg = outcome.error_message.decode("utf-8")
            if not msg:
                msg = self._client.get().last_error().decode("utf-8")
            raise GrpcError(msg or "failed to fetch incumbents")
        return [
            {
                "index": entry.index,
                "objective": entry.objective,
                "assignment": [v for v in entry.assignment],
            }
            for entry in outcome.incumbents
        ]

    def start_incumbent_stream(
        self,
        str job_id,
        callback=None,
        settings=None,
        from_index=0,
        poll_interval_ms=1000,
    ):
        """
        Poll for MIP incumbent solutions on a background thread until the job
        completes.

        ``callback`` is invoked as ``callback(index, objective, assignment,
        job_complete)``. Return ``False`` to cancel the job. ``assignment`` is
        a list of variable values.

        Alternatively pass ``settings`` with :meth:`SolverSettings.set_mip_callback`
        registered :class:`GetSolutionCallback` instances (same as local solve).

        Call :meth:`join_incumbent_stream` before :meth:`delete`.
        """
        if job_id in self._incumbent_threads:
            raise GrpcError(f"incumbent stream already running for job {job_id}")
        if callback is None and settings is None:
            raise GrpcError("callback or settings is required")

        def combined(index, objective, assignment, job_complete):
            if settings is not None:
                if _forward_incumbent_to_settings(
                    settings, index, objective, assignment, job_complete
                ) is False:
                    return False
            if callback is not None:
                return _call_incumbent_callback(
                    callback, index, objective, assignment, job_complete
                )
            return True

        incumbent_client = Client(self._host, self._port)
        thread = threading.Thread(
            target=self._run_incumbent_stream,
            args=(
                incumbent_client,
                job_id,
                combined,
                from_index,
                poll_interval_ms,
            ),
            daemon=True,
        )
        self._incumbent_threads[job_id] = thread
        thread.start()
        return thread

    def join_incumbent_stream(self, str job_id, timeout=None):
        """Wait for a background incumbent poll started by :meth:`start_incumbent_stream`."""
        thread = self._incumbent_threads.pop(job_id, None)
        if thread is not None:
            thread.join(timeout)
        exc = self._incumbent_thread_errors.pop(job_id, None)
        if exc is not None:
            raise exc

    def _run_incumbent_stream(
        self,
        incumbent_client,
        str job_id,
        callback,
        from_index,
        poll_interval_ms,
    ):
        try:
            incumbent_client._poll_incumbents(
                job_id, callback, from_index, poll_interval_ms
            )
        except Exception as exc:
            self._incumbent_thread_errors[job_id] = exc

    def _poll_incumbents(
        self, str job_id, callback, from_index=0, poll_interval_ms=1000
    ):
        cdef grpc_incumbents_result_t outcome
        cdef int64_t next_index = from_index
        cdef bint job_complete = False
        cdef double objective
        cdef list assignment
        cdef size_t i
        poll_seconds = max(poll_interval_ms, 1) / 1000.0

        while not job_complete:
            outcome = self._client.get().fetch_incumbents(
                job_id.encode("utf-8"), next_index, 0
            )
            if not outcome.success:
                msg = outcome.error_message.decode("utf-8")
                if not msg:
                    msg = self._client.get().last_error().decode("utf-8")
                raise GrpcError(msg or "incumbent poll failed")

            for entry in outcome.incumbents:
                assignment = []
                for i in range(entry.assignment.size()):
                    assignment.append(entry.assignment[i])
                if _call_incumbent_callback(
                    callback, entry.index, entry.objective, assignment, False
                ) is False:
                    self.cancel(job_id)
                    return

            next_index = outcome.next_index
            job_complete = outcome.job_complete
            if job_complete:
                _call_incumbent_callback(callback, 0, 0.0, [], True)
                return

            time.sleep(poll_seconds)

    cdef DataModel _as_data_model(self, problem):
        from cuopt.linear_programming.data_model import DataModel as PyDataModel
        from cuopt.linear_programming.problem import Problem

        if isinstance(problem, PyDataModel):
            return <DataModel>problem
        if isinstance(problem, Problem):
            if problem.model is None:
                problem._to_data_model()
            return <DataModel>problem.model
        raise TypeError(
            "submit() expects a Problem or DataModel, got "
            f"{type(problem).__name__}"
        )


def _is_mip(var_types):
    if len(var_types) == 0:
        return False
    if len(set(map(type, var_types))) == 1:
        if isinstance(var_types[0], bytes):
            return b"I" in var_types or b"S" in var_types
        return "I" in var_types or "S" in var_types
    return any(
        vt == "I" or vt == b"I" or vt == "S" or vt == b"S"
        for vt in var_types
    )
