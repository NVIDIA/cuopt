# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

"""Compiled gRPC clients for remote cuOpt solves.

One extension module holding both arms, because both wrap the same C++ object
(``cuopt::cython::grpc_python_client_t``). Keeping them in one unit means the
client can later be detached from the solver engines as a single GPU-free
package rather than two that would each need their own copy of the transport.

The two public classes stay separate -- their feature sets genuinely differ
(log streaming, incumbents and chunked upload are LP-only; VRP is unary-only) --
and both keep their original import paths via ``cuopt.grpc.linear_programming``
and ``cuopt.grpc.routing``.
"""

from cuopt.grpc.client.grpc_client cimport (
    COMPLETED,
    cpu_capacity_dimension_t,
    cpu_cost_matrix_t,
    cpu_routing_problem_t,
    cpu_routing_solution_t,
    cpu_uniform_break_t,
    cpu_vehicle_break_t,
    grpc_incumbents_result_t,
    grpc_job_status_t,
    grpc_logs_result_t,
    grpc_log_line_callback_t,
    grpc_python_client_connect_options_t,
    grpc_python_client_t,
    grpc_python_tls_mode_t,
    grpc_result_outcome_t,
    grpc_status_result_t,
    grpc_submit_result_t,
    grpc_vrp_result_outcome_t,
    routing_solver_settings_t,
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

from cython.operator cimport dereference as deref, postincrement as postinc

from enum import IntEnum
import math
import threading
import time
import warnings

from libc.stdint cimport int32_t, int64_t, uint8_t
from libc.stddef cimport size_t
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import numpy as np


# =============================================================================
# LP / MIP arm
# =============================================================================

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


def _load_pem(value):
    """Return PEM contents from a PEM string or a readable file path."""
    if value is None:
        return None
    text = str(value)
    if "-----BEGIN" in text:
        return text
    with open(text, "r", encoding="utf-8") as handle:
        return handle.read()


class TlsConfig:
    """
    TLS / mTLS settings for :class:`Client`.

    Each PEM argument may be PEM text or a path to a PEM file. For mTLS, pass
    both ``client_cert`` and ``client_key``. When ``root_certs`` is omitted,
    the client uses the system/default CA trust store.
    """

    __slots__ = ("root_certs", "client_cert", "client_key")

    def __init__(self, root_certs=None, client_cert=None, client_key=None):
        if (client_cert is None) != (client_key is None):
            raise ValueError(
                "client_cert and client_key must both be set for mTLS, "
                "or neither for server TLS only"
            )
        self.root_certs = _load_pem(root_certs) if root_certs is not None else None
        self.client_cert = _load_pem(client_cert) if client_cert is not None else None
        self.client_key = _load_pem(client_key) if client_key is not None else None


cdef grpc_python_client_connect_options_t _connect_options_from_tls(tls):
    cdef grpc_python_client_connect_options_t options
    options.tls_mode = grpc_python_tls_mode_t.ENV
    options.tls_root_certs = string()
    options.tls_client_cert = string()
    options.tls_client_key = string()

    if tls is False:
        options.tls_mode = grpc_python_tls_mode_t.DISABLED
    elif isinstance(tls, TlsConfig):
        options.tls_mode = grpc_python_tls_mode_t.EXPLICIT
        if tls.root_certs is not None:
            options.tls_root_certs = tls.root_certs.encode("utf-8")
        if tls.client_cert is not None:
            options.tls_client_cert = tls.client_cert.encode("utf-8")
            options.tls_client_key = tls.client_key.encode("utf-8")

    return options


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
    cdef dict _log_threads
    cdef dict _log_thread_errors
    cdef dict _log_stream_state
    cdef dict _incumbent_threads
    cdef dict _incumbent_thread_errors
    cdef str _host
    cdef int _port
    cdef object _tls

    def __init__(self, str host, int port, *, tls=None):
        """
        Connect to ``cuopt_grpc_server`` at ``host:port``.

        ``tls`` controls transport security:

        * ``None`` (default) — read ``CUOPT_TLS_*`` from the environment.
        * ``False`` — plain TCP; ignore ``CUOPT_TLS_*``.
        * :class:`TlsConfig` — explicit TLS/mTLS; omit ``root_certs`` to use the
          system/default CA trust store.
        """
        if tls is not None and tls is not False and not isinstance(tls, TlsConfig):
            raise TypeError("tls must be None, False, or TlsConfig")

        cdef grpc_python_client_connect_options_t options
        cdef string host_cpp = host.encode("utf-8")
        cdef string error_out

        options = _connect_options_from_tls(tls)
        self._client.reset(new grpc_python_client_t(host_cpp, port, options))
        self._log_threads = {}
        self._log_thread_errors = {}
        self._log_stream_state = {}
        self._incumbent_threads = {}
        self._incumbent_thread_errors = {}
        self._host = host
        self._port = port
        self._tls = tls
        if not self._client.get().connect(error_out):
            raise GrpcError(error_out.decode("utf-8"))

    def _spawn_client(self):
        """Create a sibling connection with the same host/port/TLS settings."""
        return Client(self._host, self._port, tls=self._tls)

    def submit(self, problem, SolverSettings settings not None):
        """
        Submit a problem for solving and return its ``job_id``.

        ``problem`` is a :class:`~cuopt.linear_programming.problem.Problem` or
        :class:`~cuopt.linear_programming.data_model.DataModel`. The job runs
        asynchronously; use :meth:`wait` or :meth:`status` to track it and
        :meth:`result` to fetch the solution. Always :meth:`delete` when done.
        """
        cdef DataModel data_model
        cdef grpc_submit_result_t submit_result
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
        return submit_result.job_id.decode("utf-8")

    def status(self, str job_id):
        """
        Return the current :class:`JobStatus` for ``job_id`` without blocking.
        """
        cdef grpc_status_result_t status_result = self._client.get().status(
            job_id.encode("utf-8")
        )
        if not status_result.success:
            raise GrpcError(status_result.error_message.decode("utf-8"))
        return JobStatus(<int>status_result.status)

    def wait(self, str job_id, timeout=None):
        """
        Block until ``job_id`` reaches a terminal state and return its
        :class:`JobStatus`.

        ``timeout`` is in whole seconds. ``None`` waits indefinitely.
        Non-``None`` values are converted with ``int(timeout)`` (so ``0.5``
        becomes ``0`` and waits indefinitely). Positive timeouts poll about
        once per second and raise :class:`GrpcError` if the deadline expires
        (they do not return a non-terminal :class:`JobStatus`).
        """
        cdef int timeout_seconds = 0 if timeout is None else int(timeout)
        cdef grpc_status_result_t wait_result = self._client.get().wait(
            job_id.encode("utf-8"), timeout_seconds
        )
        if not wait_result.success:
            raise GrpcError(wait_result.error_message.decode("utf-8"))
        return JobStatus(<int>wait_result.status)

    def cancel(self, str job_id):
        """
        Request cancellation of a running job. The job moves to
        :attr:`JobStatus.CANCELLED`; call :meth:`delete` to release its state.
        """
        cdef string error_out
        if not self._client.get().cancel(job_id.encode("utf-8"), error_out):
            raise GrpcError(error_out.decode("utf-8"))

    def delete(self, str job_id):
        """
        Cancel ``job_id`` if it is still running, then delete it on the server
        and release its state. Joins any client-side incumbent-stream thread
        for this job first. Call once you no longer need the job's result or
        logs.
        """
        if job_id in self._incumbent_threads:
            self.join_incumbent_stream(job_id)
        cdef string error_out
        if not self._client.get().delete_job(job_id.encode("utf-8"), error_out):
            raise GrpcError(error_out.decode("utf-8"))

    def result(self, str job_id, variable_names=None):
        """
        Fetch the solution for a completed job, or ``None`` if not ready.

        LP vs MIP is determined from the server response (via
        ``grpc_client_t::get_result``). Pass ``variable_names`` (column order)
        to key ``solution.get_vars()`` by name. Raises :class:`GrpcError` if the job
        failed or was cancelled.
        """
        cdef grpc_result_outcome_t outcome
        cdef unique_ptr[solver_ret_t] sol_ret

        outcome = self._client.get().result(job_id.encode("utf-8"))
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
        log_client = self._spawn_client()
        thread = threading.Thread(
            target=self._run_log_stream,
            args=(log_client, job_id, handler, from_byte),
            daemon=True,
        )
        self._log_threads[job_id] = thread
        thread.start()
        return thread

    def join_log_stream(self, str job_id, timeout=None):
        """Wait for the background log-stream thread started by :meth:`start_log_stream`.

        Returns a dict when a thread was started for ``job_id``, else ``None``.
        Useful keys:

        * ``lines`` — list of log line strings collected so far
        * ``live_lines`` — count of lines received from the live stream thread
        * ``backfilled`` — ``True`` if the live stream received no lines and
          this method then called :meth:`logs` as a client-side fallback to
          fill ``lines`` (and re-invoke the callback). That fetch is not
          destructive; the server keeps the log until :meth:`delete`.

        Other keys in the dict are internal; do not rely on them.
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
        settings,
        from_index=0,
        poll_interval_ms=1000,
    ):
        """
        Poll for MIP incumbent solutions on a background thread until the job
        completes.

        Pass ``settings`` with ``GetSolutionCallback`` instances registered via
        :meth:`~cuopt.linear_programming.solver_settings.SolverSettings.set_mip_callback`
        (same as local solve).

        Call :meth:`join_incumbent_stream` before :meth:`delete`.
        """
        if job_id in self._incumbent_threads:
            raise GrpcError(f"incumbent stream already running for job {job_id}")
        if settings is None:
            raise GrpcError("settings is required")

        def combined(index, objective, assignment, job_complete):
            return _forward_incumbent_to_settings(
                settings, index, objective, assignment, job_complete
            )

        incumbent_client = self._spawn_client()
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
        """Wait for the background incumbent-stream thread started by :meth:`start_incumbent_stream`."""
        thread = self._incumbent_threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                exc = self._incumbent_thread_errors.get(job_id)
                if exc is not None:
                    raise exc
                return
            self._incumbent_threads.pop(job_id, None)
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


# =============================================================================
# Routing (VRP) arm
# =============================================================================

class RoutingSolveError(RuntimeError):
    """A remote VRP job failed or returned no routing solution."""


# Recorded setters that _populate() maps into cpu_routing_problem_t. Must match
# the dispatch in _populate; test_grpc_serialization asserts this covers every
# name in cuopt.routing._deferred._SETTERS so a new setter cannot be missed.
HANDLED_SETTERS = frozenset({
    "add_cost_matrix",
    "add_transit_time_matrix",
    "set_order_time_windows",
    "set_vehicle_time_windows",
    "set_vehicle_locations",
    "set_pickup_delivery_pairs",
    "add_capacity_dimension",
    "set_order_service_times",
    "add_vehicle_order_match",
    "add_order_vehicle_match",
    "add_order_precedence",
    "add_break_dimension",
    "add_vehicle_break",
    "set_objective_function",
    "add_initial_solutions",
    "set_min_vehicles",
    "set_order_locations",
    "set_order_prizes",
    "set_vehicle_types",
    "set_drop_return_trips",
    "set_skip_first_trips",
    "set_vehicle_max_costs",
    "set_vehicle_max_times",
    "set_vehicle_fixed_costs",
    "set_break_locations",
})


def _to_host(x):
    """Return a host numpy array from numpy/pandas/cuDF/cupy/list input."""
    if isinstance(x, np.ndarray):
        return x
    root = type(x).__module__.split(".", 1)[0]
    if root in ("pandas", "cudf"):
        return x.to_numpy()
    if root == "cupy":
        return x.get()
    return np.asarray(x)


_NODE_TYPE_NAMES = {
    "Depot": 0,
    "Pickup": 1,
    "Delivery": 2,
    "Break": 3,
}


def _is_node_type_name_array(values):
    """True when values are node-type names rather than wire integers.

    Integer and unsigned arrays (including uint8, the local DataModel storage
    type) pass through. Name arrays are Unicode (U), byte strings (S), NumPy 2
    variable-width strings (T), or object arrays of str/bytes. Object arrays of
    ints must not take the name path.
    """
    kind = values.dtype.kind
    if kind in "UST":
        return True
    if kind != "O" or values.size == 0:
        return False
    # Types are a 1-D sequence. Flatten only for a stray column (n, 1).
    sample = values[0] if values.ndim == 1 else values.reshape(-1)[0]
    return isinstance(sample, (str, bytes, np.str_, np.bytes_))


def _node_type_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _routing_node_types(values):
    """Normalize routing node names or enum values to wire integers."""
    values = np.asarray(_to_host(values))
    if not _is_node_type_name_array(values):
        return values
    try:
        return np.asarray(
            [_NODE_TYPE_NAMES[_node_type_name(value)] for value in values.ravel()],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError(f"unknown routing node type {error.args[0]!r}") from error


# --- numpy -> std::vector fillers ------------------------------------------

cdef void _fill_i32(vector[int32_t]& v, arr) except *:
    cdef int32_t[::1] mv = np.ascontiguousarray(_to_host(arr), dtype=np.int32).ravel()
    cdef Py_ssize_t n = mv.shape[0]
    cdef Py_ssize_t i
    v.resize(n)
    for i in range(n):
        v[i] = mv[i]


cdef void _fill_u8(vector[uint8_t]& v, arr) except *:
    cdef uint8_t[::1] mv = np.ascontiguousarray(_to_host(arr), dtype=np.uint8).ravel()
    cdef Py_ssize_t n = mv.shape[0]
    cdef Py_ssize_t i
    v.resize(n)
    for i in range(n):
        v[i] = mv[i]


cdef void _fill_f32(vector[float]& v, arr) except *:
    cdef float[::1] mv = np.ascontiguousarray(_to_host(arr), dtype=np.float32).ravel()
    cdef Py_ssize_t n = mv.shape[0]
    cdef Py_ssize_t i
    v.resize(n)
    for i in range(n):
        v[i] = mv[i]


# --- std::vector -> numpy ---------------------------------------------------

cdef _i32_to_np(const vector[int32_t]& v):
    cdef Py_ssize_t n = v.size()
    out = np.empty(n, dtype=np.int32)
    cdef int32_t[::1] mv = out
    cdef Py_ssize_t i
    for i in range(n):
        mv[i] = v[i]
    return out


cdef _f64_to_np(const vector[double]& v):
    cdef Py_ssize_t n = v.size()
    out = np.empty(n, dtype=np.float64)
    cdef double[::1] mv = out
    cdef Py_ssize_t i
    for i in range(n):
        mv[i] = v[i]
    return out


# --- DataModel IR -> cpu_routing_problem_t ----------------------------------

cdef void _add_matrix(vector[cpu_cost_matrix_t]& dst, args) except *:
    # _fill_f32 already casts to float32 and C-order ravels (row-major).
    cdef cpu_cost_matrix_t cm
    cm.vehicle_type = <uint8_t>(int(args[1]) if len(args) > 1 else 0)
    _fill_f32(cm.matrix, args[0])
    dst.push_back(cm)


cdef void _populate(cpu_routing_problem_t& p, data_model) except *:
    n_loc, fleet, n_ord = data_model._init_args
    p.num_locations = <int32_t>int(n_loc)
    p.fleet_size = <int32_t>int(fleet)
    p.num_orders = <int32_t>(int(n_loc) if int(n_ord) == -1 else int(n_ord))

    cdef vector[int32_t] tmp_i
    cdef cpu_capacity_dimension_t cap
    cdef cpu_uniform_break_t ub
    cdef cpu_vehicle_break_t vb
    cdef int32_t vid

    for name, args, _ in data_model._calls:
        if name == "add_cost_matrix":
            _add_matrix(p.cost_matrices, args)
        elif name == "add_transit_time_matrix":
            _add_matrix(p.transit_time_matrices, args)
        elif name == "set_order_time_windows":
            _fill_i32(p.order_tw_earliest, args[0])
            _fill_i32(p.order_tw_latest, args[1])
        elif name == "set_vehicle_time_windows":
            _fill_i32(p.vehicle_tw_earliest, args[0])
            _fill_i32(p.vehicle_tw_latest, args[1])
        elif name == "set_vehicle_locations":
            _fill_i32(p.vehicle_start_locations, args[0])
            _fill_i32(p.vehicle_return_locations, args[1])
        elif name == "set_pickup_delivery_pairs":
            _fill_i32(p.pickup_indices, args[0])
            _fill_i32(p.delivery_indices, args[1])
        elif name == "add_capacity_dimension":
            cap = cpu_capacity_dimension_t()
            cap.name = str(args[0]).encode("utf-8")
            _fill_i32(cap.demand, args[1])
            _fill_i32(cap.capacity, args[2])
            p.capacity_dimensions.push_back(cap)
        elif name == "set_order_service_times":
            vid = <int32_t>(int(args[1]) if len(args) > 1 else -1)
            tmp_i.clear()
            _fill_i32(tmp_i, args[0])
            p.order_service_times[vid] = tmp_i
        elif name == "add_vehicle_order_match":
            vid = <int32_t>int(args[0])
            tmp_i.clear()
            _fill_i32(tmp_i, args[1])
            p.vehicle_order_match[vid] = tmp_i
        elif name == "add_order_vehicle_match":
            vid = <int32_t>int(args[0])
            tmp_i.clear()
            _fill_i32(tmp_i, args[1])
            p.order_vehicle_match[vid] = tmp_i
        elif name == "add_order_precedence":
            vid = <int32_t>int(args[0])
            tmp_i.clear()
            _fill_i32(tmp_i, args[1])
            p.order_precedence[vid] = tmp_i
        elif name == "add_break_dimension":
            ub = cpu_uniform_break_t()
            _fill_i32(ub.earliest, args[0])
            _fill_i32(ub.latest, args[1])
            _fill_i32(ub.duration, args[2])
            p.uniform_breaks.push_back(ub)
        elif name == "add_vehicle_break":
            vid = <int32_t>int(args[0])
            vb = cpu_vehicle_break_t()
            vb.earliest = <int32_t>int(args[1])
            vb.latest = <int32_t>int(args[2])
            vb.duration = <int32_t>int(args[3])
            if len(args) > 4 and args[4] is not None:
                _fill_i32(vb.locations, args[4])
            p.vehicle_breaks[vid].push_back(vb)
        elif name == "set_objective_function":
            _fill_i32(p.objectives, args[0])
            _fill_f32(p.objective_weights, args[1])
        elif name == "add_initial_solutions":
            _fill_i32(p.initial_solutions.vehicle_ids, args[0])
            _fill_i32(p.initial_solutions.routes, args[1])
            _fill_i32(
                p.initial_solutions.types, _routing_node_types(args[2])
            )
            _fill_i32(p.initial_solutions.sol_offsets, args[3])
        elif name == "set_min_vehicles":
            p.min_vehicles = <int32_t>int(args[0])
        elif name == "set_order_locations":
            _fill_i32(p.order_locations, args[0])
        elif name == "set_order_prizes":
            _fill_f32(p.order_prizes, args[0])
        elif name == "set_vehicle_types":
            _fill_u8(p.vehicle_types, args[0])
        elif name == "set_drop_return_trips":
            _fill_u8(p.drop_return_trips, args[0])
        elif name == "set_skip_first_trips":
            _fill_u8(p.skip_first_trips, args[0])
        elif name == "set_vehicle_max_costs":
            _fill_f32(p.vehicle_max_costs, args[0])
        elif name == "set_vehicle_max_times":
            _fill_f32(p.vehicle_max_times, args[0])
        elif name == "set_vehicle_fixed_costs":
            _fill_f32(p.vehicle_fixed_costs, args[0])
        elif name == "set_break_locations":
            _fill_i32(p.break_locations, args[0])
        else:
            raise KeyError(
                f"no VRP gRPC mapping for recorded setter {name!r}; add a case "
                "to cuopt.grpc.routing.grpc_client._populate"
            )


def problem_summary(data_model):
    """Populate a ``cpu_routing_problem_t`` from ``data_model`` and return a
    ``{field: size}`` summary. Runs the exact ``_populate`` path used by
    ``submit`` (so a mis-mapped or unmapped setter fails here too), without a
    server. Intended for tests.
    """
    cdef cpu_routing_problem_t p
    _populate(p, data_model)
    return {
        "num_locations": int(p.num_locations),
        "fleet_size": int(p.fleet_size),
        "num_orders": int(p.num_orders),
        "min_vehicles": int(p.min_vehicles),
        "cost_matrices": p.cost_matrices.size(),
        "transit_time_matrices": p.transit_time_matrices.size(),
        "vehicle_start_locations": p.vehicle_start_locations.size(),
        "vehicle_return_locations": p.vehicle_return_locations.size(),
        "vehicle_tw_earliest": p.vehicle_tw_earliest.size(),
        "vehicle_tw_latest": p.vehicle_tw_latest.size(),
        "vehicle_types": p.vehicle_types.size(),
        "drop_return_trips": p.drop_return_trips.size(),
        "skip_first_trips": p.skip_first_trips.size(),
        "vehicle_max_costs": p.vehicle_max_costs.size(),
        "vehicle_max_times": p.vehicle_max_times.size(),
        "vehicle_fixed_costs": p.vehicle_fixed_costs.size(),
        "order_locations": p.order_locations.size(),
        "order_tw_earliest": p.order_tw_earliest.size(),
        "order_tw_latest": p.order_tw_latest.size(),
        "order_prizes": p.order_prizes.size(),
        "order_service_times": p.order_service_times.size(),
        "pickup_indices": p.pickup_indices.size(),
        "delivery_indices": p.delivery_indices.size(),
        "capacity_dimensions": p.capacity_dimensions.size(),
        "break_locations": p.break_locations.size(),
        "uniform_breaks": p.uniform_breaks.size(),
        "vehicle_breaks": p.vehicle_breaks.size(),
        "vehicle_order_match": p.vehicle_order_match.size(),
        "order_vehicle_match": p.order_vehicle_match.size(),
        "order_precedence": p.order_precedence.size(),
        "objectives": p.objectives.size(),
        "objective_weights": p.objective_weights.size(),
        "initial_solutions_routes": p.initial_solutions.routes.size(),
    }


cdef _solution_to_py(cpu_routing_solution_t s):
    cdef dict objectives = {}
    cdef cpp_map[int32_t, double].iterator it = s.objective_values.begin()
    while it != s.objective_values.end():
        objectives[int(deref(it).first)] = float(deref(it).second)
        postinc(it)
    return {
        "status": int(s.status),
        "status_message": s.status_message.decode("utf-8"),
        "error_message": s.error_message.decode("utf-8"),
        "vehicle_count": int(s.vehicle_count),
        "total_objective_value": float(s.total_objective_value),
        "objective_values": objectives,
        "route": _i32_to_np(s.route),
        "truck_id": _i32_to_np(s.truck_id),
        "locations": _i32_to_np(s.locations),
        "node_types": _i32_to_np(s.node_types),
        "arrival_stamp": _f64_to_np(s.arrival_stamp),
        "unserviced_nodes": _i32_to_np(s.unserviced_nodes),
        "accepted": _i32_to_np(s.accepted),
    }


# dump_best_results is intentionally not forwarded. The proto and C++ mapper
# already carry dump_best_results_path, but that writes a debug file on the
# gRPC server host rather than returning data to the client. Wire it up later
# if a remote-debug use case appears.
cdef void _apply_routing_settings(
    routing_solver_settings_t[int, float]& s, settings
) except *:
    if settings is None:
        return
    if isinstance(settings, dict):
        tl = settings.get("time_limit")
        if tl is not None:
            s.set_time_limit(<float>float(tl))
        verbose = settings.get("verbose_mode", settings.get("verbose"))
        if verbose is not None:
            s.set_verbose_mode(<bint>bool(verbose))
        error_logging = settings.get("error_logging")
        if error_logging is not None:
            s.set_error_logging_mode(<bint>bool(error_logging))
        return

    get_time_limit = getattr(settings, "get_time_limit", None)
    if get_time_limit is not None:
        tl = get_time_limit()
        if tl is not None:
            s.set_time_limit(<float>float(tl))
    get_verbose_mode = getattr(settings, "get_verbose_mode", None)
    if get_verbose_mode is not None:
        s.set_verbose_mode(<bint>bool(get_verbose_mode()))
    get_error_logging_mode = getattr(
        settings, "get_error_logging_mode", None
    )
    if get_error_logging_mode is not None:
        s.set_error_logging_mode(<bint>bool(get_error_logging_mode()))


cdef class RoutingClient:
    """Client for solving VRP problems on a remote cuOpt gRPC server."""

    cdef unique_ptr[grpc_python_client_t] _client

    def __cinit__(self, str host, int port, *, tls=None):
        """
        Connect to ``cuopt_grpc_server`` at ``host:port``.

        ``tls`` controls transport security, same as :class:`Client`:

        * ``None`` (default) — read ``CUOPT_TLS_*`` from the environment.
        * ``False`` — plain TCP; ignore ``CUOPT_TLS_*``.
        * :class:`TlsConfig` — explicit TLS/mTLS; omit ``root_certs`` to use the
          system/default CA trust store.
        """
        if tls is not None and tls is not False and not isinstance(tls, TlsConfig):
            raise TypeError("tls must be None, False, or TlsConfig")

        cdef grpc_python_client_connect_options_t options
        cdef string host_cpp = host.encode("utf-8")
        cdef string err

        options = _connect_options_from_tls(tls)
        self._client.reset(new grpc_python_client_t(host_cpp, port, options))
        if not self._client.get().connect(err):
            raise RoutingSolveError(
                "failed to connect: " + err.decode("utf-8")
            )

    cdef _apply_settings(self, routing_solver_settings_t[int, float]& s, settings):
        _apply_routing_settings(s, settings)

    def submit(self, data_model, settings=None):
        """Serialize and submit a VRP problem; return its ``job_id``."""
        cdef cpu_routing_problem_t problem
        cdef routing_solver_settings_t[int, float] cpp_settings
        _populate(problem, data_model)
        self._apply_settings(cpp_settings, settings)
        cdef grpc_submit_result_t sub = self._client.get().submit_vrp(
            &problem, &cpp_settings
        )
        if not sub.success:
            raise RoutingSolveError(sub.error_message.decode("utf-8"))
        return sub.job_id.decode("utf-8")

    def wait(self, str job_id, int timeout=0):
        """Block until the job finishes; return the terminal status int.

        Raises ``RoutingSolveError`` if the wait itself fails (e.g. transport
        error or unknown job), mirroring the LP/MILP client.
        """
        cdef grpc_status_result_t st = self._client.get().wait(
            job_id.encode("utf-8"), timeout
        )
        if not st.success:
            raise RoutingSolveError(st.error_message.decode("utf-8"))
        return <int>st.status

    def result(self, str job_id):
        """Fetch and parse the routing solution for a completed job.

        Returns ``None`` if the job is still in flight (mirrors the LP client).
        """
        cdef grpc_vrp_result_outcome_t out = self._client.get().result_vrp(
            job_id.encode("utf-8")
        )
        if out.not_ready:
            return None
        if not out.success:
            raise RoutingSolveError(out.error_message.decode("utf-8"))
        return _solution_to_py(out.solution)

    def delete(self, str job_id):
        """Delete a job's server-side result; raise ``RoutingSolveError`` on failure."""
        cdef string err
        if not self._client.get().delete_job(job_id.encode("utf-8"), err):
            raise RoutingSolveError(err.decode("utf-8"))

    def solve(self, data_model, settings=None, *, int timeout=0, bint delete=True):
        """Submit, wait, and return the solution (the common path)."""
        job_id = self.submit(data_model, settings)
        try:
            status = self.wait(job_id, timeout)
            if status != <int>COMPLETED:
                # A non-completed terminal status means the solve failed;
                # result() surfaces the server's error_message. If it somehow
                # doesn't raise, fall back to a status-only message.
                self.result(job_id)
                raise RoutingSolveError(
                    f"job {job_id} did not complete (status {status})"
                )
            return self.result(job_id)
        finally:
            if delete:
                self.delete(job_id)
