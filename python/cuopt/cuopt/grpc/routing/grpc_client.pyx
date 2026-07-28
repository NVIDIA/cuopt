# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++

"""Compiled gRPC client for remote VRP solves, consistent with the LP/MILP
client. Builds a host ``cpu_routing_problem_t`` from a routing ``DataModel``
(walking its store-then-build IR) and serializes it in C++; no pure-Python
protobuf path.
"""

from cython.operator cimport dereference as deref, postincrement as postinc
from libc.stdint cimport int32_t, uint8_t
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np

from cuopt.grpc.routing.grpc_client cimport (
    COMPLETED,
    cpu_capacity_dimension_t,
    cpu_cost_matrix_t,
    cpu_routing_problem_t,
    cpu_routing_solution_t,
    cpu_uniform_break_t,
    cpu_vehicle_break_t,
    grpc_python_client_t,
    grpc_status_result_t,
    grpc_submit_result_t,
    grpc_vrp_result_outcome_t,
    solver_settings_t,
)


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
            _fill_i32(p.initial_solutions.types, args[2])
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


cdef class RoutingClient:
    """Client for solving VRP problems on a remote cuOpt gRPC server."""

    cdef unique_ptr[grpc_python_client_t] _client

    def __cinit__(self, str target="localhost:50051"):
        host, _, port = target.rpartition(":")
        if not host:
            host, port = target, "50051"
        cdef string host_cpp = host.encode("utf-8")
        cdef string err
        self._client.reset(new grpc_python_client_t(host_cpp, int(port)))
        if not self._client.get().connect(err):
            raise RoutingSolveError(
                "failed to connect: " + err.decode("utf-8")
            )

    cdef _apply_settings(self, solver_settings_t[int, float]& s, settings):
        if settings is None:
            return
        if isinstance(settings, dict):
            tl = settings.get("time_limit")
            if tl is not None:
                s.set_time_limit(<float>float(tl))
            return
        get_time_limit = getattr(settings, "get_time_limit", None)
        if get_time_limit is not None:
            tl = get_time_limit()
            if tl is not None:
                s.set_time_limit(<float>float(tl))

    def submit(self, data_model, settings=None):
        """Serialize and submit a VRP problem; return its ``job_id``."""
        cdef cpu_routing_problem_t problem
        cdef solver_settings_t[int, float] cpp_settings
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
