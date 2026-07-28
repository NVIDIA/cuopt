# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uint8_t
from libcpp cimport bool
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cuopt/routing/cpu_routing_problem.hpp" namespace "cuopt::routing":  # noqa
    cdef cppclass cpu_cost_matrix_t:
        cpu_cost_matrix_t() except +
        uint8_t vehicle_type
        vector[float] matrix

    cdef cppclass cpu_capacity_dimension_t:
        cpu_capacity_dimension_t() except +
        string name
        vector[int32_t] demand
        vector[int32_t] capacity

    cdef cppclass cpu_uniform_break_t:
        cpu_uniform_break_t() except +
        vector[int32_t] earliest
        vector[int32_t] latest
        vector[int32_t] duration

    cdef cppclass cpu_vehicle_break_t:
        cpu_vehicle_break_t() except +
        int32_t earliest
        int32_t latest
        int32_t duration
        vector[int32_t] locations

    cdef cppclass cpu_initial_solution_t:
        cpu_initial_solution_t() except +
        vector[int32_t] vehicle_ids
        vector[int32_t] routes
        vector[int32_t] types
        vector[int32_t] sol_offsets

    cdef cppclass cpu_routing_problem_t:
        int32_t num_locations
        int32_t fleet_size
        int32_t num_orders
        vector[cpu_cost_matrix_t] cost_matrices
        vector[cpu_cost_matrix_t] transit_time_matrices
        vector[int32_t] vehicle_start_locations
        vector[int32_t] vehicle_return_locations
        vector[int32_t] vehicle_tw_earliest
        vector[int32_t] vehicle_tw_latest
        vector[uint8_t] vehicle_types
        vector[uint8_t] drop_return_trips
        vector[uint8_t] skip_first_trips
        vector[float] vehicle_max_costs
        vector[float] vehicle_max_times
        vector[float] vehicle_fixed_costs
        vector[int32_t] order_locations
        vector[int32_t] order_tw_earliest
        vector[int32_t] order_tw_latest
        vector[float] order_prizes
        cpp_map[int32_t, vector[int32_t]] order_service_times
        vector[int32_t] pickup_indices
        vector[int32_t] delivery_indices
        vector[cpu_capacity_dimension_t] capacity_dimensions
        vector[int32_t] break_locations
        vector[cpu_uniform_break_t] uniform_breaks
        cpp_map[int32_t, vector[cpu_vehicle_break_t]] vehicle_breaks
        cpp_map[int32_t, vector[int32_t]] vehicle_order_match
        cpp_map[int32_t, vector[int32_t]] order_vehicle_match
        cpp_map[int32_t, vector[int32_t]] order_precedence
        vector[int32_t] objectives
        vector[float] objective_weights
        int32_t min_vehicles
        cpu_initial_solution_t initial_solutions

    cdef cppclass cpu_routing_solution_t:
        vector[int32_t] route
        vector[double] arrival_stamp
        vector[int32_t] truck_id
        vector[int32_t] locations
        vector[int32_t] node_types
        vector[int32_t] unserviced_nodes
        vector[int32_t] accepted
        int32_t vehicle_count
        double total_objective_value
        cpp_map[int32_t, double] objective_values
        int32_t status
        string status_message
        string error_message


cdef extern from "cuopt/routing/solver_settings.hpp" namespace "cuopt::routing":  # noqa
    cdef cppclass solver_settings_t[i_t, f_t]:
        solver_settings_t() except +
        void set_time_limit(f_t seconds) except +
        void set_verbose_mode(bool verbose) except +
        void set_error_logging_mode(bool logging) except +


cdef extern from "cuopt/grpc/cython_grpc_client.hpp" namespace "cuopt::cython":  # noqa
    ctypedef enum grpc_job_status_t "cuopt::cython::grpc_job_status_t":
        QUEUED "cuopt::cython::grpc_job_status_t::QUEUED"
        PROCESSING "cuopt::cython::grpc_job_status_t::PROCESSING"
        COMPLETED "cuopt::cython::grpc_job_status_t::COMPLETED"
        FAILED "cuopt::cython::grpc_job_status_t::FAILED"
        CANCELLED "cuopt::cython::grpc_job_status_t::CANCELLED"
        NOT_FOUND "cuopt::cython::grpc_job_status_t::NOT_FOUND"

    cdef cppclass grpc_submit_result_t:
        bool success
        string error_message
        string job_id

    cdef cppclass grpc_status_result_t:
        bint success
        string error_message
        grpc_job_status_t status
        string message
        long long result_size_bytes

    cdef cppclass grpc_vrp_result_outcome_t:
        bool not_ready
        bool success
        string error_message
        cpu_routing_solution_t solution

    cdef cppclass grpc_python_client_t:
        grpc_python_client_t(const string& host, int port) except +
        bool connect(string& error_out) except +
        grpc_submit_result_t submit_vrp(
            cpu_routing_problem_t* problem,
            solver_settings_t[int, float]* settings) except +
        grpc_status_result_t wait(const string& job_id, int timeout_seconds) except +
        grpc_vrp_result_outcome_t result_vrp(const string& job_id) except +
        bool delete_job(const string& job_id, string& error_out) except +
