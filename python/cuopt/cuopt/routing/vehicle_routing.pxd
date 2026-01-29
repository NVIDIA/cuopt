# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibraft.common.handle cimport *

from cuopt.routing.structure.routing_utilities cimport *


cdef extern from "cuopt/routing/solve.hpp" namespace "cuopt::routing":

    ctypedef enum objective_t "cuopt::routing::objective_t":
        COST "cuopt::routing::objective_t::COST"
        TRAVEL_TIME "cuopt::routing::objective_t::TRAVEL_TIME"
        VARIANCE_ROUTE_SIZE "cuopt::routing::objective_t::VARIANCE_ROUTE_SIZE"
        VARIANCE_ROUTE_SERVICE_TIME "cuopt::routing::objective_t::VARIANCE_ROUTE_SERVICE_TIME" # noqa
        PRIZE "cuopt::routing::objective_t::PRIZE"
        VEHICLE_FIXED_COST "cuopt::routing::objective_t::VEHICLE_FIXED_COST"

    ctypedef enum node_type_t "cuopt::routing::node_type_t":
        DEPOT "cuopt::routing::node_type_t::DEPOT"
        PICKUP "cuopt::routing::node_type_t::PICKUP"
        DELIVERY "cuopt::routing::node_type_t::DELIVERY"
        BREAK "cuopt::routing::node_type_t::BREAK"

    cdef cppclass data_model_view_t[i_t, f_t]:
        data_model_view_t() except +
        data_model_view_t(
            const handle_t *handle,
            i_t num_locations,
            i_t fleet_size,
            i_t num_orders
        ) except +
        void add_cost_matrix(
            const f_t* matrix,
            uint8_t vehicle_type
        ) except +
        void add_transit_time_matrix(
            const f_t* secondary_matrix,
            uint8_t vehicle_type
        ) except +
        void set_objective_function(
            const objective_t* objectives,
            const f_t* objective_weights,
            i_t n_objectives
        ) except +
        void add_initial_solutions(
            const i_t* vehicle_ids, const i_t* routes,
            const node_type_t *types, const i_t* sol_offsets,
            i_t n_nodes, i_t n_solutions
        ) except +
        void set_order_locations(const i_t* order_locations) except +
        void set_break_locations(
            const i_t* break_locations, i_t n_break_locations
        ) except +
        void set_vehicle_types(const uint8_t* vehicle_types) except +
        void set_pickup_delivery_pairs(
            const i_t* pickup_indices, const i_t* delivery_indices
        ) except +
        void set_vehicle_time_windows(
            const i_t* vehicle_earliest, const i_t* vehicle_latest
        ) except +
        void set_vehicle_locations(
            const i_t* vehicle_start_locations,
            const i_t* vehicle_return_locations
        ) except +
        void set_drop_return_trips(const bool* drop_return_trips) except +
        void set_skip_first_trips(const bool* skip_first_trips) except +
        void add_vehicle_order_match(
            const int vehicle_id,
            const int* orders,
            const int norders) except +
        void add_order_vehicle_match(
            const int order_id,
            const int* vehicles,
            const int nvehicles) except +
        void set_order_service_times(
            const int* service_times,
            const int vehicle_id) except +
        void add_break_dimension(
            const i_t *break_earliest,
            const i_t *break_latest,
            const i_t *break_duration
        ) except +
        void add_vehicle_break(
            const int vehicle_id,
            const int earliest,
            const int latest,
            const int duration,
            const i_t *break_locations,
            const int n_break_locations
        ) except +
        void add_capacity_dimension(
            const string &name, const i_t *demand, const i_t *capacity
        ) except +
        void set_order_time_windows(
            const i_t *earliest,
            const i_t *latest) except +
        void set_order_prizes(
            const f_t *prizes) except +
        void add_order_precedence(
            i_t node_id,
            const i_t *preceding_nodes,
            i_t n_prec_nodes) except +
        void set_min_vehicles(i_t min_vehicles) except+
        void set_vehicle_max_costs(const f_t *max_costs) except+
        void set_vehicle_max_times(const f_t *max_times) except+
        void set_vehicle_fixed_costs(const f_t *vehicle_fixed_costs) except+
        i_t get_num_locations() except+
        i_t get_fleet_size() except+
        i_t get_num_orders() except+
        i_t get_min_vehicles() except+

    cdef cppclass solver_settings_t[i_t, f_t]:
        solver_settings_t() except +
        void set_time_limit(f_t seconds) except+
        void set_verbose_mode(bool verbose) except+
        void set_error_logging_mode(bool logging) except+
        void dump_best_results(const string &file_path, i_t interval) except+

        f_t get_time_limit() except+

cdef extern from "cuopt/routing/cython/cython.hpp" namespace "cuopt::cython": # noqa
    cdef unique_ptr[vehicle_routing_ret_t] call_solve(
        data_model_view_t[int, float]* data_model,
        solver_settings_t[int, float]* solver_settings
    ) except +

    cdef vector[unique_ptr[vehicle_routing_ret_t]] call_batch_solve(
        vector[data_model_view_t[int, float] *] data_models,
        solver_settings_t[int, float]* solver_settings
    ) except +
