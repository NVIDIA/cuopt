# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libc.stdint cimport int32_t, uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from pylibraft.common.handle cimport *
from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "cuopt/routing/assignment.hpp" namespace "cuopt::routing":
    ctypedef enum solution_status_t "cuopt::routing::solution_status_t":
        SUCCESS "cuopt::routing::solution_status_t::SUCCESS"
        UNFEASIBLE "cuopt::routing::solution_status_t::UNFEASIBLE"
        TIMEOUT "cuopt::routing::solution_status_t::TIMEOUT"
        EMPTY "cuopt::routing::solution_status_t::EMPTY"

cdef extern from "cuopt/error.hpp" namespace "cuopt": # noqa
    ctypedef enum error_type_t "cuopt::error_type_t": # noqa
        Success "cuopt::error_type_t::Success" # noqa
        ValidationError "cuopt::error_type_t::ValidationError" # noqa
        OutOfMemoryError "cuopt::error_type_t::OutOfMemoryError" # noqa
        RuntimeError "cuopt::error_type_t::RuntimeError" # noqa        

cdef extern from "cuopt/routing/routing_structures.hpp" namespace "cuopt::routing": # noqa
    ctypedef enum objective_t "cuopt::routing::objective_t":
        COST "cuopt::routing::objective_t::COST"
        TRAVEL_TIME "cuopt::routing::objective_t::TRAVEL_TIME"
        VARIANCE_ROUTE_SIZE "cuopt::routing::objective_t::VARIANCE_ROUTE_SIZE"
        VARIANCE_ROUTE_SERVICE_TIME "cuopt::routing::objective_t::VARIANCE_ROUTE_SERVICE_TIME" # noqa
        PRIZE "cuopt::routing::objective_t::PRIZE"
        VEHICLE_FIXED_COST "cuopt::routing::objective_t::VEHICLE_FIXED_COST"


cdef extern from "cuopt/routing/cython/generator.hpp" namespace "cuopt::routing::generator": # noqa
    ctypedef enum dataset_distribution_t "cuopt::routing::generator::dataset_distribution_t": # noqa
        CLUSTERED "cuopt::routing::generator::dataset_distribution_t::CLUSTERED" # noqa
        RANDOM "cuopt::routing::generator::dataset_distribution_t::RANDOM" # noqa
        RANDOM_CLUSTERED "cuopt::routing::generator::dataset_distribution_t::RANDOM_CLUSTERED" # noqa

    cdef cppclass dataset_params_t[i_t, f_t]:
        i_t n_locations
        bool asymmetric
        i_t dim
        i_t *min_demand
        i_t *max_demand
        i_t *min_capacities
        i_t *max_capacities
        i_t min_service_time,
        i_t max_service_time,
        f_t tw_tightness
        f_t drop_return_trips
        i_t n_shifts
        i_t n_vehicle_types,
        i_t n_matrix_types,
        dataset_distribution_t distrib
        f_t center_box_min
        f_t center_box_max
        i_t seed

cdef extern from "cuopt/routing/cython/cython.hpp" namespace "cuopt::cython": # noqa
    cdef cppclass vehicle_routing_ret_t:
        int vehicle_count_
        double total_objective_value_
        map[objective_t, double] objective_values_
        unique_ptr[device_buffer] d_route_
        unique_ptr[device_buffer] d_route_locations_
        unique_ptr[device_buffer] d_route_id_
        unique_ptr[device_buffer] d_arrival_stamp_
        unique_ptr[device_buffer] d_truck_id_
        unique_ptr[device_buffer] d_node_types_
        unique_ptr[device_buffer] d_unserviced_nodes_
        unique_ptr[device_buffer] d_accepted_
        solution_status_t status_
        string solution_string_
        error_type_t error_status_
        string error_message_

    cdef cppclass dataset_ret_t:
        int n_locations_
        unique_ptr[device_buffer] d_x_pos_
        unique_ptr[device_buffer] d_y_pos_
        unique_ptr[device_buffer] d_matrices_
        unique_ptr[device_buffer] d_earliest_time_
        unique_ptr[device_buffer] d_latest_time_
        unique_ptr[device_buffer] d_service_time_
        unique_ptr[device_buffer] d_vehicle_earliest_time_
        unique_ptr[device_buffer] d_vehicle_latest_time_
        unique_ptr[device_buffer] d_drop_return_trips_
        unique_ptr[device_buffer] d_skip_first_trips_
        unique_ptr[device_buffer] d_vehicle_types_
        unique_ptr[device_buffer] d_demands_
        unique_ptr[device_buffer] d_caps_

    cdef void populate_dataset_params[i_t, f_t](
        dataset_params_t[i_t, f_t] &params,
        i_t n_locations,
        bool asymmetric,
        i_t dim,
        const int32_t *min_demand,
        const int32_t *max_demand,
        const int32_t *min_capacities,
        const int32_t *max_capacities,
        i_t min_service_time,
        i_t max_service_time,
        f_t tw_tightness,
        f_t drop_return_trips,
        i_t n_shifts,
        i_t n_vehicle_types,
        i_t n_matrix_types,
        dataset_distribution_t distrib,
        f_t center_box_min,
        f_t center_box_max,
        i_t seed) except +

    cdef unique_ptr[dataset_ret_t] call_generate_dataset(
        const handle_t& handle,
        const dataset_params_t[int, float]& params
    ) except +

cdef extern from "<utility>" namespace "std" nogil:
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
