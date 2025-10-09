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

from pylibraft.common.handle cimport *

from cuopt.routing.structure.routing_utilities cimport *
from cuopt.routing.vehicle_routing cimport (
    call_solve,
    data_model_view_t,
    node_type_t,
    objective_t,
    solver_settings_t,
)

from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from cuopt.routing.assignment import Assignment
from cuopt.utilities import type_cast

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, strcpy, strlen
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from rmm.pylibrmm.device_buffer cimport DeviceBuffer

import math
import sys
import warnings
from enum import IntEnum

import cupy as cp
import numpy as np
from numba import cuda

import cudf
from cudf.core.buffer import as_buffer


class ErrorStatus(IntEnum):
    Success = error_type_t.Success
    ValidationError = error_type_t.ValidationError
    OutOfMemoryError = error_type_t.OutOfMemoryError
    RuntimeError = error_type_t.RuntimeError


def type_cast(cudf_obj, np_type, name):
    if isinstance(cudf_obj, cudf.Series):
        cudf_type = cudf_obj.dtype
    elif isinstance(cudf_obj, cudf.DataFrame):
        if all([np.issubdtype(dtype, np.number) for dtype in cudf_obj.dtypes]):  # noqa
            cudf_type = cudf_obj.dtypes[0]
        else:
            msg = "All columns in " + name + " should be numeric"
            raise Exception(msg)
    if ((np.issubdtype(np_type, np.floating) and
         (not np.issubdtype(cudf_type, np.floating)))
       or (np.issubdtype(np_type, np.integer) and
           (not np.issubdtype(cudf_type, np.integer)))
       or (np.issubdtype(np_type, np.bool_) and
           (not np.issubdtype(cudf_type, np.bool_)))):
        msg = "Casting " + name + " from " + str(cudf_type) + " to " + str(np.dtype(np_type))  # noqa
        warnings.warn(msg)
    cudf_obj = cudf_obj.astype(np.dtype(np_type))
    return cudf_obj


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_message_line = exc_value.__str__().split("\n")[0]
    error_message = ""
    if error_message_line[:13] == "cuOpt failure":
        for i in error_message_line.split(":")[1:]:
            error_message = error_message + i
        print(error_message)
    else:
        print(exc_value.__str__())
    with open('error_log.txt', 'w') as f:
        f.write(exc_value.__str__())


sys.excepthook = handle_exception


class Objective(IntEnum):
    """
    Enums to configure objective of the solution

    COST                - Models with respect to total cost

    TRAVEL_TIME         - Models with respect to travel time (This includes drive, service and wait time) # noqa

    VARIANCE_ROUTE_SIZE - Models with respect to dissimilarity of route sizes

        It computes the L2 variance (squared) in the number of orders served
        by each route.

    VARIANCE_ROUTE_SERVICE_TIME - Models with respect to disssimilarty of route
                                  service times

        It computes L2 variance (squared) of the accumulated service times of
        of each route

    PRIZE               - Models with respect to prizes collected by the
                          serviced orders
    VEHICLE_FIXED_COST                - Models cost per vehicle. Enabled when set_vehicle_fixed_costs is used.
    """

    COST = objective_t.COST
    TRAVEL_TIME = objective_t.TRAVEL_TIME
    VARIANCE_ROUTE_SIZE = objective_t.VARIANCE_ROUTE_SIZE
    VARIANCE_ROUTE_SERVICE_TIME = objective_t.VARIANCE_ROUTE_SERVICE_TIME
    PRIZE = objective_t.PRIZE
    VEHICLE_FIXED_COST = objective_t.VEHICLE_FIXED_COST


class NodeType(IntEnum):
    """
    Types for nodes/route returned by the solver

    DEPOT    - Indicates whether a given node is depot
    PICKUP   - Indicates whether a given node corresponds to a pickup
    DELIVERY - Indicates whether a given node corresponds to a delivery
    BREAK    - Indicates whether a given node corresponds to a break
    """
    DEPOT = node_type_t.DEPOT
    PICKUP = node_type_t.PICKUP
    DELIVERY = node_type_t.DELIVERY
    BREAK = node_type_t.BREAK


cdef class DataModel:

    cdef unique_ptr[data_model_view_t[int, float]] c_data_model_view
    cdef unique_ptr[handle_t] handle_ptr

    def __init__(self, int num_locations, int fleet_size, int n_orders=-1):
        cdef handle_t* handle_ = <handle_t*><size_t>NULL

        self.handle_ptr.reset(new handle_t())
        handle_ = self.handle_ptr.get()

        self.c_data_model_view.reset(new data_model_view_t[int, float](
            handle_,
            num_locations,
            fleet_size,
            n_orders
        ))
        self.costs = {}
        self.transit_times = {}

        self.demand_name = []
        self.demand = []
        self.capacity = []
        self.order_locations = cudf.Series()
        self.order_earliest = cudf.Series()
        self.order_latest = cudf.Series()
        self.order_prizes = cudf.Series()
        self.pickup_indices = cudf.Series()
        self.delivery_indices = cudf.Series()
        self.objectives = cudf.Series()
        self.objective_weights = cudf.Series()
        self.break_locations = cudf.Series()
        self.break_earliest = []
        self.break_latest = []
        self.break_duration = []

        self.non_uniform_breaks = {}

        self.vehicle_earliest = cudf.Series()
        self.vehicle_latest = cudf.Series()
        self.vehicle_types = cudf.Series()
        self.vehicle_start_locations = cudf.Series()
        self.vehicle_return_locations = cudf.Series()
        self.vehicle_drop_return_trips = cudf.Series()
        self.vehicle_skip_first_trips = cudf.Series()
        self.vehicle_max_costs = cudf.Series()
        self.vehicle_max_times = cudf.Series()
        self.vehicle_fixed_costs = cudf.Series()

        self.vehicle_order_match = {}
        self.order_vehicle_match = {}
        self.order_service_times = {}

        self.initial_vehicle_ids = cudf.Series()
        self.initial_routes = cudf.Series()
        self.initial_types = cudf.Series()
        self.initial_sol_offsets = cudf.Series()

    def add_cost_matrix(self, costs, vehicle_type):
        costs = type_cast(costs, np.float32, "cost_matrix")

        costs = cp.array(costs.to_cupy(), order='C', dtype=np.float32)
        self.costs[vehicle_type] = costs
        cdef uintptr_t c_costs = self.costs[vehicle_type].data.ptr
        self.c_data_model_view.get().add_cost_matrix(
            <const float *> c_costs, <uint8_t> vehicle_type
        )

    def add_transit_time_matrix(self, times, vehicle_type):
        times = type_cast(times, np.float32, "transit_time_matrix")

        times = cp.array(times.to_cupy(), order='C', dtype=np.float32)
        self.transit_times[vehicle_type] = times
        cdef uintptr_t c_times = self.transit_times[vehicle_type].data.ptr
        self.c_data_model_view.get().add_transit_time_matrix(
            <const float *> c_times, <uint8_t> vehicle_type
        )

    def set_break_locations(self, break_locations):
        self.break_locations = type_cast(break_locations, np.int32,
                                         "break_locations")
        cdef uintptr_t c_break_locations = (
            self.break_locations.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_break_locations(
            <const int *>c_break_locations, self.break_locations.shape[0]
        )

    def set_objective_function(self, objectives, objective_weights):
        self.objectives = type_cast(objectives, np.int32, "objectives")
        cdef uintptr_t c_objectives = (
            self.objectives.__cuda_array_interface__['data'][0]
        )
        self.objective_weights = type_cast(objective_weights, np.float32,
                                           "objective_weights")
        cdef uintptr_t c_objective_weights = (
            self.objective_weights.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_objective_function(
            <const objective_t*> c_objectives,
            <const float*> c_objective_weights,
            self.objectives.shape[0]
        )

    def add_initial_solutions(self, vehicle_ids, routes, types, sol_offsets):
        def get_type_from_str(type_in_str):
            if type_in_str == "Depot":
                return 0
            elif type_in_str == "Pickup":
                return 1
            elif type_in_str == "Delivery":
                return 2
            elif type_in_str == "Break":
                return 3
        self.initial_vehicle_ids = type_cast(vehicle_ids, np.int32,
                                             "initial_vehicle_ids")
        self.initial_routes = type_cast(routes, np.int32,
                                        "initial_routes")
        self.initial_sol_offsets = type_cast(sol_offsets, np.int32,
                                             "initial_sol_offsets")
        node_types_int = cudf.Series([
            get_type_from_str(type_in_str)
            for type_in_str in types.to_pandas()])
        self.initial_types = type_cast(node_types_int, np.uint8, "types")

        cdef uintptr_t c_initial_vehicle_ids = (
            self.initial_vehicle_ids.__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_initial_routes = (
            self.initial_routes.__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_initial_sol_offsets = (
            self.initial_sol_offsets.__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_initial_types = (
            self.initial_types.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().add_initial_solutions(
            <const int *>c_initial_vehicle_ids,
            <const int*>c_initial_routes,
            <const node_type_t*>c_initial_types,
            <const int*>c_initial_sol_offsets,
            self.initial_routes.shape[0],
            self.initial_sol_offsets.shape[0]
        )

    def set_order_locations(self, order_locations):
        self.order_locations = type_cast(order_locations, np.int32,
                                         "order_locations")
        cdef uintptr_t c_order_locations = (
            self.order_locations.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_order_locations(
            <const int *> c_order_locations
        )

    def set_vehicle_types(self, vehicle_types):
        self.vehicle_types = type_cast(
            vehicle_types, np.uint8, "vehicle_types"
        )

        cdef uintptr_t c_vehicle_types = (
            self.vehicle_types.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_vehicle_types(
            <const uint8_t *> c_vehicle_types
        )

    def set_pickup_delivery_pairs(self, pickup_indices, delivery_indices):
        self.pickup_indices = type_cast(pickup_indices, np.int32,
                                        "pickup_indices")
        self.delivery_indices = type_cast(delivery_indices, np.int32,
                                          "delivery_indices")
        cdef uintptr_t c_pickup_indices = (
            self.pickup_indices.__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_delivery_indices = (
            self.delivery_indices.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_pickup_delivery_pairs(
            <const int *> c_pickup_indices,
            <const int *> c_delivery_indices
        )

    def set_vehicle_time_windows(self, vehicle_earliest, vehicle_latest):
        self.vehicle_earliest = type_cast(
            vehicle_earliest, np.int32, "vehicle_earliest"
        )

        self.vehicle_latest = type_cast(
            vehicle_latest, np.int32, "vehicle_latest"
        )

        cdef uintptr_t c_vehicle_earliest = (
            self.vehicle_earliest.__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_vehicle_latest = (
            self.vehicle_latest.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_vehicle_time_windows(
            <const int *> c_vehicle_earliest,
            <const int *> c_vehicle_latest
        )

    def set_vehicle_locations(
        self, vehicle_start_locations, vehicle_return_locations
    ):
        self.vehicle_start_locations = type_cast(
            vehicle_start_locations, np.int32, "vehicle_start_locations"
        )
        self.vehicle_return_locations = type_cast(
            vehicle_return_locations, np.int32, "vehicle_return_locations"
        )

        cdef uintptr_t c_vehicle_start_locations = (
            self.vehicle_start_locations.__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_vehicle_return_locations = (
            self.vehicle_return_locations.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_vehicle_locations(
            <const int *> c_vehicle_start_locations,
            <const int *> c_vehicle_return_locations
        )

    def set_drop_return_trips(self, set_drop_return_trips):
        self.vehicle_drop_return_trips = type_cast(
            set_drop_return_trips, np.bool_, "drop return trips"
        )
        cdef uintptr_t c_drop_return_trips = (
            self.vehicle_drop_return_trips.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_drop_return_trips(
            <bool*>c_drop_return_trips
        )

    def set_skip_first_trips(self, set_skip_first_trips):
        self.vehicle_skip_first_trips = type_cast(
            set_skip_first_trips, np.bool_, "skip first trips"
        )
        cdef uintptr_t c_skip_first_trips = (
            self.vehicle_skip_first_trips.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_skip_first_trips(
            <bool*>c_skip_first_trips
        )

    def add_vehicle_order_match(self, v_id, orders):
        self.vehicle_order_match[v_id] = type_cast(
            orders, np.int32,
            "orders in vehicle_order_match"
        )
        cdef uintptr_t c_orders =\
            self.vehicle_order_match[v_id].__cuda_array_interface__['data'][0]
        self.c_data_model_view.get().add_vehicle_order_match(
            v_id, <const int *> c_orders, len(orders))

    def add_order_vehicle_match(self, o_id, vehicles):
        n_fleet = self.get_fleet_size()
        self.order_vehicle_match[o_id] = type_cast(
            vehicles, np.int32, "vehicles"
        )
        cdef uintptr_t c_vehicles =\
            self.order_vehicle_match[o_id].__cuda_array_interface__['data'][0]
        self.c_data_model_view.get().add_order_vehicle_match(
            o_id, <const int *> c_vehicles, len(vehicles))

    def set_order_service_times(self, service_times, vehicle_id=-1):
        n_fleet = self.get_fleet_size()
        self.order_service_times[vehicle_id] = type_cast(
            service_times, np.int32, "service_times"
        )
        cdef uintptr_t c_service_times =\
            self.order_service_times[vehicle_id] \
            .__cuda_array_interface__['data'][0]
        self.c_data_model_view.get().set_order_service_times(
            <const int *> c_service_times, vehicle_id)

    def add_break_dimension(
        self, break_earliest, break_latest, break_duration
    ):
        self.break_earliest.append(
            type_cast(break_earliest, np.int32, "break_earliest")
        )
        self.break_latest.append(
            type_cast(break_latest, np.int32, "break_latest")
        )
        self.break_duration.append(
            type_cast(break_duration, np.int32, "break_duration")
        )
        cdef uintptr_t c_break_earliest = (
            self.break_earliest[-1].__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_break_latest = (
            self.break_latest[-1].__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_break_duration = (
            self.break_duration[-1].__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().add_break_dimension(
            <const int*>c_break_earliest,
            <const int*> c_break_latest,
            <const int*> c_break_duration
        )

    def add_vehicle_break(
        self, vehicle_id, earliest, latest, duration, locations
    ):
        dim = 0
        if vehicle_id in self.non_uniform_breaks:
            dim = len(self.non_uniform_breaks[vehicle_id])

        if dim == 0:
            self.non_uniform_breaks[vehicle_id] = {}

        self.non_uniform_breaks[vehicle_id][dim] = {
            "earliest": earliest,
            "latest": latest,
            "duration": duration,
            "locations": type_cast(
                locations, np.int32, "breaklocations"
            )
        }

        current_breaks = self.non_uniform_breaks[vehicle_id][dim]["locations"]

        cdef uintptr_t c_locations_ptr = (
            current_breaks.__cuda_array_interface__['data'][0]
        )

        self.c_data_model_view.get().add_vehicle_break(
            vehicle_id, earliest, latest, duration,
            <const int *> c_locations_ptr, len(locations))

    def add_capacity_dimension(self, name, demand, capacity):
        self.demand_name.append(name)
        self.demand.append(type_cast(demand, np.int32, "demand"))
        self.capacity.append(type_cast(capacity, np.int32, "capacity"))
        cdef uintptr_t c_demand_ptr = (
            self.demand[-1].__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_capacity_ptr = (
            self.capacity[-1].__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().add_capacity_dimension(
            name.encode('utf-8'), <int*>c_demand_ptr, <int*>c_capacity_ptr
        )

    def set_order_time_windows(
        self, earliest, latest
    ):
        self.order_earliest = type_cast(earliest, np.int32, "earliest")
        self.order_latest = type_cast(latest, np.int32, "latest")

        cdef uintptr_t c_earliest = (
            self.order_earliest.__cuda_array_interface__['data'][0]
        )
        cdef uintptr_t c_latest = (
            self.order_latest.__cuda_array_interface__['data'][0]
        )

        self.c_data_model_view.get().set_order_time_windows(
            <int*>c_earliest,
            <int*>c_latest
        )

    def set_order_prizes(self, prizes):
        self.order_prizes = prizes.astype(np.dtype(np.float32))

        cdef uintptr_t c_prizes = (
            self.order_prizes.__cuda_array_interface__['data'][0]
        )

        self.c_data_model_view.get().set_order_prizes(
            <const float *> c_prizes
        )

    def add_order_precedence(self, node_id, preceding_nodes):
        self.node_ids_of_precedence.append(node_id)
        self.preceding_nodes_list.append(type_cast(preceding_nodes, np.int32,
                                                   "preceding_nodes"))
        cdef uintptr_t c_prec_ptr = (
            self.preceding_nodes_list[-1].__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().add_order_precedence(
            node_id, <int*>c_prec_ptr, preceding_nodes.shape[0]
        )

    def set_vehicle_max_costs(self, vehicle_max_costs):
        self.vehicle_max_costs = type_cast(
            vehicle_max_costs,
            np.float32,
            "vehicle_max_costs"
        )

        cdef uintptr_t c_vehicle_max_costs = (
            self.vehicle_max_costs.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_vehicle_max_costs(
            <float*>c_vehicle_max_costs
        )

    def set_vehicle_max_times(self, vehicle_max_times):
        self.vehicle_max_times = type_cast(
            vehicle_max_times,
            np.float32,
            "vehicle_max_times"
        )

        cdef uintptr_t c_vehicle_max_times = (
            self.vehicle_max_times.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_vehicle_max_times(
            <float*>c_vehicle_max_times
        )

    def set_vehicle_fixed_costs(self, vehicle_fixed_costs):
        self.vehicle_fixed_costs = type_cast(
            vehicle_fixed_costs,
            np.float32,
            "vehicle_fixed_costs"
        )

        cdef uintptr_t c_vehicle_fixed_costs = (
            self.vehicle_fixed_costs.__cuda_array_interface__['data'][0]
        )
        self.c_data_model_view.get().set_vehicle_fixed_costs(
            <float*>c_vehicle_fixed_costs
        )

    def set_min_vehicles(self, min_vehicles):
        self.c_data_model_view.get().set_min_vehicles(<int>min_vehicles)

    def get_num_locations(self):
        return self.c_data_model_view.get().get_num_locations()

    def get_fleet_size(self):
        return self.c_data_model_view.get().get_fleet_size()

    def get_num_orders(self):
        return self.c_data_model_view.get().get_num_orders()

    def get_cost_matrix(self, vehicle_type):
        if self.costs and vehicle_type in self.costs:
            return self.costs[vehicle_type]
        else:
            raise ValueError(
                f"There is no cost matrix for given vehicle type: {vehicle_type}", # noqa
            )

    def get_transit_time_matrix(self, vehicle_type):
        if self.transit_times and vehicle_type in self.transit_times:
            return self.transit_times[vehicle_type]
        else:
            raise ValueError(
                f"There is no transit time matrix for given vehicle type: {vehicle_type}", # noqa
            )

    def get_transit_time_matrices(self):
        return self.transit_times

    def get_initial_solutions(self):
        return {self.initial_vehicle_ids,
                self.initial_routes,
                self.initial_types,
                self.initial_sol_offsets
                }

    def get_order_locations(self):
        return self.order_locations

    def get_vehicle_types(self):
        return self.vehicle_types

    def get_pickup_delivery_pairs(self):
        return self.pickup_indices, self.delivery_indices

    def get_vehicle_time_windows(self):
        return self.vehicle_earliest, self.vehicle_latest

    def get_vehicle_locations(self):
        return self.vehicle_start_locations, self.vehicle_return_locations

    def get_drop_return_trips(self):
        return self.vehicle_drop_return_trips

    def get_skip_first_trips(self):
        return self.vehicle_skip_first_trips

    def get_capacity_dimensions(self):
        return {
            name: {
                "demand": self.demand[i],
                "capacity": self.capacity[i]
            }
            for i, name in enumerate(self.demand_name)
        }

    def get_vehicle_order_match(self):
        return self.vehicle_order_match

    def get_order_vehicle_match(self):
        return {
            ord_id: vehs for ord_id, vehs in self.order_vehicle_match.items()
        }

    def get_order_service_times(self, vehicle_id):
        return self.order_service_times[vehicle_id]

    def get_order_time_windows(self):
        return (
            self.order_earliest,
            self.order_latest,
        )

    def get_order_prizes(self):
        return self.order_prizes

    def get_break_locations(self):
        return self.break_locations

    def get_break_dimensions(self):
        num_breaks = len(self.break_earliest)
        return {
            'break_'+str(i): {
                "earliest": self.break_earliest[i],
                "latest": self.break_latest[i],
                "duration": self.break_duration[i],
            }
            for i in range(num_breaks)
        }

    def get_non_uniform_breaks(self):
        return self.non_uniform_breaks

    def get_objective_function(self):
        return self.objectives, self.objective_weights

    def get_vehicle_max_costs(self):
        return self.vehicle_max_costs

    def get_vehicle_max_times(self):
        return self.vehicle_max_times

    def get_vehicle_fixed_costs(self):
        return self.vehicle_fixed_costs

    def get_min_vehicles(self):
        return self.c_data_model_view.get().get_min_vehicles()


cdef class SolverSettings:
    cdef unique_ptr[solver_settings_t[int, float]] c_solver_settings
    file_path = None
    interval = None
    config_file_path = None

    def __init__(self):
        self.c_solver_settings.reset(new solver_settings_t[int, float]())

    def set_time_limit(self, seconds):
        self.c_solver_settings.get().set_time_limit(<float>seconds)

    def set_verbose_mode(self, bool verbose):
        self.c_solver_settings.get().set_verbose_mode(verbose)

    def set_error_logging_mode(self, bool logging):
        self.c_solver_settings.get().set_error_logging_mode(logging)

    def dump_best_results(self, file_path, interval):
        self.file_path = file_path
        self.interval = interval
        self.c_solver_settings.get().dump_best_results(
            file_path.encode('utf-8'), interval
        )

    def dump_config_file(self, file_name):
        self.config_file_path = file_name

    def get_time_limit(self):
        return self.c_solver_settings.get().get_time_limit()

    def get_best_results_file_path(self):
        return self.file_path

    def get_config_file_name(self):
        return self.config_file_path

    def get_best_results_interval(self):
        return self.interval

cdef char* c_get_string(string in_str):
    cdef char* c_string = <char *> malloc((in_str.length()+1) * sizeof(char))
    if not c_string:
        return NULL  # malloc failed
    # copy except the terminating char
    strcpy(c_string, in_str.c_str())
    return c_string


def Solve(DataModel data_model, SolverSettings solver_settings):
    cdef data_model_view_t[int, float]* c_data_model_view = (
        data_model.c_data_model_view.get()
    )
    cdef solver_settings_t[int, float]* c_solver_settings = (
        solver_settings.c_solver_settings.get()
    )

    vr_ret_ptr = move(call_solve(
        c_data_model_view,
        c_solver_settings
    ))

    vr_ret = move(vr_ret_ptr.get()[0])
    vehicle_count = vr_ret.vehicle_count_
    total_objective_value = vr_ret.total_objective_value_
    objective_values = vr_ret.objective_values_

    objective_values = {}
    for k in vr_ret.objective_values_:
        obj = Objective(int(k.first))
        objective_values[obj] = k.second

    status = vr_ret.status_
    cdef char* c_sol_string = c_get_string(vr_ret.solution_string_)
    try:
        # Performs a copy of the data
        solver_status_string = \
            c_sol_string[:vr_ret.solution_string_.length()].decode('UTF-8')
    finally:
        free(c_sol_string)

    route = DeviceBuffer.c_from_unique_ptr(move(vr_ret.d_route_))
    route_locations = DeviceBuffer.c_from_unique_ptr(
        move(vr_ret.d_route_locations_)
    )
    arrival_stamp = DeviceBuffer.c_from_unique_ptr(
        move(vr_ret.d_arrival_stamp_)
    )
    truck_id = DeviceBuffer.c_from_unique_ptr(move(vr_ret.d_truck_id_))
    node_types = DeviceBuffer.c_from_unique_ptr(move(vr_ret.d_node_types_))
    unserviced_nodes = \
        DeviceBuffer.c_from_unique_ptr(move(vr_ret.d_unserviced_nodes_))
    accepted = \
        DeviceBuffer.c_from_unique_ptr(move(vr_ret.d_accepted_))

    route = as_buffer(route)
    route_locations = as_buffer(route_locations)
    arrival_stamp = as_buffer(arrival_stamp)
    truck_id = as_buffer(truck_id)
    node_types = as_buffer(node_types)
    unserviced_nodes = as_buffer(unserviced_nodes)
    accepted = as_buffer(accepted)

    route_df = cudf.DataFrame()
    route_df['route'] = cudf.core.column.build_column(
        route, dtype=np.dtype(np.int32)
    )
    route_df['arrival_stamp'] = cudf.core.column.build_column(
        arrival_stamp, dtype=np.dtype(np.float64)
    )
    route_df['truck_id'] = cudf.core.column.build_column(
        truck_id, dtype=np.dtype(np.int32)
    )
    route_df['location'] = cudf.core.column.build_column(
        route_locations, dtype=np.dtype(np.int32)
    )
    route_df['type'] = cudf.core.column.build_column(
        node_types, dtype=np.dtype(np.int32)
    )

    unserviced_nodes = cudf.Series._from_column(
        cudf.core.column.build_column(
            unserviced_nodes, dtype=np.dtype(np.int32)
        )
    )
    accepted = cudf.Series._from_column(
        cudf.core.column.build_column(accepted, dtype=np.dtype(np.int32))
    )

    def get_type_from_int(type_in_int):
        if type_in_int == int(NodeType.DEPOT):
            return "Depot"
        elif type_in_int == int(NodeType.PICKUP):
            return "Pickup"
        elif type_in_int == int(NodeType.DELIVERY):
            return "Delivery"
        elif type_in_int == int(NodeType.BREAK):
            return "Break"

    node_types_string = [
        get_type_from_int(type_in_int)
        for type_in_int in route_df['type'].to_pandas()]
    route_df['type'] = node_types_string
    error_status = vr_ret.error_status_
    error_message = vr_ret.error_message_

    return Assignment(
        vehicle_count,
        total_objective_value,
        objective_values,
        route_df,
        accepted,
        <solution_status_t> status,
        solver_status_string,
        <error_type_t> error_status,
        error_message,
        unserviced_nodes
    )
