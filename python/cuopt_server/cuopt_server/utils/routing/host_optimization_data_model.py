# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-array variant of the routing HTTP optimization data model.

The methods intentionally mirror ``OptimizationDataModel``. Only the container
types differ: pandas/numpy keep dense HTTP requests on the CPU until gRPC
serialization. Waypoint graph preparation may still produce device matrices.
"""

import numpy as np
import pandas as pd

from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
    get_none_for_empty_list,
    get_objectives_as_lists,
)
from cuopt_server.utils.routing.validation_cost_matrix import (
    validate_cost_matrix,
)
from cuopt_server.utils.routing.validation_fleet_data import (
    validate_fleet_data,
)
from cuopt_server.utils.routing.validation_solver_config import (
    validate_solver_config,
)
from cuopt_server.utils.routing.validation_task_data import validate_task_data


class HostOptimizationDataModel(OptimizationDataModel):
    """OptimizationDataModel whose request data uses host containers.

    Incremental ``update_*`` methods that wrap cudf on the parent are
    unimplemented here. When the GPU model is deprecated, this class can
    absorb ``OptimizationDataModel`` rather than keep a parallel type.
    """

    def update_cost_matrix(self, *args, **kwargs):
        raise NotImplementedError(
            "HostOptimizationDataModel.update_cost_matrix is unimplemented"
        )

    def update_travel_time_matrix(self, *args, **kwargs):
        raise NotImplementedError(
            "HostOptimizationDataModel.update_travel_time_matrix "
            "is unimplemented"
        )

    def update_fleet_data(self, *args, **kwargs):
        raise NotImplementedError(
            "HostOptimizationDataModel.update_fleet_data is unimplemented"
        )

    def update_task_data(self, *args, **kwargs):
        raise NotImplementedError(
            "HostOptimizationDataModel.update_task_data is unimplemented"
        )

    def update_solver_config(self, *args, **kwargs):
        raise NotImplementedError(
            "HostOptimizationDataModel.update_solver_config is unimplemented"
        )

    def set_cost_matrix(self, cost_matrix):
        is_valid = validate_cost_matrix(
            cost_matrix,
            is_travel_time=False,
            updating=False,
            comparison_matrix=None,
        )
        if is_valid[0]:
            self.is_route_detail_set = True
            self.cost_matrix = {}
            for v_type, matrix in cost_matrix.items():
                np_cost_matrix = np.array(matrix, dtype=np.float32)
                self.cost_matrix[v_type] = pd.DataFrame(np_cost_matrix)

        return is_valid

    def set_travel_time_matrix(self, travel_time_matrix):
        is_valid = validate_cost_matrix(
            travel_time_matrix,
            is_travel_time=True,
            updating=False,
            comparison_matrix=self.cost_matrix,
        )
        if is_valid[0]:
            self.travel_time_matrix = {}
            for v_type, matrix in travel_time_matrix.items():
                np_travel_time_matrix = np.array(matrix, dtype=np.float32)
                self.travel_time_matrix[v_type] = pd.DataFrame(
                    np_travel_time_matrix
                )

        return is_valid

    def set_fleet_data(
        self,
        vehicle_ids,
        vehicle_locations,
        capacities,
        vehicle_time_windows,
        vehicle_breaks,
        vehicle_break_time_windows,
        vehicle_break_durations,
        vehicle_break_locations,
        vehicle_types,
        vehicle_order_match,
        skip_first_trips,
        drop_return_trips,
        min_vehicles,
        vehicle_max_costs,
        vehicle_max_times,
        vehicle_fixed_costs,
        vehicle_distance_breaks=None,
    ):
        if not self.is_route_detail_set:
            return (
                False,
                "Cost matrix/Waypoint graph needs to be set before setting fleet data",  # noqa
            )
        vehicle_types_dict = {}
        vehicle_types_dict["Cost Matrix"] = list(self.cost_matrix.keys())
        vehicle_types_dict["Travel Time Matrix"] = list(
            self.travel_time_matrix.keys()
        )
        vehicle_types_dict["Waypoint Graph"] = list(self.waypoint_graph.keys())
        vehicle_types_dict["Travel Time Waypoint Graph"] = list(
            self.travel_time_waypoint_graph.keys()
        )

        vehicle_ids = get_none_for_empty_list(vehicle_ids)
        capacities = get_none_for_empty_list(capacities)
        vehicle_max_costs = get_none_for_empty_list(vehicle_max_costs)
        vehicle_max_times = get_none_for_empty_list(vehicle_max_times)
        vehicle_fixed_costs = get_none_for_empty_list(vehicle_fixed_costs)
        vehicle_time_windows = get_none_for_empty_list(vehicle_time_windows)
        vehicle_break_time_windows = get_none_for_empty_list(
            vehicle_break_time_windows
        )
        vehicle_break_durations = get_none_for_empty_list(
            vehicle_break_durations
        )
        vehicle_break_locations = get_none_for_empty_list(
            vehicle_break_locations
        )
        vehicle_types = get_none_for_empty_list(vehicle_types)
        vehicle_order_match = get_none_for_empty_list(vehicle_order_match)
        skip_first_trips = get_none_for_empty_list(skip_first_trips)
        drop_return_trips = get_none_for_empty_list(drop_return_trips)
        vehicle_distance_breaks = get_none_for_empty_list(
            vehicle_distance_breaks
        )

        is_valid = validate_fleet_data(
            vehicle_ids,
            vehicle_locations,
            capacities,
            vehicle_time_windows,
            vehicle_breaks,
            vehicle_break_time_windows,
            vehicle_break_durations,
            vehicle_break_locations,
            vehicle_types,
            vehicle_types_dict,
            vehicle_order_match,
            skip_first_trips,
            drop_return_trips,
            min_vehicles,
            vehicle_max_costs,
            vehicle_max_times,
            vehicle_fixed_costs,
            updating=False,
            comparison_locations=None,
            vehicle_distance_breaks=vehicle_distance_breaks,
        )

        if is_valid[0]:
            if vehicle_ids is not None:
                self.fleet_data["vehicle_ids"] = pd.Series(vehicle_ids)
            else:
                self.fleet_data["vehicle_ids"] = pd.Series(
                    range(len(vehicle_locations))
                )
            if vehicle_locations is not None:
                self.fleet_data["vehicle_locations"] = pd.DataFrame(
                    vehicle_locations,
                    columns=["start_location", "end_location"],
                    dtype=np.int32,
                )
            if capacities:
                self.fleet_data["capacities"] = pd.DataFrame(
                    capacities, dtype=np.int32
                ).T
            if vehicle_max_costs is not None:
                self.fleet_data["vehicle_max_costs"] = pd.Series(
                    vehicle_max_costs, dtype=np.float32
                )
            if vehicle_max_times is not None:
                self.fleet_data["vehicle_max_times"] = pd.Series(
                    vehicle_max_times, dtype=np.float32
                )
            if vehicle_fixed_costs is not None:
                self.fleet_data["vehicle_fixed_costs"] = pd.Series(
                    vehicle_fixed_costs, dtype=np.float32
                )
            if vehicle_time_windows:
                self.fleet_data["vehicle_time_windows"] = pd.DataFrame(
                    vehicle_time_windows,
                    columns=["earliest", "latest"],
                    dtype=np.int32,
                )
            if skip_first_trips:
                self.fleet_data["skip_first_trips"] = pd.Series(
                    skip_first_trips, dtype=bool
                )
            if drop_return_trips:
                self.fleet_data["drop_return_trips"] = pd.Series(
                    drop_return_trips, dtype=bool
                )
            if vehicle_break_time_windows and vehicle_break_durations:
                self.fleet_data["vehicle_break_time_windows"] = [
                    pd.DataFrame(
                        val, columns=["earliest", "latest"], dtype=np.int32
                    )
                    for val in vehicle_break_time_windows
                ]

                self.fleet_data["vehicle_break_durations"] = [
                    pd.Series(val, dtype=np.int32)
                    for val in vehicle_break_durations
                ]
            if vehicle_breaks is not None:
                self.fleet_data["vehicle_breaks"] = [
                    {
                        "vehicle_id": data.vehicle_id,
                        "earliest": data.earliest,
                        "latest": data.latest,
                        "duration": data.duration,
                        "locations": data.locations,
                    }
                    for data in vehicle_breaks
                ]
            if vehicle_distance_breaks is not None:
                self.fleet_data["vehicle_distance_breaks"] = [
                    {
                        "vehicle_id": data.vehicle_id,
                        "distance_min": data.distance_min,
                        "distance_max": data.distance_max,
                        "duration": data.duration,
                        "locations": data.locations,
                    }
                    for data in vehicle_distance_breaks
                ]
            if vehicle_order_match is not None:
                self.fleet_data["vehicle_order_match"] = [
                    {
                        "vehicle_id": data.vehicle_id,
                        "order_ids": data.order_ids,
                    }
                    for data in vehicle_order_match
                ]
            if vehicle_break_locations is not None:
                self.fleet_data["vehicle_break_locations"] = pd.Series(
                    vehicle_break_locations, dtype=np.int32
                )  # noqa
            if vehicle_types is not None:
                self.fleet_data["vehicle_types"] = pd.Series(
                    vehicle_types, dtype=np.uint8
                )
            if min_vehicles is not None:
                self.fleet_data["min_vehicles"] = min_vehicles

        return is_valid

    def set_task_data(
        self,
        task_ids,
        task_locations,
        demand,
        pickup_and_delivery_pairs,
        task_time_windows,
        task_service_times,
        prizes,
        order_vehicle_match,
    ):
        if not self.is_route_detail_set:
            return (
                False,
                "Cost matrix/Waypoint graph needs to be set before setting task data",  # noqa
            )

        task_ids = get_none_for_empty_list(task_ids)
        task_locations = pd.Series(
            task_locations, name="task_id", dtype=np.int32
        )

        demand = get_none_for_empty_list(demand)
        pickup_and_delivery_pairs = get_none_for_empty_list(
            pickup_and_delivery_pairs
        )
        task_time_windows = get_none_for_empty_list(task_time_windows)
        task_service_times = get_none_for_empty_list(task_service_times)
        prizes = get_none_for_empty_list(prizes)
        order_vehicle_match = get_none_for_empty_list(order_vehicle_match)

        is_valid = validate_task_data(
            task_ids,
            task_locations,
            demand,
            pickup_and_delivery_pairs,
            task_time_windows,
            task_service_times,
            prizes,
            order_vehicle_match,
            updating=False,
            comparison_locations=None,
        )

        if is_valid[0]:
            if task_ids is not None:
                self.task_data["task_ids"] = pd.Series(task_ids)
            else:
                self.task_data["task_ids"] = pd.Series(
                    range(len(task_locations))
                )
            self.task_data["task_locations"] = task_locations

            if demand:
                self.task_data["demand"] = pd.DataFrame(
                    demand, dtype=np.int32
                ).T
            if pickup_and_delivery_pairs:
                self.task_data["pickup_and_delivery_pairs"] = pd.DataFrame(
                    pickup_and_delivery_pairs,
                    columns=["pickup_ind", "delivery_ind"],
                    dtype=np.int32,
                )
            if task_time_windows:
                self.task_data["task_time_windows"] = pd.DataFrame(
                    task_time_windows,
                    columns=["earliest", "latest"],
                    dtype=np.int32,
                )
            if task_service_times:
                self.task_data["service_times"] = task_service_times
            if prizes is not None:
                self.task_data["prizes"] = pd.Series(prizes, dtype=np.float32)
            if order_vehicle_match is not None:
                self.task_data["order_vehicle_match"] = [
                    {
                        "order_id": data.order_id,
                        "vehicle_ids": data.vehicle_ids,
                    }
                    for data in order_vehicle_match
                ]

        return is_valid

    def set_solver_config(
        self,
        time_limit,
        objectives,
        config_file,
        verbose_mode,
        error_logging,
    ):
        is_valid = validate_solver_config(
            time_limit,
            objectives,
            config_file,
            verbose_mode,
            error_logging,
            updating=False,
            comparison_time_limit=None,
        )

        if is_valid[0]:
            self.solver_config["time_limit"] = time_limit
            if objectives is not None:
                cuopt_objectives, objective_weights = get_objectives_as_lists(
                    objectives
                )  # noqa
                if len(cuopt_objectives) > 0:
                    self.solver_config["objectives"] = pd.Series(
                        cuopt_objectives, dtype=np.int32
                    )  # noqa
                    self.solver_config["objective_weights"] = pd.Series(
                        objective_weights, dtype=np.float32
                    )  # noqa
            if config_file is not None:
                self.solver_config["config_file"] = config_file
            if verbose_mode is not None:
                self.solver_config["verbose_mode"] = verbose_mode
            if error_logging is not None:
                self.solver_config["error_logging"] = error_logging

        return is_valid
