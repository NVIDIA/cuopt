# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf
from cuopt import routing

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
from cuopt_server.utils.routing.validation_waypoint_graph import (
    validate_waypoint_graph,
)


def get_none_for_empty_list(data):
    return data if data is not None and len(data) > 0 else None


def get_objectives_as_lists(objectives):
    cuopt_objectives = []
    objective_weights = []

    if objectives.cost is not None:
        cuopt_objectives.append(routing.Objective.COST)
        objective_weights.append(objectives.cost)
    if objectives.travel_time is not None:
        cuopt_objectives.append(routing.Objective.TRAVEL_TIME)
        objective_weights.append(objectives.travel_time)
    if objectives.variance_route_size is not None:
        cuopt_objectives.append(routing.Objective.VARIANCE_ROUTE_SIZE)
        objective_weights.append(objectives.variance_route_size)
    if objectives.variance_route_service_time is not None:
        cuopt_objectives.append(routing.Objective.VARIANCE_ROUTE_SERVICE_TIME)
        objective_weights.append(objectives.variance_route_service_time)
    if objectives.prize is not None:
        cuopt_objectives.append(routing.Objective.PRIZE)
        objective_weights.append(objectives.prize)
    if objectives.vehicle_fixed_cost is not None:
        cuopt_objectives.append(routing.Objective.VEHICLE_FIXED_COST)
        objective_weights.append(objectives.vehicle_fixed_cost)

    return cuopt_objectives, objective_weights


objective_names = {
    routing.Objective.COST: "cost",
    routing.Objective.TRAVEL_TIME: "travel_time",
    routing.Objective.VARIANCE_ROUTE_SIZE: "variance_route_size",
    routing.Objective.VARIANCE_ROUTE_SERVICE_TIME: "variance_route_service_time",  # noqa
    routing.Objective.PRIZE: "prize",
    routing.Objective.VEHICLE_FIXED_COST: "vehicle_fixed_cost",
}


class OptimizationDataModel:
    def __init__(self) -> None:
        self.waypoint_graph = {}

        self.travel_time_waypoint_graph = {}

        self.locations = []

        self.is_route_detail_set = False

        self.cost_matrix = {}
        self.travel_time_matrix = {}

        self.fleet_data = self.reset_fleet_data()

        self.task_data = self.reset_task_data()

        self.initial_solution = []

        self.solver_config = self.reset_solver_config()

    # CLASS UTILITY FUNCTIONS
    def reset_fleet_data(self):
        return {
            "vehicle_ids": None,
            "vehicle_locations": None,
            "capacities": None,
            "vehicle_time_windows": None,
            "vehicle_break_time_windows": None,
            "vehicle_break_durations": None,
            "vehicle_break_locations": None,
            "vehicle_breaks": None,
            "vehicle_types": None,
            "vehicle_order_match": None,
            "skip_first_trips": None,
            "drop_return_trips": None,
            "min_vehicles": None,
            "vehicle_max_costs": None,
            "vehicle_max_times": None,
            "vehicle_fixed_costs": None,
        }

    def reset_task_data(self):
        return {
            "task_locations": None,
            "task_ids": None,
            "demand": None,
            "pickup_and_delivery_pairs": None,
            "task_time_windows": None,
            "service_times": None,
            "prizes": None,
            "order_vehicle_match": None,
        }

    def reset_solver_config(self):
        return {
            "time_limit": None,
            "objectives": None,
            "objective_weights": None,
            "config_file": None,
            "verbose_mode": None,
            "error_logging": None,
        }

    def get_cost_waypoint_graph(self):
        cost_waypoint_graph_data = {}
        for v_type, graph in self.waypoint_graph.items():
            cost_waypoint_graph_data[v_type] = {
                key: (value.tolist() if value is not None else None)
                for key, value in graph.items()
            }
        return cost_waypoint_graph_data

    def get_travel_time_waypoint_graph(self):
        cost_waypoint_graph_data = {}
        for v_type, graph in self.travel_time_waypoint_graph.items():
            cost_waypoint_graph_data[v_type] = {
                key: (value.tolist() if value is not None else None)
                for key, value in graph.items()
            }
        return cost_waypoint_graph_data

    def get_cost_matrix(self):
        return {
            key: value.to_numpy().tolist()
            for key, value in self.cost_matrix.items()
        }

    def get_travel_time_matrix(self):
        return {
            key: value.to_numpy().tolist()
            for key, value in self.travel_time_matrix.items()
        }

    def get_fleet_data(self):
        return {
            "vehicle_ids": self.fleet_data["vehicle_ids"]
            .to_arrow()
            .to_pylist()
            if self.fleet_data["vehicle_ids"] is not None
            else None,
            "vehicle_locations": self.fleet_data["vehicle_locations"]
            .to_numpy()
            .tolist()
            if self.fleet_data["vehicle_locations"] is not None
            else None,
            "capacities": self.fleet_data["capacities"].T.to_numpy().tolist()
            if self.fleet_data["capacities"] is not None
            else None,
            "vehicle_max_costs": self.fleet_data["vehicle_max_costs"]
            .to_arrow()
            .to_pylist()
            if self.fleet_data["vehicle_max_costs"] is not None
            else None,
            "vehicle_max_times": self.fleet_data["vehicle_max_times"]
            .to_arrow()
            .to_pylist()
            if self.fleet_data["vehicle_max_times"] is not None
            else None,
            "vehicle_fixed_costs": self.fleet_data["vehicle_fixed_costs"]
            .to_arrow()
            .to_pylist()
            if self.fleet_data["vehicle_fixed_costs"] is not None
            else None,
            "vehicle_time_windows": self.fleet_data["vehicle_time_windows"]
            .to_numpy()
            .tolist()
            if self.fleet_data["vehicle_time_windows"] is not None
            else None,
            "vehicle_break_time_windows": [
                data.to_numpy().tolist()
                for data in self.fleet_data["vehicle_break_time_windows"]  # noqa
            ]
            if self.fleet_data["vehicle_break_time_windows"] is not None
            else None,
            "vehicle_break_durations": [
                data.to_arrow().to_pylist()
                for data in self.fleet_data["vehicle_break_durations"]  # noqa
            ]
            if self.fleet_data["vehicle_break_durations"] is not None
            else None,
            "vehicle_break_locations": self.fleet_data[
                "vehicle_break_locations"
            ]  # noqa
            .to_arrow()
            .to_pylist()
            if self.fleet_data["vehicle_break_locations"] is not None
            else None,
            "vehicle_breaks": self.fleet_data["vehicle_breaks"],
            "vehicle_order_match": self.fleet_data["vehicle_order_match"],
            "skip_first_trips": self.fleet_data["skip_first_trips"]
            .to_arrow()
            .to_pylist()
            if self.fleet_data["skip_first_trips"] is not None
            else None,
            "drop_return_trips": self.fleet_data["drop_return_trips"]
            .to_arrow()
            .to_pylist()
            if self.fleet_data["drop_return_trips"] is not None
            else None,
            "vehicle_types": self.fleet_data["vehicle_types"]
            .to_arrow()
            .to_pylist()
            if self.fleet_data["vehicle_types"] is not None
            else None,
            "min_vehicles": self.fleet_data["min_vehicles"],
        }

    def get_task_data(self):
        return {
            "task_ids": self.task_data["task_ids"].to_arrow().to_pylist()
            if self.task_data["task_ids"] is not None
            else None,
            "task_locations": self.task_data["task_locations"]
            .to_arrow()
            .to_pylist()
            if self.task_data["task_locations"] is not None
            else None,
            "demand": self.task_data["demand"].T.to_numpy().tolist()
            if self.task_data["demand"] is not None
            else None,
            "pickup_and_delivery_pairs": self.task_data[
                "pickup_and_delivery_pairs"
            ]
            .to_numpy()
            .tolist()
            if self.task_data["pickup_and_delivery_pairs"] is not None
            else None,
            "task_time_windows": self.task_data["task_time_windows"]
            .to_numpy()
            .tolist()
            if self.task_data["task_time_windows"] is not None
            else None,
            "service_times": self.task_data["service_times"],
            "prizes": self.task_data["prizes"].to_arrow().to_pylist()
            if self.task_data["prizes"] is not None
            else None,
            "order_vehicle_match": self.task_data["order_vehicle_match"],
        }

    def get_solver_config_data(self):
        solver_config = self.solver_config.copy()

        if solver_config["objectives"] is not None:
            objectives = solver_config["objectives"].to_arrow().to_pylist()
            objective_weights = (
                solver_config["objective_weights"].to_arrow().to_pylist()
            )  # noqa

            solver_config["objectives"] = {
                objective_names[objectives[i]]: objective_weights[i]
                for i in range(len(objectives))
            }
        # Weights have been already added to objectives if they were available
        del solver_config["objective_weights"]

        return solver_config

    def get_optimization_data(self):
        return {
            "cost_waypoint_graph": self.get_cost_waypoint_graph(),
            "travel_time_waypoint_graph": self.get_travel_time_waypoint_graph(),  # noqa
            "cost_matrix": self.get_cost_matrix(),
            "travel_time_matrix": self.get_travel_time_matrix(),
            "fleet_data": self.get_fleet_data(),
            "task_data": self.get_task_data(),
            "solver_config": self.get_solver_config_data(),
        }

    # WAYPOINT GRAPH DATA
    def set_cost_waypoint_graph(self, waypoint_graph):
        cost_waypoint_graph_data = {}
        for v_type, graph in waypoint_graph.items():
            weights = get_none_for_empty_list(graph.weights)
            cost_waypoint_graph_data[v_type] = {
                "edges": np.array(graph.edges),
                "offsets": np.array(graph.offsets),
                "weights": np.array(weights)
                if weights is not None
                else weights,
            }

        is_valid = validate_waypoint_graph(
            cost_waypoint_graph_data,
            is_travel_time=False,
            updating=False,
            comparison_waypoint_graph=None,
        )

        if is_valid[0]:
            self.is_route_detail_set = True
            self.waypoint_graph = cost_waypoint_graph_data

        return is_valid

    def update_cost_waypoint_graph_weights(self, waypoint_graph_weights):
        cost_waypoint_graph_data = {}
        for v_type, weights in waypoint_graph_weights.items():
            weights = get_none_for_empty_list(weights)
            cost_waypoint_graph_data[v_type] = {
                "edges": None,
                "offsets": None,
                "weights": np.array(weights)
                if weights is not None
                else weights,
            }

        is_valid = validate_waypoint_graph(
            cost_waypoint_graph_data,
            is_travel_time=False,
            updating=True,
            comparison_waypoint_graph=self.waypoint_graph,
        )

        if is_valid[0]:
            for v_type, weights in waypoint_graph_weights.items():
                self.waypoint_graph[v_type]["weights"] = (
                    waypoint_graph_weights[v_type]
                )  # noqa

        return is_valid

    def set_travel_time_waypoint_graph(self, travel_time_waypoint_graph):
        travel_time_waypoint_graph_data = {}
        for v_type, graph in travel_time_waypoint_graph.items():
            weights = get_none_for_empty_list(graph.weights)
            travel_time_waypoint_graph_data[v_type] = {
                "edges": np.array(graph.edges),
                "offsets": np.array(graph.offsets),
                "weights": np.array(weights)
                if weights is not None
                else weights,
            }

        is_valid = validate_waypoint_graph(
            travel_time_waypoint_graph_data,
            is_travel_time=True,
            updating=False,
            comparison_waypoint_graph=self.waypoint_graph,
        )

        if is_valid[0]:
            self.is_route_detail_set = True
            self.travel_time_waypoint_graph = travel_time_waypoint_graph_data

        return is_valid

    def update_travel_time_waypoint_graph_weights(
        self, travel_time_waypoint_graph_weights
    ):
        travel_time_waypoint_graph_data = {}
        for (
            v_type,
            travel_time_weights,
        ) in travel_time_waypoint_graph_weights.items():  # noqa
            travel_time_weights = get_none_for_empty_list(travel_time_weights)
            travel_time_waypoint_graph_data[v_type] = {
                "edges": None,
                "offsets": None,
                "weights": np.array(travel_time_weights)
                if travel_time_weights is not None
                else travel_time_weights,  # noqa
            }

        is_valid = validate_waypoint_graph(
            travel_time_waypoint_graph_data,
            is_travel_time=True,
            updating=True,
            comparison_waypoint_graph=self.travel_time_waypoint_graph,
        )

        if is_valid[0]:
            for v_type, weights in travel_time_waypoint_graph_weights.items():
                self.travel_time_waypoint_graph[v_type]["weights"] = (
                    travel_time_waypoint_graph_weights[v_type]
                )  # noqa

        return is_valid

    # COST MATRIX DATA
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
                self.cost_matrix[v_type] = cudf.DataFrame(np_cost_matrix)

        return is_valid

    def update_cost_matrix(self, cost_matrix):
        is_valid = validate_cost_matrix(
            cost_matrix,
            is_travel_time=False,
            updating=True,
            comparison_matrix=self.cost_matrix,
        )
        if is_valid[0]:
            self.is_route_detail_set = True
            for v_type, matrix in cost_matrix.items():
                np_cost_matrix = np.array(matrix, dtype=np.float32)
                self.cost_matrix[v_type] = cudf.DataFrame(np_cost_matrix)

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
                self.travel_time_matrix[v_type] = cudf.DataFrame(
                    np_travel_time_matrix
                )

        return is_valid

    def update_travel_time_matrix(self, travel_time_matrix):
        is_valid = validate_cost_matrix(
            travel_time_matrix,
            is_travel_time=True,
            updating=True,
            comparison_matrix=self.travel_time_matrix,
        )
        if is_valid[0]:
            for v_type, matrix in travel_time_matrix.items():
                np_travel_time_matrix = np.array(matrix, dtype=np.float32)
                self.travel_time_matrix[v_type] = cudf.DataFrame(
                    np_travel_time_matrix
                )

        return is_valid

    # FLEET DATA
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
        )

        if is_valid[0]:
            if vehicle_ids is not None:
                self.fleet_data["vehicle_ids"] = cudf.Series(vehicle_ids)
            else:
                self.fleet_data["vehicle_ids"] = cudf.Series(
                    range(len(vehicle_locations))
                )
            if vehicle_locations is not None:
                self.fleet_data["vehicle_locations"] = cudf.DataFrame(
                    vehicle_locations,
                    columns=["start_location", "end_location"],
                    dtype=np.int32,
                )
            if capacities:
                self.fleet_data["capacities"] = cudf.DataFrame(
                    capacities, dtype=np.int32
                ).T
            if vehicle_max_costs is not None:
                self.fleet_data["vehicle_max_costs"] = cudf.Series(
                    vehicle_max_costs, dtype=np.float32
                )
            if vehicle_max_times is not None:
                self.fleet_data["vehicle_max_times"] = cudf.Series(
                    vehicle_max_times, dtype=np.float32
                )
            if vehicle_fixed_costs is not None:
                self.fleet_data["vehicle_fixed_costs"] = cudf.Series(
                    vehicle_fixed_costs, dtype=np.float32
                )
            if vehicle_time_windows:
                self.fleet_data["vehicle_time_windows"] = cudf.DataFrame(
                    vehicle_time_windows,
                    columns=["earliest", "latest"],
                    dtype=np.int32,
                )
            if skip_first_trips:
                self.fleet_data["skip_first_trips"] = cudf.Series(
                    skip_first_trips, dtype=bool
                )
            if drop_return_trips:
                self.fleet_data["drop_return_trips"] = cudf.Series(
                    drop_return_trips, dtype=bool
                )
            if vehicle_break_time_windows and vehicle_break_durations:
                self.fleet_data["vehicle_break_time_windows"] = [
                    cudf.DataFrame(
                        val, columns=["earliest", "latest"], dtype=np.int32
                    )
                    for val in vehicle_break_time_windows
                ]

                self.fleet_data["vehicle_break_durations"] = [
                    cudf.Series(val, dtype=np.int32)
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
            if vehicle_order_match is not None:
                self.fleet_data["vehicle_order_match"] = [
                    {
                        "vehicle_id": data.vehicle_id,
                        "order_ids": data.order_ids,
                    }
                    for data in vehicle_order_match
                ]
            if vehicle_break_locations is not None:
                self.fleet_data["vehicle_break_locations"] = cudf.Series(
                    vehicle_break_locations, dtype=np.int32
                )  # noqa
            if vehicle_types is not None:
                self.fleet_data["vehicle_types"] = cudf.Series(
                    vehicle_types, dtype=np.uint8
                )
            if min_vehicles is not None:
                self.fleet_data["min_vehicles"] = min_vehicles

        return is_valid

    def update_fleet_data(
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
    ):
        if not self.is_route_detail_set:
            return (
                False,
                "Cost matrix/Waypoint graph needs to be set before updating fleet data",  # noqa
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
        vehicle_locations = get_none_for_empty_list(vehicle_locations)
        capacities = get_none_for_empty_list(capacities)
        vehicle_max_costs = get_none_for_empty_list(vehicle_max_costs)
        vehicle_max_times = get_none_for_empty_list(vehicle_max_times)
        vehicle_fixed_costs = get_none_for_empty_list(vehicle_fixed_costs)
        vehicle_time_windows = get_none_for_empty_list(vehicle_time_windows)
        skip_first_trips = get_none_for_empty_list(skip_first_trips)
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
            updating=True,
            comparison_locations=self.fleet_data["vehicle_locations"],
        )

        if is_valid[0]:
            if vehicle_ids is not None:
                self.fleet_data["vehicle_ids"] = cudf.Series(vehicle_ids)
            if vehicle_locations is not None:
                self.fleet_data["vehicle_locations"] = cudf.DataFrame(
                    vehicle_locations,
                    columns=["start_location", "end_location"],
                    dtype=np.int32,
                )
            if capacities:
                self.fleet_data["capacities"] = cudf.DataFrame(
                    capacities, dtype=np.int32
                ).T
            if vehicle_max_costs is not None:
                self.fleet_data["vehicle_max_costs"] = cudf.Series(
                    vehicle_max_costs, dtype=np.float32
                )
            if vehicle_max_times is not None:
                self.fleet_data["vehicle_max_times"] = cudf.Series(
                    vehicle_max_times, dtype=np.float32
                )
            if vehicle_fixed_costs is not None:
                self.fleet_data["vehicle_fixed_costs"] = cudf.Series(
                    vehicle_fixed_costs, dtype=np.float32
                )
            if vehicle_time_windows:
                self.fleet_data["vehicle_time_windows"] = cudf.DataFrame(
                    vehicle_time_windows,
                    columns=["earliest", "latest"],
                    dtype=np.int32,
                )
            if vehicle_break_time_windows and vehicle_break_durations:
                self.fleet_data["vehicle_break_time_windows"] = [
                    cudf.DataFrame(
                        val, columns=["earliest", "latest"], dtype=np.int32
                    )
                    for val in vehicle_break_time_windows
                ]

                self.fleet_data["vehicle_break_durations"] = [
                    cudf.Series(val, dtype=np.int32)
                    for val in vehicle_break_durations
                ]
            if vehicle_break_locations is not None:
                self.fleet_data["vehicle_break_locations"] = cudf.Series(
                    vehicle_break_locations, dtype=np.int32
                )  # noqa
            if vehicle_types is not None:
                self.fleet_data["vehicle_types"] = cudf.Series(
                    vehicle_types, dtype=np.uint8
                )
            if vehicle_order_match is not None:
                self.fleet_data["vehicle_order_match"] = [
                    {
                        "vehicle_id": data.vehicle_id,
                        "order_ids": data.order_ids,
                    }
                    for data in vehicle_order_match
                ]
            if skip_first_trips:
                self.fleet_data["skip_first_trips"] = cudf.Series(
                    skip_first_trips, dtype=bool
                )
            if drop_return_trips:
                self.fleet_data["drop_return_trips"] = cudf.Series(
                    drop_return_trips, dtype=bool
                )
            if min_vehicles is not None:
                self.fleet_data["min_vehicles"] = min_vehicles

        return is_valid

    # TASK DATA
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
        task_locations = cudf.Series(
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
                self.task_data["task_ids"] = cudf.Series(task_ids)
            else:
                self.task_data["task_ids"] = cudf.Series(
                    range(len(task_locations))
                )
            self.task_data["task_locations"] = task_locations

            if demand:
                self.task_data["demand"] = cudf.DataFrame(
                    demand, dtype=np.int32
                ).T
            if pickup_and_delivery_pairs:
                self.task_data["pickup_and_delivery_pairs"] = cudf.DataFrame(
                    pickup_and_delivery_pairs,
                    columns=["pickup_ind", "delivery_ind"],
                    dtype=np.int32,
                )
            if task_time_windows:
                self.task_data["task_time_windows"] = cudf.DataFrame(
                    task_time_windows,
                    columns=["earliest", "latest"],
                    dtype=np.int32,
                )
            if task_service_times:
                self.task_data["service_times"] = task_service_times
            if prizes is not None:
                self.task_data["prizes"] = cudf.Series(
                    prizes, dtype=np.float32
                )
            if order_vehicle_match is not None:
                self.task_data["order_vehicle_match"] = [
                    {
                        "order_id": data.order_id,
                        "vehicle_ids": data.vehicle_ids,
                    }
                    for data in order_vehicle_match
                ]

        return is_valid

    def update_task_data(
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
                "Cost matrix/Waypoint graph needs to be set before updating task data",  # noqa
            )

        task_ids = get_none_for_empty_list(task_ids)
        if task_locations is None or len(task_locations) == 0:
            task_locations = self.task_data["task_locations"]
        else:
            task_locations = cudf.Series(
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
            updating=True,
            comparison_locations=self.task_data["task_locations"],
        )

        if is_valid[0]:
            if task_ids is not None:
                self.task_data["task_ids"] = cudf.Series(task_ids)
            if task_locations is not None:
                self.task_data["task_locations"] = task_locations
            if demand:
                self.task_data["demand"] = cudf.DataFrame(
                    demand, dtype=np.int32
                ).T
            if pickup_and_delivery_pairs:
                self.task_data["pickup_and_delivery_pairs"] = cudf.DataFrame(
                    pickup_and_delivery_pairs,
                    columns=["pickup_ind", "delivery_ind"],
                    dtype=np.int32,
                )
            if task_time_windows:
                self.task_data["task_time_windows"] = cudf.DataFrame(
                    task_time_windows,
                    columns=["earliest", "latest"],
                    dtype=np.int32,
                )
            if task_service_times:
                if type(task_service_times) is dict:
                    self.task_data["service_times"] = {}
                    for v_id, service_times in task_service_times:
                        self.task_data["service_times"][v_id] = cudf.Series(
                            task_service_times, dtype=np.int32
                        )
                else:
                    self.task_data["service_times"] = cudf.Series(
                        task_service_times, dtype=np.int32
                    )
            if prizes is not None:
                self.task_data["prizes"] = cudf.Series(
                    prizes, dtype=np.float32
                )
            if order_vehicle_match is not None:
                self.task_data["order_vehicle_match"] = [
                    {
                        "order_id": data.order_id,
                        "vehicle_ids": data.vehicle_ids,
                    }
                    for data in order_vehicle_match
                ]

        return is_valid

    def set_initial_solution(self, initial_sols):
        is_valid = [True, "valid"]  # Check for Validation
        self.initial_solution = initial_sols
        return is_valid

    # SOLVER CONFIG DATA
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
                    self.solver_config["objectives"] = cudf.Series(
                        cuopt_objectives, dtype=np.int32
                    )  # noqa
                    self.solver_config["objective_weights"] = cudf.Series(
                        objective_weights, dtype=np.float32
                    )  # noqa
            if config_file is not None:
                self.solver_config["config_file"] = config_file
            if verbose_mode is not None:
                self.solver_config["verbose_mode"] = verbose_mode
            if error_logging is not None:
                self.solver_config["error_logging"] = error_logging

        return is_valid

    def update_solver_config(
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
            updating=True,
            comparison_time_limit=self.solver_config["time_limit"],
        )

        if is_valid[0]:
            if time_limit is not None:
                self.solver_config["time_limit"] = time_limit
            if objectives is not None:
                cuopt_objectives, objective_weights = get_objectives_as_lists(
                    objectives
                )  # noqa
                if len(cuopt_objectives) > 0:
                    self.solver_config["objectives"] = cudf.Series(
                        cuopt_objectives, dtype=np.int32
                    )  # noqa
                    self.solver_config["objective_weights"] = cudf.Series(
                        objective_weights, dtype=np.float32
                    )  # noqa
            if config_file is not None:
                self.solver_config["config_file"] = config_file
            if verbose_mode is not None:
                self.solver_config["verbose_mode"] = verbose_mode
            if error_logging is not None:
                self.solver_config["error_logging"] = error_logging

        return is_valid
