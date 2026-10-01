# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def check_valid(key, data):
    if key in data.keys():
        if data[key] is not None:
            return True
    return False


def generate_metadata(cuopt_data):
    try:
        metadata = {}

        fleet_data = dict(cuopt_data["fleet_data"])
        task_data = dict(cuopt_data["task_data"])

        metadata["problem_size"] = len(task_data["task_locations"])
        metadata["number_of_available_vehicles"] = len(
            fleet_data["vehicle_locations"]
        )

        metadata["problem_type"] = (
            "PDP"
            if check_valid("pickup_and_delivery_pairs", task_data)
            else "VRP"
        )

        metadata["vehicle_time_windows"] = check_valid(
            "vehicle_time_windows", fleet_data
        )

        metadata["task_time_windows"] = check_valid(
            "task_time_windows", task_data
        )

        if check_valid("service_times", task_data):
            metadata["service_time_dimensions"] = (
                len(task_data["service_times"].keys())
                if isinstance(task_data["service_times"], dict)
                else 1
            )

        if check_valid("capacities", fleet_data):
            metadata["capacity_dimension"] = len(fleet_data["capacities"])
        if check_valid("demand", task_data):
            metadata["demand_dimension"] = len(task_data["demand"])

        metadata["order_vehicle_match"] = check_valid(
            "order_vehicle_match", task_data
        )
        metadata["vehicle_order_match"] = check_valid(
            "vehicle_order_match", fleet_data
        )
        metadata["min_vehicles"] = check_valid("min_vehicles", fleet_data)
        metadata["skip_first_trips"] = check_valid(
            "skip_first_trips", fleet_data
        )

        metadata["drop_return_trips"] = check_valid(
            "drop_return_trips", fleet_data
        )

        metadata["route_constraints"] = []
        for constraint in [
            "vehicle_max_costs",
            "vehicle_max_times",
        ]:
            if check_valid(constraint, fleet_data):
                metadata["route_constraints"].append(constraint)
        vehicle_types = 1
        if check_valid("cost_waypoint_graph_data", cuopt_data):
            cost_waypoint_graph_data = dict(
                cuopt_data["cost_waypoint_graph_data"]
            )
            if check_valid("waypoint_graph", cost_waypoint_graph_data):
                waypoint_graph = dict(
                    cost_waypoint_graph_data["waypoint_graph"]
                )
                for key in waypoint_graph.keys():
                    metadata["num_locations"] = (
                        len(waypoint_graph[key].offsets) - 1
                    )
                    break
                vehicle_types = len(cost_waypoint_graph_data["waypoint_graph"])
        if check_valid("cost_matrix_data", cuopt_data):
            cost_matrix_data = dict(cuopt_data["cost_matrix_data"])
            if check_valid("cost_matrix", cost_matrix_data):
                cost_matrices = dict(cost_matrix_data["cost_matrix"])
                for key in cost_matrices:
                    metadata["num_locations"] = len(cost_matrices[key])
                    break
                vehicle_types = len(cost_matrix_data["cost_matrix"])
        metadata["vehicle_types"] = vehicle_types

        if check_valid("solver_config", cuopt_data):
            solver_config = dict(cuopt_data["solver_config"])
            metadata["objective_functions"] = check_valid(
                "objectives", solver_config
            )
            if check_valid("time_limit", solver_config):
                metadata["time_limit"] = solver_config["time_limit"]

        if check_valid("vehicle_break_time_windows", fleet_data):
            metadata["break_dimensions"] = len(
                fleet_data["vehicle_break_time_windows"]
            )

    except Exception as e:
        metadata = {"error_msg": "Error logging metadata: " + str(e)}

    return {"msg": "metadata", "data": metadata}
