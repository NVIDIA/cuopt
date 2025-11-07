# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os

import numpy as np
from microservice_perf import (
    convert_cost_matrix_to_csr,
    generate_random_cost_matrix,
)


def generate_service_data(p_size):
    np_cost_matrix = generate_random_cost_matrix(p_size, 100)
    cost_matrix = {"cost_matrix": {0: np_cost_matrix.tolist()}}

    (
        generated_offsets,
        generated_edges,
        generated_weights,
    ) = convert_cost_matrix_to_csr(np_cost_matrix)

    waypoint_graph = {
        "waypoint_graph": {
            0: {
                "edges": generated_edges,
                "offsets": generated_offsets,
                "weights": generated_weights,
            }
        }
    }

    # generate minimal task data
    task_locations = np.random.choice(
        p_size, p_size // 5, replace=False
    ).tolist()
    demand = [[1] * len(task_locations)]
    task_data = {
        "task_locations": task_locations,
        "demand": demand,
    }
    assert len(task_locations) == len(demand[0])

    # generate minimal fleet data
    vehicle_locations = [[0, 0]] * (p_size // 10)
    capacities = [
        [1 + (len(demand[0]) // (p_size // 10))] * len(vehicle_locations)
    ]
    fleet_data = {
        "vehicle_locations": vehicle_locations,
        "capacities": capacities,
    }
    assert len(vehicle_locations) == len(capacities[0])
    assert np.sum(demand[0]) <= np.sum(capacities[0])

    # generate reasonable config
    config_data = {
        "time_limit": p_size / 100.0,
    }

    cost_matrix_data = {
        "cost_matrix_data": cost_matrix,
        "cost_waypoint_graph_data": [],
        "travel_time_matrix_data": [],
        "travel_time_waypoint_graph_data": [],
        "fleet_data": fleet_data,
        "task_data": task_data,
        "solver_config": config_data,
    }

    waypoint_graph_data = {
        "cost_matrix_data": [],
        "cost_waypoint_graph_data": waypoint_graph,
        "travel_time_matrix_data": [],
        "travel_time_waypoint_graph_data": [],
        "fleet_data": fleet_data,
        "task_data": task_data,
        "solver_config": config_data,
    }

    return (cost_matrix_data, waypoint_graph_data)


def create_and_save_dataset(p_size):
    matrix_dataset, wpg_dataset = generate_service_data(p_size)

    # the r in this file designates random problems
    file_name_matrix = f"service_data_{p_size}r_matrix.json"
    file_name_wpg = f"service_data_{p_size}r_wpg.json"

    if not os.path.exists("random_service_dataset"):
        os.makedirs("random_service_dataset")

    with open(
        os.path.join("random_service_dataset", file_name_matrix), "w"
    ) as f:
        json.dump(matrix_dataset, f, indent=4)

    with open(os.path.join("random_service_dataset", file_name_wpg), "w") as f:
        json.dump(wpg_dataset, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create random problem instances formatted for the cuOpt service"
        )
    )

    parser.add_argument(
        "--problem_sizes",
        "-ps",
        metavar="N",
        type=int,
        nargs="+",
        default=[10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        help="List of integer sizes for problem set generation",
    )

    args = parser.parse_args()

    for p_size in args.problem_sizes:
        create_and_save_dataset(p_size)


if __name__ == "__main__":
    main()
