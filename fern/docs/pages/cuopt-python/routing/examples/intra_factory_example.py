# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Intra-factory transport example.
#
# Scenario: a small factory floor where a fleet of autonomous mobile robots
# (AMRs) pick up parts at processing stations and deliver them to other
# stations (or off the floor). cuOpt plans the routes for each robot such
# that every order is served within its pickup/delivery time windows and
# no robot exceeds its carrying capacity.
#
# This is a Capacitated Pickup-and-Delivery Problem with Time Windows
# (PDPTW) solved on a weighted waypoint graph.

import cudf
import numpy as np

from cuopt import distance_engine, routing

# --- Factory layout --------------------------------------------------------
# Waypoints in the factory, referenced by integer id:
#     0 = AMR depot (robots start/return here)
#     4 = Station A
#     5 = Station B
#     6 = Station C
# Other waypoints (1, 2, 3, 7, 8, 9) are intermediate nodes that robots
# travel through but never stop at.
DEPOT = 0
STATION_A = 4
STATION_B = 5
STATION_C = 6
TARGET_LOCATIONS = np.array([DEPOT, STATION_A, STATION_B, STATION_C])

# Factory operating hours (in time units).
FACTORY_OPEN = 0
FACTORY_CLOSE = 100

# Weighted waypoint graph of the factory floor in Compressed Sparse Row
# (CSR) form: for node i, GRAPH_OFFSETS[i]:GRAPH_OFFSETS[i+1] indexes into
# GRAPH_EDGES (destination nodes) and GRAPH_WEIGHTS (edge costs). See the
# `cuopt.distance_engine.WaypointMatrix` API reference in the cuOpt User
# Guide for input requirements.
GRAPH_OFFSETS = np.array([0, 1, 3, 7, 9, 11, 13, 15, 17, 20, 22])
GRAPH_EDGES = np.array(
    [2, 2, 4, 0, 1, 3, 5, 2, 6, 1, 7, 2, 8, 3, 9, 4, 8, 5, 7, 9, 6, 8]
)
GRAPH_WEIGHTS = np.array(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2]
)


def build_cost_matrix():
    """Compute an all-pairs cost matrix over the target locations."""
    graph = distance_engine.WaypointMatrix(
        GRAPH_OFFSETS, GRAPH_EDGES, GRAPH_WEIGHTS
    )
    cost_matrix = graph.compute_cost_matrix(TARGET_LOCATIONS)
    # waypoint id -> cost-matrix index (e.g. Station A [waypoint 4] -> index 1)
    wp_to_idx = {int(wp): i for i, wp in enumerate(TARGET_LOCATIONS)}
    return graph, cost_matrix, wp_to_idx


def build_orders():
    """Six transport orders. Each row is one pickup/delivery pair."""
    return cudf.DataFrame(
        {
            "pickup_location": [
                STATION_A,
                STATION_B,
                STATION_C,
                STATION_C,
                STATION_B,
                STATION_A,
            ],
            "delivery_location": [
                STATION_B,
                STATION_C,
                DEPOT,
                STATION_B,
                STATION_A,
                DEPOT,
            ],
            "demand": [1, 1, 1, 1, 1, 1],
            "earliest_pickup": [0, 0, 0, 0, 0, 0],
            "latest_pickup": [10, 20, 30, 10, 20, 30],
            "earliest_delivery": [0, 0, 0, 0, 0, 0],
            "latest_delivery": [45, 45, 45, 45, 45, 45],
            "pickup_service_time": [2, 2, 2, 2, 2, 2],
            "delivery_service_time": [2, 2, 2, 2, 2, 2],
        }
    )


def build_fleet():
    """Two AMRs, each able to carry two parts at once."""
    return cudf.DataFrame({"robot_id": [0, 1], "capacity": [2, 2]}).set_index(
        "robot_id"
    )


def build_data_model(cost_matrix, orders, fleet, wp_to_idx):
    """Assemble the cuOpt routing DataModel from the problem inputs."""
    n_locations = len(cost_matrix)
    n_vehicles = len(fleet)
    # Each order contributes two stops: one pickup and one delivery.
    n_orders = len(orders) * 2

    data_model = routing.DataModel(n_locations, n_vehicles, n_orders)
    data_model.add_cost_matrix(cost_matrix)

    # Capacity: pickups add load, deliveries remove it.
    demand = cudf.concat(
        [orders["demand"], -orders["demand"]], ignore_index=True
    )
    data_model.add_capacity_dimension("parts", demand, fleet["capacity"])

    # Order locations are expressed as cost-matrix indices, not waypoint ids.
    pickup_idx = orders["pickup_location"].map(wp_to_idx)
    delivery_idx = orders["delivery_location"].map(wp_to_idx)
    data_model.set_order_locations(
        cudf.concat([pickup_idx, delivery_idx], ignore_index=True)
    )

    # Pickup at row i must be served before its delivery at row i + n.
    n = len(orders)
    data_model.set_pickup_delivery_pairs(
        cudf.Series(range(n)), cudf.Series(range(n, 2 * n))
    )

    # Time windows.
    data_model.set_order_time_windows(
        cudf.concat(
            [orders["earliest_pickup"], orders["earliest_delivery"]],
            ignore_index=True,
        ),
        cudf.concat(
            [orders["latest_pickup"], orders["latest_delivery"]],
            ignore_index=True,
        ),
    )
    data_model.set_order_service_times(
        cudf.concat(
            [orders["pickup_service_time"], orders["delivery_service_time"]],
            ignore_index=True,
        )
    )
    data_model.set_vehicle_time_windows(
        cudf.Series([FACTORY_OPEN] * n_vehicles),
        cudf.Series([FACTORY_CLOSE] * n_vehicles),
    )
    return data_model


def print_schedule(solution, graph, wp_to_idx):
    """Print a per-robot, human-readable schedule of stops and waypoint paths."""
    idx_to_wp = {i: wp for wp, i in wp_to_idx.items()}
    route = solution.get_route()

    print(f"\nTotal route cost: {solution.get_total_objective():g}")
    print(f"Robots used:      {solution.get_vehicle_count()}\n")

    for robot_id in sorted(route["truck_id"].unique().to_arrow().to_pylist()):
        stops_gpu = route[route["truck_id"] == robot_id]
        stops = stops_gpu.to_pandas()

        print(f"Robot {robot_id}:")
        for _, s in stops.iterrows():
            print(
                f"  t={s['arrival_stamp']:>5g}  "
                f"waypoint {idx_to_wp[s['location']]:<2}  {s['type']}"
            )

        # compute_waypoint_sequence mutates its input, so hand it a fresh copy.
        wp_path = graph.compute_waypoint_sequence(
            TARGET_LOCATIONS, stops_gpu.copy()
        )
        path_str = " -> ".join(
            str(w) for w in wp_path["waypoint_sequence"].to_arrow().to_pylist()
        )
        print(f"  path: {path_str}\n")


def main():
    graph, cost_matrix, wp_to_idx = build_cost_matrix()
    orders = build_orders()
    fleet = build_fleet()

    print("Target locations (waypoint -> cost-matrix index):", wp_to_idx)
    print("\nCost matrix between target locations:")
    print(cost_matrix)
    print(f"\n{len(orders)} transport orders, {len(fleet)} AMRs.")

    data_model = build_data_model(cost_matrix, orders, fleet, wp_to_idx)

    settings = routing.SolverSettings()
    settings.set_time_limit(5)

    solution = routing.Solve(data_model, settings)
    if solution.get_status() != 0:
        print(
            f"cuOpt failed to find a solution (status={solution.get_status()})"
        )
        return

    print_schedule(solution, graph, wp_to_idx)


if __name__ == "__main__":
    main()
