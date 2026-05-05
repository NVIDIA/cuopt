# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EV break example: distance-triggered charging stops for electric vehicles.
# Use this when vehicles must recharge at fixed distance intervals along a route.

import cudf
import numpy as np

from cuopt import routing

# 2-D coordinates (km): depot, customers 1-5, charging stations A/B/C.
COORDS = np.array(
    [
        [0.0, 0.0],  # 0  depot
        [90.0, 8.0],  # 1  customer 1  (between chargers A and B)
        [155.0, 8.0],  # 2  customer 2  (between chargers B and C)
        [105.0, -8.0],  # 3  customer 3  (between chargers A and B)
        [170.0, -8.0],  # 4  customer 4  (between chargers B and C)
        [220.0, 0.0],  # 5  customer 5  (past charger C)
        [60.0, 0.0],  # 6  charging station A  (60 km from depot)
        [135.0, 0.0],  # 7  charging station B  (135 km from depot)
        [210.0, 0.0],  # 8  charging station C  (210 km from depot)
    ],
    dtype=np.float32,
)

MIN_RANGE = 0.0  # minimum km before a charge stop is allowed
MAX_RANGE = 75.0  # maximum km between charges
CHARGE_DURATION = (
    10  # service time at a charging station (same unit as service times)
)
N_CYCLES = 5  # five charge windows per route: [0,75), [75,150), ..., [300,375)


def build_cost_matrix(coords):
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    return cudf.DataFrame(np.linalg.norm(diff, axis=-1).astype(np.float32))


def main():
    cost_matrix = build_cost_matrix(COORDS)

    n_locations = len(COORDS)
    n_vehicles = 2
    n_orders = 5  # customers 1–5

    data_model = routing.DataModel(n_locations, n_vehicles, n_orders)
    data_model.add_cost_matrix(cost_matrix)

    # Customers are at cost-matrix indices 1–5.
    data_model.set_order_locations(cudf.Series([1, 2, 3, 4, 5]))

    # All three charging stations are eligible for any charging stop.
    charging_stations = cudf.Series([6, 7, 8], dtype=np.int32)

    # Apply the same EV break schedule to both vehicles.
    data_model.add_ev_break(
        vehicle_ids=[0, 1],
        max_range=MAX_RANGE,
        charge_duration=CHARGE_DURATION,
        charging_stations=charging_stations,
        min_range=MIN_RANGE,
        n_cycles=N_CYCLES,
    )

    settings = routing.SolverSettings()
    settings.set_time_limit(5.0)

    solution = routing.Solve(data_model, settings)
    if solution.get_status() != 0:
        print(f"No solution found (status={solution.get_status()})")
        return

    route = solution.get_route()
    print(f"Total route cost: {solution.get_total_objective():.1f} km")
    print(f"Vehicles used:    {solution.get_vehicle_count()}\n")

    location_labels = {
        0: "depot",
        1: "customer 1",
        2: "customer 2",
        3: "customer 3",
        4: "customer 4",
        5: "customer 5",
        6: "charger A",
        7: "charger B",
        8: "charger C",
    }

    for vid in sorted(route["truck_id"].unique().to_arrow().to_pylist()):
        stops = route[route["truck_id"] == vid].to_pandas()
        print(f"Vehicle {vid}:")
        for _, stop in stops.iterrows():
            label = location_labels.get(
                stop["location"], f"loc {stop['location']}"
            )
            print(f"  {stop['type']:<10}  {label}")
        print()


if __name__ == "__main__":
    main()
