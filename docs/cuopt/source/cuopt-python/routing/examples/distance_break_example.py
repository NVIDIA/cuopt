# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Distance break example: a vehicle must take a mandatory break at a
# break location before its cumulative route distance exceeds 75 km.

import cudf
import numpy as np

from cuopt import routing

# 2-D coordinates (km): depot, 2 customers, 1 break location.
# Geometry forces the route depot -> customer 1 -> break -> customer 2 -> depot:
# customer 2 is 63 km from the depot, so any route that does not break
# between the two customers either lands the break past its 75 km window
# or backtracks far enough to be strictly more expensive.
COORDS = np.array(
    [
        [0.0, 0.0],  # 0  depot
        [40.0, 0.0],  # 1  customer 1
        [60.0, 20.0],  # 2  customer 2
        [60.0, 0.0],  # 3  break location
    ],
    dtype=np.float32,
)


def build_cost_matrix(coords):
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    return cudf.DataFrame(np.linalg.norm(diff, axis=-1).astype(np.float32))


def main():
    data_model = routing.DataModel(n_locations=4, n_fleet=1, n_orders=2)
    data_model.add_cost_matrix(build_cost_matrix(COORDS))
    data_model.set_order_locations(cudf.Series([1, 2], dtype=np.int32))

    data_model.add_distance_break(
        vehicle_ids=0,
        max_range=75.0,
        duration=10,
        locations=cudf.Series([3], dtype=np.int32),
    )

    settings = routing.SolverSettings()
    settings.set_time_limit(5.0)

    solution = routing.Solve(data_model, settings)
    if solution.get_status() != 0:
        print(f"No solution found (status={solution.get_status()})")
        return

    labels = {0: "depot", 1: "customer 1", 2: "customer 2", 3: "break"}
    route = solution.get_route().to_pandas()
    print(f"Total route cost: {solution.get_total_objective():.1f} km\n")
    for _, stop in route.iterrows():
        print(f"  {stop['type']:<10}  {labels[stop['location']]}")


if __name__ == "__main__":
    main()
