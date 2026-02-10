# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TSP batch solve example: solve multiple TSP instances in one call for higher throughput.
# Use this when you have many similar routing problems (e.g., many small TSPs) to solve.

import cudf
import numpy as np

from cuopt import routing


def create_tsp_cost_matrix(n_locations):
    """Create a simple symmetric cost matrix for a TSP of size n_locations."""
    cost_matrix = np.zeros((n_locations, n_locations), dtype=np.float32)
    for i in range(n_locations):
        for j in range(n_locations):
            cost_matrix[i, j] = abs(i - j)
    return cudf.DataFrame(cost_matrix)


def main():
    # Define multiple TSP sizes to solve in one batch
    tsp_sizes = [5, 8, 10, 6, 7, 9]

    # Build one DataModel per TSP
    data_models = []
    for n_locations in tsp_sizes:
        cost_matrix = create_tsp_cost_matrix(n_locations)
        dm = routing.DataModel(n_locations, 1)  # n_locations, 1 vehicle (TSP)
        dm.add_cost_matrix(cost_matrix)
        data_models.append(dm)

    # Shared solver settings for the batch
    settings = routing.SolverSettings()
    settings.set_time_limit(5.0)

    # Solve all TSPs in batch (parallel execution)
    solutions = routing.BatchSolve(data_models, settings)

    # Inspect results
    print(f"Solved {len(solutions)} TSPs in batch.")
    for i, (size, solution) in enumerate(zip(tsp_sizes, solutions)):
        status = solution.get_status()
        status_str = (
            "SUCCESS" if status == routing.SolutionStatus.SUCCESS else status
        )
        vehicle_count = solution.get_vehicle_count()
        print(
            f"  TSP {i} (size {size}): status={status_str}, vehicles={vehicle_count}"
        )


if __name__ == "__main__":
    main()
