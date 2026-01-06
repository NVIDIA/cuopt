# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import numpy as np

from cuopt import routing


def create_tsp_cost_matrix(n_locations):
    """Creates a simple symmetric cost matrix for TSP."""
    cost_matrix = np.zeros((n_locations, n_locations), dtype=np.float32)
    for i in range(n_locations):
        for j in range(n_locations):
            cost_matrix[i, j] = abs(i - j)
    return cudf.DataFrame(cost_matrix)


def test_batch_solve_varying_sizes():
    """Test batch solving TSPs of varying sizes."""
    tsp_sizes = [
        5,
        8,
        10,
        6,
        7,
        9,
        12,
        15,
        11,
        4,
        13,
        14,
        8,
        6,
        10,
        9,
        7,
        11,
        5,
        12,
    ]

    # Create data models for each TSP
    data_models = []
    for n_locations in tsp_sizes:
        cost_matrix = create_tsp_cost_matrix(n_locations)
        dm = routing.DataModel(n_locations, 1)
        dm.add_cost_matrix(cost_matrix)
        data_models.append(dm)

    # Configure solver settings
    settings = routing.SolverSettings()
    settings.set_time_limit(5.0)

    # Call batch solve
    solutions = routing.BatchSolve(data_models, settings)

    # Verify results
    assert len(solutions) == len(tsp_sizes)
    for i, solution in enumerate(solutions):
        assert solution.get_status() == 0, (
            f"TSP {i} (size {tsp_sizes[i]}) failed"
        )
        assert solution.get_vehicle_count() == 1, (
            f"TSP {i} (size {tsp_sizes[i]}) used multiple vehicles"
        )
