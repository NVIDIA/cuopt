# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    tsp_sizes = [5, 8, 10, 6, 7, 9]

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
    solutions, solve_time = routing.BatchSolve(data_models, settings)

    # Verify results
    assert len(solutions) == len(tsp_sizes)
    for i, solution in enumerate(solutions):
        assert solution.get_status() == routing.SolutionStatus.SUCCESS, (
            f"TSP {i} (size {tsp_sizes[i]}) failed"
        )
        assert solution.get_vehicle_count() == 1, (
            f"TSP {i} (size {tsp_sizes[i]}) used multiple vehicles"
        )

    # Verify solve time is reasonable
    assert solve_time > 0.0, "Solve time should be positive"


def test_batch_solve_same_size():
    """Test batch solving multiple TSPs of the same size."""
    n_problems = 10
    n_locations = 6

    # Create data models
    data_models = []
    for _ in range(n_problems):
        cost_matrix = create_tsp_cost_matrix(n_locations)
        dm = routing.DataModel(n_locations, 1)
        dm.add_cost_matrix(cost_matrix)
        data_models.append(dm)

    # Configure solver settings
    settings = routing.SolverSettings()
    settings.set_time_limit(2.0)

    # Call batch solve
    solutions, solve_time = routing.BatchSolve(data_models, settings)

    # Verify all solutions succeeded
    assert len(solutions) == n_problems
    for i, solution in enumerate(solutions):
        assert solution.get_status() == routing.SolutionStatus.SUCCESS, (
            f"TSP {i} failed"
        )


def test_batch_solve_single_problem():
    """Test batch solve with a single problem."""
    n_locations = 5

    cost_matrix = create_tsp_cost_matrix(n_locations)
    dm = routing.DataModel(n_locations, 1)
    dm.add_cost_matrix(cost_matrix)

    settings = routing.SolverSettings()
    settings.set_time_limit(2.0)

    solutions, solve_time = routing.BatchSolve([dm], settings)

    assert len(solutions) == 1
    assert solutions[0].get_status() == routing.SolutionStatus.SUCCESS


def test_batch_solve_default_settings():
    """Test batch solve with default solver settings."""
    tsp_sizes = [5, 6, 7]

    data_models = []
    for n_locations in tsp_sizes:
        cost_matrix = create_tsp_cost_matrix(n_locations)
        dm = routing.DataModel(n_locations, 1)
        dm.add_cost_matrix(cost_matrix)
        data_models.append(dm)

    # Call batch solve without explicit settings
    solutions, solve_time = routing.BatchSolve(data_models)

    assert len(solutions) == len(tsp_sizes)
    for solution in solutions:
        assert solution.get_status() == routing.SolutionStatus.SUCCESS
