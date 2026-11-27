# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cuopt import routing


def test_duplicate_distance_matrix_is_rejected():
    model = routing.DataModel(2, 1)
    matrix = np.zeros((2, 2), dtype=np.float32)
    model.add_distance_matrix(matrix)
    with pytest.raises(ValueError, match="already been added"):
        model.add_distance_matrix(matrix)


@pytest.mark.parametrize("vehicle_type", [-1, 256, 1.5])
def test_distance_matrix_rejects_invalid_vehicle_type(vehicle_type):
    model = routing.DataModel(2, 1)
    with pytest.raises((ValueError, TypeError)):
        model.add_distance_matrix(
            np.zeros((2, 2), dtype=np.float32), vehicle_type
        )


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 3)),
        np.zeros(2),
        np.zeros((3, 3)),
        [[0, -1], [1, 0]],
        [[0, np.nan], [1, 0]],
        [[0, -np.inf], [1, 0]],
        [[0, float(np.finfo(np.float32).max) * 2.0], [1, 0]],
    ],
)
def test_distance_matrix_rejects_invalid_values_or_shape(matrix):
    model = routing.DataModel(2, 1)
    with pytest.raises((ValueError, TypeError)):
        model.add_distance_matrix(np.asarray(matrix, dtype=np.float64))


@pytest.mark.parametrize("shape", [(4,), (2, 3), (3, 3)])
def test_skip_distance_validation_still_checks_shape(shape):
    model = routing.DataModel(2, 1)
    with pytest.raises(ValueError):
        model.add_distance_matrix(np.zeros(shape), skip_validation=True)


@pytest.mark.parametrize("skip_validation", [False, True])
@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0, 2 + 4j], [3, 0]]),
        pd.DataFrame({0: [0, 3], 1: [2 + 4j, 0]}),
    ],
)
def test_distance_matrix_rejects_complex_input(matrix, skip_validation):
    model = routing.DataModel(2, 1)
    with pytest.raises(TypeError, match="real"):
        model.add_distance_matrix(matrix, skip_validation=skip_validation)


def test_distance_matrix_rejects_masked_distance():
    model = routing.DataModel(2, 1)
    with pytest.raises(ValueError, match="NULL"):
        model.add_distance_matrix(
            np.ma.array([[0, 7], [3, 0]], mask=[[False, True], [False, False]])
        )


@pytest.mark.parametrize(
    "backend", ["numpy", "numpy_longdouble", "cudf", "pandas_nullable"]
)
def test_auxiliary_distance_matrix_preserves_cost_and_time(backend):
    import cudf

    def matrix(values):
        array = np.asarray(values, dtype=np.float32)
        return cudf.DataFrame(array) if backend == "cudf" else array

    model = routing.DataModel(2, 1, 1)
    model.add_cost_matrix(matrix([[0, 3], [5, 0]]))
    if backend == "pandas_nullable":
        model.add_distance_matrix(
            pd.DataFrame([[0, 100], [200, 0]], dtype="Int64")
        )
    elif backend == "numpy_longdouble":
        model.add_distance_matrix(
            np.array([[0, 100], [200, 0]], dtype=np.longdouble)
        )
    else:
        model.add_distance_matrix(matrix([[0, np.inf], [200, 0]]))
    model.add_transit_time_matrix(matrix([[0, 11], [17, 0]]))
    model.set_order_locations(np.array([1], dtype=np.int32))
    settings = routing.SolverSettings()
    settings.set_time_limit(0.1)
    solution = routing.Solve(model, settings)
    assert solution.get_status() == 0
    np.testing.assert_allclose(
        solution.get_objective_values()[routing.Objective.COST], 8
    )
    route = solution.get_route().to_pandas()
    np.testing.assert_allclose(route["arrival_stamp"].to_numpy(), [0, 11, 28])
