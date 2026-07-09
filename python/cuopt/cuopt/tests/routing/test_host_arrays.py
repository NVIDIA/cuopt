# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf

from cuopt import routing

# Small self-contained problem: 5 locations (depot + 4 orders), 2 vehicles.
COST = np.array(
    [
        [0, 4, 5, 2, 7],
        [3, 0, 6, 8, 1],
        [5, 2, 0, 4, 9],
        [6, 3, 7, 0, 2],
        [1, 8, 4, 5, 0],
    ],
    dtype=np.float32,
)
EARLIEST = np.array([0, 0, 0, 0, 0], dtype=np.int32)
LATEST = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int32)
DEMAND = np.array([0, 1, 1, 1, 1], dtype=np.int32)
CAPACITY = np.array([10, 10], dtype=np.int32)

# matrix constructor, 1-D series constructor, keyed by input backend
CONVERTERS = {
    "cudf": (lambda m: cudf.DataFrame(m), lambda v: cudf.Series(v)),
    "numpy": (lambda m: np.asarray(m), lambda v: np.asarray(v)),
    "pandas": (lambda m: pd.DataFrame(m), lambda v: pd.Series(v)),
}


def _build(backend):
    as_matrix, as_series = CONVERTERS[backend]
    d = routing.DataModel(COST.shape[0], CAPACITY.shape[0])
    d.add_cost_matrix(as_matrix(COST))
    d.set_order_time_windows(as_series(EARLIEST), as_series(LATEST))
    d.add_capacity_dimension("demand", as_series(DEMAND), as_series(CAPACITY))
    return d


@pytest.mark.parametrize("backend", ["cudf", "numpy", "pandas"])
def test_host_inputs_land_on_device(backend):
    d = _build(backend)

    # Matrix comes back row-major and identical regardless of input backend
    # (cudf.to_cupy is column-major; the wrapper reorders it to C order).
    ret_cost = cp.asnumpy(d.get_cost_matrix())
    np.testing.assert_array_equal(ret_cost, COST)

    earliest, latest = d.get_order_time_windows()
    np.testing.assert_array_equal(earliest.to_numpy(), EARLIEST)
    np.testing.assert_array_equal(latest.to_numpy(), LATEST)


def test_numpy_matrix_is_c_contiguous_single_copy():
    # A C-contiguous numpy matrix must not be transposed by the reorder path.
    d = _build("numpy")
    ret = d.get_cost_matrix()
    assert ret.flags["C_CONTIGUOUS"]
    # Asymmetric matrix: transpose would change values -> guards against a
    # silent column/row-major mix-up.
    np.testing.assert_array_equal(cp.asnumpy(ret), COST)
    np.testing.assert_array_equal(cp.asnumpy(ret).T, COST.T)


@pytest.mark.parametrize("backend", ["numpy", "pandas"])
def test_solve_with_host_inputs(backend):
    d = _build(backend)
    settings = routing.SolverSettings()
    settings.set_time_limit(2)
    sol = routing.Solve(d, settings)
    status = sol.get_status()
    assert getattr(status, "value", status) == 0  # SolutionStatus.SUCCESS
    assert sol.get_vehicle_count() >= 1


def test_python_list_rejected_matrix():
    d = routing.DataModel(COST.shape[0], CAPACITY.shape[0])
    with pytest.raises(TypeError, match="lists/tuples are not supported"):
        d.add_cost_matrix(COST.tolist())


def test_python_list_rejected_series():
    d = routing.DataModel(COST.shape[0], CAPACITY.shape[0])
    d.add_cost_matrix(COST)
    with pytest.raises(TypeError, match="lists/tuples are not supported"):
        d.set_order_time_windows(EARLIEST.tolist(), LATEST.tolist())
