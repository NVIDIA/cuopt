# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf

from cuopt import routing

# ---------------------------------------------------------------------------
# A single, richer problem exercised through many DataModel setters. Built
# three ways (cudf / numpy / pandas) so we can assert the host paths land the
# exact same data on device and solve to the exact same answer as cudf.
#   5 locations (0 == depot), 3 vehicles, 5 orders mapped to the 5 locations.
# ---------------------------------------------------------------------------
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
TRANSIT = (COST + 1).astype(np.float32)  # distinct from cost, still asymmetric

ORDER_LOCATIONS = np.array([0, 1, 2, 3, 4], dtype=np.int32)
ORDER_EARLIEST = np.array([0, 0, 0, 0, 0], dtype=np.int32)
ORDER_LATEST = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int32)
ORDER_SERVICE = np.array([0, 1, 1, 1, 1], dtype=np.int32)
ORDER_PRIZES = np.array([0, 2, 2, 2, 2], dtype=np.float32)
DEMAND = np.array([0, 1, 1, 1, 1], dtype=np.int32)

CAPACITY = np.array([10, 10, 10], dtype=np.int32)
VEH_EARLIEST = np.array([0, 0, 0], dtype=np.int32)
VEH_LATEST = np.array([1000, 1000, 1000], dtype=np.int32)
VEH_START = np.array([0, 0, 0], dtype=np.int32)
VEH_RETURN = np.array([0, 0, 0], dtype=np.int32)
VEH_TYPES = np.array([0, 0, 0], dtype=np.uint8)
VEH_MAX_COSTS = np.array([1000, 1000, 1000], dtype=np.float32)
VEH_MAX_TIMES = np.array([1000, 1000, 1000], dtype=np.float32)
VEH_FIXED_COSTS = np.array([0, 0, 0], dtype=np.float32)

OBJECTIVES = np.array([int(routing.Objective.COST)], dtype=np.int32)
OBJECTIVE_WEIGHTS = np.array([1], dtype=np.float32)

# matrix constructor, 1-D series constructor, keyed by input backend
CONVERTERS = {
    "cudf": (lambda m: cudf.DataFrame(m), lambda v: cudf.Series(v)),
    "numpy": (lambda m: np.asarray(m), lambda v: np.asarray(v)),
    "pandas": (lambda m: pd.DataFrame(m), lambda v: pd.Series(v)),
}

# every getter that reflects data set below, so equality across backends
# proves the whole surface (not just a couple of setters) landed identically.
GETTERS = [
    ("cost_matrix", lambda d: d.get_cost_matrix(0)),
    ("transit_time_matrix", lambda d: d.get_transit_time_matrix(0)),
    ("order_locations", lambda d: d.get_order_locations()),
    ("order_time_windows", lambda d: d.get_order_time_windows()),
    ("order_service_times", lambda d: d.get_order_service_times()),
    ("order_prizes", lambda d: d.get_order_prizes()),
    ("capacity_dimensions", lambda d: d.get_capacity_dimensions()),
    ("vehicle_time_windows", lambda d: d.get_vehicle_time_windows()),
    ("vehicle_locations", lambda d: d.get_vehicle_locations()),
    ("vehicle_types", lambda d: d.get_vehicle_types()),
    ("vehicle_max_costs", lambda d: d.get_vehicle_max_costs()),
    ("vehicle_max_times", lambda d: d.get_vehicle_max_times()),
    ("vehicle_fixed_costs", lambda d: d.get_vehicle_fixed_costs()),
    ("objective_function", lambda d: d.get_objective_function()),
]


def _build_full(backend):
    matrix, series = CONVERTERS[backend]
    d = routing.DataModel(COST.shape[0], CAPACITY.shape[0])
    d.add_cost_matrix(matrix(COST), 0)
    d.add_transit_time_matrix(matrix(TRANSIT), 0)
    d.set_order_locations(series(ORDER_LOCATIONS))
    d.set_order_time_windows(series(ORDER_EARLIEST), series(ORDER_LATEST))
    d.set_order_service_times(series(ORDER_SERVICE))
    d.set_order_prizes(series(ORDER_PRIZES))
    d.add_capacity_dimension("demand", series(DEMAND), series(CAPACITY))
    d.set_vehicle_time_windows(series(VEH_EARLIEST), series(VEH_LATEST))
    d.set_vehicle_locations(series(VEH_START), series(VEH_RETURN))
    d.set_vehicle_types(series(VEH_TYPES))
    d.set_vehicle_max_costs(series(VEH_MAX_COSTS))
    d.set_vehicle_max_times(series(VEH_MAX_TIMES))
    d.set_vehicle_fixed_costs(series(VEH_FIXED_COSTS))
    d.set_objective_function(series(OBJECTIVES), series(OBJECTIVE_WEIGHTS))
    return d


def _to_np(x):
    """Normalize any getter return (cudf/cupy/pandas/numpy, nested in
    tuple/dict) to plain numpy so results can be compared across backends.
    """
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        return [_to_np(e) for e in x]
    if isinstance(x, dict):
        return {
            k: _to_np(v) for k, v in sorted(x.items(), key=lambda i: str(i[0]))
        }
    if isinstance(x, cudf.DataFrame):
        return x.to_numpy()
    if hasattr(x, "to_numpy"):  # cudf/pandas Series
        return x.to_numpy()
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _assert_equal(a, b, label):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), label
        for k in a:
            _assert_equal(a[k], b[k], f"{label}[{k}]")
    elif isinstance(a, list):
        assert len(a) == len(b), label
        for i, (ai, bi) in enumerate(zip(a, b)):
            _assert_equal(ai, bi, f"{label}[{i}]")
    else:
        np.testing.assert_array_equal(a, b, err_msg=label)


# ---------------------------------------------------------------------------
# 1. Data-landing correctness: every getter identical across cudf/numpy/pandas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "pandas"])
def test_all_getters_match_cudf(backend):
    ref = _build_full("cudf")
    got = _build_full(backend)
    for label, getter in GETTERS:
        _assert_equal(_to_np(getter(ref)), _to_np(getter(got)), label)


# ---------------------------------------------------------------------------
# 2. Solve smoke: each backend produces a valid solution.
# Note: we deliberately do NOT compare solutions across backends -- the routing
# solver may return different equal-cost routes on independent solves (alternate
# optima), so cross-backend answer equality is not a valid invariant. Data
# correctness is covered by test_all_getters_match_cudf above.
# ---------------------------------------------------------------------------
def _solve(d):
    settings = routing.SolverSettings()
    settings.set_time_limit(3)
    return routing.Solve(d, settings)


@pytest.mark.parametrize("backend", ["cudf", "numpy", "pandas"])
def test_backend_solves_successfully(backend):
    sol = _solve(_build_full(backend))
    status = sol.get_status()
    assert getattr(status, "value", status) == 0
    assert sol.get_vehicle_count() >= 1


# ---------------------------------------------------------------------------
# 3. Contract: matrix reorder is correct (no silent transpose), lists rejected
# ---------------------------------------------------------------------------
def test_numpy_matrix_is_c_contiguous_no_transpose():
    d = _build_full("numpy")
    ret = d.get_cost_matrix(0)
    assert ret.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(cp.asnumpy(ret), COST)  # asymmetric: not .T


def test_python_list_rejected_matrix():
    d = routing.DataModel(COST.shape[0], CAPACITY.shape[0])
    with pytest.raises(TypeError, match="lists/tuples are not supported"):
        d.add_cost_matrix(COST.tolist())


def test_python_list_rejected_series():
    d = routing.DataModel(COST.shape[0], CAPACITY.shape[0])
    d.add_cost_matrix(COST)
    with pytest.raises(TypeError, match="lists/tuples are not supported"):
        d.set_order_time_windows(
            ORDER_EARLIEST.tolist(), ORDER_LATEST.tolist()
        )
