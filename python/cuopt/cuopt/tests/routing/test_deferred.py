# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuopt import routing
from cuopt.routing import vehicle_routing_wrapper
from cuopt.routing._deferred import _SKIP_GETTERS


def test_deferred_covers_wrapper_surface():
    """Every public wrapper DataModel method must be handled by the recording
    layer (installed as a recorder/getter or an explicit override). This fails
    loudly if a new wrapper method -- e.g. a mutator not named set_*/add_* --
    is added without being recorded, instead of silently dropping its data.
    """
    dm = routing.DataModel(1, 1)  # triggers _install_methods
    handled = set(dir(type(dm)))
    missing = [
        name
        for name in dir(vehicle_routing_wrapper.DataModel)
        if not name.startswith("_")
        and name not in _SKIP_GETTERS
        and name not in handled
    ]
    assert not missing, (
        f"deferred-build layer does not handle wrapper methods {missing}; "
        "add a recorder/getter or list them in _SKIP_GETTERS"
    )


def test_unknown_method_is_not_recorded():
    """A call to a method that is not part of the DataModel surface raises
    AttributeError rather than being silently recorded. The recording layer
    installs only the declared setters/getters (no ``__getattr__`` catch-all),
    so a typo'd or unknown call fails loudly and never enters the IR.
    """
    dm = routing.DataModel(3, 1)
    with pytest.raises(AttributeError):
        dm.random_func_call(1, 2, 3)
    assert dm._calls == []


def test_vehicle_max_distance_accepts_zero_and_rejects_nonfinite():
    dm = routing.DataModel(2, 1)
    dm.set_vehicle_max_distances(np.array([0.0], dtype=np.float32))

    for invalid_value in (np.inf, np.nan):
        with pytest.raises(ValueError, match="finite"):
            dm.set_vehicle_max_distances(
                np.array([invalid_value], dtype=np.float32)
            )

    with pytest.raises(ValueError, match="representable as float32"):
        dm.set_vehicle_max_distances(np.array([1e100], dtype=np.float64))


def test_vehicle_max_cost_accepts_zero_and_rejects_nonfinite():
    dm = routing.DataModel(2, 1)
    dm.set_vehicle_max_costs(np.array([0.0], dtype=np.float32))

    for invalid_value in (np.inf, np.nan):
        with pytest.raises(ValueError, match="finite"):
            dm.set_vehicle_max_costs(
                np.array([invalid_value], dtype=np.float32)
            )

    with pytest.raises(ValueError, match="representable as float32"):
        dm.set_vehicle_max_costs(np.array([1e100], dtype=np.float64))


@pytest.mark.parametrize(
    "vehicle_types",
    [
        np.array([1.5], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
    ],
)
def test_vehicle_types_must_contain_integers(vehicle_types):
    dm = routing.DataModel(2, 1)
    with pytest.raises(TypeError, match="must contain integers"):
        dm.set_vehicle_types(vehicle_types)


def test_distance_tiers_require_float32_open_ended_threshold():
    dm = routing.DataModel(2, 1)
    vehicle_ids = np.array([0], dtype=np.int32)
    costs = np.array([1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="last distance tier threshold"):
        dm.set_vehicle_distance_tiers(
            vehicle_ids,
            np.array([1e9], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            costs,
        )

    dm.set_vehicle_distance_tiers(
        vehicle_ids,
        np.array([np.finfo(np.float32).max], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        costs,
    )


def test_distance_tier_thresholds_remain_distinct_after_float32_cast():
    dm = routing.DataModel(2, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        dm.set_vehicle_distance_tiers(
            np.array([0, 0, 0], dtype=np.int32),
            np.array(
                [1.00000001, 1.00000002, np.finfo(np.float32).max],
                dtype=np.float64,
            ),
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )


@pytest.mark.parametrize("vehicle_type", [-1, 256, 1.5])
def test_matrix_vehicle_type_must_fit_uint8(vehicle_type):
    dm = routing.DataModel(2, 1)
    matrix = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises((TypeError, ValueError)):
        dm.add_distance_matrix(matrix, vehicle_type)


def test_distance_matrix_rejects_oversized_finite_values_but_accepts_infinity():
    dm = routing.DataModel(2, 1)
    with pytest.raises(
        ValueError, match="finite values must be representable"
    ):
        dm.add_distance_matrix(
            np.array([[0.0, 1e100], [1.0, 0.0]], dtype=np.float64)
        )

    dm.add_distance_matrix(
        np.array([[0.0, np.inf], [1.0, 0.0]], dtype=np.float32)
    )
