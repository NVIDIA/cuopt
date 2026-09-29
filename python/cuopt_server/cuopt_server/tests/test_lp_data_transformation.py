# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np

from cuopt_server.utils.linear_programming.data_definition import LPData
from cuopt_server.utils.linear_programming.data_transformation import (
    transform_lp_data,
)


def _sample_lp_dict():
    return {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {
            "upper_bounds": [5000.0],
            "lower_bounds": ["ninf"],
            "types": ["L"],
        },
        "objective_data": {
            "coefficients": [1.2, 1.7],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": ["inf", "inf"],
            "lower_bounds": [0.0, 0.0],
        },
        "initial_solution": {
            "primal": [1.0, 2.0],
            "dual": [0.5],
        },
        "variable_types": ["C", "I"],
        "variable_names": ["x", "y"],
        "maximize": False,
    }


def test_transform_lp_data_converts_dict_lists_to_numpy_dtypes():
    data = _sample_lp_dict()

    transform_lp_data(data)

    csr = data["csr_constraint_matrix"]
    assert csr["offsets"].dtype == np.int32
    assert csr["indices"].dtype == np.int32
    assert csr["values"].dtype == np.float64
    np.testing.assert_array_equal(
        csr["offsets"], np.array([0, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        csr["indices"], np.array([0, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        csr["values"], np.array([1.0, 1.0], dtype=np.float64)
    )

    assert data["constraint_bounds"]["upper_bounds"].dtype == np.float64
    assert data["constraint_bounds"]["types"].dtype == np.dtype("U1")
    np.testing.assert_array_equal(
        data["constraint_bounds"]["types"], np.array(["L"], dtype="U1")
    )

    assert data["objective_data"]["coefficients"].dtype == np.float64
    assert data["initial_solution"]["primal"].dtype == np.float64
    assert data["initial_solution"]["dual"].dtype == np.float64
    assert data["variable_types"].dtype == np.dtype("U1")
    np.testing.assert_array_equal(
        data["variable_types"], np.array(["C", "I"], dtype="U1")
    )

    # Unmapped fields must keep their original Python types.
    assert data["variable_names"] == ["x", "y"]
    assert data["maximize"] is False
    assert data["objective_data"]["offset"] == 0.0


def test_transform_lp_data_converts_inf_and_ninf_in_dict():
    data = _sample_lp_dict()

    transform_lp_data(data)

    np.testing.assert_array_equal(
        data["constraint_bounds"]["lower_bounds"],
        np.array([-np.inf], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        data["variable_bounds"]["upper_bounds"],
        np.array([np.inf, np.inf], dtype=np.float64),
    )
    assert data["variable_bounds"]["lower_bounds"].dtype == np.float64


def test_transform_lp_data_converts_lpdata_lists_to_numpy_dtypes():
    lp_data = LPData.parse_obj(_sample_lp_dict())

    transform_lp_data(lp_data)

    csr = lp_data.csr_constraint_matrix
    assert csr.offsets.dtype == np.int32
    assert csr.indices.dtype == np.int32
    assert csr.values.dtype == np.float64
    np.testing.assert_array_equal(
        csr.offsets, np.array([0, 2], dtype=np.int32)
    )

    assert lp_data.objective_data.coefficients.dtype == np.float64
    assert lp_data.initial_solution.primal.dtype == np.float64
    assert lp_data.initial_solution.dual.dtype == np.float64
    assert lp_data.variable_bounds.lower_bounds.dtype == np.float64

    # LPData path omits dtype for types / variable_types (numpy default).
    assert isinstance(lp_data.constraint_bounds.types, np.ndarray)
    assert isinstance(lp_data.variable_types, np.ndarray)
    np.testing.assert_array_equal(lp_data.constraint_bounds.types, ["L"])
    np.testing.assert_array_equal(lp_data.variable_types, ["C", "I"])


def test_transform_lp_data_converts_inf_and_ninf_in_lpdata():
    lp_data = LPData.parse_obj(_sample_lp_dict())

    transform_lp_data(lp_data)

    np.testing.assert_array_equal(
        lp_data.constraint_bounds.lower_bounds,
        np.array([-np.inf], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        lp_data.variable_bounds.upper_bounds,
        np.array([np.inf, np.inf], dtype=np.float64),
    )


def test_transform_lp_data_leaves_non_list_values_unchanged():
    existing = np.array([1.0, 2.0], dtype=np.float32)
    data = {
        "objective_data": {"coefficients": existing},
        "variable_names": ["x", "y"],
    }

    transform_lp_data(data)

    assert data["objective_data"]["coefficients"] is existing
    assert data["variable_names"] == ["x", "y"]


def test_transform_lp_data_mutates_in_place():
    data = _sample_lp_dict()
    original = data
    nested = data["csr_constraint_matrix"]

    transform_lp_data(data)

    assert data is original
    assert data["csr_constraint_matrix"] is nested
    assert isinstance(nested["offsets"], np.ndarray)


def test_transform_lp_data_mixed_inf_and_finite_values():
    data = {
        "variable_bounds": {
            "upper_bounds": [1.5, "inf", 3.0, "ninf"],
            "lower_bounds": ["ninf", 0.0, "inf"],
        }
    }

    transform_lp_data(data)

    np.testing.assert_array_equal(
        data["variable_bounds"]["upper_bounds"],
        np.array([1.5, np.inf, 3.0, -np.inf], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        data["variable_bounds"]["lower_bounds"],
        np.array([-np.inf, 0.0, np.inf], dtype=np.float64),
    )


def test_transform_lp_data_does_not_touch_unrelated_dict_keys():
    data = copy.deepcopy(_sample_lp_dict())
    data["solver_config"] = {"time_limit": 5}

    transform_lp_data(data)

    assert data["solver_config"] == {"time_limit": 5}
