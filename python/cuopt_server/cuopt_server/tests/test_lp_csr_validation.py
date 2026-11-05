# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuopt_server.utils.linear_programming.data_definition import (
    CSRConstraintMatrix,
    LPData,
)
from cuopt_server.utils.linear_programming.data_transformation import (
    transform_lp_data,
)
from cuopt_server.utils.linear_programming.data_validation import (
    validate_csr_matrix,
    validate_LP_data,
)


def _csr_matrix(offsets, indices, values, as_numpy):
    if as_numpy:
        offsets = np.array(offsets, dtype=np.int32)
        indices = np.array(indices, dtype=np.int32)
        values = np.array(values, dtype=np.float64)
    return CSRConstraintMatrix(offsets=offsets, indices=indices, values=values)


@pytest.mark.parametrize("as_numpy", [False, True], ids=["list", "numpy"])
@pytest.mark.parametrize(
    "offsets, indices, values",
    [
        ([0], [], []),
        ([0, 0], [], []),
        ([0, 0, 0], [], []),
        ([0, 0, 2, 2], [0, 1], [1.0, -1.0]),
    ],
)
def test_validate_csr_matrix_accepts_empty_rows(
    offsets, indices, values, as_numpy
):
    matrix = _csr_matrix(offsets, indices, values, as_numpy)

    assert validate_csr_matrix(matrix) == (True, "Valid CSR Matrix")


@pytest.mark.parametrize("as_numpy", [False, True], ids=["list", "numpy"])
@pytest.mark.parametrize(
    "offsets, indices, values, message",
    [
        (
            [0, 1],
            [-1],
            [1.0],
            "indices values must be greater than or equal to 0",
        ),
        ([-1, 0], [], [], "offset values must be greater than or equal to 0"),
        (
            [0, 1],
            [0],
            [1.0, 2.0],
            "Length of values array must be equal to indices array",
        ),
        (
            [0, 2],
            [0, 1],
            [1.0],
            "Length of values array must be equal to indices array",
        ),
        (
            [0, 0],
            [],
            [1.0],
            "Length of values array must be equal to indices array",
        ),
        (
            [0, 1],
            [0],
            [],
            "Length of values array must be equal to indices array",
        ),
    ],
)
def test_validate_csr_matrix_rejects_invalid_data(
    offsets, indices, values, message, as_numpy
):
    matrix = _csr_matrix(offsets, indices, values, as_numpy)

    assert validate_csr_matrix(matrix) == (False, message)


@pytest.mark.parametrize("num_rows", [1, 2])
def test_validate_lp_data_accepts_zero_nonzeros(num_rows):
    data = {
        "csr_constraint_matrix": {
            "offsets": [0] * (num_rows + 1),
            "indices": [],
            "values": [],
        },
        "constraint_bounds": {
            "lower_bounds": [0.0] * num_rows,
            "upper_bounds": [1.0] * num_rows,
        },
        "objective_data": {"coefficients": [1.0]},
        "variable_bounds": {"lower_bounds": [0.0], "upper_bounds": [1.0]},
    }
    transform_lp_data(data)

    validate_LP_data(LPData.parse_obj(data))
