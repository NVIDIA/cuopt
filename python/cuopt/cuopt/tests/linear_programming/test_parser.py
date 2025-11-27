# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import cuopt_mps_parser
import numpy as np
import pytest
from cuopt_mps_parser.utilities import InputValidationError

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.getcwd()
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")


def test_bad_mps_files():
    NumMpsFiles = 13
    for i in range(1, NumMpsFiles + 1):
        file_path = (
            RAPIDS_DATASET_ROOT_DIR + f"/linear_programming/bad-mps-{i}.mps"
        )
        if os.path.exists(file_path):
            with pytest.raises(InputValidationError):
                cuopt_mps_parser.ParseMps(file_path, True)


def test_good_mps_file():
    file_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/good-mps-free-var.mps"
    )
    data_model = cuopt_mps_parser.ParseMps(file_path)

    assert not data_model.get_sense()

    assert 3.0 == data_model.get_constraint_matrix_values()[0]
    assert 4.0 == data_model.get_constraint_matrix_values()[1]
    assert 2.7 == data_model.get_constraint_matrix_values()[2]
    assert 10.1 == data_model.get_constraint_matrix_values()[3]

    assert 0 == data_model.get_constraint_matrix_indices()[0]
    assert 1 == data_model.get_constraint_matrix_indices()[1]
    assert 0 == data_model.get_constraint_matrix_indices()[2]
    assert 1 == data_model.get_constraint_matrix_indices()[3]

    assert 0 == data_model.get_constraint_matrix_offsets()[0]
    assert 2 == data_model.get_constraint_matrix_offsets()[1]
    assert 4 == data_model.get_constraint_matrix_offsets()[2]

    assert 5.4 == data_model.get_constraint_bounds()[0]
    assert 4.9 == data_model.get_constraint_bounds()[1]

    assert 0.2 == data_model.get_objective_coefficients()[0]
    assert 0.1 == data_model.get_objective_coefficients()[1]

    assert 1.0 == data_model.get_objective_scaling_factor()
    assert 0.0 == data_model.get_objective_offset()

    assert -np.inf == data_model.get_variable_lower_bounds()[0]
    assert 0.0 == data_model.get_variable_lower_bounds()[1]

    assert np.inf == data_model.get_variable_upper_bounds()[0]
    assert np.inf == data_model.get_variable_upper_bounds()[1]

    assert -np.inf == data_model.get_constraint_lower_bounds()[0]
    assert -np.inf == data_model.get_constraint_lower_bounds()[1]

    assert 5.4 == data_model.get_constraint_upper_bounds()[0]
    assert 4.9 == data_model.get_constraint_upper_bounds()[1]
