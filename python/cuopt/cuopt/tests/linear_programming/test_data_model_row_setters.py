# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuopt.linear_programming import data_model, solver, solver_settings
from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD
from cuopt.linear_programming.solver_settings import SolverMethod
from cuopt.utilities import InputValidationError


def _base_model():
    # min -x - y  s.t.  x + y <= 4,  x >= 0  (second row), x, y in [0, 3]
    dm = data_model.DataModel()
    dm.set_csr_constraint_matrix(
        np.array([1.0, 1.0, 1.0]),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 2, 3], dtype=np.int32),
    )
    dm.set_objective_coefficients(np.array([-1.0, -1.0]))
    dm.set_variable_lower_bounds(np.array([0.0, 0.0]))
    dm.set_variable_upper_bounds(np.array([3.0, 3.0]))
    return dm


def _objective(dm):
    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    sol = solver.Solve(dm, settings)
    return sol.get_primal_objective()


def test_rows_from_bounds_clears_rhs_and_row_types():
    dm = _base_model()
    dm.set_constraint_rows_from_rhs(np.array([4.0, 0.0]), np.array(["L", "G"]))
    dm.set_constraint_rows_from_bounds(
        np.array([-np.inf, 0.0]), np.array([4.0, np.inf])
    )
    assert len(dm.get_constraint_bounds()) == 0
    assert len(dm.get_row_types()) == 0
    assert dm.get_constraint_lower_bounds().tolist() == [-np.inf, 0.0]
    assert dm.get_constraint_upper_bounds().tolist() == [4.0, np.inf]


def test_rows_from_rhs_clears_bounds():
    dm = _base_model()
    dm.set_constraint_rows_from_bounds(
        np.array([-np.inf, 0.0]), np.array([4.0, np.inf])
    )
    dm.set_constraint_rows_from_rhs(np.array([4.0, 0.0]), np.array(["L", "G"]))
    assert len(dm.get_constraint_lower_bounds()) == 0
    assert len(dm.get_constraint_upper_bounds()) == 0
    assert dm.get_constraint_bounds().tolist() == [4.0, 0.0]
    assert list(dm.get_row_types()) == ["L", "G"]


def test_rows_from_rhs_accepts_str_row_types():
    dm = _base_model()
    dm.set_constraint_rows_from_rhs(np.array([4.0, 0.0]), "LG")
    assert len(dm.get_ascii_row_types()) == 2


def test_rows_from_bounds_length_mismatch_raises():
    dm = _base_model()
    with pytest.raises(InputValidationError):
        dm.set_constraint_rows_from_bounds(
            np.array([0.0]), np.array([1.0, 2.0])
        )


def test_rows_from_rhs_length_mismatch_raises():
    dm = _base_model()
    with pytest.raises(InputValidationError):
        dm.set_constraint_rows_from_rhs(np.array([1.0, 2.0]), np.array(["L"]))


def test_both_setters_give_the_same_solution():
    by_rhs = _base_model()
    by_rhs.set_constraint_rows_from_rhs(
        np.array([4.0, 0.0]), np.array(["L", "G"])
    )
    by_bounds = _base_model()
    by_bounds.set_constraint_rows_from_bounds(
        np.array([-np.inf, 0.0]), np.array([4.0, np.inf])
    )
    assert _objective(by_rhs) == pytest.approx(-4.0, abs=1e-4)
    assert _objective(by_bounds) == pytest.approx(-4.0, abs=1e-4)
