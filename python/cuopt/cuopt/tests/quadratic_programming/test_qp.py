# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from cuopt.linear_programming import data_model, solver, solver_settings
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_ITERATION_LIMIT,
)


def test_solver():
    data_model_obj = data_model.DataModel()

    # Minimize x1 ^2 + 4 x2 ^2 - 8 x1 - 16 x2
    # subject to x1 + x2 >= 5
    #         x1 >= 3
    #         x2 >= 0

    # A
    A_values = np.array([1.0, 1.0])
    A_indices = np.array([0, 1])
    A_offsets = np.array([0, 2])
    data_model_obj.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    # constr_lb
    constr_lower_bounds = np.array([5.0])
    data_model_obj.set_constraint_lower_bounds(constr_lower_bounds)

    # constr_upper_bounds
    constr_upper_bounds = np.array([np.inf])
    data_model_obj.set_constraint_upper_bounds(constr_upper_bounds)

    #  variable bounds
    lb = np.array([0.0, 0.0])
    # ub = np.array([np.inf, np.inf])
    ub = np.array([10.0, 10.0])
    data_model_obj.set_variable_lower_bounds(lb)
    data_model_obj.set_variable_upper_bounds(ub)

    # c
    c = np.array([-8.0, -16.0])
    data_model_obj.set_objective_coefficients(c)

    # Q
    Q_values = np.array([1.0, 4.0])
    Q_indices = np.array([0, 1])
    Q_offsets = np.array([0, 1, 2])

    data_model_obj.set_quadratic_objective_matrix(
        Q_values, Q_indices, Q_offsets
    )

    test_q = data_model_obj.get_quadratic_objective_values()
    print(f"test_q: {test_q}")
    test_q_indices = data_model_obj.get_quadratic_objective_indices()
    print(f"test_q_indices: {test_q_indices}")
    test_q_offsets = data_model_obj.get_quadratic_objective_offsets()
    print(f"test_q_offsets: {test_q_offsets}")

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_ITERATION_LIMIT, 10)
    solution = solver.Solve(data_model_obj, settings)
    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(-32)
    assert solution.get_primal_solution()[0] == pytest.approx(4.0)
    assert solution.get_primal_solution()[1] == pytest.approx(2.0)
