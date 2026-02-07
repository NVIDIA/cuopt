# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest

from cuopt.linear_programming import SolverSettings
from cuopt.linear_programming.internals import (
    GetSolutionCallback,
    SetSolutionCallback,
)
from cuopt.linear_programming.problem import (
    CONTINUOUS,
    INTEGER,
    MAXIMIZE,
    MINIMIZE,
    CType,
    Problem,
    VType,
    sense,
    QuadraticExpression,
)
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_AUGMENTED,
    CUOPT_BARRIER_DUAL_INITIAL_POINT,
    CUOPT_CUDSS_DETERMINISTIC,
    CUOPT_DUALIZE,
    CUOPT_ELIMINATE_DENSE_COLUMNS,
    CUOPT_FOLDING,
    CUOPT_INFEASIBILITY_DETECTION,
    CUOPT_METHOD,
    CUOPT_ORDERING,
    CUOPT_PDLP_SOLVER_MODE,
    CUOPT_PRESOLVE,
)
from cuopt.linear_programming.solver_settings import (
    PDLPSolverMode,
    SolverMethod,
)

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.getcwd()
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")


def test_model():
    prob = Problem("Simple MIP")
    assert prob.Name == "Simple MIP"

    # Adding Variable
    x = prob.addVariable(lb=0, vtype=VType.INTEGER, name="V_x")
    y = prob.addVariable(lb=10, ub=50, vtype=INTEGER, name="V_y")

    assert x.getVariableName() == "V_x"
    assert y.getUpperBound() == 50
    assert y.getLowerBound() == 10
    assert x.getVariableType() == VType.INTEGER
    assert y.getVariableType() == "I"
    assert [x.getIndex(), y.getIndex()] == [0, 1]
    assert prob.IsMIP

    # Adding Constraints
    prob.addConstraint(2 * x + 4 * y >= 230, name="C1")
    prob.addConstraint(3 * x + 2 * y + 10 <= 200, name="C2")

    expected_name = ["C1", "C2"]
    expected_coefficient_x = [2, 3]
    expected_coefficient_y = [4, 2]
    expected_sense = [CType.GE, "L"]
    expected_rhs = [230, 190]
    for i, c in enumerate(prob.getConstraints()):
        assert c.getConstraintName() == expected_name[i]
        assert c.getSense() == expected_sense[i]
        assert c.getRHS() == expected_rhs[i]
        assert c.getCoefficient(x) == expected_coefficient_x[i]
        assert c.getCoefficient(y) == expected_coefficient_y[i]

    assert prob.NumVariables == 2
    assert prob.NumConstraints == 2
    assert prob.NumNZs == 4

    # Setting Objective
    expr = 5 * x + 3 * y + 50
    prob.setObjective(expr, sense=MAXIMIZE)

    expected_obj_coeff = [5, 3]
    assert expr.getVariables() == [x, y]
    assert expr.getCoefficients() == expected_obj_coeff
    assert expr.getConstant() == 50
    assert prob.ObjSense == sense.MAXIMIZE
    assert prob.getObjective().vars == [x, y]
    assert prob.getObjective().coefficients == [5, 3]
    assert prob.getObjective().constant == prob.ObjConstant

    # Initialize Settings
    settings = SolverSettings()
    settings.set_parameter("time_limit", 5)

    assert not prob.solved
    # Solving Problem
    prob.solve(settings)
    assert prob.solved
    assert prob.Status.name == "Optimal"
    assert prob.SolveTime < 5

    csr = prob.getCSR()
    expected_row_pointers = [0, 2, 4]
    expected_column_indices = [0, 1, 0, 1]
    expected_values = [2.0, 4.0, 3.0, 2.0]

    assert csr.row_pointers == expected_row_pointers
    assert csr.column_indices == expected_column_indices
    assert csr.values == expected_values

    expected_slack = [-6, 0]
    expected_var_values = [36, 41]

    for i, var in enumerate(prob.getVariables()):
        assert var.Value == pytest.approx(expected_var_values[i])
        assert var.getObjectiveCoefficient() == expected_obj_coeff[i]

    assert prob.ObjValue == 353

    for i, c in enumerate(prob.getConstraints()):
        assert c.Slack == pytest.approx(expected_slack[i])

    assert hasattr(prob.SolutionStats, "mip_gap")

    # Change Objective
    prob.setObjective(expr + 20, sense.MINIMIZE)
    assert not prob.solved

    # Check if values reset
    for i, var in enumerate(prob.getVariables()):
        assert math.isnan(var.Value) and math.isnan(var.ReducedCost)
    for i, c in enumerate(prob.getConstraints()):
        assert math.isnan(c.Slack) and math.isnan(c.DualValue)

    # Change Problem to LP
    x.VariableType = VType.CONTINUOUS
    y.VariableType = CONTINUOUS
    y.UB = 45.5
    assert not prob.IsMIP

    prob.solve(settings)
    assert prob.solved
    assert prob.Status.name == "Optimal"
    assert hasattr(prob.SolutionStats, "primal_residual")

    assert x.getValue() == 24
    assert y.getValue() == pytest.approx(45.5)

    assert prob.ObjValue == pytest.approx(5 * x.Value + 3 * y.Value + 70)


def test_linear_expression():
    prob = Problem()

    x = prob.addVariable()
    y = prob.addVariable()
    z = prob.addVariable()

    expr1 = 2 * x + 5 + 3 * y
    expr2 = y - z + 2 * x - 3

    expr3 = expr1 + expr2
    expr4 = expr1 - expr2

    # Test expr1 and expr 2 is unchanged
    assert expr1.getCoefficients() == [2, 3]
    assert expr1.getVariables() == [x, y]
    assert expr1.getConstant() == 5
    assert expr2.getCoefficients() == [1, -1, 2]
    assert expr2.getVariables() == [y, z, x]
    assert expr2.getConstant() == -3

    # Testing add and sub
    assert expr3.getCoefficients() == [2, 3, 1, -1, 2]
    assert expr3.getVariables() == [x, y, y, z, x]
    assert expr3.getConstant() == 2
    assert expr4.getCoefficients() == [2, 3, -1, 1, -2]
    assert expr4.getVariables() == [x, y, y, z, x]
    assert expr4.getConstant() == 8

    expr5 = 8 * y - x - 5
    expr6 = expr5 / 2
    expr7 = expr5 * 2

    # Test expr5 is unchanged
    assert expr5.getCoefficients() == [8, -1]
    assert expr5.getVariables() == [y, x]
    assert expr5.getConstant() == -5

    # Test mul and truediv
    assert expr6.getCoefficients() == [4, -0.5]
    assert expr6.getVariables() == [y, x]
    assert expr6.getConstant() == -2.5
    assert expr7.getCoefficients() == [16, -2]
    assert expr7.getVariables() == [y, x]
    assert expr7.getConstant() == -10

    expr6 *= 2
    expr7 /= 2

    # Test imul and itruediv
    assert expr6.getCoefficients() == [8, -1]
    assert expr6.getVariables() == [y, x]
    assert expr6.getConstant() == -5
    assert expr7.getCoefficients() == [8, -1]
    assert expr7.getVariables() == [y, x]
    assert expr7.getConstant() == -5


def test_constraint_matrix():
    prob = Problem()

    a = prob.addVariable(lb=0, ub=float("inf"), vtype="C", name="a")
    b = prob.addVariable(lb=0, ub=float("inf"), vtype="C", name="b")
    c = prob.addVariable(lb=0, ub=float("inf"), vtype="C", name="c")
    d = prob.addVariable(lb=0, ub=float("inf"), vtype="C", name="d")
    e = prob.addVariable(lb=0, ub=float("inf"), vtype="C", name="e")
    f = prob.addVariable(lb=0, ub=float("inf"), vtype="C", name="f")

    # 2*a + 3*e + 1 + 4*d - 2*e + f - 8 <= 90    i.e.    2a + e + 4d + f <= 97
    prob.addConstraint(2 * a + 3 * e + 1 + 4 * d - 2 * e + f - 8 <= 90, "C1")
    # d + 5*c - a - 4*d - 2 + 5*b - 20 >= 10    i.e.    -3d + 5c - a + 5b >= 32
    prob.addConstraint(d + 5 * c - a - 4 * d - 2 + 5 * b - 20 >= 10, "C2")
    # 7*f + 3 - 2*b + c == 3*f - 61 + 8*e    i.e.    4f - 2b + c - 8e == -64
    prob.addConstraint(7 * f + 3 - 2 * b + c == 3 * f - 61 + 8 * e, "C3")
    # a <= 5
    prob.addConstraint(a <= 5, "C4")
    # d >= 7*f - b - 27   i.e.   d - 7*f + b >= -27
    prob.addConstraint(d >= 7 * f - b - 27, "C5")
    # c == e   i.e.   c - e == 0
    prob.addConstraint(c == e, "C6")

    sense = []
    rhs = []
    for c in prob.getConstraints():
        sense.append(c.Sense)
        rhs.append(c.RHS)

    csr = prob.getCSR()

    exp_row_pointers = [0, 4, 8, 12, 13, 16, 18]
    exp_column_indices = [0, 4, 3, 5, 2, 3, 0, 1, 5, 1, 2, 4, 0, 5, 1, 3, 2, 4]
    exp_values = [
        2.0,
        1.0,
        4.0,
        1.0,
        5.0,
        -3.0,
        -1.0,
        5.0,
        4.0,
        -2.0,
        1.0,
        -8.0,
        1.0,
        -7.0,
        1.0,
        1.0,
        1.0,
        -1.0,
    ]
    exp_sense = ["L", "G", "E", "L", "G", "E"]
    exp_rhs = [97, 32, -64, 5, -27, 0]

    assert csr.row_pointers == exp_row_pointers
    assert csr.column_indices == exp_column_indices
    assert csr.values == exp_values
    assert sense == exp_sense
    assert rhs == exp_rhs


def test_read_write_mps_and_relaxation():
    # Create MIP model
    m = Problem("SMALLMIP")

    # Vars: continuous, nonnegative by default
    x1 = m.addVariable(name="x1", lb=0.0, vtype=INTEGER)
    x2 = m.addVariable(name="x2", lb=0.0, ub=4.0, vtype=INTEGER)
    x3 = m.addVariable(name="x3", lb=0.0, ub=6.0, vtype=INTEGER)
    x4 = m.addVariable(name="x4", lb=0.0, vtype=INTEGER)
    x5 = m.addVariable(name="x5", lb=0.0, vtype=INTEGER)

    # Objective (minimize)
    m.setObjective(2 * x1 + 3 * x2 + x3 + 1 * x4 + 4 * x5, MINIMIZE)

    # Constraints (5 total)
    m.addConstraint(x1 + x2 + x3 <= 10, name="c1")
    m.addConstraint(2 * x1 + x3 - x4 >= 3, name="c2")
    m.addConstraint(x2 + 3 * x5 == 7)
    m.addConstraint(x4 + x5 <= 8)
    m.addConstraint(x1 + x2 + x3 + x4 + x5 >= 5, name="c5")

    # Write MPS
    m.writeMPS("small_mip.mps")

    # Read MPS and solve
    prob = Problem.readMPS("small_mip.mps")
    assert prob.Name == "SMALLMIP"
    assert prob.IsMIP
    prob.solve()

    expected_values_mip = [1.0, 1.0, 1.0, 0.0, 2.0]
    assert prob.Status.name == "Optimal"
    for i, v in enumerate(prob.getVariables()):
        assert v.getValue() == pytest.approx(expected_values_mip[i])

    # Relax the Problem into LP and solve
    lp_prob = prob.relax()
    assert not lp_prob.IsMIP
    lp_prob.solve()

    expected_values_lp = [0.33333333, 0.0, 2.33333333, 0.0, 2.33333333]
    assert lp_prob.Status.name == "Optimal"
    for i, v in enumerate(lp_prob.getVariables()):
        assert v.getValue() == pytest.approx(expected_values_lp[i])


def _run_incumbent_solutions(include_set_callback):
    # Callback for incumbent solution
    class CustomGetSolutionCallback(GetSolutionCallback):
        def __init__(self, user_data):
            super().__init__()
            self.n_callbacks = 0
            self.solutions = []
            self.user_data = user_data

        def get_solution(
            self, solution, solution_cost, solution_bound, user_data
        ):
            assert user_data is self.user_data
            self.n_callbacks += 1
            assert len(solution) > 0
            assert len(solution_cost) == 1
            assert len(solution_bound) == 1

            self.solutions.append(
                {
                    "solution": solution.tolist(),
                    "cost": float(solution_cost[0]),
                    "bound": float(solution_bound[0]),
                }
            )

    class CustomSetSolutionCallback(SetSolutionCallback):
        def __init__(self, get_callback, user_data):
            super().__init__()
            self.n_callbacks = 0
            self.get_callback = get_callback
            self.user_data = user_data

        def set_solution(
            self, solution, solution_cost, solution_bound, user_data
        ):
            assert user_data is self.user_data
            self.n_callbacks += 1
            assert len(solution_bound) == 1
            if self.get_callback.solutions:
                solution[:] = self.get_callback.solutions[-1]["solution"]
                solution_cost[0] = float(
                    self.get_callback.solutions[-1]["cost"]
                )

    prob = Problem()
    x = prob.addVariable(vtype=VType.INTEGER)
    y = prob.addVariable(vtype=VType.INTEGER)
    prob.addConstraint(2 * x + 4 * y >= 230)
    prob.addConstraint(3 * x + 2 * y <= 190)
    prob.setObjective(5 * x + 3 * y, sense=sense.MAXIMIZE)

    user_data = {"source": "test_incumbent_solutions"}
    get_callback = CustomGetSolutionCallback(user_data)
    set_callback = (
        CustomSetSolutionCallback(get_callback, user_data)
        if include_set_callback
        else None
    )
    settings = SolverSettings()
    settings.set_mip_callback(get_callback, user_data)
    if include_set_callback:
        settings.set_mip_callback(set_callback, user_data)
    settings.set_parameter("time_limit", 1)

    prob.solve(settings)

    assert get_callback.n_callbacks > 0

    for sol in get_callback.solutions:
        x_val = sol["solution"][0]
        y_val = sol["solution"][1]
        cost = sol["cost"]
        tol = 1e-6
        assert 2 * x_val + 4 * y_val >= 230 - tol
        assert 3 * x_val + 2 * y_val <= 190 + tol
        assert abs(5 * x_val + 3 * y_val - cost) < tol


def test_incumbent_get_solutions():
    _run_incumbent_solutions(include_set_callback=False)


def test_incumbent_get_set_solutions():
    _run_incumbent_solutions(include_set_callback=True)


def test_warm_start():
    file_path = RAPIDS_DATASET_ROOT_DIR + "/linear_programming/a2864/a2864.mps"
    problem = Problem.readMPS(file_path)

    settings = SolverSettings()
    settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Stable2)
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    # warm start works only with presolve disabled
    settings.set_parameter(CUOPT_PRESOLVE, 0)
    settings.set_optimality_tolerance(1e-3)
    settings.set_parameter(CUOPT_INFEASIBILITY_DETECTION, False)

    problem.solve(settings)
    iterations_first_solve = problem.SolutionStats.nb_iterations

    settings.set_optimality_tolerance(1e-2)
    problem.solve(settings)
    iterations_second_solve = problem.SolutionStats.nb_iterations

    settings.set_optimality_tolerance(1e-3)
    warmstart_data = problem.get_pdlp_warm_start_data()
    settings.set_pdlp_warm_start_data(warmstart_data)
    problem.solve(settings)

    iterations_third_solve = problem.SolutionStats.nb_iterations

    assert (
        iterations_third_solve + iterations_second_solve
        == iterations_first_solve
    )


def test_problem_update():
    prob = Problem()
    x1 = prob.addVariable(vtype=INTEGER, lb=0, name="x1")
    x2 = prob.addVariable(vtype=INTEGER, lb=0, name="x2")

    prob.addConstraint(2 * x1 + x2 <= 7, name="c1")
    prob.addConstraint(x1 + x2 <= 5, name="c2")

    prob.setObjective(4 * x1 + 5 * x2 + 4 - 4 * x2, MAXIMIZE)
    prob.solve()

    assert prob.ObjValue == pytest.approx(17)

    prob.updateObjective(coeffs=[(x1, 1.0), (x2, 3.0)])
    prob.solve()
    assert prob.ObjValue == pytest.approx(19)

    c1 = prob.getConstraint("c1")
    c2 = prob.getConstraint(1)

    prob.updateConstraint(c1, coeffs=[(x1, 1)], rhs=10)
    prob.updateConstraint(c2, rhs=10)
    prob.solve()
    assert prob.ObjValue == pytest.approx(34)

    assert prob.getVariable("x1").Value == pytest.approx(0)
    assert prob.getVariable("x2").Value == pytest.approx(10)

    prob.updateObjective(constant=5, sense=MINIMIZE)
    prob.solve()
    assert prob.ObjValue == pytest.approx(5)


@pytest.mark.parametrize(
    "test_name,settings_config",
    [
        (
            "automatic",
            {
                CUOPT_FOLDING: -1,
                CUOPT_DUALIZE: -1,
                CUOPT_ORDERING: -1,
                CUOPT_AUGMENTED: -1,
            },
        ),
        (
            "forced_on",
            {
                CUOPT_FOLDING: 1,
                CUOPT_DUALIZE: 1,
                CUOPT_ORDERING: 1,
                CUOPT_AUGMENTED: 1,
                CUOPT_ELIMINATE_DENSE_COLUMNS: True,
                CUOPT_CUDSS_DETERMINISTIC: True,
            },
        ),
        (
            "disabled",
            {
                CUOPT_FOLDING: 0,
                CUOPT_DUALIZE: 0,
                CUOPT_ORDERING: 0,
                CUOPT_AUGMENTED: 0,
                CUOPT_ELIMINATE_DENSE_COLUMNS: False,
                CUOPT_CUDSS_DETERMINISTIC: False,
            },
        ),
        (
            "mixed",
            {
                CUOPT_FOLDING: 1,
                CUOPT_DUALIZE: 0,
                CUOPT_ORDERING: -1,
                CUOPT_AUGMENTED: 1,
            },
        ),
        (
            "folding_on",
            {
                CUOPT_FOLDING: 1,
            },
        ),
        (
            "folding_off",
            {
                CUOPT_FOLDING: 0,
            },
        ),
        (
            "dualize_on",
            {
                CUOPT_DUALIZE: 1,
            },
        ),
        (
            "dualize_off",
            {
                CUOPT_DUALIZE: 0,
            },
        ),
        (
            "amd_ordering",
            {
                CUOPT_ORDERING: 1,
            },
        ),
        (
            "cudss_ordering",
            {
                CUOPT_ORDERING: 0,
            },
        ),
        (
            "augmented_system",
            {
                CUOPT_AUGMENTED: 1,
            },
        ),
        (
            "adat_system",
            {
                CUOPT_AUGMENTED: 0,
            },
        ),
        (
            "no_dense_elim",
            {
                CUOPT_ELIMINATE_DENSE_COLUMNS: False,
            },
        ),
        (
            "cudss_deterministic",
            {
                CUOPT_CUDSS_DETERMINISTIC: True,
            },
        ),
        (
            "combo1",
            {
                CUOPT_FOLDING: 1,
                CUOPT_DUALIZE: 1,
                CUOPT_ORDERING: 1,
            },
        ),
        (
            "combo2",
            {
                CUOPT_FOLDING: 0,
                CUOPT_AUGMENTED: 0,
                CUOPT_ELIMINATE_DENSE_COLUMNS: False,
            },
        ),
        (
            "dual_initial_point_automatic",
            {
                CUOPT_BARRIER_DUAL_INITIAL_POINT: -1,
            },
        ),
        (
            "dual_initial_point_lustig",
            {
                CUOPT_BARRIER_DUAL_INITIAL_POINT: 0,
            },
        ),
        (
            "dual_initial_point_least_squares",
            {
                CUOPT_BARRIER_DUAL_INITIAL_POINT: 1,
            },
        ),
        (
            "combo3_with_dual_init",
            {
                CUOPT_AUGMENTED: 1,
                CUOPT_BARRIER_DUAL_INITIAL_POINT: 1,
                CUOPT_ELIMINATE_DENSE_COLUMNS: True,
            },
        ),
    ],
)
def test_barrier_solver_settings(test_name, settings_config):
    """
    Parameterized test for barrier solver with different configurations.

    Tests the barrier solver across various settings combinations to ensure
    correctness and robustness. Each configuration tests different aspects
    of the barrier solver implementation.

    Problem:
        maximize   5*xs + 20*xl
        subject to  1*xs +  3*xl <= 200
                    3*xs +  2*xl <= 160
                    xs, xl >= 0

    Expected Solution:
        Optimal objective: 1333.33
        xs = 0, xl = 66.67 (corner solution where constraint 1 is binding)

    Args
    ----
        test_name: Descriptive name for the test configuration
        settings_config: Dictionary of barrier solver parameters to set
    """
    prob = Problem(f"Barrier Test - {test_name}")

    # Add variables
    xs = prob.addVariable(lb=0, vtype=VType.CONTINUOUS, name="xs")
    xl = prob.addVariable(lb=0, vtype=VType.CONTINUOUS, name="xl")

    # Add constraints
    prob.addConstraint(xs + 3 * xl <= 200, name="constraint1")
    prob.addConstraint(3 * xs + 2 * xl <= 160, name="constraint2")

    # Set objective: maximize 5*xs + 20*xl
    prob.setObjective(5 * xs + 20 * xl, sense=MAXIMIZE)

    # Configure solver settings
    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    settings.set_parameter("time_limit", 10)

    # Apply test-specific settings
    for param_name, param_value in settings_config.items():
        settings.set_parameter(param_name, param_value)

    print(f"\nTesting configuration: {test_name}")
    print(f"Settings: {settings_config}")

    # Solve the problem
    prob.solve(settings)

    print(f"Status: {prob.Status.name}")
    print(f"Objective: {prob.ObjValue}")
    print(f"xs = {xs.Value}, xl = {xl.Value}")

    # Verify solution
    assert prob.solved, f"Problem not solved for {test_name}"
    assert prob.Status.name == "Optimal", f"Not optimal for {test_name}"
    assert prob.ObjValue == pytest.approx(1333.33, rel=0.01), (
        f"Incorrect objective for {test_name}"
    )
    assert xs.Value == pytest.approx(0.0, abs=1e-4), (
        f"Incorrect xs value for {test_name}"
    )
    assert xl.Value == pytest.approx(66.67, rel=0.01), (
        f"Incorrect xl value for {test_name}"
    )

    # Verify constraint slacks are non-negative
    for c in prob.getConstraints():
        assert c.Slack >= -1e-6, (
            f"Negative slack for {c.getConstraintName()} in {test_name}"
        )


def test_quadratic_expression_and_matrix():
    problem = Problem()
    x = problem.addVariable(lb=9.0, vtype="I", name="x")
    y = problem.addVariable(name="y")
    z = problem.addVariable(name="z")

    # Test Quadratic Expressions
    expr1 = x * x  # var * var
    expr2 = expr1 + y * z + 2  # QP + QP + const
    expr3 = expr2 * 4  # QP * const

    assert expr1.getVariables() == [(x, x)]
    assert expr1.getCoefficients() == [1]
    assert expr2.getVariables() == [(x, x), (y, z)]
    assert expr2.getCoefficients() == [1, 1]
    assert expr3.getVariables() == [(x, x), (y, z)]
    assert expr3.getCoefficients() == [4, 4]
    assert expr3.getLinearExpression().getConstant() == 8
    assert expr3.getLinearExpression().getVariables() == []
    assert expr3.getLinearExpression().getCoefficients() == []

    expr3 /= 4  # QP / const
    expr4 = expr3 - y  # QP - var

    assert expr4.getVariables() == [(x, x), (y, z)]
    assert expr4.getCoefficients() == [1, 1]
    assert expr4.getLinearExpression().getConstant() == 2
    assert expr4.getLinearExpression().getVariables() == [y]
    assert expr4.getLinearExpression().getCoefficients() == [-1]

    expr5 = z + 7 * y + 1  # LP
    expr6 = y * expr5  # var * LP
    expr7 = expr5 - x * x  # LP - QP
    expr8 = (
        expr4 - expr5
    )  # QP - LP # x2 + yz + 2 - y - 7y -z - 1 + 7y2 + yz + y
    expr8 += expr6  # QP + QP

    assert expr6.getVariable1(0) is y
    assert expr6.getVariable2(0) is y
    assert expr6.getVariable1(1) is y
    assert expr6.getVariable2(1) is z
    assert expr6.getCoefficients() == [7, 1]
    assert expr6.getLinearExpression().getConstant() == 0
    assert expr6.getLinearExpression().getVariables() == [y]
    assert expr6.getLinearExpression().getCoefficients() == [1]

    assert expr7.getVariable1(0) is x
    assert expr7.getVariable2(0) is x
    assert expr7.getCoefficients() == [-1]
    assert expr7.getLinearExpression().getConstant() == 1
    assert expr7.getLinearExpression().getVariable(0) is y
    assert expr7.getLinearExpression().getVariable(1) is z
    assert expr7.getLinearExpression().getCoefficients() == [7, 1]

    assert len(expr8.getVariables()) == 4
    assert expr8.getCoefficients() == [1, 1, 7, 1]
    assert expr8.getLinearExpression().getConstant() == 1
    assert expr8.getLinearExpression().getCoefficients() == [-1, -7, -1, 1]

    expr9 = expr5 * (3 * x - y + z + 3)  # LP * LP
    # expr9 = 21*y*x + 7*y*z + 21*y - 7*y*y + 3*x + z + 3 - y + 3*z*x + z*z +3*z - z*y

    qvariables = [(y, x), (y, y), (y, z), (z, x), (z, y), (z, z)]
    qcoeffs = [21, -7, 7, 3, -1, 1]
    for i, (var1, var2) in enumerate(expr9.getVariables()):
        assert var1 is qvariables[i][0]
        assert var2 is qvariables[i][1]
    assert expr9.getCoefficients() == qcoeffs

    linexpr = expr9.getLinearExpression()
    lvariables = [y, z, x, y, z]
    lcoeffs = [21, 3, 3, -1, 1]
    constant = 3
    for i, var in enumerate(linexpr.getVariables()):
        assert var is lvariables[i]
    assert linexpr.getCoefficients() == lcoeffs
    assert linexpr.getConstant() == constant

    # Test Quadratic Matrix
    problem.setObjective(expr9)
    Qcsr = problem.getQCSR()

    exp_row_ptrs = [0, 0, 3, 6]
    exp_col_inds = [0, 1, 2, 0, 1, 2]
    exp_vals = [21, -7, 7, 3, -1, 1]

    assert list(Qcsr.row_pointers) == exp_row_ptrs
    assert list(Qcsr.column_indices) == exp_col_inds
    assert list(Qcsr.values) == exp_vals


def test_quadratic_objective_1():
    # Minimize x1 ^2 + 4 x2 ^2 - 8 x1 - 16 x2
    # subject to x1 + x2 >= 5
    #         x1 >= 3
    #         x2 >= 0

    problem = Problem()
    x1 = problem.addVariable(lb=3.0, name="x")
    x2 = problem.addVariable(lb=0, name="y")

    problem.addConstraint(x1 + x2 >= 5)
    problem.setObjective(x1 * x1 + 4 * x2 * x2 - 8 * x1 - 16 * x2)

    problem.solve()

    assert problem.Status.name == "Optimal"
    assert x1.getValue() == pytest.approx(4.0)
    assert x2.getValue() == pytest.approx(2.0)
    assert problem.ObjValue == pytest.approx(-32.0)


def test_quadratic_objective_2():
    # Minimize 4 x1^2 + 2 x2^2 + 3 x3^2 + 1.5 x1 x3 - 2 x1 + 0.5 x2 - x3
    # subject to x1 + 2*x2 + x3 <= 3
    #         x1 >= 0
    #         x2 >= 0
    #         x3 >= 0

    problem = Problem()
    x1 = problem.addVariable(lb=0, name="x")
    x2 = problem.addVariable(lb=0, name="y")
    x3 = problem.addVariable(lb=0, name="z")

    problem.addConstraint(x1 + 2 * x2 + x3 <= 3)
    problem.setObjective(
        2 * x1 * x1
        + 2 * x2 * x2
        + 3 * x3 * x3
        + 1.5 * x1 * x3
        - 2 * x1
        + 0.5 * x2
        - 2 * x1 * x2
        - 1.0 * x3
        + 2 * x1 * x1
        + 2 * x1 * x2
    )

    problem.solve()

    assert problem.Status.name == "Optimal"
    assert x1.getValue() == pytest.approx(0.2295081, abs=1e-3)
    assert x2.getValue() == pytest.approx(0.0000000, abs=0.000001)
    assert x3.getValue() == pytest.approx(0.1092896, abs=1e-3)
    assert problem.ObjValue == pytest.approx(-0.284153, abs=1e-3)


def test_quadratic_matrix_1():
    problem = Problem()
    x1 = problem.addVariable(lb=1, name="x1")
    x2 = problem.addVariable(lb=1, name="x2")
    x3 = problem.addVariable(lb=2.0, name="x3")
    x4 = problem.addVariable(lb=1, name="x4")

    # Constraints
    problem.addConstraint(x1 + x2 + x3 + x4 <= 10, "c1")
    problem.addConstraint(2 * x1 - x2 + x4 >= 5, "c2")

    # Quadratic objective
    # Minimize 2 x1^2 + 3 x2^2 + x3^2 + 4 x4^2 + 1.5 x1 x2 - 2 x3 x4 - 4 x1 + x2 + 3 x3 + 5

    quad_matrix = [[2, 1.5, 0, 0], [0, 3, 0, 0], [0, 0, 1, -2], [0, 0, 0, 4]]
    lin_terms = x2 + 3 * x3 - 4 * x1 + 5
    quad_expr = (
        2 * x1 * x1
        + 3 * x2 * x2
        + 1 * x3 * x3
        + 4 * x4 * x4
        + 1.5 * x1 * x2
        - 2 * x3 * x4
        - 4 * x1
        + 1 * x2
        + 3 * x3
        + 5
    )

    # Break down obj into multiple expressions
    lin_mix_1 = x2 + 3 * x3
    lin_mix_2 = 4 * x1
    quad_mix_1 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
    quad_mix_2 = 3 * x2 * x2
    quad_mix_3 = [[1, 1.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]
    quad_mix_4 = 2 * x3 * x4 - 5

    # Expected Solution
    obj_value_exp = 25.25
    x1_exp = 2.5
    x2_exp = 1
    x3_exp = 2
    x4_exp = 1

    # Solve 1
    problem.setObjective(quad_expr)
    problem.solve()
    assert problem.ObjValue == pytest.approx(obj_value_exp, abs=1e-3)

    # Solve 2
    quad_obj = QuadraticExpression(quad_matrix, problem.getVariables())
    problem.setObjective(quad_obj + lin_terms)
    problem.solve()
    assert problem.ObjValue == pytest.approx(obj_value_exp, abs=1e-3)

    # Solve 3
    vars = problem.getVariables()
    qmatrix1 = QuadraticExpression(quad_mix_1, vars)
    qmatrix3 = QuadraticExpression(quad_mix_3, vars)
    quad_obj = (
        lin_mix_1 + qmatrix1 + quad_mix_2 + qmatrix3 - quad_mix_4 - lin_mix_2
    )
    problem.setObjective(quad_obj)
    problem.solve()
    assert problem.ObjValue == pytest.approx(obj_value_exp, abs=1e-3)

    # Verify accessor functions
    q_vars = quad_obj.getVariables()
    q_coeffs = quad_obj.getCoefficients()
    lin_expr = quad_obj.getLinearExpression()
    obj_value = 0.0
    for i, (var1, var2) in enumerate(q_vars):
        obj_value += var1.Value * var2.Value * q_coeffs[i]
    obj_value += lin_expr.getValue()
    assert obj_value == pytest.approx(obj_value_exp, abs=1e-3)
    assert quad_obj.getValue() == pytest.approx(obj_value_exp, abs=1e-3)
    assert x1.Value == pytest.approx(x1_exp, abs=1e-3)
    assert x2.Value == pytest.approx(x2_exp, abs=1e-3)
    assert x3.Value == pytest.approx(x3_exp, abs=1e-3)
    assert x4.Value == pytest.approx(x4_exp, abs=1e-3)


def test_quadratic_matrix_2():
    # Minimize 4 x1^2 + 2 x2^2 + 3 x3^2 + 1.5 x1 x3 - 2 x1 + 0.5 x2 - x3 + 4
    # subject to x1 + 2*x2 + x3 <= 3
    #         x1 >= 0
    #         x2 >= 0
    #         x3 >= 0

    problem = Problem()
    x1 = problem.addVariable(lb=0, name="x")
    x2 = problem.addVariable(lb=0, name="y")
    x3 = problem.addVariable(lb=0, name="z")

    problem.addConstraint(x1 + 2 * x2 + x3 <= 3)

    Q = [[4, 0, 1.5], [0, 2, 0], [0, 0, 3]]
    quad_expr = QuadraticExpression(qmatrix=Q, qvars=problem.getVariables())
    quad_expr1 = quad_expr + 4  # Quad_matrix add constant
    quad_expr2 = quad_expr1 - x3  # Quad_matrix sub variable
    quad_expr2 -= 2 * x1  # Quad_matrix isub lin_expr
    quad_expr2 += 0.5 * x2  # Quad_matrix iadd lin_expr

    problem.setObjective(quad_expr2)

    problem.solve()

    assert problem.Status.name == "Optimal"
    assert x1.getValue() == pytest.approx(0.2295081, abs=1e-3)
    assert x2.getValue() == pytest.approx(0.0000000, abs=1e-3)
    assert x3.getValue() == pytest.approx(0.1092896, abs=1e-3)
    assert problem.ObjValue == pytest.approx(3.715847, abs=1e-3)
