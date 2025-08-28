# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

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
    CType,
    Problem,
    VType,
    sense,
)


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
    assert prob.getObjective() is expr

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


def test_incumbent_solutions():

    # Callback for incumbent solution
    class CustomGetSolutionCallback(GetSolutionCallback):
        def __init__(self):
            super().__init__()
            self.n_callbacks = 0
            self.solutions = []

        def get_solution(self, solution, solution_cost):

            self.n_callbacks += 1
            assert len(solution) > 0
            assert len(solution_cost) == 1

            self.solutions.append(
                {
                    "solution": solution.copy_to_host(),
                    "cost": solution_cost.copy_to_host()[0],
                }
            )

    class CustomSetSolutionCallback(SetSolutionCallback):
        def __init__(self, get_callback):
            super().__init__()
            self.n_callbacks = 0
            self.get_callback = get_callback

        def set_solution(self, solution, solution_cost):
            self.n_callbacks += 1
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

    get_callback = CustomGetSolutionCallback()
    set_callback = CustomSetSolutionCallback(get_callback)
    settings = SolverSettings()
    settings.set_mip_callback(get_callback)
    settings.set_mip_callback(set_callback)
    settings.set_parameter("time_limit", 1)

    prob.solve(settings)

    assert get_callback.n_callbacks > 0

    for sol in get_callback.solutions:
        x_val = sol["solution"][0]
        y_val = sol["solution"][1]
        cost = sol["cost"]
        assert 2 * x_val + 4 * y_val >= 230
        assert 3 * x_val + 2 * y_val <= 190
        assert 5 * x_val + 3 * y_val == cost
