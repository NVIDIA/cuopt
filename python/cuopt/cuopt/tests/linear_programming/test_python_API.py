# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import os

import numpy as np
import pytest

from cuopt.linear_programming.problem import Problem
from cuopt.linear_programming.problem import sense, vtype, ctype


def test_model():

    prob = Problem("Simple MIP")
    assert prob.Name == "Simple MIP"

    # Adding Variable
    x = prob.addVariable(lb=0, vtype=vtype.INTEGER, name="V_x")
    y = prob.addVariable(lb=10, ub=50, vtype=vtype.INTEGER, name="V_y")

    assert x.getVariableName() == "V_x"
    assert y.getUpperBound() == 50
    assert y.getLowerBound() == 10
    assert x.getVariableType() == vtype.INTEGER
    assert y.getVariableType() == "I"

    # Adding Constraints
    prob.addConstraint(2*x + 4*y >= 230, name="C1")
    prob.addConstraint(3*x + 2*y <= 190, name="C2")

    expected_name = ["C1", "C2"]
    expected_coefficient_x = [2, 3]
    expected_coefficient_y = [4, 2]
    expected_sense = [ctype.GE, "L"]
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
    expr = 5*x + 3*y
    prob.setObjective(expr, sense=sense.MAXIMIZE)

    expected_obj_coeff = [5, 3]
    assert expr.getVariables() == [x, y]
    assert expr.getCoefficients() == expected_obj_coeff
    assert prob.ObjSense == sense.MAXIMIZE
    assert prob.getObjective() is expr

    # Adding Settings
    prob.Settings.set_parameter("time_limit", 60)

    # Solving Problem
    prob.solve()
    assert prob.Status.name == "Optimal"

    csr = prob.getCSR()
    expected_row_pointers = [0, 2, 4]
    expected_column_indices = [0, 1, 0, 1]
    expected_values = [2.0, 4.0, 3.0, 2.0]

    assert csr.row_pointers == expected_row_pointers
    assert csr.column_indices ==  expected_column_indices
    assert csr.values == expected_values

    expected_slack = [-6, 0]
    expected_var_values = [36, 41]

    for i, var in enumerate(prob.getVariables()):
        assert var.Value == pytest.approx(expected_var_values[i])
        assert var.getObjectiveCoefficient() == expected_obj_coeff[i]

    assert prob.ObjVal == 303

    for i, c in enumerate(prob.getConstraints()):
        assert c.Slack == pytest.approx(expected_slack[i])
