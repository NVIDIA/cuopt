# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
# SPDX-License-Identifier: Apache-2.0
# AFFILIATES. All rights reserved.
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
"""
Simple Linear Programming Example

This example demonstrates how to:
- Create a linear programming problem
- Add continuous variables
- Add linear constraints
- Set an objective function
- Solve the problem and retrieve results

Problem:
    Maximize: x + y
    Subject to:
        x + y <= 10
        x - y >= 0
        x, y >= 0

Expected Output:
    Optimal solution found in 0.01 seconds
    x = 10.0
    y = 0.0
    Objective value = 10.0
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    """Run the simple LP example."""
    # Create a new problem
    problem = Problem("Simple LP")

    # Add variables
    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    # Add constraints
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - y >= 0, name="c2")

    # Set objective function
    problem.setObjective(x + y, sense=MAXIMIZE)

    # Configure solver settings
    settings = SolverSettings()
    settings.set_parameter("time_limit", 60)

    # Solve the problem
    problem.solve(settings)

    # Check solution status
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {problem.ObjValue}")
    else:
        print(f"Problem status: {problem.Status.name}")


if __name__ == "__main__":
    main()
