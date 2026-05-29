# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Working with Incumbent Solutions Example

This example demonstrates:
- Using callbacks to receive intermediate solutions during MIP solving
- Using Problem.getIncumbentValues() to extract variable values from
  incumbent solutions
- Tracking solution progress as the solver improves the solution
- Accessing incumbent (best so far) solutions before final optimum
- Custom callback class implementation

Incumbent solutions are intermediate feasible solutions found during the MIP
solving process. They represent the best integer-feasible solution discovered
so far.

Note:
    Incumbent solutions are only available for Mixed Integer Programming (MIP)
    problems, not for pure Linear Programming (LP) problems.

Problem:
    Maximize: 5*x + 3*y
    Subject to:
        2*x + 4*y >= 230
        3*x + 2*y <= 190
        x, y are integers

Expected Output:
    Incumbent 1: x=36.0, y=41.0, cost: 303.00

    === Final Results ===
    Problem status: Optimal
    Solve time: 0.27 seconds
    Final solution: x=36.0, y=41.0
    Final objective value: 303.00
"""

from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT
from cuopt.linear_programming.internals import GetSolutionCallback


class IncumbentCallback(GetSolutionCallback):
    """Callback to receive and track incumbent solutions during solving.

    Uses Problem.getIncumbentValues() to extract variable values from the
    raw incumbent solution array.
    """

    def __init__(self, problem, variables, user_data):
        super().__init__()
        self.problem = problem
        self.variables = variables
        self.solutions = []
        self.n_callbacks = 0
        self.user_data = user_data

    def get_solution(self, solution, solution_cost, solution_bound, user_data):
        """Called whenever the solver finds a new incumbent solution."""
        assert user_data is self.user_data
        self.n_callbacks += 1

        # Use getIncumbentValues to extract values for specific variables
        values = self.problem.getIncumbentValues(solution, self.variables)

        incumbent = {
            "values": values,
            "cost": float(solution_cost[0]),
            "bound": float(solution_bound[0]),
            "iteration": self.n_callbacks,
        }
        self.solutions.append(incumbent)

        print(f"Incumbent {self.n_callbacks}:", end=" ")
        for i, var in enumerate(self.variables):
            print(f"{var.VariableName}={values[i]}", end=" ")
        print(f"cost: {incumbent['cost']:.2f}")


def main():
    """Run the incumbent solutions example."""
    problem = Problem("Incumbent Example")

    # Add integer variables
    x = problem.addVariable(vtype=INTEGER, name="x")
    y = problem.addVariable(vtype=INTEGER, name="y")

    # Add constraints
    problem.addConstraint(2 * x + 4 * y >= 230)
    problem.addConstraint(3 * x + 2 * y <= 190)

    # Set objective to maximize
    problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

    # Configure solver settings with callback
    settings = SolverSettings()
    user_data = {"source": "incumbent_solutions_example"}
    incumbent_callback = IncumbentCallback(problem, [x, y], user_data)
    settings.set_mip_callback(incumbent_callback, user_data)
    settings.set_parameter(CUOPT_TIME_LIMIT, 30)

    # Solve the problem
    problem.solve(settings)

    # Display final results
    print("\n=== Final Results ===")
    print(f"Problem status: {problem.Status.name}")
    print(f"Solve time: {problem.SolveTime:.2f} seconds")
    print("Final solution: ", end=" ")
    for i, var in enumerate(problem.getVariables()):
        print(f"{var.VariableName}={var.getValue()} ", end=" ")
    print(f"\nFinal objective value: {problem.ObjValue:.2f}")

    print(
        f"\nTotal incumbent solutions found: "
        f"{len(incumbent_callback.solutions)}"
    )


if __name__ == "__main__":
    main()
