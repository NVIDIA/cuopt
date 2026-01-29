# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Working with Incumbent Solutions Example

This example demonstrates:
- Using callbacks to receive intermediate solutions during MIP solving
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
    Incumbent 1: [ 0. 58.], cost: 174.00
    Incumbent 2: [36. 41.], cost: 303.00

    === Final Results ===
    Problem status: Optimal
    Solve time: 0.16 seconds
    Final solution: x=36.0, y=41.0
    Final objective value: 303.00
"""

from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT
from cuopt.linear_programming.internals import GetSolutionCallback


# Create a callback class to receive incumbent solutions
class IncumbentCallback(GetSolutionCallback):
    """Callback to receive and track incumbent solutions during solving."""

    def __init__(self):
        super().__init__()
        self.solutions = []
        self.n_callbacks = 0

    def get_solution(self, solution, solution_cost):
        """
        Called whenever the solver finds a new incumbent solution.

        Parameters
        ----------
        solution : array-like
            The variable values of the incumbent solution
        solution_cost : array-like
            The objective value of the incumbent solution
        """
        self.n_callbacks += 1

        # Store the incumbent solution
        incumbent = {
            "solution": solution.copy_to_host(),
            "cost": solution_cost.copy_to_host()[0],
            "iteration": self.n_callbacks,
        }
        self.solutions.append(incumbent)

        print(
            f"Incumbent {self.n_callbacks}: {incumbent['solution']}, "
            f"cost: {incumbent['cost']:.2f}"
        )


def main():
    """Run the incumbent solutions example."""
    # Create a more complex MIP problem that will generate multiple incumbents
    problem = Problem("Incumbent Example")

    # Add integer variables
    x = problem.addVariable(vtype=INTEGER)
    y = problem.addVariable(vtype=INTEGER)

    # Add constraints to create a problem that will generate multiple
    # incumbents
    problem.addConstraint(2 * x + 4 * y >= 230)
    problem.addConstraint(3 * x + 2 * y <= 190)

    # Set objective to maximize
    problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

    # Configure solver settings with callback
    settings = SolverSettings()
    # Set the incumbent callback
    incumbent_callback = IncumbentCallback()
    settings.set_mip_callback(incumbent_callback)
    # Allow enough time to find multiple incumbents
    settings.set_parameter(CUOPT_TIME_LIMIT, 30)

    # Solve the problem
    problem.solve(settings)

    # Display final results
    print("\n=== Final Results ===")
    print(f"Problem status: {problem.Status.name}")
    print(f"Solve time: {problem.SolveTime:.2f} seconds")
    print(f"Final solution: x={x.getValue()}, y={y.getValue()}")
    print(f"Final objective value: {problem.ObjValue:.2f}")

    # Display all incumbents found
    print(
        f"\nTotal incumbent solutions found: "
        f"{len(incumbent_callback.solutions)}"
    )


if __name__ == "__main__":
    main()
