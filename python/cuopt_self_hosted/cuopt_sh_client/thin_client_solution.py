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


class ThinClientSolution:
    """
    A container of LP solver output

    Parameters
    ----------
    problem_category : str
        Whether it is a LP, MIP or IP solution
    vars : Dict[str, float64]
        Dictionary mapping each variable (name) to its value.
    primal_solution : numpy.array
        Primal solution of the LP problem
    dual_solution : numpy.array
        Note: Applicable to only LP
        Dual solution of the LP problem
    reduced_cost : numpy.array
        Note: Applicable to only LP
        The reduced cost.
        It contains the dual multipliers for the linear constraints.
    termination_status: str
        Termination status .
    primal_residual: Float64
        L2 norm of the primal residual: measurement of the primal infeasibility
    dual_residual: Float64
        Note: Applicable to only LP
        L2 norm of the dual residual: measurement of the dual infeasibility
    primal_objective: Float64
        Value of the primal objective
    dual_objective: Float64
        Note: Applicable to only LP
        Value of the dual objective
    gap: Float64
        Difference between the primal and dual objective
    nb_iterations: Int
        Number of iterations the LP solver did before converging
    mip_gap: float64
        Note: Applicable to only MILP
        The relative difference between the best integer objective value
        found so far and the objective bound. A value of 0.01 means the
        solution is guaranteed to be within 1% of optimal.
    solution_bound: float64
        Note: Applicable to only MILP
        The best known bound on the optimal objective value.
        For minimization problems, this is a lower bound on the optimal value.
        For maximization problems, this is an upper bound.
    max_constraint_violation: float64
        Note: Applicable to only MILP
        The maximum amount by which any constraint is violated in
        the current solution. Should be close to zero for a feasible solution.
    max_int_violation: float64
        Note: Applicable to only MILP
        The maximum amount by which any integer variable deviates from being
        an integer. A value of 0 means all integer variables have
        integral values.
    max_variable_bound_violation: float64
        Note: Applicable to only MILP
        The maximum amount by which any variable violates its upper or
        lower bounds in the current solution. Should be zero for a
        feasible solution.
    presolve_time: float64
        Note: Applicable to only MILP
        Time used for pre-solve
    solve_time: Float64
        Solve time in seconds
    solved_by_pdlp: bool
        Whether the problem was solved by PDLP or Dual Simplex
    """

    def __init__(
        self,
        problem_category,
        vars,
        solve_time=0.0,
        primal_solution=None,
        dual_solution=None,
        reduced_cost=None,
        termination_status="Error",
        error_status=0,
        error_message="",
        primal_residual=0.0,
        dual_residual=0.0,
        primal_objective=0.0,
        dual_objective=0.0,
        gap=0.0,
        nb_iterations=0,
        solved_by_pdlp=None,
        mip_gap=0.0,
        solution_bound=0.0,
        presolve_time=0.0,
        max_constraint_violation=0.0,
        max_int_violation=0.0,
        max_variable_bound_violation=0.0,
        num_nodes=0,
        num_simplex_iterations=0,
    ):
        self.problem_category = problem_category
        self.primal_solution = primal_solution
        self.dual_solution = dual_solution
        self.termination_status = termination_status
        self.error_status = error_status
        self.error_message = error_message

        self.primal_objective = primal_objective
        self.dual_objective = dual_objective
        self.solve_time = solve_time
        self.solved_by_pdlp = solved_by_pdlp
        self.vars = vars
        self.lp_stats = {
            "primal_residual": primal_residual,
            "dual_residual": dual_residual,
            "gap": gap,
            "nb_iterations": nb_iterations,
        }
        self.reduced_cost = reduced_cost
        self.milp_stats = {
            "mip_gap": mip_gap,
            "solution_bound": solution_bound,
            "presolve_time": presolve_time,
            "max_constraint_violation": max_constraint_violation,
            "max_int_violation": max_int_violation,
            "max_variable_bound_violation": max_variable_bound_violation,
            "num_nodes": num_nodes,
            "num_simplex_iterations": num_simplex_iterations,
        }

    def raise_if_milp_solution(self, function_name):
        if self.problem_category in ("MIP", "IP"):
            raise AttributeError(
                f"Attribute {function_name} is not supported for milp solution"
            )

    def raise_if_lp_solution(self, function_name):
        if self.problem_category == "LP":
            raise AttributeError(
                f"Attribute {function_name} is not supported for lp solution"
            )

    def get_primal_solution(self):
        """
        Returns the primal solution as numpy.array with float64 type.
        """
        return self.primal_solution

    def get_dual_solution(self):
        """
        Note: Applicable to only LP
        Returns the dual solution as numpy.array with float64 type.
        """
        self.raise_if_milp_solution(__name__)
        return self.dual_solution

    def get_primal_objective(self):
        """
        Returns the primal objective as a float64.
        """
        return self.primal_objective

    def get_dual_objective(self):
        """
        Note: Applicable to only LP
        Returns the dual objective as a float64.
        """
        self.raise_if_milp_solution(__name__)
        return self.dual_objective

    def get_termination_status(self):
        """
        Returns the termination status.
        """
        return self.termination_status

    def get_error_status(self):
        """
        Returns the error status as per ErrorStatus.
        """
        return self.error_status

    def get_error_message(self):
        """
        Returns the error message as per ErrorMessage.
        """
        return self.error_message

    def get_solve_time(self):
        """
        Returns the engine solve time in seconds as a float64.
        """
        return self.solve_time

    def get_solved_by_pdlp(self):
        """
        Returns whether the problem was solved by PDLP or Dual Simplex
        """
        return self.solved_by_pdlp

    def get_vars(self):
        """
        Returns the dictionnary mapping each variable (name) to its value.
        """
        return self.vars

    def get_lp_stats(self):
        """
        Note: Applicable to only LP
        Returns the convergence statistics as a dictionary:

        "primal_residual": float64
          Measurement of the primal infeasibility.
          This quantity is being reduced until primal tolerance is met
          (see SolverSettings primal_tolerance).

        "dual_residual": float64,
          Measurement of the dual infeasibility.
          This quantity is being reduced until dual tolerance is met
          (see SolverSettings dual_tolerance).

        "gap": float64
          Difference between the primal and dual objective.
          This quantity is being reduced until gap tolerance is met
          (see SolverSettings gap_tolerance).

        - "nb_iterations": int
            Number of iterations the LP solver did before converging.
        """

        self.raise_if_milp_solution(__name__)

        return self.lp_stats

    def get_reduced_cost(self):
        """
        Returns the reduced cost as numpy.array with float64 type.
        """
        return self.reduced_cost

    def get_milp_stats(self):
        """
        Note: Applicable to only MILP
        Returns the convergence statistics as a dictionary:

        mip_gap: float64
            The relative difference between the best integer objective value
            found so far and the objective bound. A value of 0.01 means the
            solution is guaranteed to be within 1% of optimal.

        presolve_time: float64
            Time took for pre-solve

        max_constraint_violation: float64
            The maximum amount by which any constraint is violated
            in the current solution.
            Should be close to zero for a feasible solution
            .
        max_int_violation: float64
            The maximum amount by which any integer variable deviates
            from being an integer. A value of 0 means all integer variables
            have integral values.

        max_variable_bound_violation: float64
            The maximum amount by which any variable violates
            its upper or lower bounds in the current solution.
            Should be zero for a feasible solution.

        solution_bound: float64
            The best known bound on the optimal objective value.
            For minimization problems, this is a lower bound on the optimal
            value.
            For maximization problems, this is an upper bound.

        num_nodes: int
            Number of nodes explored during the MIP solve

        num_simplex_iterations: int
            Number of simplex iterations performed during the MIP solve
        """

        self.raise_if_lp_solution(__name__)

        return self.milp_stats

    def get_problem_category(self):
        """
        Returns one of the problem category from ProblemCategory

        LP
        MIP
        IP
        """

        return self.problem_category
