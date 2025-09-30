/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "relaxed_lp.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <linear_programming/solve.cuh>
#include <mip/mip_constants.hpp>
#include <mip/utils.cuh>

#include <linear_programming/pdlp.cuh>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/linalg/binary_op.cuh>

#include <thrust/tabulate.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  solution_t<i_t, f_t>& solution,
  const relaxed_lp_settings_t& settings)
{
  return get_relaxed_lp_solution(op_problem, solution.assignment, solution.lp_state, settings);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<f_t>& assignment,
  lp_state_t<i_t, f_t>& lp_state,
  const relaxed_lp_settings_t& settings)
{
  raft::common::nvtx::range fun_scope("get_relaxed_lp_solution");
  pdlp_solver_settings_t<i_t, f_t> pdlp_settings{};
  pdlp_settings.detect_infeasibility = settings.check_infeasibility;
  pdlp_settings.set_optimality_tolerance(settings.tolerance);
  pdlp_settings.tolerances.relative_primal_tolerance = settings.tolerance / 100.;
  pdlp_settings.tolerances.relative_dual_tolerance   = settings.tolerance / 100.;
  pdlp_settings.time_limit                           = settings.time_limit;
  pdlp_settings.concurrent_halt                      = settings.concurrent_halt;
  pdlp_settings.per_constraint_residual              = settings.per_constraint_residual;
  pdlp_settings.first_primal_feasible                = settings.return_first_feasible;
  pdlp_settings.pdlp_solver_mode                     = pdlp_solver_mode_t::Stable2;
  set_pdlp_solver_mode(pdlp_settings);
  // TODO: set Stable3 here?
  pdlp_solver_t<i_t, f_t> lp_solver(op_problem, pdlp_settings);
  if (settings.has_initial_primal) {
    i_t prev_size = lp_state.prev_dual.size();
    CUOPT_LOG_DEBUG(
      "setting initial primal solution of size %d dual size %d problem vars %d cstrs %d",
      assignment.size(),
      lp_state.prev_dual.size(),
      op_problem.n_variables,
      op_problem.n_constraints);
    lp_state.resize(op_problem, op_problem.handle_ptr->get_stream());
    clamp_within_var_bounds(assignment, &op_problem, op_problem.handle_ptr);
    // The previous dual sometimes contain invalid values w.r.t current problem
    // Adjust better dual values when we use warm start
    thrust::tabulate(op_problem.handle_ptr->get_thrust_policy(),
                     lp_state.prev_dual.data(),
                     lp_state.prev_dual.data() + op_problem.n_constraints,
                     [prev_size, dual = make_span(lp_state.prev_dual)] __device__(i_t i) {
                       f_t x = dual[i];
                       if (!isfinite(x) || i >= prev_size) { return 0.0; }
                       return x;
                     });
    lp_solver.set_initial_primal_solution(assignment);
    lp_solver.set_initial_dual_solution(lp_state.prev_dual);
  }
  CUOPT_LOG_DEBUG(
    "running LP with n_vars %d n_cstr %d", op_problem.n_variables, op_problem.n_constraints);
  // before LP flush the logs as it takes quite some time
  cuopt::default_logger().flush();
  // temporarily add timer
  auto start_time = timer_t(pdlp_settings.time_limit);
  lp_solver.set_inside_mip(true);
  auto solver_response = lp_solver.run_solver(start_time);

  if (solver_response.get_primal_solution().size() != 0 &&
      solver_response.get_dual_solution().size() != 0 && settings.save_state) {
    CUOPT_LOG_DEBUG("saving initial primal solution of size %d", lp_state.prev_primal.size());
    lp_state.set_state(solver_response.get_primal_solution(), solver_response.get_dual_solution());
  }
  if (solver_response.get_primal_solution().size() != 0) {
    // copy the solution no matter what, because in the worst case we are closer to the polytope
    raft::copy(assignment.data(),
               solver_response.get_primal_solution().data(),
               solver_response.get_primal_solution().size(),
               op_problem.handle_ptr->get_stream());
  }
  if (solver_response.get_termination_status() == pdlp_termination_status_t::Optimal) {
    CUOPT_LOG_DEBUG("feasible solution found with LP objective %f",
                    solver_response.get_objective_value());
  } else {
    CUOPT_LOG_DEBUG("LP returned with reason %d", solver_response.get_termination_status());
  }

  return solver_response;
}

// Run LP with variables fixed to specific values
template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            problem_t<i_t, f_t>& fixed_problem,
                            solution_t<i_t, f_t>& solution,
                            rmm::device_uvector<f_t>& fixed_assignment,
                            rmm::device_uvector<i_t>& variable_map,
                            relaxed_lp_settings_t& settings,
                            bound_presolve_t<i_t, f_t>* bound_presolve,
                            bool check_fixed_assignment_feasibility)
{
  if (check_fixed_assignment_feasibility) {
    solution_t<i_t, f_t> temp_solution(fixed_problem);
    raft::copy(temp_solution.assignment.data(),
               fixed_assignment.data(),
               fixed_assignment.size(),
               fixed_problem.handle_ptr->get_stream());
    bool temp_solution_feasible = temp_solution.compute_feasibility();
    if (!temp_solution_feasible) {
      CUOPT_LOG_DEBUG(
        "Infeasible solution detected with fixed vars LP. Sol excess %f fixed sol excess %f",
        solution.get_total_excess(),
        temp_solution.get_total_excess());
      settings.time_limit = 1;
    }
  }
  if (bound_presolve != nullptr) {
    bound_presolve->resize(fixed_problem);
    // run bounds prop to quickly discover inevitably infeasible
    bound_presolve->settings.time_limit = (settings.time_limit / 10);
    auto term_crit                      = bound_presolve->solve(fixed_problem);
    bound_presolve->settings            = {};
    if (bound_presolve->infeas_constraints_count > 0) {
      solution.unfix_variables(fixed_assignment, variable_map);
      bound_presolve->resize(op_problem);
      CUOPT_LOG_DEBUG("Infeasible problem detected with LP with fixed vars");
      return true;
    }
  }
  fixed_problem.check_problem_representation(true);
  // if we are on the original problem and fixing the integers, save the state
  // if we are in recombiners and on a smaller problem, don't update the state with integers fixed
  CUOPT_LOG_TRACE("save_state %d", settings.save_state);
  auto& lp_state = fixed_problem.lp_state;
  auto solver_response =
    get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, settings);
  // unfix the assignment on given result no matter if it is feasible
  solution.unfix_variables(fixed_assignment, variable_map);
  if (bound_presolve != nullptr) { bound_presolve->resize(op_problem); }
  return false;
}

// returns true if the problem is inevitably infeasible
template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            solution_t<i_t, f_t>& solution,
                            const rmm::device_uvector<i_t>& variables_to_fix,
                            relaxed_lp_settings_t& settings,
                            bound_presolve_t<i_t, f_t>* bound_presolve,
                            bool check_fixed_assignment_feasibility,
                            bool use_integer_fixed_problem)
{
  // if we are fixing all vars, there is no lp to be run
  if (variables_to_fix.size() == (size_t)op_problem.n_variables) { return true; }
  if (use_integer_fixed_problem) {
    op_problem.fill_integer_fixed_problem(solution.assignment, op_problem.handle_ptr);
    auto fixed_assignment =
      op_problem.get_fixed_assignment_from_integer_fixed_problem(solution.assignment);
    return run_lp_with_vars_fixed(op_problem,
                                  *op_problem.integer_fixed_problem,
                                  solution,
                                  fixed_assignment,
                                  op_problem.integer_fixed_variable_map,
                                  settings,
                                  bound_presolve,
                                  check_fixed_assignment_feasibility);
  } else {
    auto [fixed_problem, fixed_assignment, variable_map] = solution.fix_variables(variables_to_fix);
    return run_lp_with_vars_fixed(op_problem,
                                  fixed_problem,
                                  solution,
                                  fixed_assignment,
                                  variable_map,
                                  settings,
                                  bound_presolve,
                                  check_fixed_assignment_feasibility);
  }
}

#define INSTANTIATE(F_TYPE)                                                                   \
  template optimization_problem_solution_t<int, F_TYPE> get_relaxed_lp_solution<int, F_TYPE>( \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    solution_t<int, F_TYPE> & solution,                                                       \
    const relaxed_lp_settings_t& settings);                                                   \
  template optimization_problem_solution_t<int, F_TYPE> get_relaxed_lp_solution<int, F_TYPE>( \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    rmm::device_uvector<F_TYPE> & assignment,                                                 \
    lp_state_t<int, F_TYPE> & lp_state,                                                       \
    const relaxed_lp_settings_t& settings);                                                   \
  template bool run_lp_with_vars_fixed<int, F_TYPE>(                                          \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    solution_t<int, F_TYPE> & solution,                                                       \
    const rmm::device_uvector<int>& variables_to_fix,                                         \
    relaxed_lp_settings_t& settings,                                                          \
    bound_presolve_t<int, F_TYPE>* bound_presolve,                                            \
    bool check_fixed_assignment_feasibility,                                                  \
    bool use_integer_fixed_problem);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
