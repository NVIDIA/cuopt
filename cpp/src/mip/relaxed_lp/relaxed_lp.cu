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
#include <mip/mip_constants.hpp>
#include <mip/utils.cuh>

#include <linear_programming/pdlp.cuh>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/linalg/binary_op.cuh>

#include <thrust/tabulate.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(problem_t<i_t, f_t>& op_problem,
                                                                  solution_t<i_t, f_t>& solution,
                                                                  f_t tolerance,
                                                                  f_t time_limit,
                                                                  bool check_infeasibility,
                                                                  bool return_first_feasible)
{
  return get_relaxed_lp_solution(op_problem,
                                 solution.assignment,
                                 solution.lp_state,
                                 tolerance,
                                 time_limit,
                                 check_infeasibility,
                                 return_first_feasible);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<f_t>& assignment,
  lp_state_t<i_t, f_t>& lp_state,
  f_t tolerance,
  f_t time_limit,
  bool check_infeasibility,
  bool return_first_feasible,
  bool save_state)
{
  raft::common::nvtx::range fun_scope("get_relaxed_lp_solution");
  pdlp_solver_settings_t<i_t, f_t> settings{};
  settings.detect_infeasibility = check_infeasibility;
  settings.set_optimality_tolerance(tolerance);
  settings.tolerances.relative_primal_tolerance = tolerance / 100.;
  settings.tolerances.relative_dual_tolerance   = tolerance / 100.;
  settings.time_limit                           = time_limit;
  if (return_first_feasible) { settings.per_constraint_residual = true; }
  // settings.set_save_best_primal_so_far(true);
  // currently disable first primal setting as it is not supported without per constraint mode
  settings.first_primal_feasible = return_first_feasible;
  pdlp_solver_t<i_t, f_t> lp_solver(op_problem, settings);
  if (save_state) {
    i_t prev_size = lp_state.prev_dual.size();
    CUOPT_LOG_DEBUG(
      "setting initial primal solution of size %d dual size %d problem vars %d cstrs %d",
      lp_state.prev_primal.size(),
      lp_state.prev_dual.size(),
      op_problem.n_variables,
      op_problem.n_constraints);
    lp_state.resize(op_problem, op_problem.handle_ptr->get_stream());
    raft::copy(lp_state.prev_primal.data(),
               assignment.data(),
               assignment.size(),
               op_problem.handle_ptr->get_stream());
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
    lp_solver.set_initial_primal_solution(lp_state.prev_primal);
    lp_solver.set_initial_dual_solution(lp_state.prev_dual);
  }
  CUOPT_LOG_DEBUG(
    "running LP with n_vars %d n_cstr %d", op_problem.n_variables, op_problem.n_constraints);
  // before LP flush the logs as it takes quite some time
  cuopt::default_logger().flush();
  // temporarily add timer
  auto start_time = std::chrono::high_resolution_clock::now();
  lp_solver.set_inside_mip(true);
  auto solver_response = lp_solver.run_solver(start_time);

  if (solver_response.get_primal_solution().size() != 0 &&
      solver_response.get_dual_solution().size() != 0 && save_state) {
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

// returns true if the problem is inevitablyinfeasible
template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            solution_t<i_t, f_t>& solution,
                            const rmm::device_uvector<i_t>& variables_to_fix,
                            typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols,
                            lp_state_t<i_t, f_t>& lp_state,
                            f_t time_limit,
                            bool return_first_feasible,
                            bound_presolve_t<i_t, f_t>* bound_presolve)
{
  // if we are fixing all vars, there is no lp to be run
  if (variables_to_fix.size() == (size_t)op_problem.n_variables) { return true; }
  auto [fixed_problem, fixed_assignment, variable_map] = solution.fix_variables(variables_to_fix);
  if (bound_presolve != nullptr) {
    bound_presolve->resize(fixed_problem);
    // run bounds prop to quickly discover inevitably infeasible
    bound_presolve->settings.time_limit = (time_limit / 10);
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
  const bool check_feas = false;
  // if we are on the original problem and fixing the integers, save the state
  // if we are in recombiners and on a smaller problem, don't update the state with integers fixed
  bool save_state      = false;
  auto solver_response = get_relaxed_lp_solution(fixed_problem,
                                                 fixed_assignment,
                                                 lp_state,
                                                 tols.absolute_tolerance,
                                                 time_limit,
                                                 check_feas,
                                                 return_first_feasible,
                                                 save_state);
  // unfix the assignment on given result no matter if it is feasible
  solution.unfix_variables(fixed_assignment, variable_map);
  if (bound_presolve != nullptr) { bound_presolve->resize(op_problem); }
  return false;
}

#define INSTANTIATE(F_TYPE)                                                                   \
  template optimization_problem_solution_t<int, F_TYPE> get_relaxed_lp_solution<int, F_TYPE>( \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    solution_t<int, F_TYPE> & solution,                                                       \
    F_TYPE tolerance,                                                                         \
    F_TYPE time_limit,                                                                        \
    bool check_infeasibility,                                                                 \
    bool return_first_feasible);                                                              \
  template optimization_problem_solution_t<int, F_TYPE> get_relaxed_lp_solution<int, F_TYPE>( \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    rmm::device_uvector<F_TYPE> & assignment,                                                 \
    lp_state_t<int, F_TYPE> & lp_state,                                                       \
    F_TYPE tolerance,                                                                         \
    F_TYPE time_limit,                                                                        \
    bool check_infeasibility,                                                                 \
    bool return_first_feasible,                                                               \
    bool save_state);                                                                         \
  template bool run_lp_with_vars_fixed<int, F_TYPE>(                                          \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    solution_t<int, F_TYPE> & solution,                                                       \
    const rmm::device_uvector<int>& variables_to_fix,                                         \
    typename mip_solver_settings_t<int, F_TYPE>::tolerances_t tols,                           \
    lp_state_t<int, F_TYPE>& lp_state,                                                        \
    F_TYPE time_limit,                                                                        \
    bool return_first_feasible,                                                               \
    bound_presolve_t<int, F_TYPE>* bound_presolve);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
