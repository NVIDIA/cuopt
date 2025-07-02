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

#include "lagrangian.cuh"
#include "local_search.cuh"

#include <cuopt/error.hpp>

#include <mip/mip_constants.hpp>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/utils.cuh>
#include <utilities/seed_generator.cuh>
#include <utilities/timer.hpp>

#include <cuda_profiler_api.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
local_search_t<i_t, f_t>::local_search_t(mip_solver_context_t<i_t, f_t>& context_,
                                         rmm::device_uvector<f_t>& lp_optimal_solution_)
  : context(context_),
    lp_optimal_solution(lp_optimal_solution_),
    fj_sol_on_lp_opt(context.problem_ptr->n_variables,
                     context.problem_ptr->handle_ptr->get_stream()),
    fj(context),
    // fj_tree(fj),
    constraint_prop(context),
    lb_constraint_prop(context),
    line_segment_search(fj),
    fp(context,
       fj,
       // fj_tree,
       constraint_prop,
       lb_constraint_prop,
       line_segment_search,
       lp_optimal_solution_),
    rng(cuopt::seed_generator::get_seed())
{
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::generate_fast_solution(solution_t<i_t, f_t>& solution, timer_t timer)
{
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.n_of_minimums_for_exit = 500;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.termination            = fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT;
  fj.settings.time_limit             = std::min(30., timer.remaining_time());
  while (!timer.check_time_limit()) {
    timer_t constr_prop_timer = timer_t(std::min(timer.remaining_time(), 2.));
    // do constraint prop on lp optimal solution
    constraint_prop.apply_round(solution, 1., constr_prop_timer);
    if (solution.compute_feasibility()) { return; }
    if (timer.check_time_limit()) { return; };
    fj.settings.time_limit = std::min(3., timer.remaining_time());
    // run fj on the solution
    fj.solve(solution);
    // TODO check if FJ returns the same solution
    // check if the solution is feasible
    if (solution.compute_feasibility()) { return; }
  }
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_local_search(solution_t<i_t, f_t>& solution,
                                                const weight_t<i_t, f_t>& weights,
                                                timer_t timer)
{
  raft::common::nvtx::range fun_scope("local search");
  fj_settings_t fj_settings;
  if (timer.check_time_limit()) return false;
  // adjust these time limits
  fj_settings.time_limit      = timer.remaining_time();
  fj_settings.update_weights  = false;
  fj_settings.feasibility_run = false;
  fj.set_fj_settings(fj_settings);
  fj.copy_weights(weights, solution.handle_ptr);
  cuopt_func_call(bool is_feasible_before_search = solution.get_feasible());
  bool is_feas;
  i_t rd = std::uniform_int_distribution(0, 1)(rng);
  if (rd == 0 && lp_optimal_exists) {
    is_feas = run_fj_line_segment(solution, timer);
    cuopt_assert(!is_feasible_before_search || is_feas,
                 "Line segment search should not change feasibility");
  } else {
    solution_t<i_t, f_t> solution_before_annealing(solution);
    bool is_feasible_before_annealing = solution.get_feasible();
    is_feas                           = run_fj_annealing(solution, timer);
    // we should remove this logic once, FJ is fixed
    if (is_feasible_before_annealing && !solution.get_feasible()) {
      solution.copy_from(solution_before_annealing);
      solution.handle_ptr->sync_stream();
    }
    cuopt_func_call(bool is_feasible_after_annealing = solution.get_feasible());
    cuopt_assert(!is_feasible_before_search || is_feasible_after_annealing,
                 "Annealing should not change feasibility");
  }
  // TODO add a test for the quality improvement too
  return is_feas;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_until_timer(solution_t<i_t, f_t>& solution,
                                                  const weight_t<i_t, f_t>& weights,
                                                  timer_t timer)
{
  bool is_feasible;
  fj.settings.n_of_minimums_for_exit = 1e6;
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.time_limit             = timer.remaining_time() * 0.95;
  fj.settings.update_weights         = false;
  fj.settings.feasibility_run        = false;
  fj.copy_weights(weights, solution.handle_ptr);
  fj.solve(solution);
  CUOPT_LOG_DEBUG("Initial FJ feasibility done");
  is_feasible = solution.compute_feasibility();
  if (fj.settings.feasibility_run || timer.check_time_limit()) { return is_feasible; }
  return is_feasible;
}

// SIMULATED ANNEALING not fully implemented yet, placeholder
template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_annealing(solution_t<i_t, f_t>& solution, timer_t timer)
{
  auto prev_settings = fj.settings;

  // run in FEASIBLE_FIRST to priorize feasibility-improving moves
  fj.settings.n_of_minimums_for_exit = 50;
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.candidate_selection    = fj_candidate_selection_t::FEASIBLE_FIRST;
  fj.settings.termination            = fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT;
  fj.settings.time_limit             = std::min(10., timer.remaining_time());
  fj.settings.parameters.allow_infeasibility_iterations = 100;
  fj.settings.update_weights                            = 1;
  fj.solve(solution);
  bool is_feasible = solution.compute_feasibility();

  fj.settings = prev_settings;
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_line_segment(solution_t<i_t, f_t>& solution, timer_t timer)
{
  rmm::device_uvector<f_t> starting_point(solution.assignment, solution.handle_ptr->get_stream());
  bool feas = line_segment_search.search_line_segment(solution,
                                                      starting_point,
                                                      lp_optimal_solution,
                                                      /*n_points_to_search=*/5,
                                                      /*feasibility_run=*/false,
                                                      timer);
  return feas;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::check_fj_on_lp_optimal(solution_t<i_t, f_t>& solution,
                                                      bool perturb,
                                                      timer_t timer)
{
  raft::copy(solution.assignment.data(),
             lp_optimal_solution.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  if (perturb) {
    CUOPT_LOG_DEBUG("Perturbating solution on initial fj on optimal run!");
    f_t perturbation_ratio = 0.2;
    solution.assign_random_within_bounds(perturbation_ratio);
  }
  cuopt_func_call(solution.test_variable_bounds(false));
  f_t lp_run_time_after_feasible = 1.;
  timer_t bounds_prop_timer      = timer_t(std::min(timer.remaining_time(), 10.));
  bool is_feasible =
    constraint_prop.apply_round(solution, lp_run_time_after_feasible, bounds_prop_timer);
  if (!is_feasible) {
    const f_t lp_run_time = 2.;
    run_lp_with_vars_fixed(*solution.problem_ptr,
                           solution,
                           solution.problem_ptr->integer_indices,
                           solution.problem_ptr->tolerances,
                           context.lp_state,
                           lp_run_time);
  } else {
    return is_feasible;
  }
  cuopt_func_call(solution.test_variable_bounds());
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.n_of_minimums_for_exit = 20000;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.termination            = fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT;
  fj.settings.time_limit             = std::min(30., timer.remaining_time());
  fj.solve(solution);
  return solution.get_feasible();
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_on_zero(solution_t<i_t, f_t>& solution, timer_t timer)
{
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.n_of_minimums_for_exit = 20000;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.termination            = fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT;
  fj.settings.time_limit             = std::min(30., timer.remaining_time());
  bool is_feasible                   = fj.solve(solution);
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_staged_fp(solution_t<i_t, f_t>& solution,
                                             timer_t timer,
                                             bool& early_exit)
{
  auto n_vars         = solution.problem_ptr->n_variables;
  auto n_binary_vars  = solution.problem_ptr->get_n_binary_variables();
  auto n_integer_vars = solution.problem_ptr->n_integer_vars;

  auto binary_only  = (n_binary_vars == n_integer_vars);
  auto integer_only = (n_binary_vars == 0);
  bool is_feasible  = false;

  // TODO return the best solution instead of the last
  if (binary_only || integer_only) {
    return run_fp(solution, timer);
  } else {
    const i_t n_fp_iterations = 1000000;
    fp.cycle_queue.reset(solution);
    fp.reset();
    fp.resize_vectors(*solution.problem_ptr, solution.handle_ptr);
    for (i_t i = 0; i < n_fp_iterations && !timer.check_time_limit(); ++i) {
      if (early_exit) { return false; }
      CUOPT_LOG_DEBUG("Running staged FP from beginning it %d", i);
      fp.relax_general_integers(solution);
      timer_t binary_timer(timer.remaining_time() / 3);
      i_t binary_it_counter = 0;
      for (; binary_it_counter < 100; ++binary_it_counter) {
        if (early_exit) { return false; }
        CUOPT_LOG_DEBUG(
          "Running binary problem from it %d large_restart_it %d", binary_it_counter, i);
        is_feasible = fp.run_single_fp_descent(solution);
        if (is_feasible) { break; }
        if (timer.check_time_limit()) {
          fp.revert_relaxation(solution);
          solution.round_nearest();
          CUOPT_LOG_DEBUG("Time limit reached during binary stage!");
          return false;
        }
        is_feasible = fp.restart_fp(solution);
        if (is_feasible) { break; }
        // give the integer FP some chance
        if (binary_timer.check_time_limit()) {
          CUOPT_LOG_DEBUG("Binary FP time limit reached during binary stage!");
          break;
        }
      }
      CUOPT_LOG_DEBUG("Exited binary problem at it %d large_restart_it %d feas %d",
                      binary_it_counter,
                      i,
                      is_feasible);
      // TODO try resetting and not resetting the alpha
      fp.revert_relaxation(solution);
      fp.last_distances.resize(0);

      for (i_t integer_it_counter = 0; integer_it_counter < 500; ++integer_it_counter) {
        CUOPT_LOG_DEBUG(
          "Running integer problem from it %d large_restart_it %d", integer_it_counter, i);
        is_feasible = fp.run_single_fp_descent(solution);
        if (is_feasible) { return true; }
        if (timer.check_time_limit()) {
          CUOPT_LOG_DEBUG("FP time limit reached during integer stage!");
          solution.round_nearest();
          return false;
        }
        is_feasible = fp.restart_fp(solution);
        if (is_feasible) { return true; }
      }
    }
  }
  return false;
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::resize_vectors(problem_t<i_t, f_t>& problem,
                                              const raft::handle_t* handle_ptr)
{
  fj_sol_on_lp_opt.resize(problem.n_variables, handle_ptr->get_stream());
  fj.resize_vectors(handle_ptr);
  // TODO resize LNS when it is used
  fp.resize_vectors(problem, handle_ptr);
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fp(solution_t<i_t, f_t>& solution, timer_t timer)
{
  const i_t n_fp_iterations = 1000000;
  bool is_feasible          = false;
  for (i_t i = 0; i < n_fp_iterations && !timer.check_time_limit(); ++i) {
    if (timer.check_time_limit()) { return false; }
    CUOPT_LOG_DEBUG("fp_loop it %d", i);
    is_feasible = fp.run_single_fp_descent(solution);
    // if feasible return true
    if (is_feasible) {
      return true;
    }
    // if not feasible, it means it is a cycle
    else {
      if (timer.check_time_limit()) { return false; }
      is_feasible = fp.restart_fp(solution);
      if (is_feasible) { return true; }
    }
  }
  return false;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::generate_solution(solution_t<i_t, f_t>& solution,
                                                 bool perturb,
                                                 bool& early_exit,
                                                 f_t time_limit)
{
  raft::common::nvtx::range fun_scope("LS FP Loop");

  timer_t timer(time_limit);
  auto n_vars         = solution.problem_ptr->n_variables;
  auto n_binary_vars  = solution.problem_ptr->get_n_binary_variables();
  auto n_integer_vars = solution.problem_ptr->n_integer_vars;
  bool is_feasible    = check_fj_on_lp_optimal(solution, perturb, timer);
  if (is_feasible) {
    CUOPT_LOG_DEBUG("Solution generated with FJ on LP optimal: is_feasible %d", is_feasible);
    return true;
  }
  if (!perturb) {
    raft::copy(fj_sol_on_lp_opt.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    fj.reset_weights(solution.handle_ptr->get_stream());
    is_feasible = run_fj_on_zero(solution, timer);
    if (is_feasible) {
      CUOPT_LOG_DEBUG("Solution generated with FJ on zero solution: is_feasible %d", is_feasible);
      return true;
    }
    raft::copy(solution.assignment.data(),
               fj_sol_on_lp_opt.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
  }
  fp.timer = timer;
  // continue with the solution with fj on lp optimal
  fp.cycle_queue.reset(solution);
  fp.reset();
  fp.resize_vectors(*solution.problem_ptr, solution.handle_ptr);
  is_feasible = run_staged_fp(solution, timer, early_exit);
  // is_feasible = run_fp(solution, timer);
  CUOPT_LOG_DEBUG("Solution generated with FP: is_feasible %d", is_feasible);
  return is_feasible;
}

#if MIP_INSTANTIATE_FLOAT
template class local_search_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class local_search_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
