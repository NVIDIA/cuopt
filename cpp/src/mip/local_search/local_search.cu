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

#include <mip/feasibility_jump/fj_cpu.cuh>

#include <cuda_profiler_api.h>

#include <future>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
cpu_fj_thread_t<i_t, f_t>::cpu_fj_thread_t()
{
  cpu_worker = std::thread(&cpu_fj_thread_t<i_t, f_t>::cpu_worker_thread, this);
}

template <typename i_t, typename f_t>
cpu_fj_thread_t<i_t, f_t>::~cpu_fj_thread_t()
{
  if (!cpu_thread_terminate) { kill_cpu_solver(); }
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::cpu_worker_thread()
{
  while (!cpu_thread_terminate) {
    // Wait for start signal
    {
      std::unique_lock<std::mutex> lock(cpu_mutex);
      cpu_cv.wait(lock, [this] { return cpu_thread_should_start || cpu_thread_terminate; });
    }

    if (cpu_thread_terminate) break;

    // Run CPU solver
    {
      raft::common::nvtx::range fun_scope("Running CPU FJ");
      cpu_fj_solution_found = fj_ptr->cpu_solve(*fj_cpu);
    }

    cpu_thread_should_start = false;
    cpu_thread_done         = true;
  }
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::kill_cpu_solver()
{
  cpu_thread_terminate = true;
  if (fj_cpu) fj_cpu->halted = true;
  cpu_cv.notify_one();
  cpu_worker.join();
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::start_cpu_solver()
{
  cuopt_assert(fj_cpu != nullptr, "fj_cpu must not be null");
  // Reset flags
  cpu_thread_done         = false;
  cpu_thread_should_start = true;
  fj_cpu->halted          = false;
  cpu_cv.notify_one();
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::stop_cpu_solver()
{
  fj_cpu->halted = true;
}

template <typename i_t, typename f_t>
bool cpu_fj_thread_t<i_t, f_t>::wait_for_cpu_solver()
{
  while (!cpu_thread_done && !cpu_thread_terminate) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  return cpu_fj_solution_found;
}

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
    line_segment_search(fj, constraint_prop),
    fp(context,
       fj,
       // fj_tree,
       constraint_prop,
       line_segment_search,
       lp_optimal_solution_),
    rng(cuopt::seed_generator::get_seed()),
    problem_with_objective_cut(*context.problem_ptr, context.problem_ptr->handle_ptr)
{
  for (auto& cpu_fj : ls_cpu_fj) {
    cpu_fj.fj_ptr = &fj;
  }
  for (auto& cpu_fj : scratch_cpu_fj) {
    cpu_fj.fj_ptr = &fj;
  }
  scratch_cpu_fj_on_lp_opt.fj_ptr = &fj;
}

static double local_search_best_obj       = std::numeric_limits<double>::max();
static population_t<int, double>* pop_ptr = nullptr;

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::start_cpufj_scratch_threads(population_t<i_t, f_t>& population)
{
  pop_ptr = &population;

  std::vector<f_t> default_weights(context.problem_ptr->n_constraints, 1.);

  solution_t<i_t, f_t> solution(*context.problem_ptr);
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();
  i_t counter = 0;
  for (auto& cpu_fj : scratch_cpu_fj) {
    if (counter > 0) solution.assign_random_within_bounds(0.4);
    cpu_fj.fj_cpu = cpu_fj.fj_ptr->create_cpu_climber(solution,
                                                      default_weights,
                                                      default_weights,
                                                      0.,
                                                      fj_settings_t{},
                                                      /*randomize=*/counter > 0);

    cpu_fj.fj_cpu->log_prefix           = "******* scratch " + std::to_string(counter) + ": ";
    cpu_fj.fj_cpu->improvement_callback = [this, &population, &cpu_fj](
                                            f_t obj, const std::vector<f_t>& h_vec) {
      population.add_external_solution(h_vec, obj, solution_origin_t::CPUFJ);
      if (obj < local_search_best_obj) {
        CUOPT_LOG_DEBUG("******* New local search best obj %g, best overall %g",
                        context.problem_ptr->get_user_obj_from_solver_obj(obj),
                        context.problem_ptr->get_user_obj_from_solver_obj(
                          population.is_feasible() ? population.best_feasible().get_objective()
                                                   : std::numeric_limits<f_t>::max()));
        local_search_best_obj = obj;
      }
    };
    counter++;
  };

  // default weights
  cudaDeviceSynchronize();

  for (auto& cpu_fj : scratch_cpu_fj) {
    cpu_fj.start_cpu_solver();
  }
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::start_cpufj_lptopt_scratch_threads(
  population_t<i_t, f_t>& population)
{
  pop_ptr = &population;

  std::vector<f_t> default_weights(context.problem_ptr->n_constraints, 1.);

  solution_t<i_t, f_t> solution_lp(*context.problem_ptr);
  solution_lp.copy_new_assignment(host_copy(lp_optimal_solution));
  solution_lp.round_random_nearest(500);
  scratch_cpu_fj_on_lp_opt.fj_cpu =
    fj.create_cpu_climber(solution_lp, default_weights, default_weights, 0.);
  scratch_cpu_fj_on_lp_opt.fj_cpu->log_prefix = "******* scratch on LP optimal: ";
  scratch_cpu_fj_on_lp_opt.fj_cpu->improvement_callback =
    [this, &population](f_t obj, const std::vector<f_t>& h_vec) {
      population.add_external_solution(h_vec, obj, solution_origin_t::CPUFJ);
      if (obj < local_search_best_obj) {
        CUOPT_LOG_DEBUG("******* New local search best obj %g, best overall %g",
                        context.problem_ptr->get_user_obj_from_solver_obj(obj),
                        context.problem_ptr->get_user_obj_from_solver_obj(
                          population.is_feasible() ? population.best_feasible().get_objective()
                                                   : std::numeric_limits<f_t>::max()));
        local_search_best_obj = obj;
      }
    };

  // default weights
  cudaDeviceSynchronize();
  scratch_cpu_fj_on_lp_opt.start_cpu_solver();
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::stop_cpufj_scratch_threads()
{
  for (auto& cpu_fj : scratch_cpu_fj) {
    cpu_fj.kill_cpu_solver();
  }
  scratch_cpu_fj_on_lp_opt.kill_cpu_solver();
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::do_fj_solve(solution_t<i_t, f_t>& solution,
                                           fj_t<i_t, f_t>& in_fj,
                                           const std::string& source)
{
  auto h_weights          = cuopt::host_copy(in_fj.cstr_weights, solution.handle_ptr->get_stream());
  auto h_objective_weight = in_fj.objective_weight.value(solution.handle_ptr->get_stream());
  for (auto& cpu_fj : ls_cpu_fj) {
    cpu_fj.fj_cpu = cpu_fj.fj_ptr->create_cpu_climber(
      solution, h_weights, h_weights, h_objective_weight, fj_settings_t{}, true);
  }

  auto solution_copy = solution;

  // Start CPU solver in background thread
  for (auto& cpu_fj : ls_cpu_fj) {
    cpu_fj.start_cpu_solver();
  }

  // Run GPU solver and measure execution time
  auto gpu_fj_start = std::chrono::high_resolution_clock::now();
  in_fj.solve(solution);

  // Stop CPU solver
  for (auto& cpu_fj : ls_cpu_fj) {
    cpu_fj.stop_cpu_solver();
  }

  auto gpu_fj_end        = std::chrono::high_resolution_clock::now();
  double gpu_fj_duration = std::chrono::duration<double>(gpu_fj_end - gpu_fj_start).count();

  solution_t<i_t, f_t> solution_cpu(*solution.problem_ptr);

  f_t best_cpu_obj = std::numeric_limits<f_t>::max();
  // // Wait for CPU solver to finish
  for (auto& cpu_fj : ls_cpu_fj) {
    bool cpu_sol_found = cpu_fj.wait_for_cpu_solver();
    if (cpu_sol_found) {
      f_t cpu_obj = cpu_fj.fj_cpu->h_best_objective;
      if (cpu_obj < best_cpu_obj) {
        best_cpu_obj = cpu_obj;
        solution_cpu.copy_new_assignment(cpu_fj.fj_cpu->h_best_assignment);
        solution_cpu.compute_feasibility();
      }
    }
  }
  bool cpu_sol_found = best_cpu_obj < std::numeric_limits<f_t>::max();

  bool gpu_feasible = solution.get_feasible();
  bool cpu_feasible = cpu_sol_found && solution_cpu.get_feasible();

  static std::unordered_map<std::string, int> total_calls;
  static std::unordered_map<std::string, int> cpu_better;

  CUOPT_LOG_DEBUG("GPU FJ returns feas %d, obj %g", gpu_feasible, solution.get_user_objective());
  CUOPT_LOG_DEBUG("CPU FJ returns feas %d, obj %g, stats %d/%d",
                  cpu_feasible,
                  solution_cpu.get_user_objective(),
                  total_calls[source],
                  cpu_better[source]);

  total_calls[source]++;
  if (cpu_feasible && !gpu_feasible ||
      (cpu_feasible && solution_cpu.get_objective() < solution.get_objective())) {
    CUOPT_LOG_DEBUG(
      "CPU FJ returns better solution! cpu_obj %g, gpu_obj %g, stats %d/%d, source %s",
      solution_cpu.get_user_objective(),
      solution.get_user_objective(),
      total_calls[source],
      cpu_better[source],
      source.c_str());
    solution.copy_from(solution_cpu);
    cpu_better[source]++;
  }
  solution.compute_feasibility();

  return cpu_feasible;
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
  fj.settings.time_limit             = std::min(30., timer.remaining_time());
  while (!timer.check_time_limit()) {
    timer_t constr_prop_timer = timer_t(std::min(timer.remaining_time(), 2.));
    // do constraint prop on lp optimal solution
    constraint_prop.apply_round(solution, 1., constr_prop_timer);
    if (solution.compute_feasibility()) { return; }
    if (timer.check_time_limit()) { return; };
    fj.settings.time_limit = std::min(3., timer.remaining_time());
    // run fj on the solution
    do_fj_solve(solution, fj, "fast");
    // TODO check if FJ returns the same solution
    // check if the solution is feasible
    if (solution.compute_feasibility()) { return; }
  }
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_local_search(solution_t<i_t, f_t>& solution,
                                                const weight_t<i_t, f_t>& weights,
                                                timer_t timer,
                                                const ls_config_t<i_t, f_t>& ls_config)
{
  raft::common::nvtx::range fun_scope("local search");
  fj_settings_t fj_settings;
  if (timer.check_time_limit()) return false;
  // adjust these time limits
  if (!solution.get_feasible()) {
    if (ls_config.at_least_one_parent_feasible) {
      fj_settings.time_limit = 1.;
      timer                  = timer_t(1.);
    } else {
      fj_settings.time_limit = 0.5;
      timer                  = timer_t(0.5);
    }
  } else {
    fj_settings.time_limit = std::min(10., timer.remaining_time());
  }
  fj_settings.update_weights  = false;
  fj_settings.feasibility_run = false;
  fj.set_fj_settings(fj_settings);
  bool is_feas   = false;
  ls_method_t rd = static_cast<ls_method_t>(
    std::uniform_int_distribution(static_cast<int>(ls_method_t::FJ_ANNEALING),
                                  static_cast<int>(ls_method_t::FJ_LINE_SEGMENT))(rng));
  if (ls_config.ls_method == ls_method_t::FJ_LINE_SEGMENT) {
    rd = ls_method_t::FJ_LINE_SEGMENT;
  } else if (ls_config.ls_method == ls_method_t::FJ_ANNEALING) {
    rd = ls_method_t::FJ_ANNEALING;
  }
  if (rd == ls_method_t::FJ_LINE_SEGMENT && lp_optimal_exists) {
    fj.copy_weights(weights, solution.handle_ptr);
    is_feas = run_fj_line_segment(solution, timer, ls_config);
  } else {
    fj.copy_weights(weights, solution.handle_ptr);
    is_feas = run_fj_annealing(solution, timer, ls_config);
    if (lp_optimal_exists) { is_feas = run_fj_line_segment(solution, timer, ls_config); }
  }
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
  do_fj_solve(solution, fj, "until_timer");
  CUOPT_LOG_DEBUG("Initial FJ feasibility done");
  is_feasible = solution.compute_feasibility();
  if (fj.settings.feasibility_run || timer.check_time_limit()) { return is_feasible; }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_annealing(solution_t<i_t, f_t>& solution,
                                                timer_t timer,
                                                const ls_config_t<i_t, f_t>& ls_config)
{
  auto prev_settings = fj.settings;

  solution.compute_feasibility();
  CUOPT_LOG_DEBUG("Running FJ Annealing on solution with obj %g/%g, feas? %d",
                  solution.get_user_objective(),
                  solution.get_objective(),
                  solution.get_feasible());

  // run in FEASIBLE_FIRST to priorize feasibility-improving moves
  fj.settings.n_of_minimums_for_exit                    = ls_config.n_local_mins;
  fj.settings.mode                                      = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.candidate_selection                       = fj_candidate_selection_t::FEASIBLE_FIRST;
  fj.settings.iteration_limit                           = ls_config.iteration_limit;
  fj.settings.time_limit                                = std::min(10., timer.remaining_time());
  fj.settings.parameters.allow_infeasibility_iterations = 100;
  fj.settings.update_weights                            = 1;
  fj.settings.baseline_objective_for_longer_run         = ls_config.best_objective_of_parents;
  do_fj_solve(solution, fj, "annealing");
  bool is_feasible = solution.compute_feasibility();

  fj.settings = prev_settings;
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_line_segment(solution_t<i_t, f_t>& solution,
                                                   timer_t timer,
                                                   const ls_config_t<i_t, f_t>& ls_config)
{
  rmm::device_uvector<f_t> starting_point(solution.assignment, solution.handle_ptr->get_stream());
  line_segment_search.settings.best_of_parents_cost = ls_config.best_objective_of_parents;
  line_segment_search.settings.parents_infeasible   = !ls_config.at_least_one_parent_feasible;
  line_segment_search.settings.recombiner_mode      = false;
  line_segment_search.settings.n_local_min          = ls_config.n_local_mins_for_line_segment;
  line_segment_search.settings.n_points_to_search   = ls_config.n_points_to_search_for_line_segment;
  line_segment_search.settings.iteration_limit      = ls_config.iteration_limit_for_line_segment;

  bool feas = line_segment_search.search_line_segment(solution,
                                                      starting_point,
                                                      lp_optimal_solution,
                                                      /*feasibility_run=*/false,
                                                      timer);
  return feas;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::check_fj_on_lp_optimal(solution_t<i_t, f_t>& solution,
                                                      bool perturb,
                                                      timer_t timer)
{
  if (lp_optimal_exists) {
    raft::copy(solution.assignment.data(),
               lp_optimal_solution.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    cuopt_func_call(solution.test_variable_bounds(false));
  }
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
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit = lp_run_time;
    lp_settings.tolerance  = solution.problem_ptr->tolerances.absolute_tolerance;
    run_lp_with_vars_fixed(
      *solution.problem_ptr, solution, solution.problem_ptr->integer_indices, lp_settings);
  } else {
    return is_feasible;
  }
  cuopt_func_call(solution.test_variable_bounds());
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.n_of_minimums_for_exit = 20000;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = false;
  fj.settings.time_limit             = std::min(30., timer.remaining_time());
  do_fj_solve(solution, fj, "on_lp_optimal");
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
  fj.settings.feasibility_run        = false;
  fj.settings.time_limit             = std::min(30., timer.remaining_time());
  bool is_feasible                   = do_fj_solve(solution, fj, "on_zero");
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_staged_fp(solution_t<i_t, f_t>& solution,
                                             timer_t timer,
                                             population_t<i_t, f_t>* population_ptr)
{
  cuopt_assert(population_ptr != nullptr, "Population pointer must not be null");
  auto n_vars         = solution.problem_ptr->n_variables;
  auto n_binary_vars  = solution.problem_ptr->get_n_binary_variables();
  auto n_integer_vars = solution.problem_ptr->n_integer_vars;

  auto binary_only  = (n_binary_vars == n_integer_vars);
  auto integer_only = (n_binary_vars == 0);
  bool is_feasible  = false;

  if (binary_only || integer_only) {
    return run_fp(solution, timer, population_ptr);
  } else {
    const i_t n_fp_iterations = 1000000;
    fp.cycle_queue.reset(solution);
    fp.reset();
    fp.resize_vectors(*solution.problem_ptr, solution.handle_ptr);
    for (i_t i = 0; i < n_fp_iterations && !timer.check_time_limit(); ++i) {
      if (population_ptr->preempt_heuristic_solver_.load()) {
        CUOPT_LOG_DEBUG("Preempting heuristic solver!");
        return false;
      }
      CUOPT_LOG_DEBUG("Running staged FP from beginning it %d", i);
      fp.relax_general_integers(solution);
      timer_t binary_timer(timer.remaining_time() / 3);
      i_t binary_it_counter = 0;
      for (; binary_it_counter < 100; ++binary_it_counter) {
        if (population_ptr->preempt_heuristic_solver_.load()) {
          CUOPT_LOG_DEBUG("Preempting heuristic solver!");
          return false;
        }
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
  fp.resize_vectors(problem, handle_ptr);
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::save_solution_and_add_cutting_plane(
  solution_t<i_t, f_t>& solution, rmm::device_uvector<f_t>& best_solution, f_t& best_objective)
{
  if (solution.get_objective() < best_objective) {
    raft::copy(best_solution.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    best_objective = solution.get_objective();
    f_t objective_cut =
      best_objective - std::max(std::abs(0.001 * best_objective), OBJECTIVE_EPSILON);
    problem_with_objective_cut.add_cutting_plane_at_objective(objective_cut);
  }
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::resize_to_new_problem()
{
  resize_vectors(problem_with_objective_cut, problem_with_objective_cut.handle_ptr);
  // hint for next PR in case load balanced is reintroduced
  // lb_constraint_prop.temp_problem.setup(problem_with_objective_cut);
  // lb_constraint_prop.bounds_update.setup(lb_constraint_prop.temp_problem);
  constraint_prop.bounds_update.resize(problem_with_objective_cut);
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::resize_to_old_problem(problem_t<i_t, f_t>* old_problem_ptr)
{
  resize_vectors(*old_problem_ptr, old_problem_ptr->handle_ptr);
  // hint for next PR in case load balanced is reintroduced
  // lb_constraint_prop.temp_problem.setup(*old_problem_ptr);
  // lb_constraint_prop.bounds_update.setup(lb_constraint_prop.temp_problem);
  constraint_prop.bounds_update.resize(*old_problem_ptr);
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::reset_alpha_and_run_recombiners(
  solution_t<i_t, f_t>& solution,
  problem_t<i_t, f_t>* old_problem_ptr,
  population_t<i_t, f_t>* population_ptr,
  i_t i,
  i_t last_unimproved_iteration,
  rmm::device_uvector<f_t>& best_solution,
  f_t& best_objective)
{
  fp.config.alpha = default_alpha;
  solution_t<i_t, f_t> solution_copy(solution);
  solution_copy.problem_ptr = old_problem_ptr;
  solution_copy.resize_to_problem();
  population_ptr->add_solution(std::move(solution_copy));
  constexpr i_t iterations_for_stagnation = 3;
  if (population_ptr->current_size() > 1 &&
      i - last_unimproved_iteration > iterations_for_stagnation) {
    solution_t<i_t, f_t> best_feasible_copy(population_ptr->best_feasible());
    population_ptr->run_all_recombiners(best_feasible_copy);
  }
  auto new_sol_vector = population_ptr->get_external_solutions();
  population_ptr->add_solutions_from_vec(std::move(new_sol_vector));
  if (!cutting_plane_added_for_active_run) {
    fj.copy_weights(
      population_ptr->weights, solution.handle_ptr, problem_with_objective_cut.n_constraints);
    solution.problem_ptr = &problem_with_objective_cut;
    solution.resize_to_problem();
    resize_to_new_problem();
    cutting_plane_added_for_active_run = true;
  }
  save_solution_and_add_cutting_plane(
    population_ptr->best_feasible(), best_solution, best_objective);
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fp(solution_t<i_t, f_t>& solution,
                                      timer_t timer,
                                      population_t<i_t, f_t>* population_ptr)
{
  cuopt_assert(population_ptr != nullptr, "Population pointer must not be null");
  const i_t n_fp_iterations                  = 1000000;
  constexpr i_t n_sol_in_population_for_exit = 4;
  bool is_feasible                           = solution.compute_feasibility();
  cutting_plane_added_for_active_run         = is_feasible;
  double best_objective =
    is_feasible ? solution.get_objective() : std::numeric_limits<double>::max();
  rmm::device_uvector<f_t> best_solution(solution.assignment, solution.handle_ptr->get_stream());
  problem_t<i_t, f_t>* old_problem_ptr = solution.problem_ptr;
  fp.timer                             = timer_t(timer.remaining_time());
  // if it has not been initialized yet, create a new problem and move it to the cut problem
  if (!problem_with_objective_cut.cutting_plane_added) {
    problem_with_objective_cut = std::move(problem_t<i_t, f_t>(*old_problem_ptr));
  }
  if (is_feasible) {
    f_t objective_cut =
      best_objective - std::max(std::abs(0.001 * best_objective), OBJECTIVE_EPSILON);
    problem_with_objective_cut.add_cutting_plane_at_objective(objective_cut);
    // Do the copy here for proper handling of the added constraints weight
    fj.copy_weights(
      population_ptr->weights, solution.handle_ptr, problem_with_objective_cut.n_constraints);
    solution.problem_ptr = &problem_with_objective_cut;
    solution.resize_to_problem();
    resize_to_new_problem();
  }
  i_t last_unimproved_iteration = 0;
  for (i_t i = 0; i < n_fp_iterations && !timer.check_time_limit(); ++i) {
    if (timer.check_time_limit()) {
      is_feasible = false;
      break;
    }
    CUOPT_LOG_DEBUG("fp_loop it %d", i);
    if (population_ptr->preempt_heuristic_solver_.load()) {
      CUOPT_LOG_DEBUG("Preempting heuristic solver!");
      break;
    }
    is_feasible = fp.run_single_fp_descent(solution);
    if (population_ptr->preempt_heuristic_solver_.load()) {
      CUOPT_LOG_DEBUG("Preempting heuristic solver!");
      break;
    }
    // if feasible return true
    if (is_feasible) {
      CUOPT_LOG_DEBUG("Found feasible in FP with obj %f. Continue with FJ!",
                      solution.get_objective());
      reset_alpha_and_run_recombiners(solution,
                                      old_problem_ptr,
                                      population_ptr,
                                      i,
                                      last_unimproved_iteration,
                                      best_solution,
                                      best_objective);
      if (population_ptr->current_size() >= n_sol_in_population_for_exit) { break; }
    }
    // if not feasible, it means it is a cycle
    else {
      if (timer.check_time_limit()) {
        is_feasible = false;
        break;
      }
      is_feasible = fp.restart_fp(solution);
      if (population_ptr->preempt_heuristic_solver_.load()) {
        CUOPT_LOG_DEBUG("Preempting heuristic solver!");
        break;
      }
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Found feasible during restart with obj %f. Continue with FJ!",
                        solution.get_objective());
        reset_alpha_and_run_recombiners(solution,
                                        old_problem_ptr,
                                        population_ptr,
                                        i,
                                        last_unimproved_iteration,
                                        best_solution,
                                        best_objective);
        if (population_ptr->current_size() >= n_sol_in_population_for_exit) { break; }
      } else {
        last_unimproved_iteration = i;
      }
    }
  }
  raft::copy(solution.assignment.data(),
             best_solution.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  solution.problem_ptr = old_problem_ptr;
  solution.resize_to_problem();
  resize_to_old_problem(old_problem_ptr);
  solution.handle_ptr->sync_stream();
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::generate_solution(solution_t<i_t, f_t>& solution,
                                                 bool perturb,
                                                 population_t<i_t, f_t>* population_ptr,
                                                 f_t time_limit)
{
  raft::common::nvtx::range fun_scope("LS FP Loop");
  cuopt_assert(population_ptr != nullptr, "Population pointer must not be null");
  timer_t timer(time_limit);
  auto n_vars         = solution.problem_ptr->n_variables;
  auto n_binary_vars  = solution.problem_ptr->get_n_binary_variables();
  auto n_integer_vars = solution.problem_ptr->n_integer_vars;
  bool is_feasible    = check_fj_on_lp_optimal(solution, perturb, timer);
  if (is_feasible) {
    CUOPT_LOG_DEBUG("Solution generated with FJ on LP optimal: is_feasible %d", is_feasible);
    return true;
  }
  if (population_ptr->preempt_heuristic_solver_.load()) {
    CUOPT_LOG_DEBUG("Preempting heuristic solver!");
    return is_feasible;
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
  if (population_ptr->preempt_heuristic_solver_.load()) {
    CUOPT_LOG_DEBUG("Preempting heuristic solver!");
    return is_feasible;
  }
  fp.timer = timer;
  // continue with the solution with fj on lp optimal
  fp.cycle_queue.reset(solution);
  fp.reset();
  fp.resize_vectors(*solution.problem_ptr, solution.handle_ptr);
  is_feasible = run_staged_fp(solution, timer, population_ptr);
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
