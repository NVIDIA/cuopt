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

#include <mip/mip_constants.hpp>
#include <mip/presolve/probing_cache.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/problem/problem_helpers.cuh>
#include "diversity_manager.cuh"

#include <utilities/scope_guard.hpp>

#include "cuda_profiler_api.h"

constexpr bool from_dir    = false;
constexpr bool fj_only_run = false;

namespace cuopt::linear_programming::detail {

size_t fp_recombiner_config_t::max_n_of_vars_from_other =
  fp_recombiner_config_t::initial_n_of_vars_from_other;
size_t ls_recombiner_config_t::max_n_of_vars_from_other =
  ls_recombiner_config_t::initial_n_of_vars_from_other;
size_t bp_recombiner_config_t::max_n_of_vars_from_other =
  bp_recombiner_config_t::initial_n_of_vars_from_other;
size_t sub_mip_recombiner_config_t::max_n_of_vars_from_other =
  sub_mip_recombiner_config_t::initial_n_of_vars_from_other;

template <typename i_t, typename f_t>
diversity_manager_t<i_t, f_t>::diversity_manager_t(mip_solver_context_t<i_t, f_t>& context_)
  : context(context_),
    problem_ptr(context.problem_ptr),
    diversity_config(),
    population("population",
               context,
               *this,
               diversity_config.max_var_diff,
               diversity_config.max_solutions,
               diversity_config.initial_infeasibility_weight * context.problem_ptr->n_constraints),
    lp_optimal_solution(context.problem_ptr->n_variables,
                        context.problem_ptr->handle_ptr->get_stream()),
    lp_dual_optimal_solution(context.problem_ptr->n_constraints,
                             context.problem_ptr->handle_ptr->get_stream()),
    ls(context, lp_optimal_solution),
    timer(diversity_config.default_time_limit),
    bound_prop_recombiner(context,
                          context.problem_ptr->n_variables,
                          ls.constraint_prop,
                          context.problem_ptr->handle_ptr),
    fp_recombiner(context,
                  context.problem_ptr->n_variables,
                  ls.fj,
                  ls.constraint_prop,
                  ls.line_segment_search,
                  lp_optimal_solution,
                  context.problem_ptr->handle_ptr),
    line_segment_recombiner(context,
                            context.problem_ptr->n_variables,
                            ls.line_segment_search,
                            context.problem_ptr->handle_ptr),
    sub_mip_recombiner(
      context, population, context.problem_ptr->n_variables, context.problem_ptr->handle_ptr),
    rng(cuopt::seed_generator::get_seed()),
    stats(context.stats),
    mab_recombiner(static_cast<int>(recombiner_enum_t::SIZE),
                   cuopt::seed_generator::get_seed(),
                   recombiner_alpha,
                   "recombiner"),
    mab_ls(mab_ls_config_t<i_t, f_t>::n_of_arms, cuopt::seed_generator::get_seed(), ls_alpha, "ls"),
    ls_hash_map(*context.problem_ptr)
{
  // Read configuration ID from environment variable
  int max_config = -1;
  // Read max configuration value from environment variable
  const char* env_max_config = std::getenv("CUOPT_MAX_CONFIG");
  if (env_max_config != nullptr) {
    try {
      max_config = std::stoi(env_max_config);
      CUOPT_LOG_INFO("Using maximum configuration value from environment: %d", max_config);
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("Failed to parse CUOPT_MAX_CONFIG environment variable: %s", e.what());
    }
  }
  if (max_config > 1) {
    int config_id             = -1;  // Default value
    const char* env_config_id = std::getenv("CUOPT_CONFIG_ID");
    if (env_config_id != nullptr) {
      try {
        config_id = std::stoi(env_config_id);
        CUOPT_LOG_INFO("Using configuration ID from environment: %d", config_id);
      } catch (const std::exception& e) {
        CUOPT_LOG_WARN("Failed to parse CUOPT_CONFIG_ID environment variable: %s", e.what());
      }
    }
  }
}

// this function is to specialize the local search with config from diversity manager
template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_local_search(solution_t<i_t, f_t>& solution,
                                                     const weight_t<i_t, f_t>& weights,
                                                     timer_t& timer,
                                                     ls_config_t<i_t, f_t>& ls_config)
{
  i_t ls_mab_option = mab_ls.select_mab_option();
  mab_ls_config_t<i_t, f_t>::get_local_search_and_lm_from_config(ls_mab_option, ls_config);
  ls_hash_map.insert(solution);
  constexpr i_t skip_solutions_threshold = 3;
  if (ls_hash_map.check_skip_solution(solution, skip_solutions_threshold)) { return false; }
  ls.run_local_search(solution, weights, timer, ls_config);
  return true;
}

// There should be at least 3 solutions in the population
template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::regenerate_solutions()
{
  f_t time_limit     = 5;
  i_t counter        = 0;
  const i_t min_size = 2;
  while (population.current_size() <= min_size && (current_step == 0 || counter < 5)) {
    CUOPT_LOG_DEBUG("Trying to regenerate solution, pop size %d\n", population.current_size());
    time_limit = std::min(time_limit, timer.remaining_time());
    ls.fj.randomize_weights(problem_ptr->handle_ptr);
    population.add_solution(generate_solution(time_limit));
    if (timer.check_time_limit()) { return false; }
    // increase the time limit as we couldn't add a valid solution
    time_limit += 5;
    counter++;
  }
  ++current_step;
  // if there is at least two sols still return true
  return population.current_size() >= min_size;
}

// There should be at least 3 solutions in the population
template <typename i_t, typename f_t>
std::vector<solution_t<i_t, f_t>> diversity_manager_t<i_t, f_t>::generate_more_solutions()
{
  std::vector<solution_t<i_t, f_t>> solutions;
  timer_t total_time_to_generate = timer_t(timer.remaining_time() / 5.);
  f_t time_limit                 = std::min(60., total_time_to_generate.remaining_time());
  f_t ls_limit                   = std::min(5., timer.remaining_time() / 20.);
  const i_t n_sols_to_generate   = 3;
  for (i_t i = 0; i < n_sols_to_generate; ++i) {
    CUOPT_LOG_DEBUG("Trying to generate more solutions");
    time_limit = std::min(time_limit, timer.remaining_time());
    ls.fj.randomize_weights(problem_ptr->handle_ptr);
    auto sol = generate_solution(time_limit);
    population.run_solution_callbacks(sol);
    solutions.emplace_back(solution_t<i_t, f_t>(sol));
    if (total_time_to_generate.check_time_limit()) { return solutions; }
    timer_t ls_timer(std::min(ls_limit, timer.remaining_time()));
    ls_config_t<i_t, f_t> ls_config;
    run_local_search(sol, population.weights, ls_timer, ls_config);
    population.run_solution_callbacks(sol);
    solutions.emplace_back(std::move(sol));
    if (total_time_to_generate.check_time_limit()) { return solutions; }
  }
  return solutions;
}

template <typename i_t, typename f_t>
solution_t<i_t, f_t> diversity_manager_t<i_t, f_t>::generate_solution(f_t time_limit,
                                                                      bool random_start)
{
  solution_t<i_t, f_t> sol(*problem_ptr);
  sol.compute_feasibility();
  ls.generate_solution(sol, random_start, &population, time_limit);
  return sol;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::add_user_given_solutions(
  std::vector<solution_t<i_t, f_t>>& initial_sol_vector)
{
  for (const auto& init_sol : context.settings.initial_solutions) {
    solution_t<i_t, f_t> sol(*problem_ptr);
    rmm::device_uvector<f_t> init_sol_assignment(*init_sol, sol.handle_ptr->get_stream());
    if (problem_ptr->pre_process_assignment(init_sol_assignment)) {
      raft::copy(sol.assignment.data(),
                 init_sol_assignment.data(),
                 init_sol_assignment.size(),
                 sol.handle_ptr->get_stream());
      bool is_feasible = sol.compute_feasibility();
      cuopt_func_call(sol.test_variable_bounds(true));
      CUOPT_LOG_INFO("Adding initial solution success! feas %d objective %f excess %f",
                     is_feasible,
                     sol.get_user_objective(),
                     sol.get_total_excess());
      population.run_solution_callbacks(sol);
      initial_sol_vector.emplace_back(std::move(sol));
    } else {
      CUOPT_LOG_ERROR(
        "Error cannot add the provided initial solution! \
    Assignment size %lu \
    initial solution size %lu",
        sol.assignment.size(),
        init_sol_assignment.size());
    }
  }
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_presolve(f_t time_limit)
{
  CUOPT_LOG_INFO("Running presolve!");
  timer_t presolve_timer(time_limit);
  auto term_crit = ls.constraint_prop.bounds_update.solve(*problem_ptr);
  if (ls.constraint_prop.bounds_update.infeas_constraints_count > 0) {
    stats.presolve_time = timer.elapsed_time();
    return false;
  }
  if (termination_criterion_t::NO_UPDATE != term_crit) {
    ls.constraint_prop.bounds_update.set_updated_bounds(*problem_ptr);
    trivial_presolve(*problem_ptr);
    if (!problem_ptr->empty && !check_bounds_sanity(*problem_ptr)) { return false; }
  }
  if (!problem_ptr->empty) {
    // do the resizing no-matter what, bounds presolve might not change the bounds but initial
    // trivial presolve might have
    ls.constraint_prop.bounds_update.resize(*problem_ptr);
    ls.constraint_prop.conditional_bounds_update.update_constraint_bounds(
      *problem_ptr, ls.constraint_prop.bounds_update);
    if (!check_bounds_sanity(*problem_ptr)) { return false; }
  }
  stats.presolve_time = presolve_timer.elapsed_time();
  lp_optimal_solution.resize(problem_ptr->n_variables, problem_ptr->handle_ptr->get_stream());
  lp_dual_optimal_solution.resize(problem_ptr->n_constraints,
                                  problem_ptr->handle_ptr->get_stream());
  problem_ptr->handle_ptr->sync_stream();
  CUOPT_LOG_INFO("After trivial presolve #constraints %d #variables %d objective offset %f.",
                 problem_ptr->n_constraints,
                 problem_ptr->n_variables,
                 problem_ptr->presolve_data.objective_offset);
  return true;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::generate_quick_feasible_solution()
{
  solution_t<i_t, f_t> solution(*problem_ptr);
  // min 1 second, max 10 seconds
  const f_t generate_fast_solution_time =
    std::min(diversity_config.max_fast_sol_time, std::max(1., timer.remaining_time() / 20.));
  timer_t sol_timer(generate_fast_solution_time);
  // do very short LP run to get somewhere close to the optimal point
  ls.generate_fast_solution(solution, sol_timer);
  if (solution.get_feasible()) {
    population.run_solution_callbacks(solution);
    initial_sol_vector.emplace_back(std::move(solution));
    problem_ptr->handle_ptr->sync_stream();
    solution_t<i_t, f_t> searched_sol(initial_sol_vector.back());
    ls_config_t<i_t, f_t> ls_config;
    run_local_search(searched_sol, population.weights, sol_timer, ls_config);
    population.run_solution_callbacks(searched_sol);
    initial_sol_vector.emplace_back(std::move(searched_sol));
    auto& feas_sol = initial_sol_vector.back().get_feasible()
                       ? initial_sol_vector.back()
                       : initial_sol_vector[initial_sol_vector.size() - 2];
    CUOPT_LOG_INFO("Generated fast solution in %f seconds with objective %f",
                   timer.elapsed_time(),
                   feas_sol.get_user_objective());
  }
  problem_ptr->handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::check_b_b_preemption()
{
  if (population.preempt_heuristic_solver_.load()) {
    if (population.current_size() == 0) { population.allocate_solutions(); }
    auto new_sol_vector = population.get_external_solutions();
    population.add_solutions_from_vec(std::move(new_sol_vector));
    return true;
  }
  return false;
}

// returns the best feasible solution
template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_fj_alone(solution_t<i_t, f_t>& solution)
{
  CUOPT_LOG_INFO("Running FJ alone!");
  solution.round_nearest();
  ls.fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  ls.fj.settings.n_of_minimums_for_exit = 20000 * 1000;
  ls.fj.settings.update_weights         = true;
  ls.fj.settings.feasibility_run        = false;
  ls.fj.settings.time_limit             = timer.remaining_time();
  ls.fj.solve(solution);
  CUOPT_LOG_INFO("FJ alone finished!");
}

// returns the best feasible solution
template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_fp_alone(solution_t<i_t, f_t>& solution)
{
  CUOPT_LOG_INFO("Running FP alone!");
  ls.run_fp(solution, timer, &population);
  CUOPT_LOG_INFO("FP alone finished!");
}

// returns the best feasible solution
template <typename i_t, typename f_t>
solution_t<i_t, f_t> diversity_manager_t<i_t, f_t>::run_solver()
{
  population.timer     = timer;
  const f_t time_limit = timer.remaining_time();
  const f_t lp_time_limit =
    std::min(diversity_config.max_time_on_lp, time_limit * diversity_config.time_ratio_on_init_lp);
  // to automatically compute the solving time on scope exit
  auto timer_raii_guard =
    cuopt::scope_guard([&]() { stats.total_solve_time = timer.elapsed_time(); });
  // after every change to the problem, we should resize all the relevant vars
  // we need to encapsulate that to prevent repetitions

  ls.resize_vectors(*problem_ptr, problem_ptr->handle_ptr);
  ls.constraint_prop.bounds_update.resize(*problem_ptr);
  problem_ptr->check_problem_representation(true);
  // have the structure ready for reusing later
  problem_ptr->compute_integer_fixed_problem();

  // test problem is not ii
  cuopt_func_call(
    ls.constraint_prop.bounds_update.calculate_activity_on_problem_bounds(*problem_ptr));
  cuopt_assert(
    ls.constraint_prop.bounds_update.calculate_infeasible_redundant_constraints(*problem_ptr),
    "The problem must not be ii");
  population.initialize_population();
  if (check_b_b_preemption()) { return population.best_feasible(); }
  // before probing cache or LP, run FJ to generate initial primal feasible solution
  if (!from_dir && !fj_only_run) { generate_quick_feasible_solution(); }
  const f_t time_ratio_of_probing_cache = diversity_config.time_ratio_of_probing_cache;
  const f_t max_time_on_probing         = diversity_config.max_time_on_probing;
  f_t time_for_probing_cache =
    std::min(max_time_on_probing, time_limit * time_ratio_of_probing_cache);
  timer_t probing_timer{time_for_probing_cache};
  if (check_b_b_preemption()) { return population.best_feasible(); }
  if (!fj_only_run) {
    compute_probing_cache(ls.constraint_prop.bounds_update, *problem_ptr, probing_timer);
  }

  if (check_b_b_preemption()) { return population.best_feasible(); }
  lp_state_t<i_t, f_t>& lp_state = problem_ptr->lp_state;
  // resize because some constructor might be called before the presolve
  lp_state.resize(*problem_ptr, problem_ptr->handle_ptr->get_stream());
  bool bb_thread_solution_exists = simplex_solution_exists.load();
  if (bb_thread_solution_exists) {
    ls.lp_optimal_exists = true;
  } else if (!fj_only_run) {
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit            = lp_time_limit;
    lp_settings.tolerance             = context.settings.tolerances.absolute_tolerance;
    lp_settings.return_first_feasible = false;
    lp_settings.save_state            = true;
    lp_settings.concurrent_halt       = &global_concurrent_halt;
    lp_settings.has_initial_primal    = false;
    rmm::device_uvector<f_t> lp_optimal_solution_copy(lp_optimal_solution.size(),
                                                      problem_ptr->handle_ptr->get_stream());
    auto lp_result =
      get_relaxed_lp_solution(*problem_ptr, lp_optimal_solution_copy, lp_state, lp_settings);
    {
      std::lock_guard<std::mutex> guard(relaxed_solution_mutex);
      if (!simplex_solution_exists.load()) {
        raft::copy(lp_optimal_solution.data(),
                   lp_optimal_solution_copy.data(),
                   lp_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
      } else {
        // copy the lp state
        raft::copy(lp_state.prev_primal.data(),
                   lp_optimal_solution.data(),
                   lp_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
        raft::copy(lp_state.prev_dual.data(),
                   lp_dual_optimal_solution.data(),
                   lp_dual_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
      }
    }
    problem_ptr->handle_ptr->sync_stream();
    ls.lp_optimal_exists = true;
    if (lp_result.get_termination_status() == pdlp_termination_status_t::Optimal) {
      set_new_user_bound(lp_result.get_objective_value());
    } else if (lp_result.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible) {
      CUOPT_LOG_ERROR("Problem is primal infeasible, continuing anyway!");
      ls.lp_optimal_exists = false;
    } else if (lp_result.get_termination_status() == pdlp_termination_status_t::DualInfeasible) {
      CUOPT_LOG_ERROR("PDLP detected dual infeasibility, continuing anyway!");
      ls.lp_optimal_exists = false;
    } else if (lp_result.get_termination_status() == pdlp_termination_status_t::TimeLimit) {
      CUOPT_LOG_DEBUG(
        "Initial LP run exceeded time limit, continuing solver with partial LP result!");
      // note to developer, in debug mode the LP run might be too slow and it might cause PDLP not
      // to bring variables within the bounds
    }
    // in case the pdlp returned var boudns that are out of bounds
    clamp_within_var_bounds(lp_optimal_solution, problem_ptr, problem_ptr->handle_ptr);
  }

  population.allocate_solutions();
  population.add_solutions_from_vec(std::move(initial_sol_vector));
  if (check_b_b_preemption()) { return population.best_feasible(); }

  if (context.settings.benchmark_info_ptr != nullptr) {
    context.settings.benchmark_info_ptr->objective_of_initial_population =
      population.best_feasible().get_user_objective();
  }

  if (fj_only_run) {
    solution_t<i_t, f_t> sol(*problem_ptr);
    run_fj_alone(sol);
    return sol;
  }

  auto sol = generate_solution(timer.remaining_time(), false);
  population.add_solution(std::move(solution_t<i_t, f_t>(sol)));
  if (timer.check_time_limit()) {
    auto new_sol_vector = population.get_external_solutions();
    population.add_solutions_from_vec(std::move(new_sol_vector));
    return population.best_feasible();
  }
  run_fp_alone(sol);
  population.update_weights();

  if (timer.check_time_limit()) {
    auto new_sol_vector = population.get_external_solutions();
    population.add_solutions_from_vec(std::move(new_sol_vector));
    return population.best_feasible();
  }
  main_loop();

  return population.best_feasible();
};

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::diversity_step()
{
  // TODO when the solver is faster, increase this number
  const i_t max_iterations_without_improvement =
    diversity_config.max_iterations_without_improvement;
  bool improved = true;
  while (improved) {
    int k    = max_iterations_without_improvement;
    improved = false;
    while (k-- > 0) {
      if (check_b_b_preemption()) { return; }
      auto new_sol_vector = population.get_external_solutions();
      recombine_and_ls_with_all(new_sol_vector);
      population.adjust_weights_according_to_best_feasible();
      cuopt_assert(population.test_invariant(), "");
      if (population.current_size() < 2) {
        CUOPT_LOG_DEBUG("Population degenerated in diversity step");
        return;
      }
      if (timer.check_time_limit()) return;
      constexpr bool tournament = true;
      auto [sol1, sol2]         = population.get_two_random(tournament);
      cuopt_assert(population.test_invariant(), "");
      auto [lp_offspring, offspring] = recombine_and_local_search(sol1, sol2);
      i_t inserted_pos_1             = population.add_solution(std::move(lp_offspring));
      i_t inserted_pos_2             = population.add_solution(std::move(offspring));
      cuopt_assert(population.test_invariant(), "");
      if ((inserted_pos_1 != -1 && inserted_pos_1 <= 3) ||
          (inserted_pos_2 != -1 && inserted_pos_2 <= 3)) {
        improved = true;
        recombine_stats.print();
        break;
      }
    }
  }
  recombine_stats.print();
}

// TODO check if the new bound is actually better than the previous one.
// consider max problems too!
template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::set_new_user_bound(f_t new_bound)
{
  stats.solution_bound = new_bound;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::recombine_and_ls_with_all(solution_t<i_t, f_t>& solution,
                                                              bool add_only_feasible)
{
  raft::common::nvtx::range fun_scope("recombine_and_ls_with_all");
  if (population.population_hash_map.check_skip_solution(solution, 1)) { return; }
  auto population_vector = population.population_to_vector();
  for (auto& curr_sol : population_vector) {
    for (const auto recombiner_type : recombiner_types) {
      if (check_b_b_preemption()) { return; }
      if (curr_sol.get_feasible()) {
        auto [offspring, lp_offspring] =
          recombine_and_local_search(curr_sol, solution, recombiner_type);
        if (!add_only_feasible || lp_offspring.get_feasible()) {
          population.add_solution(std::move(lp_offspring));
        }
        if (!add_only_feasible || offspring.get_feasible()) {
          population.add_solution(std::move(offspring));
        }
        if (timer.check_time_limit()) { return; }
      }
    }
  }
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::recombine_and_ls_with_all(
  std::vector<solution_t<i_t, f_t>>& solutions, bool add_only_feasible)
{
  raft::common::nvtx::range fun_scope("recombine_and_ls_with_all");
  if (solutions.size() > 0) {
    CUOPT_LOG_INFO("Running recombiners on B&B solutions with size %lu", solutions.size());
    // add all solutions because time limit might have been consumed and we might have exited before
    for (auto& sol : solutions) {
      cuopt_func_call(sol.test_feasibility(true));
      population.add_solution(std::move(solution_t<i_t, f_t>(sol)));
    }
    for (auto& sol : solutions) {
      if (timer.check_time_limit()) { return; }
      solution_t<i_t, f_t> ls_solution(sol);
      ls_config_t<i_t, f_t> ls_config;
      run_local_search(ls_solution, population.weights, timer, ls_config);
      if (timer.check_time_limit()) { return; }
      // TODO try if running LP with integers fixed makes it feasible
      if (ls_solution.get_feasible()) {
        CUOPT_LOG_DEBUG("External LS searched solution feasible, running recombiners!");
        recombine_and_ls_with_all(ls_solution, add_only_feasible);
      } else {
        CUOPT_LOG_DEBUG("External solution feasible, running recombiners!");
        recombine_and_ls_with_all(sol, add_only_feasible);
      }
    }
  }
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::main_loop()
{
  population.start_threshold_adjustment();
  recombine_stats.reset();
  population.print();
  while (true) {
    if (check_b_b_preemption()) { break; }
    CUOPT_LOG_DEBUG("Running a new step");
    bool enough_solutions = regenerate_solutions();
    if (!enough_solutions) {
      // do a longer search on the best solution then exit
      auto best_sol = population.is_feasible() ? population.best_feasible() : population.best();
      ls.run_fj_until_timer(best_sol, population.weights, timer);
      population.add_solution(std::move(best_sol));
      CUOPT_LOG_WARN("Enough solutions couldn't be generated,exiting heuristics!");
      break;
    }
    if (timer.check_time_limit()) { break; }
    diversity_step();
    if (timer.check_time_limit()) { break; }

    if (diversity_config.halve_population) {
      population.adjust_threshold(timer);
      i_t prev_threshold = population.var_threshold;
      population.halve_the_population();
      auto new_solutions      = generate_more_solutions();
      auto current_population = population.population_to_vector();
      population.clear();
      current_population.insert(current_population.end(),
                                std::make_move_iterator(new_solutions.begin()),
                                std::make_move_iterator(new_solutions.end()));
      population.find_diversity(current_population, diversity_config.use_avg_diversity);
      // if the threshold is lower than the threshold we progress with time
      // set it to the higher threshold
      population.add_solutions_from_vec(std::move(current_population));
    } else {
      // increase the threshold/decrease the diversity
      population.adjust_threshold(timer);
    }
    // idea to try, we can average the weights of the new solutions
    population.update_weights();
    population.print();
    if (timer.check_time_limit()) { break; }
  }
  auto new_sol_vector = population.get_external_solutions();
  recombine_and_ls_with_all(new_sol_vector);
  population.print();
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::check_better_than_both(solution_t<i_t, f_t>& offspring,
                                                           solution_t<i_t, f_t>& sol1,
                                                           solution_t<i_t, f_t>& sol2)
{
  bool better_than_both = false;
  if (sol1.get_feasible() && sol2.get_feasible()) {
    better_than_both = offspring.get_objective() <
                       (std::min(sol1.get_objective(), sol2.get_objective()) - OBJECTIVE_EPSILON);
  } else if (sol1.get_feasible()) {
    better_than_both = offspring.get_objective() < (sol1.get_objective() - OBJECTIVE_EPSILON);
  } else if (sol2.get_feasible()) {
    better_than_both = offspring.get_objective() < (sol2.get_objective() - OBJECTIVE_EPSILON);
  } else {
    better_than_both = offspring.get_feasible();
  }
  if (offspring.get_feasible() && better_than_both) {
    context.settings.benchmark_info_ptr->last_improvement_after_recombination =
      timer.elapsed_time();
  }
}

template <typename i_t, typename f_t>
std::pair<solution_t<i_t, f_t>, solution_t<i_t, f_t>>
diversity_manager_t<i_t, f_t>::recombine_and_local_search(solution_t<i_t, f_t>& sol1,
                                                          solution_t<i_t, f_t>& sol2,
                                                          recombiner_enum_t recombiner_type)
{
  raft::common::nvtx::range fun_scope("recombine_and_local_search");
  CUOPT_LOG_DEBUG("Recombining sol cost:feas %f : %d and %f : %d",
                  sol1.get_quality(population.weights),
                  sol1.get_feasible(),
                  sol2.get_quality(population.weights),
                  sol2.get_feasible());
  double best_objective_of_parents  = std::min(sol1.get_objective(), sol2.get_objective());
  bool at_least_one_parent_feasible = sol1.get_feasible() || sol2.get_feasible();
  // randomly choose among 3 recombiners
  auto [offspring, success] = recombine(sol1, sol2, recombiner_type);
  if (!success) {
    // add the attempt
    mab_recombiner.add_mab_reward(static_cast<int>(recombine_stats.get_last_attempt()),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::max(),
                                  recombiner_work_normalized_reward_t(0.0));
    return std::make_pair(solution_t<i_t, f_t>(sol1), solution_t<i_t, f_t>(sol2));
  }
  cuopt_assert(population.test_invariant(), "");
  cuopt_func_call(offspring.test_variable_bounds(false));
  CUOPT_LOG_DEBUG("Recombiner offspring sol cost:feas %f : %d",
                  offspring.get_quality(population.weights),
                  offspring.get_feasible());
  cuopt_assert(offspring.test_number_all_integer(), "All must be integers before LS");
  bool feasibility_before = offspring.get_feasible();
  ls_config_t<i_t, f_t> ls_config;
  ls_config.best_objective_of_parents    = best_objective_of_parents;
  ls_config.at_least_one_parent_feasible = at_least_one_parent_feasible;
  success = this->run_local_search(offspring, population.weights, timer, ls_config);
  if (!success) {
    // add the attempt
    mab_recombiner.add_mab_reward(static_cast<int>(recombine_stats.get_last_attempt()),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::max(),
                                  recombiner_work_normalized_reward_t(0.0));
    return std::make_pair(solution_t<i_t, f_t>(sol1), solution_t<i_t, f_t>(sol2));
  }
  cuopt_assert(offspring.test_number_all_integer(), "All must be integers after LS");
  cuopt_assert(population.test_invariant(), "");
  offspring.compute_feasibility();
  CUOPT_LOG_DEBUG("After LS offspring sol cost:feas %f : %d",
                  offspring.get_quality(population.weights),
                  offspring.get_feasible());
  cuopt_assert(population.test_invariant(), "");
  // run LP with the vars
  solution_t<i_t, f_t> lp_offspring(offspring);
  cuopt_assert(population.test_invariant(), "");
  cuopt_assert(lp_offspring.test_number_all_integer(), "All must be integers before LP");
  f_t lp_run_time = offspring.get_feasible() ? diversity_config.lp_run_time_if_feasible
                                             : diversity_config.lp_run_time_if_infeasible;
  lp_run_time     = std::min(lp_run_time, timer.remaining_time());
  relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit              = lp_run_time;
  lp_settings.tolerance               = context.settings.tolerances.absolute_tolerance;
  lp_settings.return_first_feasible   = false;
  lp_settings.save_state              = true;
  lp_settings.per_constraint_residual = true;
  run_lp_with_vars_fixed(*lp_offspring.problem_ptr,
                         lp_offspring,
                         lp_offspring.problem_ptr->integer_indices,
                         lp_settings,
                         &ls.constraint_prop.bounds_update,
                         true /* check fixed assignment is feasible */,
                         true /* use integer fixed problem */);
  cuopt_assert(population.test_invariant(), "");
  cuopt_assert(lp_offspring.test_number_all_integer(), "All must be integers after LP");
  f_t lp_qual = lp_offspring.get_quality(population.weights);
  CUOPT_LOG_DEBUG("After LP offspring sol cost:feas %f : %d", lp_qual, lp_offspring.get_feasible());
  f_t offspring_qual = std::min(offspring.get_quality(population.weights), lp_qual);
  recombine_stats.update_improve_stats(
    offspring_qual, sol1.get_quality(population.weights), sol2.get_quality(population.weights));
  f_t best_quality_of_parents =
    std::min(sol1.get_quality(population.weights), sol2.get_quality(population.weights));
  mab_recombiner.add_mab_reward(
    static_cast<int>(recombine_stats.get_last_attempt()),
    best_quality_of_parents,
    population.best().get_quality(population.weights),
    offspring_qual,
    recombiner_work_normalized_reward_t(recombine_stats.get_last_recombiner_time()));
  mab_ls.add_mab_reward(mab_ls_config_t<i_t, f_t>::last_ls_mab_option,
                        best_quality_of_parents,
                        population.best_feasible().get_quality(population.weights),
                        offspring_qual,
                        ls_work_normalized_reward_t(mab_ls_config_t<i_t, f_t>::last_lm_config));
  if (context.settings.benchmark_info_ptr != nullptr) {
    check_better_than_both(offspring, sol1, sol2);
    check_better_than_both(lp_offspring, sol1, sol2);
  }
  return std::make_pair(std::move(offspring), std::move(lp_offspring));
}

template <typename i_t, typename f_t>
std::pair<solution_t<i_t, f_t>, bool> diversity_manager_t<i_t, f_t>::recombine(
  solution_t<i_t, f_t>& a, solution_t<i_t, f_t>& b, recombiner_enum_t recombiner_type)
{
  recombiner_enum_t recombiner;
  if (run_only_ls_recombiner) {
    recombiner = recombiner_enum_t::LINE_SEGMENT;
  } else if (run_only_bp_recombiner) {
    recombiner = recombiner_enum_t::BOUND_PROP;
  } else if (run_only_fp_recombiner) {
    recombiner = recombiner_enum_t::FP;
  } else if (run_only_sub_mip_recombiner) {
    recombiner = recombiner_enum_t::SUB_MIP;
  } else {
    // only run the given recombiner unless it is defult
    if (recombiner_type == recombiner_enum_t::SIZE) {
      recombiner = static_cast<recombiner_enum_t>(mab_recombiner.select_mab_option());
    } else {
      recombiner = recombiner_type;
    }
  }
  recombine_stats.add_attempt((recombiner_enum_t)recombiner);
  recombine_stats.start_recombiner_time();
  // Refactored code using a switch statement
  switch (recombiner) {
    case recombiner_enum_t::BOUND_PROP: {
      auto [sol, success] = bound_prop_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::FP: {
      auto [sol, success] = fp_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::LINE_SEGMENT: {
      auto [sol, success] = line_segment_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::SUB_MIP: {
      auto [sol, success] = sub_mip_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::SIZE: {
      CUOPT_LOG_ERROR("Invalid or unhandled recombiner type: %d", recombiner);
      return std::make_pair(solution_t<i_t, f_t>(a), false);
    }
  }
  CUOPT_LOG_ERROR("Invalid or unhandled recombiner type: %d", recombiner);
  return std::make_pair(solution_t<i_t, f_t>(a), false);
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::set_simplex_solution(const std::vector<f_t>& solution,
                                                         const std::vector<f_t>& dual_solution,
                                                         f_t objective)
{
  CUOPT_LOG_DEBUG("Setting simplex solution with objective %f", objective);
  using sol_t = solution_t<i_t, f_t>;
  cudaSetDevice(context.handle_ptr->get_device());
  context.handle_ptr->sync_stream();
  cuopt_func_call(sol_t new_sol(*problem_ptr));
  cuopt_assert(new_sol.assignment.size() == solution.size(), "Assignment size mismatch");
  cuopt_assert(problem_ptr->n_constraints == dual_solution.size(), "Dual assignment size mismatch");
  cuopt_func_call(new_sol.copy_new_assignment(solution));
  cuopt_func_call(new_sol.compute_feasibility());
  cuopt_assert(integer_equal(new_sol.get_user_objective(), objective, 1e-3), "Objective mismatch");
  std::lock_guard<std::mutex> lock(relaxed_solution_mutex);
  simplex_solution_exists = true;
  global_concurrent_halt.store(1, std::memory_order_release);
  // it is safe to use lp_optimal_solution while executing the copy operation
  // the operations are ordered as long as they are on the same stream
  raft::copy(
    lp_optimal_solution.data(), solution.data(), solution.size(), context.handle_ptr->get_stream());
  raft::copy(lp_dual_optimal_solution.data(),
             dual_solution.data(),
             dual_solution.size(),
             context.handle_ptr->get_stream());
  set_new_user_bound(objective);
  context.handle_ptr->sync_stream();
}

#if MIP_INSTANTIATE_FLOAT
template class diversity_manager_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class diversity_manager_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
