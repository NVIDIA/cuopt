/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "fix_and_propagate.cuh"
#include "fix_propagate_host.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/problem_helpers.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <pdlp/solve.cuh>
#include <utilities/copy_helpers.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <random>
#include <string>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
fix_and_propagate_t<i_t, f_t>::fix_and_propagate_t(mip_solver_context_t<i_t, f_t>& context_,
                                                   problem_t<i_t, f_t>* problem_ptr_,
                                                   population_t<i_t, f_t>& population_,
                                                   local_search_t<i_t, f_t>& local_search_,
                                                   cuopt::timer_t& timer_,
                                                   const rmm::device_uvector<f_t>& lp_primal,
                                                   const rmm::device_uvector<f_t>& lp_dual,
                                                   std::function<bool()> preempted_)
  : context(context_),
    problem_ptr(problem_ptr_),
    population(population_),
    local_search(local_search_),
    timer(timer_),
    lp_optimal_solution(lp_primal),
    lp_dual_optimal_solution(lp_dual),
    preempted(std::move(preempted_))
{
}

template <typename i_t, typename f_t>
typename fix_and_propagate_t<i_t, f_t>::pattern_eval_t
fix_and_propagate_t<i_t, f_t>::eval_fixed_pattern(const std::vector<f_t>& assignment,
                                                  f_t tolerance,
                                                  f_t time_limit,
                                                  const std::vector<f_t>* warm_primal,
                                                  const std::vector<f_t>* warm_dual,
                                                  bool repair,
                                                  f_t repair_time_limit,
                                                  f_t repair_below,
                                                  f_t max_push_time)
{
  raft::common::nvtx::range fun_scope("eval_fixed_pattern");
  pattern_eval_t result;
  auto* handle_ptr             = problem_ptr->handle_ptr;
  auto stream                  = handle_ptr->get_stream();
  const i_t n_constraints      = problem_ptr->n_constraints;
  const f_t absolute_tolerance = problem_ptr->tolerances.absolute_tolerance;
  const double start_time      = timer.elapsed_time();
  solution_t<i_t, f_t> solution(*problem_ptr);
  solution.copy_new_assignment(assignment);
  auto [fixed_problem, fixed_assignment, variable_map] =
    solution.fix_variables(problem_ptr->integer_indices);
  fixed_problem.check_problem_representation(true);
  if (fixed_problem.n_constraints != n_constraints) { return result; }
  const bool use_warm_start = warm_primal != nullptr && warm_dual != nullptr &&
                              warm_primal->size() == (size_t)fixed_problem.n_variables &&
                              warm_dual->size() == (size_t)n_constraints;
  // Stage 1: ranking LP (the stages are described at pattern_eval_t in fix_and_propagate.cuh)
  {
    pdlp_solver_settings_t<i_t, f_t> lp_settings{};
    lp_settings.set_optimality_tolerance(tolerance);
    lp_settings.time_limit = std::max<double>(1., time_limit);
    lp_settings.method     = method_t::PDLP;
    if (use_warm_start) { lp_settings.presolver = presolver_t::None; }
    std::atomic<int> local_halt{0};
    lp_settings.concurrent_halt = &local_halt;
    lp_settings.inside_mip      = false;
    lp_settings.num_gpus        = context.settings.num_gpus;
    if (use_warm_start) {
      raft::copy(fixed_assignment.data(), warm_primal->data(), warm_primal->size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(
        fixed_assignment.data(), (i_t)fixed_assignment.size(), stream);
      lp_settings.set_initial_dual_solution(warm_dual->data(), (i_t)warm_dual->size(), stream);
    }
    auto lp_result =
      solve_lp_with_method<i_t, f_t>(fixed_problem, lp_settings, timer_t(lp_settings.time_limit));
    const auto& termination_info = lp_result.get_additional_termination_information();
    result.iterations            = termination_info.number_of_steps_taken;
    result.l2_primal_residual    = termination_info.l2_primal_residual;
    result.optimal = lp_result.get_termination_status() == pdlp_termination_status_t::Optimal;
    if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables ||
        lp_result.get_dual_solution().size() != (size_t)n_constraints) {
      result.time = timer.elapsed_time() - start_time;
      return result;
    }
    result.primal = cuopt::host_copy(lp_result.get_primal_solution(), stream);
    result.dual   = cuopt::host_copy(lp_result.get_dual_solution(), stream);
    raft::copy(fixed_assignment.data(),
               lp_result.get_primal_solution().data(),
               fixed_assignment.size(),
               stream);
  }
  clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
  solution.unfix_variables(fixed_assignment, variable_map);
  // full objective (fixed integers included) of the (slightly infeasible) LP point
  result.lp_objective = solution.get_user_objective();
  result.usable       = std::isfinite(result.lp_objective);
  // the repair rarely improves the objective of the LP point, so a point that does not beat the
  // incumbent cannot be published: skip the push and the repair (solver-space objectives)
  if (result.usable && population.is_feasible()) {
    const f_t incumbent = population.best_feasible().get_objective();
    if (solution.get_objective() > incumbent + f_t(1e-4) * std::max<f_t>(1., std::abs(incumbent))) {
      result.cannot_beat_incumbent = true;
      result.time                  = timer.elapsed_time() - start_time;
      return result;
    }
  }
  if (repair && result.usable && result.lp_objective < repair_below) {
    // Stage 2: pushes. The evaluated LP point can stall with a sizable primal residual, and
    // repairing it directly costs objective; warm-started pushes with a frozen large primal weight
    // first drive the residual down at a nearly unchanged objective, so the repair moves the point
    // less (the ranking objective stays the stage-1 one)
    const double push_start_time = timer.elapsed_time();
    struct push_t {
      double weight;
      double cap;
    };
    const double push_scale = std::clamp<double>(max_push_time / 11., 0., 1.);
    const push_t pushes[]   = {{1e4, 6. * push_scale}, {1e5, 5. * push_scale}};
    auto current_primal     = cuopt::device_copy(result.primal, stream);
    auto current_dual       = cuopt::device_copy(result.dual, stream);
    bool pushed             = false;
    for (const auto& push : pushes) {
      if (timer.check_time_limit() || check_b_b_preemption() || timer.remaining_time() < 30.) {
        break;
      }
      pdlp_solver_settings_t<i_t, f_t> lp_settings{};
      lp_settings.set_optimality_tolerance(1e-7);
      lp_settings.time_limit = push.cap;
      lp_settings.method     = method_t::PDLP;
      lp_settings.presolver  = presolver_t::None;
      std::atomic<int> local_halt{0};
      lp_settings.concurrent_halt = &local_halt;
      lp_settings.inside_mip      = false;
      lp_settings.num_gpus        = context.settings.num_gpus;
      raft::copy(fixed_assignment.data(), current_primal.data(), current_primal.size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(
        fixed_assignment.data(), (i_t)fixed_assignment.size(), stream);
      lp_settings.set_initial_dual_solution(current_dual.data(), (i_t)current_dual.size(), stream);
      lp_settings.set_initial_primal_weight(push.weight);
      lp_settings.hyper_params.restart_k_p = 0.;
      lp_settings.hyper_params.restart_k_i = 0.;
      lp_settings.hyper_params.restart_k_d = 0.;
      auto lp_result =
        solve_lp_with_method<i_t, f_t>(fixed_problem, lp_settings, timer_t(lp_settings.time_limit));
      if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables ||
          lp_result.get_dual_solution().size() != (size_t)n_constraints) {
        break;
      }
      result.pushed_l2_primal_residual =
        lp_result.get_additional_termination_information().l2_primal_residual;
      raft::copy(current_primal.data(),
                 lp_result.get_primal_solution().data(),
                 current_primal.size(),
                 stream);
      raft::copy(
        current_dual.data(), lp_result.get_dual_solution().data(), current_dual.size(), stream);
      pushed = true;
    }
    if (pushed) {
      raft::copy(fixed_assignment.data(), current_primal.data(), current_primal.size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      solution.unfix_variables(fixed_assignment, variable_map);
      result.pushed_lp_objective = solution.get_user_objective();
    }
    result.push_time = timer.elapsed_time() - push_start_time;
    bool is_feasible = solution.get_feasible();
    // Stage 3: repair, a zero-objective LP that only seeks feasibility
    if (!is_feasible) {
      timer_t repair_timer(repair_time_limit);
      thrust::fill(handle_ptr->get_thrust_policy(),
                   fixed_problem.objective_coefficients.begin(),
                   fixed_problem.objective_coefficients.end(),
                   f_t(0));
      fixed_problem.presolve_data.objective_offset = 0;
      auto& lp_state                               = fixed_problem.lp_state;
      lp_state.resize(fixed_problem, stream);
      thrust::fill(handle_ptr->get_thrust_policy(),
                   lp_state.prev_dual.begin(),
                   lp_state.prev_dual.end(),
                   f_t(0));
      relaxed_lp_settings_t repair_settings;
      repair_settings.tolerance               = 0.5 * absolute_tolerance;
      repair_settings.return_first_feasible   = true;
      repair_settings.save_state              = true;
      repair_settings.check_infeasibility     = false;
      repair_settings.per_constraint_residual = true;
      repair_settings.has_initial_primal      = true;
      for (int chunk_id = 0; chunk_id < 3 && !is_feasible; ++chunk_id) {
        if (repair_timer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption()) {
          break;
        }
        repair_settings.time_limit =
          std::min<double>(repair_timer.remaining_time(), timer.remaining_time() - 1.);
        if (repair_settings.time_limit < 1.) { break; }
        get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, repair_settings);
        solution.unfix_variables(fixed_assignment, variable_map);
        is_feasible = solution.get_feasible();
      }
    }
    if (is_feasible) {
      result.repaired_objective  = solution.get_user_objective();
      result.repaired_assignment = solution.get_host_assignment();
      // like the other heuristics, hand every feasible solution to the population and let it decide
      result.published = population.add_solution(solution_t<i_t, f_t>(solution)).second;
      if (result.published) {
        CUOPT_LOG_INFO(
          "New solution from primal heuristics (CPU fix-and-propagate). Objective %+.6e. Time %.3f",
          result.repaired_objective,
          timer.elapsed_time());
      }
    }
  }
  result.time = timer.elapsed_time() - start_time;
  return result;
}

template <typename i_t, typename f_t>
bool fix_and_propagate_t<i_t, f_t>::run()
{
  raft::common::nvtx::range fun_scope("run_root_fix_and_propagate");
  if (problem_ptr->n_integer_vars == 0 || problem_ptr->n_integer_vars == problem_ptr->n_variables) {
    return false;
  }
  const auto& heuristic_params = context.settings.heuristic_params;
  const f_t max_budget =
    std::min<f_t>(heuristic_params.fix_propagate_max_time,
                  heuristic_params.fix_propagate_time_ratio * context.settings.time_limit);
  if (max_budget <= 1.) { return false; }
  CUOPT_LOG_DEBUG(
    "Root fix-and-propagate start: budget=%.1fs elapsed=%.2f", max_budget, timer.elapsed_time());
  auto* handle_ptr = problem_ptr->handle_ptr;
  auto stream      = handle_ptr->get_stream();
  const f_t budget = std::min<f_t>(timer.remaining_time() * 0.8, max_budget);
  timer_t fix_and_propagate_timer(budget);
  const f_t integrality_tolerance = problem_ptr->tolerances.integrality_tolerance;
  const double build_start_time   = timer.elapsed_time();

  // host copies of the model and the LP optimum
  const i_t n_variables  = problem_ptr->n_variables;
  auto h_integer_indices = cuopt::host_copy(problem_ptr->integer_indices, stream);
  auto [h_variable_lower_bounds, h_variable_upper_bounds] =
    cuopt::extract_host_bounds<f_t>(problem_ptr->variable_bounds, handle_ptr);
  // reference point: the root LP optimum
  auto h_lp_solution = cuopt::host_copy(lp_optimal_solution, stream);
  h_fix_propagate_t<i_t, f_t> h_fix_propagate;
  h_fix_propagate.column_offsets     = cuopt::host_copy(problem_ptr->reverse_offsets, stream);
  h_fix_propagate.column_rows        = cuopt::host_copy(problem_ptr->reverse_constraints, stream);
  h_fix_propagate.column_values      = cuopt::host_copy(problem_ptr->reverse_coefficients, stream);
  h_fix_propagate.absolute_tolerance = problem_ptr->tolerances.absolute_tolerance;
  h_fix_propagate.integrality_tolerance = problem_ptr->tolerances.integrality_tolerance;
  h_fix_propagate.continuous_min_improvement =
    heuristic_params.fix_propagate_continuous_min_improvement;
  {
    simplex::user_problem_t<i_t, f_t> user_problem(problem_ptr->handle_ptr);
    problem_ptr->get_host_user_problem(user_problem);
    h_fix_propagate.build_domain(user_problem, problem_ptr->clique_table.get());
  }
  // a pass may process max(3e8, 20 x the model's nonzeros) nonzeros before it gives up, so that
  // a pass on a model with very dense integer rows cannot hold local search for long (Arow is
  // built by build_domain)
  constexpr int64_t pass_work_floor   = 300000000;
  constexpr int64_t pass_work_per_nnz = 20;
  const int64_t model_nnz    = std::max<int64_t>(1, (int64_t)h_fix_propagate.Arow.row_start.back());
  h_fix_propagate.work_limit = std::max(pass_work_floor, pass_work_per_nnz * model_nnz);
  i_t n_fractional = 0, n_at_one = 0;
  for (auto variable : h_integer_indices) {
    f_t reference_value = h_lp_solution[variable];
    if ((reference_value - std::floor(reference_value) > integrality_tolerance) &&
        (std::ceil(reference_value) - reference_value > integrality_tolerance))
      n_fractional++;
    if (reference_value >= 1. - integrality_tolerance) n_at_one++;
  }
  CUOPT_LOG_DEBUG(
    "Fix-and-propagate setup: budget=%.1fs n_int=%zu lp_fractional=%d lp_ge_one=%d build=%.2fs "
    "elapsed=%.2f",
    budget,
    h_integer_indices.size(),
    n_fractional,
    n_at_one,
    timer.elapsed_time() - build_start_time,
    timer.elapsed_time());

  // LP-guided fix-and-propagate: one pass with the configured variable / value strategies
  // (see fix_propagate_host.hpp), the pattern evaluated by a short cost-aware fixed LP (its
  // objective settles within a few thousand PDLP iterations even when the residual stalls),
  // then pushed, repaired and added to the population. All steps are scaled to the budget.
  auto h_objective_coefficients = cuopt::host_copy(problem_ptr->objective_coefficients, stream);
  auto h_lp_dual_solution       = cuopt::host_copy(lp_dual_optimal_solution, stream);
  std::vector<f_t> reduced_cost(n_variables, 0);
  std::vector<char> is_binary(n_variables, 0);
  for (auto variable : h_integer_indices) {
    f_t dual_activity = 0;
    for (i_t entry = h_fix_propagate.column_offsets[variable];
         entry < h_fix_propagate.column_offsets[variable + 1];
         ++entry) {
      const i_t row         = h_fix_propagate.column_rows[entry];
      const f_t coefficient = h_fix_propagate.column_values[entry];
      dual_activity += coefficient * h_lp_dual_solution[row];
    }
    reduced_cost[variable] = h_objective_coefficients[variable] - dual_activity;
    is_binary[variable] =
      h_variable_lower_bounds[variable] == 0 && h_variable_upper_bounds[variable] == 1;
  }
  struct candidate_t {
    std::string name;  // log name
    fix_propagate_variable_strategy_t var_strategy;
    fix_propagate_value_strategy_t value_strategy;
    f_t value_param;  // reduced-cost or fraction threshold of the value strategy
    unsigned seed;    // randomized value strategies (fix_propagate_randomized)
  };
  std::vector<std::vector<f_t>> seen;
  f_t best_lp_objective = std::numeric_limits<f_t>::infinity();
  int n_evaluated       = 0;
  std::string best_name;
  std::vector<f_t> best_repaired;
  constexpr f_t rank_tolerance = 1e-4;
  const f_t rank_time_limit    = std::min<f_t>(8., 0.25 * budget);
  const f_t push_time_limit    = std::min<f_t>(11., 0.3 * budget);
  // a successful repair finishes well inside this cap; a failing one uses all of it
  const f_t repair_time_limit = std::min<f_t>(5., 0.4 * budget);
  auto run_candidate = [&](const candidate_t& candidate, const std::vector<f_t>& reference_point) {
    if (fix_and_propagate_timer.check_time_limit() || timer.check_time_limit() ||
        check_b_b_preemption() || timer.remaining_time() < 2. * budget) {
      return;
    }
    std::vector<i_t> order(h_integer_indices.begin(), h_integer_indices.end());
    std::vector<f_t> key(n_variables, 0);
    std::mt19937 generator(candidate.seed);
    std::uniform_real_distribution<double> uniform(0., 1.);
    for (auto variable : h_integer_indices) {
      key[variable] = fix_propagate_order_key<f_t>(
        candidate.var_strategy, reference_point[variable], reduced_cost[variable]);
    }
    const bool group = fix_propagate_groups_by_type(candidate.var_strategy);
    std::stable_sort(order.begin(), order.end(), [&](i_t first, i_t second) {
      if (group && is_binary[first] != is_binary[second]) {
        return is_binary[first] > is_binary[second];
      }
      return key[first] > key[second];
    });
    std::vector<f_t> preferred_values(n_variables, 0.);
    const bool randomized = fix_propagate_randomized(candidate.value_strategy);
    for (auto variable : h_integer_indices) {
      const f_t uniform_draw = randomized ? f_t(uniform(generator)) : f_t(0);
      preferred_values[variable] =
        fix_propagate_preferred_value<f_t>(candidate.value_strategy,
                                           reference_point[variable],
                                           h_variable_lower_bounds[variable],
                                           h_variable_upper_bounds[variable],
                                           reduced_cost[variable],
                                           candidate.value_param,
                                           integrality_tolerance,
                                           uniform_draw,
                                           h_objective_coefficients[variable]);
    }
    const double pass_start_time = timer.elapsed_time();
    i_t n_flipped = 0, n_implied = 0;
    h_fix_propagate.work = 0;
    i_t n_conflicts =
      h_fix_propagate.fix_and_propagate(order, preferred_values, n_flipped, n_implied);
    std::vector<f_t> pattern(h_integer_indices.size());
    f_t integer_cost = 0;
    i_t n_ones       = 0;
    for (size_t i = 0; i < h_integer_indices.size(); ++i) {
      pattern[i] = h_fix_propagate.pass_values[h_integer_indices[i]];
      integer_cost += h_objective_coefficients[h_integer_indices[i]] * pattern[i];
      n_ones += pattern[i] > 0.5;
    }
    bool duplicate = false;
    for (auto& seen_pattern : seen) {
      if (seen_pattern == pattern) { duplicate = true; }
    }
    CUOPT_LOG_DEBUG(
      "Fix-and-propagate pass %s: time=%.3fs conflicts=%d flipped=%d implied_off_pref=%d ones=%d "
      "int_cost=%g dup=%d cont_tightenings=%ld work=%ld (%.1fx nnz) work_limit_hit=%d "
      "elapsed=%.2f",
      candidate.name.c_str(),
      timer.elapsed_time() - pass_start_time,
      n_conflicts,
      n_flipped,
      n_implied,
      n_ones,
      integer_cost,
      (int)duplicate,
      (long)h_fix_propagate.n_continuous_tightenings,
      (long)h_fix_propagate.work,
      (double)h_fix_propagate.work / (double)model_nnz,
      (int)h_fix_propagate.work_limit_hit,
      timer.elapsed_time());
    if (duplicate) { return; }
    seen.push_back(pattern);
    std::vector<f_t> assignment = reference_point;
    for (size_t i = 0; i < h_integer_indices.size(); ++i) {
      assignment[h_integer_indices[i]] = pattern[i];
    }
    // feasibility jump hand-off of a pattern the pass or its evaluation left infeasible
    auto fj_repair = [&]() {
      if (heuristic_params.fix_propagate_repair == 0 ||
          fix_and_propagate_timer.check_time_limit() || timer.check_time_limit() ||
          check_b_b_preemption()) {
        return;
      }
      auto repair_result = fj_repair_pattern(assignment, repair_time_limit);
      CUOPT_LOG_DEBUG("Fix-and-propagate FJ repair %s: feasible=%d obj=%g published=%d time=%.2fs",
                      candidate.name.c_str(),
                      (int)repair_result.feasible,
                      repair_result.objective,
                      (int)repair_result.published,
                      repair_result.time);
      if (repair_result.feasible && best_repaired.empty()) {
        best_name     = candidate.name + "+fj";
        best_repaired = std::move(repair_result.assignment);
      }
    };
    if (n_conflicts > 0) {
      fj_repair();
      return;
    }
    auto evaluation = eval_fixed_pattern(assignment,
                                         rank_tolerance,
                                         rank_time_limit,
                                         nullptr,
                                         nullptr,
                                         true,
                                         repair_time_limit,
                                         best_lp_objective,
                                         push_time_limit);
    n_evaluated++;
    const bool better = evaluation.usable && evaluation.lp_objective < best_lp_objective;
    CUOPT_LOG_DEBUG(
      "Pattern rank %s: ones=%d lp_obj=%g optimal=%d iters=%d l2_primal_res=%g time=%.2fs "
      "better=%d pushed_lp_obj=%g pushed_res=%g push_time=%.2fs repaired=%g published=%d "
      "cannot_beat=%d elapsed=%.2f",
      candidate.name.c_str(),
      n_ones,
      evaluation.lp_objective,
      (int)evaluation.optimal,
      evaluation.iterations,
      evaluation.l2_primal_residual,
      evaluation.time,
      (int)better,
      evaluation.pushed_lp_objective,
      evaluation.pushed_l2_primal_residual,
      evaluation.push_time,
      evaluation.repaired_objective,
      (int)evaluation.published,
      (int)evaluation.cannot_beat_incumbent,
      timer.elapsed_time());
    if (better && !evaluation.repaired_assignment.empty()) {
      best_lp_objective = evaluation.lp_objective;
      best_name         = candidate.name;
      best_repaired     = std::move(evaluation.repaired_assignment);
    } else if (evaluation.repaired_assignment.empty() && !evaluation.cannot_beat_incumbent) {
      fj_repair();
    }
  };
  const candidate_t configured{
    "lp_guided",
    static_cast<fix_propagate_variable_strategy_t>(
      heuristic_params.fix_propagate_variable_strategy),
    static_cast<fix_propagate_value_strategy_t>(heuristic_params.fix_propagate_value_strategy),
    heuristic_params.fix_propagate_value_param,
    12u};
  run_candidate(configured, h_lp_solution);
  if (best_repaired.empty()) {
    CUOPT_LOG_DEBUG("Root fix-and-propagate: no repaired pattern (evaluated %d) elapsed=%.2f",
                    n_evaluated,
                    timer.elapsed_time());
    return false;
  }
  CUOPT_LOG_DEBUG("Root fix-and-propagate best: %s lp_obj=%g evaluated=%d elapsed=%.2f",
                  best_name.c_str(),
                  best_lp_objective,
                  n_evaluated,
                  timer.elapsed_time());
  // the pattern was already pushed, repaired and added to the population by its evaluation
  return true;
}

template <typename i_t, typename f_t>
typename fix_and_propagate_t<i_t, f_t>::fj_repair_t
fix_and_propagate_t<i_t, f_t>::fj_repair_pattern(const std::vector<f_t>& assignment, f_t time_limit)
{
  raft::common::nvtx::range fun_scope("fj_repair_pattern");
  fj_repair_t result;
  const double start_time = timer.elapsed_time();
  timer_t repair_timer(time_limit);
  solution_t<i_t, f_t> solution(*problem_ptr);
  solution.copy_new_assignment(assignment);
  // as for the rounded root LP point in local search: complete the continuous part with a short
  // fixed-integer LP, then let feasibility jump move every variable
  if (problem_ptr->n_integer_vars < problem_ptr->n_variables) {
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit = std::min<f_t>(2., 0.4 * time_limit);
    lp_settings.tolerance  = problem_ptr->tolerances.absolute_tolerance;
    run_lp_with_vars_fixed(*problem_ptr, solution, problem_ptr->integer_indices, lp_settings);
  }
  local_search.fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  local_search.fj.settings.n_of_minimums_for_exit = 20000;
  local_search.fj.settings.update_weights         = true;
  local_search.fj.settings.feasibility_run        = false;
  local_search.do_fj_solve(solution,
                           local_search.fj,
                           std::max<f_t>(repair_timer.remaining_time(), 0.1),
                           "fix_propagate_repair");
  solution.compute_feasibility();
  result.feasible = solution.get_feasible();
  if (result.feasible) {
    result.objective  = solution.get_user_objective();
    result.assignment = solution.get_host_assignment();
    result.published  = population.add_solution(solution_t<i_t, f_t>(solution)).second;
    if (result.published) {
      CUOPT_LOG_INFO(
        "New solution from primal heuristics (CPU fix-and-propagate, FJ repair). Objective %+.6e. "
        "Time %.3f",
        result.objective,
        timer.elapsed_time());
    }
  }
  result.time = timer.elapsed_time() - start_time;
  return result;
}

#if MIP_INSTANTIATE_FLOAT
template class fix_and_propagate_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class fix_and_propagate_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
