/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "audit.hpp"
#include "internal.hpp"
#include "problem.hpp"
#include "search/api.hpp"
#include "search/escape.hpp"
#include "search/moves.hpp"
#include "search/score.hpp"
#include "search/update.hpp"
#include "setup/bounds.hpp"
#include "setup/lp.hpp"
#include "setup/structure.hpp"
#include "starts/starts.hpp"

#include <mip_heuristics/feasibility_jump/fj_cpu_binary.cuh>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
static std::vector<f_t> lift_equality_substituted_assignment(
  fj_cpu_climber_t<i_t, f_t>& c,
  const std::vector<f_t>& assignment,
  const std::vector<i_t>& retained,
  const std::vector<fj_equality_substitution_t<i_t, f_t>>& substitutions)
{
  std::vector<f_t> lifted(c.problem->n_variables, 0);
  for (size_t j = 0; j < retained.size(); ++j)
    lifted[retained[j]] = assignment[j];
  for (auto it = substitutions.rbegin(); it != substitutions.rend(); ++it) {
    const auto coefficients = thrust::make_transform_iterator(
      it->terms.begin(), [](const auto& term) { return term.second; });
    const auto values = thrust::make_transform_iterator(
      it->terms.begin(), [&lifted](const auto& term) { return lifted[term.first]; });
    lifted[it->variable] = it->constant + compensated_dot2(coefficients, values, it->terms.size());
  }
  for (const auto& sub : substitutions) {
    const auto bounds = c.h_var_bounds[sub.variable].get();
    f_t value         = std::clamp(lifted[sub.variable], get_lower(bounds), get_upper(bounds));
    if (is_integer_var(c, sub.variable)) value = std::round(value);
    lifted[sub.variable] = value;
  }
  // Substitution and the final bound/integrality repair must both survive a check in the unchanged
  // parent model
  const auto& p = *c.problem;
  for (i_t v = 0; v < p.n_variables; ++v) {
    if (!std::isfinite(lifted[v]) || !check_variable_within_bounds(c, v, lifted[v]) ||
        (is_integer_var(c, v) && !p.is_integer(lifted[v])))
      return {};
  }
  for (i_t r = 0; r < p.n_constraints; ++r) {
    const f_t activity = compensated_dot2_csr(p, lifted, r);
    const f_t tol      = p.tolerances.absolute_tolerance;
    if (!std::isfinite(activity) || activity < p.cstr_lb[r] - tol || activity > p.cstr_ub[r] + tol)
      return {};
  }
  return lifted;
}

// Solve a lane in equality-reduced coordinates, then lift every candidate back into the unchanged
// parent model before reporting it.
template <typename i_t, typename f_t>
bool try_equality_substituted_solve(fj_cpu_climber_t<i_t, f_t>& c,
                                    double time_limit,
                                    double work_unit_limit)
{
  if (!c.use_equality_substitution || c.feasible_found || c.producer_sync || time_limit <= 0)
    return false;
  const auto started = std::chrono::steady_clock::now();
  std::vector<fj_equality_substitution_t<i_t, f_t>> substitutions;
  std::vector<i_t> retained;
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> child;
  {
    phase_timer_t timer(c.t_start);
    child =
      make_equality_reduced_climber(c, std::min(0.75, 0.15 * time_limit), substitutions, retained);
  }
  if (!child) return false;
  const double remaining =
    time_limit - std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  if (remaining <= 0) return false;

  bool rejected_lift    = false;
  child->work_unit_bias = c.work_unit_bias;
  child->improvement_callback =
    [&](f_t child_objective, const std::vector<f_t>& assignment, double work) {
      if (assignment.size() != retained.size()) {
        rejected_lift = true;
        child->halted = true;
        return;
      }
      const std::vector<f_t> lifted =
        lift_equality_substituted_assignment(c, assignment, retained, substitutions);
      if (lifted.empty()) {
        rejected_lift = true;
        child->halted = true;
        return;
      }
      const f_t objective = child_objective + child->problem->objective_offset;
      if (!c.feasible_found || objective < c.h_best_objective) {
        c.h_best_assignment = lifted;
        c.h_best_objective  = objective;
        c.feasible_found    = true;
      }
      report_cpu_incumbent(c, objective, lifted, work);
    };

  const auto setup_stats = static_cast<const fj_stats_t<i_t>&>(c);
  cpufj_solve(child.get(), remaining, work_unit_limit);
  static_cast<fj_stats_t<i_t>&>(c) = static_cast<const fj_stats_t<i_t>&>(*child);
  c.t_start += setup_stats.t_start;
  c.t_bound_prop += setup_stats.t_bound_prop;
  c.t_lp_start += setup_stats.t_lp_start;
  c.t_features += setup_stats.t_features;
  c.t_init_lhs += setup_stats.t_init_lhs;
  c.iterations = child->iterations;
  c.work_units_elapsed.store(child->work_units_elapsed.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);

  if (child->feasible_found && child->h_best_assignment.size() == retained.size()) {
    std::vector<f_t> lifted = lift_equality_substituted_assignment(
      c, child->h_best_assignment.underlying(), retained, substitutions);
    if (lifted.empty()) return c.feasible_found;
    const f_t objective = compensated_dot2(
      thrust::make_permutation_iterator(c.problem->h_obj_coeffs.data(),
                                        c.problem->h_objective_vars.data()),
      thrust::make_permutation_iterator(lifted.data(), c.problem->h_objective_vars.data()),
      c.problem->h_objective_vars.size());
    if (!c.feasible_found || objective < c.h_best_objective) {
      c.h_best_assignment = std::move(lifted);
      c.h_best_objective  = objective;
      c.feasible_found    = true;
    }
  }
  return !rejected_lift || c.feasible_found;
}

template <typename i_t, typename f_t>
void cpufj_solve(fj_cpu_climber_t<i_t, f_t>* fj_cpu, double time_limit, double work_unit_limit)
{
  const auto solve_start = std::chrono::steady_clock::now();
  if (fj_cpu->use_precedence_start) apply_precedence_completion_start(*fj_cpu);
  apply_bound_propagation(*fj_cpu);
  if (fj_cpu->use_equality_substitution) {
    const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    if (try_equality_substituted_solve(*fj_cpu, time_limit - elapsed, work_unit_limit)) return;
  }
  const double setup_time_left =
    fj_cpu->use_equality_substitution
      ? std::max(
          0.0,
          time_limit -
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count())
      : time_limit;
  if (!fj_cpu->feasible_found) { apply_lp_rounded_start(*fj_cpu, setup_time_left); }

  const bool paid_setup = fj_cpu->use_bound_prop || fj_cpu->use_lp_start ||
                          fj_cpu->use_precedence_start || fj_cpu->use_equality_substitution;
  const double setup_seconds =
    paid_setup
      ? std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count()
      : 0.0;
  const double remaining = std::max(0.0, time_limit - setup_seconds);
  if (remaining <= 0.0) return;

  if (try_cpufj_binary_solve(*fj_cpu, remaining, work_unit_limit)) return;

  clamp_start_magnitude(*fj_cpu, fj_cpu->problem->n_variables);

  // Past this point the search runs on one-sided rows. Everything above reasons about the original
  // model, which is why the rows are built here and not at construction.
  build_one_sided_rows(*fj_cpu);

  // Publish a feasible structural start, then keep the lane active for objective improvement.
  if (fj_cpu->violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(*fj_cpu)) {
    fj_cpu->h_best_assignment = fj_cpu->h_assignment;
    fj_cpu->h_best_objective =
      fj_cpu->h_incumbent_objective - fj_cpu->settings.parameters.breakthrough_move_epsilon;
    fj_cpu->feasible_found = true;
    report_cpu_incumbent(*fj_cpu);
  }

  [[maybe_unused]] i_t local_mins = 0;
  const auto loop_start           = paid_setup ? solve_start : std::chrono::steady_clock::now();
  bool first_cross_needs_polish   = fj_cpu->use_lp_polish;

  fj_cpu->rng.set_seed(fj_cpu->settings.seed);

  // Initialize feature tracking
  fj_cpu->iterations_since_best = 0;

  // The recompute is O(nnz), so a fixed period costs a growing share of the budget.
  cuopt_assert(fj_cpu->settings.parameters.lhs_refresh_period > 0,
               "lhs_refresh_period should be positive");
  const i_t nnz_stretch = std::min<i_t>(fj_cpu->problem->nnz / fj_cpu->hp.nnz_per_refresh_stretch,
                                        fj_cpu->hp.max_refresh_stretch);
  const i_t refresh_period = fj_cpu->settings.parameters.lhs_refresh_period * (1 + nnz_stretch);
  // const i_t refresh_period = 5000 * (1 + nnz_stretch);
  cuopt_assert(refresh_period > 0, "refresh period overflowed");
  fj_cpu->lhs_refresh_period_used = refresh_period;

  // Whatever the start left behind, these rows are satisfiable on their own, so the walk should not
  // start with them in the violated set competing for the sampler's attention.
  for (i_t var : fj_cpu->epigraph_vars) {
    const f_t current = fj_cpu->h_assignment[var];
    const f_t delta   = project_epigraph_variable(*fj_cpu, var) - current;
    if (delta == f_t{0}) continue;
    if (!fj_cpu->move_numerically_stable(
          current, current + delta, fj_cpu->total_violations, fj_cpu->total_violations))
      continue;
    apply_move(*fj_cpu, var, delta, false);
    ++fj_cpu->n_epigraph_projections;
  }

  while (!fj_cpu->halted && !fj_cpu->preemption_flag.load()) {
    const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed > time_limit) {
      CUOPT_LOG_TRACE("%sTime limit of %.4f seconds reached, breaking loop at iteration %d",
                      fj_cpu->log_prefix.c_str(),
                      time_limit,
                      fj_cpu->iterations);
      break;
    }
    if (fj_cpu->iterations >= fj_cpu->settings.iteration_limit) {
      CUOPT_LOG_TRACE("%sIteration limit of %d reached, breaking loop at iteration %d",
                      fj_cpu->log_prefix.c_str(),
                      fj_cpu->settings.iteration_limit,
                      fj_cpu->iterations);
      break;
    }

    if (first_cross_needs_polish && fj_cpu->feasible_found) {
      first_cross_needs_polish = false;
      const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
      apply_lp_polish(*fj_cpu, fj_cpu->hp.lp_polish_budget_share * (time_limit - elapsed));
    }

    // periodically recompute the slacks and violation scores
    // to correct any accumulated numerical errors
    if (fj_cpu->trigger_early_lhs_recomputation) {
      ++fj_cpu->n_lhs_recompute_bigval;
      recompute_slack(*fj_cpu);
      fj_cpu->trigger_early_lhs_recomputation = false;
    } else if (fj_cpu->iterations % refresh_period == 0) {
      ++fj_cpu->n_lhs_recompute_periodic;
      recompute_slack(*fj_cpu);
    }

    fj_move_t move          = fj_move_t{-1, 0};
    fj_staged_score_t score = fj_staged_score_t::invalid();
    bool is_lift            = false;
    bool is_mtm_viol        = false;
    bool is_mtm_sat         = false;

    // Perform lift moves
    fj_move_t lift_companion = fj_move_t{-1, 0};
    if (fj_cpu->violated_constraints.empty()) {
      thrust::tie(move, score) = find_lift_move(*fj_cpu);
      if (score > fj_staged_score_t::zero()) {
        is_lift = true;
      } else {
        // Pairs are only reachable once no single improving flip preserves feasibility.
        fj_move_t first, second;
        fj_staged_score_t pair_score;
        thrust::tie(first, second, pair_score) = find_lift_2opt_move(*fj_cpu);
        if (pair_score > fj_staged_score_t::zero()) {
          move           = first;
          lift_companion = second;
          score          = pair_score;
          is_lift        = true;
        }
      }
    }
    // Regular MTM
    if (!(score > fj_staged_score_t::zero())) {
      thrust::tie(move, score) = find_mtm_move_viol(*fj_cpu, fj_cpu->mtm_viol_samples);
      if (score > fj_staged_score_t::zero()) is_mtm_viol = true;
    }
    // try with MTM in satisfied constraints
    if (fj_cpu->feasible_found && !(score > fj_staged_score_t::zero())) {
      thrust::tie(move, score) = find_mtm_move_sat(*fj_cpu, fj_cpu->mtm_sat_samples);
      if (score > fj_staged_score_t::zero()) is_mtm_sat = true;
    }

    // The scorers target one row at a time, so on an epigraph variable they climb toward the bound
    // its rows already imply. The projection lands there in one move at the same O(degree) cost.
    if (move.var_idx >= 0 && fj_cpu->epigraph_push[move.var_idx] != 0) {
      const f_t projected =
        project_epigraph_variable(*fj_cpu, move.var_idx) - (f_t)fj_cpu->h_assignment[move.var_idx];
      if (projected != f_t{0}) {
        move.value = projected;
        ++fj_cpu->n_epigraph_projections;
      }
    }

    // if we're in the feasible region but haven't found improvements in the last n iterations,
    // perturb
    bool should_perturb = false;
    if (fj_cpu->violated_constraints.empty() &&
        fj_cpu->iterations_since_best > fj_cpu->perturb_interval) {
      should_perturb = true;
      // Without this the counter stays above the interval and every later iteration perturbs.
      fj_cpu->iterations_since_best = 0;
      if (fj_cpu->use_lp_polish && fj_cpu->feasible_found) {
        const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
        apply_lp_polish(*fj_cpu, fj_cpu->hp.lp_polish_budget_share * (time_limit - elapsed));
      }
    }

    if (score > fj_staged_score_t::zero() && !should_perturb) {
      apply_move(*fj_cpu, move.var_idx, move.value, false);
      if (lift_companion.var_idx >= 0) {
        apply_move(*fj_cpu, lift_companion.var_idx, lift_companion.value, false);
      }
      // Track move types
    } else {
      update_weights(*fj_cpu);
      if (should_perturb) {
        perturb(*fj_cpu);
        invalidate_mtm_cache(*fj_cpu);
      }

      if (!fj_cpu->violated_constraints.empty()) {
        thrust::tie(move, score) =
          find_mtm_move_viol(*fj_cpu, 1, true);  // pick a single random violated constraint
        i_t var_idx = move.var_idx >= 0 ? move.var_idx : 0;
        f_t delta   = move.var_idx >= 0 ? move.value : 0;
        apply_move(*fj_cpu, var_idx, delta, true);
      } else {
        // Feasible and stuck with nothing violated to move against: find_mtm_move_viol above
        // would sample an empty set and force a delta-0 no-op that still bumps every row version
        // the fallback variable touches. A forced satisfied-row move is a real step instead, and
        // when even that finds nothing the iteration is simply skipped rather than faked.
        thrust::tie(move, score) = find_mtm_move_sat(*fj_cpu, fj_cpu->mtm_sat_samples, true);
        if (move.var_idx >= 0) { apply_move(*fj_cpu, move.var_idx, move.value, true); }
      }
      ++local_mins;
    }

    if (fj_cpu->log_interval && fj_cpu->iterations % fj_cpu->log_interval == 0) {
      CUOPT_LOG_DEBUG(
        "%sCPUFJ iteration: %d/%d, local mins: %d, best_objective: %g, viol: %zu, obj weight %g, "
        "maxw %g",
        fj_cpu->log_prefix.c_str(),
        fj_cpu->iterations,
        fj_cpu->settings.iteration_limit != std::numeric_limits<i_t>::max()
          ? fj_cpu->settings.iteration_limit
          : -1,
        local_mins,
        fj_cpu->get_user_objective(fj_cpu->h_best_objective),
        fj_cpu->violated_constraints.size(),
        fj_cpu->h_objective_weight,
        fj_cpu->max_weight);
    }

    if (fj_cpu->iterations % 100 == 0 && fj_cpu->iterations > 0) {
      // Use cumulative byte counts (collect() without flush). Each window's contribution to
      // work_units_elapsed therefore grows roughly with the running total of bytes touched,
      // i.e. quadratically in iterations rather than linearly. This is intentional: the
      // memory_aggregator is calibrated for medium/large MIPs, and a strictly-linear scheme
      // forces tiny instances (few KB per iteration) to run for tens of seconds before the
      // accumulated bytes cross a 0.5 horizon, causing the deterministic producer_sync to
      // stall and B&B to time out on instances that should solve in milliseconds. The
      // accumulation is still deterministic across runs of the same problem, which is what
      // the producer_sync contract actually requires.
      auto [loads, stores] = fj_cpu->memory_aggregator.collect();
      double biased_work   = (loads + stores) * fj_cpu->work_unit_bias / 1e10;
      fj_cpu->work_units_elapsed += biased_work;

      if (fj_cpu->producer_sync != nullptr) { fj_cpu->producer_sync->notify_progress(); }
      if (fj_cpu->work_units_elapsed >= work_unit_limit) { break; }
    }

    cuopt_func_call(sanity_checks(*fj_cpu));
    if (fj_audit_every_iteration) {
      cuopt_func_call(audit_incremental_state(*fj_cpu, "iteration"));
    }
    fj_cpu->iterations++;
    fj_cpu->iterations_since_best++;
  }
  const double total_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
  [[maybe_unused]] double avg_time_per_iter =
    fj_cpu->iterations > 0 ? total_time / fj_cpu->iterations : 0;
  CUOPT_LOG_TRACE("%sCPUFJ Average time per iteration: %.8fms",
                  fj_cpu->log_prefix.c_str(),
                  avg_time_per_iter * 1000.0);
}

#if MIP_INSTANTIATE_FLOAT
template void cpufj_solve(fj_cpu_climber_t<int, float>*, double, double);
template void report_cpu_incumbent<int, float>(fj_cpu_climber_t<int, float>&,
                                               float,
                                               const std::vector<float>&,
                                               double);
template void report_cpu_incumbent<int, float>(fj_cpu_climber_t<int, float>&);
template void recompute_lhs<int, float>(fj_cpu_climber_t<int, float>&);
template void recompute_slack<int, float>(fj_cpu_climber_t<int, float>&);
template void invalidate_mtm_cache<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void cpufj_solve(fj_cpu_climber_t<int, double>*, double, double);
template void report_cpu_incumbent<int, double>(fj_cpu_climber_t<int, double>&,
                                                double,
                                                const std::vector<double>&,
                                                double);
template void report_cpu_incumbent<int, double>(fj_cpu_climber_t<int, double>&);
template void recompute_lhs<int, double>(fj_cpu_climber_t<int, double>&);
template void recompute_slack<int, double>(fj_cpu_climber_t<int, double>&);
template void invalidate_mtm_cache<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
