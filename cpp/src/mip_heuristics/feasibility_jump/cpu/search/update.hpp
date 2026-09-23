/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../audit.hpp"
#include "../internal.hpp"
#include "api.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void report_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c,
                          f_t objective,
                          const std::vector<f_t>& assignment,
                          double work_units)
{
  // Constructive starts and lifted solves report through this same point as ordinary moves.
  const f_t last_reported = c.h_last_reported_objective;
  if (!(last_reported - objective > f_t{1e-6} * std::max(f_t{1}, std::fabs(last_reported)))) return;
  c.h_last_reported_objective = objective;
  if (!c.suppress_incumbent_log)
    CUOPT_LOG_DEBUG("%sCPUFJ new incumbent: objective %.17g",
                    c.log_prefix.c_str(),
                    c.get_user_objective(objective));
  if (!c.improvement_callback) return;
  c.improvement_callback(objective, assignment, work_units);
}

template <typename i_t, typename f_t>
void report_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c)
{
  report_cpu_incumbent(c,
                       c.h_incumbent_objective,
                       c.h_assignment.underlying(),
                       c.work_units_elapsed.load(std::memory_order_acquire));
}

template <typename i_t, typename f_t>
void share_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c,
                         f_t objective,
                         const std::vector<f_t>& assignment)
{
  if (c.shared_incumbent)
    c.shared_incumbent->publish(objective, c.get_user_objective(objective), assignment);
}

template <typename i_t, typename f_t>
void share_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c)
{
  share_cpu_incumbent(c, c.h_incumbent_objective, c.h_assignment.underlying());
}

template <typename i_t, typename f_t>
void apply_move(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx, f_t delta, bool localmin = false)
{
  CPUFJ_NVTX_RANGE("CPUFJ::apply_move");

  cuopt::pcgenerator_t rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  cuopt_assert(var_idx < fj_cpu.problem->n_variables, "variable index out of bounds");
  f_t old_val = fj_cpu.h_assignment[var_idx];
  f_t new_val = old_val + delta;
  if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) {
    cuopt_assert(fj_cpu.problem->integer_equal(new_val, std::round(new_val)),
                 "new_val is not integer");
    new_val = std::round(new_val);
  }
  // clamp to var bounds, then to the magnitude the whole assignment is held to. A per-move guard
  // bounds the step and not the position, so without this a run of individually legal steps walks a
  // variable to a magnitude where one ulp of its rows' slacks passes the row tolerance. Every
  // generator is bounded here rather than each having to know; delta is recomputed below, so the
  // slack bookkeeping stays exact whatever the clamp does.
  const auto var_bounds = fj_cpu.h_var_bounds[var_idx].get();
  new_val               = std::min(std::max(new_val, get_lower(var_bounds)), get_upper(var_bounds));
  const f_t floor_      = std::max(get_lower(var_bounds), (f_t)-fj_cpu.hp.start_magnitude_limit);
  const f_t ceil_       = std::min(get_upper(var_bounds), (f_t)fj_cpu.hp.start_magnitude_limit);
  if (floor_ <= ceil_) new_val = std::min(std::max(new_val, floor_), ceil_);
  delta              = new_val - old_val;
  const f_t delta_z  = delta - new_val;
  const f_t delta_lo = (new_val - (delta - delta_z)) + (-old_val - delta_z);
  cuopt_assert(std::isfinite(new_val), "assignment is not finite");
  cuopt_assert(std::isfinite(delta), "applied delta is not finite");
  cuopt_assert(check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, new_val),
               "assignment not within bounds");

  // Update the slack of every search row the variable appears in.
  auto [offset_begin, offset_end] = fj_cpu.range_for_variable(var_idx);

  fj_cpu.nnz_processed_window += (offset_end - offset_begin);
  const size_t nnz_touched = (size_t)(offset_end - offset_begin);
  using row_state_t        = typename fj_cpu_climber_t<i_t, f_t>::row_state_t;
  ++fj_cpu.n_moves_applied;
  fj_cpu.apply_move_nnz += (int64_t)nnz_touched;
  fj_cpu.n_version_bumps_apply += (int64_t)nnz_touched;
  fj_cpu.h_reverse_constraints.byte_loads += nnz_touched * sizeof(i_t);
  fj_cpu.h_reverse_coefficients.byte_loads += nnz_touched * sizeof(f_t);
  fj_cpu.h_row_state.byte_loads += nnz_touched * sizeof(row_state_t);
  fj_cpu.h_row_state.byte_stores += nnz_touched * sizeof(row_state_t);
  fj_cpu.h_slack_sumcomp.byte_loads += nnz_touched * sizeof(f_t);
  fj_cpu.h_slack_sumcomp.byte_stores += nnz_touched * sizeof(f_t);

  const i_t* const rev_cstr  = fj_cpu.h_reverse_constraints.data();
  const f_t* const rev_coeff = fj_cpu.h_reverse_coefficients.data();
  f_t* const row_sumcomp     = fj_cpu.h_slack_sumcomp.data();
  row_state_t* const state   = fj_cpu.row_state();
  const f_t cstr_tolerance   = fj_cpu.row_tolerance;

  for (auto i = offset_begin; i < offset_end; i++) {
    cuopt_assert(i < (i_t)fj_cpu.h_reverse_constraints.size(), "");

    const i_t cstr_idx   = rev_cstr[i];
    const f_t cstr_coeff = rev_coeff[i];

    // The row is a'x <= b holding slack b - a'x, so the move lowers the slack by its own
    // coefficient times the delta. Dot2, as the activity was: the carry holds the correction to add
    // to the stored slack, covering the rounding of both the product and the addition.
    const f_t old_slack   = state[cstr_idx].slack;
    const f_t old_sumcomp = row_sumcomp[cstr_idx];
    const f_t h           = -cstr_coeff * delta;
    f_t t                 = old_slack + h;
    const f_t z           = t - old_slack;
    f_t new_sumcomp =
      old_sumcomp + (((old_slack - (t - z)) + (h - z)) + std::fma(-cstr_coeff, delta, -h)) -
      cstr_coeff * delta_lo;

    const f_t old_value = old_slack + old_sumcomp;
    f_t new_value       = t + new_sumcomp;
    if (fj_cpu.h_row_is_integral[cstr_idx]) {
      cuopt_assert(old_value == std::round(old_value), "integral row state is fractional");
      cuopt_assert(fj_cpu.problem->is_integer(new_value),
                   "integral row received a fractional move");
      new_value   = std::round(new_value);
      t           = new_value;
      new_sumcomp = 0;
    }
    row_sumcomp[cstr_idx] = new_sumcomp;
    state[cstr_idx].slack = t;

    const f_t old_cost = old_value < f_t{0} ? old_value : f_t{0};
    const f_t new_cost = new_value < f_t{0} ? new_value : f_t{0};

    // trigger early slack recomputation if the sumcomp term gets too large
    // to avoid large numerical errors
    if (std::fabs(new_sumcomp) > (f_t)fj_cpu.hp.bigval_threshold)
      fj_cpu.trigger_early_lhs_recomputation = true;

    const bool was_violated = fj_cpu.violated_constraints.contains(cstr_idx);
    const bool now_violated = new_value < -cstr_tolerance;

    // total_violations sums the excess over the violated set alone, so a row crossing the boundary
    // contributes its whole cost rather than a difference. Kahan compensated, as the slack is: this
    // is now the only place the total is maintained between refreshes.
    const f_t viol_delta = (now_violated ? new_cost : f_t{0}) - (was_violated ? old_cost : f_t{0});
    if (viol_delta != f_t{0}) {
      const f_t viol_old              = fj_cpu.total_violations;
      const f_t viol_y                = viol_delta - fj_cpu.total_violations_sumcomp;
      const f_t viol_t                = viol_old + viol_y;
      fj_cpu.total_violations_sumcomp = (viol_t - viol_old) - viol_y;
      fj_cpu.total_violations         = viol_t;
    }

    if (now_violated && !was_violated) {
      fj_cpu.violated_constraints.insert(cstr_idx);
      cuopt_assert(fj_cpu.satisfied_constraints.contains(cstr_idx), "");
      fj_cpu.satisfied_constraints.remove(cstr_idx);
    } else if (!now_violated && was_violated) {
      cuopt_assert(!fj_cpu.satisfied_constraints.contains(cstr_idx), "");
      fj_cpu.violated_constraints.remove(cstr_idx);
      fj_cpu.satisfied_constraints.insert(cstr_idx);
    }

    cuopt_assert(std::isfinite(delta), "delta should be finite");
    cuopt_assert(std::isfinite(t), "assignment should be finite");

    // Invalidate related cached move scores
    fj_cpu.h_cstr_version[cstr_idx]++;
  }

  // update the assignment and objective proper
  fj_cpu.h_assignment[var_idx] = new_val;
  // The clamp above passes a NaN straight through, and every comparison against one is false.
  cuopt_assert(fj_cpu.check_variable_within_bounds(var_idx, new_val),
               "apply_move left the variable bounds");
  // After the assignment write, which is what a fresh sum reads.
  cuopt_func_call(audit_row_updates(fj_cpu, var_idx, old_val, delta, offset_begin, offset_end));

  // Kahan compensated summation, as for the slacks. The incumbent objective is reported as-is, so
  // it cannot carry the drift of a long uncompensated chain of deltas.
  const f_t obj_old = fj_cpu.h_incumbent_objective;
  const f_t obj_coeff = fj_cpu.problem->h_obj_coeffs[var_idx];
  const f_t obj_y     = obj_coeff * delta - (fj_cpu.h_objective_sumcomp - obj_coeff * delta_lo);
  const f_t obj_t   = obj_old + obj_y;
  fj_cpu.h_objective_sumcomp   = (obj_t - obj_old) - obj_y;
  fj_cpu.h_incumbent_objective = obj_t;
  // The result of this addition carries the ulp of its larger operand, not of itself, and the
  // compensation cannot see it when the loss is in the product rather than the addition. Once that
  // exceeds the granularity an incumbent has to beat, the objective comparison below is reading
  // noise, so rebuild here rather than setting the deferred flag: the gate is in this same function
  // and would otherwise latch on the value this move just made unreliable. The row loop and the
  // assignment write are done, so the state a rebuild reads is consistent.
  const f_t obj_resolution =
    std::numeric_limits<f_t>::epsilon() * std::max(std::fabs(obj_old), std::fabs(obj_y));
  if (obj_resolution > (f_t)fj_cpu.settings.parameters.breakthrough_move_epsilon) {
    recompute_slack(fj_cpu);
  }
  cuopt_func_call(audit_objective_update(fj_cpu, var_idx, old_val, delta, obj_old, obj_y));

  if (fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective &&
      fj_cpu.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(fj_cpu)) {
    cuopt_assert((i_t)fj_cpu.satisfied_constraints.size() == fj_cpu.n_rows, "");
    cuopt_func_call(audit_incremental_state(fj_cpu, "incumbent gate"));
    fj_cpu.h_best_objective =
      fj_cpu.h_incumbent_objective - fj_cpu.settings.parameters.breakthrough_move_epsilon;
    fj_cpu.h_best_assignment     = fj_cpu.h_assignment;
    fj_cpu.iterations_since_best = 0;
    fj_cpu.perturb_streak        = 0;
    report_cpu_incumbent(fj_cpu);
    fj_cpu.feasible_found = true;
    // The true objective of the assignment, not the epsilon-reduced threshold stored above, so
    // another lane comparing against it is not misled into adopting something no better.
    share_cpu_incumbent(fj_cpu);
    // Feasibility-only descent intentionally starts at zero objective pressure.  Once it has an
    // incumbent, activate this lane's post-crossing objective persona before applying the incumbent
    // bonus; otherwise objective_weight_floor is only a decay floor and never actually diversifies
    // the objective phase.
    fj_cpu.h_objective_weight = std::max(fj_cpu.h_objective_weight, fj_cpu.objective_weight_floor);
    fj_cpu.h_objective_weight =
      std::min((f_t)fj_cpu.hp.obj_weight_incumbent_cap,
               fj_cpu.h_objective_weight + (f_t)fj_cpu.hp.obj_weight_incumbent_bump);
    // The weight enters every score, and row versions cannot see it move.
    retire_var_best_moves<i_t, f_t>(fj_cpu);
  }

  i_t tabu_tenure = fj_cpu.settings.parameters.tabu_tenure_min +
                    rng.next_u32() % (fj_cpu.settings.parameters.tabu_tenure_max -
                                      fj_cpu.settings.parameters.tabu_tenure_min);
  if (delta > 0) {
    fj_cpu.h_tabu_lastinc[var_idx]     = fj_cpu.iterations;
    fj_cpu.h_tabu_nodec_until[var_idx] = fj_cpu.iterations + tabu_tenure;
    fj_cpu.h_tabu_noinc_until[var_idx] = fj_cpu.iterations + tabu_tenure / 2;
    // CUOPT_LOG_TRACE("CPU: tabu nodec_until: %d\n", fj_cpu.h_tabu_nodec_until[var_idx]);
  } else {
    fj_cpu.h_tabu_lastdec[var_idx]     = fj_cpu.iterations;
    fj_cpu.h_tabu_noinc_until[var_idx] = fj_cpu.iterations + tabu_tenure;
    fj_cpu.h_tabu_nodec_until[var_idx] = fj_cpu.iterations + tabu_tenure / 2;
    // CUOPT_LOG_TRACE("CPU: tabu noinc_until: %d\n", fj_cpu.h_tabu_noinc_until[var_idx]);
  }

  ++fj_cpu.flip_move_epoch;
}

template <typename i_t, typename f_t>
f_t project_epigraph_variable(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx)
{
  cuopt_assert(fj_cpu.epigraph_push[var_idx] != 0, "variable is not a certified epigraph variable");
  const bool push_up = fj_cpu.epigraph_push[var_idx] > 0;
  const f_t current  = fj_cpu.h_assignment[var_idx];
  const auto bounds  = fj_cpu.h_var_bounds[var_idx].get();
  f_t target         = push_up ? get_lower(bounds) : get_upper(bounds);

  auto [offset_begin, offset_end] = fj_cpu.range_for_variable(var_idx);
  const size_t nnz_read           = (size_t)(offset_end - offset_begin);
  fj_cpu.h_reverse_constraints.byte_loads += nnz_read * sizeof(i_t);
  fj_cpu.h_reverse_coefficients.byte_loads += nnz_read * sizeof(f_t);
  fj_cpu.h_row_state.byte_loads +=
    nnz_read * sizeof(typename fj_cpu_climber_t<i_t, f_t>::row_state_t);

  const i_t* const rev_cstr  = fj_cpu.h_reverse_constraints.data();
  const f_t* const rev_coeff = fj_cpu.h_reverse_coefficients.data();
  const typename fj_cpu_climber_t<i_t, f_t>::row_state_t* const state = fj_cpu.row_state();

  // The row is a'x <= b holding slack b - a'x, so the value it allows this variable is
  // current + slack / coefficient. A certified epigraph variable has one sign throughout, so every
  // incidence gives a limit on the same side and the tightest is the extreme one.
  for (i_t p = offset_begin; p < offset_end; ++p) {
    const f_t coeff = rev_coeff[p];
    if (coeff == f_t{0}) continue;
    const f_t implied =
      current + (state[rev_cstr[p]].slack + fj_cpu.h_slack_sumcomp[rev_cstr[p]]) / coeff;
    if (!std::isfinite(implied)) continue;
    target = push_up ? std::max(target, implied) : std::min(target, implied);
  }

  target = std::min(std::max(target, get_lower(bounds)), get_upper(bounds));

  // An epigraph variable is unbounded in the push direction by construction, so the value its rows
  // imply is unbounded too, and landing on it puts every row it touches at a magnitude where one
  // ulp of the slack exceeds the row tolerance. Held to the range the start is held to, which keeps
  // the rows decidable at the cost of reaching the implied value over several moves instead of one.
  const f_t floor_ = std::max(get_lower(bounds), (f_t)-fj_cpu.hp.start_magnitude_limit);
  const f_t ceil_  = std::min(get_upper(bounds), (f_t)fj_cpu.hp.start_magnitude_limit);
  if (floor_ <= ceil_) target = std::min(std::max(target, floor_), ceil_);

  cuopt_assert(std::isfinite(target), "epigraph projection is not finite");
  return target;
}

template <typename i_t, typename f_t>
static void prepare_full_recompute(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  ++fj_cpu.n_lhs_recompute_total;
  // clamp to var bounds - defensive; apply_move should already have clamped appropriately
  for (i_t var_idx = 0; var_idx < fj_cpu.problem->n_variables; ++var_idx) {
    fj_cpu.h_assignment[var_idx] = std::min(
      std::max(fj_cpu.h_assignment[var_idx].get(), get_lower(fj_cpu.h_var_bounds[var_idx].get())),
      get_upper(fj_cpu.h_var_bounds[var_idx].get()));
  }
  fj_cpu.violated_constraints.clear();
  fj_cpu.satisfied_constraints.clear();
  fj_cpu.total_violations         = 0;
  fj_cpu.total_violations_sumcomp = 0;
}

template <typename i_t, typename f_t>
static void finish_full_recompute(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                  const f_t* objective_lhs,
                                  const f_t* objective_rhs)
{
  fj_cpu.h_incumbent_objective =
    compensated_dot2(objective_lhs, objective_rhs, fj_cpu.problem->n_variables);
  fj_cpu.h_objective_sumcomp = 0;
}

template <typename i_t, typename f_t>
void recompute_lhs(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::recompute_lhs");
  cuopt_assert(fj_cpu.h_lhs.size() == fj_cpu.problem->n_constraints, "h_lhs size mismatch");
  // Repopulates the membership sets over model rows and leaves row_state().slack untouched, so it
  // is only correct before build_one_sided_rows. recompute_slack is the counterpart afterwards.
  cuopt_assert(fj_cpu.violated_constraints.max_size() == fj_cpu.problem->n_constraints,
               "recompute_lhs keys the sets over model rows");
  cuopt_assert(fj_cpu.satisfied_constraints.max_size() == fj_cpu.problem->n_constraints,
               "recompute_lhs keys the sets over model rows");
  prepare_full_recompute(fj_cpu);
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.problem->n_constraints; ++cstr_idx) {
    fj_cpu.h_lhs[cstr_idx] = compensated_dot2_csr(*fj_cpu.problem, fj_cpu.h_assignment, cstr_idx);
    fj_cpu.h_lhs_sumcomp[cstr_idx] = 0;

    f_t new_cost = fj_cpu.excess_score(cstr_idx,
                                       fj_cpu.h_lhs[cstr_idx],
                                       fj_cpu.problem->cstr_lb[cstr_idx],
                                       fj_cpu.problem->cstr_ub[cstr_idx]);
    if (new_cost < -fj_cpu.row_tolerance) {
      fj_cpu.violated_constraints.insert(cstr_idx);
      fj_cpu.total_violations += new_cost;
    } else {
      fj_cpu.satisfied_constraints.insert(cstr_idx);
    }
  }

  finish_full_recompute(fj_cpu, fj_cpu.h_assignment.data(), fj_cpu.problem->h_obj_coeffs.data());
}

template <typename i_t, typename f_t>
void recompute_slack(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::recompute_slack");
  const i_t n_rows = fj_cpu.n_rows;
  cuopt_assert(n_rows > 0, "no rows");
  cuopt_assert((i_t)fj_cpu.h_row_state.size() == n_rows, "row state size mismatch");
  prepare_full_recompute(fj_cpu);
  const f_t* const assignment = fj_cpu.h_assignment.data();

  for (i_t r = 0; r < n_rows; ++r) {
    f_t slack = fresh_row_slack<i_t, f_t>(fj_cpu, r, assignment);
    if (fj_cpu.h_row_is_integral[r]) slack = std::round(slack);
    fj_cpu.row_state()[r].slack = slack;
    fj_cpu.h_slack_sumcomp[r]   = 0;
    if (slack < -fj_cpu.row_tolerance) {
      fj_cpu.violated_constraints.insert(r);
      fj_cpu.total_violations += slack;
    } else {
      fj_cpu.satisfied_constraints.insert(r);
    }
  }

  finish_full_recompute(fj_cpu, fj_cpu.problem->h_obj_coeffs.data(), assignment);
}

template <typename i_t, typename f_t>
void invalidate_mtm_cache(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  ++fj_cpu.n_mtm_cache_invalidations;
  for (size_t c = 0; c < fj_cpu.h_cstr_version.size(); ++c)
    fj_cpu.h_cstr_version[c]++;
}

}  // namespace cuopt::mathematical_optimization::mip
