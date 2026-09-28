/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../internal.hpp"
#include "api.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename f_t>
void fj_simd_score_rows(const int32_t* rows,
                        const f_t* coefficients,
                        const f_t* row_state,
                        int32_t begin,
                        int32_t end,
                        f_t delta,
                        f_t tolerance,
                        f_t excess_weight,
                        f_t& base,
                        f_t& bonus);

template <typename i_t, typename f_t, MTMMoveType move_type>
f_t get_mtm_for_constraint(f_t cstr_coeff, f_t slack, f_t row_tolerance)
{
  cuopt_assert(cstr_coeff != f_t{0}, "zero coefficient moves no row");
  const bool violated = slack < -row_tolerance;
  if (move_type == MTMMoveType::FJ_MTM_VIOLATED ? !violated : violated) return f_t{0};
  return slack / cstr_coeff;
}

template <typename i_t, typename f_t>
std::pair<f_t, f_t> feas_score_constraint(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, f_t delta, f_t cstr_coeff, f_t old_slack, f_t cstr_weight)
{
  cuopt_assert(std::isfinite(delta), "invalid delta");
  // A model may store explicit zeros, and a zero coefficient contributes nothing to the row.
  cuopt_assert(std::isfinite(cstr_coeff), "invalid coefficient");
  cuopt_assert(std::isfinite(cstr_weight), "invalid weight");
  cuopt_assert(cstr_weight >= 0, "invalid weight");

  const f_t tol       = fj_cpu.row_tolerance;
  const f_t new_slack = old_slack - cstr_coeff * delta;
  cuopt_assert(std::isfinite(old_slack) && std::isfinite(new_slack), "");

  const bool old_sat = old_slack > -tol;
  const bool new_sat = new_slack > -tol;

  f_t base_feas = 0;
  if (!old_sat && new_sat) {
    base_feas += cstr_weight;
  } else if (old_sat && !new_sat) {
    base_feas -= cstr_weight;
  } else if (!old_sat && !new_sat && new_slack > old_slack) {
    // Keep the fractional excess signal.  Converting through i_t made the default
    // 0.5 improvement weight vanish for unit-weight rows, leaving FJ blind to
    // progress on a row until a move crossed its bound.
    base_feas += cstr_weight * fj_cpu.settings.parameters.excess_improvement_weight;
  } else if (!old_sat && !new_sat && new_slack < old_slack) {
    base_feas -= cstr_weight * fj_cpu.settings.parameters.excess_improvement_weight;
  }

  f_t bonus_robust      = 0;
  const bool old_stable = old_slack > tol;
  const bool new_stable = new_slack > tol;
  if (!old_stable && new_stable) {
    bonus_robust += cstr_weight;
  } else if (old_stable && !new_stable) {
    bonus_robust -= cstr_weight;
  }

  return {base_feas, bonus_robust};
}

template <typename i_t, typename f_t>
inline bool tabu_check(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                       i_t var_idx,
                       f_t delta,
                       bool localmin = false)
{
  if (localmin) {
    return (delta < 0 && fj_cpu.iterations == fj_cpu.h_tabu_lastinc[var_idx] + 1) ||
           (delta >= 0 && fj_cpu.iterations == fj_cpu.h_tabu_lastdec[var_idx] + 1);
  } else {
    return (delta < 0 && fj_cpu.iterations < fj_cpu.h_tabu_nodec_until[var_idx]) ||
           (delta >= 0 && fj_cpu.iterations < fj_cpu.h_tabu_noinc_until[var_idx]);
  }
}

template <typename i_t, typename f_t>
inline std::pair<fj_staged_score_t, f_t> compute_score(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                                       i_t var_idx,
                                                       f_t delta)
{
  f_t obj_diff = fj_cpu.problem->h_obj_coeffs[var_idx] * delta;

  cuopt_assert(std::isfinite(delta), "");

  cuopt_assert(var_idx < fj_cpu.problem->n_variables, "variable index out of bounds");

  f_t base_feas_sum    = 0;
  f_t bonus_robust_sum = 0;

  auto [offset_begin, offset_end] = fj_cpu.range_for_variable(var_idx);
  fj_cpu.nnz_processed_window += (offset_end - offset_begin);

  const size_t nnz_read = (size_t)(offset_end - offset_begin);
  ++fj_cpu.n_compute_score_calls;
  fj_cpu.compute_score_nnz += (int64_t)nnz_read;
  fj_cpu.h_reverse_constraints.byte_loads += nnz_read * sizeof(i_t);
  fj_cpu.h_reverse_coefficients.byte_loads += nnz_read * sizeof(f_t);
  fj_cpu.h_row_state.byte_loads +=
    nnz_read * sizeof(typename fj_cpu_climber_t<i_t, f_t>::row_state_t);

  const i_t* const rev_cstr  = fj_cpu.h_reverse_constraints.data();
  const f_t* const rev_coeff = fj_cpu.h_reverse_coefficients.data();
  const typename fj_cpu_climber_t<i_t, f_t>::row_state_t* const state = fj_cpu.row_state();

  static_assert(std::is_same_v<i_t, int32_t>);
  fj_simd_score_rows(rev_cstr,
                     rev_coeff,
                     reinterpret_cast<const f_t*>(state),
                     offset_begin,
                     offset_end,
                     delta,
                     fj_cpu.row_tolerance,
                     (f_t)fj_cpu.settings.parameters.excess_improvement_weight,
                     base_feas_sum,
                     bonus_robust_sum);

  f_t base_obj = 0;
  if (fj_cpu.h_objective_weight > 0 && obj_diff != 0) {
    // Scaling base is only meaningful where there is feasibility impact to trade against.
    f_t weighted = fj_cpu.h_objective_weight;
    if (base_feas_sum != 0) {
      cuopt_assert(fj_cpu.problem->obj_magnitude > 0, "objective magnitude unit must be positive");
      weighted *= std::min(
        (f_t)fj_obj_mult_max,
        std::max((f_t)fj_obj_mult_min, std::fabs(obj_diff) / fj_cpu.problem->obj_magnitude));
    }
    base_obj = obj_diff < 0 ? weighted : -weighted;
  }

  f_t bonus_breakthrough = 0;

  bool old_obj_better = fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective;
  bool new_obj_better = fj_cpu.h_incumbent_objective + obj_diff < fj_cpu.h_best_objective;
  if (!old_obj_better && new_obj_better)
    bonus_breakthrough += fj_cpu.h_objective_weight;
  else if (old_obj_better && !new_obj_better) {
    bonus_breakthrough -= fj_cpu.h_objective_weight;
  }

  fj_staged_score_t score;
  score.base  = std::round(base_obj + base_feas_sum);
  score.bonus = std::round(bonus_breakthrough + bonus_robust_sum);
  return std::make_pair(score, base_feas_sum);
}

template <typename i_t, typename f_t>
void smooth_weights(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::smooth_weights");
  for (i_t row = 0; row < fj_cpu.n_rows; ++row) {
    if (fj_cpu.violated_constraints.contains(row)) continue;
    f_t& weight = fj_cpu.row_state()[row].weight;
    weight      = std::max((f_t)0, weight - 1);
  }

  if (fj_cpu.h_objective_weight > 0 && fj_cpu.h_incumbent_objective >= fj_cpu.h_best_objective) {
    fj_cpu.h_objective_weight = std::max(f_t{0}, fj_cpu.h_objective_weight - 1);
  }
}

template <typename i_t, typename f_t>
void update_weights(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::update_weights");

  cuopt::pcgenerator_t rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);
  bool smoothing = rng.next_float() <= fj_cpu.settings.parameters.weight_smoothing_probability;

  if (smoothing) {
    smooth_weights<i_t, f_t>(fj_cpu);
    return;
  }

  for (auto cstr_idx : fj_cpu.violated_constraints) {
    const f_t old_weight = fj_cpu.row_state()[cstr_idx].weight;
    cuopt_assert(fj_cpu.row_state()[cstr_idx].slack + fj_cpu.h_slack_sumcomp[cstr_idx] < 0,
                 "constraint not violated");

    f_t new_weight = std::round(old_weight + f_t{1});
    new_weight     = std::min(new_weight, (f_t)fj_cpu.hp.weight_cap);

    fj_cpu.row_state()[cstr_idx].weight = new_weight;
    fj_cpu.max_weight                   = std::max(fj_cpu.max_weight, new_weight);

    // Invalidate related cached move scores
    ++fj_cpu.n_version_bumps_weights;
    fj_cpu.h_cstr_version[cstr_idx]++;
  }

  if (fj_cpu.violated_constraints.empty()) { fj_cpu.h_objective_weight += 1; }
}

}  // namespace cuopt::mathematical_optimization::mip
