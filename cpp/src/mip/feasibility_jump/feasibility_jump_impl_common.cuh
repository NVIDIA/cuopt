/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "feasibility_jump.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t, typename Iterator>
HDI f_t fj_kahan_babushka_neumaier_sum(Iterator begin, Iterator end)
{
  f_t sum = 0;
  f_t c   = 0;
  for (Iterator it = begin; it != end; ++it) {
    f_t delta = *it;
    f_t t     = sum + delta;
    if (fabs(sum) > fabs(delta)) {
      c += (sum - t) + delta;
    } else {
      c += (delta - t) + sum;
    }
    sum = t;
  }
  return sum + c;
}

// Returns the current slack, and the variable delta that would nullify this slack ("tighten" it)
template <typename i_t, typename f_t>
HDI thrust::tuple<f_t, f_t> get_mtm_for_bound(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
  i_t var_idx,
  i_t cstr_idx,
  f_t cstr_coeff,
  f_t bound,
  f_t sign)
{
  f_t delta_ij = 0;
  f_t slack    = 0;
  f_t old_val  = fj.incumbent_assignment[var_idx];

  f_t lhs = fj.incumbent_lhs[cstr_idx] * sign;
  f_t rhs = bound * sign;
  slack   = rhs - lhs;  // bound might be infinite. let the caller handle this case

  delta_ij = slack / (cstr_coeff * sign);

  return {delta_ij, slack};
}

template <typename i_t, typename f_t, MTMMoveType move_type>
HDI thrust::tuple<f_t, f_t, f_t, f_t> get_mtm_for_constraint(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
  i_t var_idx,
  i_t cstr_idx,
  f_t cstr_coeff,
  f_t c_lb,
  f_t c_ub)
{
  f_t sign     = -1;
  f_t delta_ij = 0;
  f_t slack    = 0;

  f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx);

  f_t old_val = fj.incumbent_assignment[var_idx];

  // process each bound as two separate constraints
  f_t bounds[2] = {c_lb, c_ub};
  cuopt_assert(isfinite(bounds[0]) || isfinite(bounds[1]), "bounds are not finite");

  for (i_t bound_idx = 0; bound_idx < 2; ++bound_idx) {
    if (!isfinite(bounds[bound_idx])) continue;

    // factor to correct the lhs/rhs to turn a lb <= lhs <= ub constraint into
    // two virtual constraints lhs <= ub and -lhs <= -lb
    sign    = bound_idx == 0 ? -1 : 1;
    f_t lhs = fj.incumbent_lhs[cstr_idx] * sign;
    f_t rhs = bounds[bound_idx] * sign;
    slack   = rhs - lhs;

    // skip constraints that are violated/satisfied based on the MTM move type
    bool violated = slack < -cstr_tolerance;
    if (move_type == MTMMoveType::FJ_MTM_VIOLATED ? !violated : violated) continue;

    f_t new_val = old_val;

    delta_ij = slack / (cstr_coeff * sign);
    break;
  }

  return {delta_ij, sign, slack, cstr_tolerance};
}

template <typename i_t, typename f_t>
HDI std::pair<f_t, f_t> feas_score_constraint(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
  i_t var_idx,
  f_t delta,
  i_t cstr_idx,
  f_t cstr_coeff,
  f_t c_lb,
  f_t c_ub,
  f_t current_lhs)
{
  cuopt_assert(isfinite(delta), "invalid delta");
  cuopt_assert(cstr_coeff != 0 && isfinite(cstr_coeff), "invalid coefficient");

  f_t base_feas    = 0;
  f_t bonus_robust = 0;

  f_t bounds[2] = {c_lb, c_ub};
  cuopt_assert(isfinite(c_lb) || isfinite(c_ub), "no range");
  for (i_t bound_idx = 0; bound_idx < 2; ++bound_idx) {
    if (!isfinite(bounds[bound_idx])) continue;

    // factor to correct the lhs/rhs to turn a lb <= lhs <= ub constraint into
    // two virtual leq constraints "lhs <= ub" and "-lhs <= -lb" in order to match
    // the convention of the paper

    // TODO: broadcast left/right weights to a csr_offset-indexed table? local minimums
    // usually occur on a rarer basis (around 50 iteratiosn to 1 local minimum)
    // likely unreasonable and overkill however
    f_t cstr_weight =
      bound_idx == 0 ? fj.cstr_left_weights[cstr_idx] : fj.cstr_right_weights[cstr_idx];
    f_t sign      = bound_idx == 0 ? -1 : 1;
    f_t rhs       = bounds[bound_idx] * sign;
    f_t old_lhs   = current_lhs * sign;
    f_t new_lhs   = (current_lhs + cstr_coeff * delta) * sign;
    f_t old_slack = rhs - old_lhs;
    f_t new_slack = rhs - new_lhs;

    cuopt_assert(isfinite(cstr_weight), "invalid weight");
    cuopt_assert(cstr_weight >= 0, "invalid weight");
    cuopt_assert(isfinite(old_lhs), "");
    cuopt_assert(isfinite(new_lhs), "");
    cuopt_assert(isfinite(old_slack) && isfinite(new_slack), "");

    f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx);

    bool old_viol = fj.excess_score(cstr_idx, current_lhs, c_lb, c_ub) < -cstr_tolerance;
    bool new_viol =
      fj.excess_score(cstr_idx, current_lhs + cstr_coeff * delta, c_lb, c_ub) < -cstr_tolerance;

    bool old_sat = old_lhs < rhs + cstr_tolerance;
    bool new_sat = new_lhs < rhs + cstr_tolerance;

    // equality
    if (fj.pb.integer_equal(c_lb, c_ub)) {
      if (!old_viol) cuopt_assert(old_sat == !old_viol, "");
      if (!new_viol) cuopt_assert(new_sat == !new_viol, "");
    }

    // if it would feasibilize this constraint
    if (!old_sat && new_sat) {
      cuopt_assert(old_viol, "");
      base_feas += cstr_weight;
    }
    // would cause this constraint to be violated
    else if (old_sat && !new_sat) {
      cuopt_assert(new_viol, "");
      base_feas -= cstr_weight;
    }
    // simple improvement
    else if (!old_sat && !new_sat && old_lhs > new_lhs) {
      cuopt_assert(old_viol && new_viol, "");
      base_feas += (i_t)(cstr_weight * fj.settings->parameters.excess_improvement_weight);
    }
    // simple worsening
    else if (!old_sat && !new_sat && old_lhs <= new_lhs) {
      cuopt_assert(old_viol && new_viol, "");
      base_feas -= (i_t)(cstr_weight * fj.settings->parameters.excess_improvement_weight);
    }

    // robustness score bonus if this would leave some strick slack
    bool old_stable = old_lhs < rhs - cstr_tolerance;
    bool new_stable = new_lhs < rhs - cstr_tolerance;
    if (!old_stable && new_stable) {
      bonus_robust += cstr_weight;
    } else if (old_stable && !new_stable) {
      bonus_robust -= cstr_weight;
    }
  }

  return {base_feas, bonus_robust};
}

template <typename i_t, typename f_t>
HDI f_t get_breakthrough_move(typename fj_t<i_t, f_t>::climber_data_t::view_t fj, i_t var_idx)
{
  f_t obj_coeff = fj.pb.objective_coefficients[var_idx];
  auto bounds   = fj.pb.variable_bounds[var_idx];
  f_t v_lb      = get_lower(bounds);
  f_t v_ub      = get_upper(bounds);
  cuopt_assert(isfinite(v_lb) || isfinite(v_ub), "unexpected free variable");
  cuopt_assert(v_lb <= v_ub, "invalid bounds");
  cuopt_assert(fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx]),
               "invalid incumbent assignment");
  cuopt_assert(isfinite(obj_coeff), "invalid objective coefficient");
  cuopt_assert(obj_coeff != 0, "objective coefficient shouldn't be null");

  f_t excess = (*fj.best_objective) - *fj.incumbent_objective;

  cuopt_assert(isfinite(excess) && excess < 0,
               "breakthru move invoked during invalid solver state");

  f_t old_val = fj.incumbent_assignment[var_idx];
  f_t new_val = old_val;

  f_t delta_ij = excess / obj_coeff;

  if (fj.pb.is_integer_var(var_idx)) {
    new_val = obj_coeff > 0 ? floor(old_val + delta_ij + fj.pb.tolerances.integrality_tolerance)
                            : ceil(old_val + delta_ij - fj.pb.tolerances.integrality_tolerance);
  } else {
    new_val = old_val + delta_ij;
  }

  // fallback
  if (!fj.pb.check_variable_within_bounds(var_idx, new_val)) {
    new_val = obj_coeff > 0 ? v_lb : v_ub;
  }

  return new_val;
}

}  // namespace cuopt::linear_programming::detail
