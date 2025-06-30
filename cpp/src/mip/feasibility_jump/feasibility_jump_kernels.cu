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
#include "feasibility_jump.cuh"
#include "feasibility_jump_kernels.cuh"

#include <mip/logger.cuh>
#include <utilities/device_utils.cuh>

#include <raft/random/rng.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CONSTRAINT_FLAG_INSERT 0
#define CONSTRAINT_FLAG_REMOVE 1

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t, typename Iterator>
static DI f_t KahanBabushkaNeumaierSum(Iterator begin, Iterator end)
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

template <typename i_t, typename f_t>
DI thrust::pair<f_t, f_t> move_objective_score(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj, i_t var_idx, f_t delta)
{
  f_t obj_diff = fj.pb.objective_coefficients[var_idx] * delta;

  f_t base_obj = 0;
  if (obj_diff < 0)  // improving move wrt objective
    base_obj = *fj.objective_weight;
  else if (obj_diff > 0)
    base_obj = -*fj.objective_weight;

  f_t bonus_breakthrough = 0;

  bool old_obj_better = *fj.incumbent_objective < *fj.best_objective;
  bool new_obj_better = *fj.incumbent_objective + obj_diff < *fj.best_objective;
  if (!old_obj_better && new_obj_better)
    bonus_breakthrough += *fj.objective_weight;
  else if (old_obj_better && !new_obj_better) {
    bonus_breakthrough -= *fj.objective_weight;
  }

  return {base_obj, bonus_breakthrough};
}

template <typename i_t, typename f_t>
DI std::pair<f_t, f_t> feas_score_constraint(
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
    cuopt_assert(isfinite(old_lhs), "");
    cuopt_assert(isfinite(new_lhs), "");
    cuopt_assert(isfinite(old_slack) && isfinite(new_slack), "");

    f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx);

    bool old_viol = fj.excess_score(cstr_idx, current_lhs) < -cstr_tolerance;
    bool new_viol = fj.excess_score(cstr_idx, current_lhs + cstr_coeff * delta) < -cstr_tolerance;

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
DI void smooth_weights(typename fj_t<i_t, f_t>::climber_data_t::view_t& fj)
{
  for (i_t cstr_idx = TH_ID_X; cstr_idx < fj.pb.n_constraints; cstr_idx += GRID_STRIDE) {
    // consider only satisfied constraints
    if (fj.violated_constraints.contains(cstr_idx)) continue;

    f_t weight_l = max((f_t)0, fj.cstr_left_weights[cstr_idx] - 1);
    f_t weight_r = max((f_t)0, fj.cstr_right_weights[cstr_idx] - 1);

    fj.cstr_left_weights[cstr_idx]  = weight_l;
    fj.cstr_right_weights[cstr_idx] = weight_r;
    fj.cstr_weights[cstr_idx]       = max(weight_l, weight_r);
  }

  if (FIRST_THREAD) {
    if (*fj.incumbent_objective >= *fj.best_objective && *fj.objective_weight > 0) {
      *fj.objective_weight       = max((f_t)0, *fj.objective_weight - 1);
      *fj.full_refresh_iteration = *fj.iterations;
    }
  }
}

// update the weight of violated constraints and the score of the corresponding moves
template <typename i_t, typename f_t>
DI void update_weights(typename fj_t<i_t, f_t>::climber_data_t::view_t& fj)
{
  raft::random::PCGenerator rng(fj.settings->seed + *fj.iterations, 0, 0);
  bool smoothing = rng.next_float() <= fj.settings->parameters.weight_smoothing_probability;

  if (smoothing) return smooth_weights<i_t, f_t>(fj);

  for (i_t i = blockIdx.x; i < fj.violated_constraints.size(); i += gridDim.x) {
    i_t cstr_idx           = fj.violated_constraints.contents[i];
    f_t curr_incumbent_lhs = fj.incumbent_lhs[cstr_idx];
    f_t curr_lower_excess  = fj.lower_excess_score(cstr_idx, curr_incumbent_lhs);
    f_t curr_upper_excess  = fj.upper_excess_score(cstr_idx, curr_incumbent_lhs);
    f_t curr_excess_score  = curr_lower_excess + curr_upper_excess;

    f_t old_weight;
    if (curr_lower_excess < 0.) {
      old_weight = fj.cstr_left_weights[cstr_idx];
    } else {
      old_weight = fj.cstr_right_weights[cstr_idx];
    }

    cuopt_assert(curr_excess_score < 0, "constraint not violated");

    i_t int_delta = fj.weight_update_increment;
    f_t delta     = int_delta;

    f_t new_weight = old_weight + delta;
    new_weight     = round(new_weight);

    f_t new_left_weight  = fj.cstr_left_weights[cstr_idx];
    f_t new_right_weight = fj.cstr_right_weights[cstr_idx];
    if (curr_lower_excess < 0.) {
      new_left_weight = new_weight;
    } else {
      new_right_weight = new_weight;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      // DEVICE_LOG_DEBUG("weight of con %d updated to %g, excess %f\n", cstr_idx, new_weight,
      //     curr_excess_score);
      fj.cstr_weights[cstr_idx] = max(fj.cstr_weights[cstr_idx], new_weight);
      if (curr_lower_excess < 0.) {
        fj.cstr_left_weights[cstr_idx] = new_weight;
      } else {
        fj.cstr_right_weights[cstr_idx] = new_weight;
      }
      atomicMax(fj.max_cstr_weight, new_weight);
    }
  }

  if (FIRST_THREAD) {
    if (fj.violated_constraints.empty() && !fj.settings->feasibility_run) {
      *fj.objective_weight += 1;
      *fj.full_refresh_iteration = *fj.iterations;
    }
  }
}

template <typename i_t, typename f_t>
__global__ void init_lhs_and_violation(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  for (i_t cstr_idx = TH_ID_X; cstr_idx < fj.pb.n_constraints; cstr_idx += GRID_STRIDE) {
    auto [offset_begin, offset_end] = fj.pb.range_for_constraint(cstr_idx);

    auto delta_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), [fj] __device__(i_t j) {
        return fj.pb.coefficients[j] * fj.incumbent_assignment[fj.pb.variables[j]];
      });
    fj.incumbent_lhs[cstr_idx] =
      KahanBabushkaNeumaierSum<i_t, f_t>(delta_it + offset_begin, delta_it + offset_end);
    fj.incumbent_lhs_sumcomp[cstr_idx] = 0;

    f_t th_violation       = fj.excess_score(cstr_idx, fj.incumbent_lhs[cstr_idx]);
    f_t weighted_violation = th_violation * fj.cstr_weights[cstr_idx];
    atomicAdd(fj.violation_score, th_violation);
    atomicAdd(fj.weighted_violation_score, weighted_violation);
    f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx);
    if (th_violation < -cstr_tolerance) { fj.violated_constraints.insert(cstr_idx); }
  }
}

template <typename i_t, typename f_t, i_t TPB, bool IgnoreCstrWeights = false>
DI typename fj_t<i_t, f_t>::move_score_info_t compute_new_score(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj, i_t var_idx, f_t delta)
{
  typedef cub::BlockReduce<f_t, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ typename BlockReduce::TempStorage temp_storage_2;

  f_t obj_diff = fj.pb.objective_coefficients[var_idx] * delta;

  if (threadIdx.x == 0) {
    cuopt_assert(isfinite(delta), "");
    cuopt_assert(blockDim.x == TPB, "invalid threads per block");

    // Objective part of the move score
    cuopt_assert(isfinite(fj.pb.objective_coefficients[var_idx]), "");
    cuopt_assert(isfinite(*fj.objective_weight), "");
    cuopt_assert(isfinite(fj.pb.objective_scaling_factor), "");
  }

  f_t base_feas                   = 0;
  f_t bonus_robust                = 0;
  auto [offset_begin, offset_end] = fj.pb.reverse_range_for_var(var_idx);
  for (i_t i = threadIdx.x + offset_begin; i < offset_end; i += blockDim.x) {
    auto cstr_idx   = fj.pb.reverse_constraints[i];
    auto cstr_coeff = fj.pb.reverse_coefficients[i];

    f_t c_lb = fj.pb.constraint_lower_bounds[cstr_idx];
    f_t c_ub = fj.pb.constraint_upper_bounds[cstr_idx];

    auto [cstr_base_feas, cstr_bonus_robust] = feas_score_constraint<i_t, f_t>(
      fj, var_idx, delta, cstr_idx, cstr_coeff, c_lb, c_ub, fj.incumbent_lhs[cstr_idx]);

    base_feas += cstr_base_feas;
    bonus_robust += cstr_bonus_robust;
  }
  f_t base_feas_sum    = BlockReduce(temp_storage).Sum(base_feas);
  f_t bonus_robust_sum = BlockReduce(temp_storage_2).Sum(bonus_robust);
  if (threadIdx.x == 0) {
    cuopt_assert(isfinite(base_feas_sum), "");
    cuopt_assert(isfinite(bonus_robust_sum), "");
  }

  // New scoring approach according to An Efficient Local Search Solver for Mixed Integer
  // Programming LIPIcs.CP.2024.19

  f_t base_obj = 0;
  if (obj_diff < 0)  // improving move wrt objective
    base_obj = *fj.objective_weight;
  else if (obj_diff > 0)
    base_obj = -*fj.objective_weight;

  f_t bonus_breakthrough = 0;

  bool old_obj_better = *fj.incumbent_objective < *fj.best_objective;
  bool new_obj_better = *fj.incumbent_objective + obj_diff < *fj.best_objective;
  if (!old_obj_better && new_obj_better)
    bonus_breakthrough += *fj.objective_weight;
  else if (old_obj_better && !new_obj_better) {
    bonus_breakthrough -= *fj.objective_weight;
  }

  typename fj_t<i_t, f_t>::move_score_info_t score_info;

  score_info.score.base    = round(base_obj + base_feas_sum);
  score_info.score.bonus   = round(bonus_breakthrough + bonus_robust_sum);
  score_info.infeasibility = base_feas_sum;

  if (threadIdx.x == 0) {
    cuopt_assert(
      fj.pb.is_integer(score_info.score.base) && fj.pb.is_integer(score_info.score.bonus),
      "score should be integer");
  }

  return score_info;
}

template <typename i_t, typename f_t, MTMMoveType move_type>
DI thrust::tuple<f_t, f_t, f_t, f_t> get_mtm_for_constraint(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
  i_t var_idx,
  i_t cstr_idx,
  f_t cstr_coeff)
{
  f_t sign     = -1;
  f_t delta_ij = 0;
  f_t slack    = 0;

  f_t c_lb           = fj.pb.constraint_lower_bounds[cstr_idx];
  f_t c_ub           = fj.pb.constraint_upper_bounds[cstr_idx];
  f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx);

  f_t old_val = fj.incumbent_assignment[var_idx];

  // process each bound as two separate constraints
  f_t bounds[2] = {c_lb, c_ub};
  cuopt_assert(isfinite(bounds[0]) || isfinite(bounds[1]), "");

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

// find best mixed tight move
template <typename i_t, typename f_t, i_t TPB, MTMMoveType move_type>
DI std::pair<f_t, typename fj_t<i_t, f_t>::move_score_info_t> compute_best_mtm(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj, i_t var_idx)
{
  f_t best_val         = fj.incumbent_assignment[var_idx];
  auto best_score_info = fj_t<i_t, f_t>::move_score_info_t::invalid();

  // fixed variables
  if (fj.pb.integer_equal(fj.pb.variable_lower_bounds[var_idx],
                          fj.pb.variable_upper_bounds[var_idx])) {
    return std::make_pair(fj.pb.variable_lower_bounds[var_idx],
                          fj_t<i_t, f_t>::move_score_info_t::invalid());
  }

  f_t old_val   = fj.incumbent_assignment[var_idx];
  f_t obj_coeff = fj.pb.objective_coefficients[var_idx];
  f_t v_lb      = fj.pb.variable_lower_bounds[var_idx];
  f_t v_ub      = fj.pb.variable_upper_bounds[var_idx];
  raft::random::PCGenerator rng(fj.settings->seed + *fj.iterations, 0, 0);
  cuopt_assert(isfinite(v_lb) || isfinite(v_ub), "unexpected free variable");

  auto [offset_begin, offset_end] = fj.pb.reverse_range_for_var(var_idx);
  for (auto i = offset_begin; i < offset_end; i += 1) {
    auto cstr_idx   = fj.pb.reverse_constraints[i];
    auto cstr_coeff = fj.pb.reverse_coefficients[i];

    cuopt_assert(isfinite(fj.incumbent_lhs[cstr_idx]), "");

    if constexpr (move_type == MTMMoveType::FJ_MTM_VIOLATED) {
      // only consider violated constraints
      if (!fj.violated_constraints.contains(cstr_idx)) continue;
    } else if constexpr (move_type == MTMMoveType::FJ_MTM_SATISFIED) {
      // only consider satisfied constraints
      if (fj.violated_constraints.contains(cstr_idx)) continue;
      // sample only a limited amounts of constraints in satisfied mode
      // reservoir sampling to get 20 random constraints
      f_t selection_probability = min(1.0, 20.0 / (offset_end - offset_begin));
      if (rng.next_double() > selection_probability) continue;
    }

    f_t new_val;
    auto [delta_ij, sign, slack, cstr_tolerance] =
      get_mtm_for_constraint<i_t, f_t, move_type>(fj, var_idx, cstr_idx, cstr_coeff);
    if (fj.pb.is_integer_var(var_idx)) {
      new_val = cstr_coeff * sign > 0
                  ? floor(old_val + delta_ij + fj.pb.tolerances.integrality_tolerance)
                  : ceil(old_val + delta_ij - fj.pb.tolerances.integrality_tolerance);
    } else {
      new_val = old_val + delta_ij;
    }

    // fallback
    if (!fj.pb.check_variable_within_bounds(var_idx, new_val)) {
      if (fj.settings->feasibility_run) {
        new_val = min(max(new_val, v_lb), v_ub);
      } else {
        new_val = cstr_coeff * sign > 0 ? v_lb : v_ub;
      }
    }

    if (fj.pb.integer_equal(new_val, old_val) || !isfinite(new_val)) continue;

    if (threadIdx.x == 0) {
      cuopt_assert(isfinite(new_val), "");
      cuopt_assert(fj.pb.check_variable_within_bounds(var_idx, new_val), "");
    }

    auto new_score_info = compute_new_score<i_t, f_t, TPB>(fj, var_idx, new_val - old_val);
    if (threadIdx.x == 0) {
      // reject this move if it would increase the target variable to a numerically unstable value
      if (fj.move_numerically_stable(old_val, new_val, new_score_info.infeasibility)) {
        if (new_score_info.score > best_score_info.score ||
            (new_score_info.score == best_score_info.score && new_val < best_val)) {
          best_score_info = new_score_info;
          best_val        = new_val;
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    cuopt_assert(fj.pb.check_variable_within_bounds(var_idx, best_val), "Var not within bounds!");
    cuopt_assert(isfinite(fj.incumbent_assignment[var_idx]), "");
    cuopt_assert(isfinite(best_val - fj.incumbent_assignment[var_idx]),
                 "Jump move delta is not finite!");
    if (best_score_info.score.valid()) {
      cuopt_assert(!fj.pb.integer_equal(best_val, old_val), "Jump move delta is null!");
    }
  }

  // only returns valid data on thread 0
  return std::make_pair(best_val, best_score_info);
}

template <typename i_t, typename f_t, MTMMoveType move_type, bool is_binary_pb = false>
DI void update_jump_value(typename fj_t<i_t, f_t>::climber_data_t::view_t fj, i_t var_idx)
{
  cuopt_assert(var_idx >= 0 && var_idx < fj.pb.n_variables, "invalid variable index");

  // skip binary variables when considering MTM satisfied moves
  if constexpr (move_type == MTMMoveType::FJ_MTM_SATISFIED) {
    if (fj.pb.is_binary_variable[var_idx]) return;
  }

  // Fast path for binary variables: the only valid jump value not equal to x is ~x
  f_t delta;
  auto best_score_info = fj_t<i_t, f_t>::move_score_info_t::invalid();
  if constexpr (!is_binary_pb) {
    if (fj.pb.is_binary_variable[var_idx] && fj.pb.is_integer(fj.incumbent_assignment[var_idx])) {
      delta = 1.0 - 2 * fj.incumbent_assignment[var_idx];
      if (threadIdx.x == 0) {
        cuopt_assert(fj.incumbent_assignment[var_idx] == 0 || fj.incumbent_assignment[var_idx] == 1,
                     "Current assignment is not binary!");
        cuopt_assert(
          fj.pb.variable_lower_bounds[var_idx] == 0 && fj.pb.variable_upper_bounds[var_idx] == 1,
          "");
        cuopt_assert(
          fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx] + delta),
          "Var not within bounds!");
      }
      best_score_info = compute_new_score<i_t, f_t, TPB_resetmoves>(fj, var_idx, delta);
    } else {
      auto [best_val, score_info] =
        compute_best_mtm<i_t, f_t, TPB_resetmoves, move_type>(fj, var_idx);
      delta           = best_val - fj.incumbent_assignment[var_idx];
      best_score_info = score_info;
    }
  } else {
    delta = 1.0 - 2 * fj.incumbent_assignment[var_idx];
    if (threadIdx.x == 0) {
      cuopt_assert(
        fj.pb.variable_lower_bounds[var_idx] == 0 && fj.pb.variable_upper_bounds[var_idx] == 1, "");
      cuopt_assert(
        fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx] + delta),
        "Var not within bounds!");
    }
    best_score_info = compute_new_score<i_t, f_t, TPB_resetmoves>(fj, var_idx, delta);
  }

  if (threadIdx.x == 0) {
    cuopt_assert(
      fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx] + delta),
      "Var not within bounds!");
    fj.jump_move_delta[var_idx]         = delta;
    fj.jump_move_scores[var_idx]        = best_score_info.score;
    fj.jump_move_infeasibility[var_idx] = best_score_info.infeasibility;
  }
}

template <typename i_t, typename f_t>
DI void check_variable_feasibility(const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
                                   bool check_integer)
{
  for (i_t var_idx = threadIdx.x; var_idx < fj.pb.n_variables; var_idx += blockDim.x) {
    auto val      = fj.incumbent_assignment[var_idx];
    bool feasible = fj.pb.check_variable_within_bounds(var_idx, val);

    cuopt_assert(feasible, "invalid variable assignment");
    if (check_integer && fj.pb.is_integer_var(var_idx))
      cuopt_assert(fj.pb.is_integer(fj.incumbent_assignment[var_idx]), "integrality violation");
  }
}

template <typename i_t, typename f_t>
DI bool check_feasibility(const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
                          bool check_integer = true)
{
  for (i_t cIdx = threadIdx.x; cIdx < fj.pb.n_constraints; cIdx += blockDim.x) {
    auto [offset_begin, offset_end] = fj.pb.range_for_constraint(cIdx);
    auto delta_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), [fj] __device__(i_t j) {
        return fj.pb.coefficients[j] * fj.incumbent_assignment[fj.pb.variables[j]];
      });

    f_t lhs = KahanBabushkaNeumaierSum<i_t, f_t>(delta_it + offset_begin, delta_it + offset_end);
    cuopt_assert(fj.cstr_satisfied(cIdx, lhs), "constraint violated");
  }
  cuopt_func_call((check_variable_feasibility<i_t, f_t>(fj, check_integer)));

  return true;
}

template <typename i_t, typename f_t>
DI void save_best_l1_distance_solution(typename fj_t<i_t, f_t>::climber_data_t::view_t& fj)
{
  __shared__ f_t shmem[raft::WarpSize];
  __shared__ f_t res;
  // compute l1 distance on integer indices
  compute_l1_distance_block<i_t, f_t>(
    fj.incumbent_assignment, fj.farthest_l1_sol, fj.pb.integer_indices, shmem, res);
  __syncthreads();
  if (res > *fj.best_l1_distance) {
    __syncthreads();
    *fj.best_l1_distance = res;
    for (i_t i = threadIdx.x; i < fj.pb.n_variables; i += blockDim.x) {
      fj.farthest_l1_sol[i] = fj.incumbent_assignment[i];
    }
  }
}

template <typename i_t, typename f_t>
DI bool save_best_solution(typename fj_t<i_t, f_t>::climber_data_t::view_t& fj)
{
  cuopt_assert(blockIdx.x == 0, "Only a single block can run!");
  __shared__ bool save_sol;
  if (fj.settings->keep_farthest_l1_sol) { save_best_l1_distance_solution<i_t, f_t>(fj); }

  if (FIRST_THREAD) {
    bool better_objective = *fj.incumbent_objective < *fj.saved_solution_objective;
    save_sol              = better_objective && fj.violated_constraints.size() == 0;
    // save least infeasible solution
    if (*fj.best_excess < 0 && *fj.violation_score > *fj.best_excess) {
      if (fj.violated_constraints.size() == 0)
        *fj.best_excess = 0;
      else
        *fj.best_excess = *fj.violation_score;
      save_sol = true;
    }
  }

  __syncthreads();
  if (save_sol) {
    if (FIRST_THREAD) {
      // DEVICE_LOG_TRACE("Updating best objective from %f to %f. vio %f \n",
      //                  *fj.saved_solution_objective,
      //                  *fj.incumbent_objective,
      //                  *fj.violation_score);
      if (*fj.violation_score == 0.) {
        cuopt_assert(fj.violated_constraints.size() == 0, "Violated constraint and score mismatch");
      }

      if (*fj.best_excess == 0) { *fj.saved_solution_objective = *fj.incumbent_objective; }
    }
    cuopt_func_call((check_variable_feasibility<i_t, f_t>(fj, false)));
    for (i_t i = threadIdx.x; i < fj.pb.n_variables; i += blockDim.x) {
      fj.best_assignment[i] = fj.incumbent_assignment[i];
    }
    __syncthreads();
  }
  if (fj.violated_constraints.size() == 0) {
    cuopt_assert(
      *fj.weighted_violation_score <= *fj.max_cstr_weight * fj.pb.tolerances.absolute_tolerance,
      "Violated constraint and score mismatch");
    cuopt_func_call((check_feasibility<i_t, f_t>(fj)));
  }
  // return whether it is an improving local minimum
  return save_sol;
}

template <typename i_t, typename f_t>
DI void check_exit_condition(typename fj_t<i_t, f_t>::climber_data_t::view_t& view)
{
  // if the mode is GREEDY_DESCENT and we reached a local minimum exit
  if (view.settings->mode == fj_mode_t::GREEDY_DESCENT) {
    *view.temp_break_condition = 1;
  } else if (view.settings->mode == fj_mode_t::EXIT_NON_IMPROVING) {
    i_t last_improving_minimum = *view.last_improving_minimum;

    // exit condition is there if we are still improving and max weight reached a threshold
    bool local_min_cond = (*view.local_minimums_reached - last_improving_minimum) >
                          view.settings->n_of_minimums_for_exit;
    bool is_feasible = view.settings->feasibility_run && view.violated_constraints.size() == 0;
    // bool max_weight_reached = *view.max_cstr_weight > view.stop_threshold;
    if (local_min_cond || is_feasible) { *view.temp_break_condition = 1; }
  }
}

template <typename i_t, typename f_t>
__global__ void update_assignment_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
                                         bool IgnoreLoadBalancing)
{
  raft::random::PCGenerator rng(fj.settings->seed, *fj.iterations, 0);

  // in a graph mode, we run multiple iterations and we want to stop when break condition is reached
  if (*fj.break_condition) return;
  i_t var_idx = *fj.selected_var;

  // no valid moves found this iteration (might happen because of tabu)
  if (var_idx == std::numeric_limits<i_t>::max()) return;

  cuopt_assert(var_idx >= 0 && var_idx < fj.pb.n_variables, "invalid selected variable index");
  cuopt_assert(isfinite(fj.jump_move_delta[var_idx]), "jump value isn't finite");

  // ensure the move we took is valid for the current running mode
  if (FIRST_THREAD) {
    if (*fj.iterations_until_feasible_counter > 0) --*fj.iterations_until_feasible_counter;
  }

  // Update the LHSs of all involved constraints.
  auto [offset_begin, offset_end] = fj.pb.reverse_range_for_var(var_idx);

  for (auto i = offset_begin + blockIdx.x; i < offset_end; i += gridDim.x) {
    cuopt_assert(i < (i_t)fj.pb.reverse_constraints.size(), "");

    auto cstr_idx   = fj.pb.reverse_constraints[i];
    auto cstr_coeff = fj.pb.reverse_coefficients[i];

    f_t old_lhs  = fj.incumbent_lhs[cstr_idx];
    f_t new_lhs  = old_lhs + cstr_coeff * fj.jump_move_delta[var_idx];
    f_t old_cost = fj.excess_score(cstr_idx, old_lhs);
    f_t new_cost = fj.excess_score(cstr_idx, new_lhs);

    if (threadIdx.x == 0) {
      cuopt_assert(
        *fj.constraints_changed_count >= 0 && *fj.constraints_changed_count <= fj.pb.n_constraints,
        "");
      f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx);
      if (new_cost < -cstr_tolerance && !fj.violated_constraints.contains(cstr_idx))
        fj.constraints_changed[atomicAdd(fj.constraints_changed_count, 1)] =
          (cstr_idx << 1) | CONSTRAINT_FLAG_INSERT;
      else if (!(new_cost < -cstr_tolerance) && fj.violated_constraints.contains(cstr_idx))
        fj.constraints_changed[atomicAdd(fj.constraints_changed_count, 1)] =
          cstr_idx << 1 | CONSTRAINT_FLAG_REMOVE;
    }

    __syncthreads();

    cuopt_assert(isfinite(fj.jump_move_delta[var_idx]), "delta should be finite");
    // Kahan compensated summation
    // fj.incumbent_lhs[cstr_idx] = old_lhs + cstr_coeff * fj.jump_move_delta[var_idx];
    f_t y = cstr_coeff * fj.jump_move_delta[var_idx] - fj.incumbent_lhs_sumcomp[cstr_idx];
    f_t t = old_lhs + y;
    fj.incumbent_lhs_sumcomp[cstr_idx] = (t - old_lhs) - y;
    fj.incumbent_lhs[cstr_idx]         = t;
    cuopt_assert(isfinite(fj.incumbent_lhs[cstr_idx]), "assignment should be finite");
  }

  // update the assignment and objective proper
  if (FIRST_THREAD) {
    f_t new_val = fj.incumbent_assignment[var_idx] + fj.jump_move_delta[var_idx];

    cuopt_assert(fj.pb.check_variable_within_bounds(var_idx, new_val),
                 "assignment not within bounds");
    cuopt_assert(isfinite(new_val), "assignment is not finite");

    if (fj.pb.is_integer_var(var_idx)) {
      cuopt_assert(fj.pb.is_integer(new_val), "The variable must be integer");
      new_val = round(new_val);
    }

#if defined(FJ_SINGLE_STEP)
    DEVICE_LOG_TRACE(
      "=---- FJ: updated %d [%g/%g] :%.4g+{%.4g}=%.4g score {%g,%g}, d_obj %.2g+%.2g=%.2g, "
      "infeas %.2g, total viol %d\n",
      var_idx,
      fj.pb.variable_lower_bounds[var_idx],
      fj.pb.variable_upper_bounds[var_idx],
      fj.incumbent_assignment[var_idx],
      fj.jump_move_delta[var_idx],
      fj.incumbent_assignment[var_idx] + fj.jump_move_delta[var_idx],
      fj.jump_move_scores[var_idx].base,
      fj.jump_move_scores[var_idx].bonus,
      *fj.incumbent_objective,
      fj.jump_move_delta[var_idx] * fj.pb.objective_coefficients[var_idx],
      *fj.incumbent_objective + fj.jump_move_delta[var_idx] * fj.pb.objective_coefficients[var_idx],
      fj.jump_move_infeasibility[var_idx],
      fj.violated_constraints.size());
#endif
    // reset the score
    fj.jump_move_scores[var_idx]        = fj_t<i_t, f_t>::move_score_t::invalid();
    fj.jump_move_infeasibility[var_idx] = -std::numeric_limits<f_t>::infinity();

    fj.incumbent_assignment[var_idx] = new_val;

    *fj.incumbent_objective += fj.pb.objective_coefficients[var_idx] * fj.jump_move_delta[var_idx];

    cuopt_assert(
      fj.settings->parameters.tabu_tenure_max >= 0 &&
        fj.settings->parameters.tabu_tenure_max >= fj.settings->parameters.tabu_tenure_min,
      "Invalid tabu tenure values");
    i_t tabu_tenure = fj.settings->parameters.tabu_tenure_min +
                      rng.next_u32() % (fj.settings->parameters.tabu_tenure_max -
                                        fj.settings->parameters.tabu_tenure_min);
    if (fj.jump_move_delta[var_idx] > 0) {
      fj.tabu_lastinc[var_idx]     = *fj.iterations;
      fj.tabu_nodec_until[var_idx] = *fj.iterations + tabu_tenure;
    } else if (fj.jump_move_delta[var_idx] < 0) {
      fj.tabu_lastdec[var_idx]     = *fj.iterations;
      fj.tabu_noinc_until[var_idx] = *fj.iterations + tabu_tenure;
    }
  }
}

template <typename i_t, typename f_t>
DI void update_violation_score_block(typename fj_t<i_t, f_t>::climber_data_t::view_t& fj)
{
  __shared__ f_t shbuf[raft::WarpSize];
  f_t th_violation          = 0.;
  f_t th_weighted_violation = 0.;
  for (i_t i = threadIdx.x; i < fj.violated_constraints.size(); i += blockDim.x) {
    i_t cstr_idx       = fj.violated_constraints.contents[i];
    f_t lhs            = fj.incumbent_lhs[cstr_idx];
    f_t score          = fj.excess_score(cstr_idx, lhs);
    f_t weighted_score = score * fj.cstr_weights[cstr_idx];
    th_violation += score;
    th_weighted_violation += weighted_score;
  }
  f_t global_violation_score = raft::blockReduce(th_violation, (char*)shbuf);
  __syncthreads();
  f_t weighted_violation_score = raft::blockReduce(th_weighted_violation, (char*)shbuf);
  if (threadIdx.x == 0) {
    *fj.weighted_violation_score = weighted_violation_score;
    *fj.violation_score          = global_violation_score;
    *fj.incumbent_quality =
      weighted_violation_score + (*fj.incumbent_objective) * (-*fj.objective_weight);
  }
}

template <typename i_t, typename f_t, i_t TPB>
DI void update_lift_moves(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  if (*fj.break_condition) return;
  // ignore if the solution isn't feasible right now
  if (!fj.violated_constraints.empty()) return;
  cuopt_assert(fj.violated_constraints.size() == 0, "");

  cuopt_assert(TPB == blockDim.x, "Invalid TPB");
  typedef cub::BlockReduce<f_t, TPB> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage cub;
    f_t raft[2 * raft::WarpSize];
  } shmem;

  for (i_t i = blockIdx.x; i < fj.objective_vars.size(); i += gridDim.x) {
    i_t var_idx   = fj.objective_vars[i];
    f_t obj_coeff = fj.pb.objective_coefficients[var_idx];
    f_t delta     = -std::numeric_limits<f_t>::infinity();

    f_t th_lower_domain             = fj.pb.variable_lower_bounds[var_idx];
    f_t th_upper_domain             = fj.pb.variable_upper_bounds[var_idx];
    auto [offset_begin, offset_end] = fj.pb.reverse_range_for_var(var_idx);
    for (i_t j = threadIdx.x + offset_begin; j < offset_end; j += blockDim.x) {
      auto cstr_idx   = fj.pb.reverse_constraints[j];
      auto cstr_coeff = fj.pb.reverse_coefficients[j];
      f_t c_lb        = fj.pb.constraint_lower_bounds[cstr_idx];
      f_t c_ub        = fj.pb.constraint_upper_bounds[cstr_idx];
      cuopt_assert(c_lb <= c_ub, "invalid bounds");
      cuopt_assert(fj.cstr_satisfied(cstr_idx, fj.incumbent_lhs[cstr_idx]),
                   "cstr should be satisfied");

      auto [delta, sign, slack, cstr_tolerance] =
        get_mtm_for_constraint<i_t, f_t, MTMMoveType::FJ_MTM_SATISFIED>(
          fj, var_idx, cstr_idx, cstr_coeff);

      // skip this variable if there is no slack
      if (fabs(slack) <= cstr_tolerance) {
        th_lower_domain = th_upper_domain = fj.incumbent_assignment[var_idx];
        continue;
      }

      cuopt_assert(isfinite(delta), "");
      cuopt_assert(isfinite(fj.incumbent_assignment[var_idx] + delta), "");
      if (cstr_coeff * sign < 0) {
        if (fj.pb.is_integer_var(var_idx)) delta = ceil(delta);
        th_lower_domain = max(th_lower_domain, fj.incumbent_assignment[var_idx] + delta);
      } else {
        if (fj.pb.is_integer_var(var_idx)) delta = floor(delta);
        th_upper_domain = min(th_upper_domain, fj.incumbent_assignment[var_idx] + delta);
      }
    }

    // cub::BlockReduce because raft::blockReduce has a bug when using min() w/ floats
    // lfd = lift feasible domain
    f_t lfd_lb = BlockReduce(shmem.cub).Reduce(th_lower_domain, cuda::maximum());
    __syncthreads();
    f_t lfd_ub = BlockReduce(shmem.cub).Reduce(th_upper_domain, cuda::minimum());

    if (lfd_lb >= lfd_ub) continue;

    // Now that the life move domain is computed, compute the correct lift move
    cuopt_assert(isfinite(fj.incumbent_assignment[var_idx]), "invalid assignment value");
    delta = obj_coeff < 0 ? lfd_ub : lfd_lb;
    delta = delta - fj.incumbent_assignment[var_idx];

    // get the score
    auto score = fj_t<i_t, f_t>::move_score_t::zero();

    f_t obj_score = -1 * obj_coeff * delta;  // negated to turn this into a positive score
    score.base    = obj_score;
    if (threadIdx.x == 0 && isfinite(delta) && !fj.pb.integer_equal(delta, (f_t)0)) {
      fj.move_delta(FJ_MOVE_LIFT, var_idx)       = delta;
      fj.move_score(FJ_MOVE_LIFT, var_idx)       = score;
      fj.move_last_update(FJ_MOVE_LIFT, var_idx) = *fj.iterations;
    }
  }

  // if (TH_ID_X == 0) DEVICE_LOG_TRACE("lift move scan\n");
}

template <typename i_t, typename f_t>
__global__ void update_lift_moves_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  update_lift_moves<i_t, f_t, TPB_liftmoves>(fj);
}

template <typename i_t, typename f_t>
DI f_t get_breakthrough_move(typename fj_t<i_t, f_t>::climber_data_t::view_t fj, i_t var_idx)
{
  f_t obj_coeff = fj.pb.objective_coefficients[var_idx];
  f_t v_lb      = fj.pb.variable_lower_bounds[var_idx];
  f_t v_ub      = fj.pb.variable_upper_bounds[var_idx];
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

template <typename i_t, typename f_t, i_t TPB>
DI void update_breakthrough_moves(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  if (*fj.break_condition) return;
  // breakthru moves are only considered if we have found a feasible solution before and the
  // incumbent is worse
  if (*fj.best_objective == std::numeric_limits<f_t>::infinity() ||
      *fj.incumbent_objective <= *fj.best_objective)
    return;

  cuopt_assert(TPB == blockDim.x, "Invalid TPB");

  for (i_t i = blockIdx.x; i < fj.objective_vars.size(); i += gridDim.x) {
    i_t var_idx = fj.objective_vars[i];
    if (fj.pb.is_binary_variable[var_idx]) continue;

    f_t old_val = fj.incumbent_assignment[var_idx];
    f_t new_val = get_breakthrough_move<i_t, f_t>(fj, var_idx);

    if (fj.pb.integer_equal(new_val, old_val) || !isfinite(new_val)) continue;

    f_t delta           = new_val - old_val;
    auto new_score_info = compute_new_score<i_t, f_t, TPB>(fj, var_idx, delta);
    if (threadIdx.x == 0) {
      cuopt_assert(fj.pb.check_variable_within_bounds(var_idx, new_val), "");
      cuopt_assert(isfinite(delta), "");
      fj.move_delta(FJ_MOVE_BREAKTHROUGH, var_idx)       = delta;
      fj.move_score(FJ_MOVE_BREAKTHROUGH, var_idx)       = new_score_info.score;
      fj.move_last_update(FJ_MOVE_BREAKTHROUGH, var_idx) = *fj.iterations;
    }

    __syncthreads();
  }
}

template <typename i_t, typename f_t>
__global__ void update_breakthrough_moves_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  update_breakthrough_moves<i_t, f_t, TPB_liftmoves>(fj);
}

// TODO: rewrite the constraint violation update code for better parallelization (bitsets,
// compact/select?)
template <typename i_t, typename f_t>
DI void update_changed_constraints(typename fj_t<i_t, f_t>::climber_data_t::view_t& fj)
{
  if (*fj.break_condition) return;

  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      for (i_t i = 0; i < *fj.constraints_changed_count; ++i) {
        i_t idx = fj.constraints_changed[i];
        if ((idx & 1) == CONSTRAINT_FLAG_INSERT) {
          fj.violated_constraints.insert(idx >> 1);
        } else {
          fj.violated_constraints.remove(idx >> 1);
        }
      }
      *fj.constraints_changed_count = 0;
    }

    __syncthreads();
    // in order to have deterministic violation score update it in a block
    update_violation_score_block<i_t, f_t>(fj);
    __syncthreads();

    // important to always keep this up to date (not only at minimas) as it is used in the
    // generation of moves
    __shared__ bool update_sol;
    if (threadIdx.x == 0) {
      if (fj.violated_constraints.size() == 0 && *fj.incumbent_objective < *fj.best_objective) {
        *fj.best_objective =
          *fj.incumbent_objective - fj.settings->parameters.breakthrough_move_epsilon;
        update_sol = true;
      } else {
        update_sol = false;
      }
    }
    __syncthreads();
    if (update_sol) { save_best_solution<i_t, f_t>(fj); }
  }
}
template <typename i_t, typename f_t>
__global__ void update_changed_constraints_kernel(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  if (*fj.break_condition) return;

  update_changed_constraints<i_t, f_t>(fj);
}

template <typename i_t, typename f_t>
__global__ void update_best_solution_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  // dont update the solution if we already explored local minimum for greedy descent
  if (blockIdx.x != 0) return;
  save_best_solution<i_t, f_t>(fj);
}

template <typename i_t, typename f_t>
DI void compute_iteration_related_variables(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  auto [offset_begin, offset_end] = fj.pb.reverse_range_for_var(*fj.selected_var);
  for (i_t i = blockIdx.x + offset_begin; i < offset_end; i += gridDim.x) {
    auto cstr_idx = fj.pb.reverse_constraints[i];

    auto [cstr_offset_begin, cstr_offset_end] = fj.pb.range_for_constraint(cstr_idx);

    for (i_t j = cstr_offset_begin + threadIdx.x; j < cstr_offset_end; j += blockDim.x) {
      i_t var_idx = fj.pb.variables[j];

      fj.iteration_related_variables.set(var_idx);
    }
  }
}

template <typename i_t, typename f_t>
__global__ void compute_iteration_related_variables_kernel(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  // perform a full refresh if this was the first iteration (no selected variable yet)
  if (*fj.selected_var == std::numeric_limits<i_t>::max()) {
    for (i_t var_idx = TH_ID_X; var_idx < fj.pb.n_variables; var_idx += GRID_STRIDE) {
      fj.iteration_related_variables.set(var_idx);
    }
    return;
  }

  // if the related var table was precomputed, use it directly
  if (fj.pb.related_variables.size() > 0) {
    auto range = fj.pb.range_for_related_vars(*fj.selected_var);
    for (i_t i = TH_ID_X + range.first; i < range.second; i += GRID_STRIDE) {
      i_t var_idx = fj.pb.related_variables[i];
      fj.iteration_related_variables.set(var_idx);
    }
    return;
  }

  compute_iteration_related_variables<i_t, f_t>(fj);
}

template <typename i_t, typename f_t, MTMMoveType move_type, bool is_binary_pb>
__device__ void compute_mtm_moves(typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
                                  bool ForceRefresh)
{
  if (*fj.break_condition) return;

  bool full_refresh = false;
  if (ForceRefresh || *fj.iterations == 0) full_refresh = true;
  if (*fj.full_refresh_iteration == *fj.iterations) full_refresh = true;
  if (*fj.selected_var == std::numeric_limits<i_t>::max()) full_refresh = true;

  // always do a full sweep when looking for satisfied mtm moves
  if constexpr (move_type == MTMMoveType::FJ_MTM_SATISFIED) full_refresh = true;

  // only update related variables
  i_t split_begin, split_end;
  if (full_refresh) {
    split_begin = 0;
    split_end   = fj.pb.n_variables;
  }
  // related variable table couldn't be computed ahead of time, get related variables dynamically
  else if (fj.pb.related_variables.size() == 0) {
    compute_iteration_related_variables<i_t, f_t>(fj);
    cg::this_grid().sync();
    split_begin = 0;
    split_end   = fj.pb.n_variables;
  }
  // related variable table available
  else {
    cuopt_assert(*fj.selected_var != std::numeric_limits<i_t>::max(), "");
    auto range  = fj.pb.range_for_related_vars(*fj.selected_var);
    split_begin = range.first;
    split_end   = range.second;
  }

  if (FIRST_THREAD) *fj.relvar_count_last_update = split_end - split_begin;

  for (i_t i = blockIdx.x + split_begin; i < split_end; i += gridDim.x) {
    i_t var_idx = full_refresh                          ? i
                  : fj.pb.related_variables.size() == 0 ? i
                                                        : fj.pb.related_variables[i];

    // skip if we couldnt precompute a related var table and
    // this variable isnt in the dynamic related variable table
    if (!full_refresh && fj.pb.related_variables.size() == 0 &&
        !fj.iteration_related_variables.contains(var_idx))
      continue;

    bool exclude_from_search = false;
    // "fixed" variables are to be excluded (as they cannot take any other value)
    exclude_from_search |= fj.pb.integer_equal(fj.pb.variable_lower_bounds[var_idx],
                                               fj.pb.variable_upper_bounds[var_idx]);

    if (exclude_from_search) {
      if (threadIdx.x == 0) {
        fj.jump_move_scores[var_idx]        = fj_t<i_t, f_t>::move_score_t::invalid();
        fj.jump_move_infeasibility[var_idx] = -std::numeric_limits<f_t>::infinity();
        fj.jump_move_delta[var_idx]         = 0;
      }
      continue;
    }

    cuopt_assert(var_idx >= 0 && var_idx < fj.pb.n_variables, "");
    update_jump_value<i_t, f_t, move_type, is_binary_pb>(fj, var_idx);
  }
}

template <typename i_t, typename f_t, MTMMoveType move_type, bool is_binary_pb>
__global__ void compute_mtm_moves_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
                                         bool ForceRefresh)
{
  compute_mtm_moves<i_t, f_t, move_type, is_binary_pb>(fj, ForceRefresh);
}

template <typename i_t, typename f_t>
__global__ void select_variable_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  // in a graph mode, we run multiple iterations and we want to stop when break condition is reached
  if (*fj.break_condition) return;
  raft::random::PCGenerator rng(
    fj.settings->seed, *fj.iterations * fj.settings->parameters.max_sampled_moves, 0);

  using move_score_t = typename fj_t<i_t, f_t>::move_score_t;
  __shared__ alignas(move_score_t) char shmem_storage[2 * raft::WarpSize * sizeof(move_score_t)];
  auto* const shmem = (move_score_t*)shmem_storage;

  auto th_best_score  = fj_t<i_t, f_t>::move_score_t::invalid();
  i_t th_selected_var = std::numeric_limits<i_t>::max();
  i_t selected_var    = std::numeric_limits<i_t>::max();

  i_t good_var_count       = fj.candidate_variables.size();
  *fj.last_iter_candidates = good_var_count;

  if (good_var_count > 0) {
    // FIXME
    auto sample_size = std::min<i_t>(200000, good_var_count);

    // Pick a random variable among a sample of all candidate variables
    // a shuffle would be more accurate, but is slower and more complex for little to no gains
    i_t offset = rng.next_u32() % good_var_count;
    for (i_t i = threadIdx.x; i < sample_size; i += blockDim.x) {
      auto setidx     = (i + offset) % good_var_count;
      auto var_idx    = fj.candidate_variables.contents[setidx];
      auto move_score = fj.jump_move_scores[var_idx];

      if (move_score > th_best_score ||
          (move_score == th_best_score && var_idx > th_selected_var)) {
        th_best_score   = move_score;
        th_selected_var = var_idx;
      }
    }
    // Block level reduction to get the best variable from the sample
    auto [best_score, reduced_selected_var] =
      raft::blockRankedReduce(th_best_score, shmem, th_selected_var, raft::max_op{});
    if (FIRST_THREAD) {
      // assign it to print the value outside
      th_best_score = best_score;
      selected_var  = reduced_selected_var;

      // TODO: test if this actually matters w/ latest improvements / turn into parameters
      // used to emulate the paper's sampling behavior causing some local minima to be triggered
      // even if some improving moves are available (seems to avoid infinite loops on some
      // instances?)
      i_t random_move = 0;
      if (good_var_count < 20) random_move = (rng.next_u32() % 100) < 2;

      if (selected_var == -1 ||
          random_move)  // may happen if all moves are excluded by the tabu list
        selected_var = std::numeric_limits<i_t>::max();
      // only select a random variable if we allow possibly worsening moves
      else if (rng.next_float() < fj.settings->parameters.random_var_probability &&
               (fj.settings->candidate_selection != fj_candidate_selection_t::FEASIBLE_FIRST ||
                *fj.iterations_until_feasible_counter > 0))
        selected_var = fj.candidate_variables.contents[rng.next_u32() % good_var_count];
    }
  }

  // Reset the size of the candidate variable list, which will be updated by reset_moves
  if (FIRST_THREAD) {
    *fj.iterations += 1;
    *fj.candidate_variables.set_size = 0;
    *fj.selected_var                 = selected_var;
    if (selected_var != std::numeric_limits<i_t>::max()) {
#if FJ_SINGLE_STEP
      DEVICE_LOG_INFO(
        "=---- FJ: selected %d [%g/%g] :%.2g+{%.2g}=%.2g score {%g,%g}, d_obj %.2g+%.2g->%.2g, "
        "infeas %.2g, total viol %d, out of %d\n",
        selected_var,
        fj.pb.variable_lower_bounds[selected_var],
        fj.pb.variable_upper_bounds[selected_var],
        fj.incumbent_assignment[selected_var],
        fj.jump_move_delta[selected_var],
        fj.incumbent_assignment[selected_var] + fj.jump_move_delta[selected_var],
        fj.jump_move_scores[selected_var].base,
        fj.jump_move_scores[selected_var].bonus,
        *fj.incumbent_objective,
        fj.jump_move_delta[selected_var] * fj.pb.objective_coefficients[selected_var],
        *fj.incumbent_objective +
          fj.jump_move_delta[selected_var] * fj.pb.objective_coefficients[selected_var],
        fj.jump_move_infeasibility[selected_var],
        fj.violated_constraints.size(),
        good_var_count);
#endif
      cuopt_assert(fj.jump_move_scores[selected_var].valid(), "");
    }
  }
}

template <typename i_t,
          typename f_t,
          i_t TPB,
          bool WeakTabu,
          bool recompute_score,
          typename Callback,
          typename CandidateIterator>
DI thrust::tuple<i_t, f_t, typename fj_t<i_t, f_t>::move_score_t> gridwide_reduce_best_move(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
  CandidateIterator candidates_begin,
  CandidateIterator candidates_end,
  Callback get_move)
{
  auto best_score = fj_t<i_t, f_t>::move_score_t::invalid();
  i_t best_var    = std::numeric_limits<i_t>::max();
  f_t best_delta  = 0;

  for (auto it = candidates_begin + blockIdx.x; it < candidates_end; it += gridDim.x) {
    i_t var_idx = *it;

    // affected by tabu
    f_t delta = get_move(var_idx);
    if constexpr (WeakTabu) {
      if ((delta < 0 && *fj.iterations == fj.tabu_lastinc[var_idx] + 1) ||
          (delta > 0 && *fj.iterations == fj.tabu_lastdec[var_idx] + 1))
        continue;
    } else {
      if ((delta < 0 && *fj.iterations < fj.tabu_nodec_until[var_idx]) ||
          (delta > 0 && *fj.iterations < fj.tabu_noinc_until[var_idx]))
        continue;
    }

    if (!fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx] + delta))
      continue;
    if (fabs(delta) < fj.pb.tolerances.absolute_tolerance) continue;

    typename fj_t<i_t, f_t>::move_score_info_t loc_best_score_info = {};
    loc_best_score_info.score                                      = fj.jump_move_scores[var_idx];
    if constexpr (recompute_score) {
      loc_best_score_info = compute_new_score<i_t, f_t, TPB>(fj, var_idx, delta);
    }

    if (threadIdx.x == 0) {
      if (loc_best_score_info.score > best_score ||
          (loc_best_score_info.score == best_score && var_idx > best_var)) {
        best_score = loc_best_score_info.score;
        best_var   = var_idx;
        best_delta = delta;
      }
    }
  }

  if (threadIdx.x == 0) {
    fj.grid_score_buf[blockIdx.x] = best_score;
    fj.grid_var_buf[blockIdx.x]   = best_var;
    fj.grid_delta_buf[blockIdx.x] = best_delta;
  }

  // grid-wide reduce
  // will be replaced by a proper load balancing scheme
  cg::this_grid().sync();

  if (blockIdx.x == 0) {
    using move_score_t = typename fj_t<i_t, f_t>::move_score_t;
    __shared__ alignas(move_score_t) char shmem_storage[2 * raft::WarpSize * sizeof(move_score_t)];
    auto* const shmem = (move_score_t*)shmem_storage;

    auto th_best_score = fj_t<i_t, f_t>::move_score_t::invalid();
    i_t th_best_block  = 0;
    for (i_t i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      auto var_idx    = fj.grid_var_buf[i];
      auto move_score = fj.grid_score_buf[i];

      if (move_score > th_best_score ||
          (move_score == th_best_score && var_idx > fj.grid_var_buf[th_best_block])) {
        th_best_score = move_score;
        th_best_block = i;
      }
    }
    // Block level reduction to get the best variable from all blocks
    auto [reduced_best_score, reduced_best_block] =
      raft::blockRankedReduce(th_best_score, shmem, th_best_block, raft::max_op{});

    if (reduced_best_score.valid() && threadIdx.x == 0) {
      cuopt_assert(th_best_block < gridDim.x, "");
      best_var   = fj.grid_var_buf[reduced_best_block];
      best_delta = fj.grid_delta_buf[reduced_best_block];
      best_score = fj.grid_score_buf[reduced_best_block];
      cuopt_assert(fj.pb.check_variable_within_bounds(
                     best_var, fj.incumbent_assignment[best_var] + best_delta),
                   "");
    }
  }

  return {best_var, best_delta, best_score};
}

template <typename i_t, typename f_t, i_t TPB>
DI thrust::tuple<i_t, f_t, typename fj_t<i_t, f_t>::move_score_t> best_random_mtm_move(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  cuopt_assert(!fj.violated_constraints.empty(),
               "should only be called when the incumbent is infeasible");
  raft::random::PCGenerator rng(fj.settings->seed + *fj.iterations, 0, 0);

  i_t cstr_idx = fj.violated_constraints.contents[rng.next_u32() % fj.violated_constraints.size()];
  auto [offset_begin, offset_end] = fj.pb.range_for_constraint(cstr_idx);

  return gridwide_reduce_best_move<i_t, f_t, TPB, /*WeakTabu=*/true, /*recompute_score=*/true>(
    fj,
    fj.pb.variables.begin() + offset_begin,
    fj.pb.variables.begin() + offset_end,
    [fj] __device__(i_t var_idx) { return fj.jump_move_delta[var_idx]; });
}

template <typename i_t, typename f_t, i_t TPB>
DI thrust::tuple<i_t, f_t, typename fj_t<i_t, f_t>::move_score_t> best_sat_cstr_mtm_move(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  // compute all MTM moves within satisfied constraints
  compute_mtm_moves<i_t, f_t, MTMMoveType::FJ_MTM_SATISFIED, false>(fj, true);
  return gridwide_reduce_best_move<i_t, f_t, TPB, /*WeakTabu=*/false, /*recompute_score=*/false>(
    fj, fj.objective_vars.begin(), fj.objective_vars.end(), [fj] __device__(i_t var_idx) {
      return fj.jump_move_delta[var_idx];
    });
}

template <typename i_t, typename f_t, i_t TPB>
DI thrust::tuple<i_t, f_t, typename fj_t<i_t, f_t>::move_score_t>
best_breakthrough_move_at_local_min(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  return gridwide_reduce_best_move<i_t, f_t, TPB, /*WeakTabu=*/false, /*recompute_score=*/true>(
    fj, fj.objective_vars.begin(), fj.objective_vars.end(), [fj] __device__(i_t var_idx) {
      return get_breakthrough_move<i_t, f_t>(fj, var_idx) - fj.incumbent_assignment[var_idx];
    });
}

// when we reach the bottom of a greedy descent, increase the weight of the violated constraints
// to escape the local minimum (as outlined in the paper)
template <typename i_t, typename f_t>
__global__ void handle_local_minimum_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  raft::random::PCGenerator rng(fj.settings->seed + *fj.iterations, 0, 0);
  __shared__ typename fj_t<i_t, f_t>::move_score_t shmem[2 * raft::WarpSize];
  if (*fj.break_condition) return;
  // did we reach a local minimum?
  if (*fj.selected_var != std::numeric_limits<i_t>::max()) return;

  auto best_score = fj_t<i_t, f_t>::move_score_t::invalid();
  i_t best_var    = std::numeric_limits<i_t>::max();
  f_t best_delta  = 0;

  char best_movetype = 'M';

  // Local minimum, update weights.
  if (fj.settings->update_weights) update_weights<i_t, f_t>(fj);

  cuopt_assert(blockDim.x == TPB_localmin, "invalid TPB");

  if (blockIdx.x == 0) {
    bool improving = save_best_solution<i_t, f_t>(fj);
    if (threadIdx.x == 0) {
      *fj.last_minimum_iteration = *fj.iterations;
      if (improving) { *fj.last_improving_minimum = *fj.local_minimums_reached; }
      check_exit_condition<i_t, f_t>(fj);
      *fj.local_minimums_reached += 1;
    }
  }

  // if we're in greedy-descent mode, stop here.
  if (fj.settings->mode == fj_mode_t::GREEDY_DESCENT) return;

  // Pick the best move among the variables involved in a random violated constraint.
  if (!fj.violated_constraints.empty()) {
    cg::this_grid().sync();
    thrust::tie(best_var, best_delta, best_score) =
      best_random_mtm_move<i_t, f_t, TPB_localmin>(fj);
    if (FIRST_THREAD && best_score.valid())
      cuopt_assert(fj.pb.check_variable_within_bounds(
                     best_var, fj.incumbent_assignment[best_var] + best_delta),
                   "assignment not within bounds");
  }

  // also consider breakthrough moves
  if (*fj.best_objective < std::numeric_limits<f_t>::infinity() &&
      *fj.incumbent_objective > *fj.best_objective) {
    cg::this_grid().sync();
    auto [bm_best_var, bm_best_delta, bm_best_score] =
      best_breakthrough_move_at_local_min<i_t, f_t, TPB_localmin>(fj);
    if (bm_best_score > best_score) {
      best_score    = bm_best_score;
      best_var      = bm_best_var;
      best_delta    = bm_best_delta;
      best_movetype = 'B';
      cuopt_assert(fj.pb.check_variable_within_bounds(
                     best_var, fj.incumbent_assignment[best_var] + best_delta),
                   "assignment not within bounds");
    }
  }

  if (FIRST_THREAD) *fj.selected_var = best_var;
  cg::this_grid().sync();
  // still nothing? try sat MTM moves if we are in the feasible region
  // Attempt to find a valid move by going over MTM moves in valid constraints
  if (*fj.selected_var == std::numeric_limits<i_t>::max() &&
      *fj.incumbent_objective < std::numeric_limits<f_t>::infinity()) {
    auto [sat_best_var, sat_best_delta, sat_best_score] =
      best_sat_cstr_mtm_move<i_t, f_t, TPB_localmin>(fj);

    if (FIRST_THREAD && sat_best_score.valid())
      cuopt_assert(fj.pb.check_variable_within_bounds(
                     sat_best_var, fj.incumbent_assignment[sat_best_var] + sat_best_delta),
                   "assignment not within bounds");

    if (sat_best_score.base > 0 && sat_best_score > best_score) {
      if (FIRST_THREAD) {
        best_score = sat_best_score;
        best_var   = sat_best_var;
        best_delta = sat_best_delta;
      }
    }
  }

  if (FIRST_THREAD) {
    *fj.selected_var = best_var;
    if (best_var != std::numeric_limits<i_t>::max()) {
      cuopt_assert(fj.pb.check_variable_within_bounds(
                     best_var, fj.incumbent_assignment[best_var] + best_delta),
                   "assignment not within bounds");
      fj.jump_move_delta[best_var] = best_delta;
    }

    // force a full refresh
    *fj.full_refresh_iteration = *fj.iterations;
  }
}

#include "load_balancing.cuh"

// to save from compilation time, separate those and instantiate separately rather being part of a
// class

#define INSTANTIATE(F_TYPE)                                                           \
  template __global__ void compute_iteration_related_variables_kernel<int, F_TYPE>(   \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);                           \
  template __global__ void load_balancing_prepare_iteration<int, F_TYPE>(             \
    const __grid_constant__ typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);   \
  template __global__ void load_balancing_compute_workid_mappings<int, F_TYPE>(       \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj,                            \
    raft::device_span<int> row_size_prefix_sum,                                       \
    raft::device_span<int> var_indices,                                               \
    raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_var_idx);        \
  template __global__ void load_balancing_init_cstr_bounds_csr<int, F_TYPE>(          \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj,                            \
    raft::device_span<int> row_size_prefix_sum,                                       \
    raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_var_idx);        \
  template __global__ void load_balancing_compute_scores_binary<int, F_TYPE>(         \
    const __grid_constant__ typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);   \
  template __global__ void load_balancing_mtm_compute_candidates<int, F_TYPE>(        \
    const __grid_constant__ typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);   \
  template __global__ void load_balancing_mtm_compute_scores<int, F_TYPE>(            \
    const __grid_constant__ typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);   \
  template __global__ void load_balancing_sanity_checks<int, F_TYPE>(                 \
    const __grid_constant__ typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);   \
  template __global__ void init_lhs_and_violation<int, F_TYPE>(                       \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);                           \
  template __global__ void update_lift_moves_kernel<int, F_TYPE>(                     \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);                           \
  template __global__ void update_breakthrough_moves_kernel<int, F_TYPE>(             \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);                           \
  template __global__ void handle_local_minimum_kernel<int, F_TYPE>(                  \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);                           \
  template __global__ void update_assignment_kernel<int, F_TYPE>(                     \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj, bool IgnoreLoadBalancing); \
  template __global__ void update_changed_constraints_kernel<int, F_TYPE>(            \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);                           \
  template __global__ void update_best_solution_kernel<int, F_TYPE>(                  \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);                           \
  template __global__ void                                                            \
  compute_mtm_moves_kernel<int, F_TYPE, MTMMoveType::FJ_MTM_VIOLATED, false>(         \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj, bool);                     \
  template __global__ void                                                            \
  compute_mtm_moves_kernel<int, F_TYPE, MTMMoveType::FJ_MTM_VIOLATED, true>(          \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj, bool);                     \
  template __global__ void                                                            \
  compute_mtm_moves_kernel<int, F_TYPE, MTMMoveType::FJ_MTM_SATISFIED, false>(        \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj, bool);                     \
  template __global__ void                                                            \
  compute_mtm_moves_kernel<int, F_TYPE, MTMMoveType::FJ_MTM_SATISFIED, true>(         \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj, bool);                     \
  template __global__ void select_variable_kernel<int, F_TYPE>(                       \
    typename fj_t<int, F_TYPE>::climber_data_t::view_t fj);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
