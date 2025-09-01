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

#include <mip/mip_constants.hpp>

#include "feasibility_jump.cuh"
#include "feasibility_jump_utils.cuh"

#include <unordered_set>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct fj_cpu_t {
  fj_cpu_t(fj_t<i_t, f_t>& fj) : fj(fj) {}

  fj_t<i_t, f_t>& fj;
  fj_settings_t settings;
  typename fj_t<i_t, f_t>::climber_data_t::view_t view;
  // Host copies of device data as struct members
  std::vector<f_t> h_reverse_coefficients;
  std::vector<i_t> h_reverse_constraints;
  std::vector<i_t> h_reverse_offsets;
  std::vector<f_t> h_coefficients;
  std::vector<i_t> h_offsets;
  std::vector<i_t> h_variables;
  std::vector<f_t> h_obj_coeffs;
  std::vector<f_t> h_var_lb;
  std::vector<f_t> h_var_ub;
  std::vector<f_t> h_cstr_lb;
  std::vector<f_t> h_cstr_ub;
  std::vector<var_t> h_var_types;
  std::vector<i_t> h_is_binary_variable;

  std::vector<i_t> h_tabu_nodec_until;
  std::vector<i_t> h_tabu_noinc_until;
  std::vector<i_t> h_tabu_lastdec;
  std::vector<i_t> h_tabu_lastinc;

  std::vector<f_t> h_lhs;
  std::vector<f_t> h_lhs_sumcomp;
  std::vector<f_t> h_cstr_left_weights;
  std::vector<f_t> h_cstr_right_weights;
  std::vector<f_t> h_assignment;
  f_t h_objective_weight;
  f_t h_incumbent_objective;
  f_t h_best_objective;
  i_t iterations;
  std::unordered_set<i_t> violated_constraints;
};

template <typename i_t, typename f_t>
fj_staged_score_t compute_score(fj_cpu_t<i_t, f_t>& fj_cpu, i_t var_idx, f_t delta)
{
  f_t obj_diff = fj_cpu.h_obj_coeffs[var_idx] * delta;

  cuopt_assert(isfinite(delta), "");

  cuopt_assert(var_idx < fj_cpu.view.pb.n_variables, "variable index out of bounds");

  f_t base_feas_sum               = 0;
  f_t bonus_robust_sum            = 0;
  auto [offset_begin, offset_end] = fj_cpu.view.pb.reverse_range_for_var(var_idx);
  for (i_t i = offset_begin; i < offset_end; i++) {
    auto cstr_idx   = fj_cpu.h_reverse_constraints[i];
    auto cstr_coeff = fj_cpu.h_reverse_coefficients[i];

    f_t c_lb = fj_cpu.h_cstr_lb[cstr_idx];
    f_t c_ub = fj_cpu.h_cstr_ub[cstr_idx];
    cuopt_assert(c_lb <= c_ub, "invalid bounds");

    auto [cstr_base_feas, cstr_bonus_robust] = feas_score_constraint<i_t, f_t>(
      fj_cpu.view, var_idx, delta, cstr_idx, cstr_coeff, c_lb, c_ub, fj_cpu.h_lhs[cstr_idx]);

    base_feas_sum += cstr_base_feas;
    bonus_robust_sum += cstr_bonus_robust;
  }

  f_t base_obj = 0;
  if (obj_diff < 0)  // improving move wrt objective
    base_obj = fj_cpu.h_objective_weight;
  else if (obj_diff > 0)
    base_obj = -fj_cpu.h_objective_weight;

  f_t bonus_breakthrough = 0;

  bool old_obj_better = fj_cpu.h_objective_weight < fj_cpu.h_best_objective;
  bool new_obj_better = fj_cpu.h_objective_weight + obj_diff < fj_cpu.h_best_objective;
  if (!old_obj_better && new_obj_better)
    bonus_breakthrough += fj_cpu.h_objective_weight;
  else if (old_obj_better && !new_obj_better) {
    bonus_breakthrough -= fj_cpu.h_objective_weight;
  }

  fj_staged_score_t score;
  score.base  = round(base_obj + base_feas_sum);
  score.bonus = round(bonus_breakthrough + bonus_robust_sum);
  return score;
}

template <typename i_t, typename f_t>
void smooth_weights(fj_cpu_t<i_t, f_t>& fj_cpu)
{
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.view.pb.n_constraints; cstr_idx++) {
    // consider only satisfied constraints
    if (fj_cpu.violated_constraints.count(cstr_idx)) continue;

    f_t weight_l = max((f_t)0, fj_cpu.h_cstr_left_weights[cstr_idx] - 1);
    f_t weight_r = max((f_t)0, fj_cpu.h_cstr_right_weights[cstr_idx] - 1);

    fj_cpu.h_cstr_left_weights[cstr_idx]  = weight_l;
    fj_cpu.h_cstr_right_weights[cstr_idx] = weight_r;
  }

  if (fj_cpu.h_objective_weight > 0 && fj_cpu.h_incumbent_objective >= fj_cpu.h_best_objective) {
    fj_cpu.h_objective_weight = max((f_t)0, fj_cpu.h_objective_weight - 1);
  }
}

template <typename i_t, typename f_t>
void update_weights(fj_cpu_t<i_t, f_t>& fj_cpu)
{
  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);
  bool smoothing = rng.next_float() <= fj_cpu.settings.parameters.weight_smoothing_probability;

  if (smoothing && false) return smooth_weights<i_t, f_t>(fj_cpu);

  for (auto cstr_idx : fj_cpu.violated_constraints) {
    f_t curr_incumbent_lhs = fj_cpu.h_lhs[cstr_idx];
    f_t curr_lower_excess  = fj_cpu.view.lower_excess_score(cstr_idx, curr_incumbent_lhs);
    f_t curr_upper_excess  = fj_cpu.view.upper_excess_score(cstr_idx, curr_incumbent_lhs);
    f_t curr_excess_score  = curr_lower_excess + curr_upper_excess;

    f_t old_weight;
    if (curr_lower_excess < 0.) {
      old_weight = fj_cpu.h_cstr_left_weights[cstr_idx];
    } else {
      old_weight = fj_cpu.h_cstr_right_weights[cstr_idx];
    }

    cuopt_assert(curr_excess_score < 0, "constraint not violated");

    i_t int_delta = fj_cpu.fj.weight_update_increment;
    f_t delta     = int_delta;

    f_t new_weight = old_weight + delta;
    new_weight     = round(new_weight);

    if (curr_lower_excess < 0.) {
      fj_cpu.h_cstr_left_weights[cstr_idx] = new_weight;
    } else {
      fj_cpu.h_cstr_right_weights[cstr_idx] = new_weight;
    }
  }

  if (fj_cpu.violated_constraints.empty()) { fj_cpu.h_objective_weight += 1; }
}

template <typename i_t, typename f_t>
void apply_move(fj_cpu_t<i_t, f_t>& fj_cpu, i_t var_idx, f_t delta)
{
  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  // printf("    applying move: %d, %g, unsat %zu\n", var_idx, delta,
  // fj_cpu.violated_constraints.size());

  cuopt_assert(var_idx < fj_cpu.view.pb.n_variables, "variable index out of bounds");
  // Update the LHSs of all involved constraints.
  auto [offset_begin, offset_end] = fj_cpu.view.pb.reverse_range_for_var(var_idx);

  for (auto i = offset_begin; i < offset_end; i++) {
    cuopt_assert(i < (i_t)fj_cpu.h_reverse_constraints.size(), "");

    auto cstr_idx   = fj_cpu.h_reverse_constraints[i];
    auto cstr_coeff = fj_cpu.h_reverse_coefficients[i];

    f_t old_lhs        = fj_cpu.h_lhs[cstr_idx];
    f_t new_lhs        = old_lhs + cstr_coeff * delta;
    f_t old_cost       = fj_cpu.view.excess_score(cstr_idx, old_lhs);
    f_t new_cost       = fj_cpu.view.excess_score(cstr_idx, new_lhs);
    f_t cstr_tolerance = fj_cpu.view.get_corrected_tolerance(cstr_idx);

    if (new_cost < -cstr_tolerance && !fj_cpu.violated_constraints.count(cstr_idx)) {
      fj_cpu.violated_constraints.insert(cstr_idx);
    } else if (!(new_cost < -cstr_tolerance) && fj_cpu.violated_constraints.count(cstr_idx)) {
      fj_cpu.violated_constraints.erase(cstr_idx);
    }

    cuopt_assert(isfinite(delta), "delta should be finite");
    // Kahan compensated summation
    f_t y                          = cstr_coeff * delta - fj_cpu.h_lhs_sumcomp[cstr_idx];
    f_t t                          = old_lhs + y;
    fj_cpu.h_lhs_sumcomp[cstr_idx] = (t - old_lhs) - y;
    fj_cpu.h_lhs[cstr_idx]         = t;
    cuopt_assert(isfinite(fj_cpu.h_lhs[cstr_idx]), "assignment should be finite");
  }
  // update the assignment and objective proper
  f_t new_val                  = fj_cpu.h_assignment[var_idx] + delta;
  fj_cpu.h_assignment[var_idx] = new_val;

  cuopt_assert(fj_cpu.view.pb.check_variable_within_bounds(var_idx, new_val),
               "assignment not within bounds");
  cuopt_assert(isfinite(new_val), "assignment is not finite");

  fj_cpu.h_incumbent_objective += fj_cpu.h_obj_coeffs[var_idx] * delta;
  if (fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective &&
      fj_cpu.violated_constraints.empty()) {
    fj_cpu.h_best_objective = fj_cpu.h_incumbent_objective;
    printf("CPU: new best objective: %g\n", fj_cpu.h_best_objective);
  }

  i_t tabu_tenure = fj_cpu.settings.parameters.tabu_tenure_min +
                    rng.next_u32() % (fj_cpu.settings.parameters.tabu_tenure_max -
                                      fj_cpu.settings.parameters.tabu_tenure_min);
  if (delta > 0) {
    fj_cpu.h_tabu_lastinc[var_idx]     = fj_cpu.iterations;
    fj_cpu.h_tabu_nodec_until[var_idx] = fj_cpu.iterations + tabu_tenure;
  } else if (delta < 0) {
    fj_cpu.h_tabu_lastdec[var_idx]     = fj_cpu.iterations;
    fj_cpu.h_tabu_noinc_until[var_idx] = fj_cpu.iterations + tabu_tenure;
  }
}

template <typename i_t, typename f_t, MTMMoveType move_type>
thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move(fj_cpu_t<i_t, f_t>& fj_cpu,
                                                          const std::vector<i_t>& target_cstrs)
{
  auto& problem = *fj_cpu.fj.pb_ptr;

  // Maps a candidate move to its score
  std::set<fj_move_t> candidate_moves;
  fj_move_t best_move          = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::zero();

  for (size_t cstr_idx : target_cstrs) {
    cuopt_assert(cstr_idx < fj_cpu.h_cstr_lb.size(), "cstr_idx is out of bounds");
    auto [offset_begin, offset_end] = fj_cpu.view.pb.range_for_constraint(cstr_idx);
    for (auto i = offset_begin; i < offset_end; i++) {
      auto var_idx = fj_cpu.h_variables[i];
      // // Special case for binary variables
      // if (fj_cpu.h_is_binary_variable[var_idx])
      // {
      //     f_t val = fj_cpu.h_assignment[var_idx];
      //     f_t new_val = 1 - val;
      //     auto move = fj_move_t{var_idx, new_val};
      //     if (!candidate_moves.count(move))
      //     {
      //         candidate_moves.insert(move);
      //         fj_staged_score_t score = compute_score<i_t, f_t>(fj_cpu, var_idx, move.value);
      //         if (best_score < score)
      //         {
      //             best_score = score;
      //             best_move = move;
      //         }
      //     }
      //     continue;
      // }

      auto cstr_coeff = fj_cpu.h_coefficients[i];
      f_t val         = fj_cpu.h_assignment[var_idx];
      f_t new_val     = val;

      f_t c_lb = fj_cpu.h_cstr_lb[cstr_idx];
      f_t c_ub = fj_cpu.h_cstr_ub[cstr_idx];
      auto [delta, sign, slack, cstr_tolerance] =
        get_mtm_for_constraint<i_t, f_t, MTMMoveType::FJ_MTM_VIOLATED>(
          fj_cpu.view, var_idx, cstr_idx, cstr_coeff, c_lb, c_ub);
      if (fj_cpu.view.pb.is_integer_var(var_idx)) {
        new_val = cstr_coeff * sign > 0
                    ? floor(val + delta + fj_cpu.view.pb.tolerances.integrality_tolerance)
                    : ceil(val + delta - fj_cpu.view.pb.tolerances.integrality_tolerance);
      } else {
        new_val = val + delta;
      }
      // fallback
      if (new_val < fj_cpu.h_var_lb[var_idx] || new_val > fj_cpu.h_var_ub[var_idx]) {
        new_val = cstr_coeff * sign > 0 ? fj_cpu.h_var_lb[var_idx] : fj_cpu.h_var_ub[var_idx];
      }
      cuopt_assert(isfinite(new_val), "new_val is not finite");
      cuopt_assert(new_val >= fj_cpu.h_var_lb[var_idx], "new_val is not greater than lower bound");
      cuopt_assert(new_val <= fj_cpu.h_var_ub[var_idx], "new_val is not less than upper bound");
      delta = new_val - val;
      if (delta < 0 && fj_cpu.iterations < fj_cpu.h_tabu_nodec_until[var_idx] ||
          delta > 0 && fj_cpu.iterations < fj_cpu.h_tabu_noinc_until[var_idx])
        continue;
      if (fabs(delta) < fj_cpu.view.get_corrected_tolerance(cstr_idx)) continue;

      // Check if we already have a move for this variable
      auto move = fj_move_t{var_idx, new_val};
      cuopt_assert(move.var_idx < fj_cpu.h_assignment.size(), "move.var_idx is out of bounds");
      cuopt_assert(move.var_idx >= 0, "move.var_idx is not positive");

      if (candidate_moves.count(move)) continue;

      candidate_moves.insert(move);
      fj_staged_score_t score = compute_score<i_t, f_t>(fj_cpu, var_idx, delta);
      if (best_score < score) {
        best_score = score;
        best_move  = move;
      }
    }
  }

  // printf("best_move: %d, %g, score %d, subscore %d\n", best_move.var_idx, best_move.value,
  // (int)best_score.base, (int)best_score.bonus);
  return thrust::make_tuple(best_move, best_score);
}

template <typename i_t, typename f_t>
thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move_viol(fj_cpu_t<i_t, f_t>& fj_cpu,
                                                               i_t sample_size = 100)
{
  std::vector<i_t> sampled_cstrs;
  sampled_cstrs.reserve(sample_size);
  std::sample(fj_cpu.violated_constraints.begin(),
              fj_cpu.violated_constraints.end(),
              std::back_inserter(sampled_cstrs),
              sample_size,
              std::mt19937(fj_cpu.settings.seed + fj_cpu.iterations));

  return find_mtm_move<i_t, f_t, MTMMoveType::FJ_MTM_VIOLATED>(fj_cpu, sampled_cstrs);
}

template <typename i_t, typename f_t>
void init_lhs(fj_cpu_t<i_t, f_t>& fj_cpu)
{
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.view.pb.n_constraints; ++cstr_idx) {
    auto [offset_begin, offset_end] = fj_cpu.view.pb.range_for_constraint(cstr_idx);
    f_t lhs                         = 0;
    for (i_t i = offset_begin; i < offset_end; ++i) {
      lhs += fj_cpu.h_coefficients[i] * fj_cpu.h_assignment[fj_cpu.h_variables[i]];
    }
    fj_cpu.h_lhs[cstr_idx] = lhs;

    f_t cstr_tolerance = fj_cpu.view.get_corrected_tolerance(cstr_idx);
    f_t new_cost       = fj_cpu.view.excess_score(cstr_idx, lhs);
    if (new_cost < -cstr_tolerance) { fj_cpu.violated_constraints.insert(cstr_idx); }
  }

  // compute incumbent objective
  fj_cpu.h_incumbent_objective = thrust::inner_product(
    fj_cpu.h_assignment.begin(), fj_cpu.h_assignment.end(), fj_cpu.h_obj_coeffs.begin(), 0.);
  fj_cpu.h_best_objective = +std::numeric_limits<f_t>::infinity();
}

template <typename i_t, typename f_t>
void init_fj_cpu(fj_cpu_t<i_t, f_t>& fj_cpu, solution_t<i_t, f_t>& solution)
{
  auto& fj        = fj_cpu.fj;
  auto& problem   = *fj.pb_ptr;
  auto handle_ptr = fj_cpu.fj.handle_ptr;

  // build a cpu-based fj_view_t
  fj_cpu.view = fj.climbers[0]->view();

  // Get host copies of device data
  fj_cpu.h_reverse_coefficients =
    cuopt::host_copy(problem.reverse_coefficients, handle_ptr->get_stream());
  fj_cpu.h_reverse_constraints =
    cuopt::host_copy(problem.reverse_constraints, handle_ptr->get_stream());
  fj_cpu.h_reverse_offsets = cuopt::host_copy(problem.reverse_offsets, handle_ptr->get_stream());
  fj_cpu.h_coefficients    = cuopt::host_copy(problem.coefficients, handle_ptr->get_stream());
  fj_cpu.h_offsets         = cuopt::host_copy(problem.offsets, handle_ptr->get_stream());
  fj_cpu.h_variables       = cuopt::host_copy(problem.variables, handle_ptr->get_stream());
  fj_cpu.h_obj_coeffs = cuopt::host_copy(problem.objective_coefficients, handle_ptr->get_stream());
  fj_cpu.h_var_lb     = cuopt::host_copy(problem.variable_lower_bounds, handle_ptr->get_stream());
  fj_cpu.h_var_ub     = cuopt::host_copy(problem.variable_upper_bounds, handle_ptr->get_stream());
  fj_cpu.h_cstr_lb    = cuopt::host_copy(problem.constraint_lower_bounds, handle_ptr->get_stream());
  fj_cpu.h_cstr_ub    = cuopt::host_copy(problem.constraint_upper_bounds, handle_ptr->get_stream());
  fj_cpu.h_var_types  = cuopt::host_copy(problem.variable_types, handle_ptr->get_stream());
  fj_cpu.h_is_binary_variable =
    cuopt::host_copy(problem.is_binary_variable, handle_ptr->get_stream());

  fj_cpu.h_cstr_left_weights  = cuopt::host_copy(fj.cstr_left_weights, handle_ptr->get_stream());
  fj_cpu.h_cstr_right_weights = cuopt::host_copy(fj.cstr_right_weights, handle_ptr->get_stream());
  fj_cpu.h_objective_weight   = fj.objective_weight.value(handle_ptr->get_stream());
  fj_cpu.h_assignment         = solution.get_host_assignment();
  fj_cpu.h_lhs.resize(fj.pb_ptr->n_constraints);
  fj_cpu.h_lhs_sumcomp.resize(fj.pb_ptr->n_constraints, 0);
  fj_cpu.h_tabu_nodec_until.resize(fj.pb_ptr->n_variables, 0);
  fj_cpu.h_tabu_noinc_until.resize(fj.pb_ptr->n_variables, 0);
  fj_cpu.h_tabu_lastdec.resize(fj.pb_ptr->n_variables, 0);
  fj_cpu.h_tabu_lastinc.resize(fj.pb_ptr->n_variables, 0);
  fj_cpu.iterations = 0;

  // set pointers to host copies
  // technically not 'device_span's but raft doesn't have a universal span.
  // cuda::std::span?
  fj_cpu.view.cstr_left_weights =
    raft::device_span<f_t>(fj_cpu.h_cstr_left_weights.data(), fj_cpu.h_cstr_left_weights.size());
  fj_cpu.view.cstr_right_weights =
    raft::device_span<f_t>(fj_cpu.h_cstr_right_weights.data(), fj_cpu.h_cstr_right_weights.size());
  fj_cpu.view.objective_weight = &fj_cpu.h_objective_weight;
  fj_cpu.view.incumbent_assignment =
    raft::device_span<f_t>(fj_cpu.h_assignment.data(), fj_cpu.h_assignment.size());
  fj_cpu.view.incumbent_lhs = raft::device_span<f_t>(fj_cpu.h_lhs.data(), fj_cpu.h_lhs.size());
  fj_cpu.view.tabu_nodec_until =
    raft::device_span<i_t>(fj_cpu.h_tabu_nodec_until.data(), fj_cpu.h_tabu_nodec_until.size());
  fj_cpu.view.tabu_noinc_until =
    raft::device_span<i_t>(fj_cpu.h_tabu_noinc_until.data(), fj_cpu.h_tabu_noinc_until.size());
  fj_cpu.view.tabu_lastdec =
    raft::device_span<i_t>(fj_cpu.h_tabu_lastdec.data(), fj_cpu.h_tabu_lastdec.size());
  fj_cpu.view.tabu_lastinc =
    raft::device_span<i_t>(fj_cpu.h_tabu_lastinc.data(), fj_cpu.h_tabu_lastinc.size());

  fj_cpu.view.settings = &fj.settings;
  fj_cpu.view.pb.constraint_lower_bounds =
    raft::device_span<f_t>(fj_cpu.h_cstr_lb.data(), fj_cpu.h_cstr_lb.size());
  fj_cpu.view.pb.constraint_upper_bounds =
    raft::device_span<f_t>(fj_cpu.h_cstr_ub.data(), fj_cpu.h_cstr_ub.size());
  fj_cpu.view.pb.variable_lower_bounds =
    raft::device_span<f_t>(fj_cpu.h_var_lb.data(), fj_cpu.h_var_lb.size());
  fj_cpu.view.pb.variable_upper_bounds =
    raft::device_span<f_t>(fj_cpu.h_var_ub.data(), fj_cpu.h_var_ub.size());
  fj_cpu.view.pb.variable_types =
    raft::device_span<var_t>(fj_cpu.h_var_types.data(), fj_cpu.h_var_types.size());
  fj_cpu.view.pb.is_binary_variable =
    raft::device_span<i_t>(fj_cpu.h_is_binary_variable.data(), fj_cpu.h_is_binary_variable.size());
  fj_cpu.view.pb.coefficients =
    raft::device_span<f_t>(fj_cpu.h_coefficients.data(), fj_cpu.h_coefficients.size());
  fj_cpu.view.pb.offsets = raft::device_span<i_t>(fj_cpu.h_offsets.data(), fj_cpu.h_offsets.size());
  fj_cpu.view.pb.variables =
    raft::device_span<i_t>(fj_cpu.h_variables.data(), fj_cpu.h_variables.size());
  fj_cpu.view.pb.reverse_coefficients = raft::device_span<f_t>(
    fj_cpu.h_reverse_coefficients.data(), fj_cpu.h_reverse_coefficients.size());
  fj_cpu.view.pb.reverse_constraints = raft::device_span<i_t>(fj_cpu.h_reverse_constraints.data(),
                                                              fj_cpu.h_reverse_constraints.size());
  fj_cpu.view.pb.reverse_offsets =
    raft::device_span<i_t>(fj_cpu.h_reverse_offsets.data(), fj_cpu.h_reverse_offsets.size());

  // scratch thread: fill all weights with 1
  for (i_t cstr_idx = 0; cstr_idx < fj.pb_ptr->n_constraints; ++cstr_idx) {
    fj_cpu.h_cstr_left_weights[cstr_idx]  = 1;
    fj_cpu.h_cstr_right_weights[cstr_idx] = 1;
  }
  fj_cpu.h_objective_weight = 0;

  init_lhs(fj_cpu);
}

template <typename i_t, typename f_t>
i_t fj_t<i_t, f_t>::cpu_solve(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range scope("fj_cpu");

  auto& fj = *this;
  fj_cpu_t fj_cpu{fj};

  // Initialize fj_cpu with all the data
  init_fj_cpu(fj_cpu, solution);

  fprintf(stderr, "running bespoke CPU FJ!\n");

  i_t local_mins = 0;
  while (fj_cpu.iterations < 10000) {
    auto [move, score] = find_mtm_move_viol(fj_cpu, 1500 * 1000);
    if (score > fj_staged_score_t::zero()) {
      apply_move(fj_cpu, move.var_idx, move.value - fj_cpu.h_assignment[move.var_idx]);
    } else {
      // printf("local min\n");
      update_weights(fj_cpu);
      auto [move, score] =
        find_mtm_move_viol(fj_cpu, 1);  // pick a single random violated constraint
      i_t var_idx = move.var_idx >= 0 ? move.var_idx : 0;
      f_t delta   = move.var_idx >= 0 ? move.value - fj_cpu.h_assignment[move.var_idx] : 0;
      apply_move(fj_cpu, var_idx, delta);
      ++local_mins;
    }
    if (fj_cpu.iterations % 100 == 0) {
      printf("iteration: %d, local mins: %d, best_objective: %g, viol: %zu\n",
             fj_cpu.iterations,
             local_mins,
             fj_cpu.h_best_objective,
             fj_cpu.violated_constraints.size());
    }
    fj_cpu.iterations++;
  }

  return 0;
}

#if MIP_INSTANTIATE_FLOAT
template class fj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class fj_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
