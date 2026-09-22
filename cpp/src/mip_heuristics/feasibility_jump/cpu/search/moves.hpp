/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../internal.hpp"
#include "score.hpp"

namespace cuopt::mathematical_optimization::mip {

struct two_opt_move_t {
  fj_move_t first{-1, 0};
  fj_move_t second{-1, 0};
  fj_staged_score_t score{fj_staged_score_t::invalid()};
  double objective_delta{0};
  int age{std::numeric_limits<int>::max()};

  bool operator>(const two_opt_move_t& other) const
  {
    if (score != other.score) return score > other.score;
    if (objective_delta != other.objective_delta) return objective_delta < other.objective_delta;
    if (age != other.age) return age < other.age;
    if (first.var_idx != other.first.var_idx) return first.var_idx < other.first.var_idx;
    return second.var_idx < other.second.var_idx;
  }
};

template <typename i_t, typename f_t>
static fj_staged_score_t two_opt_compute_pair_score(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t first, f_t first_delta, i_t second, f_t second_delta)
{
  auto& row_deltas = fj_cpu.two_opt_row_deltas;
  row_deltas.clear();
  const fj_move_t endpoints[2] = {{first, first_delta}, {second, second_delta}};
  for (const auto& [var_idx, delta] : endpoints) {
    const auto [offset_begin, offset_end] = fj_cpu.range_for_variable(var_idx);
    fj_cpu.nnz_processed_window += offset_end - offset_begin;
    for (i_t i = offset_begin; i < offset_end; ++i) {
      const i_t cstr_idx = fj_cpu.h_reverse_constraints[i];
      const f_t coeff    = fj_cpu.h_reverse_coefficients[i];
      row_deltas.emplace_back(cstr_idx, coeff * delta);
    }
  }
  // Brings the entries of a shared row next to each other
  std::sort(row_deltas.begin(), row_deltas.end());

  f_t base_feas_sum    = 0;
  f_t bonus_robust_sum = 0;
  for (size_t pos = 0; pos < row_deltas.size();) {
    const i_t cstr_idx = row_deltas[pos].first;
    f_t lhs_delta      = 0;
    do {
      lhs_delta += row_deltas[pos++].second;
    } while (pos < row_deltas.size() && row_deltas[pos].first == cstr_idx);

    // The coefficients are already folded into lhs_delta, hence the unit coefficient
    auto [cstr_base_feas, cstr_bonus_robust] =
      feas_score_constraint<i_t, f_t>(fj_cpu,
                                      lhs_delta,
                                      1,
                                      fj_cpu.row_state()[cstr_idx].slack,
                                      fj_cpu.row_state()[cstr_idx].weight);
    base_feas_sum += cstr_base_feas;
    bonus_robust_sum += cstr_bonus_robust;
  }

  const f_t obj_diff = fj_cpu.problem->h_obj_coeffs[first] * first_delta +
                       fj_cpu.problem->h_obj_coeffs[second] * second_delta;
  f_t base_obj = 0;
  if (obj_diff < 0)
    base_obj = fj_cpu.h_objective_weight;
  else if (obj_diff > 0)
    base_obj = -fj_cpu.h_objective_weight;

  f_t bonus_breakthrough = 0;
  bool old_obj_better    = fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective;
  bool new_obj_better    = fj_cpu.h_incumbent_objective + obj_diff < fj_cpu.h_best_objective;
  if (!old_obj_better && new_obj_better)
    bonus_breakthrough += fj_cpu.h_objective_weight;
  else if (old_obj_better && !new_obj_better)
    bonus_breakthrough -= fj_cpu.h_objective_weight;

  fj_staged_score_t score;
  score.base  = std::round(base_obj + base_feas_sum);
  score.bonus = std::round(bonus_breakthrough + bonus_robust_sum);
  return score;
}

template <typename i_t, typename f_t, MTMMoveType move_type>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, const std::vector<i_t>& target_cstrs, bool localmin = false)
{
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move");

  cuopt::pcgenerator_t rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  fj_move_t best_move          = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::invalid();
  double best_objective_delta  = std::numeric_limits<double>::infinity();
  auto improves_best           = [&](fj_staged_score_t score, i_t var, f_t delta) {
    if (score > best_score) return true;
    // Magnitude ordering paid on the dedicated objective trajectories but displaced useful neutral
    // moves on feasibility-biased lanes. Keep it behind the same high-pressure persona gate.
    if (fj_cpu.h_objective_weight < f_t{16} || !(score == best_score)) return false;
    const double objective_delta = static_cast<double>(fj_cpu.problem->h_obj_coeffs[var]) * delta;
    return objective_delta < best_objective_delta;
  };
  auto store_best = [&](fj_staged_score_t score, i_t var, f_t delta) {
    best_score           = score;
    best_move            = fj_move_t{var, delta};
    best_objective_delta = static_cast<double>(fj_cpu.problem->h_obj_coeffs[var]) * delta;
  };

  ++fj_cpu.n_mtm_calls;

  // Each row contributes at most its share of the sampling budget. The gate below sits inside the
  // walk, so an uncapped wide row is walked in full whatever the budget says.
  const i_t per_row_cap =
    std::max<i_t>(1, fj_cpu.nnz_samples / std::max<i_t>(1, (i_t)target_cstrs.size()));

  i_t entries = 0;
  for (size_t cstr_idx : target_cstrs) {
    auto [offset_begin, offset_end] = fj_cpu.range_for_row((i_t)cstr_idx);
    const i_t width                 = offset_end - offset_begin;
    entries += std::min(width, per_row_cap);
    fj_cpu.mtm_entries_capped += (int64_t)std::max<i_t>(0, width - per_row_cap);
  }
  fj_cpu.mtm_row_entries += (int64_t)entries;

  // The exact sum over the candidate variables costs one random offset read each to set a single
  // sampling rate. The mean reverse degree estimates it in constant time.
  const f_t mean_reverse_degree =
    (f_t)fj_cpu.h_coefficients.size() / (f_t)std::max<i_t>(1, fj_cpu.problem->n_variables);
  const f_t nnz_sum = (f_t)entries * mean_reverse_degree;

  f_t nnz_pick_probability = 1;
  if (nnz_sum > (f_t)fj_cpu.nnz_samples) nnz_pick_probability = (f_t)fj_cpu.nnz_samples / nnz_sum;

  for (size_t cstr_idx : target_cstrs) {
    cuopt_assert((i_t)cstr_idx < fj_cpu.n_rows, "cstr_idx is out of bounds");
    auto [offset_begin, offset_end] = fj_cpu.range_for_row((i_t)cstr_idx);
    const i_t width                 = offset_end - offset_begin;
    const i_t visit                 = std::min(width, per_row_cap);
    const i_t start =
      visit == width ? offset_begin : offset_begin + (i_t)(rng.next_u32() % (uint32_t)width);
    for (i_t q = 0, i = start; q < visit; ++q, i = (i + 1 == offset_end ? offset_begin : i + 1)) {
      const i_t var_idx = fj_cpu.h_variables[i];
      // early cached check
      cuopt_assert(fj_cpu.cached_mtm_moves_version[i] <= fj_cpu.h_cstr_version[cstr_idx],
                   "cached move newer than its constraint");
      if (auto& cached_move = fj_cpu.cached_mtm_moves[i];
          cached_move.first != 0 &&
          fj_cpu.cached_mtm_moves_version[i] == fj_cpu.h_cstr_version[cstr_idx]) {
        if (improves_best(cached_move.second, var_idx, cached_move.first)) {
          if (check_variable_within_bounds<i_t, f_t>(
                fj_cpu, var_idx, fj_cpu.h_assignment[var_idx] + cached_move.first)) {
            store_best(cached_move.second, var_idx, cached_move.first);
          }
          // cuopt_assert(fj_cpu.check_variable_within_bounds(var_idx,
          // fj_cpu.h_assignment[var_idx] + cached_move.first), "best move is not within bounds");
        }
        fj_cpu.hit_count++;
        continue;
      }

      // random chance to skip this nnz if there are many to consider
      if (nnz_pick_probability < 1)
        if (rng.next_float() > nnz_pick_probability) continue;

      f_t val     = fj_cpu.h_assignment[var_idx];
      f_t new_val = val;
      f_t delta   = 0;

      // Special case for binary variables
      if (fj_cpu.h_is_binary_variable[var_idx]) {
        if (fj_cpu.flip_move_stamp[var_idx] == fj_cpu.flip_move_epoch) continue;
        fj_cpu.flip_move_stamp[var_idx] = fj_cpu.flip_move_epoch;
        new_val                         = 1 - val;
      } else {
        const f_t cstr_coeff = fj_cpu.h_coefficients[i];

        const f_t delta = get_mtm_for_constraint<i_t, f_t, move_type>(
          cstr_coeff, fj_cpu.row_state()[cstr_idx].slack, fj_cpu.row_tolerance);
        if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) {
          // The sign the two-sided form applied here is already folded into the coefficient.
          new_val = cstr_coeff > 0
                      ? std::floor(val + delta + fj_cpu.problem->tolerances.integrality_tolerance)
                      : std::ceil(val + delta - fj_cpu.problem->tolerances.integrality_tolerance);
        } else {
          new_val = val + delta;
        }
        // fallback
        if (new_val < get_lower(fj_cpu.h_var_bounds[var_idx].get()) ||
            new_val > get_upper(fj_cpu.h_var_bounds[var_idx].get())) {
          new_val = cstr_coeff > 0 ? get_lower(fj_cpu.h_var_bounds[var_idx].get())
                                   : get_upper(fj_cpu.h_var_bounds[var_idx].get());
        }
      }
      if (!std::isfinite(new_val)) continue;
      cuopt_assert(check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, new_val),
                   "new_val is not within bounds");
      delta = new_val - val;
      // more permissive tabu in the case of local minima
      if (tabu_check<i_t, f_t>(fj_cpu, var_idx, delta, localmin)) continue;
      if (std::fabs(delta) < fj_cpu.row_tolerance) continue;

      auto move = fj_move_t{var_idx, delta};
      cuopt_assert(move.var_idx < fj_cpu.h_assignment.size(), "move.var_idx is out of bounds");
      cuopt_assert(move.var_idx >= 0, "move.var_idx is not positive");

      auto [score, infeasibility] = compute_score<i_t, f_t>(fj_cpu, var_idx, delta);
      fj_cpu.miss_count++;
      // reject this move if it would increase the target variable to a numerically unstable value
      if (!fj_cpu.move_numerically_stable(val, new_val, infeasibility, fj_cpu.total_violations))
        continue;
      fj_cpu.cached_mtm_moves[i]         = std::make_pair(delta, score);
      fj_cpu.cached_mtm_moves_version[i] = fj_cpu.h_cstr_version[cstr_idx];
      if (improves_best(score, move.var_idx, move.value))
        store_best(score, move.var_idx, move.value);
    }
  }

  // also consider BM moves if we have found a feasible solution at least once
  if (move_type == MTMMoveType::FJ_MTM_VIOLATED &&
      fj_cpu.h_best_objective < std::numeric_limits<f_t>::infinity() &&
      fj_cpu.h_incumbent_objective >=
        fj_cpu.h_best_objective + fj_cpu.settings.parameters.breakthrough_move_epsilon) {
    for (auto var_idx : fj_cpu.problem->h_objective_vars) {
      f_t old_val = fj_cpu.h_assignment[var_idx];
      f_t new_val = fj_cpu.breakthrough_value(var_idx);

      if (fj_cpu.problem->integer_equal(new_val, old_val) || !std::isfinite(new_val)) continue;

      f_t delta = new_val - old_val;

      // Check if we already have a move for this variable
      auto move = fj_move_t{var_idx, delta};
      cuopt_assert(move.var_idx < fj_cpu.h_assignment.size(), "move.var_idx is out of bounds");
      cuopt_assert(move.var_idx >= 0, "move.var_idx is not positive");

      if (tabu_check<i_t, f_t>(fj_cpu, var_idx, delta)) continue;

      auto [score, infeasibility] = compute_score<i_t, f_t>(fj_cpu, var_idx, delta);

      cuopt_assert(check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, new_val), "");
      cuopt_assert(std::isfinite(delta), "");

      if (fj_cpu.move_numerically_stable(
            old_val, new_val, infeasibility, fj_cpu.total_violations)) {
        if (improves_best(score, move.var_idx, move.value))
          store_best(score, move.var_idx, move.value);
      }
    }
  }

  return thrust::make_tuple(best_move, best_score);
}

template <typename i_t>
void sample_with_replacement(const host_contiguous_set_t<i_t>& pool,
                             i_t sample_size,
                             uint64_t seed,
                             std::vector<i_t>& out)
{
  cuopt_assert(sample_size > 0, "invalid sample size");
  out.clear();
  const i_t pool_size = pool.size();
  if (pool_size == 0) { return; }
  if (pool_size <= sample_size) {
    out.assign(pool.begin(), pool.end());
    return;
  }
  out.reserve(sample_size);
  cuopt::pcgenerator_t rng(seed);
  for (i_t i = 0; i < sample_size; ++i) {
    out.push_back(pool.contents[rng.next_u32() % (uint32_t)pool_size]);
  }
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move_viol(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t sample_size = 100, bool localmin = false)
{
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move_viol");

  std::vector<i_t> sampled_cstrs;
  sample_with_replacement(fj_cpu.violated_constraints,
                          sample_size,
                          fj_cpu.settings.seed + fj_cpu.iterations,
                          sampled_cstrs);

  return find_mtm_move<i_t, f_t, MTMMoveType::FJ_MTM_VIOLATED>(fj_cpu, sampled_cstrs, localmin);
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move_sat(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t sample_size = 100, bool localmin = false)
{
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move_sat");

  std::vector<i_t> sampled_cstrs;
  sample_with_replacement(fj_cpu.satisfied_constraints,
                          sample_size,
                          fj_cpu.settings.seed + fj_cpu.iterations,
                          sampled_cstrs);

  return find_mtm_move<i_t, f_t, MTMMoveType::FJ_MTM_SATISFIED>(fj_cpu, sampled_cstrs, localmin);
}

template <typename i_t, typename f_t>
bool paired_flip_keeps_feasible(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var1, f_t delta1, i_t var2, f_t delta2)
{
  const auto range1 = fj_cpu.range_for_variable(var1);
  const auto range2 = fj_cpu.range_for_variable(var2);
  i_t i = range1.first, ie = range1.second;
  i_t j = range2.first, je = range2.second;

  while (i < ie || j < je) {
    const i_t r1 = i < ie ? (i_t)fj_cpu.h_reverse_constraints[i] : std::numeric_limits<i_t>::max();
    const i_t r2 = j < je ? (i_t)fj_cpu.h_reverse_constraints[j] : std::numeric_limits<i_t>::max();
    const i_t r  = r1 < r2 ? r1 : r2;

    f_t change = 0;
    if (r1 == r) {
      change += (f_t)fj_cpu.h_reverse_coefficients[i] * delta1;
      ++i;
    }
    if (r2 == r) {
      change += (f_t)fj_cpu.h_reverse_coefficients[j] * delta2;
      ++j;
    }

    const f_t new_slack = (fj_cpu.row_state()[r].slack + fj_cpu.h_slack_sumcomp[r]) - change;
    if (new_slack < -fj_cpu.row_tolerance) return false;
  }
  return true;
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_move_t, fj_staged_score_t> find_lift_2opt_move(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::find_lift_2opt_move");
  cuopt_assert(fj_cpu.violated_constraints.empty(), "lift moves require a feasible incumbent");

  fj_move_t best_first         = fj_move_t{-1, 0};
  fj_move_t best_second        = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::zero();
  f_t best_improvement         = 0;

  const i_t n_obj = (i_t)fj_cpu.problem->h_objective_vars.size();
  if (n_obj == 0) return thrust::make_tuple(best_first, best_second, best_score);

  cuopt::pcgenerator_t rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);
  const i_t n_draws =
    n_obj < fj_cpu.hp.two_opt_candidates ? n_obj : fj_cpu.hp.two_opt_candidates;

  for (i_t t = 0; t < n_draws; ++t) {
    const i_t var1 = fj_cpu.problem->h_objective_vars[rng.next_u32() % (uint32_t)n_obj];
    if (!fj_cpu.h_is_binary_variable[var1]) continue;

    const f_t coeff1 = fj_cpu.problem->h_obj_coeffs[var1];
    const f_t val1   = fj_cpu.h_assignment[var1];
    const f_t delta1 = std::round(1.0 - 2 * val1);
    if (delta1 * coeff1 >= 0) continue;
    if (tabu_check<i_t, f_t>(fj_cpu, var1, delta1)) continue;

    // Breaking nothing is the single-flip lift's job; breaking several rows cannot be repaired by
    // one companion.
    const auto range1 = fj_cpu.range_for_variable(var1);
    i_t broken        = -1;
    bool multiple     = false;
    for (i_t k = range1.first; k < range1.second && !multiple; ++k) {
      const i_t r         = fj_cpu.h_reverse_constraints[k];
      const f_t new_slack = (fj_cpu.row_state()[r].slack + fj_cpu.h_slack_sumcomp[r]) -
                            (f_t)fj_cpu.h_reverse_coefficients[k] * delta1;
      if (new_slack < -fj_cpu.row_tolerance) {
        if (broken >= 0)
          multiple = true;
        else
          broken = r;
      }
    }
    if (multiple || broken < 0) continue;

    const auto row = fj_cpu.range_for_row(broken);
    for (i_t k = row.first; k < row.second; ++k) {
      const i_t var2 = fj_cpu.h_variables[k];
      if (var2 == var1) continue;
      if (!fj_cpu.h_is_binary_variable[var2]) continue;

      const f_t coeff2   = fj_cpu.problem->h_obj_coeffs[var2];
      const f_t val2     = fj_cpu.h_assignment[var2];
      const f_t delta2   = std::round(1.0 - 2 * val2);
      const f_t combined = delta1 * coeff1 + delta2 * coeff2;
      if (combined >= 0) continue;
      if (tabu_check<i_t, f_t>(fj_cpu, var2, delta2)) continue;
      if (!paired_flip_keeps_feasible<i_t, f_t>(fj_cpu, var1, delta1, var2, delta2)) continue;

      // Both lift operators rank on the objective gain in its own units: the score quantization
      // used elsewhere counts weights, so rounding a gain below 0.5 into it discards the move.
      const f_t improvement = -combined;
      if (improvement > best_improvement) {
        best_improvement = improvement;
        best_score.base  = 1;  // sign only, never compared against another operator's score
        best_first       = fj_move_t{var1, delta1};
        best_second      = fj_move_t{var2, delta2};
      }
    }
  }
  cuopt_assert((best_first.var_idx < 0) == (best_improvement <= 0),
               "pair and score must agree on whether a move was found");
  return thrust::make_tuple(best_first, best_second, best_score);
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_lift_move(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::find_lift_move");

  fj_move_t best_move          = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::zero();
  f_t best_improvement         = 0;

  for (auto var_idx : fj_cpu.problem->h_objective_vars) {
    cuopt_assert(var_idx < fj_cpu.problem->h_obj_coeffs.size(), "var_idx is out of bounds");
    cuopt_assert(var_idx >= 0, "var_idx is out of bounds");

    f_t obj_coeff = fj_cpu.problem->h_obj_coeffs[var_idx];
    f_t delta     = -std::numeric_limits<f_t>::infinity();
    f_t val       = fj_cpu.h_assignment[var_idx];

    // special path for binary variables
    if (fj_cpu.h_is_binary_variable[var_idx]) {
      cuopt_assert(fj_cpu.problem->is_integer(val), "binary variable is not integer");
      cuopt_assert(fj_cpu.problem->integer_equal(val, 0) || fj_cpu.problem->integer_equal(val, 1),
                   "Current assignment is not binary!");
      delta = std::round(1.0 - 2 * val);
      // flip move wouldn't improve
      if (delta * obj_coeff >= 0) continue;

      auto [offset_begin, offset_end] = fj_cpu.range_for_variable(var_idx);

      const i_t* const rev_cstr    = fj_cpu.h_reverse_constraints.data();
      const f_t* const rev_coeff   = fj_cpu.h_reverse_coefficients.data();
      const f_t* const row_sumcomp = fj_cpu.h_slack_sumcomp.data();
      const typename fj_cpu_climber_t<i_t, f_t>::row_state_t* const state = fj_cpu.row_state();

      bool breaks_a_row = false;
      i_t scanned       = 0;
      for (i_t j = offset_begin; j < offset_end; ++j) {
        ++scanned;
        const i_t cstr_idx   = rev_cstr[j];
        const f_t cstr_coeff = rev_coeff[j];
        const f_t new_slack  = (state[cstr_idx].slack + row_sumcomp[cstr_idx]) - cstr_coeff * delta;
        if (new_slack < -fj_cpu.row_tolerance) {
          breaks_a_row = true;
          break;
        }
      }

      const size_t nnz_scanned = (size_t)scanned;
      fj_cpu.h_reverse_constraints.byte_loads += nnz_scanned * sizeof(i_t);
      fj_cpu.h_reverse_coefficients.byte_loads += nnz_scanned * sizeof(f_t);
      fj_cpu.h_row_state.byte_loads +=
        nnz_scanned * sizeof(typename fj_cpu_climber_t<i_t, f_t>::row_state_t);
      fj_cpu.h_slack_sumcomp.byte_loads += nnz_scanned * sizeof(f_t);

      if (breaks_a_row) continue;
    } else {
      f_t lfd_lb                      = get_lower(fj_cpu.h_var_bounds[var_idx].get()) - val;
      f_t lfd_ub                      = get_upper(fj_cpu.h_var_bounds[var_idx].get()) - val;
      auto [offset_begin, offset_end] = fj_cpu.range_for_variable(var_idx);
      for (i_t j = offset_begin; j < offset_end; j += 1) {
        const i_t cstr_idx   = fj_cpu.h_reverse_constraints[j];
        const f_t cstr_coeff = fj_cpu.h_reverse_coefficients[j];
        if (cstr_coeff == f_t{0}) continue;
        const f_t slack = fj_cpu.row_state()[cstr_idx].slack;
        cuopt_assert(!(slack < -fj_cpu.row_tolerance), "cstr should be satisfied");

        // One bound per row here, and the sign the two-sided form carried is in the coefficient.
        f_t delta_j = slack / cstr_coeff;
        if (is_integer_var<i_t, f_t>(fj_cpu, var_idx))
          delta_j = cstr_coeff < 0 ? std::ceil(delta_j) : std::floor(delta_j);

        // skip this variable if there is no slack
        if (std::fabs(slack) <= fj_cpu.row_tolerance) {
          if (cstr_coeff > 0) {
            lfd_ub = 0;
          } else {
            lfd_lb = 0;
          }
        } else if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val + delta_j)) {
          continue;
        } else {
          if (cstr_coeff < 0) {
            lfd_lb = std::max(lfd_lb, delta_j);
          } else {
            lfd_ub = std::min(lfd_ub, delta_j);
          }
        }
        if (lfd_lb >= lfd_ub) break;
      }

      // invalid crossing bounds
      if (lfd_lb >= lfd_ub) { lfd_lb = lfd_ub = 0; }

      if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val + lfd_lb)) { lfd_lb = 0; }
      if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val + lfd_ub)) { lfd_ub = 0; }

      // Now that the lift move domain is computed, compute the correct lift move
      cuopt_assert(std::isfinite(val), "invalid assignment value");
      delta = obj_coeff < 0 ? lfd_ub : lfd_lb;
    }

    if (!std::isfinite(delta)) delta = 0;
    if (fj_cpu.problem->integer_equal(delta, (f_t)0)) continue;
    if (tabu_check<i_t, f_t>(fj_cpu, var_idx, delta)) continue;
    // The continuous branch takes its step straight from a row residual, bounded only by the
    // variable's own bounds, so an unbounded variable gets an unbounded step. This is the same bar
    // find_mtm_move holds its candidates to; total_violations twice because nothing here scores the
    // move, which leaves only the step and value clauses meaningful.
    if (!fj_cpu.move_numerically_stable(
          val, val + delta, fj_cpu.total_violations, fj_cpu.total_violations))
      continue;

    cuopt_assert(delta * obj_coeff < 0, "lift move doesn't improve the objective!");

    const f_t improvement = -obj_coeff * delta;
    if (improvement > best_improvement) {
      best_improvement = improvement;
      best_score.base  = 1;
      best_move        = fj_move_t{var_idx, delta};
    }
  }

  cuopt_assert((best_move.var_idx < 0) == (best_improvement <= 0),
               "move and score must agree on whether a move was found");
  return thrust::make_tuple(best_move, best_score);
}

}  // namespace cuopt::mathematical_optimization::mip
