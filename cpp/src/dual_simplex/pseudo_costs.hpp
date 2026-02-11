/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/branch_and_bound_worker.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>
#include <utilities/pcgenerator.hpp>

#include <omp.h>
#include <cmath>
#include <cstdint>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t>
struct branch_variable_t {
  i_t variable;
  rounding_direction_t direction;
};

template <typename i_t, typename f_t>
struct pseudo_cost_update_t {
  i_t variable;
  rounding_direction_t direction;
  f_t delta;
  double work_timestamp;
  int worker_id;

  bool operator<(const pseudo_cost_update_t& other) const
  {
    if (work_timestamp != other.work_timestamp) return work_timestamp < other.work_timestamp;
    if (variable != other.variable) return variable < other.variable;
    if (delta != other.delta) return delta < other.delta;
    return worker_id < other.worker_id;
  }
};

template <typename f_t>
struct pseudo_cost_averages_t {
  f_t down_avg;
  f_t up_avg;
};

// used to get T from omp_atomic_t<T> based on the fact that omp_atomic_t<T>::operator++ returns T
template <typename T>
using underlying_type = decltype(std::declval<T&>()++);

// Necessary because omp_atomic_t<f_t> may be passed instead of f_t
template <typename MaybeWrappedI, typename MaybeWrappedF>
auto compute_pseudo_cost_averages(const MaybeWrappedF* pc_sum_down,
                                  const MaybeWrappedF* pc_sum_up,
                                  const MaybeWrappedI* pc_num_down,
                                  const MaybeWrappedI* pc_num_up,
                                  size_t n)
{
  using underlying_f_t = underlying_type<MaybeWrappedF>;
  using underlying_i_t = underlying_type<MaybeWrappedI>;

  underlying_i_t num_initialized_down = 0;
  underlying_i_t num_initialized_up   = 0;
  underlying_f_t pseudo_cost_down_avg = 0.0;
  underlying_f_t pseudo_cost_up_avg   = 0.0;

  for (size_t j = 0; j < n; ++j) {
    if (pc_num_down[j] > 0) {
      ++num_initialized_down;
      if (std::isfinite(pc_sum_down[j])) {
        pseudo_cost_down_avg += pc_sum_down[j] / pc_num_down[j];
      }
    }
    if (pc_num_up[j] > 0) {
      ++num_initialized_up;
      if (std::isfinite(pc_sum_up[j])) { pseudo_cost_up_avg += pc_sum_up[j] / pc_num_up[j]; }
    }
  }

  pseudo_cost_down_avg =
    (num_initialized_down > 0) ? pseudo_cost_down_avg / num_initialized_down : 1.0;
  pseudo_cost_up_avg = (num_initialized_up > 0) ? pseudo_cost_up_avg / num_initialized_up : 1.0;

  return pseudo_cost_averages_t<underlying_f_t>{pseudo_cost_down_avg, pseudo_cost_up_avg};
}

// Variable selection using pseudo-cost product scoring
// Returns the best variable to branch on
template <typename i_t, typename f_t>
i_t variable_selection_from_pseudo_costs(const f_t* pc_sum_down,
                                         const f_t* pc_sum_up,
                                         const i_t* pc_num_down,
                                         const i_t* pc_num_up,
                                         i_t n_vars,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return -1;

  auto [pc_down_avg, pc_up_avg] =
    compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  i_t branch_var    = fractional[0];
  f_t max_score     = std::numeric_limits<f_t>::lowest();
  constexpr f_t eps = f_t(1e-6);

  for (i_t j : fractional) {
    f_t pc_down      = pc_num_down[j] != 0 ? pc_sum_down[j] / pc_num_down[j] : pc_down_avg;
    f_t pc_up        = pc_num_up[j] != 0 ? pc_sum_up[j] / pc_num_up[j] : pc_up_avg;
    const f_t f_down = solution[j] - std::floor(solution[j]);
    const f_t f_up   = std::ceil(solution[j]) - solution[j];
    f_t score        = std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);
    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  return branch_var;
}

// Objective estimate using pseudo-costs (lock-free implementation)
// Returns lower_bound + estimated cost to reach integer feasibility
template <typename i_t, typename f_t>
f_t obj_estimate_from_arrays(const f_t* pc_sum_down,
                             const f_t* pc_sum_up,
                             const i_t* pc_num_down,
                             const i_t* pc_num_up,
                             i_t n_vars,
                             const std::vector<i_t>& fractional,
                             const std::vector<f_t>& solution,
                             f_t lower_bound)
{
  auto [pc_down_avg, pc_up_avg] =
    compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  f_t estimate      = lower_bound;
  constexpr f_t eps = f_t(1e-6);

  for (i_t j : fractional) {
    f_t pc_down      = pc_num_down[j] != 0 ? pc_sum_down[j] / pc_num_down[j] : pc_down_avg;
    f_t pc_up        = pc_num_up[j] != 0 ? pc_sum_up[j] / pc_num_up[j] : pc_up_avg;
    const f_t f_down = solution[j] - std::floor(solution[j]);
    const f_t f_up   = std::ceil(solution[j]) - solution[j];
    estimate += std::min(std::max(pc_down * f_down, eps), std::max(pc_up * f_up, eps));
  }

  return estimate;
}

template <typename i_t, typename f_t, typename MaybeWrappedI = i_t, typename MaybeWrappedF = f_t>
branch_variable_t<i_t> pseudocost_diving_from_arrays(const MaybeWrappedF* pc_sum_down,
                                                     const MaybeWrappedF* pc_sum_up,
                                                     const MaybeWrappedI* pc_num_down,
                                                     const MaybeWrappedI* pc_num_up,
                                                     i_t n_vars,
                                                     const std::vector<i_t>& fractional,
                                                     const std::vector<f_t>& solution,
                                                     const std::vector<f_t>& root_solution)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return {-1, rounding_direction_t::NONE};

  auto avgs = compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  i_t branch_var                 = fractional[0];
  f_t max_score                  = std::numeric_limits<f_t>::lowest();
  rounding_direction_t round_dir = rounding_direction_t::DOWN;
  constexpr f_t eps              = f_t(1e-6);

  for (i_t j : fractional) {
    f_t f_down  = solution[j] - std::floor(solution[j]);
    f_t f_up    = std::ceil(solution[j]) - solution[j];
    f_t pc_down = pc_num_down[j] != 0 ? (f_t)pc_sum_down[j] / (f_t)pc_num_down[j] : avgs.down_avg;
    f_t pc_up   = pc_num_up[j] != 0 ? (f_t)pc_sum_up[j] / (f_t)pc_num_up[j] : avgs.up_avg;

    f_t score_down = std::sqrt(f_up) * (1 + pc_up) / (1 + pc_down);
    f_t score_up   = std::sqrt(f_down) * (1 + pc_down) / (1 + pc_up);

    f_t score                = 0;
    rounding_direction_t dir = rounding_direction_t::DOWN;

    f_t root_val = (j < static_cast<i_t>(root_solution.size())) ? root_solution[j] : solution[j];

    if (solution[j] < root_val - f_t(0.4)) {
      score = score_down;
      dir   = rounding_direction_t::DOWN;
    } else if (solution[j] > root_val + f_t(0.4)) {
      score = score_up;
      dir   = rounding_direction_t::UP;
    } else if (f_down < f_t(0.3)) {
      score = score_down;
      dir   = rounding_direction_t::DOWN;
    } else if (f_down > f_t(0.7)) {
      score = score_up;
      dir   = rounding_direction_t::UP;
    } else if (pc_down < pc_up + eps) {
      score = score_down;
      dir   = rounding_direction_t::DOWN;
    } else {
      score = score_up;
      dir   = rounding_direction_t::UP;
    }

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  if (round_dir == rounding_direction_t::NONE) {
    branch_var = fractional[0];
    round_dir  = rounding_direction_t::DOWN;
  }

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t, typename MaybeWrappedI = i_t, typename MaybeWrappedF = f_t>
branch_variable_t<i_t> guided_diving_from_arrays(const MaybeWrappedF* pc_sum_down,
                                                 const MaybeWrappedF* pc_sum_up,
                                                 const MaybeWrappedI* pc_num_down,
                                                 const MaybeWrappedI* pc_num_up,
                                                 i_t n_vars,
                                                 const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& solution,
                                                 const std::vector<f_t>& incumbent)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return {-1, rounding_direction_t::NONE};

  auto avgs = compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  i_t branch_var                 = fractional[0];
  f_t max_score                  = std::numeric_limits<f_t>::lowest();
  rounding_direction_t round_dir = rounding_direction_t::DOWN;
  constexpr f_t eps              = f_t(1e-6);

  for (i_t j : fractional) {
    f_t f_down    = solution[j] - std::floor(solution[j]);
    f_t f_up      = std::ceil(solution[j]) - solution[j];
    f_t down_dist = std::abs(incumbent[j] - std::floor(solution[j]));
    f_t up_dist   = std::abs(std::ceil(solution[j]) - incumbent[j]);
    rounding_direction_t dir =
      down_dist < up_dist + eps ? rounding_direction_t::DOWN : rounding_direction_t::UP;

    f_t pc_down = pc_num_down[j] != 0 ? (f_t)pc_sum_down[j] / (f_t)pc_num_down[j] : avgs.down_avg;
    f_t pc_up   = pc_num_up[j] != 0 ? (f_t)pc_sum_up[j] / (f_t)pc_num_up[j] : avgs.up_avg;

    f_t score1 = dir == rounding_direction_t::DOWN ? 5 * pc_down * f_down : 5 * pc_up * f_up;
    f_t score2 = dir == rounding_direction_t::DOWN ? pc_up * f_up : pc_down * f_down;
    f_t score  = (score1 + score2) / 6;

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
class pseudo_cost_snapshot_t {
 public:
  pseudo_cost_snapshot_t() = default;

  pseudo_cost_snapshot_t(std::vector<f_t> sum_down,
                         std::vector<f_t> sum_up,
                         std::vector<i_t> num_down,
                         std::vector<i_t> num_up)
    : sum_down_(std::move(sum_down)),
      sum_up_(std::move(sum_up)),
      num_down_(std::move(num_down)),
      num_up_(std::move(num_up))
  {
  }

  i_t variable_selection(const std::vector<i_t>& fractional, const std::vector<f_t>& solution) const
  {
    return variable_selection_from_pseudo_costs(sum_down_.data(),
                                                sum_up_.data(),
                                                num_down_.data(),
                                                num_up_.data(),
                                                n_vars(),
                                                fractional,
                                                solution);
  }

  f_t obj_estimate(const std::vector<i_t>& fractional,
                   const std::vector<f_t>& solution,
                   f_t lower_bound) const
  {
    return obj_estimate_from_arrays(sum_down_.data(),
                                    sum_up_.data(),
                                    num_down_.data(),
                                    num_up_.data(),
                                    n_vars(),
                                    fractional,
                                    solution,
                                    lower_bound);
  }

  branch_variable_t<i_t> pseudocost_diving(const std::vector<i_t>& fractional,
                                           const std::vector<f_t>& solution,
                                           const std::vector<f_t>& root_solution) const
  {
    return pseudocost_diving_from_arrays(sum_down_.data(),
                                         sum_up_.data(),
                                         num_down_.data(),
                                         num_up_.data(),
                                         n_vars(),
                                         fractional,
                                         solution,
                                         root_solution);
  }

  branch_variable_t<i_t> guided_diving(const std::vector<i_t>& fractional,
                                       const std::vector<f_t>& solution,
                                       const std::vector<f_t>& incumbent) const
  {
    return guided_diving_from_arrays(sum_down_.data(),
                                     sum_up_.data(),
                                     num_down_.data(),
                                     num_up_.data(),
                                     n_vars(),
                                     fractional,
                                     solution,
                                     incumbent);
  }

  void queue_update(
    i_t variable, rounding_direction_t direction, f_t delta, double clock, int worker_id)
  {
    updates_.push_back({variable, direction, delta, clock, worker_id});
    if (direction == rounding_direction_t::DOWN) {
      sum_down_[variable] += delta;
      num_down_[variable]++;
    } else {
      sum_up_[variable] += delta;
      num_up_[variable]++;
    }
  }

  std::vector<pseudo_cost_update_t<i_t, f_t>> take_updates()
  {
    std::vector<pseudo_cost_update_t<i_t, f_t>> result;
    result.swap(updates_);
    return result;
  }

  i_t n_vars() const { return (i_t)sum_down_.size(); }

  std::vector<f_t> sum_down_;
  std::vector<f_t> sum_up_;
  std::vector<i_t> num_down_;
  std::vector<i_t> num_up_;

 private:
  std::vector<pseudo_cost_update_t<i_t, f_t>> updates_;
};

template <typename i_t, typename f_t>
struct reliability_branching_settings_t {
  // Lower bound for the maximum number of LP iterations for a single trial branching
  i_t lower_max_lp_iter = 10;

  // Upper bound for the maximum number of LP iterations for a single trial branching
  i_t upper_max_lp_iter = 500;

  // Priority of the tasks created when running the trial branching in parallel.
  // Set to 1 to have the same priority as the other tasks.
  i_t task_priority = 5;

  // The maximum number of candidates initialized by strong branching in a single
  // node
  i_t max_num_candidates = 100;

  // Define the maximum number of iteration spent in strong branching.
  // Let `bnb_lp_iter` = total number of iterations in B&B, then
  // `max iter in strong branching = bnb_lp_factor * bnb_lp_iter + bnb_lp_offset`.
  // This is used for determining the `reliable_threshold`.
  f_t bnb_lp_factor = 0.5;
  i_t bnb_lp_offset = 100000;

  // Maximum and minimum points in curve to determine the value
  // of the `reliable_threshold` based on the current number of LP
  // iterations in strong branching and B&B. Since it is a
  // a curve, the actual value of `reliable_threshold` may be
  // higher than `max_reliable_threshold`.
  // Only used when `reliable_threshold` is negative
  i_t max_reliable_threshold = 5;
  i_t min_reliable_threshold = 1;
};

template <typename i_t, typename f_t>
class pseudo_costs_t {
 public:
  explicit pseudo_costs_t(i_t num_variables)
    : pseudo_cost_sum_down(num_variables),
      pseudo_cost_sum_up(num_variables),
      pseudo_cost_num_down(num_variables),
      pseudo_cost_num_up(num_variables),
      pseudo_cost_mutex_up(num_variables),
      pseudo_cost_mutex_down(num_variables)
  {
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_objective);

  pseudo_cost_snapshot_t<i_t, f_t> create_snapshot() const
  {
    const i_t n = (i_t)pseudo_cost_sum_down.size();
    std::vector<f_t> sd(n), su(n);
    std::vector<i_t> nd(n), nu(n);
    for (i_t j = 0; j < n; ++j) {
      sd[j] = pseudo_cost_sum_down[j];
      su[j] = pseudo_cost_sum_up[j];
      nd[j] = pseudo_cost_num_down[j];
      nu[j] = pseudo_cost_num_up[j];
    }
    return pseudo_cost_snapshot_t<i_t, f_t>(
      std::move(sd), std::move(su), std::move(nd), std::move(nu));
  }

  void merge_updates(const std::vector<pseudo_cost_update_t<i_t, f_t>>& updates)
  {
    for (const auto& upd : updates) {
      if (upd.direction == rounding_direction_t::DOWN) {
        pseudo_cost_sum_down[upd.variable] += upd.delta;
        pseudo_cost_num_down[upd.variable]++;
      } else {
        pseudo_cost_sum_up[upd.variable] += upd.delta;
        pseudo_cost_num_up[upd.variable]++;
      }
    }
  }

  void resize(i_t num_variables)
  {
    pseudo_cost_sum_down.assign(num_variables, 0);
    pseudo_cost_sum_up.assign(num_variables, 0);
    pseudo_cost_num_down.assign(num_variables, 0);
    pseudo_cost_num_up.assign(num_variables, 0);
    pseudo_cost_mutex_up.resize(num_variables);
    pseudo_cost_mutex_down.resize(num_variables);
  }

  void initialized(i_t& num_initialized_down,
                   i_t& num_initialized_up,
                   f_t& pseudo_cost_down_avg,
                   f_t& pseudo_cost_up_avg) const;

  f_t obj_estimate(const std::vector<i_t>& fractional,
                   const std::vector<f_t>& solution,
                   f_t lower_bound,
                   logger_t& log);

  i_t variable_selection(const std::vector<i_t>& fractional,
                         const std::vector<f_t>& solution,
                         logger_t& log);

  i_t reliable_variable_selection(mip_node_t<i_t, f_t>* node_ptr,
                                  const std::vector<i_t>& fractional,
                                  const std::vector<f_t>& solution,
                                  const simplex_solver_settings_t<i_t, f_t>& settings,
                                  const std::vector<variable_type_t>& var_types,
                                  branch_and_bound_worker_t<i_t, f_t>* worker,
                                  const branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
                                  f_t upper_bound,
                                  int max_num_tasks,
                                  logger_t& log);

  void update_pseudo_costs_from_strong_branching(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& root_soln);

  uint32_t compute_state_hash() const
  {
    return detail::compute_hash(pseudo_cost_sum_down) ^ detail::compute_hash(pseudo_cost_sum_up) ^
           detail::compute_hash(pseudo_cost_num_down) ^ detail::compute_hash(pseudo_cost_num_up);
  }

  uint32_t compute_strong_branch_hash() const
  {
    return detail::compute_hash(strong_branch_down) ^ detail::compute_hash(strong_branch_up);
  }

  f_t calculate_pseudocost_score(i_t j,
                                 const std::vector<f_t>& solution,
                                 f_t pseudo_cost_up_avg,
                                 f_t pseudo_cost_down_avg) const;

  reliability_branching_settings_t<i_t, f_t> reliability_branching_settings;

  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_up;
  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_down;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_up;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_down;
  std::vector<f_t> strong_branch_down;
  std::vector<f_t> strong_branch_up;
  std::vector<omp_mutex_t> pseudo_cost_mutex_up;
  std::vector<omp_mutex_t> pseudo_cost_mutex_down;
  omp_atomic_t<i_t> num_strong_branches_completed = 0;
  omp_atomic_t<int64_t> strong_branching_lp_iter  = 0;
};

template <typename i_t, typename f_t>
void strong_branching(const user_problem_t<i_t, f_t>& original_problem,
                      const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc);

}  // namespace cuopt::linear_programming::dual_simplex
