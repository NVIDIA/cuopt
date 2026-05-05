/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/constants.hpp>
#include <branch_and_bound/mip_node.hpp>
#include <branch_and_bound/worker.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>

#include <utilities/omp_helpers.hpp>
#include <utilities/pcgenerator.hpp>

#include <cmath>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <limits>

namespace cuopt::linear_programming::dual_simplex {

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

  // Estimate the objective change of each fractional variable
  // using a single pivot of dual simplex. Then rank the candidates
  // based on this estimation.
  bool rank_candidates_with_dual_pivot = true;
};

template <typename i_t>
struct branch_variable_t {
  i_t variable;
  branch_direction_t direction;
};

template <typename i_t, typename f_t>
struct batch_pdlp_warm_cache_t {
  const raft::handle_t batch_pdlp_handle{};
  rmm::device_uvector<f_t> initial_primal{0, batch_pdlp_handle.get_stream()};
  rmm::device_uvector<f_t> initial_dual{0, batch_pdlp_handle.get_stream()};
  f_t step_size{std::numeric_limits<f_t>::signaling_NaN()};
  f_t primal_weight{std::numeric_limits<f_t>::signaling_NaN()};
  i_t pdlp_iteration{-1};
  f_t percent_solved_by_batch_pdlp_at_root{f_t(0.0)};
  bool populated{false};
};

template <typename i_t, typename f_t>
struct pseudo_cost_update_t {
  i_t variable;
  branch_direction_t direction;
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

template <typename i_t, typename f_t>
class pseudo_costs_t {
 public:
  explicit pseudo_costs_t(i_t num_variables, const simplex_solver_settings_t<i_t, f_t>& settings)
    : settings(settings),
      pseudo_cost_sum_down(num_variables),
      pseudo_cost_sum_up(num_variables),
      pseudo_cost_num_down(num_variables),
      pseudo_cost_num_up(num_variables),
      pseudo_cost_mutex_up(num_variables),
      pseudo_cost_mutex_down(num_variables),
      AT(std::make_shared<csc_matrix_t<i_t, f_t>>(1, 1, 1)),
      pdlp_warm_cache(std::make_shared<batch_pdlp_warm_cache_t<i_t, f_t>>())
  {
  }

  pseudo_costs_t(const pseudo_costs_t<i_t, f_t>& other) : pseudo_costs_t(1, other.settings)
  {
    *this = other;
  }

  pseudo_costs_t& operator=(const pseudo_costs_t& other)
  {
    if (this != &other) {
      this->AT                   = other.AT;
      this->pdlp_warm_cache      = other.pdlp_warm_cache;
      this->pseudo_cost_num_down = other.pseudo_cost_num_down;
      this->pseudo_cost_num_up   = other.pseudo_cost_num_up;
      this->pseudo_cost_sum_down = other.pseudo_cost_sum_down;
      this->pseudo_cost_sum_up   = other.pseudo_cost_sum_up;
    }
    return *this;
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_objective);

  void merge_updates(const std::vector<pseudo_cost_update_t<i_t, f_t>>& updates)
  {
    for (const auto& upd : updates) {
      if (upd.direction == branch_direction_t::DOWN) {
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

  f_t get_pseudocost_down(i_t j, f_t avg) const
  {
    i_t num = pseudo_cost_num_down[j];
    f_t sum = pseudo_cost_sum_down[j];
    return num > 0 ? sum / num : avg;
  }

  f_t get_pseudocost_up(i_t j, f_t avg) const
  {
    i_t num = pseudo_cost_num_up[j];
    f_t sum = pseudo_cost_sum_up[j];
    return num > 0 ? sum / num : avg;
  }

  f_t compute_pseudocost_average_down();
  f_t compute_pseudocost_average_up();

  f_t obj_estimate(const std::vector<i_t>& fractional,
                   const std::vector<f_t>& solution,
                   f_t lower_bound);

  i_t variable_selection(const std::vector<i_t>& fractional, const std::vector<f_t>& solution);

  i_t reliable_variable_selection(const mip_node_t<i_t, f_t>* node_ptr,
                                  const std::vector<i_t>& fractional,
                                  branch_and_bound_worker_t<i_t, f_t>* worker,
                                  const std::vector<variable_type_t>& var_types,
                                  const branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
                                  f_t upper_bound,
                                  int max_num_tasks,
                                  const std::vector<i_t>& new_slacks,
                                  const lp_problem_t<i_t, f_t>& original_lp);

  void update_pseudo_costs_from_strong_branching(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& strong_branch_down,
                                                 const std::vector<f_t>& strong_branch_up,
                                                 const std::vector<f_t>& root_soln);

  uint32_t compute_state_hash() const
  {
    return detail::compute_hash(pseudo_cost_sum_down) ^ detail::compute_hash(pseudo_cost_sum_up) ^
           detail::compute_hash(pseudo_cost_num_down) ^ detail::compute_hash(pseudo_cost_num_up);
  }

  f_t calculate_pseudocost_score(i_t j,
                                 const std::vector<f_t>& solution,
                                 f_t avg_down,
                                 f_t avg_up) const;

  std::shared_ptr<csc_matrix_t<i_t, f_t>> AT;  // Transpose of the constraint matrix A
  std::shared_ptr<batch_pdlp_warm_cache_t<i_t, f_t>> pdlp_warm_cache;

  reliability_branching_settings_t<i_t, f_t> reliability_branching_settings;
  simplex_solver_settings_t<i_t, f_t> settings;

 protected:
  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_up;
  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_down;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_up;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_down;
  std::vector<omp_mutex_t> pseudo_cost_mutex_up;
  std::vector<omp_mutex_t> pseudo_cost_mutex_down;

  omp_atomic_t<int64_t> strong_branching_lp_iter = 0;
};

template <typename i_t, typename f_t>
class pseudo_cost_snapshot_t : public pseudo_costs_t<i_t, f_t> {
 public:
  using Base = pseudo_costs_t<i_t, f_t>;
  using Base::Base;

  pseudo_cost_snapshot_t(const pseudo_costs_t<i_t, f_t>& other) : Base(1, other.settings)
  {
    Base::operator=(other);
  }

  pseudo_cost_snapshot_t operator=(const pseudo_costs_t<i_t, f_t>& other)
  {
    return Base::operator=(other);
  }

  void queue_update(
    i_t variable, branch_direction_t direction, f_t delta, double clock, int worker_id)
  {
    updates_.push_back({variable, direction, delta, clock, worker_id});
    if (direction == branch_direction_t::DOWN) {
      this->pseudo_cost_sum_down[variable] += delta;
      ++this->pseudo_cost_num_down[variable];
    } else {
      this->pseudo_cost_sum_up[variable] += delta;
      ++this->pseudo_cost_num_up[variable];
    }
  }

  std::vector<pseudo_cost_update_t<i_t, f_t>> take_updates()
  {
    std::vector<pseudo_cost_update_t<i_t, f_t>> result;
    result.swap(updates_);
    return result;
  }

  i_t n_vars() const { return this->pseudo_cost_sum_down.size(); }

 private:
  std::vector<pseudo_cost_update_t<i_t, f_t>> updates_;
};

template <typename i_t, typename f_t>
void strong_branching(const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<i_t>& new_slacks,
                      const std::vector<variable_type_t>& var_types,
                      const lp_solution_t<i_t, f_t>& root_solution,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      f_t upper_bound,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      const std::vector<i_t>& basic_list,
                      const std::vector<i_t>& nonbasic_list,
                      basis_update_mpf_t<i_t, f_t>& basis_factors,
                      pseudo_costs_t<i_t, f_t>& pc);

}  // namespace cuopt::linear_programming::dual_simplex
