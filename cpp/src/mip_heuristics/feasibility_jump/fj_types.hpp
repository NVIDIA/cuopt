/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>

namespace cuopt::mathematical_optimization::mip {

struct fj_hyper_parameters_t {
  int max_sampled_moves                   = 32 * 16;
  double random_var_probability           = 0.04;
  double random_cstr_probability          = 0.16;
  int global_move_update_period           = 10;
  int heavy_move_update_period            = 50;
  int sync_period                         = 200;
  int lhs_refresh_period                  = 500;
  int allow_infeasibility_iterations      = 200;
  double objective_weight_increment       = 0.01;
  int load_balancing_variable_threshold   = 300;
  int load_balancing_constraint_threshold = 5000;
  int load_balancing_variable_split_size  = 50;

  double breakthrough_move_epsilon    = 1e-4;
  int tabu_tenure_min                 = 3;
  int tabu_tenure_max                 = 13;
  double excess_improvement_weight    = (1.0 / 2.0);
  double weight_smoothing_probability = 0.0003;

  double fractional_score_multiplier = 100;
  double rounding_second_stage_split = 0.1;

  double small_move_tabu_threshold = 1e-6;
  int small_move_tabu_tenure       = 4;

  int two_opt_max_rows     = 4;
  int two_opt_max_row_vars = 256;
  int two_opt_max_pairs    = 256;

  int old_codepath_total_var_to_relvar_ratio_threshold = 200;
  int load_balancing_codepath_min_varcount             = 3200;
};

enum class fj_mode_t { FIRST_FEASIBLE, GREEDY_DESCENT, TREE, ROUNDING, EXIT_NON_IMPROVING };

enum class MTMMoveType { FJ_MTM_VIOLATED, FJ_MTM_SATISFIED, FJ_MTM_ALL };

enum class fj_load_balancing_mode_t { ALWAYS_ON, AUTO, ALWAYS_OFF };

enum class fj_candidate_selection_t { WEIGHTED_SCORE, FEASIBLE_FIRST };

struct fj_settings_t {
  int seed{0};
  fj_mode_t mode{fj_mode_t::FIRST_FEASIBLE};
  fj_candidate_selection_t candidate_selection{fj_candidate_selection_t::WEIGHTED_SCORE};
  double time_limit{60.0};
  int iteration_limit{std::numeric_limits<int>::max()};
  fj_hyper_parameters_t parameters{};
  int n_of_minimums_for_exit  = 7000;
  double infeasibility_weight = 1.0;
  bool update_weights         = true;
  bool feasibility_run        = true;
  fj_load_balancing_mode_t load_balancing_mode{fj_load_balancing_mode_t::AUTO};
  double baseline_objective_for_longer_run{std::numeric_limits<double>::lowest()};
};

struct fj_move_t {
  int var_idx;
  double value;

  bool operator<(const fj_move_t& rhs) const
  {
    if (var_idx == rhs.var_idx) return value < rhs.value;
    return var_idx < rhs.var_idx;
  }
  bool operator==(const fj_move_t& rhs) const
  {
    return var_idx == rhs.var_idx && value == rhs.value;
  }
  bool operator!=(const fj_move_t& rhs) const { return !(*this == rhs); }
};

struct fj_staged_score_t {
  float base{-std::numeric_limits<float>::infinity()};
  float bonus{-std::numeric_limits<float>::infinity()};

#if defined(__CUDACC__)
#define CUOPT_FJ_HOST_DEVICE inline __host__ __device__
#else
#define CUOPT_FJ_HOST_DEVICE inline
#endif

  CUOPT_FJ_HOST_DEVICE bool operator<(fj_staged_score_t other) const noexcept
  {
    return base == other.base ? bonus < other.bonus : base < other.base;
  }
  CUOPT_FJ_HOST_DEVICE bool operator>(fj_staged_score_t other) const noexcept
  {
    return base == other.base ? bonus > other.bonus : base > other.base;
  }
  CUOPT_FJ_HOST_DEVICE bool operator==(fj_staged_score_t other) const noexcept
  {
    return base == other.base && bonus == other.bonus;
  }
  CUOPT_FJ_HOST_DEVICE bool operator!=(fj_staged_score_t other) const noexcept
  {
    return !(*this == other);
  }

  CUOPT_FJ_HOST_DEVICE static fj_staged_score_t invalid()
  {
    return {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  }
  CUOPT_FJ_HOST_DEVICE static fj_staged_score_t zero() { return {0, 0}; }

  CUOPT_FJ_HOST_DEVICE bool valid() const { return *this != invalid(); }

#undef CUOPT_FJ_HOST_DEVICE
};

}  // namespace cuopt::mathematical_optimization::mip
