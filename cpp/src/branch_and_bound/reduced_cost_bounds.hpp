/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>

#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename f_t>
struct objective_bound_pair_t {
  objective_bound_pair_t()
    : objective(std::numeric_limits<f_t>::quiet_NaN()), bound(std::numeric_limits<f_t>::quiet_NaN())
  {
  }
  objective_bound_pair_t(f_t objective_in, f_t bound_in) : objective(objective_in), bound(bound_in)
  {
  }
  bool is_valid() { return objective == objective && bound == bound; }
  f_t objective;
  f_t bound;
};

template <typename i_t, typename f_t>
class reduced_cost_bounds_t {
 public:
  static constexpr i_t kBoundDominated      = -2;
  static constexpr i_t kVariableOutOfBounds = -1;
  static constexpr i_t kBoundAccepted       = 1;
  static constexpr i_t kBoundTightened      = 2;

  reduced_cost_bounds_t(i_t original_cols)
    : max_objective_(-std::numeric_limits<f_t>::infinity()),
      lower_bounds_(original_cols),
      upper_bounds_(original_cols)
  {
  }

  i_t add_lower_bound(i_t col, f_t objective, f_t bound)
  {
    if (col < static_cast<i_t>(lower_bounds_.size())) {
      if (!lower_bounds_[col].is_valid()) {
        lower_bounds_[col] = objective_bound_pair_t<f_t>(objective, bound);
        if (objective > max_objective_) { max_objective_ = objective; }
        return kBoundAccepted;
      } else {
        if (bound > lower_bounds_[col].bound) {
          lower_bounds_[col] = objective_bound_pair_t<f_t>(objective, bound);
          if (objective > max_objective_) { max_objective_ = objective; }
          return kBoundTightened;
        } else if (bound == lower_bounds_[col].bound && objective > lower_bounds_[col].objective) {
          lower_bounds_[col].objective = objective;
          if (objective > max_objective_) { max_objective_ = objective; }
          return kBoundAccepted;
        } else {
          return kBoundDominated;
        }
      }
    } else {
      return kVariableOutOfBounds;
    }
  }

  i_t add_upper_bound(i_t col, f_t objective, f_t bound)
  {
    if (col < static_cast<i_t>(upper_bounds_.size())) {
      if (!upper_bounds_[col].is_valid()) {
        upper_bounds_[col] = objective_bound_pair_t<f_t>(objective, bound);
        if (objective > max_objective_) { max_objective_ = objective; }
        return kBoundAccepted;
      } else {
        if (bound < upper_bounds_[col].bound) {
          upper_bounds_[col] = objective_bound_pair_t<f_t>(objective, bound);
          if (objective > max_objective_) { max_objective_ = objective; }
          return kBoundTightened;
        } else if (bound == upper_bounds_[col].bound && objective > upper_bounds_[col].objective) {
          upper_bounds_[col].objective = objective;
          if (objective > max_objective_) { max_objective_ = objective; }
          return kBoundAccepted;
        } else {
          return kBoundDominated;
        }
      }
    } else {
      return kVariableOutOfBounds;
    }
  }

  i_t update_bounds_from_new_incumbent(f_t incumbent_objective,
                                       const std::vector<simplex::variable_type_t>& var_types,
                                       std::vector<f_t>& lower_bounds,
                                       std::vector<f_t>& upper_bounds)
  {
    const i_t n                = static_cast<i_t>(lower_bounds_.size());
    f_t max_objective          = -std::numeric_limits<f_t>::infinity();
    i_t integer_bounds_updated = 0;
    for (i_t j = 0; j < n; ++j) {
      if (lower_bounds_[j].is_valid()) {
        if (incumbent_objective <= lower_bounds_[j].objective &&
            lower_bounds_[j].bound > lower_bounds[j]) {
          // printf("RCF Variable %d (%d): lower %e -> %e\n", j, static_cast<int>(var_types[j]),
          // lower_bounds[j], lower_bounds_[j].bound);
          lower_bounds[j] = lower_bounds_[j].bound;
          if (var_types[j] == simplex::variable_type_t::INTEGER) { integer_bounds_updated++; }
          lower_bounds_[j].bound = lower_bounds_[j].objective =
            std::numeric_limits<f_t>::quiet_NaN();
        }
        if (lower_bounds_[j].objective > max_objective) {
          max_objective = lower_bounds_[j].objective;
        }
      }
      if (upper_bounds_[j].is_valid()) {
        if (incumbent_objective <= upper_bounds_[j].objective &&
            upper_bounds_[j].bound < upper_bounds[j]) {
          // printf("RCF Variable %d (%d): upper %e -> %e\n", j, static_cast<int>(var_types[j]),
          // upper_bounds[j], upper_bounds_[j].bound);
          upper_bounds[j] = upper_bounds_[j].bound;
          if (var_types[j] == simplex::variable_type_t::INTEGER) { integer_bounds_updated++; }
          upper_bounds_[j].bound = upper_bounds_[j].objective =
            std::numeric_limits<f_t>::quiet_NaN();
        }
        if (upper_bounds_[j].objective > max_objective) {
          max_objective = upper_bounds_[j].objective;
        }
      }
    }
    max_objective_ = max_objective;
    return integer_bounds_updated;
  }

  f_t get_current_lower_bound(i_t col)
  {
    if (col < static_cast<i_t>(lower_bounds_.size())) { return lower_bounds_[col].bound; }
    return std::numeric_limits<f_t>::quiet_NaN();
  }

  f_t get_current_upper_bound(i_t col)
  {
    if (col < static_cast<i_t>(upper_bounds_.size())) { return upper_bounds_[col].bound; }
    return std::numeric_limits<f_t>::quiet_NaN();
  }

  i_t num_cols() { return static_cast<i_t>(lower_bounds_.size()); }

  f_t get_max_objective() { return max_objective_; }

 private:
  f_t max_objective_;
  std::vector<objective_bound_pair_t<f_t>> lower_bounds_;
  std::vector<objective_bound_pair_t<f_t>> upper_bounds_;
};

}  // namespace cuopt::mathematical_optimization::mip
