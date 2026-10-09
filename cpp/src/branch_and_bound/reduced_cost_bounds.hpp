/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/user_problem.hpp>

#include <algorithm>
#include <cmath>
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

  void update_reduced_cost_bounds(const simplex::lp_problem_t<i_t, f_t>& lp,
                                  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                  const std::vector<simplex::variable_type_t>& var_types,
                                  f_t relaxation_objective,
                                  const std::vector<f_t>& reduced_costs,
                                  const std::vector<simplex::variable_status_t>& var_status)
  {
    const i_t n         = num_cols();
    const f_t threshold = 100.0 * settings.integer_tol;
    const f_t tol       = 1e-2;
    for (i_t j = 0; j < n; ++j) {
      if (std::isfinite(reduced_costs[j]) && std::abs(reduced_costs[j]) > threshold &&
          var_status[j] != simplex::variable_status_t::BASIC) {
        const f_t lower_j = lp.lower[j];
        const f_t upper_j = lp.upper[j];

        // x_j <= l_j + (incumbent_objective - relaxation_objective) / reduced_costs[j]
        // Let u_tilde_j = u_j - epsilon, so that floor(u_tilde_j) = u_j - 1
        // We want to solve for what the incumbent objective needs to be to make
        // x_j <= u_tilde_j
        // This means l_j + (incumbent_objective - relaxation_objective) / reduced_costs[j] <=
        // u_tilde_j Or equivalently, incumbent_objective <= relaxation_objective + reduced_costs[j]
        // * (u_tilde_j - l_j) when reduced_costs[j] > 0
        if (lower_j > -inf && reduced_costs[j] > 0) {
          const f_t u_tilde_j = var_types[j] == simplex::variable_type_t::INTEGER
                                  ? upper_j - tol
                                  : std::max(upper_j - 1.0, lower_j);
          const f_t bound_j =
            var_types[j] == simplex::variable_type_t::INTEGER ? std::floor(u_tilde_j) : u_tilde_j;
          const f_t diff        = u_tilde_j - lower_j;
          const f_t objective_j = relaxation_objective + diff * reduced_costs[j];
          if (((var_types[j] == simplex::variable_type_t::INTEGER && bound_j == upper_j - 1.0) ||
               var_types[j] != simplex::variable_type_t::INTEGER) &&
              std::isfinite(objective_j) && std::isfinite(bound_j)) {
            add_upper_bound(j, objective_j, bound_j);
          }
        }

        // x_j >= u_j + (incumbent_objective - relaxation_objective) / reduced_costs[j] when
        // reduced_costs[j] < 0 Let l_tilde_j = l_j + epsilon, so that ceil(l_tilde_j) = l_j + 1 We
        // want to solve for what the incumbent objective needs to be to make x_j >= l_tilde_j This
        // means u_j + (incumbent_objective - relaxation_objective) / reduced_costs[j] >= l_tilde_j
        // Or equivalently, incumbent_objective <=  relaxation_objective + reduced_costs[j] *
        // (l_tilde_j
        // - u_j) when reduced_costs[j] < 0
        if (upper_j < inf && reduced_costs[j] < 0) {
          const f_t l_tilde_j = var_types[j] == simplex::variable_type_t::INTEGER
                                  ? lower_j + tol
                                  : std::min(lower_j + 1.0, upper_j);
          const f_t bound_j =
            var_types[j] == simplex::variable_type_t::INTEGER ? std::ceil(l_tilde_j) : l_tilde_j;
          const f_t diff        = l_tilde_j - upper_j;
          const f_t objective_j = relaxation_objective + diff * reduced_costs[j];
          if (((var_types[j] == simplex::variable_type_t::INTEGER && bound_j == lower_j + 1.0) ||
               var_types[j] != simplex::variable_type_t::INTEGER) &&
              std::isfinite(objective_j) && std::isfinite(bound_j)) {
            add_lower_bound(j, objective_j, bound_j);
          }
        }
      }
    }
  }

  i_t update_bounds_from_new_incumbent(const simplex::lp_problem_t<i_t, f_t>& lp,
                                       const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                       const simplex::lp_solution_t<i_t, f_t>& solution,
                                       f_t relaxation_objective,
                                       f_t incumbent_objective,
                                       const std::vector<simplex::variable_type_t>& var_types,
                                       std::vector<f_t>& lower_bounds,
                                       std::vector<f_t>& upper_bounds)
  {
    const i_t n                           = static_cast<i_t>(lower_bounds_.size());
    f_t max_objective                     = -std::numeric_limits<f_t>::infinity();
    i_t integer_bounds_updated            = 0;
    const std::vector<f_t>& reduced_costs = solution.z;
    const f_t threshold                   = 100.0 * settings.integer_tol;
    const f_t weaken                      = settings.integer_tol;
    if (std::isfinite(incumbent_objective) && std::isfinite(relaxation_objective) &&
        incumbent_objective >= relaxation_objective) {
      const f_t abs_gap = incumbent_objective - relaxation_objective;
      for (i_t j = 0; j < n; ++j) {
        if (var_types[j] != simplex::variable_type_t::INTEGER || lp.upper[j] - lp.lower[j] <= 1.0) {
          continue;
        }
        const f_t rc = reduced_costs[j];
        if (!std::isfinite(rc) || std::abs(rc) <= threshold) { continue; }
        const f_t lower_j = lp.lower[j];
        const f_t upper_j = lp.upper[j];
        if (lower_j > -inf && reduced_costs[j] > 0) {
          const f_t new_upper_bound          = lower_j + abs_gap / reduced_costs[j];
          const f_t reduced_cost_upper_bound = std::floor(new_upper_bound + weaken);
          if (reduced_cost_upper_bound < upper_j) {
            upper_bounds[j] = reduced_cost_upper_bound;
            ++integer_bounds_updated;
          }
        }
        if (upper_j < inf && reduced_costs[j] < 0) {
          const f_t new_lower_bound          = upper_j + abs_gap / reduced_costs[j];
          const f_t reduced_cost_lower_bound = std::ceil(new_lower_bound - weaken);
          if (reduced_cost_lower_bound > lower_j) {
            lower_bounds[j] = reduced_cost_lower_bound;
            ++integer_bounds_updated;
          }
        }
      }
    }
    for (i_t j = 0; j < n; ++j) {
      if (lower_bounds_[j].is_valid()) {
        if (incumbent_objective <= lower_bounds_[j].objective &&
            lower_bounds_[j].bound > lower_bounds[j]) {
          lower_bounds[j] = lower_bounds_[j].bound;
          if (var_types[j] == simplex::variable_type_t::INTEGER) { integer_bounds_updated++; }
        }
        if (lower_bounds_[j].bound <= lower_bounds[j]) {
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
          upper_bounds[j] = upper_bounds_[j].bound;
          if (var_types[j] == simplex::variable_type_t::INTEGER) { integer_bounds_updated++; }
        }
        if (upper_bounds_[j].bound >= upper_bounds[j]) {
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
