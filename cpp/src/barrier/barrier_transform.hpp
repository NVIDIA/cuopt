/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace cuopt::mathematical_optimization {

/**
 * User-to-barrier transform retained on barrier_cache_t after Optimal:
 * convert / presolve / scaling, plus the scaled LP.
 * Enough to crush a new linear objective from the original problem into barrier
 * coordinates and to uncrush a solution without rerunning those algorithms.
 */
struct barrier_transform_t {
  int user_num_cols{0};
  int user_num_rows{0};
  int original_num_cols{0};
  int original_num_rows{0};
  double obj_scale{1.0};
  double obj_constant{0.0};

  // Enough of the user problem for reuse uncrush without rebuilding A.
  std::vector<char> row_sense;
  int cone_var_start{0};
  std::vector<int> second_order_cone_dims;
  int expanded_original_num_cols{0};
  std::vector<int> original_col_to_expanded_col;

  cuopt::mathematical_optimization::simplex::presolve_info_t<int, double> presolve_info;
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  // Barrier linear objective minus crush(user c) from the first solve (Q*ell shift, etc.).
  std::vector<double> linear_obj_shift;
  std::unique_ptr<cuopt::mathematical_optimization::simplex::lp_problem_t<int, double>> barrier_lp;
};

inline std::vector<double> crush_user_linear_objective(barrier_transform_t const& xf,
                                                       double const* c,
                                                       int n)
{
  if (c == nullptr || n != xf.user_num_cols) {
    throw std::invalid_argument(
      "update_linear_objective: linear objective length must match the cached user column count.");
  }
  if (xf.original_num_cols < xf.user_num_cols) {
    throw std::invalid_argument("update_linear_objective: cached original column count is smaller than user n.");
  }
  if (xf.barrier_lp == nullptr) {
    throw std::invalid_argument("update_linear_objective: cached barrier LP is missing.");
  }

  std::vector<double> orig(static_cast<std::size_t>(xf.original_num_cols), 0.0);
  for (int j = 0; j < n; ++j) {
    orig[static_cast<std::size_t>(j)] = c[j];
  }
  for (int j : xf.presolve_info.negated_variables) {
    orig[static_cast<std::size_t>(j)] *= -1.0;
  }

  std::vector<double> presolved;
  if (!xf.presolve_info.remaining_variables.empty()) {
    presolved.resize(xf.presolve_info.remaining_variables.size());
    for (std::size_t k = 0; k < xf.presolve_info.remaining_variables.size(); ++k) {
      presolved[k] = orig[static_cast<std::size_t>(xf.presolve_info.remaining_variables[k])];
    }
  } else {
    presolved = std::move(orig);
  }

  auto const& pairs = xf.presolve_info.free_variable_pairs;
  if (!pairs.empty()) {
    if (pairs.size() % 2 != 0) {
      throw std::invalid_argument("update_linear_objective: free_variable_pairs size is not even.");
    }
    std::size_t extra = pairs.size() / 2;
    presolved.resize(presolved.size() + extra);
    for (std::size_t k = 0; k < extra; ++k) {
      int u                                  = pairs[2 * k];
      int v                                  = pairs[2 * k + 1];
      presolved[static_cast<std::size_t>(v)] = -presolved[static_cast<std::size_t>(u)];
    }
  }

  if (static_cast<int>(presolved.size()) != xf.barrier_lp->num_cols ||
      xf.column_scales.size() != presolved.size()) {
    throw std::invalid_argument(
      "update_linear_objective: crushed objective size does not match barrier columns / column_scales.");
  }
  for (std::size_t j = 0; j < presolved.size(); ++j) {
    presolved[j] /= xf.column_scales[j];
  }
  return presolved;
}

}  // namespace cuopt::mathematical_optimization
