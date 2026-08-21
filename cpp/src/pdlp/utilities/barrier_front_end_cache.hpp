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

namespace cuopt::cython {

/**
 * Convert / presolve / scaling state retained on barrier_cache_t after Optimal.
 * Enough to crush a new user-space linear objective into barrier space (C) and to
 * uncrush a solution (D) without rerunning those algorithms.
 */
struct barrier_front_end_cache_t {
  bool c_dirty{false};

  int user_num_cols{0};
  int user_num_rows{0};
  int original_num_cols{0};
  int original_num_rows{0};
  int barrier_num_cols{0};
  int barrier_num_rows{0};
  double obj_scale{1.0};
  double obj_constant{0.0};

  cuopt::mathematical_optimization::simplex::presolve_info_t<int, double> presolve_info;
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  // Barrier linear objective minus crush(user c) from the first solve (Q·ℓ shift, etc.).
  std::vector<double> linear_obj_shift;
  std::unique_ptr<cuopt::mathematical_optimization::simplex::lp_problem_t<int, double>> barrier_lp;
};

inline std::vector<double> crush_user_linear_objective(barrier_front_end_cache_t const& fe,
                                                       double const* c,
                                                       int n)
{
  if (c == nullptr || n != fe.user_num_cols) {
    throw std::invalid_argument(
      "update_q: linear objective length must match the cached user column count.");
  }
  if (fe.original_num_cols < fe.user_num_cols) {
    throw std::invalid_argument("update_q: cached original column count is smaller than user n.");
  }

  std::vector<double> orig(static_cast<std::size_t>(fe.original_num_cols), 0.0);
  for (int j = 0; j < n; ++j) {
    orig[static_cast<std::size_t>(j)] = c[j];
  }
  for (int j : fe.presolve_info.negated_variables) {
    orig[static_cast<std::size_t>(j)] *= -1.0;
  }

  std::vector<double> presolved;
  if (!fe.presolve_info.remaining_variables.empty()) {
    presolved.resize(fe.presolve_info.remaining_variables.size());
    for (std::size_t k = 0; k < fe.presolve_info.remaining_variables.size(); ++k) {
      presolved[k] = orig[static_cast<std::size_t>(fe.presolve_info.remaining_variables[k])];
    }
  } else {
    presolved = std::move(orig);
  }

  auto const& pairs = fe.presolve_info.free_variable_pairs;
  if (!pairs.empty()) {
    if (pairs.size() % 2 != 0) {
      throw std::invalid_argument("update_q: free_variable_pairs size is not even.");
    }
    std::size_t extra = pairs.size() / 2;
    presolved.resize(presolved.size() + extra);
    for (std::size_t k = 0; k < extra; ++k) {
      int u = pairs[2 * k];
      int v = pairs[2 * k + 1];
      presolved[static_cast<std::size_t>(v)] = -presolved[static_cast<std::size_t>(u)];
    }
  }

  if (static_cast<int>(presolved.size()) != fe.barrier_num_cols ||
      fe.column_scales.size() != presolved.size()) {
    throw std::invalid_argument(
      "update_q: crushed objective size does not match barrier columns / column_scales.");
  }
  for (std::size_t j = 0; j < presolved.size(); ++j) {
    presolved[j] /= fe.column_scales[j];
  }
  return presolved;
}

}  // namespace cuopt::cython
