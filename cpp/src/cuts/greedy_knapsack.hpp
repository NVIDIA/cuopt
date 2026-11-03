/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

enum class greedy_knapsack_mode_t { SCAN_ALL_WITH_BEST_SINGLE, STRICT_RATIO_PREFIX };

template <typename i_t, typename f_t>
f_t greedy_knapsack_problem(
  const std::vector<f_t>& values,
  const std::vector<f_t>& weights,
  f_t rhs,
  std::vector<f_t>& solution,
  greedy_knapsack_mode_t mode = greedy_knapsack_mode_t::SCAN_ALL_WITH_BEST_SINGLE)
{
  i_t n = weights.size();
  solution.assign(n, 0.0);

  // Build permutation
  std::vector<i_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);

  std::vector<f_t> ratios;
  ratios.resize(n);
  for (i_t i = 0; i < n; i++) {
    ratios[i] = values[i] / weights[i];
  }

  // Sort by value / weight ratio
  std::sort(perm.begin(), perm.end(), [&](i_t i, i_t j) { return ratios[i] > ratios[j]; });

  if (mode == greedy_knapsack_mode_t::STRICT_RATIO_PREFIX) {
    // Wolter Algorithm 5.1 for the single-node-flow knapsack relaxation: take
    // the ratio-sorted prefix while the strict capacity remains satisfied and
    // stop at the first item that does not fit.
    f_t total_weight = 0.0;
    f_t total_value  = 0.0;
    for (i_t item : perm) {
      if (total_weight + weights[item] < rhs) {
        solution[item] = 1.0;
        total_weight += weights[item];
        total_value += values[item];
      } else {
        break;
      }
    }
    return total_value;
  }

  // Greedy select items with the best value / weight ratio until the remaining capacity is
  // exhausted
  f_t remaining   = rhs;
  f_t total_value = 0.0;

  for (i_t j : perm) {
    if (weights[j] <= remaining) {
      solution[j] = 1.0;
      remaining -= weights[j];
      total_value += values[j];
    }
  }

  // Best single-item fallback
  f_t best_single_value = 0.0;
  i_t best_single_idx   = -1;

  for (i_t j = 0; j < n; ++j) {
    if (weights[j] <= rhs && values[j] > best_single_value) {
      best_single_value = values[j];
      best_single_idx   = j;
    }
  }

  if (best_single_value > total_value) {
    solution.assign(n, 0.0);
    solution[best_single_idx] = 1.0;
    return best_single_value;
  }

  return total_value;
}

}  // namespace cuopt::mathematical_optimization::mip
