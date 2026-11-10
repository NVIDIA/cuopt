/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>
#include <presolve/probing_cache.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct probing_implied_bound_t {
  // Probing implications stored in CSR format, indexed by binary variable x_j.
  //
  // "zero" = implications discovered when probing x_j = 0.
  // "one"  = implications discovered when probing x_j = 1.
  //
  // For a binary variable x_j, the range
  //   zero_offsets[j] .. zero_offsets[j+1]
  // indexes into the flat arrays zero_variables, zero_lower_bound, zero_upper_bound.
  //
  // For each position p in that range:
  //   zero_variables[p]    = i if variable y_i bounds were tightened
  //                          when x_j was fixed to 0 and constraints were propagated.
  //   zero_lower_bound[p]  = tightened lower bound on y_i (i.e., x_j = 0  =>  y_i >=
  //   zero_lower_bound[p]). zero_upper_bound[p]  = tightened upper bound on y_i (i.e., x_j = 0  =>
  //   y_i <= zero_upper_bound[p]).
  //
  // The one arrays are analogous for probing x_j = 1.
  //
  // Non-binary variables have empty ranges (zero_offsets[j] == zero_offsets[j+1]).
  // Offsets vectors have size num_cols + 1.

  probing_implied_bound_t() = default;

  probing_implied_bound_t(i_t num_cols)
    : zero_offsets(num_cols + 1, 0), one_offsets(num_cols + 1, 0)
  {
  }

  std::vector<i_t> zero_offsets;
  std::vector<i_t> zero_variables;
  std::vector<f_t> zero_lower_bound;
  std::vector<f_t> zero_upper_bound;

  std::vector<i_t> one_offsets;
  std::vector<i_t> one_variables;
  std::vector<f_t> one_lower_bound;
  std::vector<f_t> one_upper_bound;
};

// Extract probing cache into CPU-only CSR struct for implied bounds cuts
template <typename i_t, typename f_t>
void extract_probing_implied_bounds(
  const std::vector<i_t>& reverse_original_ids,
  const simplex::user_problem_t<i_t, f_t>& branch_and_bound_problem,
  const probing_cache_t<i_t, f_t>& probing_cache,
  probing_implied_bound_t<i_t, f_t>& probing_implied_bound);

}  // namespace cuopt::mathematical_optimization::mip
