/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>
#include <mip_heuristics/presolve/probing_cache.hpp>
#include <utilities/logger.hpp>

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
  const problem_t<i_t, f_t>& op_problem,
  const simplex::user_problem_t<i_t, f_t>& branch_and_bound_problem,
  const probing_cache_t<i_t, f_t>& probing_cache,
  probing_implied_bound_t<i_t, f_t>& probing_implied_bound)

{
  auto& pc              = probing_cache.probing_cache;
  const i_t num_cols    = branch_and_bound_problem.num_cols;
  probing_implied_bound = probing_implied_bound_t<i_t, f_t>(num_cols);

  // First pass: count entries per binary variable
  // Probing cache indices are in pre-trivial-presolve space; remap to post-presolve (B&B) space
  auto& rev_ids = op_problem.reverse_original_ids;
  i_t rev_size  = static_cast<i_t>(rev_ids.size());
  auto remap    = [&](i_t raw_idx) -> i_t {
    if (rev_size == 0) return raw_idx;
    if (raw_idx < 0 || raw_idx >= rev_size) return -1;
    return rev_ids[raw_idx];
  };
  auto is_bb_binary = [&](i_t j) {
    return branch_and_bound_problem.lower[j] == 0.0 && branch_and_bound_problem.upper[j] == 1.0;
  };
  auto bb_bounds_consistent = [&](i_t i, f_t b_lb, f_t b_ub) {
    return b_ub >= branch_and_bound_problem.lower[i] - 1e-6 &&
           b_lb <= branch_and_bound_problem.upper[i] + 1e-6;
  };
  for (auto& [var_idx, entries] : pc) {
    if (entries[0].val_interval.interval_type != interval_type_t::EQUALS) { continue; }
    i_t j = remap(var_idx);
    if (j < 0 || j >= num_cols) { continue; }
    if (!is_bb_binary(j)) { continue; }

    for (auto& [imp_var, bound] : entries[0].var_to_cached_bound_map) {
      i_t i = remap(imp_var);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub)) { continue; }
      probing_implied_bound.zero_offsets[j + 1]++;
    }
    for (auto& [imp_var, bound] : entries[1].var_to_cached_bound_map) {
      i_t i = remap(imp_var);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub)) { continue; }
      probing_implied_bound.one_offsets[j + 1]++;
    }
  }

  // Prefix sum
  for (i_t j = 0; j < num_cols; j++) {
    probing_implied_bound.zero_offsets[j + 1] += probing_implied_bound.zero_offsets[j];
    probing_implied_bound.one_offsets[j + 1] += probing_implied_bound.one_offsets[j];
  }

  // Allocate flat arrays
  i_t zero_nnz = probing_implied_bound.zero_offsets[num_cols];
  i_t one_nnz  = probing_implied_bound.one_offsets[num_cols];
  probing_implied_bound.zero_variables.resize(zero_nnz);
  probing_implied_bound.zero_lower_bound.resize(zero_nnz);
  probing_implied_bound.zero_upper_bound.resize(zero_nnz);
  probing_implied_bound.one_variables.resize(one_nnz);
  probing_implied_bound.one_lower_bound.resize(one_nnz);
  probing_implied_bound.one_upper_bound.resize(one_nnz);

  // Second pass: fill flat arrays using write cursors
  std::vector<i_t> zero_cursor(probing_implied_bound.zero_offsets);
  std::vector<i_t> one_cursor(probing_implied_bound.one_offsets);

  for (auto& [var_idx, entries] : pc) {
    if (entries[0].val_interval.interval_type != interval_type_t::EQUALS) { continue; }
    i_t j = remap(var_idx);
    if (j < 0 || j >= num_cols) { continue; }
    if (!is_bb_binary(j)) { continue; }

    for (auto& [imp_var, bound] : entries[0].var_to_cached_bound_map) {
      i_t i = remap(imp_var);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub)) { continue; }
      i_t p                                     = zero_cursor[j]++;
      probing_implied_bound.zero_variables[p]   = i;
      probing_implied_bound.zero_lower_bound[p] = bound.lb;
      probing_implied_bound.zero_upper_bound[p] = bound.ub;
    }
    for (auto& [imp_var, bound] : entries[1].var_to_cached_bound_map) {
      i_t i = remap(imp_var);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub)) { continue; }
      i_t p                                    = one_cursor[j]++;
      probing_implied_bound.one_variables[p]   = i;
      probing_implied_bound.one_lower_bound[p] = bound.lb;
      probing_implied_bound.one_upper_bound[p] = bound.ub;
    }
  }

  CUOPT_LOG_INFO("\nProbing implied bounds: %d zero entries, %d one entries", zero_nnz, one_nnz);
}

}  // namespace cuopt::mathematical_optimization::mip
