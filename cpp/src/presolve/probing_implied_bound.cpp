/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <presolve/probing_implied_bound.hpp>

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

namespace cuopt::mathematical_optimization::mip {

namespace {
template <typename i_t>
i_t remap_probing_variable(i_t raw_idx, const std::vector<i_t>& reverse_original_ids)
{
  if (reverse_original_ids.empty()) { return raw_idx; }
  if (raw_idx < 0 || raw_idx >= static_cast<i_t>(reverse_original_ids.size())) { return -1; }
  return reverse_original_ids[raw_idx];
}

}  // namespace

// Extract probing cache into CPU-only CSR struct for implied bounds cuts
template <typename i_t, typename f_t>
void extract_probing_implied_bounds(
  const std::vector<i_t>& reverse_original_ids,
  const simplex::user_problem_t<i_t, f_t>& branch_and_bound_problem,
  const probing_cache_t<i_t, f_t>& probing_cache,
  probing_implied_bound_t<i_t, f_t>& probing_implied_bound)
{
  auto& pc              = probing_cache.probing_cache;
  const i_t num_cols    = branch_and_bound_problem.num_cols;
  probing_implied_bound = probing_implied_bound_t<i_t, f_t>(num_cols);

  // First pass: count entries per binary variable
  // Probing cache indices are in pre-trivial-presolve space; remap to post-presolve (B&B) space
  for (auto& [var_idx, entries] : pc) {
    i_t j = remap_probing_variable(var_idx, reverse_original_ids);
    if (j < 0 || j >= num_cols) { continue; }
    if (branch_and_bound_problem.lower[j] != 0.0 || branch_and_bound_problem.upper[j] != 1.0) {
      continue;
    }

    // The probed interval, not the slot, decides the value of x_j it covers: {0} or (-inf, 0] for
    // x_j = 0, {1} or [1, inf) for x_j = 1.
    for (auto& entry : entries) {
      if (entry.var_to_cached_bound_map.empty()) { continue; }
      const auto& probe = entry.val_interval;
      const bool zero   = probe.interval_type != interval_type_t::GEQ && probe.val == 0;
      const bool one    = probe.interval_type != interval_type_t::LEQ && probe.val == 1;
      if (!zero && !one) { continue; }
      auto& offsets = zero ? probing_implied_bound.zero_offsets : probing_implied_bound.one_offsets;

      for (auto& [imp_var, bound] : entry.var_to_cached_bound_map) {
        i_t i = remap_probing_variable(imp_var, reverse_original_ids);
        if (i < 0 || i >= num_cols) { continue; }
        if (bound.ub < branch_and_bound_problem.lower[i] - 1e-6 ||
            bound.lb > branch_and_bound_problem.upper[i] + 1e-6) {
          continue;
        }
        offsets[j + 1]++;
      }
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
    i_t j = remap_probing_variable(var_idx, reverse_original_ids);
    if (j < 0 || j >= num_cols) { continue; }
    if (branch_and_bound_problem.lower[j] != 0.0 || branch_and_bound_problem.upper[j] != 1.0) {
      continue;
    }

    for (auto& entry : entries) {
      if (entry.var_to_cached_bound_map.empty()) { continue; }
      const auto& probe = entry.val_interval;
      const bool zero   = probe.interval_type != interval_type_t::GEQ && probe.val == 0;
      const bool one    = probe.interval_type != interval_type_t::LEQ && probe.val == 1;
      if (!zero && !one) { continue; }
      auto& cursor = zero ? zero_cursor : one_cursor;
      auto& variables =
        zero ? probing_implied_bound.zero_variables : probing_implied_bound.one_variables;
      auto& lower =
        zero ? probing_implied_bound.zero_lower_bound : probing_implied_bound.one_lower_bound;
      auto& upper =
        zero ? probing_implied_bound.zero_upper_bound : probing_implied_bound.one_upper_bound;

      for (auto& [imp_var, bound] : entry.var_to_cached_bound_map) {
        i_t i = remap_probing_variable(imp_var, reverse_original_ids);
        if (i < 0 || i >= num_cols) { continue; }
        if (bound.ub < branch_and_bound_problem.lower[i] - 1e-6 ||
            bound.lb > branch_and_bound_problem.upper[i] + 1e-6) {
          continue;
        }
        i_t p        = cursor[j]++;
        variables[p] = i;
        lower[p]     = bound.lb;
        upper[p]     = bound.ub;
      }
    }
  }

  CUOPT_LOG_INFO("\nProbing implied bounds: %d zero entries, %d one entries", zero_nnz, one_nnz);
}

#if MIP_INSTANTIATE_FLOAT
template void extract_probing_implied_bounds<int, float>(const std::vector<int>&,
                                                         const simplex::user_problem_t<int, float>&,
                                                         const probing_cache_t<int, float>&,
                                                         probing_implied_bound_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void extract_probing_implied_bounds<int, double>(
  const std::vector<int>&,
  const simplex::user_problem_t<int, double>&,
  const probing_cache_t<int, double>&,
  probing_implied_bound_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
