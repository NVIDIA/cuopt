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

template <typename i_t, typename f_t>
bool is_bb_binary(i_t j, const simplex::user_problem_t<i_t, f_t>& problem)
{
  return problem.lower[j] == 0.0 && problem.upper[j] == 1.0;
}

template <typename i_t, typename f_t>
bool bb_bounds_consistent(i_t i,
                          f_t lower,
                          f_t upper,
                          const simplex::user_problem_t<i_t, f_t>& problem)
{
  return upper >= problem.lower[i] - 1e-6 && lower <= problem.upper[i] + 1e-6;
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
    if (entries[0].val_interval.interval_type != interval_type_t::EQUALS) { continue; }
    i_t j = remap_probing_variable(var_idx, reverse_original_ids);
    if (j < 0 || j >= num_cols) { continue; }
    if (!is_bb_binary(j, branch_and_bound_problem)) { continue; }

    for (auto& [imp_var, bound] : entries[0].var_to_cached_bound_map) {
      i_t i = remap_probing_variable(imp_var, reverse_original_ids);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub, branch_and_bound_problem)) { continue; }
      probing_implied_bound.zero_offsets[j + 1]++;
    }
    for (auto& [imp_var, bound] : entries[1].var_to_cached_bound_map) {
      i_t i = remap_probing_variable(imp_var, reverse_original_ids);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub, branch_and_bound_problem)) { continue; }
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
    i_t j = remap_probing_variable(var_idx, reverse_original_ids);
    if (j < 0 || j >= num_cols) { continue; }
    if (!is_bb_binary(j, branch_and_bound_problem)) { continue; }

    for (auto& [imp_var, bound] : entries[0].var_to_cached_bound_map) {
      i_t i = remap_probing_variable(imp_var, reverse_original_ids);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub, branch_and_bound_problem)) { continue; }
      i_t p                                     = zero_cursor[j]++;
      probing_implied_bound.zero_variables[p]   = i;
      probing_implied_bound.zero_lower_bound[p] = bound.lb;
      probing_implied_bound.zero_upper_bound[p] = bound.ub;
    }
    for (auto& [imp_var, bound] : entries[1].var_to_cached_bound_map) {
      i_t i = remap_probing_variable(imp_var, reverse_original_ids);
      if (i < 0 || i >= num_cols) { continue; }
      if (!bb_bounds_consistent(i, bound.lb, bound.ub, branch_and_bound_problem)) { continue; }
      i_t p                                    = one_cursor[j]++;
      probing_implied_bound.one_variables[p]   = i;
      probing_implied_bound.one_lower_bound[p] = bound.lb;
      probing_implied_bound.one_upper_bound[p] = bound.ub;
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
