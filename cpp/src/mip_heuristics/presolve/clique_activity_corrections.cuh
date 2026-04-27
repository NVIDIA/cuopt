/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/device_span.hpp>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/pair.h>

#include <mip_heuristics/presolve/bounds_update_data.cuh>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

// Set to 1 to printf when clique-aware activity tightening fires.
#ifndef CUOPT_DEBUG_CLIQUE_TIGHTENING
#define CUOPT_DEBUG_CLIQUE_TIGHTENING 0
#endif

namespace cuopt::linear_programming::detail {

// Static CSR of non-overlapping clique groups per constraint. Built once on the
// host from clique_table_t and copied to device. Groups are sorted by
// constraint_id for deterministic per-constraint summation. Dynamic correction
// values live on bounds_update_data_t (per-probe lifetime, see
// resize_clique_buffers).
template <typename i_t, typename f_t>
struct clique_group_table_t {
  struct view_t {
    raft::device_span<const i_t> group_constraint_ids;
    raft::device_span<const i_t> group_member_offsets;
    raft::device_span<const i_t> group_member_vars;
    raft::device_span<const f_t> group_member_coeffs;
    raft::device_span<const i_t> constraint_group_offsets;
    raft::device_span<const i_t> reverse_group_id;
    // Parallel to reverse_group_id: literal sign per (var, cnst) slot.
    // +1 / -1 when group member, 0 otherwise. Top-2 stats live on the
    // effective literal coeff b_j = sign_j * a_j, so the per-var adjustment
    // needs the sign separately from the raw coeff.
    raft::device_span<const i_t> reverse_member_sign;
    i_t n_groups;
  };

  explicit clique_group_table_t(rmm::cuda_stream_view stream)
    : group_constraint_ids(0, stream),
      group_member_offsets(0, stream),
      group_member_vars(0, stream),
      group_member_coeffs(0, stream),
      constraint_group_offsets(0, stream),
      reverse_group_id(0, stream),
      reverse_member_sign(0, stream)
  {
  }

  // Greedy non-overlapping partition of each constraint's binary members into
  // clique groups: explicit large/addtl cliques first (largest first), then
  // remaining unassigned binaries against small_clique_adj. Members store the
  // effective literal coeff b_j = sign_j * a_j so the kernel handles positive
  // and complement literals uniformly. `primary_reverse_original_ids` is the
  // primary problem's N→M map for the sub-problem path; empty otherwise.
  void build_from_host(problem_t<i_t, f_t>& problem,
                       const std::vector<i_t>& primary_reverse_original_ids,
                       clique_table_t<i_t, f_t>& clique_table);

  view_t view();

  bool empty() const noexcept { return n_groups == 0; }

  // Static (built once)
  rmm::device_uvector<i_t> group_constraint_ids;
  rmm::device_uvector<i_t> group_member_offsets;
  rmm::device_uvector<i_t> group_member_vars;
  rmm::device_uvector<f_t> group_member_coeffs;
  rmm::device_uvector<i_t> constraint_group_offsets;
  rmm::device_uvector<i_t> reverse_group_id;
  rmm::device_uvector<i_t> reverse_member_sign;

  i_t n_groups{0};
};

// One warp per group. Each thread strides over members maintaining (best,
// second), then a butterfly warp reduction merges sums and top-2 pairs.
// Lane 0 writes the results. No atomics, deterministic.
//
// Skip when changed_constraints[c] == 0: the previously written corrections
// are still exact (no member bound changed). Both downstream consumers
// (apply_clique_corrections_to_activity_kernel, update_bounds_per_cnst_cliq)
// gate on the same flag, so unread slots are never observed stale.
template <typename i_t, typename f_t, i_t TPB>
__global__ void compute_clique_corrections_kernel(raft::device_span<const i_t> group_member_offsets,
                                                  raft::device_span<const i_t> group_member_vars,
                                                  raft::device_span<const f_t> group_member_coeffs,
                                                  raft::device_span<const i_t> group_constraint_ids,
                                                  raft::device_span<const i_t> changed_constraints,
                                                  raft::device_span<const f_t> lb,
                                                  raft::device_span<const f_t> ub,
                                                  raft::device_span<f_t> group_max_correction,
                                                  raft::device_span<f_t> group_min_correction,
                                                  raft::device_span<f_t> group_max_pos,
                                                  raft::device_span<f_t> group_second_max_pos,
                                                  raft::device_span<f_t> group_min_neg,
                                                  raft::device_span<f_t> group_second_min_neg,
                                                  f_t int_tol)
{
  static_assert(TPB == raft::WarpSize,
                "compute_clique_corrections_kernel requires exactly one warp per block");

  const i_t gid = blockIdx.x;
  cuopt_assert(gid + 1 < (i_t)group_member_offsets.size(), "group id out of range");

  // Skip clean constraint (uniform branch across the warp).
  cuopt_assert(gid < (i_t)group_constraint_ids.size(), "group_constraint_ids index out of range");
  const i_t cnst = group_constraint_ids[gid];
  cuopt_assert(cnst >= 0 && cnst < (i_t)changed_constraints.size(),
               "group constraint id out of range");
  if (changed_constraints[cnst] == 0) return;

  const i_t mem_begin = group_member_offsets[gid];
  const i_t mem_end   = group_member_offsets[gid + 1];
  cuopt_assert(mem_begin <= mem_end, "group member offsets not monotonic");
  cuopt_assert(mem_end <= (i_t)group_member_vars.size(), "group member offsets exceed vars array");
  cuopt_assert(group_member_vars.size() == group_member_coeffs.size(),
               "group member vars/coeffs size mismatch");

  f_t sum_pos = 0, sum_neg = 0;
  f_t max1 = 0, max2 = 0;  // top-2 of max(0, coeff)  (max1 >= max2)
  f_t min1 = 0, min2 = 0;  // top-2 of min(0, coeff)  (min1 <= min2)
  i_t n_unfixed = 0;

  for (i_t m = mem_begin + threadIdx.x; m < mem_end; m += TPB) {
    i_t var = group_member_vars[m];
    f_t a   = group_member_coeffs[m];
    cuopt_assert(var >= 0 && var < (i_t)lb.size(), "clique member var index out of range");
    if (ub[var] - lb[var] <= int_tol) continue;  // fixed → skip

    n_unfixed++;
    f_t pos = fmax(a, f_t{0});
    f_t neg = fmin(a, f_t{0});
    sum_pos += pos;
    sum_neg += neg;

    if (pos > max1) {
      max2 = max1;
      max1 = pos;
    } else if (pos > max2) {
      max2 = pos;
    }
    if (neg < min1) {
      min2 = min1;
      min1 = neg;
    } else if (neg < min2) {
      min2 = neg;
    }
  }

  // Butterfly warp reduction; top-2 merge formula for sorted pairs (a1>=a2),
  // (b1>=b2): new1 = max(a1,b1), new2 = max(min(a1,b1), max(a2,b2)).
#pragma unroll
  for (int off = TPB / 2; off > 0; off >>= 1) {
    sum_pos += __shfl_xor_sync(0xffffffff, sum_pos, off);
    sum_neg += __shfl_xor_sync(0xffffffff, sum_neg, off);
    n_unfixed += __shfl_xor_sync(0xffffffff, n_unfixed, off);

    f_t b1       = __shfl_xor_sync(0xffffffff, max1, off);
    f_t b2       = __shfl_xor_sync(0xffffffff, max2, off);
    f_t new_max1 = fmax(max1, b1);
    f_t new_max2 = fmax(fmin(max1, b1), fmax(max2, b2));
    max1         = new_max1;
    max2         = new_max2;

    f_t d1       = __shfl_xor_sync(0xffffffff, min1, off);
    f_t d2       = __shfl_xor_sync(0xffffffff, min2, off);
    f_t new_min1 = fmin(min1, d1);
    f_t new_min2 = fmin(fmax(min1, d1), fmin(min2, d2));
    min1         = new_min1;
    min2         = new_min2;
  }

  if (threadIdx.x == 0) {
    if (n_unfixed < 2) {
      group_max_pos[gid]        = 0;
      group_second_max_pos[gid] = 0;
      group_min_neg[gid]        = 0;
      group_second_min_neg[gid] = 0;
      group_max_correction[gid] = 0;
      group_min_correction[gid] = 0;
      return;
    }
    cuopt_assert(max1 >= max2, "top-2 max invariant violated");
    cuopt_assert(min1 <= min2, "top-2 min invariant violated");
    cuopt_assert(sum_pos >= max1, "sum_pos < max1 is impossible");
    cuopt_assert(sum_neg <= min1, "sum_neg > min1 is impossible");
    group_max_pos[gid]        = max1;
    group_second_max_pos[gid] = max2;
    group_min_neg[gid]        = min1;
    group_second_min_neg[gid] = min2;
    f_t max_corr              = sum_pos - max1;  // >= 0
    f_t min_corr              = sum_neg - min1;  // <= 0
    group_max_correction[gid] = max_corr;
    group_min_correction[gid] = min_corr;
#if CUOPT_DEBUG_CLIQUE_TIGHTENING
    if (max_corr > f_t{0} || min_corr < f_t{0}) {
      printf(
        "[clique-corr] gid=%d n_unfixed=%d sum_pos=%.6f max1=%.6f max_corr=%.6f | "
        "sum_neg=%.6f min1=%.6f min_corr=%.6f\n",
        (int)gid,
        (int)n_unfixed,
        (double)sum_pos,
        (double)max1,
        (double)max_corr,
        (double)sum_neg,
        (double)min1,
        (double)min_corr);
    }
#endif
  }
}

// One thread per constraint, sums its groups' corrections in fixed order.
// MUST gate on the same changed_constraints flag as calc_activity_kernel:
// calc_activity_kernel skips clean constraints, leaving min/max_activity at
// their previous (already clique-corrected) values. Re-subtracting here would
// double-correct and compound across iterations.
template <typename i_t, typename f_t>
__global__ void apply_clique_corrections_to_activity_kernel(
  raft::device_span<const i_t> constraint_group_offsets,
  raft::device_span<const f_t> group_max_correction,
  raft::device_span<const f_t> group_min_correction,
  raft::device_span<const i_t> changed_constraints,
  raft::device_span<f_t> min_activity,
  raft::device_span<f_t> max_activity,
  i_t n_constraints)
{
  i_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_constraints) return;
  cuopt_assert(c + 1 < (i_t)constraint_group_offsets.size(),
               "constraint id out of range for group offsets");
  // Must match calc_activity_kernel's gate exactly.
  if (changed_constraints[c] == 0) return;

  i_t g_begin = constraint_group_offsets[c];
  i_t g_end   = constraint_group_offsets[c + 1];
  cuopt_assert(g_begin <= g_end, "constraint group offsets not monotonic");
  cuopt_assert(g_end <= (i_t)group_max_correction.size(),
               "constraint group offsets exceed group array");
  if (g_begin == g_end) return;

  f_t max_corr = 0, min_corr = 0;
  for (i_t g = g_begin; g < g_end; ++g) {
    cuopt_assert(group_max_correction[g] >= 0, "max correction must be non-negative");
    cuopt_assert(group_min_correction[g] <= 0, "min correction must be non-positive");
    max_corr += group_max_correction[g];
    min_corr += group_min_correction[g];
  }
  max_activity[c] -= max_corr;
  min_activity[c] -= min_corr;
}

}  // namespace cuopt::linear_programming::detail
