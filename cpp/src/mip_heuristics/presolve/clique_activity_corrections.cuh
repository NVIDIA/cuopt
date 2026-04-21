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

// Debug: when set to 1, kernels will printf every time clique-aware activity
// tightening fires. Off by default; flip to 1 locally (or add -D on the
// compile line) to see trace output. Produces one line per group with a
// non-trivial correction and one line per (var, cnst) where the clique-aware
// bound is strictly tighter than what the stock formula would produce.
#ifndef CUOPT_DEBUG_CLIQUE_TIGHTENING
#define CUOPT_DEBUG_CLIQUE_TIGHTENING 0
#endif

namespace cuopt::linear_programming::detail {

// -----------------------------------------------------------------------------
// clique_group_table_t
//
// Static (per-(problem, clique_table)) CSR describing the non-overlapping
// groups of binary variables that enable clique-aware activity tightening.
// For each constraint, one or more groups are stored; within each group, at
// most one variable can be 1 (clique constraint), so their joint contribution
// to activity is the single best member rather than the sum of all members.
//
// This struct is built once on the host from clique_table_t and copied to the
// device. It holds NO per-(probe, iteration) state — the dynamic correction
// values (per-group min/max pos/neg and the resulting activity corrections)
// live on bounds_update_data_t since they share the same lifetime as
// min_activity/max_activity and are naturally per-probe.
//
// Groups are sorted by constraint_id so all groups for a constraint are
// contiguous. This enables deterministic per-constraint summation (no FP
// atomicAdd, which would be non-associative).
// -----------------------------------------------------------------------------
template <typename i_t, typename f_t>
struct clique_group_table_t {
  struct view_t {
    raft::device_span<const i_t> group_constraint_ids;
    raft::device_span<const i_t> group_member_offsets;
    raft::device_span<const i_t> group_member_vars;
    raft::device_span<const f_t> group_member_coeffs;
    raft::device_span<const i_t> constraint_group_offsets;
    raft::device_span<const i_t> reverse_group_id;
    // Parallel to reverse_group_id (indexed on reverse-CSR slot). Holds the
    // literal sign of the member for this (var, cnst) entry: +1 for positive
    // literal, -1 for complement literal, 0 when reverse_group_id[pos] == -1
    // (not a clique-group member for this cnst). Needed by the per-var
    // adjustment because the group's top-2 stats live on the EFFECTIVE literal
    // coefficient b_j = sign_j * a_j, while the per-nnz raw coeff is a_j.
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

  // Construct the group table by analyzing which constraints share ≥2 members
  // of any clique. Greedy non-overlapping partition per constraint: explicit
  // large/addtl cliques are tried first (largest first), then remaining
  // unassigned binaries are checked against adj_list_small_cliques for
  // maximal small-clique extraction.
  //
  // Sources of cliques consumed:
  //   - clique_table.first          (large explicit cliques, size ≥ min_clique_size)
  //   - clique_table.addtl_cliques  (= {vertex_idx} ∪ first[clique_idx][start_pos:])
  //   - clique_table.adj_list_small_cliques  (pairwise conflict edges; small
  //     cliques are rebuilt per constraint via greedy maximal-clique search)
  //
  // Requires a populated problem_t<i_t, f_t> (with reverse CSR) and a non-null
  // host clique table. Returns silently with n_groups=0 if no (constraint,
  // clique) pair has ≥2 members.
  //
  // Literal handling: conflict-graph vertices are literals (v < n_vars is the
  // positive literal of var v, v >= n_vars is the complement). Cliques may
  // freely mix both polarities. Each group member stores its underlying var
  // in group_member_vars and the *effective literal coefficient*
  // `sign_j * a_j` (where sign_j is +1 for positive, -1 for complement) in
  // group_member_coeffs. This choice is what lets the downstream correction
  // kernel use the same formula for positive and complement literals — the
  // constant `sum_{j in Q-} a_j` offset cancels in stock-minus-true.
  // `primary_reverse_original_ids` is the primary problem's
  // reverse_original_ids (N→M map). Only read on the sub-problem path; may be
  // empty when `problem` itself is in clique-build space.
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

// -----------------------------------------------------------------------------
// Kernel: compute per-group correction values and top-2 stats.
//
// Launch <<<n_groups, raft::WarpSize>>>. One warp per group. Threads stride
// over the group's members, each maintaining a thread-local (best, second)
// pair, then a butterfly warp reduction merges sums and top-2 pairs into a
// single (best, second) on every lane. Lane 0 writes the results.
//
// No atomics — each thread writes only to its own group's slot. Deterministic.
//
// TPB is a template argument to match the file's existing style; it is
// statically required to equal raft::WarpSize so every block is exactly one
// warp.
// -----------------------------------------------------------------------------
template <typename i_t, typename f_t, i_t TPB>
__global__ void compute_clique_corrections_kernel(raft::device_span<const i_t> group_member_offsets,
                                                  raft::device_span<const i_t> group_member_vars,
                                                  raft::device_span<const f_t> group_member_coeffs,
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
  const i_t mem_begin = group_member_offsets[gid];
  const i_t mem_end   = group_member_offsets[gid + 1];
  cuopt_assert(mem_begin <= mem_end, "group member offsets not monotonic");
  cuopt_assert(mem_end <= (i_t)group_member_vars.size(), "group member offsets exceed vars array");
  cuopt_assert(group_member_vars.size() == group_member_coeffs.size(),
               "group member vars/coeffs size mismatch");

  // Thread-local accumulators. For each lane, maintain (best, second-best).
  f_t sum_pos = 0, sum_neg = 0;
  f_t max1 = 0, max2 = 0;  // top-2 of max(0, coeff)  (max1 ≥ max2)
  f_t min1 = 0, min2 = 0;  // top-2 of min(0, coeff)  (min1 ≤ min2, most negative first)
  i_t n_unfixed = 0;

  // Strided scan over members
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

    // Insert `pos` into (max1, max2)
    if (pos > max1) {
      max2 = max1;
      max1 = pos;
    } else if (pos > max2) {
      max2 = pos;
    }
    // Insert `neg` into (min1, min2)
    if (neg < min1) {
      min2 = min1;
      min1 = neg;
    } else if (neg < min2) {
      min2 = neg;
    }
  }

  // Butterfly warp reduction. After the loop every lane holds the group's
  // fully-reduced values. Top-2 merge formula for sorted pairs (a1≥a2),
  // (b1≥b2):
  //   new1 = max(a1, b1)
  //   new2 = max(min(a1, b1), max(a2, b2))
#pragma unroll
  for (int off = TPB / 2; off > 0; off >>= 1) {
    sum_pos += __shfl_xor_sync(0xffffffff, sum_pos, off);
    sum_neg += __shfl_xor_sync(0xffffffff, sum_neg, off);
    n_unfixed += __shfl_xor_sync(0xffffffff, n_unfixed, off);

    // Merge top-2 (max side)
    f_t b1       = __shfl_xor_sync(0xffffffff, max1, off);
    f_t b2       = __shfl_xor_sync(0xffffffff, max2, off);
    f_t new_max1 = fmax(max1, b1);
    f_t new_max2 = fmax(fmin(max1, b1), fmax(max2, b2));
    max1         = new_max1;
    max2         = new_max2;

    // Merge top-2 (min side) — symmetric
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
    f_t max_corr              = sum_pos - max1;  // ≥ 0
    f_t min_corr              = sum_neg - min1;  // ≤ 0
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

// -----------------------------------------------------------------------------
// Kernel: fold per-group corrections into per-constraint activities.
//
// One thread per constraint. Each thread loops over its constraint's groups
// (contiguous range in the sorted group array) and subtracts their corrections
// from the activity. Summation order is fixed (ascending group index), so the
// FP arithmetic is bit-exact deterministic.
//
// MUST be gated on the SAME `changed_constraints[c]` flag that
// calc_activity_kernel uses. Rationale: calc_activity_kernel early-returns
// when the constraint is "not changed", leaving min/max_activity[c] at their
// previous-iteration values, which were ALREADY clique-corrected. If we
// unconditionally subtract the fresh correction here, constraints that stay
// untouched across iterations get their activity double-corrected, which
// compounds and drives max_activity below cnst_lb → spurious "infeasible in
// presolve". When changed_constraints[c] == 0, no member bound changed, so
// the correction is identical anyway and the already-corrected stored value
// is still valid — skipping is exact, not approximate.
// -----------------------------------------------------------------------------
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
  // Must match calc_activity_kernel's gate exactly. See comment above.
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
