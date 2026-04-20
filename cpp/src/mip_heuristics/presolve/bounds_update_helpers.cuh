/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <thrust/pair.h>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/utils.cuh>
#include "bounds_update_data.cuh"
#include "clique_activity_corrections.cuh"

namespace cuopt::linear_programming::detail {

// Activity calculation

template <typename f_t>
inline __device__ f_t min_act_of_var(f_t coeff, f_t var_lb, f_t var_ub)
{
  return (coeff < 0.) ? coeff * var_ub : coeff * var_lb;
}

template <typename f_t>
inline __device__ f_t max_act_of_var(f_t coeff, f_t var_lb, f_t var_ub)
{
  return (coeff < 0.) ? coeff * var_lb : coeff * var_ub;
}

template <typename f_t>
inline __device__ f_t update_lb(f_t curr_lb, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_min_act / coeff : delta_max_act / coeff;
  return max(curr_lb, comp_bnd);
}

template <typename f_t>
inline __device__ f_t update_ub(f_t curr_ub, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_max_act / coeff : delta_min_act / coeff;
  return min(curr_ub, comp_bnd);
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void calc_activity_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_0,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_1)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t cnst_idx    = blockIdx.x;
  i_t cnst_offset = pb.offsets[cnst_idx];
  i_t cnst_degree = pb.offsets[cnst_idx + 1] - cnst_offset;
  f_t min_act_0 = 0, max_act_0 = 0;
  f_t min_act_1 = 0, max_act_1 = 0;
  bool changed_0 = upd_0.changed_constraints[cnst_idx] == 1;
  bool changed_1 = upd_1.changed_constraints[cnst_idx] == 1;
  if (!changed_0 && !changed_1) { return; }

  for (i_t i = threadIdx.x; i < cnst_degree; i += blockDim.x) {
    auto coeff   = pb.coefficients[cnst_offset + i];
    auto var_idx = pb.variables[cnst_offset + i];
    if (changed_0) {
      auto var_lb_0 = upd_0.lb[var_idx];
      auto var_ub_0 = upd_0.ub[var_idx];
      min_act_0 += min_act_of_var(coeff, var_lb_0, var_ub_0);
      max_act_0 += max_act_of_var(coeff, var_lb_0, var_ub_0);
      atomicExch(&upd_0.changed_variables[var_idx], 1);
    }
    if (changed_1) {
      auto var_lb_1 = upd_1.lb[var_idx];
      auto var_ub_1 = upd_1.ub[var_idx];
      min_act_1 += min_act_of_var(coeff, var_lb_1, var_ub_1);
      max_act_1 += max_act_of_var(coeff, var_lb_1, var_ub_1);
      atomicExch(&upd_1.changed_variables[var_idx], 1);
    }
  }
  if (changed_0) {
    min_act_0 = BlockReduce(temp_storage).Sum(min_act_0);
    __syncthreads();
    max_act_0 = BlockReduce(temp_storage).Sum(max_act_0);
    __syncthreads();
  }
  if (changed_1) {
    min_act_1 = BlockReduce(temp_storage).Sum(min_act_1);
    __syncthreads();
    max_act_1 = BlockReduce(temp_storage).Sum(max_act_1);
  }

  if (threadIdx.x == 0) {
    if (changed_0) {
      upd_0.min_activity[cnst_idx] = min_act_0;
      upd_0.max_activity[cnst_idx] = max_act_0;
    }
    if (changed_1) {
      upd_1.min_activity[cnst_idx] = min_act_1;
      upd_1.max_activity[cnst_idx] = max_act_1;
    }
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void calc_activity_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t cnst_idx    = blockIdx.x;
  i_t cnst_offset = pb.offsets[cnst_idx];
  i_t cnst_degree = pb.offsets[cnst_idx + 1] - cnst_offset;
  f_t min_act = 0, max_act = 0;
  if (upd.changed_constraints[cnst_idx] == 0) { return; }

  for (i_t i = threadIdx.x; i < cnst_degree; i += blockDim.x) {
    auto coeff   = pb.coefficients[cnst_offset + i];
    auto var_idx = pb.variables[cnst_offset + i];
    auto var_lb  = upd.lb[var_idx];
    auto var_ub  = upd.ub[var_idx];
    min_act += min_act_of_var(coeff, var_lb, var_ub);
    max_act += max_act_of_var(coeff, var_lb, var_ub);
    atomicExch(&upd.changed_variables[var_idx], 1);
  }
  min_act = BlockReduce(temp_storage).Sum(min_act);
  __syncthreads();
  max_act = BlockReduce(temp_storage).Sum(max_act);

  if (threadIdx.x == 0) {
    upd.min_activity[cnst_idx] = min_act;
    upd.max_activity[cnst_idx] = max_act;
  }
}

// Update bounds

template <typename i_t, typename f_t>
inline __device__ bool check_infeasibility(f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t eps)
{
  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
}

template <typename i_t, typename f_t>
inline __device__ bool check_infeasibility(
  f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t abs_tol, f_t rel_tol)
{
  auto eps = get_cstr_tolerance<i_t, f_t>(cnst_lb, cnst_ub, abs_tol, rel_tol);
  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
}

template <typename i_t, typename f_t>
inline __device__ bool check_redundancy(
  f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t abs_tol, f_t rel_tol)
{
  auto eps = get_cstr_tolerance<i_t, f_t>(cnst_lb, cnst_ub, abs_tol, rel_tol);
  return (min_a > cnst_lb + eps) && (max_a < cnst_ub - eps);
}

template <typename f_t>
inline __device__ bool skip_update(thrust::pair<f_t, f_t> bnd, f_t int_tol)
{
  return (thrust::get<0>(bnd) + int_tol >= thrust::get<1>(bnd));
}

template <typename i_t, typename f_t>
inline __device__ thrust::pair<f_t, f_t> update_bounds_per_cnst(
  typename problem_t<i_t, f_t>::view_t pb,
  f_t coeff,
  i_t cnst_idx,
  f_t cnst_lb,
  f_t cnst_ub,
  typename bounds_update_data_t<i_t, f_t>::view_t upd,
  thrust::pair<f_t, f_t> bnd,
  thrust::pair<f_t, f_t> old_bnd)
{
  auto min_a = upd.min_activity[cnst_idx];
  auto max_a = upd.max_activity[cnst_idx];
  // don't propagate over constraints that are infeasible
  if (check_infeasibility<i_t, f_t>(min_a,
                                    max_a,
                                    cnst_lb,
                                    cnst_ub,
                                    pb.tolerances.absolute_tolerance,
                                    pb.tolerances.relative_tolerance)) {
    return bnd;
  }
  min_a -= (coeff < 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  max_a -= (coeff > 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  auto delta_min_act  = cnst_ub - min_a;
  auto delta_max_act  = cnst_lb - max_a;
  thrust::get<0>(bnd) = update_lb(thrust::get<0>(bnd), coeff, delta_min_act, delta_max_act);
  thrust::get<1>(bnd) = update_ub(thrust::get<1>(bnd), coeff, delta_min_act, delta_max_act);
  return bnd;
}

template <typename i_t, typename f_t>
inline __device__ bool write_updated_bounds(typename problem_t<i_t, f_t>::view_t pb,
                                            i_t var_idx,
                                            bool is_int,
                                            typename bounds_update_data_t<i_t, f_t>::view_t upd,
                                            thrust::pair<f_t, f_t> bnd,
                                            thrust::pair<f_t, f_t> old_bnd)
{
  auto new_lb = thrust::get<0>(bnd);
  auto new_ub = thrust::get<1>(bnd);
  new_lb      = is_int ? ceil(new_lb - pb.tolerances.integrality_tolerance) : new_lb;
  new_ub      = is_int ? floor(new_ub + pb.tolerances.integrality_tolerance) : new_ub;

  auto lb_updated =
    (abs(new_lb - thrust::get<0>(old_bnd)) > 1e3 * pb.tolerances.absolute_tolerance);
  auto ub_updated =
    (abs(new_ub - thrust::get<1>(old_bnd)) > 1e3 * pb.tolerances.absolute_tolerance);

  if (lb_updated) { upd.lb[var_idx] = new_lb; }
  if (ub_updated) { upd.ub[var_idx] = new_ub; }

  if (lb_updated || ub_updated) { atomicAdd(upd.bounds_changed, 1); }
  // bounds_changed tracks the number of significantly changed bounds, we want any small change to
  // be detected
  if (new_lb != old_bnd.first || new_ub != old_bnd.second) { return true; }
  return false;
}

template <typename i_t, typename f_t, i_t BDIM>
__device__ void update_bounds(typename problem_t<i_t, f_t>::view_t pb,
                              i_t var_idx,
                              i_t var_offset,
                              i_t var_degree,
                              bool is_int,
                              typename bounds_update_data_t<i_t, f_t>::view_t upd_0,
                              thrust::pair<f_t, f_t> old_bnd_0,
                              typename bounds_update_data_t<i_t, f_t>::view_t upd_1,
                              thrust::pair<f_t, f_t> old_bnd_1)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto bnd_0        = old_bnd_0;
  auto bnd_1        = old_bnd_1;
  i_t var_changed_0 = upd_0.changed_variables[var_idx];
  i_t var_changed_1 = upd_1.changed_variables[var_idx];
  if (!var_changed_0 && !var_changed_1) { return; }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    auto a        = pb.reverse_coefficients[var_offset + i];
    auto cnst_ub  = pb.constraint_upper_bounds[cnst_idx];
    auto cnst_lb  = pb.constraint_lower_bounds[cnst_idx];
    if (var_changed_0) {
      bool cstr_changed_0 = upd_0.changed_constraints[cnst_idx] == 1;
      if (cstr_changed_0) {
        bnd_0 = update_bounds_per_cnst(pb, a, cnst_idx, cnst_lb, cnst_ub, upd_0, bnd_0, old_bnd_0);
      }
    }
    if (var_changed_1) {
      bool cstr_changed_1 = upd_1.changed_constraints[cnst_idx] == 1;
      if (cstr_changed_1) {
        bnd_1 = update_bounds_per_cnst(pb, a, cnst_idx, cnst_lb, cnst_ub, upd_1, bnd_1, old_bnd_1);
      }
    }
  }
  __syncthreads();
  if (var_changed_0) {
    thrust::get<0>(bnd_0) =
      BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd_0), cuda::maximum());
    __syncthreads();
    thrust::get<1>(bnd_0) =
      BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd_0), cuda::minimum());
    __syncthreads();
  }
  if (var_changed_1) {
    thrust::get<0>(bnd_1) =
      BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd_1), cuda::maximum());
    __syncthreads();
    thrust::get<1>(bnd_1) =
      BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd_1), cuda::minimum());
  }
  __shared__ bool changed_0, changed_1;
  if (threadIdx.x == 0) {
    changed_0 = write_updated_bounds(pb, var_idx, is_int, upd_0, bnd_0, old_bnd_0);
    changed_1 = write_updated_bounds(pb, var_idx, is_int, upd_1, bnd_1, old_bnd_1);
  }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    if (changed_0) { atomicExch(&upd_0.next_changed_constraints[cnst_idx], 1); }
    if (changed_1) { atomicExch(&upd_1.next_changed_constraints[cnst_idx], 1); }
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__device__ void update_bounds(typename problem_t<i_t, f_t>::view_t pb,
                              i_t var_idx,
                              i_t var_offset,
                              i_t var_degree,
                              bool is_int,
                              typename bounds_update_data_t<i_t, f_t>::view_t upd,
                              thrust::pair<f_t, f_t> old_bnd)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t var_changed = upd.changed_variables[var_idx];
  if (!var_changed) { return; }

  auto bnd = old_bnd;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    bool changed  = upd.changed_constraints[cnst_idx] == 1;
    if (!changed) { continue; }
    auto a       = pb.reverse_coefficients[var_offset + i];
    auto cnst_ub = pb.constraint_upper_bounds[cnst_idx];
    auto cnst_lb = pb.constraint_lower_bounds[cnst_idx];
    bnd          = update_bounds_per_cnst(pb, a, cnst_idx, cnst_lb, cnst_ub, upd, bnd, old_bnd);
  }

  thrust::get<0>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd), cuda::maximum());
  __syncthreads();
  thrust::get<1>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd), cuda::minimum());
  __shared__ bool changed;
  if (threadIdx.x == 0) { changed = write_updated_bounds(pb, var_idx, is_int, upd, bnd, old_bnd); }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    if (changed) { atomicExch(&upd.next_changed_constraints[cnst_idx], 1); }
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void update_bounds_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t var_idx    = blockIdx.x;
  i_t var_offset = pb.reverse_offsets[var_idx];
  i_t var_degree = pb.reverse_offsets[var_idx + 1] - var_offset;
  bool is_int    = (pb.variable_types[var_idx] == var_t::INTEGER);

  auto old_bnd = thrust::make_pair(upd.lb[var_idx], upd.ub[var_idx]);

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  auto skip = skip_update(old_bnd, pb.tolerances.integrality_tolerance);
  if (skip) {
    return;
  } else {
    update_bounds<i_t, f_t, BDIM>(pb, var_idx, var_offset, var_degree, is_int, upd, old_bnd);
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void update_bounds_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_0,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_1)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t var_idx    = blockIdx.x;
  i_t var_offset = pb.reverse_offsets[var_idx];
  i_t var_degree = pb.reverse_offsets[var_idx + 1] - var_offset;
  bool is_int    = (pb.variable_types[var_idx] == var_t::INTEGER);

  auto old_bnd_0 = thrust::make_pair(upd_0.lb[var_idx], upd_0.ub[var_idx]);
  auto old_bnd_1 = thrust::make_pair(upd_1.lb[var_idx], upd_1.ub[var_idx]);

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  auto skip_0 = skip_update(old_bnd_0, pb.tolerances.integrality_tolerance);
  auto skip_1 = skip_update(old_bnd_1, pb.tolerances.integrality_tolerance);
  if (skip_0 && skip_1) {
    return;
  } else if (skip_0) {
    update_bounds<i_t, f_t, BDIM>(pb, var_idx, var_offset, var_degree, is_int, upd_1, old_bnd_1);
  } else if (skip_1) {
    update_bounds<i_t, f_t, BDIM>(pb, var_idx, var_offset, var_degree, is_int, upd_0, old_bnd_0);
  } else {
    update_bounds<i_t, f_t, BDIM>(
      pb, var_idx, var_offset, var_degree, is_int, upd_0, old_bnd_0, upd_1, old_bnd_1);
  }
}

// -----------------------------------------------------------------------------
// Clique-aware single-update variants
//
// These mirror the single-update update_bounds_per_cnst / update_bounds /
// update_bounds_kernel above but apply a per-variable correction adjustment so
// that when we peel variable v off the (already clique-corrected) activity, we
// restore the correct group contribution "without v".
//
// Derivation of the adjustment (max side; min is symmetric):
//   Group g's top-2 stats (max1, max2, min1, min2) live on the EFFECTIVE
//   literal coefficient b_j = sign_j * a_j (so positive and complement
//   literals share the same formula — see clique_activity_corrections.cu).
//   When v is a member of group g with literal sign s_v, its effective coeff
//   is b_v = s_v * coeff_v, and:
//     pos_v_b = max(b_v, 0).  Group g's corrected max contribution is max1_g
//     (the largest pos_b among unfixed u ∈ g). Removing v from the group:
//       if pos_v_b == max1_g  →  group contribution becomes max2_g (second-best)
//       else                  →  group contribution stays max1_g
//     The stock formula subtracts pos_v_b from max_a_corrected. The adjustment
//     to reach activity_without_v is:
//       max_a += (pos_v_b >= max1_g) ? max2_g : pos_v_b
//     (The degenerate case n_unfixed<2 is handled by max1=max2=0 →
//     adjustment=0.)
//   NOTE: the subsequent "peel raw coeff" step still uses the raw a_v because
//   the activity itself was built from raw coefficients; only the top-2 math
//   is in literal space.
// -----------------------------------------------------------------------------
template <typename i_t, typename f_t>
inline __device__ thrust::pair<f_t, f_t> update_bounds_per_cnst_cliq(
  typename problem_t<i_t, f_t>::view_t pb,
  i_t var_idx,
  f_t coeff,
  i_t cnst_idx,
  f_t cnst_lb,
  f_t cnst_ub,
  typename bounds_update_data_t<i_t, f_t>::view_t upd,
  thrust::pair<f_t, f_t> bnd,
  thrust::pair<f_t, f_t> old_bnd,
  i_t group_id,
  i_t member_sign,
  typename clique_group_table_t<i_t, f_t>::view_t cliq)
{
  auto min_a = upd.min_activity[cnst_idx];
  auto max_a = upd.max_activity[cnst_idx];
  if (check_infeasibility<i_t, f_t>(min_a,
                                    max_a,
                                    cnst_lb,
                                    cnst_ub,
                                    pb.tolerances.absolute_tolerance,
                                    pb.tolerances.relative_tolerance)) {
    return bnd;
  }

#if CUOPT_DEBUG_CLIQUE_TIGHTENING
  // For the "stock" (no clique correction at all) comparison baseline we need
  // the RAW min/max activity as it would have been without
  // apply_clique_corrections_to_activity_kernel having subtracted the group
  // corrections. Undo those here by adding back the per-group corrections for
  // this constraint. Snapshotting now, before the per-var adjustment below,
  // keeps the arithmetic straightforward.
  f_t raw_min_a = min_a;
  f_t raw_max_a = max_a;
  {
    i_t g_begin = cliq.constraint_group_offsets[cnst_idx];
    i_t g_end   = cliq.constraint_group_offsets[cnst_idx + 1];
    for (i_t g = g_begin; g < g_end; ++g) {
      raw_max_a += upd.group_max_correction[g];
      raw_min_a += upd.group_min_correction[g];
    }
  }
#endif

  // Apply per-variable clique adjustment before peeling off v's contribution.
  // The top-2 stats live on upd (bounds_update_data_t), matching the per-probe
  // lifetime of min_activity/max_activity. They are computed in LITERAL space
  // (b_j = sign_j * a_j), so the adjustment must compare against
  // b_v = member_sign * coeff — NOT the raw coeff. Getting this wrong is a
  // silent source of over-tightening for cliques that contain complement
  // literals (member_sign == -1 with positive a_j, or vice versa).
  if (group_id >= 0) {
    cuopt_assert(group_id < cliq.n_groups, "clique group id out of range");
    cuopt_assert(member_sign == 1 || member_sign == -1,
                 "member_sign must be +/-1 when group_id >= 0");
    f_t b_v   = static_cast<f_t>(member_sign) * coeff;
    f_t pos_v = (b_v > 0) ? b_v : f_t{0};
    f_t neg_v = (b_v < 0) ? b_v : f_t{0};
    f_t max1  = upd.group_max_pos[group_id];
    f_t max2  = upd.group_second_max_pos[group_id];
    f_t min1  = upd.group_min_neg[group_id];
    f_t min2  = upd.group_second_min_neg[group_id];
    cuopt_assert(max1 >= max2, "top-2 max invariant violated (per-cnst)");
    cuopt_assert(min1 <= min2, "top-2 min invariant violated (per-cnst)");
    f_t max_adj = (pos_v >= max1) ? max2 : pos_v;
    f_t min_adj = (neg_v <= min1) ? min2 : neg_v;
    max_a += max_adj;
    min_a += min_adj;
  } else {
    cuopt_assert(member_sign == 0, "member_sign must be 0 when group_id == -1");
  }

  min_a -= (coeff < 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  max_a -= (coeff > 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  auto delta_min_act  = cnst_ub - min_a;
  auto delta_max_act  = cnst_lb - max_a;
  thrust::get<0>(bnd) = update_lb(thrust::get<0>(bnd), coeff, delta_min_act, delta_max_act);
  thrust::get<1>(bnd) = update_ub(thrust::get<1>(bnd), coeff, delta_min_act, delta_max_act);

#if CUOPT_DEBUG_CLIQUE_TIGHTENING
  // Genuine comparison: clique-aware per-cnst (lb, ub) vs the non-clique path
  // run on the RAW activity. Fires for EVERY (var, cnst) — clique-aware
  // tightening most often shows up on variables that are NOT themselves
  // members of any clique (the bound on y in the canonical "x0+x1+x2+y ≥ 2"
  // example), so we can't limit this to group_id >= 0.
  {
    f_t s_min_a = raw_min_a;
    f_t s_max_a = raw_max_a;
    s_min_a -= (coeff < 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
    s_max_a -= (coeff > 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
    f_t s_dmin    = cnst_ub - s_min_a;
    f_t s_dmax    = cnst_lb - s_max_a;
    f_t s_lb      = update_lb(thrust::get<0>(old_bnd), coeff, s_dmin, s_dmax);
    f_t s_ub      = update_ub(thrust::get<1>(old_bnd), coeff, s_dmin, s_dmax);
    f_t c_lb      = thrust::get<0>(bnd);
    f_t c_ub      = thrust::get<1>(bnd);
    const f_t eps = (f_t)1e-9;
    if (c_lb > s_lb + eps || c_ub < s_ub - eps) {
      printf(
        "[clique-tighten] var=%d cnst=%d group=%d coeff=%.6f | "
        "stock per-cnst: lb=%.6f ub=%.6f | cliq per-cnst: lb=%.6f ub=%.6f\n",
        (int)var_idx,
        (int)cnst_idx,
        (int)group_id,
        (double)coeff,
        (double)s_lb,
        (double)s_ub,
        (double)c_lb,
        (double)c_ub);
    }
  }
#endif

  return bnd;
}

template <typename i_t, typename f_t, i_t BDIM>
__device__ void update_bounds_cliq(typename problem_t<i_t, f_t>::view_t pb,
                                   i_t var_idx,
                                   i_t var_offset,
                                   i_t var_degree,
                                   bool is_int,
                                   typename bounds_update_data_t<i_t, f_t>::view_t upd,
                                   thrust::pair<f_t, f_t> old_bnd,
                                   typename clique_group_table_t<i_t, f_t>::view_t cliq)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t var_changed = upd.changed_variables[var_idx];
  if (!var_changed) { return; }

  auto bnd = old_bnd;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    bool changed  = upd.changed_constraints[cnst_idx] == 1;
    if (!changed) { continue; }
    auto a       = pb.reverse_coefficients[var_offset + i];
    auto cnst_ub = pb.constraint_upper_bounds[cnst_idx];
    auto cnst_lb = pb.constraint_lower_bounds[cnst_idx];
    // reverse_group_id is indexed on the reverse CSR position (same order as
    // reverse_constraints / reverse_coefficients). -1 means v is not in any
    // clique group for this constraint → behaves like the stock formula.
    // reverse_member_sign is parallel; 0 when group_id == -1, ±1 otherwise
    // (carries the literal polarity of this member).
    cuopt_assert((i_t)(var_offset + i) < (i_t)cliq.reverse_group_id.size(),
                 "reverse_group_id index out of range");
    cuopt_assert((i_t)(var_offset + i) < (i_t)cliq.reverse_member_sign.size(),
                 "reverse_member_sign index out of range");
    i_t group_id    = cliq.reverse_group_id[var_offset + i];
    i_t member_sign = cliq.reverse_member_sign[var_offset + i];
    cuopt_assert(group_id == -1 || (group_id >= 0 && group_id < cliq.n_groups),
                 "reverse_group_id carries invalid group id");
    bnd = update_bounds_per_cnst_cliq<i_t, f_t>(
      pb, var_idx, a, cnst_idx, cnst_lb, cnst_ub, upd, bnd, old_bnd, group_id, member_sign, cliq);
  }

  thrust::get<0>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd), cuda::maximum());
  __syncthreads();
  thrust::get<1>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd), cuda::minimum());
  __shared__ bool changed;
  if (threadIdx.x == 0) { changed = write_updated_bounds(pb, var_idx, is_int, upd, bnd, old_bnd); }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    if (changed) { atomicExch(&upd.next_changed_constraints[cnst_idx], 1); }
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void update_bounds_kernel_cliq(typename problem_t<i_t, f_t>::view_t pb,
                                          typename bounds_update_data_t<i_t, f_t>::view_t upd,
                                          typename clique_group_table_t<i_t, f_t>::view_t cliq)
{
  i_t var_idx    = blockIdx.x;
  i_t var_offset = pb.reverse_offsets[var_idx];
  i_t var_degree = pb.reverse_offsets[var_idx + 1] - var_offset;
  bool is_int    = (pb.variable_types[var_idx] == var_t::INTEGER);

  auto old_bnd = thrust::make_pair(upd.lb[var_idx], upd.ub[var_idx]);

  auto skip = skip_update(old_bnd, pb.tolerances.integrality_tolerance);
  if (skip) {
    return;
  } else {
    update_bounds_cliq<i_t, f_t, BDIM>(
      pb, var_idx, var_offset, var_degree, is_int, upd, old_bnd, cliq);
  }
}

}  // namespace cuopt::linear_programming::detail
