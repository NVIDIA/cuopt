/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <mip/problem/problem.cuh>
#include <mip/utils.cuh>
#include "load_balanced_bounds_common_kernels.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename upd_view_t>
inline __device__ thrust::pair<bool, bool> skip_cnst_calc(upd_view_t upd_0,
                                                          upd_view_t upd_1,
                                                          i_t cnst_idx)
{
  return thrust::make_pair((upd_0.changed_constraints[cnst_idx] == i_t{0}),
                           (upd_1.changed_constraints[cnst_idx] == i_t{0}));
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t>
__device__ f_t2 calc_act(activity_view_t view, i_t tid, i_t beg, i_t end)
{
  auto act = f_t2{0., 0.};
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coeff[i];
    auto var   = view.vars[i];

    atomicExch(&view.changed_variables[var], 1);

    auto bounds      = view.vars_bnd[var];
    auto min_contrib = bounds.x;
    auto max_contrib = bounds.y;
    if (coeff < 0.0) {
      min_contrib = bounds.y;
      max_contrib = bounds.x;
    }
    act.x += coeff * min_contrib;
    act.y += coeff * max_contrib;
  }
  return act;
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t,
          typename upd_view_t>
__device__ f_t2 calc_act(activity_view_t view, upd_view_t upd, i_t tid, i_t beg, i_t end)
{
  auto act = f_t2{0., 0.};
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coeff[i];
    auto var   = view.vars[i];

    atomicExch(&upd.changed_variables[var], 1);

    auto bounds      = upd.vars_bnd[var];
    auto min_contrib = bounds.x;
    auto max_contrib = bounds.y;
    if (coeff < 0.0) {
      min_contrib = bounds.y;
      max_contrib = bounds.x;
    }
    act.x += coeff * min_contrib;
    act.y += coeff * max_contrib;
  }
  return act;
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t,
          typename upd_view_t>
__device__ thrust::pair<f_t2, f_t2> calc_act(
  activity_view_t view, upd_view_t upd_0, upd_view_t upd_1, i_t tid, i_t beg, i_t end)
{
  auto act_0 = f_t2{0., 0.};
  auto act_1 = f_t2{0., 0.};
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coeff[i];
    auto var   = view.vars[i];

    atomicExch(&upd_0.changed_variables[var], 1);
    atomicExch(&upd_1.changed_variables[var], 1);

    auto bounds_0      = upd_0.vars_bnd[var];
    auto bounds_1      = upd_1.vars_bnd[var];
    auto min_contrib_0 = bounds_0.x;
    auto max_contrib_0 = bounds_0.y;
    auto min_contrib_1 = bounds_1.x;
    auto max_contrib_1 = bounds_1.y;
    if (coeff < 0.0) {
      min_contrib_0 = bounds_0.y;
      max_contrib_0 = bounds_0.x;
      min_contrib_1 = bounds_1.y;
      max_contrib_1 = bounds_1.x;
    }
    act_0.x += coeff * min_contrib_0;
    act_0.y += coeff * max_contrib_0;
    act_1.x += coeff * min_contrib_1;
    act_1.y += coeff * max_contrib_1;
  }
  return thrust::make_pair(act_0, act_1);
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename activity_view_t,
          typename upd_view_t>
__global__ void lb_calc_act_heavy_kernel(i_t id_range_beg,
                                         raft::device_span<const i_t> ids,
                                         raft::device_span<const i_t> pseudo_block_ids,
                                         i_t work_per_block,
                                         activity_view_t view,
                                         upd_view_t upd_0,
                                         upd_view_t upd_1)
{
  auto idx       = ids[blockIdx.x] + id_range_beg;
  auto cnst_idx  = view.cnst_reorg_ids[idx];
  auto skip_calc = skip_cnst_calc(upd_0, upd_1, cnst_idx);
  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) { return; }
  auto pseudo_block_id = pseudo_block_ids[blockIdx.x];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  if (thrust::get<1>(skip_calc)) {
    auto act = calc_act<i_t, f_t, f_t2, BDIM>(view, upd_0, threadIdx.x, item_off_beg, item_off_end);

    act.x = BlockReduce(temp_storage).Sum(act.x);
    __syncthreads();
    act.y = BlockReduce(temp_storage).Sum(act.y);

    // don't subtract constraint bounds yet
    // to be done in post processing in finalize_calc_act_kernel
    if (threadIdx.x == 0) { upd_0.tmp_cnst_slack[blockIdx.x] = act; }
  } else if (thrust::get<0>(skip_calc)) {
    auto act = calc_act<i_t, f_t, f_t2, BDIM>(view, upd_1, threadIdx.x, item_off_beg, item_off_end);

    act.x = BlockReduce(temp_storage).Sum(act.x);
    __syncthreads();
    act.y = BlockReduce(temp_storage).Sum(act.y);

    // don't subtract constraint bounds yet
    // to be done in post processing in finalize_calc_act_kernel
    if (threadIdx.x == 0) { upd_1.tmp_cnst_slack[blockIdx.x] = act; }
  } else {
    auto act =
      calc_act<i_t, f_t, f_t2, BDIM>(view, upd_0, upd_1, threadIdx.x, item_off_beg, item_off_end);

    thrust::get<0>(act).x = BlockReduce(temp_storage).Sum(thrust::get<0>(act).x);
    __syncthreads();
    thrust::get<0>(act).y = BlockReduce(temp_storage).Sum(thrust::get<0>(act).y);
    __syncthreads();
    thrust::get<1>(act).x = BlockReduce(temp_storage).Sum(thrust::get<1>(act).x);
    __syncthreads();
    thrust::get<1>(act).y = BlockReduce(temp_storage).Sum(thrust::get<1>(act).y);
    if (threadIdx.x == 0) {
      upd_0.tmp_cnst_slack[blockIdx.x] = thrust::get<0>(act);
      upd_1.tmp_cnst_slack[blockIdx.x] = thrust::get<1>(act);
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename activity_view_t>
__global__ void lb_calc_act_heavy_kernel(i_t id_range_beg,
                                         raft::device_span<const i_t> ids,
                                         raft::device_span<const i_t> pseudo_block_ids,
                                         i_t work_per_block,
                                         activity_view_t view,
                                         raft::device_span<f_t2> tmp_cnst_act)
{
  auto idx      = ids[blockIdx.x] + id_range_beg;
  auto cnst_idx = view.cnst_reorg_ids[idx];
  if (view.changed_constraints[cnst_idx] == 0) { return; }
  auto pseudo_block_id = pseudo_block_ids[blockIdx.x];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto act = calc_act<i_t, f_t, f_t2, BDIM>(view, threadIdx.x, item_off_beg, item_off_end);

  act.x = BlockReduce(temp_storage).Sum(act.x);
  __syncthreads();
  act.y = BlockReduce(temp_storage).Sum(act.y);

  // don't subtract constraint bounds yet
  // to be done in post processing in finalize_calc_act_kernel
  if (threadIdx.x == 0) { tmp_cnst_act[blockIdx.x] = act; }
}

template <bool erase_inf_cnst, typename i_t, typename f_t, typename f_t2, typename activity_view_t>
inline __device__ void write_cnst_slack(
  activity_view_t view, i_t cnst_idx, f_t2 cnst_lb_ub, f_t2 act, f_t eps)
{
  auto cnst_prop = f_t2{cnst_lb_ub.y - act.x, cnst_lb_ub.x - act.y};
  if constexpr (erase_inf_cnst) {
    if ((0 > cnst_prop.x + eps) || (eps < cnst_prop.y)) {
      cnst_prop.x = std::numeric_limits<f_t>::quiet_NaN();
    }
  }
  view.cnst_slack[cnst_idx] = cnst_prop;
}

template <typename i_t, typename f_t2, typename activity_view_t>
inline __device__ void write_cnst_slack(activity_view_t view,
                                        i_t cnst_idx,
                                        f_t2 cnst_lb_ub,
                                        f_t2 act)
{
  auto cnst_prop            = f_t2{cnst_lb_ub.y - act.x, cnst_lb_ub.x - act.y};
  view.cnst_slack[cnst_idx] = cnst_prop;
}

template <typename i_t, typename f_t, typename f_t2, typename activity_view_t>
inline __device__ void write_cnst_slack(
  activity_view_t view, i_t cnst_idx, f_t2 cnst_lb_ub, f_t2 act, f_t eps)
{
  auto cnst_prop = f_t2{cnst_lb_ub.y - act.x, cnst_lb_ub.x - act.y};
  if ((0 > cnst_prop.x + eps) || (eps < cnst_prop.y)) {
    cnst_prop.x = std::numeric_limits<f_t>::quiet_NaN();
  }
  view.cnst_slack[cnst_idx] = cnst_prop;
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          typename activity_view_t,
          typename upd_view_t>
__global__ void finalize_calc_act_kernel(i_t heavy_cnst_beg_id,
                                         raft::device_span<const i_t> item_offsets,
                                         activity_view_t view,
                                         upd_view_t upd_0,
                                         upd_view_t upd_1)
{
  using warp_reduce = cub::WarpReduce<f_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage;
  i_t idx        = heavy_cnst_beg_id + blockIdx.x;
  i_t cnst_idx   = view.cnst_reorg_ids[idx];
  auto skip_calc = skip_cnst_calc(upd_0, upd_1, cnst_idx);
  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) { return; }
  auto cnst_lb_ub          = view.cnst_bnd[idx];
  [[maybe_unused]] f_t eps = {};
  if constexpr (erase_inf_cnst) {
    eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                       cnst_lb_ub.y,
                                       view.tolerances.absolute_tolerance,
                                       view.tolerances.relative_tolerance);
  }

  // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
  i_t item_off_beg = item_offsets[blockIdx.x];
  i_t item_off_end = item_offsets[blockIdx.x + 1];

  if (thrust::get<1>(skip_calc)) {
    f_t2 cnst_prop = f_t2{0., 0.};
    // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act = upd_0.tmp_cnst_slack[i];
      cnst_prop.x += act.x;
      cnst_prop.y += act.y;
    }
    cnst_prop.x = warp_reduce(temp_storage).Sum(cnst_prop.x);
    __syncwarp();
    cnst_prop.y = warp_reduce(temp_storage).Sum(cnst_prop.y);
    if (threadIdx.x == 0) {
      if constexpr (erase_inf_cnst) {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, cnst_prop, eps);
      } else {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, cnst_prop);
      }
    }
  } else if (thrust::get<0>(skip_calc)) {
    f_t2 cnst_prop = f_t2{0., 0.};
    // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act = upd_1.tmp_cnst_slack[i];
      cnst_prop.x += act.x;
      cnst_prop.y += act.y;
    }
    cnst_prop.x = warp_reduce(temp_storage).Sum(cnst_prop.x);
    __syncwarp();
    cnst_prop.y = warp_reduce(temp_storage).Sum(cnst_prop.y);
    if (threadIdx.x == 0) {
      if constexpr (erase_inf_cnst) {
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, cnst_prop, eps);
      } else {
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, cnst_prop);
      }
    }
  } else {
    f_t2 cnst_prop_0 = f_t2{0., 0.};
    f_t2 cnst_prop_1 = f_t2{0., 0.};
    // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act = upd_0.tmp_cnst_slack[i];
      cnst_prop_0.x += act.x;
      cnst_prop_0.y += act.y;
      act = upd_1.tmp_cnst_slack[i];
      cnst_prop_1.x += act.x;
      cnst_prop_1.y += act.y;
    }
    cnst_prop_0.x = warp_reduce(temp_storage).Sum(cnst_prop_0.x);
    __syncwarp();
    cnst_prop_0.y = warp_reduce(temp_storage).Sum(cnst_prop_0.y);
    __syncwarp();
    cnst_prop_1.x = warp_reduce(temp_storage).Sum(cnst_prop_1.x);
    __syncwarp();
    cnst_prop_1.y = warp_reduce(temp_storage).Sum(cnst_prop_1.y);
    if (threadIdx.x == 0) {
      if constexpr (erase_inf_cnst) {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, cnst_prop_0, eps);
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, cnst_prop_1, eps);
      } else {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, cnst_prop_0);
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, cnst_prop_1);
      }
    }
  }
}

template <bool erase_inf_cnst, typename i_t, typename f_t, typename f_t2, typename activity_view_t>
__global__ void finalize_calc_act_kernel(i_t heavy_cnst_beg_id,
                                         raft::device_span<const i_t> item_offsets,
                                         raft::device_span<f_t2> tmp_act,
                                         activity_view_t view)
{
  using warp_reduce = cub::WarpReduce<f_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage;
  i_t idx      = heavy_cnst_beg_id + blockIdx.x;
  i_t cnst_idx = view.cnst_reorg_ids[idx];
  if (view.changed_constraints[cnst_idx] == 0) { return; }
  auto cnst_lb_ub          = view.cnst_bnd[idx];
  [[maybe_unused]] f_t eps = {};
  if constexpr (erase_inf_cnst) {
    eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                       cnst_lb_ub.y,
                                       view.tolerances.absolute_tolerance,
                                       view.tolerances.relative_tolerance);
  }

  // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
  i_t item_off_beg = item_offsets[blockIdx.x];
  i_t item_off_end = item_offsets[blockIdx.x + 1];
  f_t2 cnst_prop   = f_t2{0., 0.};
  // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
  for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
    auto act = tmp_act[i];
    cnst_prop.x += act.x;
    cnst_prop.y += act.y;
  }
  cnst_prop.x = warp_reduce(temp_storage).Sum(cnst_prop.x);
  __syncwarp();
  cnst_prop.y = warp_reduce(temp_storage).Sum(cnst_prop.y);
  if (threadIdx.x == 0) {
    write_cnst_slack<erase_inf_cnst>(view, cnst_idx, cnst_lb_ub, cnst_prop, eps);
  }
}

// template <bool erase_inf_cnst, typename i_t, typename f_t, typename f_t2, typename
// activity_view_t, typename upd_view_t>
//__global__ void finalize_calc_act_kernel(i_t heavy_cnst_beg_id,
//                                          raft::device_span<const i_t> item_offsets,
//                                          activity_view_t view,
//                                          upd_view_t upd)
//{
//   using warp_reduce = cub::WarpReduce<f_t>;
//   __shared__ typename warp_reduce::TempStorage temp_storage;
//   i_t idx      = heavy_cnst_beg_id + blockIdx.x;
//   i_t cnst_idx = view.cnst_reorg_ids[idx];
//   if (view.changed_constraints[cnst_idx] == 0) { return; }
//   auto cnst_lb_ub          = view.cnst_bnd[idx];
//   [[maybe_unused]] f_t eps = {};
//   if constexpr (erase_inf_cnst) {
//     eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
//                                        cnst_lb_ub.y,
//                                        view.tolerances.absolute_tolerance,
//                                        view.tolerances.relative_tolerance);
//   }
//
//   // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
//   i_t item_off_beg = item_offsets[blockIdx.x];
//   i_t item_off_end = item_offsets[blockIdx.x + 1];
//   f_t2 cnst_prop   = f_t2{0., 0.};
//   // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
//   for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
//     auto act = upd.tmp_act[i];
//     cnst_prop.x += act.x;
//     cnst_prop.y += act.y;
//   }
//   cnst_prop.x = warp_reduce(temp_storage).Sum(cnst_prop.x);
//   __syncwarp();
//   cnst_prop.y = warp_reduce(temp_storage).Sum(cnst_prop.y);
//   if (threadIdx.x == 0) {
//     write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, cnst_prop, eps);
//   }
// }

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename activity_view_t>
__global__ void lb_calc_act_block_kernel(i_t id_range_beg, activity_view_t view)

{
  i_t idx      = id_range_beg + blockIdx.x;
  i_t cnst_idx = view.cnst_reorg_ids[idx];
  if (view.changed_constraints[cnst_idx] == 0) { return; }
  auto cnst_lb_ub          = view.cnst_bnd[idx];
  i_t item_off_beg         = view.offsets[idx];
  i_t item_off_end         = view.offsets[idx + 1];
  [[maybe_unused]] f_t eps = {};
  if constexpr (erase_inf_cnst) {
    eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                       cnst_lb_ub.y,
                                       view.tolerances.absolute_tolerance,
                                       view.tolerances.relative_tolerance);
  }

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto act = calc_act<i_t, f_t, f_t2, BDIM>(view, threadIdx.x, item_off_beg, item_off_end);

  act.x = BlockReduce(temp_storage).Sum(act.x);
  __syncthreads();
  act.y = BlockReduce(temp_storage).Sum(act.y);

  if (threadIdx.x == 0) { write_cnst_slack<erase_inf_cnst>(view, cnst_idx, cnst_lb_ub, act, eps); }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename activity_view_t,
          typename upd_view_t>
__global__ void lb_calc_act_block_kernel(i_t id_range_beg, activity_view_t view, upd_view_t upd)

{
  i_t idx      = id_range_beg + blockIdx.x;
  i_t cnst_idx = view.cnst_reorg_ids[idx];
  if (view.changed_constraints[cnst_idx] == 0) { return; }
  auto cnst_lb_ub          = view.cnst_bnd[idx];
  i_t item_off_beg         = view.offsets[idx];
  i_t item_off_end         = view.offsets[idx + 1];
  [[maybe_unused]] f_t eps = {};
  if constexpr (erase_inf_cnst) {
    eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                       cnst_lb_ub.y,
                                       view.tolerances.absolute_tolerance,
                                       view.tolerances.relative_tolerance);
  }

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto act = calc_act<i_t, f_t, f_t2, BDIM>(view, upd, threadIdx.x, item_off_beg, item_off_end);

  act.x = BlockReduce(temp_storage).Sum(act.x);
  __syncthreads();
  act.y = BlockReduce(temp_storage).Sum(act.y);

  if (threadIdx.x == 0) { write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, act, eps); }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename activity_view_t,
          typename upd_view_t>
__global__ void lb_calc_act_block_kernel(i_t id_range_beg,
                                         activity_view_t view,
                                         upd_view_t upd_0,
                                         upd_view_t upd_1)

{
  i_t idx        = id_range_beg + blockIdx.x;
  i_t cnst_idx   = view.cnst_reorg_ids[idx];
  auto skip_calc = skip_cnst_calc(upd_0, upd_1, cnst_idx);
  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) { return; }
  auto cnst_lb_ub          = view.cnst_bnd[idx];
  i_t item_off_beg         = view.offsets[idx];
  i_t item_off_end         = view.offsets[idx + 1];
  [[maybe_unused]] f_t eps = {};
  if constexpr (erase_inf_cnst) {
    eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                       cnst_lb_ub.y,
                                       view.tolerances.absolute_tolerance,
                                       view.tolerances.relative_tolerance);
  }

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  if (thrust::get<1>(skip_calc)) {
    auto act = calc_act<i_t, f_t, f_t2, BDIM>(view, upd_0, threadIdx.x, item_off_beg, item_off_end);

    act.x = BlockReduce(temp_storage).Sum(act.x);
    __syncthreads();
    act.y = BlockReduce(temp_storage).Sum(act.y);

    if (threadIdx.x == 0) {
      if constexpr (erase_inf_cnst) {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, act, eps);
      } else {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, act);
      }
    }
  } else if (thrust::get<0>(skip_calc)) {
    auto act = calc_act<i_t, f_t, f_t2, BDIM>(view, upd_1, threadIdx.x, item_off_beg, item_off_end);

    act.x = BlockReduce(temp_storage).Sum(act.x);
    __syncthreads();
    act.y = BlockReduce(temp_storage).Sum(act.y);

    if (threadIdx.x == 0) {
      if constexpr (erase_inf_cnst) {
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, act, eps);
      } else {
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, act);
      }
    }
  } else {
    auto act =
      calc_act<i_t, f_t, f_t2, BDIM>(view, upd_0, upd_1, threadIdx.x, item_off_beg, item_off_end);

    thrust::get<0>(act).x = BlockReduce(temp_storage).Sum(thrust::get<0>(act).x);
    __syncthreads();
    thrust::get<0>(act).y = BlockReduce(temp_storage).Sum(thrust::get<0>(act).y);
    __syncthreads();
    thrust::get<1>(act).x = BlockReduce(temp_storage).Sum(thrust::get<1>(act).x);
    __syncthreads();
    thrust::get<1>(act).y = BlockReduce(temp_storage).Sum(thrust::get<1>(act).y);

    if (threadIdx.x == 0) {
      if constexpr (erase_inf_cnst) {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
      } else {
        write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act));
        write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act));
      }
    }
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t,
          typename upd_view_t>
__global__ void lb_calc_act_sub_warp_kernel(
  i_t id_range_beg, i_t id_range_end, activity_view_t view, upd_view_t upd_0, upd_view_t upd_1)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_CNST;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_CNST);
  // bool continue_calc          = (idx < id_range_end);
  auto skip_calc = thrust::make_pair(!(idx < id_range_end), !(idx < id_range_end));
  i_t cnst_idx;
  [[maybe_unused]] f_t eps = {};
  f_t2 cnst_lb_ub;
  if (idx < id_range_end) {
    cnst_idx   = view.cnst_reorg_ids[idx];
    skip_calc  = skip_cnst_calc(upd_0, upd_1, cnst_idx);
    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_CNST;

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});

  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];
  if (thrust::get<1>(skip_calc)) {
    thrust::get<0>(act) =
      calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, upd_0, p_tid, item_off_beg, item_off_end);
  } else if (thrust::get<0>(skip_calc)) {
    thrust::get<1>(act) =
      calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, upd_1, p_tid, item_off_beg, item_off_end);
  } else {
    act = calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(
      view, upd_0, upd_1, p_tid, item_off_beg, item_off_end);
  }
  thrust::get<0>(act).x = warp_reduce(temp_storage).Sum(thrust::get<0>(act).x);
  __syncwarp();
  thrust::get<0>(act).y = warp_reduce(temp_storage).Sum(thrust::get<0>(act).y);
  __syncwarp();
  thrust::get<1>(act).x = warp_reduce(temp_storage).Sum(thrust::get<1>(act).x);
  __syncwarp();
  thrust::get<1>(act).y = warp_reduce(temp_storage).Sum(thrust::get<1>(act).y);

  if (head_flag && thrust::get<1>(skip_calc)) {
    if constexpr (erase_inf_cnst) {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
    } else {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act));
    }
  } else if (head_flag && thrust::get<0>(skip_calc)) {
    if constexpr (erase_inf_cnst) {
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
    } else {
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act));
    }
  } else if (head_flag) {
    if constexpr (erase_inf_cnst) {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
    } else {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act));
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act));
    }
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t>
__global__ void lb_calc_act_sub_warp_kernel(i_t id_range_beg,
                                            i_t id_range_end,
                                            activity_view_t view)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_CNST;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_CNST);
  bool continue_calc          = (idx < id_range_end);
  i_t cnst_idx;
  [[maybe_unused]] f_t eps = {};
  f_t2 cnst_lb_ub;
  if (continue_calc) {
    cnst_idx = view.cnst_reorg_ids[idx];
    if (view.changed_constraints[cnst_idx] == 0) { continue_calc = false; }
    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_CNST;

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto act = f_t2{0., 0.};

  if (continue_calc) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act = calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, p_tid, item_off_beg, item_off_end);
  }

  act.x = warp_reduce(temp_storage).Sum(act.x);
  __syncwarp();
  act.y = warp_reduce(temp_storage).Sum(act.y);

  if (head_flag && continue_calc) {
    write_cnst_slack<erase_inf_cnst>(view, cnst_idx, cnst_lb_ub, act, eps);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t>
__device__ void calc_act_sub_warp(i_t id_warp_beg, i_t id_range_end, activity_view_t view)
{
  i_t lane_id        = (threadIdx.x & 31);
  i_t idx            = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  bool continue_calc = (idx < id_range_end);
  i_t cnst_idx;
  [[maybe_unused]] f_t eps = {};
  f_t2 cnst_lb_ub;
  if (continue_calc) {
    cnst_idx   = view.cnst_reorg_ids[idx];
    cnst_lb_ub = view.cnst_bnd[idx];
    if (view.changed_constraints[cnst_idx] == 0) { continue_calc = false; }
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }
  i_t p_tid = lane_id & (MAX_EDGE_PER_CNST - 1);

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto act = f_t2{0., 0.};

  if (continue_calc) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act = calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, p_tid, item_off_beg, item_off_end);
  }

  act.x = warp_reduce(temp_storage).Sum(act.x);
  __syncwarp();
  act.y = warp_reduce(temp_storage).Sum(act.y);

  if (head_flag && continue_calc) {
    write_cnst_slack<erase_inf_cnst>(view, cnst_idx, cnst_lb_ub, act, eps);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename activity_view_t>
__global__ void lb_calc_act_sub_warp_kernel(activity_view_t view,
                                            raft::device_span<i_t> warp_cnst_offsets,
                                            raft::device_span<i_t> warp_cnst_id_offsets)
{
  i_t id_warp_beg, id_range_end, threads_per_constraints;
  detect_range_sub_warp<i_t>(
    &id_warp_beg, &id_range_end, &threads_per_constraints, warp_cnst_offsets, warp_cnst_id_offsets);

  if (threads_per_constraints == 1) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 1>(id_warp_beg, id_range_end, view);
  } else if (threads_per_constraints == 2) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 2>(id_warp_beg, id_range_end, view);
  } else if (threads_per_constraints == 4) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 4>(id_warp_beg, id_range_end, view);
  } else if (threads_per_constraints == 8) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 8>(id_warp_beg, id_range_end, view);
  } else if (threads_per_constraints == 16) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 16>(id_warp_beg, id_range_end, view);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t,
          typename upd_view_t>
__device__ void calc_act_sub_warp(
  i_t id_warp_beg, i_t id_range_end, activity_view_t view, upd_view_t upd_0, upd_view_t upd_1)
{
  i_t lane_id    = (threadIdx.x & 31);
  i_t idx        = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  auto skip_calc = thrust::make_pair(!(idx < id_range_end), !(idx < id_range_end));
  i_t cnst_idx;
  [[maybe_unused]] f_t eps = {};
  f_t2 cnst_lb_ub;
  if (idx < id_range_end) {
    cnst_idx   = view.cnst_reorg_ids[idx];
    cnst_lb_ub = view.cnst_bnd[idx];
    skip_calc  = skip_cnst_calc(upd_0, upd_1, cnst_idx);
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }
  i_t p_tid = lane_id & (MAX_EDGE_PER_CNST - 1);

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});

  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  if (thrust::get<1>(skip_calc)) {
    thrust::get<0>(act) =
      calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, upd_0, p_tid, item_off_beg, item_off_end);
  } else if (thrust::get<0>(skip_calc)) {
    thrust::get<1>(act) =
      calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, upd_1, p_tid, item_off_beg, item_off_end);
  } else {
    act = calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(
      view, upd_0, upd_1, p_tid, item_off_beg, item_off_end);
  }
  thrust::get<0>(act).x = warp_reduce(temp_storage).Sum(thrust::get<0>(act).x);
  __syncwarp();
  thrust::get<0>(act).y = warp_reduce(temp_storage).Sum(thrust::get<0>(act).y);
  __syncwarp();
  thrust::get<1>(act).x = warp_reduce(temp_storage).Sum(thrust::get<1>(act).x);
  __syncwarp();
  thrust::get<1>(act).y = warp_reduce(temp_storage).Sum(thrust::get<1>(act).y);

  if (head_flag && thrust::get<1>(skip_calc)) {
    if constexpr (erase_inf_cnst) {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
    } else {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act));
    }
  } else if (head_flag && thrust::get<0>(skip_calc)) {
    if constexpr (erase_inf_cnst) {
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
    } else {
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act));
    }
  } else if (head_flag) {
    if constexpr (erase_inf_cnst) {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
    } else {
      write_cnst_slack(upd_0, cnst_idx, cnst_lb_ub, thrust::get<0>(act));
      write_cnst_slack(upd_1, cnst_idx, cnst_lb_ub, thrust::get<1>(act));
    }
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename activity_view_t,
          typename upd_view_t>
__global__ void lb_calc_act_sub_warp_kernel(activity_view_t view,
                                            upd_view_t upd_0,
                                            upd_view_t upd_1,
                                            raft::device_span<i_t> warp_cnst_offsets,
                                            raft::device_span<i_t> warp_cnst_id_offsets)
{
  i_t id_warp_beg, id_range_end, threads_per_constraints;
  detect_range_sub_warp<i_t>(
    &id_warp_beg, &id_range_end, &threads_per_constraints, warp_cnst_offsets, warp_cnst_id_offsets);

  if (threads_per_constraints == 1) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 1>(
      id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_constraints == 2) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 2>(
      id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_constraints == 4) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 4>(
      id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_constraints == 8) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 8>(
      id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_constraints == 16) {
    calc_act_sub_warp<erase_inf_cnst, i_t, f_t, f_t2, BDIM, 16>(
      id_warp_beg, id_range_end, view, upd_0, upd_1);
  }
}

/// BOUNDS UPDATE

template <typename i_t, typename f_t, typename f_t2>
inline __device__ void update_bounds_per_cnst(f_t2& bounds,
                                              f_t2 old_bounds,
                                              f_t coeff,
                                              i_t cnst_idx,
                                              raft::device_span<f_t2> cnst_slack,
                                              raft::device_span<i_t> changed_constraints)
{
  // cnst_slack[cnst_idx].x now has cnst_ub - min_a
  // cnst_slack[cnst_idx].y now has cnst_lb - max_a
  auto cnstr_data           = cnst_slack[cnst_idx];
  bool unchanged            = (changed_constraints[cnst_idx] == 0);
  auto cnstr_ub_minus_min_a = cnstr_data.x;
  auto cnstr_lb_minus_max_a = cnstr_data.y;
  //  don't propagate over constraints that are infeasible
  if (unchanged || isnan(cnstr_data.x)) { return; }

  f_t min_contrib = old_bounds.x;
  f_t max_contrib = old_bounds.y;
  if (coeff < 0.0) {
    min_contrib = old_bounds.y;
    max_contrib = old_bounds.x;
  }

  auto delta_min_act = (cnstr_ub_minus_min_a + (coeff * min_contrib)) / coeff;
  auto delta_max_act = (cnstr_lb_minus_max_a + (coeff * max_contrib)) / coeff;

  f_t lb_contrib = delta_max_act;
  f_t ub_contrib = delta_min_act;
  if (coeff < 0.0) {
    lb_contrib = delta_min_act;
    ub_contrib = delta_max_act;
  }
  bounds.x = max(bounds.x, lb_contrib);
  bounds.y = min(bounds.y, ub_contrib);
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t,
          typename upd_view_t>
__device__ thrust::pair<f_t2, f_t2> update_bounds(bounds_update_view_t view,
                                                  upd_view_t upd_0,
                                                  upd_view_t upd_1,
                                                  i_t tid,
                                                  i_t beg,
                                                  i_t end,
                                                  thrust::pair<f_t2, f_t2> old_bounds)
{
  auto bounds = old_bounds;

  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    auto a        = view.coeff[i];
    auto cnst_idx = view.cnst[i];

    update_bounds_per_cnst(thrust::get<0>(bounds),
                           thrust::get<0>(old_bounds),
                           a,
                           cnst_idx,
                           upd_0.cnst_slack,
                           upd_0.changed_constraints);
    update_bounds_per_cnst(thrust::get<1>(bounds),
                           thrust::get<1>(old_bounds),
                           a,
                           cnst_idx,
                           upd_1.cnst_slack,
                           upd_1.changed_constraints);
  }

  return bounds;
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t>
__device__ f_t2 update_bounds(bounds_update_view_t view,
                              raft::device_span<f_t2> cnst_slack,
                              raft::device_span<i_t> changed_constraints,
                              i_t tid,
                              i_t beg,
                              i_t end,
                              f_t2 old_bounds)
{
  f_t2 bounds = old_bounds;

  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    auto a        = view.coeff[i];
    auto cnst_idx = view.cnst[i];

    update_bounds_per_cnst(bounds, old_bounds, a, cnst_idx, cnst_slack, changed_constraints);
  }

  return bounds;
}

template <typename f_t2, typename bounds_update_view_t>
inline __device__ bool write_updated_bounds(
  f_t2* ptr, bool is_int, bounds_update_view_t view, f_t2 bounds, f_t2 old_bounds)
{
  bool changed   = false;
  auto threshold = 1e3 * view.tolerances.absolute_tolerance;
  if (is_int) {
    bounds.x = ceil(bounds.x - view.tolerances.integrality_tolerance);
    bounds.y = floor(bounds.y + view.tolerances.integrality_tolerance);
  }
  auto lb_updated = (fabs(bounds.x - old_bounds.x) > threshold);
  auto ub_updated = (fabs(bounds.y - old_bounds.y) > threshold);
  if ((bounds.x != old_bounds.x) || (bounds.y != old_bounds.y)) { changed = true; }

  if (lb_updated) { old_bounds.x = bounds.x; }
  if (ub_updated) { old_bounds.y = bounds.y; }

  *ptr = old_bounds;

  if (lb_updated || ub_updated) { atomicAdd(view.bounds_changed, 1); }
  return changed;
}

template <typename f_t2, typename bounds_update_view_t, typename upd_view_t>
inline __device__ bool write_updated_bounds(
  f_t2* ptr, bool is_int, bounds_update_view_t view, upd_view_t upd, f_t2 bounds, f_t2 old_bounds)
{
  bool changed   = false;
  auto threshold = 1e3 * view.tolerances.absolute_tolerance;
  if (is_int) {
    bounds.x = ceil(bounds.x - view.tolerances.integrality_tolerance);
    bounds.y = floor(bounds.y + view.tolerances.integrality_tolerance);
  }
  auto lb_updated = (fabs(bounds.x - old_bounds.x) > threshold);
  auto ub_updated = (fabs(bounds.y - old_bounds.y) > threshold);
  if ((bounds.x != old_bounds.x) || (bounds.y != old_bounds.y)) { changed = true; }

  if (lb_updated) { old_bounds.x = bounds.x; }
  if (ub_updated) { old_bounds.y = bounds.y; }

  *ptr = old_bounds;

  if (lb_updated || ub_updated) { atomicAdd(upd.bounds_changed, 1); }
  return changed;
}

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_bnd_heavy_kernel(i_t id_range_beg,
                                        raft::device_span<const i_t> ids,
                                        raft::device_span<const i_t> pseudo_block_ids,
                                        i_t work_per_block,
                                        bounds_update_view_t view,
                                        raft::device_span<f_t2> tmp_vars_bnd)
{
  auto idx             = ids[blockIdx.x] + id_range_beg;
  auto pseudo_block_id = pseudo_block_ids[blockIdx.x];
  auto var_idx         = view.vars_reorg_ids[idx];
  // x is lb, y is ub
  auto old_bounds = view.vars_bnd[var_idx];
  if (view.changed_variables[var_idx] == 0) {
    tmp_vars_bnd[blockIdx.x] = old_bounds;
    return;
  }
  bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
  i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) {
    tmp_vars_bnd[blockIdx.x] = old_bounds;
    return;
  }
  auto bounds = update_bounds<i_t, f_t, f_t2, BDIM>(view,
                                                    view.cnst_slack,
                                                    view.changed_constraints,
                                                    threadIdx.x,
                                                    item_off_beg,
                                                    item_off_end,
                                                    old_bounds);

  bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
  __syncthreads();
  bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

  if (threadIdx.x == 0) {
    bool changed =
      write_updated_bounds(&tmp_vars_bnd[blockIdx.x], is_int, view, bounds, old_bounds);
    atomicExch(&view.var_bounds_changed[var_idx], 1);
  }
}

template <typename i_t, typename f_t, typename f_t2, typename bounds_update_view_t>
__global__ void finalize_upd_bnd_kernel(i_t heavy_vars_beg_id,
                                        raft::device_span<const i_t> item_offsets,
                                        raft::device_span<f_t2> tmp_vars_bnd,
                                        bounds_update_view_t view)
{
  using warp_reduce = cub::WarpReduce<f_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage;
  i_t idx     = heavy_vars_beg_id + blockIdx.x;
  i_t var_idx = view.vars_reorg_ids[idx];
  if (view.changed_variables[var_idx] == 0) { return; }

  // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
  i_t item_off_beg = item_offsets[blockIdx.x];
  i_t item_off_end = item_offsets[blockIdx.x + 1];
  f_t2 bounds = f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
  for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
    auto bnd = tmp_vars_bnd[i];
    bounds.x = max(bounds.x, bnd.x);
    bounds.y = min(bounds.y, bnd.y);
  }
  bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
  __syncwarp();
  bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());
  if (threadIdx.x == 0) { view.vars_bnd[var_idx] = bounds; }
}

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_bnd_block_kernel(i_t id_range_beg, bounds_update_view_t view)
{
  i_t idx     = id_range_beg + blockIdx.x;
  i_t var_idx = view.vars_reorg_ids[idx];
  if (view.changed_variables[var_idx] == 0) { return; }
  // x is lb, y is ub
  auto old_bounds  = view.vars_bnd[var_idx];
  bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { return; }
  auto bounds = update_bounds<i_t, f_t, f_t2, BDIM>(view,
                                                    view.cnst_slack,
                                                    view.changed_constraints,
                                                    threadIdx.x,
                                                    item_off_beg,
                                                    item_off_end,
                                                    old_bounds);

  bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
  __syncthreads();
  bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

  if (threadIdx.x == 0) {
    bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds, old_bounds);
    if (changed) { view.var_bounds_changed[var_idx] = changed; }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_VAR,
          typename activity_view_t>
__global__ void lb_upd_bnd_sub_warp_kernel(i_t id_range_beg, i_t id_range_end, activity_view_t view)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_VAR;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_VAR);
  i_t var_idx;
  auto old_bounds =
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  auto bounds        = old_bounds;
  bool is_int        = false;
  bool continue_calc = (idx < id_range_end);
  if (continue_calc) {
    var_idx = view.vars_reorg_ids[idx];
    if (view.changed_variables[var_idx] == 0) { continue_calc = false; }
    old_bounds = view.vars_bnd[var_idx];
    is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (continue_calc) {
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { head_flag = 0; }

    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    bounds = update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(view,
                                                             view.cnst_slack,
                                                             view.changed_constraints,
                                                             p_tid,
                                                             item_off_beg,
                                                             item_off_end,
                                                             old_bounds);
  }

  bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
  __syncwarp();
  bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());

  if (head_flag && continue_calc) {
    bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds, old_bounds);
    if (changed) { view.var_bounds_changed[var_idx] = changed; }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t>
__device__ void upd_bnd_sub_warp(i_t id_warp_beg, i_t id_range_end, bounds_update_view_t view)
{
  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_VAR);
  i_t var_idx;
  auto old_bounds =
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  auto bounds        = old_bounds;
  bool is_int        = false;
  bool continue_calc = (idx < id_range_end);
  if (continue_calc) {
    var_idx = view.vars_reorg_ids[idx];
    if (view.changed_variables[var_idx] == 0) { continue_calc = false; }
    old_bounds = view.vars_bnd[var_idx];
    is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  // Equivalent to
  // i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;
  i_t p_tid = lane_id & (MAX_EDGE_PER_VAR - 1);

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (continue_calc) {
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { head_flag = 0; }

    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    bounds = update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(view,
                                                             view.cnst_slack,
                                                             view.changed_constraints,
                                                             p_tid,
                                                             item_off_beg,
                                                             item_off_end,
                                                             old_bounds);
  }

  bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
  __syncwarp();
  bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());

  if (head_flag && continue_calc) {
    bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds, old_bounds);
    if (changed) { view.var_bounds_changed[var_idx] = changed; }
  }
}

#if 1
template <typename f_t, typename f_t2>
inline __device__ bool skip_update(f_t2 bnd, f_t int_tol)
{
  return (bnd.x + int_tol >= bnd.y);
}

template <typename i_t, typename f_t, typename f_t2, typename upd_view_t>
inline __device__ thrust::pair<bool, bool> skip_update(
  thrust::pair<f_t2, f_t2> bnd, upd_view_t upd_0, upd_view_t upd_1, i_t var_idx, f_t int_tol)
{
  return thrust::make_pair((thrust::get<0>(bnd).x + int_tol >= thrust::get<0>(bnd).y) ||
                             (upd_0.changed_variables[var_idx] == 0),

                           (thrust::get<1>(bnd).x + int_tol >= thrust::get<1>(bnd).y) ||
                             (upd_1.changed_variables[var_idx] == 0));
}

template <typename i_t, typename upd_view_t>
inline __device__ thrust::pair<bool, bool> skip_update(upd_view_t upd_0,
                                                       upd_view_t upd_1,
                                                       i_t var_idx)
{
  return thrust::make_pair((upd_0.var_bounds_changed[var_idx] == 0),
                           (upd_1.var_bounds_changed[var_idx] == 0));
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename bounds_update_view_t,
          typename upd_view_t>
__global__ void lb_upd_bnd_heavy_kernel(i_t id_range_beg,
                                        raft::device_span<const i_t> ids,
                                        raft::device_span<const i_t> pseudo_block_ids,
                                        i_t work_per_block,
                                        bounds_update_view_t view,
                                        upd_view_t upd_0,
                                        upd_view_t upd_1)
{
  auto idx             = ids[blockIdx.x] + id_range_beg;
  auto pseudo_block_id = pseudo_block_ids[blockIdx.x];
  auto var_idx         = view.vars_reorg_ids[idx];
  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  auto old_bounds = thrust::make_pair(upd_0.vars_bnd[var_idx], upd_1.vars_bnd[var_idx]);
  // auto skip_calc = thrust::make_pair(
  //   skip_update(thrust::get<0>(old_bounds), view.tolerances.integrality_tolerance) ||
  //   (upd_0.changed_variables[var_idx] == 0), skip_update(thrust::get<1>(old_bounds),
  //   view.tolerances.integrality_tolerance) || (upd_1.changed_variables[var_idx] == 0));
  auto skip_calc =
    skip_update(old_bounds, upd_0, upd_1, var_idx, view.tolerances.integrality_tolerance);

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) {
    if (threadIdx.x == 0) {
      upd_0.tmp_vars_bnd[blockIdx.x] = thrust::get<0>(old_bounds);
      upd_1.tmp_vars_bnd[blockIdx.x] = thrust::get<1>(old_bounds);
    }
    return;
  } else if (thrust::get<0>(skip_calc)) {
    if (threadIdx.x == 0) { upd_0.tmp_vars_bnd[blockIdx.x] = thrust::get<0>(old_bounds); }
    bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
    i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
    i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);
    auto bounds      = update_bounds<i_t, f_t, f_t2, BDIM>(view,
                                                      upd_1.cnst_slack,
                                                      upd_1.changed_constraints,
                                                      threadIdx.x,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<1>(old_bounds));
    bounds.x         = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
    __syncthreads();
    bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());
    if (threadIdx.x == 0) {
      bool changed = write_updated_bounds(
        &upd_1.tmp_vars_bnd[blockIdx.x], is_int, view, upd_1, bounds, thrust::get<1>(old_bounds));
      atomicExch(&upd_1.var_bounds_changed[var_idx], 1);
    }
  } else if (thrust::get<1>(skip_calc)) {
    if (threadIdx.x == 0) { upd_1.tmp_vars_bnd[blockIdx.x] = thrust::get<1>(old_bounds); }
    bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
    i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
    i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);
    auto bounds      = update_bounds<i_t, f_t, f_t2, BDIM>(view,
                                                      upd_0.cnst_slack,
                                                      upd_0.changed_constraints,
                                                      threadIdx.x,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<0>(old_bounds));
    bounds.x         = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
    __syncthreads();
    bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());
    if (threadIdx.x == 0) {
      bool changed = write_updated_bounds(
        &upd_0.tmp_vars_bnd[blockIdx.x], is_int, view, upd_0, bounds, thrust::get<0>(old_bounds));
      atomicExch(&upd_0.var_bounds_changed[var_idx], 1);
    }
  } else {
    bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
    i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
    i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);
    auto bounds      = update_bounds<i_t, f_t, f_t2, BDIM>(
      view, upd_0, upd_1, threadIdx.x, item_off_beg, item_off_end, old_bounds);
    thrust::get<0>(bounds).x =
      BlockReduce(temp_storage).Reduce(thrust::get<0>(bounds).x, cub::Max());
    __syncthreads();
    thrust::get<0>(bounds).y =
      BlockReduce(temp_storage).Reduce(thrust::get<0>(bounds).y, cub::Min());
    __syncthreads();
    thrust::get<1>(bounds).x =
      BlockReduce(temp_storage).Reduce(thrust::get<1>(bounds).x, cub::Max());
    __syncthreads();
    thrust::get<1>(bounds).y =
      BlockReduce(temp_storage).Reduce(thrust::get<1>(bounds).y, cub::Min());
    if (threadIdx.x == 0) {
      bool changed = write_updated_bounds(&upd_0.tmp_vars_bnd[blockIdx.x],
                                          is_int,
                                          view,
                                          upd_0,
                                          thrust::get<0>(bounds),
                                          thrust::get<0>(old_bounds));
      atomicExch(&upd_0.var_bounds_changed[var_idx], 1);
      changed = write_updated_bounds(&upd_1.tmp_vars_bnd[blockIdx.x],
                                     is_int,
                                     view,
                                     upd_1,
                                     thrust::get<1>(bounds),
                                     thrust::get<1>(old_bounds));
      atomicExch(&upd_1.var_bounds_changed[var_idx], 1);
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          typename bounds_update_view_t,
          typename upd_view_t>
__global__ void finalize_upd_bnd_kernel(i_t heavy_vars_beg_id,
                                        raft::device_span<const i_t> item_offsets,
                                        bounds_update_view_t view,
                                        upd_view_t upd_0,
                                        upd_view_t upd_1)
{
  i_t idx     = heavy_vars_beg_id + blockIdx.x;
  i_t var_idx = view.vars_reorg_ids[idx];

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  // auto skip_calc = thrust::make_pair(
  //  (upd_0.var_bounds_changed[var_idx] == 0) || (upd_0.changed_variables[var_idx] == 0),
  //  (upd_1.var_bounds_changed[var_idx] == 0) || (upd_1.changed_variables[var_idx] == 0));
  auto skip_calc = skip_update(upd_0, upd_1, var_idx);

  using warp_reduce = cub::WarpReduce<f_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage;
  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) {
    return;
  } else if (thrust::get<0>(skip_calc)) {
    // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
    i_t item_off_beg = item_offsets[blockIdx.x];
    i_t item_off_end = item_offsets[blockIdx.x + 1];
    f_t2 bounds = f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
    // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto bnd = upd_1.tmp_vars_bnd[i];
      bounds.x = max(bounds.x, bnd.x);
      bounds.y = min(bounds.y, bnd.y);
    }
    bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
    __syncwarp();
    bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());
    if (threadIdx.x == 0) { upd_1.vars_bnd[var_idx] = bounds; }
  } else if (thrust::get<1>(skip_calc)) {
    // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
    i_t item_off_beg = item_offsets[blockIdx.x];
    i_t item_off_end = item_offsets[blockIdx.x + 1];
    f_t2 bounds = f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
    // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto bnd = upd_0.tmp_vars_bnd[i];
      bounds.x = max(bounds.x, bnd.x);
      bounds.y = min(bounds.y, bnd.y);
    }
    bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
    __syncwarp();
    bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());
    if (threadIdx.x == 0) { upd_0.vars_bnd[var_idx] = bounds; }
  } else {
    // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
    i_t item_off_beg = item_offsets[blockIdx.x];
    i_t item_off_end = item_offsets[blockIdx.x + 1];
    f_t2 bounds_0 =
      f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
    f_t2 bounds_1 =
      f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
    // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto bnd_0 = upd_0.tmp_vars_bnd[i];
      bounds_0.x = max(bounds_0.x, bnd_0.x);
      bounds_0.y = min(bounds_0.y, bnd_0.y);
      auto bnd_1 = upd_1.tmp_vars_bnd[i];
      bounds_1.x = max(bounds_1.x, bnd_1.x);
      bounds_1.y = min(bounds_1.y, bnd_1.y);
    }
    bounds_0.x = warp_reduce(temp_storage).Reduce(bounds_0.x, cub::Max());
    __syncwarp();
    bounds_0.y = warp_reduce(temp_storage).Reduce(bounds_0.y, cub::Min());
    __syncwarp();
    bounds_1.x = warp_reduce(temp_storage).Reduce(bounds_1.x, cub::Max());
    __syncwarp();
    bounds_1.y = warp_reduce(temp_storage).Reduce(bounds_1.y, cub::Min());
    if (threadIdx.x == 0) {
      upd_0.vars_bnd[var_idx] = bounds_0;
      upd_1.vars_bnd[var_idx] = bounds_1;
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t,
          typename upd_view_t>
__device__ void upd_bnd_sub_warp(
  i_t id_warp_beg, i_t id_range_end, bounds_update_view_t view, upd_view_t upd_0, upd_view_t upd_1)
{
  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_VAR);
  i_t var_idx;
  auto old_bounds = thrust::make_pair(
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()});
  auto bounds = old_bounds;
  bool is_int = false;

  auto skip_calc = thrust::make_pair(!(idx < id_range_end), !(idx < id_range_end));

  if (idx < id_range_end) {
    var_idx                    = view.vars_reorg_ids[idx];
    thrust::get<0>(old_bounds) = upd_0.vars_bnd[var_idx];
    thrust::get<1>(old_bounds) = upd_1.vars_bnd[var_idx];
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    // skip unchanged variables
    skip_calc =
      skip_update(old_bounds, upd_0, upd_1, var_idx, view.tolerances.integrality_tolerance);
    is_int = (view.vars_types[idx] == var_t::INTEGER);
  }
  // Equivalent to
  // i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;
  i_t p_tid = lane_id & (MAX_EDGE_PER_VAR - 1);

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) {
    return;
  } else if (thrust::get<0>(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    thrust::get<1>(bounds) =
      update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(view,
                                                      upd_1.cnst_slack,
                                                      upd_1.changed_constraints,
                                                      p_tid,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<1>(old_bounds));
  } else if (thrust::get<1>(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    thrust::get<0>(bounds) =
      update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(view,
                                                      upd_0.cnst_slack,
                                                      upd_0.changed_constraints,
                                                      p_tid,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<0>(old_bounds));
  } else {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    bounds = update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(
      view, upd_0, upd_1, p_tid, item_off_beg, item_off_end, old_bounds);
  }

  thrust::get<0>(bounds).x = warp_reduce(temp_storage).Reduce(thrust::get<0>(bounds).x, cub::Max());
  __syncwarp();
  thrust::get<0>(bounds).y = warp_reduce(temp_storage).Reduce(thrust::get<0>(bounds).y, cub::Min());
  __syncwarp();
  thrust::get<1>(bounds).x = warp_reduce(temp_storage).Reduce(thrust::get<1>(bounds).x, cub::Max());
  __syncwarp();
  thrust::get<1>(bounds).y = warp_reduce(temp_storage).Reduce(thrust::get<1>(bounds).y, cub::Min());

  if (head_flag && !thrust::get<0>(skip_calc)) {
    bool changed = write_updated_bounds(&upd_0.vars_bnd[var_idx],
                                        is_int,
                                        view,
                                        upd_0,
                                        thrust::get<0>(bounds),
                                        thrust::get<0>(old_bounds));
    if (changed) { upd_0.var_bounds_changed[var_idx] = changed; }
  }
  if (head_flag && !thrust::get<1>(skip_calc)) {
    bool changed = write_updated_bounds(&upd_1.vars_bnd[var_idx],
                                        is_int,
                                        view,
                                        upd_1,
                                        thrust::get<1>(bounds),
                                        thrust::get<1>(old_bounds));
    if (changed) { upd_1.var_bounds_changed[var_idx] = changed; }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename bounds_update_view_t,
          typename upd_view_t>
__global__ void lb_upd_bnd_sub_warp_kernel(bounds_update_view_t view,
                                           upd_view_t upd_0,
                                           upd_view_t upd_1,
                                           raft::device_span<i_t> warp_vars_offsets,
                                           raft::device_span<i_t> warp_vars_id_offsets)
{
  i_t id_warp_beg, id_range_end, threads_per_variable;
  detect_range_sub_warp<i_t>(
    &id_warp_beg, &id_range_end, &threads_per_variable, warp_vars_offsets, warp_vars_id_offsets);

  if (threads_per_variable == 1) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 1>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 2) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 2>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 4) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 4>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 8) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 8>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 16) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 16>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t,
          typename upd_view_t>
__global__ void lb_upd_bnd_sub_warp_kernel(
  i_t id_range_beg, i_t id_range_end, bounds_update_view_t view, upd_view_t upd_0, upd_view_t upd_1)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_VAR;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_VAR);
  i_t var_idx;
  auto old_bounds = thrust::make_pair(
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()});
  auto bounds = old_bounds;
  bool is_int = false;

  auto skip_calc = thrust::make_pair(!(idx < id_range_end), !(idx < id_range_end));

  if (idx < id_range_end) {
    var_idx                    = view.vars_reorg_ids[idx];
    thrust::get<0>(old_bounds) = upd_0.vars_bnd[var_idx];
    thrust::get<1>(old_bounds) = upd_1.vars_bnd[var_idx];
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    // skip unchanged variables
    skip_calc =
      skip_update(old_bounds, upd_0, upd_1, var_idx, view.tolerances.integrality_tolerance);
    is_int = (view.vars_types[idx] == var_t::INTEGER);
  }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) {
    return;
  } else if (thrust::get<0>(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    thrust::get<1>(bounds) =
      update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(view,
                                                      upd_1.cnst_slack,
                                                      upd_1.changed_constraints,
                                                      p_tid,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<1>(old_bounds));
  } else if (thrust::get<1>(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    thrust::get<0>(bounds) =
      update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(view,
                                                      upd_0.cnst_slack,
                                                      upd_0.changed_constraints,
                                                      p_tid,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<0>(old_bounds));
  } else {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    bounds = update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(
      view, upd_0, upd_1, p_tid, item_off_beg, item_off_end, old_bounds);
  }

  thrust::get<0>(bounds).x = warp_reduce(temp_storage).Reduce(thrust::get<0>(bounds).x, cub::Max());
  __syncwarp();
  thrust::get<0>(bounds).y = warp_reduce(temp_storage).Reduce(thrust::get<0>(bounds).y, cub::Min());
  __syncwarp();
  thrust::get<1>(bounds).x = warp_reduce(temp_storage).Reduce(thrust::get<1>(bounds).x, cub::Max());
  __syncwarp();
  thrust::get<1>(bounds).y = warp_reduce(temp_storage).Reduce(thrust::get<1>(bounds).y, cub::Min());

  if (head_flag && !thrust::get<0>(skip_calc)) {
    bool changed = write_updated_bounds(&upd_0.vars_bnd[var_idx],
                                        is_int,
                                        view,
                                        upd_0,
                                        thrust::get<0>(bounds),
                                        thrust::get<0>(old_bounds));
    if (changed) { upd_0.var_bounds_changed[var_idx] = changed; }
  }
  if (head_flag && !thrust::get<1>(skip_calc)) {
    bool changed = write_updated_bounds(&upd_1.vars_bnd[var_idx],
                                        is_int,
                                        view,
                                        upd_1,
                                        thrust::get<1>(bounds),
                                        thrust::get<1>(old_bounds));
    if (changed) { upd_1.var_bounds_changed[var_idx] = changed; }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename bounds_update_view_t,
          typename upd_view_t>
__global__ void lb_upd_bnd_block_kernel(i_t id_range_beg,
                                        bounds_update_view_t view,
                                        upd_view_t upd_0,
                                        upd_view_t upd_1)
{
  i_t idx         = id_range_beg + blockIdx.x;
  i_t var_idx     = view.vars_reorg_ids[idx];
  auto old_bounds = thrust::make_pair(upd_0.vars_bnd[var_idx], upd_1.vars_bnd[var_idx]);

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  // auto skip_calc = thrust::make_pair(
  //  skip_update(thrust::get<0>(old_bounds), view.tolerances.integrality_tolerance) ||
  //  (upd_0.changed_variables[var_idx] == 0), skip_update(thrust::get<1>(old_bounds),
  //  view.tolerances.integrality_tolerance) || (upd_1.changed_variables[var_idx] == 0));
  auto skip_calc =
    skip_update(old_bounds, upd_0, upd_1, var_idx, view.tolerances.integrality_tolerance);

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  if (thrust::get<0>(skip_calc) && thrust::get<1>(skip_calc)) {
    return;
  } else if (thrust::get<0>(skip_calc)) {
    bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    auto bounds      = update_bounds<i_t, f_t, f_t2, BDIM>(view,
                                                      upd_1.cnst_slack,
                                                      upd_1.changed_constraints,
                                                      threadIdx.x,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<1>(old_bounds));

    bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
    __syncthreads();
    bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

    if (threadIdx.x == 0) {
      bool changed = write_updated_bounds(
        &upd_1.vars_bnd[var_idx], is_int, view, upd_1, bounds, thrust::get<1>(old_bounds));
      if (changed) { upd_1.var_bounds_changed[var_idx] = changed; }
    }
  } else if (thrust::get<1>(skip_calc)) {
    bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    auto bounds      = update_bounds<i_t, f_t, f_t2, BDIM>(view,
                                                      upd_0.cnst_slack,
                                                      upd_0.changed_constraints,
                                                      threadIdx.x,
                                                      item_off_beg,
                                                      item_off_end,
                                                      thrust::get<0>(old_bounds));

    bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
    __syncthreads();
    bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

    if (threadIdx.x == 0) {
      bool changed = write_updated_bounds(
        &upd_0.vars_bnd[var_idx], is_int, view, upd_0, bounds, thrust::get<0>(old_bounds));
      if (changed) { upd_0.var_bounds_changed[var_idx] = changed; }
    }
  } else {
    bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    auto bounds      = update_bounds<i_t, f_t, f_t2, BDIM>(
      view, upd_0, upd_1, threadIdx.x, item_off_beg, item_off_end, old_bounds);

    thrust::get<0>(bounds).x =
      BlockReduce(temp_storage).Reduce(thrust::get<0>(bounds).x, cub::Max());
    __syncthreads();
    thrust::get<0>(bounds).y =
      BlockReduce(temp_storage).Reduce(thrust::get<0>(bounds).y, cub::Min());
    __syncthreads();
    thrust::get<1>(bounds).x =
      BlockReduce(temp_storage).Reduce(thrust::get<1>(bounds).x, cub::Max());
    __syncthreads();
    thrust::get<1>(bounds).y =
      BlockReduce(temp_storage).Reduce(thrust::get<1>(bounds).y, cub::Min());

    if (threadIdx.x == 0) {
      bool changed = write_updated_bounds(&upd_0.vars_bnd[var_idx],
                                          is_int,
                                          view,
                                          upd_0,
                                          thrust::get<0>(bounds),
                                          thrust::get<0>(old_bounds));
      if (changed) { upd_0.var_bounds_changed[var_idx] = changed; }
      changed = write_updated_bounds(&upd_1.vars_bnd[var_idx],
                                     is_int,
                                     view,
                                     upd_1,
                                     thrust::get<1>(bounds),
                                     thrust::get<1>(old_bounds));
      if (changed) { upd_1.var_bounds_changed[var_idx] = changed; }
    }
  }
}

#endif

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_bnd_sub_warp_kernel(bounds_update_view_t view,
                                           raft::device_span<i_t> warp_vars_offsets,
                                           raft::device_span<i_t> warp_vars_id_offsets)
{
  i_t id_warp_beg, id_range_end, threads_per_variable;
  detect_range_sub_warp<i_t>(
    &id_warp_beg, &id_range_end, &threads_per_variable, warp_vars_offsets, warp_vars_id_offsets);

  if (threads_per_variable == 1) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 1>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 2) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 2>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 4) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 4>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 8) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 8>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 16) {
    upd_bnd_sub_warp<i_t, f_t, f_t2, BDIM, 16>(id_warp_beg, id_range_end, view);
  }
}

}  // namespace cuopt::linear_programming::detail
