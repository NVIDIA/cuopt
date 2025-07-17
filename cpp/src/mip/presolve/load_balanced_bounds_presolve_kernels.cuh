/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mip/problem/problem.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t>
__device__ __forceinline__ void detect_range_sub_warp(i_t* id_warp_beg,
                                                      i_t* id_range_end,
                                                      i_t* threads_per_item,
                                                      raft::device_span<i_t> warp_offsets,
                                                      raft::device_span<i_t> bin_offsets)
{
  i_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  i_t lane_id = threadIdx.x & 31;
  bool pred   = false;
  if (lane_id < warp_offsets.size()) { pred = (warp_id >= warp_offsets[lane_id]); }
  unsigned int m  = __ballot_sync(0xffffffff, pred);
  i_t seg         = 31 - __clz(m);
  i_t it_per_warp = (1 << (5 - seg));  // item per warp = raft::WarpSize/(2^seg)
  if (5 - seg < 0) {
    *threads_per_item = 0;
    return;
  }
  i_t beg           = bin_offsets[seg] + (warp_id - warp_offsets[seg]) * it_per_warp;
  i_t end           = bin_offsets[seg + 1];
  *id_warp_beg      = beg;
  *id_range_end     = end;
  *threads_per_item = (1 << seg);
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

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename activity_view_t>
__global__ void lb_calc_act_heavy_kernel(i_t id_range_beg,
                                         raft::device_span<const i_t> ids,
                                         raft::device_span<const i_t> pseudo_block_ids,
                                         i_t work_per_block,
                                         activity_view_t view,
                                         raft::device_span<f_t2> tmp_cnst_act)
{
  auto idx             = ids[blockIdx.x] + id_range_beg;
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

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename activity_view_t>
__global__ void lb_calc_act_heavy_kernel(i_t id_range_beg,
                                         raft::device_span<const i_t> item_block_segments,
                                         i_t work_per_block,
                                         activity_view_t view,
                                         raft::device_span<f_t2> tmp_cnst_act)
{
  __shared__ i_t id_map;
  __shared__ i_t pseudo_block_id;
  if (threadIdx.x == 0) {
    id_map = thrust::upper_bound(
               thrust::seq, item_block_segments.begin(), item_block_segments.end(), blockIdx.x) -
             item_block_segments.begin();
    pseudo_block_id = blockIdx.x - item_block_segments[id_map - 1];
  }
  __syncthreads();
  auto idx         = id_range_beg + id_map;
  i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

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

template <bool erase_inf_cnst, typename i_t, typename f_t, typename f_t2, typename activity_view_t>
__global__ void finalize_calc_act_kernel(i_t heavy_cnst_beg_id,
                                         raft::device_span<const i_t> item_offsets,
                                         raft::device_span<f_t2> tmp_act,
                                         activity_view_t view)
{
  using warp_reduce = cub::WarpReduce<f_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage;
  i_t idx                  = heavy_cnst_beg_id + blockIdx.x;
  i_t cnst_idx             = view.cnst_reorg_ids[idx];
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

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          typename f_t2,
          i_t BDIM,
          typename activity_view_t>
__global__ void lb_calc_act_block_kernel(i_t id_range_beg, activity_view_t view)

{
  i_t idx                  = id_range_beg + blockIdx.x;
  i_t cnst_idx             = view.cnst_reorg_ids[idx];
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
          i_t MAX_EDGE_PER_CNST,
          typename activity_view_t>
__global__ void lb_calc_act_sub_warp_kernel_test(i_t id_range_beg,
                                                 i_t id_range_end,
                                                 activity_view_t view)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_CNST;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_CNST);
  i_t cnst_idx;
  f_t eps;
  f_t2 cnst_lb_ub;
  if (idx < id_range_end) {
    cnst_idx   = view.cnst_reorg_ids[idx];
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

  if (idx < id_range_end) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act = calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, p_tid, item_off_beg, item_off_end);
  }

  act.x = warp_reduce(temp_storage).Sum(act.x);
  __syncwarp();
  act.y = warp_reduce(temp_storage).Sum(act.y);

  if (head_flag && (idx < id_range_end)) {
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
__global__ void lb_calc_act_sub_warp_kernel(i_t id_range_beg,
                                            i_t id_range_end,
                                            activity_view_t view)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_CNST;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_CNST);
  i_t cnst_idx;
  [[maybe_unused]] f_t eps = {};
  f_t2 cnst_lb_ub;
  if (idx < id_range_end) {
    cnst_idx   = view.cnst_reorg_ids[idx];
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

  if (idx < id_range_end) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act = calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, p_tid, item_off_beg, item_off_end);
  }

  act.x = warp_reduce(temp_storage).Sum(act.x);
  __syncwarp();
  act.y = warp_reduce(temp_storage).Sum(act.y);

  if (head_flag && (idx < id_range_end)) {
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
  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  i_t cnst_idx;
  [[maybe_unused]] f_t eps = {};
  f_t2 cnst_lb_ub;
  if (idx < id_range_end) {
    cnst_idx   = view.cnst_reorg_ids[idx];
    cnst_lb_ub = view.cnst_bnd[idx];
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

  if (idx < id_range_end) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act = calc_act<i_t, f_t, f_t2, MAX_EDGE_PER_CNST>(view, p_tid, item_off_beg, item_off_end);
  }

  act.x = warp_reduce(temp_storage).Sum(act.x);
  __syncwarp();
  act.y = warp_reduce(temp_storage).Sum(act.y);

  if (head_flag && (idx < id_range_end)) {
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

/// BOUNDS UPDATE

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t>
__device__ f_t2 update_bounds(bounds_update_view_t view, i_t tid, i_t beg, i_t end, f_t2 init)
{
  f_t2 bounds = init;

  const auto old_lb = bounds.x;
  const auto old_ub = bounds.y;

  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    auto a        = view.coeff[i];
    auto cnst_idx = view.cnst[i];

    // cnst_slack[cnst_idx].x now has cnst_ub - min_a
    // cnst_slack[cnst_idx].y now has cnst_lb - max_a
    auto cnstr_data           = view.cnst_slack[cnst_idx];
    auto cnstr_ub_minus_min_a = cnstr_data.x;
    auto cnstr_lb_minus_max_a = cnstr_data.y;
    //  don't propagate over constraints that are infeasible
    if (isnan(cnstr_data.x)) { continue; }

    f_t min_contrib = old_lb;
    f_t max_contrib = old_ub;
    if (a < 0.0) {
      min_contrib = old_ub;
      max_contrib = old_lb;
    }

    auto delta_min_act = (cnstr_ub_minus_min_a + (a * min_contrib)) / a;
    auto delta_max_act = (cnstr_lb_minus_max_a + (a * max_contrib)) / a;

    f_t lb_contrib = delta_max_act;
    f_t ub_contrib = delta_min_act;
    if (a < 0.0) {
      lb_contrib = delta_min_act;
      ub_contrib = delta_max_act;
    }
    bounds.x = max(bounds.x, lb_contrib);
    bounds.y = min(bounds.y, ub_contrib);
  }

  return bounds;
}

template <typename f_t2, typename bounds_update_view_t>
inline __device__ void write_updated_bounds(
  f_t2* ptr, bool is_int, bounds_update_view_t view, f_t2 bounds, f_t2 old_bounds)
{
  auto threshold = 1e3 * view.tolerances.absolute_tolerance;
  if (is_int) {
    bounds.x = ceil(bounds.x - view.tolerances.integrality_tolerance);
    bounds.y = floor(bounds.y + view.tolerances.integrality_tolerance);
  }
  auto lb_updated = (fabs(bounds.x - old_bounds.x) > threshold);
  auto ub_updated = (fabs(bounds.y - old_bounds.y) > threshold);

  if (lb_updated) { old_bounds.x = bounds.x; }
  if (ub_updated) { old_bounds.y = bounds.y; }

  *ptr = old_bounds;

  if (lb_updated || ub_updated) { atomicAdd(view.bounds_changed, 1); }
}

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_bnd_heavy_kernel(i_t id_range_beg,
                                        raft::device_span<const i_t> ids,
                                        raft::device_span<const i_t> pseudo_block_ids,
                                        i_t work_per_block,
                                        bounds_update_view_t view,
                                        raft::device_span<f_t2> tmp_bnd)
{
  auto idx             = ids[blockIdx.x] + id_range_beg;
  auto pseudo_block_id = pseudo_block_ids[blockIdx.x];
  auto var_idx         = view.vars_reorg_ids[idx];
  // x is lb, y is ub
  auto old_bounds  = view.vars_bnd[var_idx];
  bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
  i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) {
    tmp_bnd[blockIdx.x] = old_bounds;
    return;
  }
  auto bounds =
    update_bounds<i_t, f_t, f_t2, BDIM>(view, threadIdx.x, item_off_beg, item_off_end, old_bounds);

  bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cuda::maximum());
  __syncthreads();
  bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cuda::minimum());

  if (threadIdx.x == 0) {
    write_updated_bounds(&tmp_bnd[blockIdx.x], is_int, view, bounds, old_bounds);
  }
}

template <typename i_t, typename f_t, typename f_t2, typename bounds_update_view_t>
__global__ void finalize_upd_bnd_kernel(i_t heavy_vars_beg_id,
                                        raft::device_span<const i_t> item_offsets,
                                        raft::device_span<f_t2> tmp_bnd,
                                        bounds_update_view_t view)
{
  using warp_reduce = cub::WarpReduce<f_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage;
  i_t idx     = heavy_vars_beg_id + blockIdx.x;
  i_t var_idx = view.vars_reorg_ids[idx];

  // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
  i_t item_off_beg = item_offsets[blockIdx.x];
  i_t item_off_end = item_offsets[blockIdx.x + 1];
  f_t2 bounds = f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
  for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
    auto bnd = tmp_bnd[i];
    bounds.x = max(bounds.x, bnd.x);
    bounds.y = min(bounds.y, bnd.y);
  }
  bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cuda::maximum());
  __syncwarp();
  bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cuda::minimum());
  if (threadIdx.x == 0) { view.vars_bnd[var_idx] = bounds; }
}

template <typename i_t, typename f_t, typename f_t2, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_bnd_block_kernel(i_t id_range_beg, bounds_update_view_t view)
{
  i_t idx     = id_range_beg + blockIdx.x;
  i_t var_idx = view.vars_reorg_ids[idx];
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
  auto bounds =
    update_bounds<i_t, f_t, f_t2, BDIM>(view, threadIdx.x, item_off_beg, item_off_end, old_bounds);

  bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cuda::maximum());
  __syncthreads();
  bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cuda::minimum());

  if (threadIdx.x == 0) {
    write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds, old_bounds);
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
  auto bounds = old_bounds;
  bool is_int = false;
  if (idx < id_range_end) {
    var_idx    = view.vars_reorg_ids[idx];
    old_bounds = view.vars_bnd[var_idx];
    is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (idx < id_range_end) {
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { head_flag = 0; }

    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    bounds = update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(
      view, p_tid, item_off_beg, item_off_end, old_bounds);
  }

  bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cuda::maximum());
  __syncwarp();
  bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cuda::minimum());

  if (head_flag && (idx < id_range_end)) {
    write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds, old_bounds);
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
  auto bounds = old_bounds;
  bool is_int = false;
  if (idx < id_range_end) {
    var_idx    = view.vars_reorg_ids[idx];
    old_bounds = view.vars_bnd[var_idx];
    is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  // Equivalent to
  // i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;
  i_t p_tid = lane_id & (MAX_EDGE_PER_VAR - 1);

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (idx < id_range_end) {
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { head_flag = 0; }

    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    bounds = update_bounds<i_t, f_t, f_t2, MAX_EDGE_PER_VAR>(
      view, p_tid, item_off_beg, item_off_end, old_bounds);
  }

  bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cuda::maximum());
  __syncwarp();
  bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cuda::minimum());

  if (head_flag && (idx < id_range_end)) {
    write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds, old_bounds);
  }
}

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
