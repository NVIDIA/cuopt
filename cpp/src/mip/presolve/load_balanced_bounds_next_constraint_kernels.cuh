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
#include "load_balanced_bounds_common_kernels.cuh"

namespace cuopt::linear_programming::detail {

/// BOUNDS UPDATE

template <typename i_t, typename upd_view_t>
inline __device__ thrust::pair<bool, bool> skip_mark(upd_view_t upd_0,
                                                     upd_view_t upd_1,
                                                     i_t var_idx)
{
  return thrust::make_pair((upd_0.var_bounds_changed[var_idx] == i_t{0}),
                           (upd_1.var_bounds_changed[var_idx] == i_t{0}));
}

template <typename i_t,
          typename f_t,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t,
          typename upd_view_t>
__device__ void update_next_constraints(
  bounds_update_view_t view, upd_view_t upd, i_t tid, i_t beg, i_t end)
{
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    // auto a        = view.coeff[i];
    auto cnst_idx = view.cnst[i];
    atomicExch(&upd.next_changed_constraints[cnst_idx], 1);
  }
}

template <typename i_t,
          typename f_t,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t,
          typename upd_view_t>
__device__ void update_next_constraints(
  bounds_update_view_t view, upd_view_t upd_0, upd_view_t upd_1, i_t tid, i_t beg, i_t end)
{
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    // auto a        = view.coeff[i];
    auto cnst_idx = view.cnst[i];
    atomicExch(&upd_0.next_changed_constraints[cnst_idx], 1);
    atomicExch(&upd_1.next_changed_constraints[cnst_idx], 1);
  }
}

template <typename i_t, typename f_t, i_t MAX_EDGE_PER_VAR, typename bounds_update_view_t>
__device__ void update_next_constraints(bounds_update_view_t view, i_t tid, i_t beg, i_t end)
{
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    // auto a        = view.coeff[i];
    auto cnst_idx = view.cnst[i];
    atomicExch(&view.next_changed_constraints[cnst_idx], 1);
  }
}

template <typename i_t, typename f_t, i_t BDIM, typename bounds_update_view_t, typename upd_view_t>
__global__ void lb_upd_next_constraint_heavy_kernel(i_t id_range_beg,
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
  auto skip_mark_next  = skip_mark(upd_0, upd_1, var_idx);
  // x is lb, y is ub
  // auto old_bounds  = view.vars_bnd[var_idx];
  // bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
  i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  // typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  //__shared__ typename BlockReduce::TempStorage temp_storage;

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  // if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) {
  //  tmp_bnd[blockIdx.x] = old_bounds;
  //  return;
  //}
  // auto bounds =
  if (thrust::get<1>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, BDIM>(view, upd_0, threadIdx.x, item_off_beg, item_off_end);
  } else if (thrust::get<0>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, BDIM>(view, upd_1, threadIdx.x, item_off_beg, item_off_end);
  } else {
    update_next_constraints<i_t, f_t, BDIM>(
      view, upd_0, upd_1, threadIdx.x, item_off_beg, item_off_end);
  }

  // bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncthreads();
  // bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (threadIdx.x == 0) {
  //   bool changed = write_updated_bounds(&tmp_bnd[blockIdx.x], is_int, view, bounds, old_bounds);
  //   atomicExch(&view.var_bounds_changed[var_idx], 1);
  // }
}

template <typename i_t, typename f_t, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_next_constraint_heavy_kernel(i_t id_range_beg,
                                                    raft::device_span<const i_t> ids,
                                                    raft::device_span<const i_t> pseudo_block_ids,
                                                    i_t work_per_block,
                                                    bounds_update_view_t view)
{
  auto idx             = ids[blockIdx.x] + id_range_beg;
  auto pseudo_block_id = pseudo_block_ids[blockIdx.x];
  auto var_idx         = view.vars_reorg_ids[idx];
  if (view.var_bounds_changed[var_idx] == 0) { return; }
  // x is lb, y is ub
  // auto old_bounds  = view.vars_bnd[var_idx];
  // bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
  i_t item_off_beg = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  // typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  //__shared__ typename BlockReduce::TempStorage temp_storage;

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  // if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) {
  //  tmp_bnd[blockIdx.x] = old_bounds;
  //  return;
  //}
  // auto bounds =
  update_next_constraints<i_t, f_t, BDIM>(view, threadIdx.x, item_off_beg, item_off_end);

  // bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncthreads();
  // bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (threadIdx.x == 0) {
  //   bool changed = write_updated_bounds(&tmp_bnd[blockIdx.x], is_int, view, bounds, old_bounds);
  //   atomicExch(&view.var_bounds_changed[var_idx], 1);
  // }
}

// template <typename i_t, typename f_t, typename bounds_update_view_t>
//__global__ void finalize_upd_next_constraint_kernel(i_t heavy_vars_beg_id,
//                                         raft::device_span<const i_t> item_offsets,
//                                         raft::device_span<f_t2> tmp_bnd,
//                                         bounds_update_view_t view)
//{
//   using warp_reduce = cub::WarpReduce<f_t>;
//   __shared__ typename warp_reduce::TempStorage temp_storage;
//   i_t idx     = heavy_vars_beg_id + blockIdx.x;
//   i_t var_idx = view.vars_reorg_ids[idx];
//   if (view.var_bounds_changed[var_idx] == 0) { return; }
//
//   // assumes cnst_bnd[i].x has ub and cnst_bnd[i].y has lb
//   i_t item_off_beg = item_offsets[blockIdx.x];
//   i_t item_off_end = item_offsets[blockIdx.x + 1];
//   f_t2 bounds = f_t2{-std::numeric_limits<f_t>::infinity(),
//   std::numeric_limits<f_t>::infinity()};
//   // assumes tmp_act[i].x has min activity and tmp_act[i].y has max activity
//   for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
//     auto bnd = tmp_bnd[i];
//     bounds.x = max(bounds.x, bnd.x);
//     bounds.y = min(bounds.y, bnd.y);
//   }
//   bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
//   __syncwarp();
//   bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());
//   if (threadIdx.x == 0) { view.vars_bnd[var_idx] = bounds; }
// }

template <typename i_t, typename f_t, i_t BDIM, typename bounds_update_view_t, typename upd_view_t>
__global__ void lb_upd_next_constraint_block_kernel(i_t id_range_beg,
                                                    bounds_update_view_t view,
                                                    upd_view_t upd_0,
                                                    upd_view_t upd_1)
{
  i_t idx             = id_range_beg + blockIdx.x;
  i_t var_idx         = view.vars_reorg_ids[idx];
  auto skip_mark_next = skip_mark(upd_0, upd_1, var_idx);
  // x is lb, y is ub
  // auto old_bounds  = view.vars_bnd[var_idx];
  // bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  // typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  //__shared__ typename BlockReduce::TempStorage temp_storage;

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  // if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { return; }
  // auto bounds =
  if (thrust::get<1>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, BDIM>(view, upd_0, threadIdx.x, item_off_beg, item_off_end);
  } else if (thrust::get<0>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, BDIM>(view, upd_1, threadIdx.x, item_off_beg, item_off_end);
  } else {
    update_next_constraints<i_t, f_t, BDIM>(
      view, upd_0, upd_1, threadIdx.x, item_off_beg, item_off_end);
  }

  // bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncthreads();
  // bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (threadIdx.x == 0) {
  //   bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds,
  //   old_bounds); view.var_bounds_changed[var_idx] = changed;
  // }
}

template <typename i_t, typename f_t, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_next_constraint_block_kernel(i_t id_range_beg, bounds_update_view_t view)
{
  i_t idx     = id_range_beg + blockIdx.x;
  i_t var_idx = view.vars_reorg_ids[idx];
  if (view.var_bounds_changed[var_idx] == 0) { return; }
  // x is lb, y is ub
  // auto old_bounds  = view.vars_bnd[var_idx];
  // bool is_int      = (view.vars_types[idx] == var_t::INTEGER);
  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  // typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  //__shared__ typename BlockReduce::TempStorage temp_storage;

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  // if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { return; }
  // auto bounds =
  update_next_constraints<i_t, f_t, BDIM>(view, threadIdx.x, item_off_beg, item_off_end);

  // bounds.x = BlockReduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncthreads();
  // bounds.y = BlockReduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (threadIdx.x == 0) {
  //   bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds,
  //   old_bounds); view.var_bounds_changed[var_idx] = changed;
  // }
}

template <typename i_t,
          typename f_t,
          i_t BDIM,
          i_t MAX_EDGE_PER_VAR,
          typename activity_view_t,
          typename upd_view_t>
__global__ void lb_upd_next_constraint_sub_warp_kernel(
  i_t id_range_beg, i_t id_range_end, activity_view_t view, upd_view_t upd_0, upd_view_t upd_1)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_VAR;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_VAR);
  i_t var_idx;
  // auto old_bounds =
  //   f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  // auto bounds = old_bounds;
  // bool is_int = false;
  auto skip_mark_next = thrust::make_pair(!(idx < id_range_end), !(idx < id_range_end));
  if (idx < id_range_end) {
    var_idx        = view.vars_reorg_ids[idx];
    skip_mark_next = skip_mark(upd_0, upd_1, var_idx);
    // old_bounds = view.vars_bnd[var_idx];
    // is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;

  // i_t head_flag = (p_tid == 0);

  // using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  //__shared__ typename warp_reduce::TempStorage temp_storage;

  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];
  if (thrust::get<1>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(
      view, upd_0, p_tid, item_off_beg, item_off_end);
  } else if (thrust::get<0>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(
      view, upd_1, p_tid, item_off_beg, item_off_end);
  } else {
    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(
      view, upd_0, upd_1, p_tid, item_off_beg, item_off_end);
  }

  // bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncwarp();
  // bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (head_flag && continue_calc) {
  //   bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds,
  //   old_bounds); view.var_bounds_changed[var_idx] = changed;
  // }
}

template <typename i_t, typename f_t, i_t BDIM, i_t MAX_EDGE_PER_VAR, typename activity_view_t>
__global__ void lb_upd_next_constraint_sub_warp_kernel(i_t id_range_beg,
                                                       i_t id_range_end,
                                                       activity_view_t view)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_VAR;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_VAR);
  i_t var_idx;
  // auto old_bounds =
  //   f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  // auto bounds = old_bounds;
  // bool is_int = false;
  bool continue_calc = (idx < id_range_end);
  if (continue_calc) {
    var_idx = view.vars_reorg_ids[idx];
    if (view.var_bounds_changed[var_idx] == 0) { continue_calc = false; }
    // old_bounds = view.vars_bnd[var_idx];
    // is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;

  // i_t head_flag = (p_tid == 0);

  // using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  //__shared__ typename warp_reduce::TempStorage temp_storage;

  if (continue_calc) {
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    // if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { head_flag = 0; }

    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(view, p_tid, item_off_beg, item_off_end);
  }

  // bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncwarp();
  // bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (head_flag && continue_calc) {
  //   bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds,
  //   old_bounds); view.var_bounds_changed[var_idx] = changed;
  // }
}

template <typename i_t, typename f_t, i_t BDIM, i_t MAX_EDGE_PER_VAR, typename bounds_update_view_t>
__device__ void upd_next_constraint_sub_warp(i_t id_warp_beg,
                                             i_t id_range_end,
                                             bounds_update_view_t view)
{
  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_VAR);
  i_t var_idx;
  // auto old_bounds =
  //   f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  // auto bounds = old_bounds;
  // bool is_int = false;
  bool continue_calc = (idx < id_range_end);
  if (continue_calc) {
    var_idx = view.vars_reorg_ids[idx];
    if (view.var_bounds_changed[var_idx] == 0) { continue_calc = false; }
    // old_bounds = view.vars_bnd[var_idx];
    // is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  // Equivalent to
  // i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;
  i_t p_tid = lane_id & (MAX_EDGE_PER_VAR - 1);

  // i_t head_flag = (p_tid == 0);

  // using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  //__shared__ typename warp_reduce::TempStorage temp_storage;

  if (continue_calc) {
    // if it is a set variable then don't propagate the bound
    // consider continuous vars as set if their bounds cross or equal
    // if (old_bounds.x + view.tolerances.integrality_tolerance >= old_bounds.y) { head_flag = 0; }

    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];

    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(view, p_tid, item_off_beg, item_off_end);
  }

  // bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncwarp();
  // bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (head_flag && continue_calc) {
  //   bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds,
  //   old_bounds); view.var_bounds_changed[var_idx] = changed;
  // }
}

template <typename i_t,
          typename f_t,
          i_t BDIM,
          i_t MAX_EDGE_PER_VAR,
          typename bounds_update_view_t,
          typename upd_view_t>
__device__ void upd_next_constraint_sub_warp(
  i_t id_warp_beg, i_t id_range_end, bounds_update_view_t view, upd_view_t upd_0, upd_view_t upd_1)
{
  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_VAR);
  i_t var_idx;
  // auto old_bounds =
  //   f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  // auto bounds = old_bounds;
  // bool is_int = false;
  auto skip_mark_next = thrust::make_pair(!(idx < id_range_end), !(idx < id_range_end));
  if (idx < id_range_end) {
    var_idx        = view.vars_reorg_ids[idx];
    skip_mark_next = skip_mark(upd_0, upd_1, var_idx);
    // old_bounds = view.vars_bnd[var_idx];
    // is_int     = (view.vars_types[idx] == var_t::INTEGER);
  }
  // Equivalent to
  // i_t p_tid = threadIdx.x % MAX_EDGE_PER_VAR;
  i_t p_tid = lane_id & (MAX_EDGE_PER_VAR - 1);

  // i_t head_flag = (p_tid == 0);

  // using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_VAR>;
  //__shared__ typename warp_reduce::TempStorage temp_storage;

  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];
  if (thrust::get<1>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(
      view, upd_0, p_tid, item_off_beg, item_off_end);
  } else if (thrust::get<0>(skip_mark_next)) {
    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(
      view, upd_1, p_tid, item_off_beg, item_off_end);
  } else {
    update_next_constraints<i_t, f_t, MAX_EDGE_PER_VAR>(
      view, upd_0, upd_1, p_tid, item_off_beg, item_off_end);
  }

  // bounds.x = warp_reduce(temp_storage).Reduce(bounds.x, cub::Max());
  //__syncwarp();
  // bounds.y = warp_reduce(temp_storage).Reduce(bounds.y, cub::Min());

  // if (head_flag && continue_calc) {
  //   bool changed = write_updated_bounds(&view.vars_bnd[var_idx], is_int, view, bounds,
  //   old_bounds); view.var_bounds_changed[var_idx] = changed;
  // }
}

template <typename i_t, typename f_t, i_t BDIM, typename bounds_update_view_t>
__global__ void lb_upd_next_constraint_sub_warp_kernel(bounds_update_view_t view,
                                                       raft::device_span<i_t> warp_vars_offsets,
                                                       raft::device_span<i_t> warp_vars_id_offsets)
{
  i_t id_warp_beg, id_range_end, threads_per_variable;
  detect_range_sub_warp<i_t>(
    &id_warp_beg, &id_range_end, &threads_per_variable, warp_vars_offsets, warp_vars_id_offsets);

  if (threads_per_variable == 1) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 1>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 2) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 2>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 4) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 4>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 8) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 8>(id_warp_beg, id_range_end, view);
  } else if (threads_per_variable == 16) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 16>(id_warp_beg, id_range_end, view);
  }
}

template <typename i_t, typename f_t, i_t BDIM, typename bounds_update_view_t, typename upd_view_t>
__global__ void lb_upd_next_constraint_sub_warp_kernel(bounds_update_view_t view,
                                                       upd_view_t upd_0,
                                                       upd_view_t upd_1,
                                                       raft::device_span<i_t> warp_vars_offsets,
                                                       raft::device_span<i_t> warp_vars_id_offsets)
{
  i_t id_warp_beg, id_range_end, threads_per_variable;
  detect_range_sub_warp<i_t>(
    &id_warp_beg, &id_range_end, &threads_per_variable, warp_vars_offsets, warp_vars_id_offsets);

  if (threads_per_variable == 1) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 1>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 2) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 2>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 4) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 4>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 8) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 8>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  } else if (threads_per_variable == 16) {
    upd_next_constraint_sub_warp<i_t, f_t, BDIM, 16>(id_warp_beg, id_range_end, view, upd_0, upd_1);
  }
}

}  // namespace cuopt::linear_programming::detail
