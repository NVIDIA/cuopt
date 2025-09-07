/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/utils.cuh>
#include <raft/core/device_span.hpp>
#include <utilities/lb_common_kernels.cuh>
#include "lb_bounds_update_kernels.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t, typename upd_view_t>
inline __device__ auto skip_update(upd_view_t upd_0, upd_view_t upd_1, i_t var_idx, f_t int_tol)
{
  auto old_bounds_0 = upd_0.vars_bnd[var_idx];
  auto old_bounds_1 = upd_1.vars_bnd[var_idx];
  auto skip_var_0 =
    (old_bounds_0.x + int_tol >= old_bounds_0.y);
  auto skip_var_1 =
    (old_bounds_1.x + int_tol >= old_bounds_1.y);
  //auto skip_var_0 =
  //  (!upd_0.changed_variables[var_idx]) && (old_bounds_0.x + int_tol >= old_bounds_0.y);
  //auto skip_var_1 =
  //  (!upd_1.changed_variables[var_idx]) && (old_bounds_1.x + int_tol >= old_bounds_1.y);
  return thrust::make_tuple(thrust::make_pair(old_bounds_0, old_bounds_1),
                            thrust::make_pair(skip_var_0, skip_var_1));
}

template <int MAX_EDGE_PER_VAR, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void update_next_changed_constraints(
  csr_view_t view, upd_view_t upd0, upd_view_t upd1, i_t tid, i_t beg, i_t end)
{
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    auto cnst_idx = view.col_elem[i];
    atomicExch(&upd0.next_changed_constraints[cnst_idx], 1);
    atomicExch(&upd1.next_changed_constraints[cnst_idx], 1);
  }
}

template <typename i_t,
          int MAX_EDGE_PER_VAR,
          typename f_t2,
          typename csr_view_t,
          typename upd_view_t>
__device__ thrust::pair<f_t2, f_t2> update_bounds(csr_view_t view,
                                                  upd_view_t upd_0,
                                                  upd_view_t upd_1,
                                                  i_t tid,
                                                  i_t beg,
                                                  i_t end,
                                                  f_t2 old_bounds_0,
                                                  f_t2 old_bounds_1)
{
  f_t2 bounds_0 = old_bounds_0;
  f_t2 bounds_1 = old_bounds_1;

  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    auto coeff    = view.coefficients[i];
    auto cnst_idx = view.col_elem[i];

    // cnst_slack[cnst_idx].x now has cnst_ub - min_a
    // cnst_slack[cnst_idx].y now has cnst_lb - max_a
    auto cnst_slack_0 = upd_0.cnst_slack[cnst_idx];
    //  don't propagate over constraints that are infeasible
    // TODO : write changed_constraints = 0 for infeasible constraints while calculating activity
    bool skip_cnst_0 = (upd_0.changed_constraints[cnst_idx] == 0);
    if (!skip_cnst_0) {
      bounds_0 = update_bounds_per_cnst(coeff, cnst_slack_0, old_bounds_0, bounds_0);
    }

    auto cnst_slack_1 = upd_1.cnst_slack[cnst_idx];
    bool skip_cnst_1  = (upd_1.changed_constraints[cnst_idx] == 0);
    //  don't propagate over constraints that are infeasible
    // TODO : write changed_constraints = 0 for infeasible constraints while calculating activity
    if (!skip_cnst_1) {
      bounds_1 = update_bounds_per_cnst(coeff, cnst_slack_1, old_bounds_1, bounds_1);
    }
  }

  return thrust::make_pair(bounds_0, bounds_1);
}

template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__global__ void bnd_heavy_update_next_changed_constraints(csr_view_t view,
                                                          upd_view_t upd0,
                                                          upd_view_t upd1)
{
  auto idx = view.heavy_vertex_ids[blockIdx.x] + view.heavy_beg_id;

  auto pseudo_block_id = view.heavy_pseudo_block_ids[blockIdx.x];

  auto var_idx             = view.reorg_ids[idx];
  auto heavy_var_id_offset = var_idx - view.heavy_beg_id;

  auto bounds_updated_0 = upd0.heavy_bounds_changed_agg[heavy_var_id_offset];
  auto bounds_updated_1 = upd1.heavy_bounds_changed_agg[heavy_var_id_offset];

  if (bounds_updated_0 && (pseudo_block_id == 0)) { atomicAdd(upd0.bounds_changed, 1); }
  if (bounds_updated_1 && (pseudo_block_id == 0)) { atomicAdd(upd1.bounds_changed, 1); }

  auto changed_0 = upd0.heavy_bounds_changed[heavy_var_id_offset];
  auto changed_1 = upd1.heavy_bounds_changed[heavy_var_id_offset];

  if (!(changed_0 && changed_1)) { return; }

  i_t tid          = threadIdx.x;
  i_t item_off_beg = view.offsets[idx] + view.work_per_block * pseudo_block_id;
  i_t item_off_end = min(item_off_beg + view.work_per_block, view.offsets[idx + 1]);

  if (changed_0 && changed_1) {
    update_next_changed_constraints<BDIM>(view, upd0, upd1, tid, item_off_beg, item_off_end);
  } else if (changed_0) {
    update_next_changed_constraints<BDIM>(view, upd0, tid, item_off_beg, item_off_end);
  } else if (changed_1) {
    update_next_changed_constraints<BDIM>(view, upd1, tid, item_off_beg, item_off_end);
  }
}

template <typename f_t, int BDIM, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void bnd_heavy(i_t id_block_beg,
                          i_t id_range_end,
                          i_t work_per_block,
                          csr_view_t view,
                          upd_view_t upd0,
                          upd_view_t upd1,
                          reduction_storage_t<f_t, BDIM>& storage)
{
  auto heavy_block_id = blockIdx.x - (view.sub_warp_block_count + view.med_block_count);

  auto idx = view.heavy_vertex_ids[heavy_block_id] + view.heavy_beg_id;

  auto var_idx = view.reorg_ids[idx];
  auto [old_bounds, skip_calc] =
    skip_update(upd0, upd1, var_idx, view.tolerances.integrality_tolerance);

  if (skip_both(skip_calc)) { return; }

  bool is_int = (view.var_types[idx] == var_t::INTEGER);

  auto pseudo_block_id = view.heavy_pseudo_block_ids[heavy_block_id];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  using reduce_t = block_reduce_t<f_t, BDIM>;
  block_reduce_t<f_t, BDIM> reduce(block_storage(storage));

  if (both_valid(skip_calc)) {
    auto bounds = update_bounds<i_t, BDIM>(view,
                                           upd0,
                                           upd1,
                                           threadIdx.x,
                                           item_off_beg,
                                           item_off_end,
                                           thrust::get<0>(old_bounds),
                                           thrust::get<1>(old_bounds));
    bounds      = reduce.max_min(bounds);
    if (threadIdx.x == 0) {
      // upd0.tmp_bnd[heavy_block_id] = thrust::get<0>(bounds);
      // upd1.tmp_bnd[heavy_block_id] = thrust::get<1>(bounds);
      write_updated_bounds_heavy(
        view, upd0, var_idx, is_int, thrust::get<0>(bounds), thrust::get<0>(old_bounds));
      write_updated_bounds_heavy(
        view, upd1, var_idx, is_int, thrust::get<1>(bounds), thrust::get<1>(old_bounds));
    }
  } else if (!thrust::get<0>(skip_calc)) {
    auto bounds = update_bounds<i_t, BDIM>(
      view, upd0, threadIdx.x, item_off_beg, item_off_end, thrust::get<0>(old_bounds));
    bounds = reduce.max_min(bounds);
    if (threadIdx.x == 0) {
      // upd0.tmp_bnd[heavy_block_id] = thrust::get<0>(bounds);
      write_updated_bounds_heavy(view, upd0, var_idx, is_int, bounds, thrust::get<0>(old_bounds));
    }
  } else if (!thrust::get<1>(skip_calc)) {
    auto bounds = update_bounds<i_t, BDIM>(
      view, upd1, threadIdx.x, item_off_beg, item_off_end, thrust::get<1>(old_bounds));
    bounds = reduce.max_min(bounds);
    if (threadIdx.x == 0) {
      // upd1.tmp_bnd[heavy_block_id] = thrust::get<1>(bounds);
      write_updated_bounds_heavy(view, upd1, var_idx, is_int, bounds, thrust::get<1>(old_bounds));
    }
  }
}

template <typename f_t,
          int BDIM,
          int MAX_EDGE_PER_CNST,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void bnd_sub_warp(i_t id_warp_beg,
                             i_t id_range_end,
                             csr_view_t view,
                             upd_view_t upd0,
                             upd_view_t upd1,
                             reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  i_t var_idx = -1;
  bool is_int = false;

  thrust::pair<f_t2, f_t2> old_bounds = thrust::make_pair(
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()});

  thrust::pair<f_t2, f_t2> bounds;
  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(!valid_item, !valid_item);
  if (valid_item) {
    var_idx = view.reorg_ids[idx];
    thrust::tie(old_bounds, skip_calc) =
      skip_update(upd0, upd1, var_idx, view.tolerances.integrality_tolerance);
    is_int = (view.var_types[idx] == var_t::INTEGER);
    bounds = old_bounds;
  }

  i_t p_tid      = lane_id & (MAX_EDGE_PER_CNST - 1);
  bool head_flag = (p_tid == 0);

  using reduce_t = warp_reduce_t<f_t, MAX_EDGE_PER_CNST, BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  reduce_t reduce(warp_storage<MAX_EDGE_PER_CNST>(storage));

  i_t item_off_beg, item_off_end;
  if (valid_item && both_valid(skip_calc)) {
    item_off_beg = view.offsets[idx];
    item_off_end = view.offsets[idx + 1];
    bounds       = update_bounds<i_t, MAX_EDGE_PER_CNST>(view,
                                                   upd0,
                                                   upd1,
                                                   p_tid,
                                                   item_off_beg,
                                                   item_off_end,
                                                   thrust::get<0>(old_bounds),
                                                   thrust::get<1>(old_bounds));
  } else if (valid_item && (!thrust::get<0>(skip_calc))) {
    item_off_beg           = view.offsets[idx];
    item_off_end           = view.offsets[idx + 1];
    thrust::get<0>(bounds) = update_bounds<i_t, MAX_EDGE_PER_CNST>(
      view, upd0, p_tid, item_off_beg, item_off_end, thrust::get<0>(old_bounds));
  } else if (valid_item && (!thrust::get<1>(skip_calc))) {
    item_off_beg           = view.offsets[idx];
    item_off_end           = view.offsets[idx + 1];
    thrust::get<1>(bounds) = update_bounds<i_t, MAX_EDGE_PER_CNST>(
      view, upd1, p_tid, item_off_beg, item_off_end, thrust::get<1>(old_bounds));
  }

  bounds = reduce.max_min(bounds);

  bool changed_0, changed_1;
  //auto mask = __ballot_sync(0xFFFFFFFF, valid_item);
  if (valid_item && head_flag && (!thrust::get<0>(skip_calc))) {
    changed_0 = write_updated_bounds(
      view, upd0, var_idx, is_int, thrust::get<0>(bounds), thrust::get<0>(old_bounds));
  }
  if (valid_item && head_flag && (!thrust::get<1>(skip_calc))) {
    changed_1 = write_updated_bounds(
      view, upd1, var_idx, is_int, thrust::get<1>(bounds), thrust::get<1>(old_bounds));
  }
  //if (valid_item) {
  //  changed_0 = __shfl_sync(mask, changed_0, 0, MAX_EDGE_PER_CNST);
  //  changed_1 = __shfl_sync(mask, changed_1, 0, MAX_EDGE_PER_CNST);
  //  if (changed_0 && changed_1) {
  //    update_next_changed_constraints<MAX_EDGE_PER_CNST>(
  //      view, upd0, upd1, p_tid, item_off_beg, item_off_end);
  //  } else if (changed_0) {
  //    update_next_changed_constraints<MAX_EDGE_PER_CNST>(
  //      view, upd0, p_tid, item_off_beg, item_off_end);
  //  } else if (changed_1) {
  //    update_next_changed_constraints<MAX_EDGE_PER_CNST>(
  //      view, upd1, p_tid, item_off_beg, item_off_end);
  //  }
  //}
}

template <typename f_t, int BDIM, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void bnd_warp(i_t id_block_beg,
                         i_t id_range_end,
                         csr_view_t view,
                         upd_view_t upd0,
                         upd_view_t upd1,
                         reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / 32);
  i_t idx             = id_block_beg + id_within_block;
  i_t var_idx;
  bool is_int = false;

  thrust::pair<f_t2, f_t2> old_bounds = thrust::make_pair(
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()});

  thrust::pair<f_t2, f_t2> bounds;
  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(!valid_item, !valid_item);
  if (valid_item) {
    var_idx = view.reorg_ids[idx];
    thrust::tie(old_bounds, skip_calc) =
      skip_update(upd0, upd1, var_idx, view.tolerances.integrality_tolerance);
    is_int = (view.var_types[idx] == var_t::INTEGER);
    bounds = old_bounds;
    if (skip_both(skip_calc)) { return; }
  }

  i_t p_tid      = (threadIdx.x & 31);
  bool head_flag = (p_tid == 0);

  using reduce_t = warp_reduce_t<f_t, 32, BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  reduce_t reduce(warp_storage<32>(storage));

  i_t item_off_beg, item_off_end;
  if (valid_item && both_valid(skip_calc)) {
    item_off_beg = view.offsets[idx];
    item_off_end = view.offsets[idx + 1];
    bounds       = update_bounds<i_t, 32>(view,
                                    upd0,
                                    upd1,
                                    p_tid,
                                    item_off_beg,
                                    item_off_end,
                                    thrust::get<0>(old_bounds),
                                    thrust::get<1>(old_bounds));
  } else if (valid_item) {
    item_off_beg = view.offsets[idx];
    item_off_end = view.offsets[idx + 1];
    if (thrust::get<0>(skip_calc)) {
      thrust::get<1>(bounds) = update_bounds<i_t, 32>(
        view, upd1, p_tid, item_off_beg, item_off_end, thrust::get<1>(old_bounds));
    } else {
      thrust::get<0>(bounds) = update_bounds<i_t, 32>(
        view, upd0, p_tid, item_off_beg, item_off_end, thrust::get<0>(old_bounds));
    }
  }

  bounds = reduce.max_min(bounds);
  __syncwarp();

  if (valid_item && head_flag && (!thrust::get<0>(skip_calc))) {
    //storage.vote.changed_0[id_within_block] = write_updated_bounds(
    write_updated_bounds(
      view, upd0, var_idx, is_int, thrust::get<0>(bounds), thrust::get<0>(old_bounds));
  }
  if (valid_item && head_flag && (!thrust::get<1>(skip_calc))) {
    write_updated_bounds(
      view, upd1, var_idx, is_int, thrust::get<1>(bounds), thrust::get<1>(old_bounds));
  }
  //__syncwarp();
  //bool changed_0 = storage.vote.changed_0[id_within_block];
  //bool changed_1 = storage.vote.changed_1[id_within_block];
  //if (valid_item && changed_0 && changed_1) {
  //  update_next_changed_constraints<32>(view, upd0, upd1, p_tid, item_off_beg, item_off_end);
  //} else if (valid_item && changed_0) {
  //  update_next_changed_constraints<32>(view, upd0, p_tid, item_off_beg, item_off_end);
  //} else if (valid_item && changed_1) {
  //  update_next_changed_constraints<32>(view, upd1, p_tid, item_off_beg, item_off_end);
  //}
}

template <typename f_t,
          int BDIM,
          int PSEUDO_BDIM,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void bnd_block(i_t id_block_beg,
                          i_t id_range_end,
                          csr_view_t view,
                          upd_view_t upd0,
                          upd_view_t upd1,
                          reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / PSEUDO_BDIM);
  i_t idx             = id_block_beg + id_within_block;
  i_t var_idx;
  bool is_int = false;

  thrust::pair<f_t2, f_t2> old_bounds = thrust::make_pair(
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()});
  thrust::pair<f_t2, f_t2> bounds;
  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(!valid_item, !valid_item);
  if (valid_item) {
    var_idx = view.reorg_ids[idx];
    thrust::tie(old_bounds, skip_calc) =
      skip_update(upd0, upd1, var_idx, view.tolerances.integrality_tolerance);
    is_int = (view.var_types[idx] == var_t::INTEGER);
    bounds = old_bounds;
  }

  using reduce_t = partial_block_reduce_t<f_t, BDIM, PSEUDO_BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  reduce_t reduce(partial_block_storage<PSEUDO_BDIM>(storage));

  i_t item_off_beg, item_off_end;
  if (valid_item && both_valid(skip_calc)) {
    item_off_beg = view.offsets[idx];
    item_off_end = view.offsets[idx + 1];
    bounds       = update_bounds<i_t, PSEUDO_BDIM>(view,
                                             upd0,
                                             upd1,
                                             reduce.pseudo_thread_id(),
                                             item_off_beg,
                                             item_off_end,
                                             thrust::get<0>(old_bounds),
                                             thrust::get<1>(old_bounds));
  } else if (valid_item && !thrust::get<0>(skip_calc)) {
    item_off_beg           = view.offsets[idx];
    item_off_end           = view.offsets[idx + 1];
    thrust::get<0>(bounds) = update_bounds<i_t, PSEUDO_BDIM>(view,
                                                             upd0,
                                                             reduce.pseudo_thread_id(),
                                                             item_off_beg,
                                                             item_off_end,
                                                             thrust::get<0>(old_bounds));
  } else if (valid_item && !thrust::get<1>(skip_calc)) {
    item_off_beg           = view.offsets[idx];
    item_off_end           = view.offsets[idx + 1];
    thrust::get<1>(bounds) = update_bounds<i_t, PSEUDO_BDIM>(view,
                                                             upd1,
                                                             reduce.pseudo_thread_id(),
                                                             item_off_beg,
                                                             item_off_end,
                                                             thrust::get<1>(old_bounds));
  }

  bounds = reduce.max_min(bounds);
  __syncthreads();

  if (valid_item && reduce.is_aggregated_thread() && !thrust::get<0>(skip_calc)) {
    //storage.vote.changed_0[id_within_block] = write_updated_bounds(
    write_updated_bounds(
      view, upd0, var_idx, is_int, thrust::get<0>(bounds), thrust::get<0>(old_bounds));
  }
  if (valid_item && reduce.is_aggregated_thread() && !thrust::get<1>(skip_calc)) {
    //storage.vote.changed_1[id_within_block] = write_updated_bounds(
    write_updated_bounds(
      view, upd1, var_idx, is_int, thrust::get<1>(bounds), thrust::get<1>(old_bounds));
  }

  //__syncthreads();
  //bool changed_0 = storage.vote.changed_0[id_within_block];
  //bool changed_1 = storage.vote.changed_1[id_within_block];
  //if (valid_item && changed_0 && changed_1) {
  //  update_next_changed_constraints<PSEUDO_BDIM>(
  //    view, upd0, upd1, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  //} else if (valid_item && changed_0) {
  //  update_next_changed_constraints<PSEUDO_BDIM>(
  //    view, upd0, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  //} else if (valid_item && changed_1) {
  //  update_next_changed_constraints<PSEUDO_BDIM>(
  //    view, upd1, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  //}
}

template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__device__ void call_bnd_sub_warp(csr_view_t view,
                                  upd_view_t upd0,
                                  upd_view_t upd1,
                                  reduction_storage_t<f_t, BDIM>& storage)
{
  i_t id_warp_beg, id_range_end, t_p_v;
  get_sub_warp_bin<i_t>(&id_warp_beg,
                        &id_range_end,
                        &t_p_v,
                        view.warp_offsets,
                        view.warp_id_offsets,
                        view.sub_warp_count);

  if (t_p_v == 1) {
    bnd_sub_warp<f_t, BDIM, 1>(id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 2) {
    bnd_sub_warp<f_t, BDIM, 2>(id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 4) {
    bnd_sub_warp<f_t, BDIM, 4>(id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 8) {
    bnd_sub_warp<f_t, BDIM, 8>(id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 16) {
    bnd_sub_warp<f_t, BDIM, 16>(id_warp_beg, id_range_end, view, upd0, upd1, storage);
  }
}

template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__device__ void call_bnd_block(csr_view_t view,
                               upd_view_t upd0,
                               upd_view_t upd1,
                               reduction_storage_t<f_t, BDIM>& storage)
{
  i_t id_block_beg, id_block_end, t_p_v;
  get_block_bin<i_t>(&id_block_beg,
                     &id_block_end,
                     &t_p_v,
                     view.block_offsets,
                     view.block_id_offsets,
                     view.sub_warp_block_count,
                     view.med_block_count);

  if (t_p_v == 32) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_warp<f_t, BDIM>(id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else if (t_p_v == 64) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_block<f_t, BDIM, 64>(id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else if (t_p_v == 128) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_block<f_t, BDIM, 128>(id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else if (t_p_v == 256) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_block<f_t, BDIM, 256>(id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_heavy<f_t, BDIM>(
      id_block_beg, id_block_end, view.work_per_block, view, upd0, upd1, storage);
  }
}

// TODO : call_constraint_slack_kernel
template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__global__ void call_bnd_update(csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  __shared__ reduction_storage_t<f_t, BDIM> storage;
  if (blockIdx.x < view.sub_warp_block_count) {
    call_bnd_sub_warp<i_t, f_t, BDIM>(view, upd0, upd1, storage);
  } else {
    call_bnd_block<i_t, f_t, BDIM>(view, upd0, upd1, storage);
  }
}

}  // namespace cuopt::linear_programming::detail
