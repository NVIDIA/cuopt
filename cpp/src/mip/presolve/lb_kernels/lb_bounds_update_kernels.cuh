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

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t, typename upd_view_t>
inline __device__ auto skip_update(upd_view_t upd, i_t var_idx, f_t int_tol)
{
  auto old_bounds = upd.vars_bnd[var_idx];
  auto skip_var   = (!upd.changed_variables[var_idx]) && (old_bounds.x + int_tol >= old_bounds.y);
  return thrust::make_tuple(old_bounds, skip_var);
}

template <int MAX_EDGE_PER_VAR, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void update_next_changed_constraints(
  csr_view_t view, upd_view_t upd, i_t tid, i_t beg, i_t end)
{
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    auto cnst_idx = view.col_elem[i];
    atomicExch(&upd.next_changed_constraints[cnst_idx], 1);
  }
}

template <typename f_t, typename f_t2>
__device__ f_t2 update_bounds_per_cnst(f_t coeff, f_t2 cnst_slack, f_t2 old_bnd, f_t2 bounds)
{
  f_t min_contrib = old_bnd.x;
  f_t max_contrib = old_bnd.y;
  if (coeff < 0.0) {
    min_contrib = old_bnd.y;
    max_contrib = old_bnd.x;
  }

  auto delta_min_act = (cnst_slack.x + (coeff * min_contrib)) / coeff;
  auto delta_max_act = (cnst_slack.y + (coeff * max_contrib)) / coeff;

  f_t lb_contrib = delta_max_act;
  f_t ub_contrib = delta_min_act;
  if (coeff < 0.0) {
    lb_contrib = delta_min_act;
    ub_contrib = delta_max_act;
  }
  bounds.x = max(bounds.x, lb_contrib);
  bounds.y = min(bounds.y, ub_contrib);
  return bounds;
}

template <typename i_t,
          int MAX_EDGE_PER_VAR,
          typename f_t2,
          typename csr_view_t,
          typename upd_view_t>
__device__ f_t2
update_bounds(csr_view_t view, upd_view_t upd, i_t tid, i_t beg, i_t end, f_t2 old_bounds)
{
  f_t2 bounds = old_bounds;

  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_VAR) {
    auto coeff    = view.coefficients[i];
    auto cnst_idx = view.col_elem[i];

    // cnst_slack[cnst_idx].x now has cnst_ub - min_a
    // cnst_slack[cnst_idx].y now has cnst_lb - max_a
    auto cnst_slack = upd.cnst_slack[cnst_idx];
    //  don't propagate over constraints that are infeasible
    // TODO : write changed_constraints = 0 for infeasible constraints while calculating activity
    if ((upd.changed_constraints[cnst_idx] == 0) || isnan(cnst_slack.x)) {
      continue;
    } else {
      bounds = update_bounds_per_cnst(coeff, cnst_slack, old_bounds, bounds);
    }
  }

  return bounds;
}

template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__global__ void bnd_heavy_update_next_changed_constraints(csr_view_t view, upd_view_t upd)
{
  auto idx = view.heavy_vertex_ids[blockIdx.x] + view.heavy_beg_id;

  auto pseudo_block_id = view.heavy_pseudo_block_ids[blockIdx.x];

  auto var_idx             = view.reorg_ids[idx];
  auto heavy_var_id_offset = var_idx - view.heavy_beg_id;

  auto bounds_updated = upd.heavy_bounds_changed_agg[heavy_var_id_offset];

  if (bounds_updated && (pseudo_block_id == 0)) { atomicAdd(upd.bounds_changed, 1); }

  auto changed = upd.heavy_bounds_changed[heavy_var_id_offset];

  if (!changed) { return; }

  i_t tid          = threadIdx.x;
  i_t item_off_beg = view.offsets[idx] + view.work_per_block * pseudo_block_id;
  i_t item_off_end = min(item_off_beg + view.work_per_block, view.offsets[idx + 1]);

  if (changed) {
    update_next_changed_constraints<BDIM>(view, upd, tid, item_off_beg, item_off_end);
  }
}

template <typename f_t, int BDIM, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void bnd_heavy(i_t id_block_beg,
                          i_t id_range_end,
                          i_t work_per_block,
                          csr_view_t view,
                          upd_view_t upd,
                          reduction_storage_t<f_t, BDIM>& storage)
{
  auto heavy_block_id = blockIdx.x - (view.sub_warp_block_count + view.med_block_count);

  auto idx = view.heavy_vertex_ids[heavy_block_id] + view.heavy_beg_id;

  auto var_idx                 = view.reorg_ids[idx];
  auto [old_bounds, skip_calc] = skip_update(upd, var_idx, view.tolerances.integrality_tolerance);

  if (skip_calc) { return; }

  bool is_int = (view.var_types[idx] == var_t::INTEGER);

  auto pseudo_block_id = view.heavy_pseudo_block_ids[heavy_block_id];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  using reduce_t = block_reduce_t<f_t, BDIM>;
  block_reduce_t<f_t, BDIM> reduce(block_storage(storage));

  if (!skip_calc) {
    auto bounds =
      update_bounds<i_t, BDIM>(view, upd, threadIdx.x, item_off_beg, item_off_end, old_bounds);
    bounds = reduce.max_min(bounds);
    if (threadIdx.x == 0) {
      write_updated_bounds_heavy(view, upd, var_idx, is_int, bounds, old_bounds);
    }
  }
}

template <typename csr_view_t, typename upd_view_t, typename i_t, typename f_t2>
inline __device__ void write_updated_bounds_heavy(
  csr_view_t view, upd_view_t upd, i_t var_idx, bool is_int, f_t2 bounds, f_t2 old_bounds)
{
  auto heavy_var_id_offset = var_idx - view.heavy_beg_id;
  auto threshold           = 1e3 * view.tolerances.absolute_tolerance;
  if (is_int) {
    bounds.x = ceil(bounds.x - view.tolerances.integrality_tolerance);
    bounds.y = floor(bounds.y + view.tolerances.integrality_tolerance);
  }
  auto lb_updated = (fabs(bounds.x - old_bounds.x) > threshold);
  auto ub_updated = (fabs(bounds.y - old_bounds.y) > threshold);

  cuda::atomic_ref<double> lb(upd.vars_bnd[var_idx].x);
  cuda::atomic_ref<double> ub(upd.vars_bnd[var_idx].y);

  if (lb_updated) { lb.fetch_max(bounds.x); }
  if (ub_updated) { ub.fetch_min(bounds.y); }

  if (lb_updated || ub_updated) {
    atomicExch(&upd.heavy_bounds_changed_agg[heavy_var_id_offset], 1);
  }
  if ((bounds.x != old_bounds.x) || (bounds.y != old_bounds.y)) {
    atomicExch(&upd.heavy_bounds_changed[heavy_var_id_offset], 1);
  }
}

template <typename csr_view_t, typename upd_view_t, typename i_t, typename f_t2>
inline __device__ bool write_updated_bounds(
  csr_view_t view, upd_view_t upd, i_t var_idx, bool is_int, f_t2 bounds, f_t2 old_bounds)
{
  auto threshold = 1e3 * view.tolerances.absolute_tolerance;
  if (is_int) {
    bounds.x = ceil(bounds.x - view.tolerances.integrality_tolerance);
    bounds.y = floor(bounds.y + view.tolerances.integrality_tolerance);
  }
  auto lb_updated = (fabs(bounds.x - old_bounds.x) > threshold);
  auto ub_updated = (fabs(bounds.y - old_bounds.y) > threshold);

  if (lb_updated) { upd.vars_bnd[var_idx].x = bounds.x; }
  if (ub_updated) { upd.vars_bnd[var_idx].y = bounds.y; }

  if (lb_updated || ub_updated) { atomicAdd(upd.bounds_changed, 1); }
  if (bounds.x != old_bounds.x || bounds.y != old_bounds.y) { return true; }
  return false;
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
                             upd_view_t upd,
                             reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  i_t var_idx = -1;
  bool is_int = false;

  f_t2 old_bounds =
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  f_t2 bounds;
  bool valid_item = (idx < id_range_end);
  bool skip_calc  = !valid_item;
  if (valid_item) {
    var_idx = view.reorg_ids[idx];
    thrust::tie(old_bounds, skip_calc) =
      skip_update(upd, var_idx, view.tolerances.integrality_tolerance);
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
  if (valid_item && (!thrust::get<0>(skip_calc))) {
    item_off_beg = view.offsets[idx];
    item_off_end = view.offsets[idx + 1];
    bounds       = update_bounds<i_t, MAX_EDGE_PER_CNST>(
      view, upd, p_tid, item_off_beg, item_off_end, old_bounds);
  }

  bounds = reduce.max_min(bounds);

  bool changed;
  auto mask = __ballot_sync(0xFFFFFFFF, valid_item);
  if (valid_item && head_flag && (!skip_calc)) {
    changed = write_updated_bounds(view, upd, var_idx, is_int, bounds, old_bounds);
  }
  if (valid_item) {
    changed = __shfl_sync(mask, changed, 0, MAX_EDGE_PER_CNST);
    if (changed) {
      update_next_changed_constraints<MAX_EDGE_PER_CNST>(
        view, upd, p_tid, item_off_beg, item_off_end);
    }
  }
}

template <typename f_t, int BDIM, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void bnd_warp(i_t id_block_beg,
                         i_t id_range_end,
                         csr_view_t view,
                         upd_view_t upd,
                         reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / 32);
  i_t idx             = id_block_beg + id_within_block;
  i_t var_idx;
  bool is_int = false;

  f_t2 old_bounds =
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  f_t2 bounds;

  bool valid_item = (idx < id_range_end);
  bool skip_calc  = !valid_item;
  if (valid_item) {
    var_idx = view.reorg_ids[idx];
    thrust::tie(old_bounds, skip_calc) =
      skip_update(upd, var_idx, view.tolerances.integrality_tolerance);
    is_int = (view.var_types[idx] == var_t::INTEGER);
    bounds = old_bounds;
    if (skip_calc) { return; }
  }

  i_t p_tid      = (threadIdx.x & 31);
  bool head_flag = (p_tid == 0);

  using reduce_t = warp_reduce_t<f_t, 32, BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  reduce_t reduce(warp_storage<32>(storage));

  i_t item_off_beg, item_off_end;
  if (valid_item && !skip_calc) {
    item_off_beg = view.offsets[idx];
    item_off_end = view.offsets[idx + 1];
    bounds       = update_bounds<i_t, 32>(view, upd, p_tid, item_off_beg, item_off_end, old_bounds);
  }

  bounds = reduce.max_min(bounds);
  __syncwarp();

  if (valid_item && head_flag && (!skip_calc)) {
    storage.vote.changed_0[id_within_block] =
      write_updated_bounds(view, upd, var_idx, is_int, bounds, old_bounds);
  }
  __syncwarp();
  bool changed_0 = storage.vote.changed_0[id_within_block];
  if (valid_item && changed_0) {
    update_next_changed_constraints<32>(view, upd, p_tid, item_off_beg, item_off_end);
  }
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
                          upd_view_t upd,
                          reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / PSEUDO_BDIM);
  i_t idx             = id_block_beg + id_within_block;
  i_t var_idx;
  bool is_int = false;

  f_t2 old_bounds =
    f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
  f_t2 bounds;
  bool valid_item = (idx < id_range_end);
  bool skip_calc  = !valid_item;
  if (valid_item) {
    var_idx = view.reorg_ids[idx];
    thrust::tie(old_bounds, skip_calc) =
      skip_update(upd, var_idx, view.tolerances.integrality_tolerance);
    is_int = (view.var_types[idx] == var_t::INTEGER);
    bounds = old_bounds;
  }

  using reduce_t = partial_block_reduce_t<f_t, BDIM, PSEUDO_BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  reduce_t reduce(partial_block_storage<PSEUDO_BDIM>(storage));

  i_t item_off_beg, item_off_end;
  if (valid_item && !skip_calc) {
    item_off_beg = view.offsets[idx];
    item_off_end = view.offsets[idx + 1];
    bounds       = update_bounds<i_t, PSEUDO_BDIM>(
      view, upd, reduce.pseudo_thread_id(), item_off_beg, item_off_end, old_bounds);
  }

  bounds = reduce.max_min(bounds);
  __syncthreads();

  if (valid_item && reduce.is_aggregated_thread() && !skip_calc) {
    storage.vote.changed_0[id_within_block] =
      write_updated_bounds(view, upd, var_idx, is_int, bounds, old_bounds);
  }

  __syncthreads();
  bool changed = storage.vote.changed_0[id_within_block];
  if (valid_item && changed) {
    update_next_changed_constraints<PSEUDO_BDIM>(
      view, upd, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  }
}

template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__device__ void call_bnd_sub_warp(csr_view_t view,
                                  upd_view_t upd,
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
    bnd_sub_warp<f_t, BDIM, 1>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 2) {
    bnd_sub_warp<f_t, BDIM, 2>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 4) {
    bnd_sub_warp<f_t, BDIM, 4>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 8) {
    bnd_sub_warp<f_t, BDIM, 8>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 16) {
    bnd_sub_warp<f_t, BDIM, 16>(id_warp_beg, id_range_end, view, upd, storage);
  }
}

template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__device__ void call_bnd_block(csr_view_t view,
                               upd_view_t upd,
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
    bnd_warp<f_t, BDIM>(id_block_beg, id_block_end, view, upd, storage);
  } else if (t_p_v == 64) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_block<f_t, BDIM, 64>(id_block_beg, id_block_end, view, upd, storage);
  } else if (t_p_v == 128) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_block<f_t, BDIM, 128>(id_block_beg, id_block_end, view, upd, storage);
  } else if (t_p_v == 256) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_block<f_t, BDIM, 256>(id_block_beg, id_block_end, view, upd, storage);
  } else {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    bnd_heavy<f_t, BDIM>(id_block_beg, id_block_end, view.work_per_block, view, upd, storage);
  }
}

// TODO : call_constraint_slack_kernel
template <typename i_t, typename f_t, int BDIM, typename csr_view_t, typename upd_view_t>
__global__ void call_bnd_update(csr_view_t view, upd_view_t upd)
{
  __shared__ reduction_storage_t<f_t, BDIM> storage;
  if (blockIdx.x < view.sub_warp_block_count) {
    call_bnd_sub_warp<i_t, f_t, BDIM>(view, upd, storage);
  } else {
    call_bnd_block<i_t, f_t, BDIM>(view, upd, storage);
  }
}

}  // namespace cuopt::linear_programming::detail
