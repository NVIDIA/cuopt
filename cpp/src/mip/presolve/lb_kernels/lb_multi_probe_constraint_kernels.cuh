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
#include "lb_constraint_slack_kernels.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename upd_view_t>
inline __device__ thrust::pair<bool, bool> skip_cnst(upd_view_t upd_0,
                                                     upd_view_t upd_1,
                                                     i_t cnst_idx)
{
  return thrust::make_pair((upd_0.changed_constraints[cnst_idx] == i_t{0}),
                           (upd_1.changed_constraints[cnst_idx] == i_t{0}));
  // return thrust::make_pair(false, false);
}

template <typename upd_view_t>
inline __device__ upd_view_t& get_valid(thrust::pair<bool, bool>& skip_flag,
                                        upd_view_t& upd_0,
                                        upd_view_t& upd_1)
{
  if (thrust::get<0>(skip_flag)) {
    return upd_1;
  } else {
    return upd_0;
  }
}

template <typename i_t,
          typename f_t,
          int MAX_EDGE_PER_CNST,
          typename csr_view_t,
          typename upd_view_t>
__device__ thrust::pair<typename type_2<f_t>::type, typename type_2<f_t>::type> calc_act(
  csr_view_t view, upd_view_t upd_0, upd_view_t upd_1, i_t tid, i_t beg, i_t end)
{
  using f_t2 = typename type_2<f_t>::type;
  auto act_0 = f_t2{0., 0.};
  auto act_1 = f_t2{0., 0.};
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coefficients[i];
    auto var   = view.col_elem[i];

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

template <typename f_t, int BDIM, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void cnst_heavy(i_t id_block_beg,
                           i_t id_range_end,
                           i_t work_per_block,
                           csr_view_t view,
                           upd_view_t upd0,
                           upd_view_t upd1,
                           reduction_storage_t<f_t, BDIM>& storage)
{
  // if (heavy_block_id > view.heavy_pseudo_block_ids.size()) {

  auto heavy_block_id = blockIdx.x - (view.sub_warp_block_count + view.med_block_count);

  auto idx = view.heavy_vertex_ids[heavy_block_id] + view.heavy_beg_id;

  auto cnst_idx  = view.reorg_ids[idx];
  auto skip_calc = skip_cnst(upd0, upd1, cnst_idx);

  if (skip_both(skip_calc)) { return; }

  auto pseudo_block_id = view.heavy_pseudo_block_ids[heavy_block_id];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  using reduce_t = block_reduce_t<f_t, BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  block_reduce_t<f_t, BDIM> reduce(block_storage(storage));

  if (both_valid(skip_calc)) {
    auto act = calc_act<i_t, f_t, BDIM>(view, upd0, upd1, threadIdx.x, item_off_beg, item_off_end);
    act      = reduce.sum(act);
    if (threadIdx.x == 0) {
      upd0.tmp_act[heavy_block_id] = thrust::get<0>(act);
      upd1.tmp_act[heavy_block_id] = thrust::get<1>(act);
    }
  } else if (!thrust::get<0>(skip_calc)) {
    auto act = calc_act<i_t, f_t, BDIM>(view, upd0, threadIdx.x, item_off_beg, item_off_end);
    act      = reduce.sum(act);
    if (threadIdx.x == 0) { upd0.tmp_act[heavy_block_id] = act; }
  } else if (!thrust::get<1>(skip_calc)) {
    auto act = calc_act<i_t, f_t, BDIM>(view, upd1, threadIdx.x, item_off_beg, item_off_end);
    act      = reduce.sum(act);
    if (threadIdx.x == 0) { upd1.tmp_act[heavy_block_id] = act; }
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__global__ void finalize_cnst_heavy(csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  using f_t2 = typename type_2<f_t>::type;

  auto idx        = blockIdx.x + view.heavy_beg_id;
  i_t cnst_idx    = view.reorg_ids[idx];
  auto cnst_lb_ub = view.cnst_bnd[idx];

  auto skip_calc = skip_cnst(upd0, upd1, cnst_idx);
  if (skip_both(skip_calc)) { return; }

  [[maybe_unused]] f_t eps = {};
  if constexpr (erase_inf_cnst) {
    eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                       cnst_lb_ub.y,
                                       view.tolerances.absolute_tolerance,
                                       view.tolerances.relative_tolerance);
  }

  using reduce_t  = warp_reduce_t<f_t, 32, BDIM>;
  using storage_t = typename reduce_t::storage_t;
  __shared__ storage_t storage;
  reduce_t reduce(storage);

  i_t item_off_beg = view.heavy_block_segments[blockIdx.x];
  i_t item_off_end = view.heavy_block_segments[blockIdx.x + 1];
  if (both_valid(skip_calc)) {
    auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act0 = upd0.tmp_act[i];
      auto act1 = upd1.tmp_act[i];
      thrust::get<0>(act).x += act0.x;
      thrust::get<0>(act).y += act0.y;

      thrust::get<1>(act).x += act1.x;
      thrust::get<1>(act).y += act1.y;
    }
    act = reduce.sum(act);
    if (threadIdx.x == 0) {
      write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
      write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
    }
  } else if (!thrust::get<0>(skip_calc)) {
    auto act = f_t2{0., 0.};
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act_load = upd0.tmp_act[i];
      act.x += act_load.x;
      act.y += act_load.y;
    }
    act = reduce.sum(act);
    if (threadIdx.x == 0) {
      write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, act, eps);
    }
  } else if (!thrust::get<1>(skip_calc)) {
    auto act = f_t2{0., 0.};
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act_load = upd1.tmp_act[i];
      act.x += act_load.x;
      act.y += act_load.y;
    }
    act = reduce.sum(act);
    if (threadIdx.x == 0) {
      write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, act, eps);
    }
  }
}

template <bool erase_inf_cnst,
          typename f_t,
          int BDIM,
          int MAX_EDGE_PER_CNST,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void cnst_sub_warp(i_t id_warp_beg,
                              i_t id_range_end,
                              csr_view_t view,
                              upd_view_t upd0,
                              upd_view_t upd1,
                              reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t lane_id  = (threadIdx.x & 31);
  i_t idx      = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  i_t cnst_idx = -1;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(valid_item, valid_item);
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd0, upd1, cnst_idx);

    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }

  i_t p_tid      = lane_id & (MAX_EDGE_PER_CNST - 1);
  bool head_flag = (p_tid == 0);

  using reduce_t = warp_reduce_t<f_t, MAX_EDGE_PER_CNST, BDIM>;
  reduce_t reduce(warp_storage<MAX_EDGE_PER_CNST>(storage));

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});

  if (valid_item && both_valid(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act =
      calc_act<i_t, f_t, MAX_EDGE_PER_CNST>(view, upd0, upd1, p_tid, item_off_beg, item_off_end);
  } else if (valid_item && (!thrust::get<0>(skip_calc))) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    thrust::get<0>(act) =
      calc_act<i_t, f_t, MAX_EDGE_PER_CNST>(view, upd0, p_tid, item_off_beg, item_off_end);
  } else if (valid_item && (!thrust::get<1>(skip_calc))) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    thrust::get<1>(act) =
      calc_act<i_t, f_t, MAX_EDGE_PER_CNST>(view, upd1, p_tid, item_off_beg, item_off_end);
  }

  act = reduce.sum(act);

  if (valid_item && head_flag && (!thrust::get<0>(skip_calc))) {
    write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
  }
  if (valid_item && head_flag && (!thrust::get<1>(skip_calc))) {
    write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
  }
}

template <bool erase_inf_cnst,
          typename f_t,
          int BDIM,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void cnst_warp(i_t id_block_beg,
                          i_t id_range_end,
                          csr_view_t view,
                          upd_view_t upd0,
                          upd_view_t upd1,
                          reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / 32);
  i_t idx             = id_block_beg + id_within_block;
  i_t cnst_idx;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(valid_item, valid_item);
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd0, upd1, cnst_idx);
    if (skip_both(skip_calc)) { return; }

    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }

  i_t p_tid      = (threadIdx.x & 31);
  bool head_flag = (p_tid == 0);

  using reduce_t = warp_reduce_t<f_t, 32, BDIM>;
  reduce_t reduce(warp_storage<32>(storage));

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});

  if (valid_item && both_valid(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act              = calc_act<i_t, f_t, 32>(view, upd0, upd1, p_tid, item_off_beg, item_off_end);
  } else if (valid_item) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    if (thrust::get<0>(skip_calc)) {
      thrust::get<1>(act) = calc_act<i_t, f_t, 32>(view, upd1, p_tid, item_off_beg, item_off_end);
    } else {
      thrust::get<0>(act) = calc_act<i_t, f_t, 32>(view, upd0, p_tid, item_off_beg, item_off_end);
    }
  }

  act = reduce.sum(act);

  if (valid_item && head_flag && (!thrust::get<0>(skip_calc))) {
    write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
  }
  if (valid_item && head_flag && (!thrust::get<1>(skip_calc))) {
    write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
  }
}

template <bool erase_inf_cnst,
          typename f_t,
          int BDIM,
          int PSEUDO_BDIM,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void cnst_block(i_t id_block_beg,
                           i_t id_range_end,
                           csr_view_t view,
                           upd_view_t upd0,
                           upd_view_t upd1,
                           reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / PSEUDO_BDIM);
  i_t idx             = id_block_beg + id_within_block;
  i_t cnst_idx;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(valid_item, valid_item);
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd0, upd1, cnst_idx);

    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }

  using reduce_t = partial_block_reduce_t<f_t, BDIM, PSEUDO_BDIM>;
  reduce_t reduce(partial_block_storage<PSEUDO_BDIM>(storage));

  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});
  if (valid_item && both_valid(skip_calc)) {
    act = calc_act<i_t, f_t, PSEUDO_BDIM>(
      view, upd0, upd1, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  } else if (valid_item && !thrust::get<0>(skip_calc)) {
    thrust::get<0>(act) = calc_act<i_t, f_t, PSEUDO_BDIM>(
      view, upd0, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  } else if (valid_item && !thrust::get<1>(skip_calc)) {
    thrust::get<1>(act) = calc_act<i_t, f_t, PSEUDO_BDIM>(
      view, upd1, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  }
  act = reduce.sum(act);
  if (valid_item && reduce.is_aggregated_thread() && !thrust::get<0>(skip_calc)) {
    write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
  }
  if (valid_item && reduce.is_aggregated_thread() && !thrust::get<1>(skip_calc)) {
    write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__device__ void call_cnst_sub_warp(csr_view_t view,
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
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 1>(
      id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 2) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 2>(
      id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 4) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 4>(
      id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 8) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 8>(
      id_warp_beg, id_range_end, view, upd0, upd1, storage);
  } else if (t_p_v == 16) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 16>(
      id_warp_beg, id_range_end, view, upd0, upd1, storage);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__device__ void call_cnst_block(csr_view_t view,
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
    cnst_warp<erase_inf_cnst, f_t, BDIM>(id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else if (t_p_v == 64) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_block<erase_inf_cnst, f_t, BDIM, 64>(
      id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else if (t_p_v == 128) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_block<erase_inf_cnst, f_t, BDIM, 128>(
      id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else if (t_p_v == 256) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_block<erase_inf_cnst, f_t, BDIM, 256>(
      id_block_beg, id_block_end, view, upd0, upd1, storage);
  } else {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_heavy<f_t, BDIM>(
      id_block_beg, id_block_end, view.work_per_block, view, upd0, upd1, storage);
  }
}

// TODO : call_constraint_slack_kernel
template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__global__ void call_cnst_slack(csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  __shared__ reduction_storage_t<f_t, BDIM> storage;
  if (blockIdx.x < view.sub_warp_block_count) {
    call_cnst_sub_warp<erase_inf_cnst, i_t, f_t, BDIM>(view, upd0, upd1, storage);
  } else {
    call_cnst_block<erase_inf_cnst, i_t, f_t, BDIM>(view, upd0, upd1, storage);
  }
}

}  // namespace cuopt::linear_programming::detail
