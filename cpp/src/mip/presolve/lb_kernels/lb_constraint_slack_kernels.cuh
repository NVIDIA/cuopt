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

template <typename i_t, typename upd_view_t>
inline __device__ bool skip_cnst(upd_view_t upd, i_t cnst_idx)
{
  return (upd.changed_constraints[cnst_idx] == i_t{0});
}

template <typename i_t,
          typename f_t,
          int MAX_EDGE_PER_CNST,
          typename csr_view_t,
          typename upd_view_t>
__device__ typename type_2<f_t>::type calc_act(
  csr_view_t view, upd_view_t upd, i_t tid, i_t beg, i_t end)
{
  using f_t2 = typename type_2<f_t>::type;
  auto act   = f_t2{0., 0.};
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coefficients[i];
    auto var   = view.col_elem[i];

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

template <bool calc_act, typename i_t, typename f_t, typename f_t2, typename upd_view_t>
inline __device__ void write_cnst_slack(
  upd_view_t view, i_t cnst_idx, f_t2 cnst_lb_ub, f_t2 act, f_t eps)
{
  auto cnst_slack = f_t2{cnst_lb_ub.y - act.x, cnst_lb_ub.x - act.y};
  if ((0 > cnst_slack.x + eps) || (eps < cnst_slack.y)) { view.changed_constraints[cnst_idx] = 0; }

  if constexpr (calc_act) {
    view.cnst_slack[cnst_idx] = act;
  } else {
    view.cnst_slack[cnst_idx] = cnst_slack;
  }
}

template <typename f_t, int BDIM, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void cnst_heavy(i_t id_block_beg,
                           i_t id_range_end,
                           i_t work_per_block,
                           csr_view_t view,
                           upd_view_t upd,
                           reduction_storage_t<f_t, BDIM>& storage)
{
  auto heavy_block_id = blockIdx.x - (view.sub_warp_block_count + view.med_block_count);

  auto idx = view.heavy_vertex_ids[heavy_block_id] + view.heavy_beg_id;

  auto cnst_idx  = view.reorg_ids[idx];
  auto skip_calc = skip_cnst(upd, cnst_idx);

  if (skip_calc) { return; }

  auto pseudo_block_id = view.heavy_pseudo_block_ids[heavy_block_id];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  using reduce_t = block_reduce_t<f_t, BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  block_reduce_t<f_t, BDIM> reduce(block_storage(storage));

  auto act = calc_act<i_t, f_t, BDIM>(view, upd, threadIdx.x, item_off_beg, item_off_end);
  act      = reduce.sum(act);
  if (threadIdx.x == 0) { upd.tmp_act[heavy_block_id] = act; }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__global__ void finalize_cnst_heavy(csr_view_t view, upd_view_t upd)
{
  using f_t2 = typename type_2<f_t>::type;

  auto idx        = blockIdx.x + view.heavy_beg_id;
  i_t cnst_idx    = view.reorg_ids[idx];
  auto cnst_lb_ub = view.cnst_bnd[idx];

  auto skip_calc = skip_cnst(upd, cnst_idx);
  if (skip_calc) { return; }

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
  auto act         = f_t2{0., 0.};
  for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
    auto act = upd.tmp_act[i];
    act.x += act.x;
    act.y += act.y;
  }
  act = reduce.sum(act);
  if (threadIdx.x == 0) { write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, act, eps); }
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
                              upd_view_t upd,
                              reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t lane_id  = (threadIdx.x & 31);
  i_t idx      = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  i_t cnst_idx = -1;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item = (idx < id_range_end);
  bool skip_calc  = !valid_item;
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd, cnst_idx);

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

  auto act = f_t2{0., 0.};

  if (valid_item && (!skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act = calc_act<i_t, f_t, MAX_EDGE_PER_CNST>(view, upd, p_tid, item_off_beg, item_off_end);
  }

  act = reduce.sum(act);

  if (valid_item && head_flag && (!skip_calc)) {
    write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, act, eps);
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
                          upd_view_t upd,
                          reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / 32);
  i_t idx             = id_block_beg + id_within_block;
  i_t cnst_idx;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item = (idx < id_range_end);
  bool skip_calc  = !valid_item;
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd, cnst_idx);
    if (skip_calc) { return; }

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
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  reduce_t reduce(warp_storage<32>(storage));

  auto act = f_t2{0., 0.};

  if (valid_item) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act              = calc_act<i_t, f_t, 32>(view, upd, p_tid, item_off_beg, item_off_end);
  }

  act = reduce.sum(act);

  if (valid_item && head_flag && (!skip_calc)) {
    write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, act, eps);
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
                           upd_view_t upd,
                           reduction_storage_t<f_t, BDIM>& storage)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / PSEUDO_BDIM);
  i_t idx             = id_block_beg + id_within_block;
  i_t cnst_idx;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item = (idx < id_range_end);
  bool skip_calc  = !valid_item;
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd, cnst_idx);

    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }

  using reduce_t = partial_block_reduce_t<f_t, BDIM, PSEUDO_BDIM>;
  // using storage_t = typename reduce_t::storage_t;
  //__shared__ storage_t storage;
  reduce_t reduce(partial_block_storage<PSEUDO_BDIM>(storage));

  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  auto act = f_t2{0., 0.};
  if (valid_item && (!skip_calc)) {
    act = calc_act<i_t, f_t, PSEUDO_BDIM>(
      view, upd, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
  }
  act = reduce.sum(act);
  if (valid_item && reduce.is_aggregated_thread() && (!skip_calc)) {
    write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, act, eps);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__device__ void call_cnst_sub_warp(csr_view_t view,
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
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 1>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 2) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 2>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 4) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 4>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 8) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 8>(id_warp_beg, id_range_end, view, upd, storage);
  } else if (t_p_v == 16) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 16>(id_warp_beg, id_range_end, view, upd, storage);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__device__ void call_cnst_block(csr_view_t view,
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
    cnst_warp<erase_inf_cnst, f_t, BDIM>(id_block_beg, id_block_end, view, upd, storage);
  } else if (t_p_v == 64) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_block<erase_inf_cnst, f_t, BDIM, 64>(id_block_beg, id_block_end, view, upd, storage);
  } else if (t_p_v == 128) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_block<erase_inf_cnst, f_t, BDIM, 128>(id_block_beg, id_block_end, view, upd, storage);
  } else if (t_p_v == 256) {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_block<erase_inf_cnst, f_t, BDIM, 256>(id_block_beg, id_block_end, view, upd, storage);
  } else {
    // if (threadIdx.x == 0) { printf("block %d t_p_v %d id_beg %d id_end %d\n", blockIdx.x, t_p_v,
    // id_block_beg, id_block_end); }
    cnst_heavy<f_t, BDIM>(id_block_beg, id_block_end, view.work_per_block, view, upd, storage);
  }
}

// TODO : call_constraint_slack_kernel
template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__global__ void call_cnst_slack(csr_view_t view, upd_view_t upd)
{
  __shared__ reduction_storage_t<f_t, BDIM> storage;
  if (blockIdx.x < view.sub_warp_block_count) {
    call_cnst_sub_warp<erase_inf_cnst, i_t, f_t, BDIM>(view, upd, storage);
  } else {
    call_cnst_block<erase_inf_cnst, i_t, f_t, BDIM>(view, upd, storage);
  }
}

}  // namespace cuopt::linear_programming::detail
