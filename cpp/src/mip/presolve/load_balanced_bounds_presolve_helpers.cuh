/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "load_balanced_bounds_presolve_kernels.cuh"
#include "load_balanced_partition_helpers.cuh"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t>
i_t get_id_offset(const std::vector<i_t>& bin_offsets, i_t degree_cutoff)
{
  return bin_offsets[ceil_log_2(degree_cutoff)];
}

template <typename i_t>
std::pair<i_t, i_t> get_id_range(const std::vector<i_t>& bin_offsets,
                                 i_t degree_beg,
                                 i_t degree_end)
{
  return std::make_pair(bin_offsets[ceil_log_2(degree_beg)],
                        bin_offsets[ceil_log_2(degree_end) + 1]);
}

template <typename i_t>
struct calc_blocks_per_item_t {
  calc_blocks_per_item_t(raft::device_span<const i_t> offsets_, i_t work_per_block_)
    : offsets(offsets_), work_per_block(work_per_block_)
  {
  }
  raft::device_span<const i_t> offsets;
  i_t work_per_block;
  __device__ __forceinline__ i_t operator()(i_t item_id) const
  {
    i_t work_per_vertex = (offsets[item_id + 1] - offsets[item_id]);
    return raft::ceildiv<i_t>(work_per_vertex, work_per_block);
  }
};

template <typename i_t>
struct heavy_vertex_meta_t {
  heavy_vertex_meta_t(raft::device_span<const i_t> offsets_,
                      raft::device_span<i_t> vertex_id_,
                      raft::device_span<i_t> pseudo_block_id_)
    : offsets(offsets_), vertex_id(vertex_id_), pseudo_block_id(pseudo_block_id_)
  {
  }

  raft::device_span<const i_t> offsets;
  raft::device_span<i_t> vertex_id;
  raft::device_span<i_t> pseudo_block_id;

  __device__ __forceinline__ void operator()(i_t id) const
  {
    vertex_id[offsets[id]] = id;
    if (id != 0) {
      pseudo_block_id[offsets[id]] = offsets[id - 1] - offsets[id] + 1;
    } else {
      pseudo_block_id[offsets[0]] = 0;
    }
  }
};

template <typename i_t>
i_t create_heavy_item_block_segments(rmm::cuda_stream_view stream,
                                     rmm::device_uvector<i_t>& vertex_id,
                                     rmm::device_uvector<i_t>& pseudo_block_id,
                                     rmm::device_uvector<i_t>& item_block_segments,
                                     const i_t heavy_degree_cutoff,
                                     const std::vector<i_t>& bin_offsets,
                                     rmm::device_uvector<i_t> const& offsets)
{
  // TODO : assert that bin_offsets.back() == offsets.size() - 1
  auto heavy_id_beg   = bin_offsets[ceil_log_2(heavy_degree_cutoff)];
  auto n_items        = offsets.size() - 1;
  auto heavy_id_count = n_items - heavy_id_beg;
  item_block_segments.resize(1 + heavy_id_count, stream);

  // Amount of blocks to be launched for each item (constraint or variable).
  auto work_per_block              = heavy_degree_cutoff / 2;
  auto calc_blocks_per_vertex_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(heavy_id_beg),
    calc_blocks_per_item_t<i_t>{make_span(offsets), work_per_block});

  // Inclusive scan so that each block can determine which item it belongs to
  item_block_segments.set_element_to_zero_async(0, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         calc_blocks_per_vertex_iter,
                         calc_blocks_per_vertex_iter + heavy_id_count,
                         item_block_segments.begin() + 1);
  auto num_blocks = item_block_segments.back_element(stream);
  if (num_blocks > 0) {
    vertex_id.resize(num_blocks, stream);
    pseudo_block_id.resize(num_blocks, stream);
    thrust::fill(rmm::exec_policy(stream), vertex_id.begin(), vertex_id.end(), i_t{-1});
    thrust::fill(rmm::exec_policy(stream), pseudo_block_id.begin(), pseudo_block_id.end(), i_t{1});
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(item_block_segments.size()),
      heavy_vertex_meta_t<i_t>{
        make_span(item_block_segments), make_span(vertex_id), make_span(pseudo_block_id)});
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           vertex_id.begin(),
                           vertex_id.end(),
                           vertex_id.begin(),
                           thrust::maximum<i_t>{});
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           pseudo_block_id.begin(),
                           pseudo_block_id.end(),
                           pseudo_block_id.begin(),
                           thrust::plus<i_t>{});
  }
  // Total number of blocks that have to be launched
  return num_blocks;
}

/// CALCULATE ACTIVITY

template <typename i_t, typename f_t, typename f_t2, i_t block_dim, typename activity_view_t>
void calc_activity_heavy_cnst(managed_stream_pool& streams,
                              activity_view_t view,
                              raft::device_span<f_t2> tmp_cnst_act,
                              const rmm::device_uvector<i_t>& heavy_cnst_vertex_ids,
                              const rmm::device_uvector<i_t>& heavy_cnst_pseudo_block_ids,
                              const rmm::device_uvector<i_t>& heavy_cnst_block_segments,
                              const std::vector<i_t>& cnst_bin_offsets,
                              i_t heavy_degree_cutoff,
                              i_t num_blocks_heavy_cnst,
                              bool erase_inf_cnst,
                              bool dry_run = false)
{
  if (num_blocks_heavy_cnst != 0) {
    auto heavy_cnst_stream = streams.get_stream();
    // TODO : Check heavy_cnst_block_segments size for profiling
    if (!dry_run) {
      auto heavy_cnst_beg_id = get_id_offset(cnst_bin_offsets, heavy_degree_cutoff);
      lb_calc_act_heavy_kernel<i_t, f_t, f_t2, block_dim>
        <<<num_blocks_heavy_cnst, block_dim, 0, heavy_cnst_stream>>>(
          heavy_cnst_beg_id,
          make_span(heavy_cnst_vertex_ids),
          make_span(heavy_cnst_pseudo_block_ids),
          heavy_degree_cutoff,
          view,
          tmp_cnst_act);
      auto num_heavy_cnst = cnst_bin_offsets.back() - heavy_cnst_beg_id;
      if (erase_inf_cnst) {
        finalize_calc_act_kernel<true, i_t, f_t, f_t2>
          <<<num_heavy_cnst, 32, 0, heavy_cnst_stream>>>(
            heavy_cnst_beg_id, make_span(heavy_cnst_block_segments), tmp_cnst_act, view);
      } else {
        finalize_calc_act_kernel<false, i_t, f_t, f_t2>
          <<<num_heavy_cnst, 32, 0, heavy_cnst_stream>>>(
            heavy_cnst_beg_id, make_span(heavy_cnst_block_segments), tmp_cnst_act, view);
      }
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, i_t block_dim, typename activity_view_t>
void calc_activity_per_block(managed_stream_pool& streams,
                             activity_view_t view,
                             const std::vector<i_t>& cnst_bin_offsets,
                             i_t degree_beg,
                             i_t degree_end,
                             bool erase_inf_cnst,
                             bool dry_run)
{
  static_assert(block_dim <= 1024, "Cannot launch kernel with more than 1024 threads");

  auto [cnst_id_beg, cnst_id_end] = get_id_range(cnst_bin_offsets, degree_beg, degree_end);

  auto block_count = cnst_id_end - cnst_id_beg;
  if (block_count > 0) {
    auto block_stream = streams.get_stream();
    if (!dry_run) {
      if (erase_inf_cnst) {
        lb_calc_act_block_kernel<true, i_t, f_t, f_t2, block_dim>
          <<<block_count, block_dim, 0, block_stream>>>(cnst_id_beg, view);
      } else {
        lb_calc_act_block_kernel<false, i_t, f_t, f_t2, block_dim>
          <<<block_count, block_dim, 0, block_stream>>>(cnst_id_beg, view);
      }
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, typename activity_view_t>
void calc_activity_per_block(managed_stream_pool& streams,
                             activity_view_t view,
                             const std::vector<i_t>& cnst_bin_offsets,
                             i_t heavy_degree_cutoff,
                             bool erase_inf_cnst,
                             bool dry_run = false)
{
  if (view.nnz < 10000) {
    calc_activity_per_block<i_t, f_t, f_t2, 32>(
      streams, view, cnst_bin_offsets, 32, 32, erase_inf_cnst, dry_run);
    calc_activity_per_block<i_t, f_t, f_t2, 64>(
      streams, view, cnst_bin_offsets, 64, 64, erase_inf_cnst, dry_run);
    calc_activity_per_block<i_t, f_t, f_t2, 128>(
      streams, view, cnst_bin_offsets, 128, 128, erase_inf_cnst, dry_run);
    calc_activity_per_block<i_t, f_t, f_t2, 256>(
      streams, view, cnst_bin_offsets, 256, 256, erase_inf_cnst, dry_run);
  } else {
    //[1024, heavy_degree_cutoff/2] -> 1024 block size
    calc_activity_per_block<i_t, f_t, f_t2, 1024>(
      streams, view, cnst_bin_offsets, 1024, heavy_degree_cutoff / 2, erase_inf_cnst, dry_run);
    //[512, 512] -> 128 block size
    calc_activity_per_block<i_t, f_t, f_t2, 128>(
      streams, view, cnst_bin_offsets, 128, 512, erase_inf_cnst, dry_run);
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_constraint,
          typename activity_view_t>
void calc_activity_sub_warp(managed_stream_pool& streams,
                            activity_view_t view,
                            i_t degree_beg,
                            i_t degree_end,
                            const std::vector<i_t>& cnst_bin_offsets,
                            bool erase_inf_cnst,
                            bool dry_run)
{
  constexpr i_t block_dim         = 32;
  auto cnst_per_block             = block_dim / threads_per_constraint;
  auto [cnst_id_beg, cnst_id_end] = get_id_range(cnst_bin_offsets, degree_beg, degree_end);

  auto block_count = raft::ceildiv<i_t>(cnst_id_end - cnst_id_beg, cnst_per_block);
  if (block_count != 0) {
    auto sub_warp_thread = streams.get_stream();
    if (!dry_run) {
      if (erase_inf_cnst) {
        lb_calc_act_sub_warp_kernel<true, i_t, f_t, f_t2, block_dim, threads_per_constraint>
          <<<block_count, block_dim, 0, sub_warp_thread>>>(cnst_id_beg, cnst_id_end, view);
      } else {
        lb_calc_act_sub_warp_kernel<false, i_t, f_t, f_t2, block_dim, threads_per_constraint>
          <<<block_count, block_dim, 0, sub_warp_thread>>>(cnst_id_beg, cnst_id_end, view);
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_constraint,
          typename activity_view_t>
void calc_activity_sub_warp(managed_stream_pool& streams,
                            activity_view_t view,
                            i_t degree,
                            const std::vector<i_t>& cnst_bin_offsets,
                            bool erase_inf_cnst,
                            bool dry_run)
{
  calc_activity_sub_warp<i_t, f_t, f_t2, threads_per_constraint>(
    streams, view, degree, degree, cnst_bin_offsets, erase_inf_cnst, dry_run);
}

template <typename i_t, typename f_t, typename f_t2, typename activity_view_t>
void calc_activity_sub_warp(managed_stream_pool& streams,
                            activity_view_t view,
                            i_t cnst_sub_warp_count,
                            rmm::device_uvector<i_t>& warp_cnst_offsets,
                            rmm::device_uvector<i_t>& warp_cnst_id_offsets,
                            bool erase_inf_cnst,
                            bool dry_run)
{
  constexpr i_t block_dim = 256;

  auto block_count = raft::ceildiv<i_t>(cnst_sub_warp_count * 32, block_dim);
  if (block_count != 0) {
    auto sub_warp_stream = streams.get_stream();
    if (!dry_run) {
      if (erase_inf_cnst) {
        lb_calc_act_sub_warp_kernel<true, i_t, f_t, f_t2, block_dim>
          <<<block_count, block_dim, 0, sub_warp_stream>>>(
            view, make_span(warp_cnst_offsets), make_span(warp_cnst_id_offsets));
      } else {
        lb_calc_act_sub_warp_kernel<false, i_t, f_t, f_t2, block_dim>
          <<<block_count, block_dim, 0, sub_warp_stream>>>(
            view, make_span(warp_cnst_offsets), make_span(warp_cnst_id_offsets));
      }
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, typename activity_view_t>
void calc_activity_sub_warp(managed_stream_pool& streams,
                            activity_view_t view,
                            bool is_cnst_sub_warp_single_bin,
                            i_t cnst_sub_warp_count,
                            rmm::device_uvector<i_t>& warp_cnst_offsets,
                            rmm::device_uvector<i_t>& warp_cnst_id_offsets,
                            const std::vector<i_t>& cnst_bin_offsets,
                            bool erase_inf_cnst,
                            bool dry_run = false)
{
  if (view.nnz < 10000) {
    calc_activity_sub_warp<i_t, f_t, f_t2, 16>(
      streams, view, 16, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 8>(
      streams, view, 8, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 4>(
      streams, view, 4, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 2>(
      streams, view, 2, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 1>(
      streams, view, 1, cnst_bin_offsets, erase_inf_cnst, dry_run);
  } else {
    if (is_cnst_sub_warp_single_bin) {
      calc_activity_sub_warp<i_t, f_t, f_t2, 16>(
        streams, view, 64, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 8>(
        streams, view, 32, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 4>(
        streams, view, 16, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 2>(
        streams, view, 8, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 1>(
        streams, view, 1, 4, cnst_bin_offsets, erase_inf_cnst, dry_run);
    } else {
      calc_activity_sub_warp<i_t, f_t, f_t2>(streams,
                                             view,
                                             cnst_sub_warp_count,
                                             warp_cnst_offsets,
                                             warp_cnst_id_offsets,
                                             erase_inf_cnst,
                                             dry_run);
    }
  }
}

/// BOUNDS UPDATE

template <typename i_t, typename f_t, typename f_t2, i_t block_dim, typename bounds_update_view_t>
void upd_bounds_heavy_vars(managed_stream_pool& streams,
                           bounds_update_view_t view,
                           raft::device_span<f_t2> tmp_vars_bnd,
                           const rmm::device_uvector<i_t>& heavy_vars_vertex_ids,
                           const rmm::device_uvector<i_t>& heavy_vars_pseudo_block_ids,
                           const rmm::device_uvector<i_t>& heavy_vars_block_segments,
                           const std::vector<i_t>& vars_bin_offsets,
                           i_t heavy_degree_cutoff,
                           i_t num_blocks_heavy_vars,
                           bool dry_run = false)
{
  if (num_blocks_heavy_vars != 0) {
    auto heavy_vars_stream = streams.get_stream();
    // TODO : Check heavy_vars_block_segments size for profiling
    if (!dry_run) {
      auto heavy_vars_beg_id = get_id_offset(vars_bin_offsets, heavy_degree_cutoff);
      lb_upd_bnd_heavy_kernel<i_t, f_t, f_t2, block_dim>
        <<<num_blocks_heavy_vars, block_dim, 0, heavy_vars_stream>>>(
          heavy_vars_beg_id,
          make_span(heavy_vars_vertex_ids),
          make_span(heavy_vars_pseudo_block_ids),
          heavy_degree_cutoff,
          view,
          tmp_vars_bnd);
      auto num_heavy_vars = vars_bin_offsets.back() - heavy_vars_beg_id;
      finalize_upd_bnd_kernel<i_t, f_t, f_t2><<<num_heavy_vars, 32, 0, heavy_vars_stream>>>(
        heavy_vars_beg_id, make_span(heavy_vars_block_segments), tmp_vars_bnd, view);
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, i_t block_dim, typename bounds_update_view_t>
void upd_bounds_heavy_vars(managed_stream_pool& streams,
                           bounds_update_view_t view,
                           raft::device_span<f_t2> tmp_vars_bnd,
                           const rmm::device_uvector<i_t>& heavy_vars_block_segments,
                           const std::vector<i_t>& vars_bin_offsets,
                           i_t heavy_degree_cutoff,
                           i_t num_blocks_heavy_vars,
                           bool dry_run = false)
{
  if (num_blocks_heavy_vars != 0) {
    auto heavy_vars_stream = streams.get_stream();
    // TODO : Check heavy_vars_block_segments size for profiling
    if (!dry_run) {
      auto heavy_vars_beg_id = get_id_offset(vars_bin_offsets, heavy_degree_cutoff);
      lb_upd_bnd_heavy_kernel<i_t, f_t, f_t2, block_dim>
        <<<num_blocks_heavy_vars, block_dim, 0, heavy_vars_stream>>>(
          heavy_vars_beg_id,
          make_span(heavy_vars_block_segments, 1, heavy_vars_block_segments.size()),
          heavy_degree_cutoff,
          view,
          tmp_vars_bnd);
      auto num_heavy_vars = vars_bin_offsets.back() - heavy_vars_beg_id;
      finalize_upd_bnd_kernel<i_t, f_t, f_t2><<<num_heavy_vars, 32, 0, heavy_vars_stream>>>(
        heavy_vars_beg_id, make_span(heavy_vars_block_segments), tmp_vars_bnd, view);
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, i_t block_dim, typename bounds_update_view_t>
void upd_bounds_per_block(managed_stream_pool& streams,
                          bounds_update_view_t view,
                          const std::vector<i_t>& vars_bin_offsets,
                          i_t degree_beg,
                          i_t degree_end,
                          bool dry_run)
{
  static_assert(block_dim <= 1024, "Cannot launch kernel with more than 1024 threads");

  auto [vars_id_beg, vars_id_end] = get_id_range(vars_bin_offsets, degree_beg, degree_end);

  auto block_count = vars_id_end - vars_id_beg;
  if (block_count > 0) {
    auto block_stream = streams.get_stream();
    if (!dry_run) {
      lb_upd_bnd_block_kernel<i_t, f_t, f_t2, block_dim>
        <<<block_count, block_dim, 0, block_stream>>>(vars_id_beg, view);
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, typename bounds_update_view_t>
void upd_bounds_per_block(managed_stream_pool& streams,
                          bounds_update_view_t view,
                          const std::vector<i_t>& vars_bin_offsets,
                          i_t heavy_degree_cutoff,
                          bool dry_run = false)
{
  if (view.nnz < 10000) {
    upd_bounds_per_block<i_t, f_t, f_t2, 32>(streams, view, vars_bin_offsets, 32, 32, dry_run);
    upd_bounds_per_block<i_t, f_t, f_t2, 64>(streams, view, vars_bin_offsets, 64, 64, dry_run);
    upd_bounds_per_block<i_t, f_t, f_t2, 128>(streams, view, vars_bin_offsets, 128, 128, dry_run);
    upd_bounds_per_block<i_t, f_t, f_t2, 256>(streams, view, vars_bin_offsets, 256, 256, dry_run);
  } else {
    //[1024, heavy_degree_cutoff/2] -> 128 block size
    upd_bounds_per_block<i_t, f_t, f_t2, 256>(
      streams, view, vars_bin_offsets, 1024, heavy_degree_cutoff / 2, dry_run);
    //[64, 512] -> 32 block size
    upd_bounds_per_block<i_t, f_t, f_t2, 64>(streams, view, vars_bin_offsets, 128, 512, dry_run);
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_variable,
          typename bounds_update_view_t>
void upd_bounds_sub_warp(managed_stream_pool& streams,
                         bounds_update_view_t view,
                         i_t degree_beg,
                         i_t degree_end,
                         const std::vector<i_t>& vars_bin_offsets,
                         bool dry_run)
{
  constexpr i_t block_dim         = 32;
  auto vars_per_block             = block_dim / threads_per_variable;
  auto [vars_id_beg, vars_id_end] = get_id_range(vars_bin_offsets, degree_beg, degree_end);

  auto block_count = raft::ceildiv<i_t>(vars_id_end - vars_id_beg, vars_per_block);
  if (block_count != 0) {
    auto sub_warp_stream = streams.get_stream();
    if (!dry_run) {
      lb_upd_bnd_sub_warp_kernel<i_t, f_t, f_t2, block_dim, threads_per_variable>
        <<<block_count, block_dim, 0, sub_warp_stream>>>(vars_id_beg, vars_id_end, view);
    }
  }
}

template <typename i_t, typename f_t, typename f_t2, typename bounds_update_view_t>
void upd_bounds_sub_warp(managed_stream_pool& streams,
                         bounds_update_view_t view,
                         i_t vars_sub_warp_count,
                         rmm::device_uvector<i_t>& warp_vars_offsets,
                         rmm::device_uvector<i_t>& warp_vars_id_offsets,
                         bool dry_run)
{
  constexpr i_t block_dim = 256;

  auto block_count = raft::ceildiv<i_t>(vars_sub_warp_count * 32, block_dim);
  if (block_count != 0) {
    auto sub_warp_stream = streams.get_stream();
    if (!dry_run) {
      lb_upd_bnd_sub_warp_kernel<i_t, f_t, f_t2, block_dim>
        <<<block_count, block_dim, 0, sub_warp_stream>>>(
          view, make_span(warp_vars_offsets), make_span(warp_vars_id_offsets));
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_variable,
          typename bounds_update_view_t>
void upd_bounds_sub_warp(managed_stream_pool& streams,
                         bounds_update_view_t view,
                         i_t degree,
                         const std::vector<i_t>& vars_bin_offsets,
                         bool dry_run)
{
  upd_bounds_sub_warp<i_t, f_t, f_t2, threads_per_variable>(
    streams, view, degree, degree, vars_bin_offsets, dry_run);
}

template <typename i_t, typename f_t, typename f_t2, typename bounds_update_view_t>
void upd_bounds_sub_warp(managed_stream_pool& streams,
                         bounds_update_view_t view,
                         bool is_vars_sub_warp_single_bin,
                         i_t vars_sub_warp_count,
                         rmm::device_uvector<i_t>& warp_vars_offsets,
                         rmm::device_uvector<i_t>& warp_vars_id_offsets,
                         const std::vector<i_t>& vars_bin_offsets,
                         bool dry_run = false)
{
  if (view.nnz < 10000) {
    upd_bounds_sub_warp<i_t, f_t, f_t2, 16>(streams, view, 16, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 8>(streams, view, 8, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 4>(streams, view, 4, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 2>(streams, view, 2, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 1>(streams, view, 1, vars_bin_offsets, dry_run);
  } else {
    if (is_vars_sub_warp_single_bin) {
      upd_bounds_sub_warp<i_t, f_t, f_t2, 16>(streams, view, 64, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 8>(streams, view, 32, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 4>(streams, view, 16, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 2>(streams, view, 8, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 1>(streams, view, 1, 4, vars_bin_offsets, dry_run);
    } else {
      upd_bounds_sub_warp<i_t, f_t, f_t2>(
        streams, view, vars_sub_warp_count, warp_vars_offsets, warp_vars_id_offsets, dry_run);
    }
  }
}
}  // namespace cuopt::linear_programming::detail
