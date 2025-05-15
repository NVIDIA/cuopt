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

#include "load_balanced_bounds_next_constraint_kernels.cuh"
#include "load_balanced_bounds_presolve_helpers.cuh"
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

/// CALCULATE ACTIVITY

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t block_dim,
          typename activity_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void calc_activity_heavy_cnst(stream_pool_t& streams,
                              activity_view_t view,
                              upd_view_t upd_0,
                              upd_view_t upd_1,
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
          upd_0,
          upd_1);
      auto num_heavy_cnst = cnst_bin_offsets.back() - heavy_cnst_beg_id;
      if (erase_inf_cnst) {
        finalize_calc_act_kernel<true, i_t, f_t, f_t2>
          <<<num_heavy_cnst, 32, 0, heavy_cnst_stream>>>(
            heavy_cnst_beg_id, make_span(heavy_cnst_block_segments), view, upd_0, upd_1);
      } else {
        finalize_calc_act_kernel<false, i_t, f_t, f_t2>
          <<<num_heavy_cnst, 32, 0, heavy_cnst_stream>>>(
            heavy_cnst_beg_id, make_span(heavy_cnst_block_segments), view, upd_0, upd_1);
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t block_dim,
          typename activity_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void calc_activity_per_block(stream_pool_t& streams,
                             activity_view_t view,
                             upd_view_t upd_0,
                             upd_view_t upd_1,
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
          <<<block_count, block_dim, 0, block_stream>>>(cnst_id_beg, view, upd_0, upd_1);
      } else {
        lb_calc_act_block_kernel<false, i_t, f_t, f_t2, block_dim>
          <<<block_count, block_dim, 0, block_stream>>>(cnst_id_beg, view, upd_0, upd_1);
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          typename activity_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void calc_activity_per_block(stream_pool_t& streams,
                             activity_view_t view,
                             upd_view_t upd_0,
                             upd_view_t upd_1,
                             const std::vector<i_t>& cnst_bin_offsets,
                             i_t heavy_degree_cutoff,
                             bool erase_inf_cnst,
                             bool dry_run = false)
{
  if (view.nnz < 10000) {
    calc_activity_per_block<i_t, f_t, f_t2, 32>(
      streams, view, upd_0, upd_1, cnst_bin_offsets, 32, 32, erase_inf_cnst, dry_run);
    calc_activity_per_block<i_t, f_t, f_t2, 64>(
      streams, view, upd_0, upd_1, cnst_bin_offsets, 64, 64, erase_inf_cnst, dry_run);
    calc_activity_per_block<i_t, f_t, f_t2, 128>(
      streams, view, upd_0, upd_1, cnst_bin_offsets, 128, 128, erase_inf_cnst, dry_run);
    calc_activity_per_block<i_t, f_t, f_t2, 256>(
      streams, view, upd_0, upd_1, cnst_bin_offsets, 256, 256, erase_inf_cnst, dry_run);
  } else {
    //[1024, heavy_degree_cutoff/2] -> 1024 block size
    calc_activity_per_block<i_t, f_t, f_t2, 1024>(streams,
                                                  view,
                                                  upd_0,
                                                  upd_1,
                                                  cnst_bin_offsets,
                                                  1024,
                                                  heavy_degree_cutoff / 2,
                                                  erase_inf_cnst,
                                                  dry_run);
    //[512, 512] -> 128 block size
    calc_activity_per_block<i_t, f_t, f_t2, 128>(
      streams, view, upd_0, upd_1, cnst_bin_offsets, 128, 512, erase_inf_cnst, dry_run);
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_constraint,
          typename activity_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void calc_activity_sub_warp(stream_pool_t& streams,
                            activity_view_t view,
                            upd_view_t upd_0,
                            upd_view_t upd_1,
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
          <<<block_count, block_dim, 0, sub_warp_thread>>>(
            cnst_id_beg, cnst_id_end, view, upd_0, upd_1);
      } else {
        lb_calc_act_sub_warp_kernel<false, i_t, f_t, f_t2, block_dim, threads_per_constraint>
          <<<block_count, block_dim, 0, sub_warp_thread>>>(
            cnst_id_beg, cnst_id_end, view, upd_0, upd_1);
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_constraint,
          typename activity_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void calc_activity_sub_warp(stream_pool_t& streams,
                            activity_view_t view,
                            upd_view_t upd_0,
                            upd_view_t upd_1,
                            i_t degree,
                            const std::vector<i_t>& cnst_bin_offsets,
                            bool erase_inf_cnst,
                            bool dry_run)
{
  calc_activity_sub_warp<i_t, f_t, f_t2, threads_per_constraint>(
    streams, view, upd_0, upd_1, degree, degree, cnst_bin_offsets, erase_inf_cnst, dry_run);
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          typename activity_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void calc_activity_sub_warp(stream_pool_t& streams,
                            activity_view_t view,
                            upd_view_t upd_0,
                            upd_view_t upd_1,
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
            view, upd_0, upd_1, make_span(warp_cnst_offsets), make_span(warp_cnst_id_offsets));
      } else {
        lb_calc_act_sub_warp_kernel<false, i_t, f_t, f_t2, block_dim>
          <<<block_count, block_dim, 0, sub_warp_stream>>>(
            view, upd_0, upd_1, make_span(warp_cnst_offsets), make_span(warp_cnst_id_offsets));
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          typename activity_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void calc_activity_sub_warp(stream_pool_t& streams,
                            activity_view_t view,
                            upd_view_t upd_0,
                            upd_view_t upd_1,
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
      streams, view, upd_0, upd_1, 16, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 8>(
      streams, view, upd_0, upd_1, 8, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 4>(
      streams, view, upd_0, upd_1, 4, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 2>(
      streams, view, upd_0, upd_1, 2, cnst_bin_offsets, erase_inf_cnst, dry_run);
    calc_activity_sub_warp<i_t, f_t, f_t2, 1>(
      streams, view, upd_0, upd_1, 1, cnst_bin_offsets, erase_inf_cnst, dry_run);
  } else {
    if (is_cnst_sub_warp_single_bin) {
      calc_activity_sub_warp<i_t, f_t, f_t2, 16>(
        streams, view, upd_0, upd_1, 64, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 8>(
        streams, view, upd_0, upd_1, 32, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 4>(
        streams, view, upd_0, upd_1, 16, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 2>(
        streams, view, upd_0, upd_1, 8, cnst_bin_offsets, erase_inf_cnst, dry_run);
      calc_activity_sub_warp<i_t, f_t, f_t2, 1>(
        streams, view, upd_0, upd_1, 1, 4, cnst_bin_offsets, erase_inf_cnst, dry_run);
    } else {
      calc_activity_sub_warp<i_t, f_t, f_t2>(streams,
                                             view,
                                             upd_0,
                                             upd_1,
                                             cnst_sub_warp_count,
                                             warp_cnst_offsets,
                                             warp_cnst_id_offsets,
                                             erase_inf_cnst,
                                             dry_run);
    }
  }
}

// bounds

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t block_dim,
          typename bounds_update_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void upd_bounds_heavy_vars(stream_pool_t& streams,
                           bounds_update_view_t view,
                           upd_view_t upd_0,
                           upd_view_t upd_1,
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
          upd_0,
          upd_1);
      auto num_heavy_vars = vars_bin_offsets.back() - heavy_vars_beg_id;
      finalize_upd_bnd_kernel<i_t, f_t, f_t2><<<num_heavy_vars, 32, 0, heavy_vars_stream>>>(
        heavy_vars_beg_id, make_span(heavy_vars_block_segments), view, upd_0, upd_1);
      lb_upd_next_constraint_heavy_kernel<i_t, f_t, block_dim>
        <<<num_blocks_heavy_vars, block_dim, 0, heavy_vars_stream>>>(
          heavy_vars_beg_id,
          make_span(heavy_vars_vertex_ids),
          make_span(heavy_vars_pseudo_block_ids),
          heavy_degree_cutoff,
          view,
          upd_0,
          upd_1);
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t block_dim,
          typename bounds_update_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void upd_bounds_per_block(stream_pool_t& streams,
                          bounds_update_view_t view,
                          upd_view_t upd_0,
                          upd_view_t upd_1,
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
        <<<block_count, block_dim, 0, block_stream>>>(vars_id_beg, view, upd_0, upd_1);
      lb_upd_next_constraint_block_kernel<i_t, f_t, block_dim>
        <<<block_count, block_dim, 0, block_stream>>>(vars_id_beg, view, upd_0, upd_1);
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          typename bounds_update_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void upd_bounds_per_block(stream_pool_t& streams,
                          bounds_update_view_t view,
                          upd_view_t upd_0,
                          upd_view_t upd_1,
                          const std::vector<i_t>& vars_bin_offsets,
                          i_t heavy_degree_cutoff,
                          bool dry_run = false)
{
  if (view.nnz < 10000) {
    upd_bounds_per_block<i_t, f_t, f_t2, 32>(
      streams, view, upd_0, upd_1, vars_bin_offsets, 32, 32, dry_run);
    upd_bounds_per_block<i_t, f_t, f_t2, 64>(
      streams, view, upd_0, upd_1, vars_bin_offsets, 64, 64, dry_run);
    upd_bounds_per_block<i_t, f_t, f_t2, 128>(
      streams, view, upd_0, upd_1, vars_bin_offsets, 128, 128, dry_run);
    upd_bounds_per_block<i_t, f_t, f_t2, 256>(
      streams, view, upd_0, upd_1, vars_bin_offsets, 256, 256, dry_run);
  } else {
    //[1024, heavy_degree_cutoff/2] -> 128 block size
    upd_bounds_per_block<i_t, f_t, f_t2, 256>(
      streams, view, upd_0, upd_1, vars_bin_offsets, 1024, heavy_degree_cutoff / 2, dry_run);
    //[64, 512] -> 32 block size
    upd_bounds_per_block<i_t, f_t, f_t2, 64>(
      streams, view, upd_0, upd_1, vars_bin_offsets, 128, 512, dry_run);
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_variable,
          typename bounds_update_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void upd_bounds_sub_warp(stream_pool_t& streams,
                         bounds_update_view_t view,
                         upd_view_t upd_0,
                         upd_view_t upd_1,
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
        <<<block_count, block_dim, 0, sub_warp_stream>>>(
          vars_id_beg, vars_id_end, view, upd_0, upd_1);
      lb_upd_next_constraint_sub_warp_kernel<i_t, f_t, block_dim, threads_per_variable>
        <<<block_count, block_dim, 0, sub_warp_stream>>>(
          vars_id_beg, vars_id_end, view, upd_0, upd_1);
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          typename bounds_update_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void upd_bounds_sub_warp(stream_pool_t& streams,
                         bounds_update_view_t view,
                         upd_view_t upd_0,
                         upd_view_t upd_1,
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
          view, upd_0, upd_1, make_span(warp_vars_offsets), make_span(warp_vars_id_offsets));
      lb_upd_next_constraint_sub_warp_kernel<i_t, f_t, block_dim>
        <<<block_count, block_dim, 0, sub_warp_stream>>>(
          view, upd_0, upd_1, make_span(warp_vars_offsets), make_span(warp_vars_id_offsets));
    }
  }
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          i_t threads_per_variable,
          typename bounds_update_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void upd_bounds_sub_warp(stream_pool_t& streams,
                         bounds_update_view_t view,
                         upd_view_t upd_0,
                         upd_view_t upd_1,
                         i_t degree,
                         const std::vector<i_t>& vars_bin_offsets,
                         bool dry_run)
{
  upd_bounds_sub_warp<i_t, f_t, f_t2, threads_per_variable>(
    streams, view, upd_0, upd_1, degree, degree, vars_bin_offsets, dry_run);
}

template <typename i_t,
          typename f_t,
          typename f_t2,
          typename bounds_update_view_t,
          typename upd_view_t,
          typename stream_pool_t>
void upd_bounds_sub_warp(stream_pool_t& streams,
                         bounds_update_view_t view,
                         upd_view_t upd_0,
                         upd_view_t upd_1,
                         bool is_vars_sub_warp_single_bin,
                         i_t vars_sub_warp_count,
                         rmm::device_uvector<i_t>& warp_vars_offsets,
                         rmm::device_uvector<i_t>& warp_vars_id_offsets,
                         const std::vector<i_t>& vars_bin_offsets,
                         bool dry_run = false)
{
  if (view.nnz < 10000) {
    upd_bounds_sub_warp<i_t, f_t, f_t2, 16>(
      streams, view, upd_0, upd_1, 16, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 8>(
      streams, view, upd_0, upd_1, 8, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 4>(
      streams, view, upd_0, upd_1, 4, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 2>(
      streams, view, upd_0, upd_1, 2, vars_bin_offsets, dry_run);
    upd_bounds_sub_warp<i_t, f_t, f_t2, 1>(
      streams, view, upd_0, upd_1, 1, vars_bin_offsets, dry_run);
  } else {
    if (is_vars_sub_warp_single_bin) {
      upd_bounds_sub_warp<i_t, f_t, f_t2, 16>(
        streams, view, upd_0, upd_1, 64, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 8>(
        streams, view, upd_0, upd_1, 32, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 4>(
        streams, view, upd_0, upd_1, 16, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 2>(
        streams, view, upd_0, upd_1, 8, vars_bin_offsets, dry_run);
      upd_bounds_sub_warp<i_t, f_t, f_t2, 1>(
        streams, view, upd_0, upd_1, 1, 4, vars_bin_offsets, dry_run);
    } else {
      upd_bounds_sub_warp<i_t, f_t, f_t2>(streams,
                                          view,
                                          upd_0,
                                          upd_1,
                                          vars_sub_warp_count,
                                          warp_vars_offsets,
                                          warp_vars_id_offsets,
                                          dry_run);
    }
  }
}

}  // namespace cuopt::linear_programming::detail
