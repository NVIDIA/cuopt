/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

// THIS IS LIKELY THE INNER-MOST INCLUDE
// FOR COMPILE TIME, WE SHOULD KEEP THE INCLUDES ON THIS HEADER MINIMAL

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>

#include <utilities/macros.cuh>

#include <mip/logger.hpp>
#include <mip/problem/problem.cuh>

#include <mip/presolve/load_balanced_partition_helpers.cuh>
#include <raft/core/nvtx.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include "spmv_helpers.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class spmv_t {
 public:
  struct view_t {
    raft::device_span<const i_t> reorg_ids;
    raft::device_span<const i_t> offsets;
    raft::device_span<const i_t> elem;
    raft::device_span<const f_t> coeff;
    i_t nnz;
  };

  spmv_t(problem_t<i_t, f_t>& problem, bool debug = false);
  spmv_t() = delete;
  void setup_lb_problem(problem_t<i_t, f_t>& problem, bool debug = false);
  void setup_lb_meta();
  void setup(problem_t<i_t, f_t>& problem, bool debug = false);
  view_t get_A_view();
  view_t get_AT_view();

  template <typename functor_t = identity_functor<i_t, f_t>>
  void Ax(rmm::cuda_stream_view stream,
          raft::device_span<f_t> input,
          raft::device_span<f_t> output,
          functor_t functor = identity_functor<i_t, f_t>{});

  template <typename functor_t = identity_functor<i_t, f_t>>
  void ATy(rmm::cuda_stream_view stream,
           raft::device_span<f_t> input,
           raft::device_span<f_t> output,
           functor_t functor = identity_functor<i_t, f_t>{});

  static constexpr i_t heavy_degree_cutoff = 16 * 1024;
  problem_t<i_t, f_t>* pb;
  const raft::handle_t* handle_ptr;

  i_t n_variables;
  i_t n_constraints;
  i_t nnz;

  // csr - cnst
  rmm::device_uvector<i_t> cnst_reorg_ids;
  rmm::device_uvector<f_t> coefficients;
  rmm::device_uvector<i_t> variables;
  rmm::device_uvector<i_t> offsets;

  // csc - vars
  rmm::device_uvector<i_t> vars_reorg_ids;
  rmm::device_uvector<f_t> reverse_coefficients;
  rmm::device_uvector<i_t> reverse_constraints;
  rmm::device_uvector<i_t> reverse_offsets;

  // lb members
  rmm::device_uvector<i_t> tmp_cnst_ids;
  rmm::device_uvector<i_t> tmp_vars_ids;

  // Number of blocks for heavy ids
  rmm::device_uvector<i_t> heavy_cnst_block_segments;
  rmm::device_uvector<i_t> heavy_vars_block_segments;
  rmm::device_uvector<i_t> heavy_cnst_vertex_ids;
  rmm::device_uvector<i_t> heavy_vars_vertex_ids;
  rmm::device_uvector<i_t> heavy_cnst_pseudo_block_ids;
  rmm::device_uvector<i_t> heavy_vars_pseudo_block_ids;

  i_t num_blocks_heavy_cnst;
  i_t num_blocks_heavy_vars;

  rmm::device_uvector<f_t> tmp_ax;
  rmm::device_uvector<f_t> tmp_aty;

  // lb sub-warp opt members
  i_t cnst_heavy_beg_id;
  i_t cnst_sub_warp_count;
  i_t cnst_med_block_count;
  rmm::device_uvector<i_t> warp_cnst_offsets;
  rmm::device_uvector<i_t> warp_cnst_id_offsets;

  i_t vars_heavy_beg_id;
  i_t vars_sub_warp_count;
  i_t vars_med_block_count;
  rmm::device_uvector<i_t> warp_vars_offsets;
  rmm::device_uvector<i_t> warp_vars_id_offsets;

  // binning
  std::vector<i_t> cnst_bin_offsets;
  std::vector<i_t> vars_bin_offsets;

  vertex_bin_t<i_t> cnst_binner;
  vertex_bin_t<i_t> vars_binner;
};

template <typename i_t, typename f_t>
template <typename functor_t>
void spmv_t<i_t, f_t>::Ax(rmm::cuda_stream_view stream,
                          raft::device_span<f_t> input,
                          raft::device_span<f_t> output,
                          functor_t functor)
{
  raft::common::nvtx::range scope("ax");
  //std::cerr<<"spmv_call\n";
  //std::cerr<<"cnst_sub_warp_count "<<cnst_sub_warp_count<<"\n";
  //std::cerr<<"warp_cnst_id_offsets.back() "<<warp_cnst_id_offsets.back_element(stream)<<"\n";
  //std::cerr<<"cnst_med_block_count "<<cnst_med_block_count<<"\n";
  spmv_call(stream,
            get_A_view(),
            input,
            output,
            make_span(tmp_ax),
            cnst_sub_warp_count,
            cnst_med_block_count,
            num_blocks_heavy_cnst,
            cnst_heavy_beg_id,
            pb->n_constraints - cnst_heavy_beg_id,
            heavy_degree_cutoff,
            warp_cnst_offsets,
            warp_cnst_id_offsets,
            heavy_cnst_vertex_ids,
            heavy_cnst_pseudo_block_ids,
            heavy_cnst_block_segments,
            functor);
  //std::cerr<<"spmv_call done\n";
}

template <typename i_t, typename f_t>
template <typename functor_t>
void spmv_t<i_t, f_t>::ATy(rmm::cuda_stream_view stream,
                           raft::device_span<f_t> input,
                           raft::device_span<f_t> output,
                           functor_t functor)
{
  raft::common::nvtx::range scope("aty");
  spmv_call(stream,
            get_AT_view(),
            input,
            output,
            make_span(tmp_aty),
            vars_sub_warp_count,
            vars_med_block_count,
            num_blocks_heavy_vars,
            vars_heavy_beg_id,
            pb->n_variables - vars_heavy_beg_id,
            heavy_degree_cutoff,
            warp_vars_offsets,
            warp_vars_id_offsets,
            heavy_vars_vertex_ids,
            heavy_vars_pseudo_block_ids,
            heavy_vars_block_segments,
            functor);
}

}  // namespace cuopt::linear_programming::detail
