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

#include <mip/mip_constants.hpp>
#include "spmv.cuh"
#include "spmv_helpers.cuh"
#include "spmv_setup_helpers.cuh"

#include <nvtx3/nvtx3.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
spmv_t<i_t, f_t>::spmv_t(problem_t<i_t, f_t>& problem_, bool debug)
  : pb(&problem_),
    handle_ptr(pb->handle_ptr),
    n_constraints(pb->n_constraints),
    n_variables(pb->n_variables),
    nnz(pb->nnz),
    cnst_reorg_ids(n_constraints, handle_ptr->get_stream()),
    coefficients(nnz, handle_ptr->get_stream()),
    variables(nnz, handle_ptr->get_stream()),
    offsets(n_constraints + 1, handle_ptr->get_stream()),
    vars_reorg_ids(n_variables, handle_ptr->get_stream()),
    reverse_coefficients(nnz, handle_ptr->get_stream()),
    reverse_constraints(nnz, handle_ptr->get_stream()),
    reverse_offsets(n_variables + 1, handle_ptr->get_stream()),
    tmp_cnst_ids(n_constraints, handle_ptr->get_stream()),
    tmp_vars_ids(n_variables, handle_ptr->get_stream()),
    heavy_cnst_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_vars_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    num_blocks_heavy_cnst(0),
    num_blocks_heavy_vars(0),
    tmp_ax(0, problem_.handle_ptr->get_stream()),
    tmp_aty(0, problem_.handle_ptr->get_stream()),
    warp_cnst_offsets(0, problem_.handle_ptr->get_stream()),
    warp_cnst_id_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_id_offsets(0, problem_.handle_ptr->get_stream()),
    cnst_binner(handle_ptr),
    vars_binner(handle_ptr)
{
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::setup(problem_t<i_t, f_t>& problem_, bool debug)
{
  //std::cerr<<"set up begin\n";
  pb = &problem_;
  setup_lb_problem(problem_, debug);
  setup_lb_meta();
  //std::cerr<<"set up done\n";
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::setup_lb_problem(problem_t<i_t, f_t>& problem_, bool debug)
{
  pb            = &problem_;
  handle_ptr    = pb->handle_ptr;
  n_constraints = pb->n_constraints;
  n_variables   = pb->n_variables;
  nnz           = pb->nnz;
  cnst_reorg_ids.resize(n_constraints, handle_ptr->get_stream());
  coefficients.resize(nnz, handle_ptr->get_stream());
  variables.resize(nnz, handle_ptr->get_stream());
  offsets.resize(n_constraints + 1, handle_ptr->get_stream());
  vars_reorg_ids.resize(n_variables, handle_ptr->get_stream());
  reverse_coefficients.resize(nnz, handle_ptr->get_stream());
  reverse_constraints.resize(nnz, handle_ptr->get_stream());
  reverse_offsets.resize(n_variables + 1, handle_ptr->get_stream());
  tmp_cnst_ids.resize(n_constraints, handle_ptr->get_stream());
  tmp_vars_ids.resize(n_variables, handle_ptr->get_stream());

  cnst_binner.setup(pb->offsets.data(), nullptr, 0, n_constraints);
  auto dist_cnst = cnst_binner.run(tmp_cnst_ids, handle_ptr);
  vars_binner.setup(pb->reverse_offsets.data(), nullptr, 0, n_variables);
  auto dist_vars = vars_binner.run(tmp_vars_ids, handle_ptr);

  auto cnst_bucket = dist_cnst.degree_range();
  auto vars_bucket = dist_vars.degree_range();

  cnst_reorg_ids.resize(cnst_bucket.vertex_ids.size(), handle_ptr->get_stream());
  vars_reorg_ids.resize(vars_bucket.vertex_ids.size(), handle_ptr->get_stream());

  raft::copy(cnst_reorg_ids.data(),
             cnst_bucket.vertex_ids.data(),
             cnst_bucket.vertex_ids.size(),
             handle_ptr->get_stream());
  raft::copy(vars_reorg_ids.data(),
             vars_bucket.vertex_ids.data(),
             vars_bucket.vertex_ids.size(),
             handle_ptr->get_stream());

  create_graph<i_t, f_t>(handle_ptr,
                         cnst_reorg_ids,
                         offsets,
                         coefficients,
                         variables,
                         problem_.offsets,
                         problem_.coefficients,
                         problem_.variables,
                         debug);

  create_graph<i_t, f_t>(handle_ptr,
                         vars_reorg_ids,
                         reverse_offsets,
                         reverse_coefficients,
                         reverse_constraints,
                         problem_.reverse_offsets,
                         problem_.reverse_coefficients,
                         problem_.reverse_constraints,
                         debug);

  cnst_bin_offsets = dist_cnst.bin_offsets_;
  vars_bin_offsets = dist_vars.bin_offsets_;

  handle_ptr->sync_stream();
  //std::cerr<<"lb setup\n";
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::setup_lb_meta()
{
  auto stream = handle_ptr->get_stream();
  stream.synchronize();

  //std::cerr<<"cnst create_heavy_item_block_segments\n";
  num_blocks_heavy_cnst = create_heavy_item_block_segments(stream,
                                                           heavy_cnst_vertex_ids,
                                                           heavy_cnst_pseudo_block_ids,
                                                           heavy_cnst_block_segments,
                                                           heavy_degree_cutoff,
                                                           cnst_bin_offsets,
                                                           offsets);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  //std::cerr<<"vars create_heavy_item_block_segments\n";
  num_blocks_heavy_vars = create_heavy_item_block_segments(stream,
                                                           heavy_vars_vertex_ids,
                                                           heavy_vars_pseudo_block_ids,
                                                           heavy_vars_block_segments,
                                                           heavy_degree_cutoff,
                                                           vars_bin_offsets,
                                                           reverse_offsets);

  tmp_ax.resize(num_blocks_heavy_cnst, stream);
  tmp_aty.resize(num_blocks_heavy_vars, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  //std::cerr<<"cnst block_meta\n";
  i_t w_t_r = 4;
  std::tie(cnst_sub_warp_count, cnst_med_block_count, cnst_heavy_beg_id) =
    block_meta(stream,
               warp_cnst_offsets,
               warp_cnst_id_offsets,
               cnst_bin_offsets,
               w_t_r,
               heavy_degree_cutoff,
               true);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  //std::cerr<<"vars block_meta\n";
  std::tie(vars_sub_warp_count, vars_med_block_count, vars_heavy_beg_id) =
    block_meta(stream,
               warp_vars_offsets,
               warp_vars_id_offsets,
               vars_bin_offsets,
               w_t_r,
               heavy_degree_cutoff,
               true);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  //std::cerr<<"lb meta setup\n";
}

template <typename i_t, typename f_t>
typename spmv_t<i_t, f_t>::view_t spmv_t<i_t, f_t>::get_A_view()
{
  spmv_t::view_t v;
  v.reorg_ids = make_span(cnst_reorg_ids);
  v.coeff     = make_span(coefficients);
  v.elem      = make_span(variables);
  v.offsets   = make_span(offsets);
  v.nnz       = nnz;
  return v;
}

template <typename i_t, typename f_t>
typename spmv_t<i_t, f_t>::view_t spmv_t<i_t, f_t>::get_AT_view()
{
  spmv_t::view_t v;
  v.reorg_ids = make_span(vars_reorg_ids);
  v.coeff     = make_span(reverse_coefficients);
  v.elem      = make_span(reverse_constraints);
  v.offsets   = make_span(reverse_offsets);
  v.nnz       = nnz;
  return v;
}

#if MIP_INSTANTIATE_FLOAT
template class spmv_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class spmv_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
