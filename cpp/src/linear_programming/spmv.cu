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

#include <linear_programming/pdlp.cuh>
#include <mip/mip_constants.hpp>
#include "spmv.cuh"
#include "spmv_helpers.cuh"
#include "spmv_setup_helpers.cuh"
#include "utils.cuh"

#include <nvtx3/nvtx3.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
spmv_t<i_t, f_t>::spmv_t(problem_t<i_t, f_t>& problem_,
                         raft::device_span<f_t> ax_input_,
                         raft::device_span<f_t> ax_output_,
                         raft::device_span<f_t> aty_input_,
                         raft::device_span<f_t> aty_output_,
                         raft::device_span<f_t> aty_next_input_,
                         raft::device_span<f_t> aty_next_output_,
                         const f_t* primal_step_size,
                         raft::device_span<f_t> primal_solution,
                         raft::device_span<f_t> next_primal_solution,
                         raft::device_span<f_t> delta_primal,
                         const f_t* dual_step_size,
                         raft::device_span<f_t> dual_solution,
                         raft::device_span<f_t> next_dual_solution,
                         raft::device_span<f_t> delta_dual,
                         bool debug)
  : streams(16),
    pb(&problem_),
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
    vars_binner(handle_ptr),
    ax_input(ax_input_),
    ax_output(ax_output_),
    aty_input(aty_input_),
    aty_output(aty_output_),
    aty_next_input(aty_next_input_),
    aty_next_output(aty_next_output_),
    ax_graph_created(false),
    aty_graph_created(false),
    aty_next_graph_created(false),
    ax_exec(nullptr),
    aty_exec(nullptr),
    aty_exec_proj(nullptr),
    aty_next_exec(nullptr),
    aty_next_exec_proj(nullptr),
    current_primal_projection_functor_(primal_step_size,
                                       primal_solution.data(),
                                       problem_.objective_coefficients.data(),
                                       problem_.variable_lower_bounds.data(),
                                       problem_.variable_upper_bounds.data(),
                                       delta_primal.data(),
                                       ax_input_.data(),
                                       next_primal_solution.data()),
    next_primal_projection_functor_(primal_step_size,
                                    next_primal_solution.data(),
                                    problem_.objective_coefficients.data(),
                                    problem_.variable_lower_bounds.data(),
                                    problem_.variable_upper_bounds.data(),
                                    delta_primal.data(),
                                    ax_input_.data(),
                                    primal_solution.data()),
    current_dual_projection_functor_(dual_step_size,
                                     dual_solution.data(),
                                     problem_.constraint_lower_bounds.data(),
                                     problem_.constraint_upper_bounds.data(),
                                     next_dual_solution.data(),
                                     delta_dual.data()),
    next_dual_projection_functor_(dual_step_size,
                                  next_dual_solution.data(),
                                  problem_.constraint_lower_bounds.data(),
                                  problem_.constraint_upper_bounds.data(),
                                  dual_solution.data(),
                                  delta_dual.data()),
    current_step_size_functor_(aty_output_.data(), ax_input_.data()),
    next_step_size_functor_(aty_next_output_.data(), ax_input_.data())
{
  setup_lb_problem(problem_, debug);
  setup_lb_meta();
}
template <typename i_t, typename f_t>
spmv_t<i_t, f_t>::~spmv_t()
{
  if (ax_graph_created) { cudaGraphExecDestroy(ax_exec); }
  if (aty_graph_created) { cudaGraphExecDestroy(aty_exec); }
  if (aty_next_graph_created) { cudaGraphExecDestroy(aty_next_exec); }
  if (aty_graph_proj_created) { cudaGraphExecDestroy(aty_exec_proj); }
  if (aty_graph_proj_next_created) { cudaGraphExecDestroy(aty_next_exec_proj); }
  if (ax_graph_proj_created) { cudaGraphExecDestroy(ax_exec_proj); }
  if (ax_graph_proj_next_created) { cudaGraphExecDestroy(ax_next_exec_proj); }
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

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 0\n";

  cnst_binner.setup(pb->offsets.data(), nullptr, 0, n_constraints);
  auto dist_cnst = cnst_binner.run(tmp_cnst_ids, handle_ptr);
  vars_binner.setup(pb->reverse_offsets.data(), nullptr, 0, n_variables);
  auto dist_vars = vars_binner.run(tmp_vars_ids, handle_ptr);

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 1\n";

  auto cnst_bucket = dist_cnst.degree_range();
  auto vars_bucket = dist_vars.degree_range();

  cnst_reorg_ids.resize(cnst_bucket.vertex_ids.size(), handle_ptr->get_stream());
  vars_reorg_ids.resize(vars_bucket.vertex_ids.size(), handle_ptr->get_stream());

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 2\n";

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
  if (nnz < 10000) {
    compact_bins(cnst_bin_offsets, n_constraints);
    compact_bins(vars_bin_offsets, n_variables);
  }

  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::setup_lb_meta()
{
  auto stream = handle_ptr->get_stream();
  stream.synchronize();
  ax_graph_created            = false;
  aty_graph_created           = false;
  aty_next_graph_created      = false;
  aty_graph_proj_created      = false;
  aty_graph_proj_next_created = false;
  ax_graph_proj_created       = false;
  ax_graph_proj_next_created  = false;

  num_blocks_heavy_cnst = create_heavy_item_block_segments(stream,
                                                           heavy_cnst_vertex_ids,
                                                           heavy_cnst_pseudo_block_ids,
                                                           heavy_cnst_block_segments,
                                                           heavy_degree_cutoff,
                                                           cnst_bin_offsets,
                                                           offsets);

  num_blocks_heavy_vars = create_heavy_item_block_segments(stream,
                                                           heavy_vars_vertex_ids,
                                                           heavy_vars_pseudo_block_ids,
                                                           heavy_vars_block_segments,
                                                           heavy_degree_cutoff,
                                                           vars_bin_offsets,
                                                           reverse_offsets);

  tmp_ax.resize(num_blocks_heavy_cnst, stream);
  tmp_aty.resize(num_blocks_heavy_vars, stream);

  std::tie(is_cnst_sub_warp_single_bin, cnst_sub_warp_count) =
    sub_warp_meta(stream, warp_cnst_offsets, warp_cnst_id_offsets, cnst_bin_offsets, 4);

  std::tie(is_vars_sub_warp_single_bin, vars_sub_warp_count) =
    sub_warp_meta(stream, warp_vars_offsets, warp_vars_id_offsets, vars_bin_offsets, 4);

  RAFT_CHECK_CUDA(stream.synchronize());
  stream.synchronize();
  streams.sync_all_issued();

  // TODO remove
  if (!ax_graph_created) {
    ax_graph_created = build_graph(
      streams,
      handle_ptr,
      ax_graph,
      ax_exec,
      [&]() { this->call_Ax_graph(ax_input, ax_output, true); },
      [&]() { this->call_Ax_graph(ax_input, ax_output); });
  }

  streams.sync_all_issued();
  if (!aty_graph_created) {
    aty_graph_created = build_graph(
      streams,
      handle_ptr,
      aty_graph,
      aty_exec,
      [this]() { this->call_ATy_graph(aty_input, aty_output, true, next_step_size_functor_); },
      [this]() { this->call_ATy_graph(aty_input, aty_output, false, next_step_size_functor_); });
  }

  streams.sync_all_issued();
  if (!aty_next_graph_created) {
    aty_next_graph_created = build_graph(
      streams,
      handle_ptr,
      aty_next_graph,
      aty_next_exec,
      [this]() {
        this->call_ATy_graph(aty_next_input, aty_next_output, true, current_step_size_functor_);
      },
      [this]() {
        this->call_ATy_graph(aty_next_input, aty_next_output, false, current_step_size_functor_);
      });
  }

  if (!aty_graph_proj_created) {
    aty_graph_proj_created = build_graph(
      streams,
      handle_ptr,
      aty_graph_proj,
      aty_exec_proj,
      [this]() {
        this->call_ATy_graph(aty_input, aty_output, true, current_primal_projection_functor_);
      },
      [this]() {
        this->call_ATy_graph(aty_input, aty_output, false, current_primal_projection_functor_);
      });
  }

  if (!aty_graph_proj_next_created) {
    aty_graph_proj_next_created = build_graph(
      streams,
      handle_ptr,
      aty_graph_proj_next,
      aty_next_exec_proj,
      [this]() {
        this->call_ATy_graph(
          aty_next_input, aty_next_output, true, next_primal_projection_functor_);
      },
      [this]() {
        this->call_ATy_graph(
          aty_next_input, aty_next_output, false, next_primal_projection_functor_);
      });
  }

  if (!ax_graph_proj_created) {
    ax_graph_proj_created = build_graph(
      streams,
      handle_ptr,
      ax_graph_proj,
      ax_exec_proj,
      [this]() {
        this->call_Ax_graph(ax_input, ax_output, true, current_dual_projection_functor_);
      },
      [this]() {
        this->call_Ax_graph(ax_input, ax_output, false, current_dual_projection_functor_);
      });
  }

  if (!ax_graph_proj_next_created) {
    ax_graph_proj_next_created = build_graph(
      streams,
      handle_ptr,
      ax_graph_proj_next,
      ax_next_exec_proj,
      // No ax_next_output since we don't write the Ax result anyway
      [this]() { this->call_Ax_graph(ax_input, ax_output, true, next_dual_projection_functor_); },
      [this]() { this->call_Ax_graph(ax_input, ax_output, false, next_dual_projection_functor_); });
  }
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::Ax(const raft::handle_t* h)
{
  raft::common::nvtx::range scope("ax");
  cudaGraphLaunch(ax_exec, h->get_stream());
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::Ax_projection(const raft::handle_t* h, i_t total_pdlp_iterations)
{
  raft::common::nvtx::range scope("ax");
  if (total_pdlp_iterations % 2 == 0) {
    RAFT_CUDA_TRY(cudaGraphLaunch(ax_exec_proj, h->get_stream()));
  } else if (total_pdlp_iterations % 2 == 1) {
    RAFT_CUDA_TRY(cudaGraphLaunch(ax_next_exec_proj, h->get_stream()));
  } else {
    std::cerr << "Ax projection unexpected call\n";
  }
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::ATy_projection(const raft::handle_t* h, i_t total_pdlp_iterations)
{
  raft::common::nvtx::range scope("ay");
  if (total_pdlp_iterations % 2 == 0) {
    RAFT_CUDA_TRY(cudaGraphLaunch(aty_exec_proj, h->get_stream()));
  } else if (total_pdlp_iterations % 2 == 1) {
    RAFT_CUDA_TRY(cudaGraphLaunch(aty_next_exec_proj, h->get_stream()));
  } else {
    std::cerr << "ATy projection unexpected call\n";
  }
}

// TODO use total instead
template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::ATy(const raft::handle_t* h,
                           raft::device_span<f_t> input,
                           raft::device_span<f_t> output)
{
  raft::common::nvtx::range scope("ay");
  if (input.data() == aty_input.data() && output.data() == aty_output.data()) {
    cudaGraphLaunch(aty_exec, h->get_stream());
  } else if (input.data() == aty_next_input.data() && output.data() == aty_next_output.data()) {
    cudaGraphLaunch(aty_next_exec, h->get_stream());
  } else {
    std::cerr << "ATy unexpected call\n";
  }
}

template <typename i_t, typename f_t>
typename spmv_t<i_t, f_t>::spmv_view_t spmv_t<i_t, f_t>::get_A_view()
{
  spmv_t::spmv_view_t v;
  v.reorg_ids = make_span(cnst_reorg_ids);
  v.coeff     = make_span(coefficients);
  v.elem      = make_span(variables);
  v.offsets   = make_span(offsets);
  v.nnz       = nnz;
  return v;
}

template <typename i_t, typename f_t>
typename spmv_t<i_t, f_t>::spmv_view_t spmv_t<i_t, f_t>::get_AT_view()
{
  spmv_t::spmv_view_t v;
  v.reorg_ids = make_span(vars_reorg_ids);
  v.coeff     = make_span(reverse_coefficients);
  v.elem      = make_span(reverse_constraints);
  v.offsets   = make_span(reverse_offsets);
  v.nnz       = nnz;
  return v;
}

template <typename i_t, typename f_t>
template <typename functor_t>
void spmv_t<i_t, f_t>::call_Ax_graph(raft::device_span<f_t> input,
                                     raft::device_span<f_t> output,
                                     bool dry_run,
                                     functor_t functor)
{
  auto view = get_A_view();

  spmv_heavy<i_t, f_t, 640, spmv_view_t, functor_t>(streams,
                                                    view,
                                                    input,
                                                    output,
                                                    make_span(tmp_ax),
                                                    heavy_cnst_vertex_ids,
                                                    heavy_cnst_pseudo_block_ids,
                                                    heavy_cnst_block_segments,
                                                    cnst_bin_offsets,
                                                    heavy_degree_cutoff,
                                                    num_blocks_heavy_cnst,
                                                    dry_run,
                                                    functor);

  spmv_per_block<i_t, f_t, spmv_view_t, functor_t>(
    streams, view, input, output, cnst_bin_offsets, heavy_degree_cutoff, functor, dry_run);
  spmv_sub_warp<i_t, f_t, spmv_view_t, functor_t>(streams,
                                                  view,
                                                  input,
                                                  output,
                                                  is_cnst_sub_warp_single_bin,
                                                  cnst_sub_warp_count,
                                                  warp_cnst_offsets,
                                                  warp_cnst_id_offsets,
                                                  cnst_bin_offsets,
                                                  functor,
                                                  dry_run);
}

template <typename i_t, typename f_t>
template <typename functor_t>
void spmv_t<i_t, f_t>::call_ATy_graph(raft::device_span<f_t> input,
                                      raft::device_span<f_t> output,
                                      bool dry_run,
                                      functor_t functor)
{
  auto view = get_AT_view();

  spmv_heavy<i_t, f_t, 640, spmv_view_t, functor_t>(streams,
                                                    view,
                                                    input,
                                                    output,
                                                    make_span(tmp_aty),
                                                    heavy_vars_vertex_ids,
                                                    heavy_vars_pseudo_block_ids,
                                                    heavy_vars_block_segments,
                                                    vars_bin_offsets,
                                                    heavy_degree_cutoff,
                                                    num_blocks_heavy_vars,
                                                    dry_run,
                                                    functor);

  spmv_per_block<i_t, f_t, spmv_view_t, functor_t>(
    streams, view, input, output, vars_bin_offsets, heavy_degree_cutoff, functor, dry_run);
  spmv_sub_warp<i_t, f_t, spmv_view_t, functor_t>(streams,
                                                  view,
                                                  input,
                                                  output,
                                                  is_vars_sub_warp_single_bin,
                                                  vars_sub_warp_count,
                                                  warp_vars_offsets,
                                                  warp_vars_id_offsets,
                                                  vars_bin_offsets,
                                                  functor,
                                                  dry_run);
}

#if MIP_INSTANTIATE_FLOAT
template class spmv_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class spmv_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
