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

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <mip/mip_constants.hpp>
#include <mip/problem/load_balanced_problem.cuh>
#include <utilities/device_utils.cuh>

#include <cub/cub.cuh>
#include <raft/common/nvtx.hpp>
#include "load_balanced_bounds_presolve.cuh"
#include "load_balanced_bounds_presolve_helpers.cuh"

#include <limits>

namespace cuopt::linear_programming::detail {

// Tobias Achterberg, Robert E. Bixby, Zonghao Gu, Edward Rothberg, Dieter Weninger (2019) Presolve
// Reductions in Mixed Integer Programming. INFORMS Journal on Computing 32(2):473-506.
// https://doi.org/10.1287/ijoc.2018.0857

// This code follows the paper mentioned above, section 3.2
// The solve function runs for a set number of iterations or until the expiry
// of the time limit.
// In each iteration, the minimal activity of all the constraints are calculated
// In infeasbility is not found, then a variable is selected and its bounds are
// updated. This update will invalidate minimal activity which is recalculated
// in the next iteration.
// If no updates to the bounds are detected then the loop is broken and the new
// bounds (if found) are applied to the problem.

template <typename i_t, typename f_t>
load_balanced_bounds_presolve_t<i_t, f_t>::load_balanced_bounds_presolve_t(
  const load_balanced_problem_t<i_t, f_t>& problem_,
  mip_solver_context_t<i_t, f_t>& context_,
  settings_t in_settings,
  i_t max_stream_count_)
  : streams(max_stream_count_),
    pb(&problem_),
    bounds_changed(problem_.handle_ptr->get_stream()),
    cnst_slack(0, problem_.handle_ptr->get_stream()),
    vars_bnd(0, problem_.handle_ptr->get_stream()),
    tmp_act(0, problem_.handle_ptr->get_stream()),
    tmp_bnd(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_vars_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    num_blocks_heavy_cnst(0),
    num_blocks_heavy_vars(0),
    settings(in_settings),
    calc_slack_exec(nullptr),
    calc_slack_erase_inf_cnst_exec(nullptr),
    upd_bnd_exec(nullptr),
    calc_slack_erase_inf_cnst_graph_created(false),
    calc_slack_graph_created(false),
    upd_bnd_graph_created(false),
    warp_cnst_offsets(0, problem_.handle_ptr->get_stream()),
    warp_cnst_id_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_id_offsets(0, problem_.handle_ptr->get_stream()),
    context(context_)
{
  setup(problem_);
}

template <typename i_t, typename f_t>
load_balanced_bounds_presolve_t<i_t, f_t>::~load_balanced_bounds_presolve_t()
{
  if (calc_slack_erase_inf_cnst_graph_created) {
    cudaGraphExecDestroy(calc_slack_erase_inf_cnst_exec);
  }
  if (calc_slack_graph_created) { cudaGraphExecDestroy(calc_slack_exec); }
  if (upd_bnd_graph_created) { cudaGraphExecDestroy(upd_bnd_exec); }
}

template <typename i_t>
std::pair<bool, i_t> sub_warp_meta(rmm::cuda_stream_view stream,
                                   rmm::device_uvector<i_t>& d_warp_offsets,
                                   rmm::device_uvector<i_t>& d_warp_id_offsets,
                                   const std::vector<i_t>& bin_offsets,
                                   i_t w_t_r)
{
  // 1, 2, 4, 8, 16
  auto sub_warp_bin_count = 5;
  std::vector<i_t> warp_counts(sub_warp_bin_count);

  std::vector<i_t> warp_offsets(warp_counts.size() + 1);
  std::vector<i_t> warp_id_offsets(warp_counts.size() + 1);

  for (size_t i = 0; i < warp_id_offsets.size(); ++i) {
    warp_id_offsets[i] = bin_offsets[i + std::log2(w_t_r) + 1];
  }
  warp_id_offsets[0] = bin_offsets[0];

  i_t non_empty_bin_count = 0;
  for (size_t i = 0; i < warp_counts.size(); ++i) {
    warp_counts[i] =
      raft::ceildiv<i_t>((warp_id_offsets[i + 1] - warp_id_offsets[i]) * (1 << i), raft::WarpSize);
    if (warp_counts[i] != 0) { non_empty_bin_count++; }
  }

  warp_offsets[0] = 0;
  for (size_t i = 1; i < warp_offsets.size(); ++i) {
    warp_offsets[i] = warp_offsets[i - 1] + warp_counts[i - 1];
  }
  expand_device_copy(d_warp_offsets, warp_offsets, stream);
  expand_device_copy(d_warp_id_offsets, warp_id_offsets, stream);

  // If there is only 1 bin active, then there is no need to add logic to determine which warps work
  // on which bin
  return std::make_pair(non_empty_bin_count == 1, warp_offsets.back());
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::copy_input_bounds(
  const load_balanced_problem_t<i_t, f_t>& problem)
{
  raft::copy(vars_bnd.data(),
             problem.variable_bounds.data(),
             problem.variable_bounds.size(),
             problem.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::update_host_bounds(
  const load_balanced_problem_t<i_t, f_t>& problem)
{
  raft::copy(host_bounds.data(),
             problem.variable_bounds.data(),
             problem.variable_bounds.size(),
             problem.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::update_host_bounds(const raft::handle_t* handle_ptr)
{
  raft::copy(host_bounds.data(), vars_bnd.data(), vars_bnd.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::update_device_bounds(
  const raft::handle_t* handle_ptr)
{
  raft::copy(vars_bnd.data(), host_bounds.data(), host_bounds.size(), handle_ptr->get_stream());
}

template <typename DryRunFunc, typename CaptureGraphFunc>
bool build_graph(managed_stream_pool& streams,
                 const raft::handle_t* handle_ptr,
                 cudaGraph_t& graph,
                 cudaGraphExec_t& graph_exec,
                 DryRunFunc d_func,
                 CaptureGraphFunc g_func)
{
  bool graph_created = false;
  cudaGraphCreate(&graph, 0);
  cudaEvent_t fork_stream_event;
  cudaEventCreate(&fork_stream_event);

  cudaStreamBeginCapture(handle_ptr->get_stream(), cudaStreamCaptureModeThreadLocal);
  cudaEventRecord(fork_stream_event, handle_ptr->get_stream());

  // dry-run - managed pool tracks how many streams were issued
  d_func();
  streams.wait_issued_on_event(fork_stream_event);
  streams.reset_issued();

  g_func();
  auto activity_done = streams.create_events_on_issued();
  streams.reset_issued();
  for (auto& e : activity_done) {
    cudaStreamWaitEvent(handle_ptr->get_stream(), e);
  }

  cudaStreamEndCapture(handle_ptr->get_stream(), &graph);
  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  if (graph_exec != nullptr) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphInstantiate(&graph_exec, graph);
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  } else {
    cudaGraphInstantiate(&graph_exec, graph);
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  }

  cudaGraphDestroy(graph);
  graph_created = true;

  handle_ptr->get_stream().synchronize();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  return graph_created;
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::setup(
  const load_balanced_problem_t<i_t, f_t>& problem)
{
  pb              = &problem;
  auto handle_ptr = pb->handle_ptr;
  auto stream     = handle_ptr->get_stream();
  stream.synchronize();
  host_bounds.resize(2 * pb->n_variables);
  cnst_slack.resize(2 * pb->n_constraints, stream);
  vars_bnd.resize(2 * pb->n_variables, stream);
  calc_slack_graph_created                = false;
  calc_slack_erase_inf_cnst_graph_created = false;
  upd_bnd_graph_created                   = false;

  copy_input_bounds(problem);

  auto stream_heavy_cnst = stream;
  auto stream_heavy_vars = stream;
  num_blocks_heavy_cnst  = create_heavy_item_block_segments(stream_heavy_cnst,
                                                           heavy_cnst_vertex_ids,
                                                           heavy_cnst_pseudo_block_ids,
                                                           heavy_cnst_block_segments,
                                                           heavy_degree_cutoff,
                                                           problem.cnst_bin_offsets,
                                                           problem.offsets);
  RAFT_CHECK_CUDA(stream_heavy_cnst);

  num_blocks_heavy_vars = create_heavy_item_block_segments(stream_heavy_vars,
                                                           heavy_vars_vertex_ids,
                                                           heavy_vars_pseudo_block_ids,
                                                           heavy_vars_block_segments,
                                                           heavy_degree_cutoff,
                                                           problem.vars_bin_offsets,
                                                           problem.reverse_offsets);
  RAFT_CHECK_CUDA(stream_heavy_vars);

  tmp_act.resize(2 * num_blocks_heavy_cnst, stream_heavy_cnst);
  tmp_bnd.resize(2 * num_blocks_heavy_vars, stream_heavy_vars);

  std::tie(is_cnst_sub_warp_single_bin, cnst_sub_warp_count) =
    sub_warp_meta(stream, warp_cnst_offsets, warp_cnst_id_offsets, pb->cnst_bin_offsets, 4);

  std::tie(is_vars_sub_warp_single_bin, vars_sub_warp_count) =
    sub_warp_meta(stream, warp_vars_offsets, warp_vars_id_offsets, pb->vars_bin_offsets, 4);

  RAFT_CHECK_CUDA(stream);
  streams.sync_test_all_issued();

  if (!calc_slack_erase_inf_cnst_graph_created) {
    create_constraint_slack_graph(true);
    calc_slack_erase_inf_cnst_graph_created = true;
  }

  if (!calc_slack_graph_created) {
    create_constraint_slack_graph(false);
    calc_slack_graph_created = true;
  }

  if (!upd_bnd_graph_created) {
    create_bounds_update_graph();
    upd_bnd_graph_created = true;
  }
}

template <typename i_t, typename f_t>
typename load_balanced_bounds_presolve_t<i_t, f_t>::activity_view_t
load_balanced_bounds_presolve_t<i_t, f_t>::get_activity_view(
  const load_balanced_problem_t<i_t, f_t>& pb)
{
  load_balanced_bounds_presolve_t<i_t, f_t>::activity_view_t v;
  v.cnst_reorg_ids = make_span(pb.cnst_reorg_ids);
  v.coeff          = make_span(pb.coefficients);
  v.vars           = make_span(pb.variables);
  v.offsets        = make_span(pb.offsets);
  v.cnst_bnd       = make_span_2(pb.cnst_bounds_data);
  v.vars_bnd       = make_span_2(vars_bnd);
  v.cnst_slack     = make_span_2(cnst_slack);
  v.nnz            = pb.nnz;
  v.tolerances     = pb.tolerances;
  return v;
}

template <typename i_t, typename f_t>
typename load_balanced_bounds_presolve_t<i_t, f_t>::bounds_update_view_t
load_balanced_bounds_presolve_t<i_t, f_t>::get_bounds_update_view(
  const load_balanced_problem_t<i_t, f_t>& pb)
{
  load_balanced_bounds_presolve_t<i_t, f_t>::bounds_update_view_t v;
  v.vars_reorg_ids = make_span(pb.vars_reorg_ids);
  v.coeff          = make_span(pb.reverse_coefficients);
  v.cnst           = make_span(pb.reverse_constraints);
  v.offsets        = make_span(pb.reverse_offsets);
  v.vars_types     = make_span(pb.vars_types);
  v.vars_bnd       = make_span_2(vars_bnd);
  v.cnst_slack     = make_span_2(cnst_slack);
  v.bounds_changed = bounds_changed.data();
  v.nnz            = pb.nnz;
  v.tolerances     = pb.tolerances;
  return v;
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::calculate_activity_graph(bool erase_inf_cnst,
                                                                         bool dry_run)
{
  using f_t2 = typename type_2<f_t>::type;

  auto activity_view = get_activity_view(*pb);

  calc_activity_heavy_cnst<i_t, f_t, f_t2, 512>(streams,
                                                activity_view,
                                                make_span_2(tmp_act),
                                                heavy_cnst_vertex_ids,
                                                heavy_cnst_pseudo_block_ids,
                                                heavy_cnst_block_segments,
                                                pb->cnst_bin_offsets,
                                                heavy_degree_cutoff,
                                                num_blocks_heavy_cnst,
                                                erase_inf_cnst,
                                                dry_run);
  calc_activity_per_block<i_t, f_t, f_t2>(
    streams, activity_view, pb->cnst_bin_offsets, heavy_degree_cutoff, erase_inf_cnst, dry_run);
  calc_activity_sub_warp<i_t, f_t, f_t2>(streams,
                                         activity_view,
                                         is_cnst_sub_warp_single_bin,
                                         cnst_sub_warp_count,
                                         warp_cnst_offsets,
                                         warp_cnst_id_offsets,
                                         pb->cnst_bin_offsets,
                                         erase_inf_cnst,
                                         dry_run);
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::create_bounds_update_graph()
{
  using f_t2 = typename type_2<f_t>::type;
  cudaGraph_t upd_graph;
  cudaGraphCreate(&upd_graph, 0);
  cudaGraphNode_t bounds_changed_node;
  {
    i_t* bounds_changed_ptr = bounds_changed.data();

    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.srcArray          = NULL;
    memcpyParams.srcPos            = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr            = make_cudaPitchedPtr(bounds_changed_ptr, sizeof(i_t), 1, 1);
    memcpyParams.dstArray          = NULL;
    memcpyParams.dstPos            = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr            = make_cudaPitchedPtr(&h_bounds_changed, sizeof(i_t), 1, 1);
    memcpyParams.extent            = make_cudaExtent(sizeof(i_t), 1, 1);
    memcpyParams.kind              = cudaMemcpyDeviceToHost;
    cudaGraphAddMemcpyNode(&bounds_changed_node, upd_graph, NULL, 0, &memcpyParams);
  }

  auto bounds_update_view = get_bounds_update_view(*pb);

  create_update_bounds_heavy_vars<i_t, f_t, f_t2, 640>(upd_graph,
                                                       bounds_changed_node,
                                                       bounds_update_view,
                                                       make_span_2(tmp_bnd),
                                                       heavy_vars_vertex_ids,
                                                       heavy_vars_pseudo_block_ids,
                                                       heavy_vars_block_segments,
                                                       pb->vars_bin_offsets,
                                                       heavy_degree_cutoff,
                                                       num_blocks_heavy_vars);
  RAFT_CUDA_TRY(cudaGetLastError());
  create_update_bounds_per_block<i_t, f_t, f_t2>(
    upd_graph, bounds_changed_node, bounds_update_view, pb->vars_bin_offsets, heavy_degree_cutoff);
  RAFT_CUDA_TRY(cudaGetLastError());
  create_update_bounds_sub_warp<i_t, f_t, f_t2>(upd_graph,
                                                bounds_changed_node,
                                                bounds_update_view,
                                                is_vars_sub_warp_single_bin,
                                                vars_sub_warp_count,
                                                warp_vars_offsets,
                                                warp_vars_id_offsets,
                                                pb->vars_bin_offsets);
  RAFT_CUDA_TRY(cudaGetLastError());
  cudaGraphInstantiate(&upd_bnd_exec, upd_graph, NULL, NULL, 0);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::create_constraint_slack_graph(bool erase_inf_cnst)
{
  using f_t2 = typename type_2<f_t>::type;
  cudaGraph_t cnst_slack_graph;
  cudaGraphCreate(&cnst_slack_graph, 0);

  cudaGraphNode_t set_bounds_changed_node;
  {
    // TODO : Investigate why memset node is not captured manually
    i_t* bounds_changed_ptr = bounds_changed.data();

    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.srcArray          = NULL;
    memcpyParams.srcPos            = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr            = make_cudaPitchedPtr(&h_bounds_changed, sizeof(i_t), 1, 1);
    memcpyParams.dstArray          = NULL;
    memcpyParams.dstPos            = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr            = make_cudaPitchedPtr(bounds_changed_ptr, sizeof(i_t), 1, 1);
    memcpyParams.extent            = make_cudaExtent(sizeof(i_t), 1, 1);
    memcpyParams.kind              = cudaMemcpyHostToDevice;
    cudaGraphAddMemcpyNode(&set_bounds_changed_node, cnst_slack_graph, NULL, 0, &memcpyParams);
  }

  auto activity_view = get_activity_view(*pb);

  create_activity_heavy_cnst<i_t, f_t, f_t2, 512>(cnst_slack_graph,
                                                  set_bounds_changed_node,
                                                  activity_view,
                                                  make_span_2(tmp_act),
                                                  heavy_cnst_vertex_ids,
                                                  heavy_cnst_pseudo_block_ids,
                                                  heavy_cnst_block_segments,
                                                  pb->cnst_bin_offsets,
                                                  heavy_degree_cutoff,
                                                  num_blocks_heavy_cnst,
                                                  erase_inf_cnst);
  create_activity_per_block<i_t, f_t, f_t2>(cnst_slack_graph,
                                            set_bounds_changed_node,
                                            activity_view,
                                            pb->cnst_bin_offsets,
                                            heavy_degree_cutoff,
                                            erase_inf_cnst);
  create_activity_sub_warp<i_t, f_t, f_t2>(cnst_slack_graph,
                                           set_bounds_changed_node,
                                           activity_view,
                                           is_cnst_sub_warp_single_bin,
                                           cnst_sub_warp_count,
                                           warp_cnst_offsets,
                                           warp_cnst_id_offsets,
                                           pb->cnst_bin_offsets,
                                           erase_inf_cnst);
  if (erase_inf_cnst) {
    cudaGraphInstantiate(&calc_slack_erase_inf_cnst_exec, cnst_slack_graph, NULL, NULL, 0);
  } else {
    cudaGraphInstantiate(&calc_slack_exec, cnst_slack_graph, NULL, NULL, 0);
  }
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::calculate_bounds_update_graph(bool dry_run)
{
  using f_t2 = typename type_2<f_t>::type;

  auto bounds_update_view = get_bounds_update_view(*pb);

  upd_bounds_heavy_vars<i_t, f_t, f_t2, 640>(streams,
                                             bounds_update_view,
                                             make_span_2(tmp_bnd),
                                             heavy_vars_vertex_ids,
                                             heavy_vars_pseudo_block_ids,
                                             heavy_vars_block_segments,
                                             pb->vars_bin_offsets,
                                             heavy_degree_cutoff,
                                             num_blocks_heavy_vars,
                                             dry_run);
  upd_bounds_per_block<i_t, f_t, f_t2>(
    streams, bounds_update_view, pb->vars_bin_offsets, heavy_degree_cutoff, dry_run);
  upd_bounds_sub_warp<i_t, f_t, f_t2>(streams,
                                      bounds_update_view,
                                      is_vars_sub_warp_single_bin,
                                      vars_sub_warp_count,
                                      warp_vars_offsets,
                                      warp_vars_id_offsets,
                                      pb->vars_bin_offsets,
                                      dry_run);
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::calculate_constraint_slack_iter(
  const raft::handle_t* handle_ptr)
{
  // h_bounds_changed is copied to bounds_changed in calc_slack_exec
  h_bounds_changed = 0;
  {
    // writes nans to constraint activities that are infeasible
    //-> less expensive checks for update bounds step
    raft::common::nvtx::range scope("act_cuda_task_graph");
    cudaGraphLaunch(calc_slack_erase_inf_cnst_exec, handle_ptr->get_stream());
  }
  infeas_cnst_slack_set_to_nan = true;
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::calculate_constraint_slack(
  const raft::handle_t* handle_ptr)
{
  // h_bounds_changed is copied to bounds_changed in calc_slack_exec
  h_bounds_changed = 0;
  {
    raft::common::nvtx::range scope("act_cuda_task_graph");
    cudaGraphLaunch(calc_slack_exec, handle_ptr->get_stream());
  }
  infeas_cnst_slack_set_to_nan = false;
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool load_balanced_bounds_presolve_t<i_t, f_t>::update_bounds_from_slack(
  const raft::handle_t* handle_ptr)
{
  // bounds_changed is copied to h_bounds_changed in upd_bnd_exec
  {
    raft::common::nvtx::range scope("upd_cuda_task_graph");
    cudaGraphLaunch(upd_bnd_exec, handle_ptr->get_stream());
  }
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  constexpr i_t zero = 0;
  return (zero < h_bounds_changed);
}

template <typename i_t, typename f_t>
termination_criterion_t load_balanced_bounds_presolve_t<i_t, f_t>::bound_update_loop(
  const raft::handle_t* handle_ptr, timer_t timer)
{
  termination_criterion_t criteria = termination_criterion_t::ITERATION_LIMIT;

  i_t iter;
  for (iter = 0; iter < settings.iteration_limit; ++iter) {
    calculate_constraint_slack_iter(handle_ptr);
    if (!update_bounds_from_slack(handle_ptr)) {
      if (iter == 0) {
        criteria = termination_criterion_t::NO_UPDATE;
      } else {
        criteria = termination_criterion_t::CONVERGENCE;
      }
      break;
    }
    if (timer.check_time_limit()) {
      criteria = termination_criterion_t::TIME_LIMIT;
      CUOPT_LOG_DEBUG("Exiting bounds prop because of time limit at iter %d", iter);
      break;
    }
  }
  handle_ptr->sync_stream();
  infeas_cnst_slack_set_to_nan = true;
  calculate_infeasible_redundant_constraints(handle_ptr);
  solve_iter = iter;

  return criteria;
}

template <typename i_t, typename f_t, typename f_t2>
struct detect_infeas_t {
  __device__ __forceinline__ i_t operator()(thrust::tuple<f_t, f_t, i_t, i_t, f_t2> t) const
  {
    auto cnst_lb    = thrust::get<0>(t);
    auto cnst_ub    = thrust::get<1>(t);
    auto off_beg    = thrust::get<2>(t);
    auto off_end    = thrust::get<3>(t);
    auto cnst_slack = thrust::get<4>(t);
    // zero degree constraints are not infeasible
    if (off_beg == off_end) { return 0; }
    auto eps = get_cstr_tolerance<i_t, f_t>(
      cnst_lb, cnst_ub, tolerances.absolute_tolerance, tolerances.relative_tolerance);
    // The return statement is equivalent to
    //  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
    return (0 > cnst_slack.x + eps) || (eps < cnst_slack.y);
  }

 public:
  detect_infeas_t()                                       = delete;
  detect_infeas_t(const detect_infeas_t<i_t, f_t, f_t2>&) = default;
  detect_infeas_t(const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tols)
    : tolerances(tols)
  {
  }

 private:
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
};

template <typename i_t, typename f_t>
bool load_balanced_bounds_presolve_t<i_t, f_t>::calculate_infeasible_redundant_constraints(
  const raft::handle_t* handle_ptr)
{
  using f_t2         = typename type_2<f_t>::type;
  auto cnst_slack_sp = make_span_2(cnst_slack);
  if (infeas_cnst_slack_set_to_nan) {
    auto detect_iter =
      thrust::make_transform_iterator(cnst_slack_sp.begin(), [] __host__ __device__(f_t2 slack) {
        i_t is_infeas = isnan(slack.x);
        return is_infeas;
      });
    infeas_constraints_count =
      thrust::reduce(handle_ptr->get_thrust_policy(), detect_iter, detect_iter + pb->n_constraints);

    handle_ptr->sync_stream();
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  } else {
    auto detect_iter = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(pb->constraint_lower_bounds.begin(),
                                                   pb->constraint_upper_bounds.begin(),
                                                   pb->offsets.begin(),
                                                   pb->offsets.begin() + 1,
                                                   cnst_slack_sp.begin())),
      detect_infeas_t<i_t, f_t, f_t2>{pb->tolerances});
    infeas_constraints_count =
      thrust::reduce(handle_ptr->get_thrust_policy(), detect_iter, detect_iter + pb->n_constraints);
    handle_ptr->sync_stream();
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  }
  if (infeas_constraints_count > 0) {
    CUOPT_LOG_TRACE("LB Infeasible constraint count %d", infeas_constraints_count);
  }
  return (infeas_constraints_count == 0);
}

template <typename i_t, typename f_t>
termination_criterion_t load_balanced_bounds_presolve_t<i_t, f_t>::solve(f_t var_lb,
                                                                         f_t var_ub,
                                                                         i_t var_idx)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb->handle_ptr;
  copy_input_bounds(*pb);
  vars_bnd.set_element_async(2 * var_idx, var_lb, handle_ptr->get_stream());
  vars_bnd.set_element_async(2 * var_idx + 1, var_ub, handle_ptr->get_stream());
  return bound_update_loop(handle_ptr, timer);
}

template <typename i_t, typename f_t>
termination_criterion_t load_balanced_bounds_presolve_t<i_t, f_t>::solve(
  raft::device_span<f_t> input_bounds)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb->handle_ptr;
  if (input_bounds.size() != 0) {
    raft::copy(vars_bnd.data(), input_bounds.data(), input_bounds.size(), handle_ptr->get_stream());
  } else {
    copy_input_bounds(*pb);
  }
  return bound_update_loop(handle_ptr, timer);
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::set_bounds(
  const std::vector<thrust::pair<i_t, f_t>>& var_probe_vals, const raft::handle_t* handle_ptr)
{
  auto d_var_probe_vals = device_copy(var_probe_vals, handle_ptr->get_stream());
  auto variable_bounds  = make_span_2(vars_bnd);

  thrust::for_each(handle_ptr->get_thrust_policy(),
                   d_var_probe_vals.begin(),
                   d_var_probe_vals.end(),
                   [variable_bounds] __device__(auto pair) {
                     variable_bounds[pair.first] = f_t2{pair.second, pair.second};
                   });
}

template <typename i_t, typename f_t>
termination_criterion_t load_balanced_bounds_presolve_t<i_t, f_t>::solve(
  const std::vector<thrust::pair<i_t, f_t>>& var_probe_val_pairs, bool use_host_bounds)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb->handle_ptr;
  if (use_host_bounds) {
    update_device_bounds(handle_ptr);
  } else {
    copy_input_bounds(*pb);
  }
  set_bounds(var_probe_val_pairs, handle_ptr);

  return bound_update_loop(handle_ptr, timer);
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::set_updated_bounds(
  load_balanced_problem_t<i_t, f_t>* problem)
{
  auto& handle_ptr = problem->handle_ptr;
  raft::copy(
    problem->variable_bounds.data(), vars_bnd.data(), vars_bnd.size(), handle_ptr->get_stream());
}

#if MIP_INSTANTIATE_FLOAT
template class load_balanced_bounds_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class load_balanced_bounds_presolve_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
