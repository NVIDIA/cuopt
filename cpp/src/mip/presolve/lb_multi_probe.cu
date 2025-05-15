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

#include <mip/mip_constants.hpp>
#include "lb_multi_probe.cuh"
#include "lb_multi_probe_helpers.cuh"
#include "load_balanced_bounds_presolve_helpers.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_multi_probe_t<i_t, f_t>::lb_multi_probe_t(const load_balanced_problem_t<i_t, f_t>& problem_,
                                             mip_solver_context_t<i_t, f_t>& context_,
                                             settings_t in_settings,
                                             i_t max_stream_count_)
  : context(context_),
    pb(&problem_),
    streams(max_stream_count_),
    upd_0(problem_.handle_ptr),
    upd_1(problem_.handle_ptr),
    heavy_cnst_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_vars_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    num_blocks_heavy_cnst(0),
    num_blocks_heavy_vars(0),
    warp_cnst_offsets(0, problem_.handle_ptr->get_stream()),
    warp_cnst_id_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_id_offsets(0, problem_.handle_ptr->get_stream()),
    calc_slack_exec(nullptr),
    calc_slack_erase_inf_cnst_exec(nullptr),
    upd_bnd_exec(nullptr),
    calc_slack_erase_inf_cnst_graph_created(false),
    calc_slack_graph_created(false),
    upd_bnd_graph_created(false),
    settings(in_settings)
{
  setup(problem_);
}

template <typename i_t, typename f_t>
lb_multi_probe_t<i_t, f_t>::~lb_multi_probe_t()
{
  if (calc_slack_erase_inf_cnst_graph_created) {
    cudaGraphExecDestroy(calc_slack_erase_inf_cnst_exec);
  }
  if (calc_slack_graph_created) { cudaGraphExecDestroy(calc_slack_exec); }
  if (upd_bnd_graph_created) { cudaGraphExecDestroy(upd_bnd_exec); }
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::setup(const load_balanced_problem_t<i_t, f_t>& problem)
{
  auto handle_ptr = pb->handle_ptr;
  auto stream     = handle_ptr->get_stream();

  num_blocks_heavy_cnst = create_heavy_item_block_segments(stream,
                                                           heavy_cnst_vertex_ids,
                                                           heavy_cnst_pseudo_block_ids,
                                                           heavy_cnst_block_segments,
                                                           heavy_degree_cutoff,
                                                           problem.cnst_bin_offsets,
                                                           problem.offsets);

  num_blocks_heavy_vars = create_heavy_item_block_segments(stream,
                                                           heavy_vars_vertex_ids,
                                                           heavy_vars_pseudo_block_ids,
                                                           heavy_vars_block_segments,
                                                           heavy_degree_cutoff,
                                                           problem.vars_bin_offsets,
                                                           problem.reverse_offsets);
  upd_0.resize(
    handle_ptr, pb->n_constraints, pb->n_variables, num_blocks_heavy_cnst, num_blocks_heavy_vars);
  upd_1.resize(
    handle_ptr, pb->n_constraints, pb->n_variables, num_blocks_heavy_cnst, num_blocks_heavy_vars);

  std::tie(is_cnst_sub_warp_single_bin, cnst_sub_warp_count) =
    sub_warp_meta(stream, warp_cnst_offsets, warp_cnst_id_offsets, pb->cnst_bin_offsets, 4);

  std::tie(is_vars_sub_warp_single_bin, vars_sub_warp_count) =
    sub_warp_meta(stream, warp_vars_offsets, warp_vars_id_offsets, pb->vars_bin_offsets, 4);

  stream.synchronize();
  streams.sync_all_issued();

  if (!calc_slack_erase_inf_cnst_graph_created) {
    bool erase_inf_cnst                     = true;
    calc_slack_erase_inf_cnst_graph_created = build_graph(
      streams,
      handle_ptr,
      calc_slack_erase_inf_cnst_graph,
      calc_slack_erase_inf_cnst_exec,
      [erase_inf_cnst, this]() { this->calculate_cnst_slack_graph(erase_inf_cnst, true); },
      [erase_inf_cnst, this]() { this->calculate_cnst_slack_graph(erase_inf_cnst); });
  }

  if (!calc_slack_graph_created) {
    bool erase_inf_cnst      = false;
    calc_slack_graph_created = build_graph(
      streams,
      handle_ptr,
      calc_slack_graph,
      calc_slack_exec,
      [erase_inf_cnst, this]() { this->calculate_cnst_slack_graph(erase_inf_cnst, true); },
      [erase_inf_cnst, this]() { this->calculate_cnst_slack_graph(erase_inf_cnst); });
  }

  if (!upd_bnd_graph_created) {
    upd_bnd_graph_created = build_graph(
      streams,
      handle_ptr,
      upd_bnd_graph,
      upd_bnd_exec,
      [this]() { this->calculate_bounds_update_graph(true); },
      [this]() { this->calculate_bounds_update_graph(); });
  }
}

template <typename i_t, typename f_t>
typename lb_multi_probe_t<i_t, f_t>::activity_view_t lb_multi_probe_t<i_t, f_t>::get_activity_view(
  const load_balanced_problem_t<i_t, f_t>& pb)
{
  lb_multi_probe_t<i_t, f_t>::activity_view_t v;
  v.cnst_reorg_ids = make_span(pb.cnst_reorg_ids);
  v.coeff          = make_span(pb.coefficients);
  v.vars           = make_span(pb.variables);
  v.offsets        = make_span(pb.offsets);
  v.cnst_bnd       = make_span_2(pb.cnst_bounds_data);
  v.nnz            = pb.nnz;
  v.tolerances     = pb.tolerances;
  return v;
}

template <typename i_t, typename f_t>
typename lb_multi_probe_t<i_t, f_t>::bounds_update_view_t
lb_multi_probe_t<i_t, f_t>::get_bounds_update_view(const load_balanced_problem_t<i_t, f_t>& pb)
{
  lb_multi_probe_t<i_t, f_t>::bounds_update_view_t v;
  v.vars_reorg_ids = make_span(pb.vars_reorg_ids);
  v.coeff          = make_span(pb.reverse_coefficients);
  v.cnst           = make_span(pb.reverse_constraints);
  v.offsets        = make_span(pb.reverse_offsets);
  v.vars_types     = make_span(pb.vars_types);
  v.nnz            = pb.nnz;
  v.tolerances     = pb.tolerances;
  return v;
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::calculate_cnst_slack(const raft::handle_t* handle_ptr)
{
  cudaGraphLaunch(calc_slack_erase_inf_cnst_exec, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::calculate_cnst_slack_graph(bool erase_inf_cnst, bool dry_run)
{
  using f_t2 = typename type_2<f_t>::type;

  auto activity_view = get_activity_view(*pb);
  auto upd_0_v       = upd_0.view();
  auto upd_1_v       = upd_1.view();

  calc_activity_heavy_cnst<i_t, f_t, f_t2, 512>(streams,
                                                activity_view,
                                                upd_0_v,
                                                upd_1_v,
                                                heavy_cnst_vertex_ids,
                                                heavy_cnst_pseudo_block_ids,
                                                heavy_cnst_block_segments,
                                                pb->cnst_bin_offsets,
                                                heavy_degree_cutoff,
                                                num_blocks_heavy_cnst,
                                                erase_inf_cnst,
                                                dry_run);
  calc_activity_per_block<i_t, f_t, f_t2>(streams,
                                          activity_view,
                                          upd_0_v,
                                          upd_1_v,
                                          pb->cnst_bin_offsets,
                                          heavy_degree_cutoff,
                                          erase_inf_cnst,
                                          dry_run);
  calc_activity_sub_warp<i_t, f_t, f_t2>(streams,
                                         activity_view,
                                         upd_0_v,
                                         upd_1_v,
                                         is_cnst_sub_warp_single_bin,
                                         cnst_sub_warp_count,
                                         warp_cnst_offsets,
                                         warp_cnst_id_offsets,
                                         pb->cnst_bin_offsets,
                                         erase_inf_cnst,
                                         dry_run);
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::calculate_bounds_update_graph(bool dry_run)
{
  using f_t2 = typename type_2<f_t>::type;

  auto bounds_update_view = get_bounds_update_view(*pb);
  auto upd_0_v            = upd_0.view();
  auto upd_1_v            = upd_1.view();

  upd_bounds_heavy_vars<i_t, f_t, f_t2, 640>(streams,
                                             bounds_update_view,
                                             upd_0_v,
                                             upd_1_v,
                                             heavy_vars_vertex_ids,
                                             heavy_vars_pseudo_block_ids,
                                             heavy_vars_block_segments,
                                             pb->vars_bin_offsets,
                                             heavy_degree_cutoff,
                                             num_blocks_heavy_vars,
                                             dry_run);
  upd_bounds_per_block<i_t, f_t, f_t2>(streams,
                                       bounds_update_view,
                                       upd_0_v,
                                       upd_1_v,
                                       pb->vars_bin_offsets,
                                       heavy_degree_cutoff,
                                       dry_run);
  upd_bounds_sub_warp<i_t, f_t, f_t2>(streams,
                                      bounds_update_view,
                                      upd_0_v,
                                      upd_1_v,
                                      is_vars_sub_warp_single_bin,
                                      vars_sub_warp_count,
                                      warp_vars_offsets,
                                      warp_vars_id_offsets,
                                      pb->vars_bin_offsets,
                                      dry_run);
}

template <typename i_t, typename f_t>
bool lb_multi_probe_t<i_t, f_t>::calculate_bounds_update(const raft::handle_t* handle_ptr)
{
  constexpr i_t zero = 0;
  upd_0.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
  upd_1.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
  cudaGraphLaunch(upd_bnd_exec, handle_ptr->get_stream());
  i_t h_bounds_changed_0 = upd_0.bounds_changed.value(handle_ptr->get_stream());
  i_t h_bounds_changed_1 = upd_1.bounds_changed.value(handle_ptr->get_stream());
  skip_0                 = (h_bounds_changed_0 == zero);
  skip_1                 = (h_bounds_changed_1 == zero);
  return !(skip_0 && skip_1);
}

template <typename i_t, typename f_t>
termination_criterion_t lb_multi_probe_t<i_t, f_t>::bound_update_loop(
  const raft::handle_t* handle_ptr, timer_t timer)
{
  termination_criterion_t criteria = termination_criterion_t::ITERATION_LIMIT;
  i_t iter_0                       = 0;
  i_t iter_1                       = 0;
  if (init_changed_constraints) {
    // all changed constraints are 1, next are zero
    upd_0.init_changed_constraints(handle_ptr);
    upd_1.init_changed_constraints(handle_ptr);
  } else {
    // reset for the next calls on the same object
    init_changed_constraints = true;
  }
  // settings.iteration_limit = 1;
  for (i_t iter = 0; iter < settings.iteration_limit; ++iter) {
    if (timer.check_time_limit()) {
      criteria = termination_criterion_t::TIME_LIMIT;
      break;
    }
    // calculate activity for both probes
    calculate_cnst_slack(handle_ptr);
    if (!calculate_bounds_update(handle_ptr)) {
      if (iter == 0) {
        criteria = termination_criterion_t::NO_UPDATE;
      } else {
        criteria = termination_criterion_t::CONVERGENCE;
      }
      break;
    }
    // next_changed are updated, fill current changed with zero and swap
    // swap next and current changed constraints
    if (!skip_0) { upd_0.prepare_for_next_iteration(handle_ptr); }
    if (!skip_1) { upd_1.prepare_for_next_iteration(handle_ptr); }
    iter_0 += !skip_0;
    iter_1 += !skip_1;
  }

  return criteria;
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::copy_problem_into_probing_buffers(const raft::handle_t* handle_ptr)
{
  cuopt_assert(upd_0.vars_bnd.size() == pb->variable_bounds.size(),
               "size of variable bounds mismatch");
  raft::copy(upd_0.vars_bnd.data(),
             pb->variable_bounds.data(),
             upd_0.vars_bnd.size(),
             handle_ptr->get_stream());

  cuopt_assert(upd_1.vars_bnd.size() == pb->variable_bounds.size(),
               "size of variable bounds mismatch");
  raft::copy(upd_1.vars_bnd.data(),
             pb->variable_bounds.data(),
             upd_1.vars_bnd.size(),
             handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_interval_bounds(
  const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
  const raft::handle_t* handle_ptr)
{
  // TODO : upd.vars_bnd_changed
  using f_t2                              = typename type_2<f_t>::type;
  const i_t& probe_var                    = std::get<0>(var_interval_vals);
  const std::pair<f_t, f_t>& probe_vals_0 = std::get<1>(var_interval_vals);
  const std::pair<f_t, f_t>& probe_vals_1 = std::get<2>(var_interval_vals);
  run_device_lambda(handle_ptr->get_stream(),
                    [probe_var = probe_var,
                     lb_0      = probe_vals_0.first,
                     ub_0      = probe_vals_0.second,
                     lb_1      = probe_vals_1.first,
                     ub_1      = probe_vals_1.second,
                     upd_0_v   = upd_0.view(),
                     upd_1_v   = upd_1.view()] __device__() {
                      upd_0_v.vars_bnd[probe_var] = f_t2{lb_0, ub_0};
                      upd_1_v.vars_bnd[probe_var] = f_t2{lb_1, ub_1};
                    });
  // init changed constraints
  auto orig_pb         = pb->pb;
  i_t var_offset_begin = orig_pb->reverse_offsets.element(probe_var, handle_ptr->get_stream());
  i_t var_offset_end   = orig_pb->reverse_offsets.element(probe_var + 1, handle_ptr->get_stream());
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_0.changed_constraints.begin(),
               upd_0.changed_constraints.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_1.changed_constraints.begin(),
               upd_1.changed_constraints.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_0.next_changed_constraints.begin(),
               upd_0.next_changed_constraints.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_1.next_changed_constraints.begin(),
               upd_1.next_changed_constraints.end(),
               0);
  // set changed constraints from the vars
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   orig_pb->reverse_constraints.begin() + var_offset_begin,
                   orig_pb->reverse_constraints.begin() + var_offset_end,
                   [upd_0_v = upd_0.view(), upd_1_v = upd_1.view()] __device__(auto i) {
                     upd_0_v.changed_constraints[i] = 1;
                     upd_1_v.changed_constraints[i] = 1;
                   });
  init_changed_constraints = false;
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t lb_multi_probe_t<i_t, f_t>::solve_for_interval(
  const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
  const raft::handle_t* handle_ptr)
{
  timer_t timer(settings.time_limit);

  copy_problem_into_probing_buffers(handle_ptr);
  set_interval_bounds(var_interval_vals, handle_ptr);

  return bound_update_loop(handle_ptr, timer);
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_updated_bounds(const raft::handle_t* handle_ptr,
                                                    raft::device_span<f_t> output_bounds,
                                                    i_t select_update)
{
  auto& bnds = select_update ? upd_1.vars_bnd : upd_0.vars_bnd;

  cuopt_assert(bnds.size() == output_bounds.size(), "size of variable bound mismatch");
  raft::copy(output_bounds.data(), bnds.data(), bnds.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_updated_bounds(const raft::handle_t* handle_ptr,
                                                    raft::device_span<f_t> output_lb,
                                                    raft::device_span<f_t> output_ub,
                                                    i_t select_update)
{
  auto& bnds = select_update ? upd_1.vars_bnd : upd_0.vars_bnd;

  auto bnd_span = make_span_2(bnds);
  cuopt_assert(bnd_span.size() == output_lb.size(), "size of variable lower bound mismatch");
  cuopt_assert(bnd_span.size() == output_ub.size(), "size of variable upper bound mismatch");
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<i_t>(0),
                   thrust::make_counting_iterator<i_t>(bnd_span.size()),
                   [output_lb, output_ub, bnd_span] __device__(auto idx) {
                     auto bnd       = bnd_span[idx];
                     output_lb[idx] = bnd.x;
                     output_ub[idx] = bnd.y;
                   });
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_updated_bounds(load_balanced_problem_t<i_t, f_t>& problem,
                                                    i_t select_update,
                                                    const raft::handle_t* handle_ptr)
{
  set_updated_bounds(handle_ptr, make_span(problem.variable_bounds), select_update);
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_bounds(
  const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
  const raft::handle_t* handle_ptr)
{
  const std::vector<i_t>& probe_vars   = std::get<0>(var_probe_vals);
  const std::vector<f_t>& probe_vals_0 = std::get<1>(var_probe_vals);
  const std::vector<f_t>& probe_vals_1 = std::get<2>(var_probe_vals);
  auto d_vars                          = device_copy(probe_vars, handle_ptr->get_stream());
  auto d_vals_0                        = device_copy(probe_vals_0, handle_ptr->get_stream());
  auto d_vals_1                        = device_copy(probe_vals_1, handle_ptr->get_stream());

  auto upd_0_v = upd_0.view();
  auto upd_1_v = upd_1.view();
  auto z_iter  = thrust::make_zip_iterator(
    thrust::make_tuple(d_vars.begin(), d_vals_0.begin(), d_vals_1.begin()));
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    z_iter,
    z_iter + d_vars.size(),
    [upd_0_v, upd_1_v] __device__(auto t) {
      using f_t2                          = typename type_2<f_t>::type;
      upd_0_v.vars_bnd[thrust::get<0>(t)] = f_t2{thrust::get<1>(t), thrust::get<1>(t)};
      upd_1_v.vars_bnd[thrust::get<0>(t)] = f_t2{thrust::get<2>(t), thrust::get<2>(t)};
      // upd_0_v.ub[thrust::get<0>(t)] = thrust::get<1>(t);
      // upd_1_v.lb[thrust::get<0>(t)] = thrust::get<2>(t);
      // upd_1_v.ub[thrust::get<0>(t)] = thrust::get<2>(t);
    });
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t lb_multi_probe_t<i_t, f_t>::solve(
  const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb->handle_ptr;

  copy_problem_into_probing_buffers(handle_ptr);
  set_bounds(var_probe_vals, handle_ptr);

  return bound_update_loop(handle_ptr, timer);
}

#if MIP_INSTANTIATE_FLOAT
template class lb_multi_probe_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_multi_probe_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
