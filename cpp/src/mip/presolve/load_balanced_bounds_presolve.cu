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
#include <nvtx3/nvtx3.hpp>
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
    var_bounds_changed(0, problem_.handle_ptr->get_stream()),
    changed_constraints(0, problem_.handle_ptr->get_stream()),
    changed_variables(0, problem_.handle_ptr->get_stream()),
    next_changed_constraints(0, problem_.handle_ptr->get_stream()),
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

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::setup(
  const load_balanced_problem_t<i_t, f_t>& problem)
{
  pb              = &problem;
  auto handle_ptr = pb->handle_ptr;
  auto stream     = handle_ptr->get_stream();
  stream.synchronize();
  host_bounds.resize(2 * pb->n_variables);
  var_bounds_changed.resize(pb->n_variables, stream);
  changed_constraints.resize(pb->n_constraints, stream);
  changed_variables.resize(pb->n_variables, stream);
  next_changed_constraints.resize(pb->n_constraints, stream);
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

  num_blocks_heavy_vars = create_heavy_item_block_segments(stream_heavy_vars,
                                                           heavy_vars_vertex_ids,
                                                           heavy_vars_pseudo_block_ids,
                                                           heavy_vars_block_segments,
                                                           heavy_degree_cutoff,
                                                           problem.vars_bin_offsets,
                                                           problem.reverse_offsets);

  tmp_act.resize(2 * num_blocks_heavy_cnst, stream_heavy_cnst);
  tmp_bnd.resize(2 * num_blocks_heavy_vars, stream_heavy_vars);

  std::tie(is_cnst_sub_warp_single_bin, cnst_sub_warp_count) = sub_warp_meta(
    streams.get_stream(), warp_cnst_offsets, warp_cnst_id_offsets, pb->cnst_bin_offsets, 4);

  std::tie(is_vars_sub_warp_single_bin, vars_sub_warp_count) = sub_warp_meta(
    streams.get_stream(), warp_vars_offsets, warp_vars_id_offsets, pb->vars_bin_offsets, 4);

  stream.synchronize();
  streams.sync_all_issued();

  if (!calc_slack_erase_inf_cnst_graph_created) {
    bool erase_inf_cnst                     = true;
    calc_slack_erase_inf_cnst_graph_created = build_graph(
      streams,
      handle_ptr,
      calc_slack_erase_inf_cnst_graph,
      calc_slack_erase_inf_cnst_exec,
      [erase_inf_cnst, this]() { this->calculate_activity_graph(erase_inf_cnst, true); },
      [erase_inf_cnst, this]() { this->calculate_activity_graph(erase_inf_cnst); });
  }

  if (!calc_slack_graph_created) {
    bool erase_inf_cnst      = false;
    calc_slack_graph_created = build_graph(
      streams,
      handle_ptr,
      calc_slack_graph,
      calc_slack_exec,
      [erase_inf_cnst, this]() { this->calculate_activity_graph(erase_inf_cnst, true); },
      [erase_inf_cnst, this]() { this->calculate_activity_graph(erase_inf_cnst); });
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
typename load_balanced_bounds_presolve_t<i_t, f_t>::activity_view_t
load_balanced_bounds_presolve_t<i_t, f_t>::get_activity_view(
  const load_balanced_problem_t<i_t, f_t>& pb)
{
  load_balanced_bounds_presolve_t<i_t, f_t>::activity_view_t v;
  v.cnst_reorg_ids           = make_span(pb.cnst_reorg_ids);
  v.coeff                    = make_span(pb.coefficients);
  v.vars                     = make_span(pb.variables);
  v.offsets                  = make_span(pb.offsets);
  v.cnst_bnd                 = make_span_2(pb.cnst_bounds_data);
  v.vars_bnd                 = make_span_2(vars_bnd);
  v.cnst_slack               = make_span_2(cnst_slack);
  v.var_bounds_changed       = make_span(var_bounds_changed);
  v.changed_constraints      = make_span(changed_constraints);
  v.changed_variables        = make_span(changed_variables);
  v.next_changed_constraints = make_span(next_changed_constraints);
  v.nnz                      = pb.nnz;
  v.tolerances               = pb.tolerances;
  return v;
}

template <typename i_t, typename f_t>
typename load_balanced_bounds_presolve_t<i_t, f_t>::bounds_update_view_t
load_balanced_bounds_presolve_t<i_t, f_t>::get_bounds_update_view(
  const load_balanced_problem_t<i_t, f_t>& pb)
{
  load_balanced_bounds_presolve_t<i_t, f_t>::bounds_update_view_t v;
  v.vars_reorg_ids           = make_span(pb.vars_reorg_ids);
  v.coeff                    = make_span(pb.reverse_coefficients);
  v.cnst                     = make_span(pb.reverse_constraints);
  v.offsets                  = make_span(pb.reverse_offsets);
  v.vars_types               = make_span(pb.vars_types);
  v.vars_bnd                 = make_span_2(vars_bnd);
  v.cnst_slack               = make_span_2(cnst_slack);
  v.bounds_changed           = bounds_changed.data();
  v.var_bounds_changed       = make_span(var_bounds_changed);
  v.changed_constraints      = make_span(changed_constraints);
  v.changed_variables        = make_span(changed_variables);
  v.next_changed_constraints = make_span(next_changed_constraints);
  v.nnz                      = pb.nnz;
  v.tolerances               = pb.tolerances;
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
  {
    // writes nans to constraint activities that are infeasible
    //-> less expensive checks for update bounds step
    raft::common::nvtx::range scope("act_cuda_task_graph");
    cudaGraphLaunch(calc_slack_erase_inf_cnst_exec, handle_ptr->get_stream());
    handle_ptr->sync_stream();
  }
  infeas_cnst_slack_set_to_nan = true;
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::calculate_constraint_slack(
  const raft::handle_t* handle_ptr)
{
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
  i_t h_bounds_changed;
  bounds_changed.set_value_to_zero_async(handle_ptr->get_stream());

  {
    raft::common::nvtx::range scope("upd_cuda_task_graph");
    cudaGraphLaunch(upd_bnd_exec, handle_ptr->get_stream());
    h_bounds_changed = bounds_changed.value(handle_ptr->get_stream());
  }
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  constexpr i_t zero = 0;
  return (zero < h_bounds_changed);
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::init_changed_constraints(
  const raft::handle_t* handle_ptr)
{
  thrust::fill(
    handle_ptr->get_thrust_policy(), var_bounds_changed.begin(), var_bounds_changed.end(), 0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 1);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_constraints.begin(), changed_constraints.end(), 1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               next_changed_constraints.begin(),
               next_changed_constraints.end(),
               0);
}

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::prepare_for_next_iteration(
  const raft::handle_t* handle_ptr)
{
  std::swap(changed_constraints, next_changed_constraints);
  handle_ptr->sync_stream();
  thrust::fill(handle_ptr->get_thrust_policy(),
               next_changed_constraints.begin(),
               next_changed_constraints.end(),
               0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), var_bounds_changed.begin(), var_bounds_changed.end(), 0);
}

template <typename i_t, typename f_t>
termination_criterion_t load_balanced_bounds_presolve_t<i_t, f_t>::bound_update_loop(
  const raft::handle_t* handle_ptr, timer_t timer)
{
  termination_criterion_t criteria = termination_criterion_t::ITERATION_LIMIT;

  i_t iter;
  init_changed_constraints(handle_ptr);
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
    prepare_for_next_iteration(handle_ptr);
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
struct detect_infeas_t : public thrust::unary_function<thrust::tuple<f_t, f_t, f_t2>, i_t> {
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

template <typename i_t, typename f_t>
void load_balanced_bounds_presolve_t<i_t, f_t>::set_updated_bounds(rmm::device_uvector<f_t>& lb,
                                                                   rmm::device_uvector<f_t>& ub)
{
  auto& handle_ptr = pb->handle_ptr;
  auto out         = thrust::make_zip_iterator(thrust::make_tuple(lb.begin(), ub.begin()));
  auto bnd_span    = make_span_2(vars_bnd);
  thrust::transform(handle_ptr->get_thrust_policy(),
                    bnd_span.begin(),
                    bnd_span.end(),
                    out,
                    [] __device__(auto bnd) { return thrust::make_tuple(bnd.x, bnd.y); });
}

#if MIP_INSTANTIATE_FLOAT
template class load_balanced_bounds_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class load_balanced_bounds_presolve_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
