/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/error.hpp>

#include "feasibility_jump.cuh"
#include "feasibility_jump_kernels.cuh"

#include <mip/mip_constants.hpp>
#include <mip/utils.cuh>
#include <utilities/seed_generator.cuh>
#include <utilities/timer.hpp>

#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/random/rng.cuh>

#include <thrust/copy.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <cuda/std/functional>

#include <cooperative_groups.h>

#define FJ_LOG_PREFIX "FJ: "

namespace cuopt::linear_programming::detail {

#if FJ_SINGLE_STEP
static constexpr int iterations_per_graph = 1;
#else
static constexpr int iterations_per_graph = 50;
#endif

__constant__ fj_settings_t device_settings;

template <typename i_t, typename f_t>
fj_t<i_t, f_t>::fj_t(mip_solver_context_t<i_t, f_t>& context_, fj_settings_t in_settings)
  : context(context_),
    pb_ptr(context.problem_ptr),
    handle_ptr(const_cast<raft::handle_t*>(pb_ptr->handle_ptr)),
    settings(in_settings),
    cstr_weights(pb_ptr->n_constraints, pb_ptr->handle_ptr->get_stream()),
    cstr_right_weights(pb_ptr->n_constraints, pb_ptr->handle_ptr->get_stream()),
    cstr_left_weights(pb_ptr->n_constraints, pb_ptr->handle_ptr->get_stream()),
    weight_update_increment(1.0),
    objective_weight(0.0, pb_ptr->handle_ptr->get_stream()),
    max_cstr_weight(0, pb_ptr->handle_ptr->get_stream()),
    climber_views(0, pb_ptr->handle_ptr->get_stream()),
    objective_vars(0, pb_ptr->handle_ptr->get_stream()),
    constraint_lower_bounds_csr(pb_ptr->coefficients.size(), pb_ptr->handle_ptr->get_stream()),
    constraint_upper_bounds_csr(pb_ptr->coefficients.size(), pb_ptr->handle_ptr->get_stream()),
    cstr_coeff_reciprocal(pb_ptr->coefficients.size(), pb_ptr->handle_ptr->get_stream()),
    work_id_to_bin_var_idx(pb_ptr->coefficients.size(), pb_ptr->handle_ptr->get_stream()),
    work_id_to_nonbin_var_idx(pb_ptr->coefficients.size(), pb_ptr->handle_ptr->get_stream()),
    row_size_bin_prefix_sum(pb_ptr->binary_indices.size(), pb_ptr->handle_ptr->get_stream()),
    row_size_nonbin_prefix_sum(pb_ptr->nonbinary_indices.size(), pb_ptr->handle_ptr->get_stream()),
    work_ids_for_related_vars(pb_ptr->n_variables, pb_ptr->handle_ptr->get_stream())
{
  setval_launch_dims = get_launch_dims_max_occupancy(
    (void*)update_assignment_kernel<i_t, f_t>, TPB_setval, pb_ptr->handle_ptr);
  resetmoves_launch_dims = get_launch_dims_max_occupancy(
    (void*)compute_mtm_moves_kernel<i_t, f_t, FJ_MTM_VIOLATED>, TPB_resetmoves, pb_ptr->handle_ptr);
  resetmoves_bin_launch_dims =
    get_launch_dims_max_occupancy((void*)compute_mtm_moves_kernel<i_t, f_t, FJ_MTM_VIOLATED, true>,
                                  TPB_resetmoves,
                                  pb_ptr->handle_ptr);
  update_weights_launch_dims = get_launch_dims_max_occupancy(
    (void*)handle_local_minimum_kernel<i_t, f_t>, TPB_localmin, pb_ptr->handle_ptr);
  lift_move_launch_dims = get_launch_dims_max_occupancy(
    (void*)update_lift_moves_kernel<i_t, f_t>, TPB_liftmoves, pb_ptr->handle_ptr);
  load_balancing_workid_map_launch_dims = get_launch_dims_max_occupancy(
    (void*)load_balancing_compute_workid_mappings<i_t, f_t>, TPB_loadbalance, pb_ptr->handle_ptr);
  load_balancing_binary_launch_dims = get_launch_dims_max_occupancy(
    (void*)load_balancing_compute_scores_binary<i_t, f_t>, TPB_loadbalance, pb_ptr->handle_ptr);
  load_balancing_mtm_compute_candidates_launch_dims = get_launch_dims_max_occupancy(
    (void*)load_balancing_mtm_compute_candidates<i_t, f_t>, TPB_loadbalance, pb_ptr->handle_ptr);
  load_balancing_mtm_compute_scores_launch_dims = get_launch_dims_max_occupancy(
    (void*)load_balancing_mtm_compute_scores<i_t, f_t>, TPB_loadbalance, pb_ptr->handle_ptr);
  load_balancing_prepare_launch_dims = get_launch_dims_max_occupancy(
    (void*)load_balancing_prepare_iteration<i_t, f_t>, TPB_loadbalance, pb_ptr->handle_ptr);
  reset_weights(pb_ptr->handle_ptr->get_stream());

  // ensure the problem and its transpose are in a valid state (assert checks)
  pb_ptr->check_problem_representation(true);

  device_init(pb_ptr->handle_ptr->get_stream());
  // allocate here to avoid cuda graph updates
  // +1 comes from the main climber
  alloc_max_climbers(1);
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::reset_cuda_graph()
{
  if (graph_created) cudaGraphExecDestroy(graph_instance);
  graph_created = false;
}

template <typename i_t, typename f_t>
fj_t<i_t, f_t>::~fj_t()
{
  reset_cuda_graph();
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::reset_weights(const rmm::cuda_stream_view& climber_stream, f_t weight)
{
  // unless reset explicitly, the values are kept across runs and across climbers
  max_cstr_weight.set_value_async(weight, climber_stream);
  objective_weight.set_value_to_zero_async(climber_stream);
  thrust::uninitialized_fill(
    rmm::exec_policy(climber_stream), cstr_weights.begin(), cstr_weights.end(), weight);
  thrust::uninitialized_fill(
    rmm::exec_policy(climber_stream), cstr_left_weights.begin(), cstr_left_weights.end(), weight);
  thrust::uninitialized_fill(
    rmm::exec_policy(climber_stream), cstr_right_weights.begin(), cstr_right_weights.end(), weight);
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::randomize_weights(const raft::handle_t* handle_ptr)
{
  std::mt19937 rng(cuopt::seed_generator::get_seed());
  constexpr f_t min_weight = 10.;
  constexpr f_t max_weight = 30.;
  // generate a range of weights between 10. and 30.
  auto h_cstr_vec =
    get_random_uniform_vector<i_t, f_t>(cstr_weights.size(), rng, min_weight, max_weight);
  f_t h_max_weight = *std::max_element(h_cstr_vec.begin(), h_cstr_vec.end());
  max_cstr_weight.set_value_async(h_max_weight, handle_ptr->get_stream());
  raft::copy(cstr_weights.data(), h_cstr_vec.data(), h_cstr_vec.size(), handle_ptr->get_stream());
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
fj_t<i_t, f_t>::climber_data_t::view_t fj_t<i_t, f_t>::climber_data_t::view()
{
  view_t v;
  v.pb = fj.pb_ptr->view();

  v.violated_constraints        = violated_constraints.view();
  v.candidate_variables         = candidate_variables.view();
  v.iteration_related_variables = iteration_related_variables.view();
  v.constraints_changed         = make_span(constraints_changed);
  v.best_assignment             = make_span(best_assignment);
  v.farthest_l1_sol             = make_span(farthest_l1_sol);
  v.incumbent_assignment        = make_span(incumbent_assignment);
  v.cstr_weights                = make_span(fj.cstr_weights);
  v.cstr_left_weights           = make_span(fj.cstr_left_weights);
  v.cstr_right_weights          = make_span(fj.cstr_right_weights);
  v.max_cstr_weight             = fj.max_cstr_weight.data();
  v.objective_vars              = make_span(fj.objective_vars);
  v.constraint_lower_bounds_csr = make_span(fj.constraint_lower_bounds_csr);
  v.constraint_upper_bounds_csr = make_span(fj.constraint_upper_bounds_csr);
  v.cstr_coeff_reciprocal       = make_span(fj.cstr_coeff_reciprocal);
  v.incumbent_lhs               = make_span(incumbent_lhs);
  v.incumbent_lhs_sumcomp       = make_span(incumbent_lhs_sumcomp);
  v.jump_move_scores            = make_span(jump_move_scores);
  v.jump_move_infeasibility     = make_span(jump_move_infeasibility);
  v.jump_move_delta             = make_span(jump_move_delta);
  v.jump_move_delta_check       = make_span(jump_move_delta_check);
  v.jump_move_score_check       = make_span(jump_move_score_check);
  v.move_last_update            = raft::make_mdspan<i_t, i_t>(
    move_last_update.data(),
    raft::make_extents<i_t>((i_t)FJ_MOVE_SIZE, (i_t)fj.pb_ptr->n_variables));
  v.move_delta = raft::make_mdspan<f_t, i_t>(
    move_delta.data(), raft::make_extents<i_t>((i_t)FJ_MOVE_SIZE, (i_t)fj.pb_ptr->n_variables));
  v.move_score = raft::make_mdspan<move_score_t, i_t>(
    move_score.data(), raft::make_extents<i_t>((i_t)FJ_MOVE_SIZE, (i_t)fj.pb_ptr->n_variables));
  v.tabu_nodec_until                  = make_span(tabu_nodec_until);
  v.tabu_noinc_until                  = make_span(tabu_noinc_until);
  v.tabu_lastdec                      = make_span(tabu_lastdec);
  v.tabu_lastinc                      = make_span(tabu_lastinc);
  v.jump_candidates                   = make_span(jump_candidates);
  v.jump_candidate_count              = make_span(jump_candidate_count);
  v.jump_locks                        = make_span(jump_locks);
  v.candidate_arrived_workids         = make_span(candidate_arrived_workids);
  v.grid_score_buf                    = make_span(grid_score_buf);
  v.grid_delta_buf                    = make_span(grid_delta_buf);
  v.grid_var_buf                      = make_span(grid_var_buf);
  v.row_size_bin_prefix_sum           = make_span(fj.row_size_bin_prefix_sum);
  v.row_size_nonbin_prefix_sum        = make_span(fj.row_size_nonbin_prefix_sum);
  v.work_id_to_bin_var_idx            = make_span(fj.work_id_to_bin_var_idx);
  v.work_id_to_nonbin_var_idx         = make_span(fj.work_id_to_nonbin_var_idx);
  v.work_ids_for_related_vars         = make_span(fj.work_ids_for_related_vars);
  v.selected_var                      = selected_var.data();
  v.violation_score                   = violation_score.data();
  v.weighted_violation_score          = weighted_violation_score.data();
  v.constraints_changed_count         = constraints_changed_count.data();
  v.local_minimums_reached            = local_minimums_reached.data();
  v.iterations                        = iterations.data();
  v.best_excess                       = best_excess.data();
  v.best_l1_distance                  = best_l1_distance.data();
  v.best_objective                    = best_objective.data();
  v.saved_solution_objective          = saved_solution_objective.data();
  v.incumbent_quality                 = incumbent_quality.data();
  v.incumbent_objective               = incumbent_objective.data();
  v.weight_update_increment           = fj.weight_update_increment;
  v.objective_weight                  = fj.objective_weight.data();
  v.last_minimum_iteration            = last_minimum_iteration.data();
  v.last_improving_minimum            = last_improving_minimum.data();
  v.last_iter_candidates              = last_iter_candidates.data();
  v.relvar_count_last_update          = relvar_count_last_update.data();
  v.load_balancing_skip               = load_balancing_skip.data();
  v.break_condition                   = break_condition.data();
  v.temp_break_condition              = temp_break_condition.data();
  v.best_jump_idx                     = best_jump_idx.data();
  v.stop_threshold                    = fj.stop_threshold;
  v.iterations_until_feasible_counter = iterations_until_feasible_counter.data();
  v.full_refresh_iteration            = full_refresh_iteration.data();
  RAFT_CUDA_TRY(cudaGetSymbolAddress((void**)&v.settings, device_settings));

  return v;
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::populate_climber_views()
{
  for (i_t i = 0; i < (i_t)climbers.size(); ++i) {
    climber_views.set_element(i, climbers[i]->view(), handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::copy_weights(const weight_t<i_t, f_t>& weights,
                                  const raft::handle_t* handle_ptr)
{
  cuopt_assert(cstr_weights.size() == weights.cstr_weights.size(), "Size mismatch");
  raft::copy(cstr_weights.data(),
             weights.cstr_weights.data(),
             weights.cstr_weights.size(),
             handle_ptr->get_stream());
  thrust::transform(handle_ptr->get_thrust_policy(),
                    weights.objective_weight.data(),
                    weights.objective_weight.data() + 1,
                    objective_weight.data(),
                    cuda::std::identity());
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::climber_data_t::clear_sets(const rmm::cuda_stream_view& stream)
{
  violated_constraints.clear(stream);
  candidate_variables.clear(stream);
  iteration_related_variables.clear(stream);
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::device_init(const rmm::cuda_stream_view& stream)
{
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::counting_iterator<i_t>(0),
                   thrust::counting_iterator<i_t>(pb_ptr->n_variables),
                   [pb = pb_ptr->view(), settings = settings] __device__(i_t var_idx) {
                     auto [offset_begin, offset_end] = pb.reverse_range_for_var(var_idx);

                     cuopt_assert(var_idx < pb.is_binary_variable.size(), "");
                     if (pb.is_binary_variable[var_idx]) {
                       cuopt_assert(pb.variable_lower_bounds[var_idx] == 0 &&
                                      pb.variable_upper_bounds[var_idx] == 1,
                                    "invalid bounds for binary variable");
                     }
                   });
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::climber_init(i_t climber_idx)
{
  auto& climber       = climbers[climber_idx];
  auto climber_stream = climber->stream.view();
  if (climber_idx == 0) climber_stream = handle_ptr->get_stream();

  return climber_init(climber_idx, climber_stream);
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::climber_init(i_t climber_idx, const rmm::cuda_stream_view& climber_stream)
{
  raft::common::nvtx::range scope("climber_init");

  cuopt_assert(climber_idx >= 0 && climber_idx < climbers.size(), "");

  auto& climber = climbers[climber_idx];
  auto view     = climber->view();

  climber->clear_sets(climber_stream);

  thrust::fill(rmm::exec_policy(climber_stream),
               climber->incumbent_lhs.begin(),
               climber->incumbent_lhs.end(),
               (f_t)0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->incumbent_lhs_sumcomp.begin(),
               climber->incumbent_lhs_sumcomp.end(),
               (f_t)0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->jump_move_scores.begin(),
               climber->jump_move_scores.end(),
               move_score_t::invalid());
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->jump_move_infeasibility.begin(),
               climber->jump_move_infeasibility.end(),
               (f_t)0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->jump_move_delta.begin(),
               climber->jump_move_delta.end(),
               (f_t)0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->move_last_update.begin(),
               climber->move_last_update.end(),
               std::numeric_limits<i_t>::lowest());
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->move_delta.begin(),
               climber->move_delta.end(),
               (f_t)0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->move_score.begin(),
               climber->move_score.end(),
               move_score_t::invalid());
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->tabu_nodec_until.begin(),
               climber->tabu_nodec_until.end(),
               0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->tabu_noinc_until.begin(),
               climber->tabu_noinc_until.end(),
               0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->tabu_lastdec.begin(),
               climber->tabu_lastdec.end(),
               -1);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->tabu_lastinc.begin(),
               climber->tabu_lastinc.end(),
               -1);
  thrust::fill(rmm::exec_policy(climber_stream),
               cstr_coeff_reciprocal.begin(),
               cstr_coeff_reciprocal.end(),
               0);
  thrust::fill(
    rmm::exec_policy(climber_stream), climber->jump_locks.begin(), climber->jump_locks.end(), 0);
  thrust::fill(rmm::exec_policy(climber_stream),
               climber->candidate_arrived_workids.begin(),
               climber->candidate_arrived_workids.end(),
               0);

  // clamp the incumbent assignment to within the variable bounds and round if needed
  thrust::for_each(
    rmm::exec_policy(climber_stream),
    thrust::counting_iterator<i_t>(0),
    thrust::counting_iterator<i_t>(pb_ptr->n_variables),
    [pb                   = pb_ptr->view(),
     incumbent_assignment = climber->incumbent_assignment.data()] __device__(i_t var_idx) {
      // round if integer
      if (pb.is_integer_var(var_idx)) {
        incumbent_assignment[var_idx] = round(incumbent_assignment[var_idx]);
      }
      // clamp to bounds
      incumbent_assignment[var_idx] =
        max(pb.variable_lower_bounds[var_idx],
            min(pb.variable_upper_bounds[var_idx], incumbent_assignment[var_idx]));
    });

  objective_vars.resize(pb_ptr->n_variables, climber_stream);
  auto end = thrust::copy_if(rmm::exec_policy(climber_stream),
                             thrust::counting_iterator<i_t>(0),
                             thrust::counting_iterator<i_t>(pb_ptr->n_variables),
                             objective_vars.begin(),
                             cuda::proclaim_return_type<bool>([v = view] __device__(i_t idx) {
                               return !v.pb.integer_equal(v.pb.objective_coefficients[idx], (f_t)0);
                             }));
  objective_vars.resize(end - objective_vars.begin(), climber_stream);

  // this does a stream sync inside
  f_t h_incumbent_obj = compute_objective_from_vec<i_t, f_t>(
    climber->incumbent_assignment, pb_ptr->objective_coefficients, climber_stream);
  climber->incumbent_objective.set_value_async(h_incumbent_obj, climber_stream);
  f_t inf = std::numeric_limits<f_t>::infinity();
  climber->best_objective.set_value_async(inf, climber_stream);
  climber->saved_solution_objective.set_value_async(inf, climber_stream);
  climber->violation_score.set_value_to_zero_async(climber_stream);
  climber->best_l1_distance.set_value_to_zero_async(climber_stream);
  climber->weighted_violation_score.set_value_to_zero_async(climber_stream);
  init_lhs_and_violation<i_t, f_t><<<256, 256, 0, climber_stream.value()>>>(view);

  // initialize the best_objective values according to the initial assignment
  f_t best_obj = compute_objective_from_vec<i_t, f_t>(
    climber->incumbent_assignment, pb_ptr->objective_coefficients, climber_stream);
  if (climber->violated_constraints.set_size.value(climber_stream) == 0) {
    climber->best_excess.set_value_to_zero_async(climber_stream);
    climber->best_objective.set_value_async(best_obj, climber_stream);
    climber->saved_solution_objective.set_value_async(best_obj, climber_stream);
  } else {
    f_t excess = climber->violation_score.value(climber_stream);
    climber->best_excess.set_value_async(excess, climber_stream);
  }
  climber_stream.synchronize();

  climber->break_condition.set_value_to_zero_async(climber_stream);
  climber->temp_break_condition.set_value_to_zero_async(climber_stream);
  climber->local_minimums_reached.set_value_to_zero_async(climber_stream);
  climber->last_minimum_iteration.set_value_to_zero_async(climber_stream);
  climber->last_improving_minimum.set_value_to_zero_async(climber_stream);
  climber->last_iter_candidates.set_value_to_zero_async(climber_stream);
  climber->relvar_count_last_update.set_value_to_zero_async(climber_stream);
  climber->load_balancing_skip.set_value_to_zero_async(climber_stream);
  climber->constraints_changed_count.set_value_to_zero_async(climber_stream);
  climber->iterations.set_value_to_zero_async(climber_stream);
  climber->full_refresh_iteration.set_value_to_zero_async(climber_stream);
  climber->iterations_until_feasible_counter.set_value_to_zero_async(climber_stream);

  climber_stream.synchronize();

  view = climber->view();

  // Compute the prefix sum to map any csr offsets to the corresponding variable indices
  // var_idx = bsearch_lower_bound(prefix_sum[], i_t workid)
  // warp granularity
  // TODO: sub-warp granularity (perform the reductions using the mask parameter)
  auto row_size_func = cuda::proclaim_return_type<i_t>([v = view] __device__(i_t var_idx) {
    return (v.pb.reverse_offsets[var_idx + 1] - v.pb.reverse_offsets[var_idx] - 1) /
             raft::WarpSize +
           1;
  });

  auto row_size_it_bin =
    thrust::make_transform_iterator(pb_ptr->binary_indices.data(), row_size_func);
  auto row_size_it_nonbin =
    thrust::make_transform_iterator(pb_ptr->nonbinary_indices.data(), row_size_func);

  size_t temp_storage_bytes = 0;
  for (i_t i = 0; i < 2; ++i) {
    cub::DeviceScan::InclusiveSum(i == 0 ? nullptr : climber->cub_storage_bytes.data(),
                                  temp_storage_bytes,
                                  row_size_it_bin,
                                  row_size_bin_prefix_sum.data(),
                                  pb_ptr->binary_indices.size(),
                                  climber_stream);
    if (i == 0 && temp_storage_bytes > climber->cub_storage_bytes.size())
      climber->cub_storage_bytes.resize(temp_storage_bytes, climber_stream);
  }
  temp_storage_bytes = 0;
  for (i_t i = 0; i < 2; ++i) {
    cub::DeviceScan::InclusiveSum(i == 0 ? nullptr : climber->cub_storage_bytes.data(),
                                  temp_storage_bytes,
                                  row_size_it_nonbin,
                                  row_size_nonbin_prefix_sum.data(),
                                  pb_ptr->nonbinary_indices.size(),
                                  climber_stream);
    if (i == 0 && temp_storage_bytes > climber->cub_storage_bytes.size())
      climber->cub_storage_bytes.resize(temp_storage_bytes, climber_stream);
  }

  view = climber->view();

  if (pb_ptr->related_variables.size() > 0) {
    // for each variable, compute the number of nnzs that would be examined during a FJ move update
    // pass to help determine whether to run load balancing or not
    temp_storage_bytes = 0;
    auto row_size_it =
      thrust::make_transform_iterator(pb_ptr->related_variables.begin(), row_size_func);
    // TODO: DeviceSegmentedReduce doesn't use full-fleged load balancing (relies on a 1 thread
    // block:1 segment mapping) Write another LB kernel for this?
    for (i_t i = 0; i < 2; ++i) {
      cub::DeviceSegmentedReduce::Sum(i == 0 ? nullptr : climber->cub_storage_bytes.data(),
                                      temp_storage_bytes,
                                      row_size_it,
                                      work_ids_for_related_vars.data(),
                                      pb_ptr->n_variables,
                                      pb_ptr->related_variables_offsets.begin(),
                                      pb_ptr->related_variables_offsets.begin() + 1,
                                      climber_stream);
      if (i == 0 && temp_storage_bytes > climber->cub_storage_bytes.size())
        climber->cub_storage_bytes.resize(temp_storage_bytes, climber_stream);
    }
  }

  // compute the explicit csr_offset to var_idx array
  if (pb_ptr->binary_indices.size() > 0)
    load_balancing_compute_workid_mappings<i_t, f_t><<<4096, 128, 0, climber_stream.value()>>>(
      view, view.row_size_bin_prefix_sum, view.pb.binary_indices, view.work_id_to_bin_var_idx);
  if (pb_ptr->nonbinary_indices.size() > 0)
    load_balancing_compute_workid_mappings<i_t, f_t>
      <<<4096, 128, 0, climber_stream.value()>>>(view,
                                                 view.row_size_nonbin_prefix_sum,
                                                 view.pb.nonbinary_indices,
                                                 view.work_id_to_nonbin_var_idx);

  if (pb_ptr->binary_indices.size() > 0)
    load_balancing_init_cstr_bounds_csr<i_t, f_t><<<4096, 128, 0, climber_stream.value()>>>(
      view, view.row_size_bin_prefix_sum, view.work_id_to_bin_var_idx);
  if (pb_ptr->nonbinary_indices.size() > 0)
    load_balancing_init_cstr_bounds_csr<i_t, f_t><<<4096, 128, 0, climber_stream.value()>>>(
      view, view.row_size_nonbin_prefix_sum, view.work_id_to_nonbin_var_idx);

  cuopt_assert(
    pb_ptr->binary_indices.size() + pb_ptr->nonbinary_indices.size() == pb_ptr->n_variables,
    "invalid variable indices total");

  cuopt_assert(thrust::all_of(rmm::exec_policy(climber_stream),
                              thrust::counting_iterator<i_t>(0),
                              thrust::counting_iterator<i_t>(cstr_coeff_reciprocal.size()),
                              [v = view] __device__(i_t offset) {
                                return v.cstr_coeff_reciprocal[offset] != 0 &&
                                       isfinite(v.cstr_coeff_reciprocal[offset]);
                              }),
               "");
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::set_fj_settings(fj_settings_t settings_)
{
  settings = settings_;
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::run_step_device(i_t climber_idx, bool use_graph)
{
  auto& data          = *climbers[climber_idx];
  auto climber_stream = data.stream.view();

  return run_step_device(climber_stream, climber_idx, use_graph);
}

// TODO: switch to conditional graph nodes once we switch to CTK >= 12.4
template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::load_balancing_score_update(const rmm::cuda_stream_view& stream,
                                                 i_t climber_idx)
{
  auto [grid_load_balancing_prepare, blocks_load_balancing_prepare] =
    load_balancing_prepare_launch_dims;
  auto [grid_load_balancing_binary, blocks_load_balancing_binary] =
    load_balancing_binary_launch_dims;
  auto [grid_load_balancing_mtm_compute_scores, blocks_load_balancing_mtm_compute_scores] =
    load_balancing_mtm_compute_scores_launch_dims;
  auto [grid_load_balancing_mtm_compute_candidates, blocks_load_balancing_mtm_compute_candidates] =
    load_balancing_mtm_compute_candidates_launch_dims;

  auto& data = *climbers[climber_idx];
  auto v     = data.view();

  data.iteration_related_variables.clear(stream);

  void* kernel_args[] = {&v};
  cudaLaunchCooperativeKernel((void*)load_balancing_prepare_iteration<i_t, f_t>,
                              grid_load_balancing_prepare,
                              blocks_load_balancing_prepare,
                              kernel_args,
                              0,
                              stream);

  data.load_balancing_start_event.record(stream);

  if (pb_ptr->binary_indices.size() > 0) {
    data.load_balancing_start_event.stream_wait(data.load_balancing_bin_stream.view());
    // compute the scores for binary variables (unique delta)
    load_balancing_compute_scores_binary<i_t, f_t><<<grid_load_balancing_binary,
                                                     blocks_load_balancing_binary,
                                                     0,
                                                     data.load_balancing_bin_stream.view()>>>(v);
    data.load_balancing_bin_finished_event.record(data.load_balancing_bin_stream.view());
  }
  if (pb_ptr->nonbinary_indices.size() > 0) {
    data.load_balancing_start_event.stream_wait(data.load_balancing_nonbin_stream.view());
    load_balancing_mtm_compute_candidates<i_t, f_t>
      <<<grid_load_balancing_mtm_compute_candidates,
         blocks_load_balancing_mtm_compute_candidates,
         0,
         data.load_balancing_nonbin_stream.view()>>>(v);
    load_balancing_mtm_compute_scores<i_t, f_t><<<grid_load_balancing_mtm_compute_scores,
                                                  blocks_load_balancing_mtm_compute_scores,
                                                  0,
                                                  data.load_balancing_nonbin_stream.view()>>>(v);
    data.load_balancing_nonbin_finished_event.record(data.load_balancing_nonbin_stream.view());
  }

  if (pb_ptr->binary_indices.size() > 0) data.load_balancing_bin_finished_event.stream_wait(stream);
  if (pb_ptr->nonbinary_indices.size() > 0)
    data.load_balancing_nonbin_finished_event.stream_wait(stream);

#if FJ_DEBUG_LOAD_BALANCING
  expand_device_copy(data.jump_move_delta_check, data.jump_move_delta, stream);
  expand_device_copy(data.jump_move_score_check, data.jump_move_scores, stream);
#endif
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::run_step_device(const rmm::cuda_stream_view& climber_stream,
                                     i_t climber_idx,
                                     bool use_graph)
{
  raft::common::nvtx::range scope("run_step_device");
  auto [grid_setval, blocks_setval]                 = setval_launch_dims;
  auto [grid_resetmoves, blocks_resetmoves]         = resetmoves_launch_dims;
  auto [grid_resetmoves_bin, blocks_resetmoves_bin] = resetmoves_bin_launch_dims;
  auto [grid_update_weights, blocks_update_weights] = update_weights_launch_dims;
  auto [grid_lift_move, blocks_lift_move]           = lift_move_launch_dims;

  auto& data    = *climbers[climber_idx];
  auto v        = data.view();
  settings.seed = cuopt::seed_generator::get_seed();
  // ensure an updated copy of the settings is used device-side
  raft::copy(v.settings, &settings, 1, climber_stream);

  bool is_binary_pb       = pb_ptr->n_variables == thrust::count(handle_ptr->get_thrust_policy(),
                                                           pb_ptr->is_binary_variable.begin(),
                                                           pb_ptr->is_binary_variable.end(),
                                                           1);
  bool use_load_balancing = false;
  if (settings.load_balancing_mode == fj_load_balancing_mode_t::ALWAYS_OFF) {
    use_load_balancing = false;
  } else if (settings.load_balancing_mode == fj_load_balancing_mode_t::ALWAYS_ON) {
    use_load_balancing = true;
  } else if (settings.load_balancing_mode == fj_load_balancing_mode_t::AUTO) {
    use_load_balancing =
      pb_ptr->n_variables > settings.parameters.load_balancing_codepath_min_varcount;
  }

  cudaGraph_t graph;
  void* kernel_args[]            = {&v};
  bool force_reset               = false;
  void* reset_moves_args[]       = {&v, &force_reset};
  bool ignore_load_balancing     = false;
  void* update_assignment_args[] = {&v, &ignore_load_balancing};
  if (!graph_created || !use_graph) {
    // CUB temp storage initialization
    size_t compaction_temp_storage_bytes = 0;
    auto valid_move_iterator             = thrust::make_transform_iterator(
      thrust::counting_iterator<i_t>(0),
      cuda::proclaim_return_type<i_t>([v] __device__(i_t i) -> i_t { return v.admits_move(i); }));
    cub::DeviceSelect::Flagged((void*)nullptr,
                               compaction_temp_storage_bytes,
                               thrust::counting_iterator<i_t>(0),
                               valid_move_iterator,
                               data.candidate_variables.contents.data(),
                               data.candidate_variables.set_size.data(),
                               pb_ptr->n_variables,
                               climber_stream);
    if (compaction_temp_storage_bytes > data.cub_storage_bytes.size()) {
      data.cub_storage_bytes.resize(compaction_temp_storage_bytes, climber_stream);
    }

    if (use_graph) {
      cudaGraphCreate(&graph, 0);
      cudaStreamBeginCapture(climber_stream, cudaStreamCaptureModeThreadLocal);
    }
    for (i_t i = 0; i < (use_graph ? iterations_per_graph : 1); ++i) {
      {
        // related varialbe array has to be dynamically computed each iteration
        if (pb_ptr->related_variables.size() == 0) {
          data.iteration_related_variables.clear(climber_stream);
        }

        // don't run the LB codepath if this is a small instance (n_var below threshold)
        if (use_load_balancing) {
          load_balancing_score_update(climber_stream, climber_idx);
        } else {
          if (is_binary_pb) {
            cudaLaunchCooperativeKernel(
              (void*)compute_mtm_moves_kernel<i_t, f_t, FJ_MTM_VIOLATED, true>,
              grid_resetmoves_bin,
              blocks_resetmoves_bin,
              reset_moves_args,
              0,
              climber_stream);
          } else {
            cudaLaunchCooperativeKernel(
              (void*)compute_mtm_moves_kernel<i_t, f_t, FJ_MTM_VIOLATED, false>,
              grid_resetmoves,
              blocks_resetmoves,
              reset_moves_args,
              0,
              climber_stream);
          }
        }
#if FJ_DEBUG_LOAD_BALANCING
        if (use_load_balancing) {
          cudaLaunchCooperativeKernel((void*)compute_mtm_moves_kernel<i_t, f_t>,
                                      grid_resetmoves_bin,
                                      blocks_resetmoves_bin,
                                      reset_moves_args,
                                      0,
                                      climber_stream);
          cudaLaunchCooperativeKernel((void*)load_balancing_sanity_checks<i_t, f_t>,
                                      512,
                                      128,
                                      kernel_args,
                                      0,
                                      climber_stream);
        }
#endif

        cudaLaunchKernel((void*)update_lift_moves_kernel<i_t, f_t>,
                         grid_lift_move,
                         blocks_lift_move,
                         kernel_args,
                         0,
                         climber_stream);
        cudaLaunchKernel((void*)update_breakthrough_moves_kernel<i_t, f_t>,
                         grid_lift_move,
                         blocks_lift_move,
                         kernel_args,
                         0,
                         climber_stream);
      }

      // compaction kernel
      cub::DeviceSelect::Flagged((void*)data.cub_storage_bytes.data(),
                                 compaction_temp_storage_bytes,
                                 thrust::counting_iterator<i_t>(0),
                                 valid_move_iterator,
                                 data.candidate_variables.contents.data(),
                                 data.candidate_variables.set_size.data(),
                                 pb_ptr->n_variables,
                                 climber_stream);

      cudaLaunchKernel((void*)select_variable_kernel<i_t, f_t>,
                       dim3(1),
                       dim3(256),
                       kernel_args,
                       0,
                       climber_stream);

      cudaLaunchCooperativeKernel((void*)handle_local_minimum_kernel<i_t, f_t>,
                                  grid_update_weights,
                                  blocks_update_weights,
                                  kernel_args,
                                  0,
                                  climber_stream);
      raft::copy(data.break_condition.data(), data.temp_break_condition.data(), 1, climber_stream);
      cudaLaunchKernel((void*)update_assignment_kernel<i_t, f_t>,
                       grid_setval,
                       blocks_setval,
                       update_assignment_args,
                       0,
                       climber_stream);
      cudaLaunchKernel((void*)update_changed_constraints_kernel<i_t, f_t>,
                       1,
                       blocks_setval,
                       kernel_args,
                       0,
                       climber_stream);
    }

    if (use_graph) {
      cudaStreamEndCapture(climber_stream, &graph);
      cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0);
      RAFT_CHECK_CUDA(climber_stream);
      cudaGraphDestroy(graph);
      graph_created = true;
    }
  }

  if (use_graph) cudaGraphLaunch(graph_instance, climber_stream);
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::refresh_lhs_and_violation(const rmm::cuda_stream_view& stream, i_t climber_idx)
{
  auto& data = *climbers[climber_idx];
  auto v     = data.view();

  data.violated_constraints.clear(stream);
  data.violation_score.set_value_to_zero_async(stream);
  data.weighted_violation_score.set_value_to_zero_async(stream);
  init_lhs_and_violation<i_t, f_t><<<4096, 256, 0, stream>>>(v);
}

template <typename i_t, typename f_t>
i_t fj_t<i_t, f_t>::host_loop(solution_t<i_t, f_t>& solution, i_t climber_idx)
{
  auto& data = *climbers[climber_idx];
  auto v     = data.view();  // == climber_views[climber_idx]

  auto climber_stream = data.stream.view();
  if (climber_idx == 0) climber_stream = handle_ptr->get_stream();

  auto [grid_resetmoves, blocks_resetmoves] = resetmoves_launch_dims;
  solution.compute_feasibility();
  if (settings.feasibility_run) {
    objective_weight.set_value_to_zero_async(handle_ptr->get_stream());
  }
  f_t obj = -std::numeric_limits<f_t>::infinity();
  data.incumbent_quality.set_value_async(obj, handle_ptr->get_stream());

  cuopt_assert((settings.termination & fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT) ||
                 (settings.termination & fj_termination_flags_t::FJ_TERMINATION_ITERATION_LIMIT),
               "invalid termination criteria");
  data.incumbent_quality.set_value_async(obj, handle_ptr->get_stream());

  timer_t timer(settings.time_limit);
  i_t steps;
  bool limit_reached = false;
  for (steps = 0; steps < std::numeric_limits<i_t>::max(); steps += iterations_per_graph) {
    // to actualize time limit
    handle_ptr->sync_stream();
    if ((settings.termination & fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT) &&
        timer.check_time_limit()) {
      limit_reached = true;
    }
    if ((settings.termination & fj_termination_flags_t::FJ_TERMINATION_ITERATION_LIMIT) &&
        steps >= settings.iteration_limit) {
      limit_reached = true;
    }

#if !FJ_SINGLE_STEP
    if (steps % 500 == 0)
#endif
    {
      CUOPT_LOG_TRACE(
        "FJ "
        "step %d viol %.2g [%d], obj %.8g, best %.8g, mins %d, maxw %g, "
        "objw %g",
        steps,
        data.violation_score.value(climber_stream),
        data.violated_constraints.set_size.value(climber_stream),
        data.incumbent_objective.value(climber_stream),
        data.best_objective.value(climber_stream),
        data.local_minimums_reached.value(climber_stream),
        max_cstr_weight.value(climber_stream),
        objective_weight.value(climber_stream));
    }

    if (!limit_reached) { run_step_device(climber_stream, climber_idx); }

    // periodically recompute the LHS and violation scores
    // to correct any accumulated numerical errors
    if (steps % settings.parameters.lhs_refresh_period == 0) {
      refresh_lhs_and_violation(climber_stream, climber_idx);
    }

    // periodically synchronize and check the latest solution
    // feasible solution found!*view.break_condition
    if (steps % settings.parameters.sync_period == 0 || limit_reached) {
      i_t break_condition = data.break_condition.value(climber_stream);
      if (settings.mode == fj_mode_t::GREEDY_DESCENT) {
        if (!limit_reached && !break_condition) { continue; }
      }

      i_t iterations = data.iterations.value(climber_stream);
      // make sure we have the current incumbent saved (e.g. in the case of a timeout)
      update_best_solution_kernel<i_t, f_t><<<1, blocks_resetmoves, 0, climber_stream>>>(v);
      // check feasibility with the relative tolerance rather than the violation score
      raft::copy(solution.assignment.data(),
                 data.best_assignment.data(),
                 data.best_assignment.size(),
                 climber_stream);
      climber_stream.synchronize();
      // this solution cost computation with the changing(or not changing) weights is needed to
      // decide whether we reset the best objective on the FIRST_FEASIBLE mode. once we get rid of
      // FIRST_FEASIBLE mode, we can remove the following too.
      bool is_feasible = solution.compute_feasibility();
      solution.handle_ptr->sync_stream();

      if (limit_reached) { break; }

      if (is_feasible) {
        if (settings.mode == fj_mode_t::FIRST_FEASIBLE) { break; }
      }

      if (settings.mode != fj_mode_t::FIRST_FEASIBLE) {
        if (break_condition) {
          f_t exit_best_objective = data.best_objective.value(handle_ptr->get_stream());
          CUOPT_LOG_TRACE(FJ_LOG_PREFIX "EXIT LOCAL MINIMUM step %d, best best_objective %f",
                          iterations,
                          exit_best_objective);
          break;
        }
      }
    }
  }
#if CUOPT_LOG_ACTIVE_LEVEL == CUOPT_LOG_LEVEL_TRACE
  auto h_sol = cuopt::host_copy(solution.assignment, climber_stream);
  static std::set<std::vector<f_t>> solutions_set;
  bool same_sol = solutions_set.count(h_sol) > 0;
  solutions_set.insert(h_sol);
  CUOPT_LOG_TRACE("n_iter %d n_local_mins %d best excess %f same sol %d max cstr score %f",
                  steps,
                  data.local_minimums_reached.value(climber_stream),
                  -data.best_excess.value(climber_stream),
                  same_sol,
                  max_cstr_weight.value(climber_stream));
#endif
  CUOPT_LOG_TRACE("EXIT step %d, best objective %f best_excess %f, feas %d",
                  data.iterations.value(climber_stream),
                  solution.get_user_objective(),
                  solution.get_total_excess(),
                  solution.get_feasible());

  return steps;
}

template <typename i_t, typename f_t>
i_t fj_t<i_t, f_t>::alloc_max_climbers(i_t desired_climbers)
{
  if (climbers.size() >= (size_t)desired_climbers) return desired_climbers;

  i_t n_climbers = desired_climbers;

  bool alloc_failed = false;
  // TODO don't allocate the full pool
  // the rmm uses upstream if the allocation doesn't succeed
  // besides some thrust calls might be using allocations and there might not be any memory left
  // for now only do 1 climber allocation
  do {
    alloc_failed = false;
    try {
      climbers.resize(n_climbers);
      for (auto& climber_ptr : climbers)
        climber_ptr = std::make_unique<climber_data_t>(*this);
      climber_views.resize(climbers.size(), handle_ptr->get_stream());
    } catch (const rmm::out_of_memory&) {
      n_climbers /= 2;
      alloc_failed = true;
      cudaGetLastError();
    }
  } while (alloc_failed && n_climbers > 0);
  cuopt_expects(n_climbers > 0,
                error_type_t::OutOfMemoryError,
                "Unable to allocate enough memory for the problem");

  return n_climbers;
}

template <typename i_t, typename f_t>
void fj_t<i_t, f_t>::resize_vectors(const raft::handle_t* handle_ptr)
{
  // climber related vars
  climbers[0]->constraints_changed.resize(pb_ptr->n_constraints, handle_ptr->get_stream());
  climbers[0]->violated_constraints.resize(pb_ptr->n_constraints, handle_ptr->get_stream());
  climbers[0]->best_assignment.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->farthest_l1_sol.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->incumbent_assignment.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->incumbent_lhs.resize(pb_ptr->n_constraints, handle_ptr->get_stream());
  climbers[0]->incumbent_lhs_sumcomp.resize(pb_ptr->n_constraints, handle_ptr->get_stream());
  climbers[0]->jump_move_scores.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->jump_move_delta.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->jump_move_infeasibility.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->move_last_update.resize(pb_ptr->n_variables * FJ_MOVE_SIZE,
                                       handle_ptr->get_stream());
  climbers[0]->move_delta.resize(pb_ptr->n_variables * FJ_MOVE_SIZE, handle_ptr->get_stream());
  climbers[0]->move_score.resize(pb_ptr->n_variables * FJ_MOVE_SIZE, handle_ptr->get_stream());
  climbers[0]->tabu_nodec_until.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->tabu_noinc_until.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->tabu_lastdec.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->tabu_lastinc.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->candidate_variables.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->iteration_related_variables.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->jump_locks.resize(pb_ptr->n_variables, handle_ptr->get_stream());
  climbers[0]->candidate_arrived_workids.resize(pb_ptr->coefficients.size(),
                                                handle_ptr->get_stream());
  climbers[0]->jump_candidates.resize(pb_ptr->coefficients.size(), handle_ptr->get_stream());
  climbers[0]->jump_candidate_count.resize(pb_ptr->n_variables, handle_ptr->get_stream());

  climbers[0]->grid_score_buf.resize(update_weights_launch_dims.first.x, handle_ptr->get_stream());
  climbers[0]->grid_var_buf.resize(update_weights_launch_dims.first.x, handle_ptr->get_stream());
  climbers[0]->grid_delta_buf.resize(update_weights_launch_dims.first.x, handle_ptr->get_stream());

  // FJ related vars
  cstr_weights.resize(pb_ptr->n_constraints, handle_ptr->get_stream());
  cstr_right_weights.resize(pb_ptr->n_constraints, handle_ptr->get_stream());
  cstr_left_weights.resize(pb_ptr->n_constraints, handle_ptr->get_stream());
  constraint_lower_bounds_csr.resize(pb_ptr->coefficients.size(), handle_ptr->get_stream());
  constraint_upper_bounds_csr.resize(pb_ptr->coefficients.size(), handle_ptr->get_stream());
  cstr_coeff_reciprocal.resize(pb_ptr->coefficients.size(), handle_ptr->get_stream());
  work_id_to_bin_var_idx.resize(pb_ptr->coefficients.size(), handle_ptr->get_stream());
  work_id_to_nonbin_var_idx.resize(pb_ptr->coefficients.size(), handle_ptr->get_stream());
  row_size_bin_prefix_sum.resize(pb_ptr->binary_indices.size(), handle_ptr->get_stream());
  row_size_nonbin_prefix_sum.resize(pb_ptr->nonbinary_indices.size(), handle_ptr->get_stream());
  work_ids_for_related_vars.resize(pb_ptr->n_variables, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
i_t fj_t<i_t, f_t>::solve(solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range scope("fj_solve");
  timer_t timer(settings.time_limit);
  handle_ptr = const_cast<raft::handle_t*>(solution.handle_ptr);
  pb_ptr     = solution.problem_ptr;
  cuopt_func_call(solution.test_variable_bounds(true));
  cuopt_assert(solution.test_number_all_integer(), "All integers must be rounded");
  pb_ptr->check_problem_representation(true);
  resize_vectors(solution.handle_ptr);

  bool is_initial_feasible = solution.compute_feasibility();

  // TODO only call this when the size is different
  device_init(handle_ptr->get_stream());
  // TODO check if we are returning the initial solution
  // check the best_quality is set properly
  for (i_t i = 0; i < (i_t)climbers.size(); ++i) {
    raft::copy(climbers[i]->incumbent_assignment.data(),
               solution.assignment.data(),
               solution.problem_ptr->n_variables,
               handle_ptr->get_stream());
    // we might never call save solution and never initialize the best solution which is returned
    // that's why always have the original solution in best_assignment in the beginning
    raft::copy(climbers[i]->best_assignment.data(),
               solution.assignment.data(),
               solution.problem_ptr->n_variables,
               handle_ptr->get_stream());
  }

  climber_init(0);
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  handle_ptr->sync_stream();

  i_t iterations = host_loop(solution);
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  handle_ptr->sync_stream();

  f_t effort_rate = (f_t)iterations / timer.elapsed_time();

  CUOPT_LOG_TRACE("GPU solver took %g", timer.elapsed_time());
  CUOPT_LOG_TRACE("limit reached, effort rate %g steps/secm %d steps", effort_rate, iterations);
  reset_cuda_graph();
  cuopt_assert(solution.test_number_all_integer(), "All integers must be rounded");
  bool is_new_feasible = solution.compute_feasibility();

  if (is_initial_feasible && !is_new_feasible) {
    CUOPT_LOG_ERROR(
      "Feasibility jump caused feasible solution to become infeasible\n"
      "Best excess is %g",
      climbers[0]->best_excess.value(handle_ptr->get_stream()));
    cuopt_assert(false, "Feasibility jump caused feasible solution to become infeasible");
  }

  return is_new_feasible;
}

#if MIP_INSTANTIATE_FLOAT
template class fj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class fj_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
