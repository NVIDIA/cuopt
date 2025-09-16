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

#include "probing_cache.cuh"

#include <mip/mip_constants.hpp>
#include <mip/utils.cuh>

#include <thrust/sort.h>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void lb_probing_cache_t<i_t, f_t>::update_bounds_with_selected(
  std::vector<f_t>& host_bounds,
  const cache_entry_t<i_t, f_t>& cache_entry,
  const std::vector<i_t>& reverse_original_ids)
{
  i_t n_bounds_updated = 0;
  for (const auto& [var_idx, bound] : cache_entry.var_to_cached_bound_map) {
    i_t var_idx_in_current_problem = reverse_original_ids[var_idx];
    // -1 means that variable was fixed and doesn't exists in the current problem
    if (var_idx_in_current_problem == -1) { continue; }
    if (host_bounds[2 * var_idx_in_current_problem] < bound.lb) {
      host_bounds[2 * var_idx_in_current_problem] = bound.lb;
      n_bounds_updated++;
    }
    if (host_bounds[2 * var_idx_in_current_problem + 1] > bound.ub) {
      host_bounds[2 * var_idx_in_current_problem + 1] = bound.ub;
      n_bounds_updated++;
    }
  }
}

template <typename i_t, typename f_t>
i_t lb_probing_cache_t<i_t, f_t>::check_number_of_conflicting_vars(
  const std::vector<f_t>& host_bounds,
  const cache_entry_t<i_t, f_t>& cache_entry,
  f_t integrality_tolerance,
  const std::vector<i_t>& reverse_original_ids)
{
  i_t n_conflicting_var = 0;
  for (const auto& [var_idx, bound] : cache_entry.var_to_cached_bound_map) {
    i_t var_idx_in_current_problem = reverse_original_ids[var_idx];
    // -1 means that variable was fixed and doesn't exists in the current problem
    if (var_idx_in_current_problem == -1) { continue; }
    if (host_bounds[2 * var_idx_in_current_problem] - integrality_tolerance > bound.ub ||
        host_bounds[2 * var_idx_in_current_problem + 1] < bound.lb - integrality_tolerance) {
      ++n_conflicting_var;
    }
  }
  return n_conflicting_var;
}

template <typename i_t, typename f_t>
f_t lb_probing_cache_t<i_t, f_t>::get_least_conflicting_rounding(problem_t<i_t, f_t>& problem,
                                                                 std::vector<f_t>& host_bounds,
                                                                 i_t var_id_on_problem,
                                                                 f_t first_probe,
                                                                 f_t second_probe,
                                                                 f_t integrality_tolerance)
{
  i_t var_id      = problem.original_ids[var_id_on_problem];
  auto& cache_row = probing_cache[var_id];

  i_t hit_interval_for_first_probe  = -1;
  i_t hit_interval_for_second_probe = -1;
  for (i_t i = 0; i < 2; ++i) {
    auto& cache_entry = cache_row[i];
    // if no implied bounds found go to next interval
    if (cache_entry.var_to_cached_bound_map.empty()) { continue; }
    cache_entry.val_interval.fill_cache_hits(
      i, first_probe, second_probe, hit_interval_for_first_probe, hit_interval_for_second_probe);
  }
  i_t n_conflicting_vars = 0;
  // first probe found some interval
  if (hit_interval_for_first_probe != -1) {
    n_conflicting_vars = check_number_of_conflicting_vars(host_bounds,
                                                          cache_row[hit_interval_for_first_probe],
                                                          integrality_tolerance,
                                                          problem.reverse_original_ids);
    if (n_conflicting_vars == 0) {
      CUOPT_LOG_TRACE("No conflicting vars, returning first probe");
      update_bounds_with_selected(
        host_bounds, cache_row[hit_interval_for_first_probe], problem.reverse_original_ids);
      return first_probe;
    }
  }
  // if the interval is still -1, it means this probing doesn't have any implied bounds
  else {
    CUOPT_LOG_TRACE("No implied bounds on first probe, returning first probe");
    return first_probe;
  }
  CUOPT_LOG_TRACE("Conflicting vars %d found in first probing, searching least conflicting!",
                  n_conflicting_vars);
  // check for the other side, if it the interval includes second_probe return that, if not return
  // cutoff point second probe has a hit but it is not the same as first probe
  i_t other_interval_idx = 1 - hit_interval_for_first_probe;
  i_t n_conflicting_vars_other_probe =
    check_number_of_conflicting_vars(host_bounds,
                                     cache_row[other_interval_idx],
                                     integrality_tolerance,
                                     problem.reverse_original_ids);

  if (n_conflicting_vars_other_probe < n_conflicting_vars) {
    CUOPT_LOG_DEBUG(
      "Better conflicting vars found %d in the other probing region (cache interval)!",
      n_conflicting_vars_other_probe);
    update_bounds_with_selected(
      host_bounds, cache_row[other_interval_idx], problem.reverse_original_ids);
    if (other_interval_idx == hit_interval_for_second_probe) {
      return second_probe;
    } else {
      return cache_row[other_interval_idx].val_interval.val;
    }
  }
  update_bounds_with_selected(
    host_bounds, cache_row[hit_interval_for_first_probe], problem.reverse_original_ids);
  return first_probe;
}

template <typename i_t, typename f_t>
bool lb_probing_cache_t<i_t, f_t>::contains(problem_t<i_t, f_t>& problem, i_t var_id)
{
  return probing_cache.count(problem.original_ids[var_id]) > 0;
}

template <typename i_t, typename f_t, typename f_t2>
void inline insert_current_probing_to_cache(
  i_t var_idx,
  val_interval_t<i_t, f_t>& probe_val,
  load_balanced_bounds_presolve_t<i_t, f_t>& bound_presolve,
  const std::vector<f_t>& original_lb,
  const std::vector<f_t>& original_ub,
  const std::vector<f_t2>& modified_bounds,
  const std::vector<i_t>& h_integer_indices,
  i_t& n_implied_singletons)
{
  f_t int_tol = bound_presolve.context.settings.tolerances.integrality_tolerance;

  cache_entry_t<i_t, f_t> cache_item;
  cache_item.val_interval = probe_val;
  for (auto impacted_var_idx : h_integer_indices) {
    if (original_lb[impacted_var_idx] != modified_bounds[2 * impacted_var_idx] ||
        original_ub[impacted_var_idx] != modified_bounds[2 * impacted_var_idx + 1]) {
      if (integer_equal<f_t>(modified_bounds[2 * impacted_var_idx],
                             modified_bounds[2 * impacted_var_idx + 1],
                             int_tol)) {
        ++n_implied_singletons;
      }
      cached_bound_t<f_t> new_bound{modified_bounds[2 * impacted_var_idx],
                                    modified_bounds[2 * impacted_var_idx + 1]};
      cache_item.var_to_cached_bound_map.insert({impacted_var_idx, new_bound});
    }
  }
  if (!bound_presolve.probing_cache.probing_cache.count(var_idx) > 0) {
    std::array<cache_entry_t<i_t, f_t>, 2> entries_per_var;
    entries_per_var[0] = cache_item;
    bound_presolve.probing_cache.probing_cache.insert({var_idx, entries_per_var});
  } else {
    bound_presolve.probing_cache.probing_cache[var_idx][1] = cache_item;
  }
}

template <typename i_t, typename f_t, typename f_t2>
__global__ void compute_min_slack_per_var(typename problem_t<i_t, f_t>::view_t pb,
                                          raft::device_span<f_t2> cnst_slack,
                                          // raft::device_span<f_t> min_activity,
                                          // raft::device_span<f_t> max_activity,
                                          raft::device_span<f_t> var_slack,
                                          raft::device_span<bool> different_coefficient,
                                          raft::device_span<f_t> max_excess_per_var,
                                          raft::device_span<i_t> max_n_violated_per_constraint)
{
  i_t var_idx           = pb.integer_indices[blockIdx.x];
  i_t var_offset        = pb.reverse_offsets[var_idx];
  i_t var_degree        = pb.reverse_offsets[var_idx + 1] - var_offset;
  f_t th_var_unit_slack = std::numeric_limits<f_t>::max();
  f_t lb                = pb.variable_lower_bounds[var_idx];
  f_t ub                = pb.variable_upper_bounds[var_idx];
  f_t first_coeff       = pb.reverse_coefficients[var_offset];
  bool different_coeff  = false;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto a = pb.reverse_coefficients[var_offset + i];
    if (std::signbit(a) != std::signbit(first_coeff)) { different_coeff = true; }
    auto cnst_idx      = pb.reverse_constraints[var_offset + i];
    auto cnstr_slack   = cnst_slack[cnst_idx];
    auto delta_min_act = get_lower(cnstr_slack) + ((a < 0) ? a * ub : a * lb);
    th_var_unit_slack  = min(th_var_unit_slack, (delta_min_act / a));
    auto delta_max_act = get_upper(cnstr_slack) + ((a > 0) ? a * ub : a * lb);
    th_var_unit_slack  = min(th_var_unit_slack, (delta_max_act / a));
  }
  __shared__ f_t shmem[raft::WarpSize];
  f_t block_var_unit_slack = raft::blockReduce(th_var_unit_slack, (char*)shmem, raft::min_op{});
  __syncthreads();
  i_t block_different_coeff = raft::blockReduce((i_t)different_coeff, (char*)shmem);
  if (threadIdx.x == 0) {
    var_slack[blockIdx.x]             = block_var_unit_slack;
    different_coefficient[blockIdx.x] = block_different_coeff > 0;
  }
  __syncthreads();
  // return vars that will have no implied bounds
  if (!different_coefficient[blockIdx.x]) { return; }
  // for each variable that appers with negated coeffs in different cosntraints
  // check whether flipping the var from lb to ub in constraints with positive coefficient
  // violates the constraint. we do it for 4 situation that can be inferred.
  i_t th_n_of_excess = 0;
  f_t th_max_excess  = 0.;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto a           = pb.reverse_coefficients[var_offset + i];
    auto cnst_idx    = pb.reverse_constraints[var_offset + i];
    auto cnstr_slack = cnst_slack[cnst_idx];
    f_t diff         = (a < 0) ? a * (lb - ub) : a * (ub - lb);
    f_t excess       = max(0., diff - cnstr_slack.x);
    if (excess > 0) {
      th_max_excess = max(th_max_excess, excess);
      th_n_of_excess++;
    }
    excess = max(0., get_upper(cnstr_slack) + diff);
    if (excess > 0) {
      th_max_excess = max(th_max_excess, excess);
      th_n_of_excess++;
    }
  }
  f_t max_excess = raft::blockReduce(th_max_excess, (char*)shmem, raft::max_op{});
  __syncthreads();
  i_t total_excessed_cstr = raft::blockReduce(th_n_of_excess, (char*)shmem);
  if (threadIdx.x == 0) {
    max_excess_per_var[blockIdx.x]            = max_excess;
    max_n_violated_per_constraint[blockIdx.x] = total_excessed_cstr;
  }
}

template <typename i_t, typename f_t>
inline std::vector<i_t> compute_prioritized_integer_indices(
  load_balanced_bounds_presolve_t<i_t, f_t>& bound_presolve,
  load_balanced_problem_t<i_t, f_t>& problem)
{
  // sort the variables according to the min slack they have across constraints
  // we also need to consider the variable range
  // the priority is computed as the var_range * min_slack
  // min_slack is computed as var_range*coefficient/(b - min_act)
  rmm::device_uvector<f_t> min_slack_per_var(problem.pb->n_integer_vars,
                                             problem.handle_ptr->get_stream());
  rmm::device_uvector<i_t> priority_indices(problem.pb->integer_indices,
                                            problem.handle_ptr->get_stream());
  rmm::device_uvector<bool> different_coefficient(problem.pb->n_integer_vars,
                                                  problem.handle_ptr->get_stream());
  rmm::device_uvector<f_t> max_excess_per_var(problem.pb->n_integer_vars,
                                              problem.handle_ptr->get_stream());
  rmm::device_uvector<i_t> max_n_violated_per_constraint(problem.pb->n_integer_vars,
                                                         problem.handle_ptr->get_stream());
  thrust::fill(problem.handle_ptr->get_thrust_policy(),
               min_slack_per_var.begin(),
               min_slack_per_var.end(),
               std::numeric_limits<f_t>::max());

  thrust::fill(problem.handle_ptr->get_thrust_policy(),
               max_excess_per_var.begin(),
               max_excess_per_var.end(),
               0);
  thrust::fill(problem.handle_ptr->get_thrust_policy(),
               max_n_violated_per_constraint.begin(),
               max_n_violated_per_constraint.end(),
               0);
  // compute min and max activity first
  bound_presolve.copy_input_bounds(problem);
  bound_presolve.calculate_constraint_slack(problem.handle_ptr);
  bool res = bound_presolve.calculate_infeasible_redundant_constraints(problem.handle_ptr);
  cuopt_assert(res, "The activity computation must be feasible during probing cache!");
  CUOPT_LOG_INFO("prioritized integer_indices n_integer_vars %d", problem.pb->n_integer_vars);
  // compute the min var slack
  compute_min_slack_per_var<i_t, f_t>
    <<<problem.pb->n_integer_vars, 128, 0, problem.handle_ptr->get_stream()>>>(
      problem.pb->view(),
      make_span_2(bound_presolve.cnst_slack),
      make_span(min_slack_per_var),
      make_span(different_coefficient),
      make_span(max_excess_per_var),
      make_span(max_n_violated_per_constraint));
  auto iterator = thrust::make_zip_iterator(thrust::make_tuple(
    max_n_violated_per_constraint.begin(), max_excess_per_var.begin(), min_slack_per_var.begin()));
  // sort the vars
  thrust::sort_by_key(problem.handle_ptr->get_thrust_policy(),
                      iterator,
                      iterator + problem.pb->n_integer_vars,
                      priority_indices.begin(),
                      [] __device__(auto tuple1, auto tuple2) {
                        // if both are zero, i.e. no excess, sort it by min slack
                        if (thrust::get<0>(tuple1) == 0 && thrust::get<0>(tuple2) == 0) {
                          return thrust::get<2>(tuple1) < thrust::get<2>(tuple2);
                        } else if (thrust::get<0>(tuple1) > thrust::get<0>(tuple2)) {
                          return true;
                        } else if (thrust::get<0>(tuple1) == thrust::get<0>(tuple2)) {
                          return thrust::get<1>(tuple1) > thrust::get<1>(tuple2);
                        }
                        return false;
                      });
  auto h_priority_indices = host_copy(priority_indices);
  return h_priority_indices;
}

template <typename i_t, typename f_t>
void compute_probing_cache(load_balanced_bounds_presolve_t<i_t, f_t>& bound_presolve,
                           load_balanced_problem_t<i_t, f_t>& problem,
                           timer_t timer)
{
  // we dont want to compute the probing cache for all variables for time and computation resources
  auto priority_indices = compute_prioritized_integer_indices(bound_presolve, problem);
  // std::cout<<"priority_indices\n";
  CUOPT_LOG_DEBUG("Computing probing cache");
  auto h_integer_indices      = host_copy(problem.pb->integer_indices);
  auto h_var_upper_bounds     = host_copy(problem.pb->variable_upper_bounds);
  auto h_var_lower_bounds     = host_copy(problem.pb->variable_lower_bounds);
  size_t n_of_cached_probings = 0;
  // TODO adjust the iteration limit depending on the total time limit and time it takes for single
  // var
  bound_presolve.settings.iteration_limit = 50;
  bound_presolve.settings.time_limit      = timer.remaining_time();
  i_t n_of_implied_singletons             = 0;
  // for each integer var, loop around two possible values/intervals and save the implied bounds
  for (auto var_idx : priority_indices) {
    if (timer.check_time_limit()) { break; }
    f_t lb = h_var_lower_bounds[var_idx];
    f_t ub = h_var_upper_bounds[var_idx];
    // TODO make this more efficient, we can do double probing
    for (i_t i = 0; i < 2; ++i) {
      val_interval_t<i_t, f_t> probe_val;
      // if binary, probe both values
      if (problem.pb->integer_equal(ub - lb, 1.)) {
        probe_val.interval_type = interval_type_t::EQUALS;
        probe_val.val           = i == 0 ? lb : ub;
      }
      // if both sides are finite, probe on lower half and upper half
      else if (isfinite(lb) && isfinite(ub)) {
        probe_val.interval_type = i == 0 ? interval_type_t::LEQ : interval_type_t::GEQ;
        f_t middle              = floor((lb + ub) / 2);
        probe_val.val           = i == 0 ? middle : middle + 1;
      }
      // if only lower bound is finite, probe on lb and >lb
      else if (isfinite(lb)) {
        probe_val.interval_type = i == 0 ? interval_type_t::EQUALS : interval_type_t::GEQ;
        probe_val.val           = i == 0 ? lb : lb + 1;
      }
      // if only upper bound is finite, probe on ub and <ub
      else {
        probe_val.interval_type = i == 0 ? interval_type_t::EQUALS : interval_type_t::LEQ;
        probe_val.val           = i == 0 ? ub : ub - 1;
      }
      // now solve bounds presolve for the value or the interval
      // if the type is equals, just set the value and solve the bounds presolve
      auto bounds_presolve_result = termination_criterion_t::NO_UPDATE;
      if (probe_val.interval_type == interval_type_t::EQUALS) {
        std::vector<thrust::pair<i_t, f_t>> var_probe_val_pairs;
        var_probe_val_pairs.push_back(thrust::make_pair(var_idx, probe_val.val));
        bounds_presolve_result = bound_presolve.solve(var_probe_val_pairs);
      }
      // if it is an interval change the variable bound and solve
      else {
        if (probe_val.interval_type == interval_type_t::LEQ) {
          ub = probe_val.val;
        } else {
          lb = probe_val.val;
        }
        bounds_presolve_result = bound_presolve.solve(lb, ub, var_idx);
      }
      // save the impacted bounds
      if (bounds_presolve_result != termination_criterion_t::NO_UPDATE) {
        n_of_cached_probings++;
        CUOPT_LOG_TRACE("Cached bounds are added for var %d", var_idx);
        // TODO make this more efficient, keep the changed bounds in bounds presolve
        bound_presolve.update_host_bounds(problem.handle_ptr);
        // TODO
        insert_current_probing_to_cache(var_idx,
                                        probe_val,
                                        bound_presolve,
                                        h_var_lower_bounds,
                                        h_var_upper_bounds,
                                        bound_presolve.host_bounds,
                                        h_integer_indices,
                                        n_of_implied_singletons);
      }
    }
  }
  CUOPT_LOG_INFO("Total number of cached probings %d number of implied singletons %d",
                 n_of_cached_probings,
                 n_of_implied_singletons);
  // restore the settings
  bound_presolve.settings = {};
}

#define INSTANTIATE(F_TYPE)                                        \
  template void compute_probing_cache<int, F_TYPE>(                \
    load_balanced_bounds_presolve_t<int, F_TYPE> & bound_presolve, \
    load_balanced_problem_t<int, F_TYPE> & problem,                \
    timer_t timer);                                                \
  template class lb_probing_cache_t<int, F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
