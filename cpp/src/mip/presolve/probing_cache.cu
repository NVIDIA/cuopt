/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "probing_cache.cuh"
#include "trivial_presolve.cuh"

#include <mip/mip_constants.hpp>
#include <mip/presolve/multi_probe.cuh>
#include <mip/utilities/work_unit_ordered_queue.cuh>
#include <mip/utils.cuh>

#include <omp.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
i_t probing_cache_t<i_t, f_t>::check_number_of_conflicting_vars(
  const std::vector<f_t>& host_lb,
  const std::vector<f_t>& host_ub,
  const cache_entry_t<i_t, f_t>& cache_entry,
  f_t integrality_tolerance,
  const std::vector<i_t>& reverse_original_ids)
{
  i_t n_conflicting_var = 0;
  for (const auto& [var_idx, bound] : cache_entry.var_to_cached_bound_map) {
    i_t var_idx_in_current_problem = reverse_original_ids[var_idx];
    // -1 means that variable was fixed and doesn't exists in the current problem
    if (var_idx_in_current_problem == -1) { continue; }
    if (host_lb[var_idx_in_current_problem] - integrality_tolerance > bound.ub ||
        host_ub[var_idx_in_current_problem] < bound.lb - integrality_tolerance) {
      ++n_conflicting_var;
    }
  }
  return n_conflicting_var;
}

template <typename i_t, typename f_t>
void probing_cache_t<i_t, f_t>::update_bounds_with_selected(
  std::vector<f_t>& host_lb,
  std::vector<f_t>& host_ub,
  const cache_entry_t<i_t, f_t>& cache_entry,
  const std::vector<i_t>& reverse_original_ids)
{
  i_t n_bounds_updated = 0;
  for (const auto& [var_idx, bound] : cache_entry.var_to_cached_bound_map) {
    i_t var_idx_in_current_problem = reverse_original_ids[var_idx];
    // -1 means that variable was fixed and doesn't exists in the current problem
    if (var_idx_in_current_problem == -1) { continue; }
    if (host_lb[var_idx_in_current_problem] < bound.lb) {
      host_lb[var_idx_in_current_problem] = bound.lb;
      n_bounds_updated++;
    }
    if (host_ub[var_idx_in_current_problem] > bound.ub) {
      host_ub[var_idx_in_current_problem] = bound.ub;
      n_bounds_updated++;
    }
  }
}

template <typename i_t, typename f_t>
f_t probing_cache_t<i_t, f_t>::get_least_conflicting_rounding(problem_t<i_t, f_t>& problem,
                                                              std::vector<f_t>& host_lb,
                                                              std::vector<f_t>& host_ub,
                                                              i_t var_id_on_problem,
                                                              f_t first_probe,
                                                              f_t second_probe,
                                                              f_t integrality_tolerance)
{
  // get the var id where the probing cache was computed
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
    n_conflicting_vars = check_number_of_conflicting_vars(host_lb,
                                                          host_ub,
                                                          cache_row[hit_interval_for_first_probe],
                                                          integrality_tolerance,
                                                          problem.reverse_original_ids);
    if (n_conflicting_vars == 0) {
      CUOPT_LOG_TRACE("No conflicting vars, returning first probe");
      update_bounds_with_selected(
        host_lb, host_ub, cache_row[hit_interval_for_first_probe], problem.reverse_original_ids);
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
    check_number_of_conflicting_vars(host_lb,
                                     host_ub,
                                     cache_row[other_interval_idx],
                                     integrality_tolerance,
                                     problem.reverse_original_ids);

  if (n_conflicting_vars_other_probe < n_conflicting_vars) {
    CUOPT_LOG_TRACE(
      "For probing var %d with value %f better conflicting vars found %d in the other probing "
      "region (cache interval)!",
      var_id,
      first_probe,
      n_conflicting_vars_other_probe);
    update_bounds_with_selected(
      host_lb, host_ub, cache_row[other_interval_idx], problem.reverse_original_ids);
    if (other_interval_idx == hit_interval_for_second_probe) {
      CUOPT_LOG_TRACE("Better value on second probe val %f", second_probe);
      return second_probe;
    } else {
      CUOPT_LOG_TRACE("Better value on other interval cutoff %f",
                      cache_row[other_interval_idx].val_interval.val);
      return cache_row[other_interval_idx].val_interval.val;
    }
  }
  update_bounds_with_selected(
    host_lb, host_ub, cache_row[hit_interval_for_first_probe], problem.reverse_original_ids);
  return first_probe;
}

template <typename i_t, typename f_t>
bool probing_cache_t<i_t, f_t>::contains(problem_t<i_t, f_t>& problem, i_t var_id)
{
  return probing_cache.count(problem.original_ids[var_id]) > 0;
}

template <typename i_t, typename f_t, typename f_t2>
void inline insert_current_probing_to_cache(i_t var_idx,
                                            const val_interval_t<i_t, f_t>& probe_val,
                                            bound_presolve_t<i_t, f_t>& bound_presolve,
                                            const std::vector<f_t2>& original_bounds,
                                            const std::vector<f_t>& modified_lb,
                                            const std::vector<f_t>& modified_ub,
                                            const std::vector<i_t>& h_integer_indices,
                                            std::atomic<size_t>& n_implied_singletons)
{
  f_t int_tol = bound_presolve.context.settings.tolerances.integrality_tolerance;

  cache_entry_t<i_t, f_t> cache_item;
  cache_item.val_interval = probe_val;
  for (auto impacted_var_idx : h_integer_indices) {
    auto original_var_bounds = original_bounds[impacted_var_idx];
    if (get_lower(original_var_bounds) != modified_lb[impacted_var_idx] ||
        get_upper(original_var_bounds) != modified_ub[impacted_var_idx]) {
      if (integer_equal<f_t>(
            modified_lb[impacted_var_idx], modified_ub[impacted_var_idx], int_tol)) {
        ++n_implied_singletons;
      }
      cuopt_assert(modified_lb[impacted_var_idx] >= get_lower(original_var_bounds),
                   "Lower bound must be greater than or equal to original lower bound");
      cuopt_assert(modified_ub[impacted_var_idx] <= get_upper(original_var_bounds),
                   "Upper bound must be less than or equal to original upper bound");
      cached_bound_t<f_t> new_bound{modified_lb[impacted_var_idx], modified_ub[impacted_var_idx]};
      cache_item.var_to_cached_bound_map.insert({impacted_var_idx, new_bound});
    }
  }
  {
    std::lock_guard<std::mutex> lock(bound_presolve.probing_cache.probing_cache_mutex);
    if (!bound_presolve.probing_cache.probing_cache.count(var_idx) > 0) {
      std::array<cache_entry_t<i_t, f_t>, 2> entries_per_var;
      entries_per_var[0] = cache_item;
      bound_presolve.probing_cache.probing_cache.insert({var_idx, entries_per_var});
    } else {
      bound_presolve.probing_cache.probing_cache[var_idx][1] = cache_item;
    }
  }
}

template <typename i_t, typename f_t>
__global__ void compute_min_slack_per_var(typename problem_t<i_t, f_t>::view_t pb,
                                          raft::device_span<f_t> min_activity,
                                          raft::device_span<f_t> max_activity,
                                          raft::device_span<f_t> var_slack,
                                          raft::device_span<bool> different_coefficient,
                                          raft::device_span<f_t> max_excess_per_var,
                                          raft::device_span<i_t> max_n_violated_per_constraint)
{
  i_t var_idx           = pb.integer_indices[blockIdx.x];
  i_t var_offset        = pb.reverse_offsets[var_idx];
  i_t var_degree        = pb.reverse_offsets[var_idx + 1] - var_offset;
  f_t th_var_unit_slack = std::numeric_limits<f_t>::max();
  auto var_bounds       = pb.variable_bounds[var_idx];
  f_t lb                = get_lower(var_bounds);
  f_t ub                = get_upper(var_bounds);
  f_t first_coeff       = pb.reverse_coefficients[var_offset];
  bool different_coeff  = false;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto a = pb.reverse_coefficients[var_offset + i];
    if (std::signbit(a) != std::signbit(first_coeff)) { different_coeff = true; }
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    auto min_a    = min_activity[cnst_idx];
    auto max_a    = max_activity[cnst_idx];
    auto cnstr_ub = pb.constraint_upper_bounds[cnst_idx];
    auto cnstr_lb = pb.constraint_lower_bounds[cnst_idx];
    min_a -= (a < 0) ? a * ub : a * lb;
    auto delta_min_act = cnstr_ub - min_a;
    th_var_unit_slack  = min(th_var_unit_slack, (delta_min_act / a));
    max_a -= (a > 0) ? a * ub : a * lb;
    auto delta_max_act = cnstr_lb - max_a;
    th_var_unit_slack  = min(th_var_unit_slack, (delta_max_act / a));
    // if (var_idx == 0) {
    //   printf("\ncmp_min_slack cnst %d\n diff %f %f\n cnstr_ub %f min_a %f delta_min %f\n cnstr_lb
    //   %f max_a %f delta_max %f\n", cnst_idx,
    //       (a < 0) ? a * ub : a * lb,
    //       (a > 0) ? a * ub : a * lb,
    //       cnstr_ub, min_a, delta_min_act,
    //       cnstr_lb, max_a, delta_max_act);
    // }
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
    auto a        = pb.reverse_coefficients[var_offset + i];
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    auto min_a    = min_activity[cnst_idx];
    auto max_a    = max_activity[cnst_idx];
    auto cnstr_ub = pb.constraint_upper_bounds[cnst_idx];
    auto cnstr_lb = pb.constraint_lower_bounds[cnst_idx];
    min_a -= (a < 0) ? a * ub : a * lb;
    f_t var_max_act = (a > 0) ? a * ub : a * lb;
    f_t excess      = max(0., min_a + var_max_act - cnstr_ub);
    if (excess > 0) {
      th_max_excess = max(th_max_excess, excess);
      th_n_of_excess++;
    }
    // now add max activity of this var to see the excess
    max_a -= (a > 0) ? a * ub : a * lb;
    f_t var_min_act = (a < 0) ? a * ub : a * lb;
    excess          = max(0., cnstr_lb - (max_a + var_min_act));
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

// computes variables that appear in multiple constraints with different signs
// which means that min activity contribution in one constraint will not be valid in another
// constraint we will sort them by the violation rooted from the conflicting bounds. an example: lb:
// 0 ub: 5 cstr_1 coeff : -1  cstr_2 coeff: 1 min activity val in cstr_1 is 5 and 0 in cstr_2, they
// cannot happen at the same time we extract those variables and then sort it by the sum of
// excesses(or slack) in all constraints by setting to lb and ub
template <typename i_t, typename f_t>
inline std::vector<i_t> compute_prioritized_integer_indices(
  bound_presolve_t<i_t, f_t>& bound_presolve, problem_t<i_t, f_t>& problem)
{
  // sort the variables according to the min slack they have across constraints
  // we also need to consider the variable range
  // the priority is computed as the var_range * min_slack
  // min_slack is computed as var_range*coefficient/(b - min_act)
  rmm::device_uvector<f_t> min_slack_per_var(problem.n_integer_vars,
                                             problem.handle_ptr->get_stream());
  rmm::device_uvector<i_t> priority_indices(problem.integer_indices,
                                            problem.handle_ptr->get_stream());
  rmm::device_uvector<bool> different_coefficient(problem.n_integer_vars,
                                                  problem.handle_ptr->get_stream());
  rmm::device_uvector<f_t> max_excess_per_var(problem.n_integer_vars,
                                              problem.handle_ptr->get_stream());
  rmm::device_uvector<i_t> max_n_violated_per_constraint(problem.n_integer_vars,
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
  bound_presolve.calculate_activity_on_problem_bounds(problem);
  bool res = bound_presolve.calculate_infeasible_redundant_constraints(problem);
  cuopt_assert(res, "The activity computation must be feasible during probing cache!");
  CUOPT_LOG_DEBUG("prioritized integer_indices n_integer_vars %d", problem.n_integer_vars);
  // compute the min var slack
  compute_min_slack_per_var<i_t, f_t>
    <<<problem.n_integer_vars, 128, 0, problem.handle_ptr->get_stream()>>>(
      problem.view(),
      make_span(bound_presolve.upd.min_activity),
      make_span(bound_presolve.upd.max_activity),
      make_span(min_slack_per_var),
      make_span(different_coefficient),
      make_span(max_excess_per_var),
      make_span(max_n_violated_per_constraint));
  auto iterator = thrust::make_zip_iterator(thrust::make_tuple(
    max_n_violated_per_constraint.begin(), max_excess_per_var.begin(), min_slack_per_var.begin()));
  // sort the vars
  thrust::sort_by_key(problem.handle_ptr->get_thrust_policy(),
                      iterator,
                      iterator + problem.n_integer_vars,
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
  auto h_priority_indices = host_copy(priority_indices, problem.handle_ptr->get_stream());
  problem.handle_ptr->sync_stream();
  return h_priority_indices;
}

template <typename i_t, typename f_t, typename f_t2>
void compute_cache_for_var(i_t var_idx,
                           bound_presolve_t<i_t, f_t>& bound_presolve,
                           problem_t<i_t, f_t>& problem,
                           multi_probe_t<i_t, f_t>& multi_probe_presolve,
                           const std::vector<f_t2>& h_var_bounds,
                           const std::vector<i_t>& h_integer_indices,
                           std::atomic<size_t>& n_of_implied_singletons,
                           std::atomic<size_t>& n_of_cached_probings,
                           std::atomic<bool>& problem_is_infeasible,
                           std::vector<std::tuple<f_t, i_t, f_t, f_t>>& modification_vector,
                           std::vector<substitution_t<i_t, f_t>>& substitution_vector,
                           timer_t timer,
                           i_t device_id)
{
  RAFT_CUDA_TRY(cudaSetDevice(device_id));
  // test if we need per thread handle
  raft::handle_t handle{};
  std::vector<f_t> h_improved_lower_bounds_0(h_var_bounds.size());
  std::vector<f_t> h_improved_upper_bounds_0(h_var_bounds.size());
  std::vector<f_t> h_improved_lower_bounds_1(h_var_bounds.size());
  std::vector<f_t> h_improved_upper_bounds_1(h_var_bounds.size());
  std::pair<val_interval_t<i_t, f_t>, val_interval_t<i_t, f_t>> probe_vals;
  auto bounds = h_var_bounds[var_idx];
  f_t lb      = get_lower(bounds);
  f_t ub      = get_upper(bounds);
  // note that is_binary does not always mean the bound difference is one
  bool is_binary = ub == 1 && lb == 0;
  for (i_t i = 0; i < 2; ++i) {
    auto& probe_val = i == 0 ? probe_vals.first : probe_vals.second;
    // if binary, probe both values
    if (problem.integer_equal(ub - lb, 1.)) {
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
  }
  std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>> var_interval_vals;
  std::get<0>(var_interval_vals) = var_idx;
  for (i_t i = 0; i < 2; ++i) {
    auto& probe_val = i == 0 ? probe_vals.first : probe_vals.second;
    // first(index 1) item of tuple is the first interval, the second is the second interval
    auto& bounds = i == 0 ? std::get<1>(var_interval_vals) : std::get<2>(var_interval_vals);
    // now solve bounds presolve for the value or the interval
    // if the type is equals, just set the value and solve the bounds presolve
    if (probe_val.interval_type == interval_type_t::EQUALS) {
      bounds.first  = probe_val.val;
      bounds.second = probe_val.val;
    }
    // if it is an interval change the variable bound and solve
    else {
      if (probe_val.interval_type == interval_type_t::LEQ) {
        bounds.first  = lb;
        bounds.second = probe_val.val;
      } else {
        bounds.first  = probe_val.val;
        bounds.second = ub;
      }
    }
  }
  auto bounds_presolve_result =
    multi_probe_presolve.solve_for_interval(problem, var_interval_vals, &handle);
  if (bounds_presolve_result != termination_criterion_t::NO_UPDATE) {
    CUOPT_LOG_TRACE("Adding cached bounds for var %d", var_idx);
  }
  i_t n_of_infeasible_probings = 0;
  i_t valid_host_bounds        = 0;
  for (i_t i = 0; i < 2; ++i) {
    if (multi_probe_presolve.infeas_constraints_count_0 > 0 &&
        multi_probe_presolve.infeas_constraints_count_1 > 0) {
      problem_is_infeasible.store(true);
      return;
    }
    i_t infeas_constraints_count  = i == 0 ? multi_probe_presolve.infeas_constraints_count_0
                                           : multi_probe_presolve.infeas_constraints_count_1;
    const auto& probe_val         = i == 0 ? probe_vals.first : probe_vals.second;
    auto& h_improved_lower_bounds = i == 0 ? h_improved_lower_bounds_0 : h_improved_lower_bounds_1;
    auto& h_improved_upper_bounds = i == 0 ? h_improved_upper_bounds_0 : h_improved_upper_bounds_1;
    if (infeas_constraints_count > 0) {
      CUOPT_LOG_TRACE("Var %d is infeasible for probe %d on value %f. Fixing other interval",
                      var_idx,
                      i,
                      probe_val.val);
      const auto other_probe_val = i == 0 ? probe_vals.second : probe_vals.first;
      const auto other_probe_interval_type =
        i == 0 ? probe_vals.second.interval_type : probe_vals.first.interval_type;
      // current probe is infeasible, remove the current var bound from the bounds
      if (other_probe_interval_type == interval_type_t::EQUALS) {
        modification_vector.emplace_back(
          timer.elapsed_time(), var_idx, other_probe_val.val, other_probe_val.val);
      } else if (other_probe_interval_type == interval_type_t::GEQ) {
        modification_vector.emplace_back(
          timer.elapsed_time(), var_idx, other_probe_val.val, bounds.y);
      } else {
        modification_vector.emplace_back(
          timer.elapsed_time(), var_idx, bounds.x, other_probe_val.val);
      }
      n_of_infeasible_probings++;
      continue;
    }
    // this only tracks the number of variable intervals that have cached bounds
    n_of_cached_probings++;
    // save the impacted bounds
    if (bounds_presolve_result != termination_criterion_t::NO_UPDATE) {
      valid_host_bounds++;
      auto& d_lb = i == 0 ? multi_probe_presolve.upd_0.lb : multi_probe_presolve.upd_1.lb;
      auto& d_ub = i == 0 ? multi_probe_presolve.upd_0.ub : multi_probe_presolve.upd_1.ub;
      raft::copy(h_improved_lower_bounds.data(),
                 d_lb.data(),
                 h_improved_lower_bounds.size(),
                 handle.get_stream());
      raft::copy(h_improved_upper_bounds.data(),
                 d_ub.data(),
                 h_improved_upper_bounds.size(),
                 handle.get_stream());
      insert_current_probing_to_cache(var_idx,
                                      probe_val,
                                      bound_presolve,
                                      h_var_bounds,
                                      h_improved_lower_bounds,
                                      h_improved_upper_bounds,
                                      h_integer_indices,
                                      n_of_implied_singletons);
    }
  }
  // when both probes are feasible, we can infer some global bounds
  if (n_of_infeasible_probings == 0 && valid_host_bounds == 2) {
    // TODO do the check in parallel
    for (size_t i = 0; i < h_improved_lower_bounds_0.size(); i++) {
      if (i == (size_t)var_idx) { continue; }
      f_t lower_bound = min(h_improved_lower_bounds_0[i], h_improved_lower_bounds_1[i]);
      f_t upper_bound = max(h_improved_upper_bounds_0[i], h_improved_upper_bounds_1[i]);
      cuopt_assert(h_var_bounds[i].x <= lower_bound, "lower bound violation");
      cuopt_assert(h_var_bounds[i].y >= upper_bound, "upper bound violation");
      // check why we might have invalid lower and upper bound here
      if (h_var_bounds[i].x < lower_bound || h_var_bounds[i].y > upper_bound) {
        modification_vector.emplace_back(timer.elapsed_time(), i, lower_bound, upper_bound);
        CUOPT_LOG_TRACE(
          "Var %d global bounds inferred from probing new bounds: [%f, %f] old bounds: [%f, %f]",
          i,
          lower_bound,
          upper_bound,
          h_var_bounds[i].x,
          h_var_bounds[i].y);
      }
      f_t int_tol = bound_presolve.context.settings.tolerances.integrality_tolerance;
      if (integer_equal<f_t>(h_improved_lower_bounds_0[i], h_improved_upper_bounds_0[i], int_tol) &&
          integer_equal<f_t>(h_improved_lower_bounds_1[i], h_improved_upper_bounds_1[i], int_tol) &&
          is_binary) {
        // == case has been handled as fixing by the global bounds update
        if (!integer_equal<f_t>(
              h_improved_lower_bounds_0[i], h_improved_lower_bounds_1[i], int_tol)) {
          // trivial presolve handles eliminations
          // x_i = l_0 + (l_1 - l_0) * x_var_idx
          // this means
          CUOPT_LOG_TRACE("Variable substitution found for var %d", i);
          substitution_t<i_t, f_t> substitution;
          substitution.timestamp        = timer.elapsed_time();
          substitution.substituted_var  = i;
          substitution.substituting_var = var_idx;
          substitution.offset           = h_improved_lower_bounds_0[i];
          substitution.coefficient = h_improved_lower_bounds_1[i] - h_improved_lower_bounds_0[i];
          substitution_vector.emplace_back(substitution);
        }
      }
    }
  }
  handle.sync_stream();
}

template <typename i_t, typename f_t>
void apply_modification_queue_to_problem(
  std::vector<std::vector<std::tuple<f_t, i_t, f_t, f_t>>>& modification_vector_pool,
  problem_t<i_t, f_t>& problem)
{
  // since each thread has its own deterministic chunk and the order of insertion here is
  // deterministic this should be deterministic
  std::unordered_map<i_t, std::pair<f_t, f_t>> var_bounds_modifications;
  for (const auto& modification_vector : modification_vector_pool) {
    for (const auto& modification : modification_vector) {
      auto [time, var_idx, lb, ub] = modification;
      if (var_bounds_modifications.count(var_idx) == 0) {
        var_bounds_modifications[var_idx] = std::make_pair(lb, ub);
      } else {
        var_bounds_modifications[var_idx].first = max(var_bounds_modifications[var_idx].first, lb);
        var_bounds_modifications[var_idx].second =
          min(var_bounds_modifications[var_idx].second, ub);
      }
    }
  }
  std::vector<i_t> var_indices;
  std::vector<f_t> lb_values;
  std::vector<f_t> ub_values;
  for (const auto& [var_idx, modifications] : var_bounds_modifications) {
    var_indices.push_back(var_idx);
    lb_values.push_back(modifications.first);
    ub_values.push_back(modifications.second);
  }
  if (var_indices.size() > 0) {
    problem.update_variable_bounds(var_indices, lb_values, ub_values);
    CUOPT_LOG_DEBUG("Updated %d variable bounds", var_indices.size());
  }
}

// Ensures that if A subs B and B subs A, we only keep one deterministic direction.
template <typename i_t, typename f_t>
void sanitize_graph(
  std::unordered_map<i_t, std::vector<std::pair<i_t, substitution_t<i_t, f_t>>>>& all_substitutions)
{
  for (auto& substitution : all_substitutions) {
    auto& substituting_var = substitution.first;
    auto& list             = substitution.second;
    // Use remove_if with a lambda to clean up the vector in-place
    auto it = std::remove_if(
      list.begin(), list.end(), [&](const std::pair<i_t, substitution_t<i_t, f_t>>& item) {
        i_t substituted_var = item.first;
        // Check if the reverse edge exists, it should exists because of the nature of probing
        if (all_substitutions.count(substituted_var)) {
          const auto& reverse_list = all_substitutions[substituted_var];
          for (const auto& reverse_item : reverse_list) {
            if (reverse_item.first == substituting_var) {
              // Bidirectional edge detected!
              // Keep edge only if substituting_var < substituted_var.
              if (substituting_var > substituted_var) {
                CUOPT_LOG_TRACE("Removing cycle edge: %d -> %d (keeping %d -> %d)",
                                substituting_var,
                                substituted_var,
                                substituted_var,
                                substituting_var);
                return true;  // delete the edge
              }
            }
          }
        }
        return false;  // keep the edge
      });

    list.erase(it, list.end());
  }
}

template <typename i_t, typename f_t>
void dfs(
  std::unordered_map<i_t, std::vector<std::pair<i_t, substitution_t<i_t, f_t>>>>& all_substitutions,
  std::unordered_set<i_t>& visited,
  const substitution_t<i_t, f_t>& parent_substitution,
  i_t curr_var)
{
  // If we have already processed this node in the current traversal.
  if (visited.count(curr_var)) return;
  visited.insert(curr_var);

  // If 'curr_var' itself substitutes others, we must propagate the parent's substitution down.
  if (all_substitutions.count(curr_var)) {
    for (auto& [substituted_var_of_child, child_substitution] : all_substitutions[curr_var]) {
      // Parent: curr_var = P_offset + P_coeff * Root_Var
      // Child:  child_var = C_offset + C_coeff * curr_var
      // Result: child_var = C_offset + C_coeff * (P_offset + P_coeff * Root_Var)
      //                   = (C_offset + C_coeff * P_offset) + (C_coeff * P_coeff) * Root_Var
      child_substitution.offset =
        child_substitution.offset + child_substitution.coefficient * parent_substitution.offset;
      child_substitution.coefficient =
        child_substitution.coefficient * parent_substitution.coefficient;
      child_substitution.substituting_var = parent_substitution.substituting_var;
      CUOPT_LOG_TRACE("Merged: Var %d is now substituted by %d via %d",
                      substituted_var_of_child,
                      child_substitution.substituting_var,
                      curr_var);
      dfs(all_substitutions, visited, child_substitution, substituted_var_of_child);
    }
  }
}

template <typename i_t, typename f_t>
void merge_substitutions(
  std::unordered_map<i_t, std::vector<std::pair<i_t, substitution_t<i_t, f_t>>>>& all_substitutions)
{
  // Remove cycles (A->B and B->A) as probing always generates a pair of equivalent substitutions
  sanitize_graph(all_substitutions);

  // Identify Roots
  // A Root is a 'substituting' var that is never 'substituted' by anyone else.
  std::unordered_set<i_t> all_substituted_vars;
  for (const auto& [key, list] : all_substitutions) {
    for (const auto& item : list) {
      all_substituted_vars.insert(item.first);
    }
  }

  std::vector<i_t> roots;
  for (const auto& [key, list] : all_substitutions) {
    if (all_substituted_vars.find(key) == all_substituted_vars.end()) { roots.push_back(key); }
  }

  // Run DFS from every Root

  for (i_t root : roots) {
    // For the root, there is no "parent substitution".
    std::unordered_set<i_t> visited_in_this_path;
    visited_in_this_path.insert(root);
    for (auto& [substituted_var, substitution] : all_substitutions[root]) {
      // Pass the substitution connecting Root->Child as the "parent" for the next level
      dfs(all_substitutions, visited_in_this_path, substitution, substituted_var);
    }
  }
}

template <typename i_t, typename f_t>
void apply_substitution_queue_to_problem(
  std::vector<std::vector<substitution_t<i_t, f_t>>>& substitution_vector_pool,
  problem_t<i_t, f_t>& problem)
{
  std::unordered_map<i_t, std::vector<std::pair<i_t, substitution_t<i_t, f_t>>>> all_substitutions;

  for (const auto& substitution_vector : substitution_vector_pool) {
    for (const auto& substitution : substitution_vector) {
      all_substitutions[substitution.substituting_var].push_back(
        {substitution.substituted_var, substitution});
    }
  }

  // Flatten Graph
  merge_substitutions(all_substitutions);

  std::vector<i_t> var_indices;
  std::vector<i_t> substituting_var_indices;
  std::vector<f_t> offset_values;
  std::vector<f_t> coefficient_values;

  // Get variable_mapping to convert current indices to original indices
  auto h_variable_mapping =
    host_copy(problem.presolve_data.variable_mapping, problem.handle_ptr->get_stream());
  problem.handle_ptr->sync_stream();

  for (const auto& [substituting_var, substitutions] : all_substitutions) {
    for (const auto& [substituted_var, substitution] : substitutions) {
      CUOPT_LOG_TRACE("Applying substitution: %d -> %d",
                      substitution.substituting_var,
                      substitution.substituted_var);
      var_indices.push_back(substitution.substituted_var);
      substituting_var_indices.push_back(substitution.substituting_var);
      offset_values.push_back(substitution.offset);
      coefficient_values.push_back(substitution.coefficient);

      // Store substitution for post-processing (convert to original variable IDs)
      substitution_t<i_t, f_t> sub;
      sub.timestamp        = substitution.timestamp;
      sub.substituted_var  = h_variable_mapping[substitution.substituted_var];
      sub.substituting_var = h_variable_mapping[substitution.substituting_var];
      sub.offset           = substitution.offset;
      sub.coefficient      = substitution.coefficient;
      problem.presolve_data.variable_substitutions.push_back(sub);
      CUOPT_LOG_TRACE("Stored substitution for post-processing: x[%d] = %f + %f * x[%d]",
                      sub.substituted_var,
                      sub.offset,
                      sub.coefficient,
                      sub.substituting_var);
    }
  }

  if (!var_indices.empty()) {
    problem.substitute_variables(
      var_indices, substituting_var_indices, offset_values, coefficient_values);
  }
}

template <typename i_t, typename f_t>
std::vector<i_t> compute_priority_indices_by_implied_integers(problem_t<i_t, f_t>& problem)
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  auto input_transform_it   = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), [view = problem.view()] __device__(i_t idx) -> i_t {
      return view.is_integer_var(view.variables[idx]);
    });
  // keeps the number of constraints that contain integer variables
  rmm::device_uvector<i_t> num_int_vars_per_constraint(problem.n_constraints,
                                                       problem.handle_ptr->get_stream());
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                     temp_storage_bytes,
                                     input_transform_it,
                                     num_int_vars_per_constraint.data(),
                                     problem.n_constraints,
                                     problem.offsets.data(),
                                     problem.offsets.data() + 1,
                                     cuda::std::plus<>{},
                                     0,
                                     problem.handle_ptr->get_stream());

  rmm::device_uvector<std::uint8_t> temp_storage(temp_storage_bytes,
                                                 problem.handle_ptr->get_stream());
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                     temp_storage_bytes,
                                     input_transform_it,
                                     num_int_vars_per_constraint.data(),
                                     problem.n_constraints,
                                     problem.offsets.data(),
                                     problem.offsets.data() + 1,
                                     cuda::std::plus<>{},
                                     0,
                                     problem.handle_ptr->get_stream());
  // keeps the count of number of other integers that this variables shares a constraint with
  rmm::device_uvector<i_t> count_per_variable(problem.n_variables,
                                              problem.handle_ptr->get_stream());
  auto input_transform_it_2 = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [num_int_vars_per_constraint = make_span(num_int_vars_per_constraint),
     view                        = problem.view()] __device__(i_t idx) -> i_t {
      return num_int_vars_per_constraint[view.reverse_constraints[idx]];
    });
  // run second reduction operation, reset sizes so query works correctly
  d_temp_storage     = nullptr;
  temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                     temp_storage_bytes,
                                     input_transform_it_2,
                                     count_per_variable.data(),
                                     problem.n_variables,
                                     problem.reverse_offsets.data(),
                                     problem.reverse_offsets.data() + 1,
                                     cuda::std::plus<>{},
                                     0,
                                     problem.handle_ptr->get_stream());

  temp_storage.resize(temp_storage_bytes, problem.handle_ptr->get_stream());
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                     temp_storage_bytes,
                                     input_transform_it_2,
                                     count_per_variable.data(),
                                     problem.n_variables,
                                     problem.reverse_offsets.data(),
                                     problem.reverse_offsets.data() + 1,
                                     cuda::std::plus<>{},
                                     0,
                                     problem.handle_ptr->get_stream());
  thrust::for_each(problem.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(problem.n_variables),
                   [count_per_variable = make_span(count_per_variable),
                    view               = problem.view()] __device__(i_t idx) {
                     if (!view.is_integer_var(idx)) { count_per_variable[idx] = 0; }
                   });
  rmm::device_uvector<i_t> priority_indices(problem.n_variables, problem.handle_ptr->get_stream());
  thrust::sequence(
    problem.handle_ptr->get_thrust_policy(), priority_indices.begin(), priority_indices.end());
  thrust::sort_by_key(problem.handle_ptr->get_thrust_policy(),
                      count_per_variable.data(),
                      count_per_variable.data() + problem.n_variables,
                      priority_indices.data(),
                      thrust::greater<i_t>());
  auto h_priority_indices = host_copy(priority_indices, problem.handle_ptr->get_stream());
  // Find the index of the first 0 element in count_per_variable
  auto first_zero_it      = thrust::lower_bound(problem.handle_ptr->get_thrust_policy(),
                                           count_per_variable.begin(),
                                           count_per_variable.end(),
                                           0,
                                           thrust::greater<i_t>());
  size_t first_zero_index = (first_zero_it != count_per_variable.end())
                              ? std::distance(count_per_variable.begin(), first_zero_it)
                              : count_per_variable.size();
  h_priority_indices.erase(h_priority_indices.begin() + first_zero_index, h_priority_indices.end());
  return h_priority_indices;
}

template <typename i_t, typename f_t>
bool compute_probing_cache(bound_presolve_t<i_t, f_t>& bound_presolve,
                           problem_t<i_t, f_t>& problem,
                           timer_t timer)
{
  raft::common::nvtx::range fun_scope("compute_probing_cache");
  // we dont want to compute the probing cache for all variables for time and computation resources
  auto priority_indices = compute_priority_indices_by_implied_integers(problem);
  CUOPT_LOG_DEBUG("Computing probing cache");
  auto stream            = problem.handle_ptr->get_stream();
  auto h_integer_indices = host_copy(problem.integer_indices, stream);
  auto h_var_bounds      = host_copy(problem.variable_bounds, stream);
  // TODO adjust the iteration limit depending on the total time limit and time it takes for single
  // var
  bound_presolve.settings.iteration_limit = 50;
  bound_presolve.settings.time_limit      = timer.remaining_time();

  size_t num_threads = bound_presolve.settings.num_threads < 0
                         ? 0.2 * omp_get_max_threads()
                         : bound_presolve.settings.num_threads;
  num_threads        = std::clamp<size_t>(num_threads, 1, 8);

  // Create a vector of multi_probe_t objects
  std::vector<multi_probe_t<i_t, f_t>> multi_probe_presolve_pool;
  std::vector<std::vector<std::tuple<f_t, i_t, f_t, f_t>>> modification_vector_pool(num_threads);
  std::vector<std::vector<substitution_t<i_t, f_t>>> substitution_vector_pool(num_threads);

  // Initialize multi_probe_presolve_pool
  for (size_t i = 0; i < num_threads; i++) {
    multi_probe_presolve_pool.emplace_back(bound_presolve.context);
    multi_probe_presolve_pool[i].resize(problem);
    multi_probe_presolve_pool[i].compute_stats = true;
  }

  // Atomic variables for tracking progress
  std::atomic<size_t> n_of_implied_singletons(0);
  std::atomic<size_t> n_of_cached_probings(0);
  std::atomic<bool> problem_is_infeasible(false);
  size_t last_it_implied_singletons = 0;
  bool early_exit                   = false;
  const size_t step_size            = min((size_t)2048, priority_indices.size());

// Main parallel loop
#pragma omp parallel num_threads(num_threads)
  {
    for (size_t step_start = 0; step_start < priority_indices.size(); step_start += step_size) {
      if (timer.check_time_limit() || early_exit || problem_is_infeasible.load()) { break; }
      size_t step_end = std::min(step_start + step_size, priority_indices.size());

#pragma omp for
      for (size_t i = step_start; i < step_end; ++i) {
        auto var_idx = priority_indices[i];
        if (timer.check_time_limit()) { continue; }

        int thread_idx = omp_get_thread_num();
        CUOPT_LOG_TRACE("Computing probing cache for var %d on thread %d", var_idx, thread_idx);

        auto& multi_probe_presolve = multi_probe_presolve_pool[thread_idx];

        compute_cache_for_var<i_t, f_t>(var_idx,
                                        bound_presolve,
                                        problem,
                                        multi_probe_presolve,
                                        h_var_bounds,
                                        h_integer_indices,
                                        n_of_implied_singletons,
                                        n_of_cached_probings,
                                        problem_is_infeasible,
                                        modification_vector_pool[thread_idx],
                                        substitution_vector_pool[thread_idx],
                                        timer,
                                        problem.handle_ptr->get_device());
      }
    }
#pragma omp single
    {
      // TODO when we have determinism, check current threads work/time counter and filter queue
      // items that are smaller or equal to that
      apply_modification_queue_to_problem(modification_vector_pool, problem);
      // copy host bounds again, because we changed some problem bounds
      raft::copy(h_var_bounds.data(),
                 problem.variable_bounds.data(),
                 h_var_bounds.size(),
                 problem.handle_ptr->get_stream());
      problem.handle_ptr->sync_stream();
      if (n_of_implied_singletons - last_it_implied_singletons <
          (size_t)std::max(2, (min(100, problem.n_variables / 50)))) {
        early_exit = true;
      }
      last_it_implied_singletons = n_of_implied_singletons;
    }
  }  // end of step
  apply_substitution_queue_to_problem(substitution_vector_pool, problem);
  CUOPT_LOG_DEBUG("Total number of cached probings %lu number of implied singletons %lu",
                  n_of_cached_probings.load(),
                  n_of_implied_singletons.load());
  // restore the settings
  bound_presolve.settings = {};
  return problem_is_infeasible.load();
}

#define INSTANTIATE(F_TYPE)                                                                        \
  template bool compute_probing_cache<int, F_TYPE>(bound_presolve_t<int, F_TYPE> & bound_presolve, \
                                                   problem_t<int, F_TYPE> & problem,               \
                                                   timer_t timer);                                 \
  template class probing_cache_t<int, F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
