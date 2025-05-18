/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/mip_constants.hpp>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>
#include "lb_constraint_prop.cuh"
#include "simple_rounding.cuh"

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_constraint_prop_t<i_t, f_t>::lb_constraint_prop_t(mip_solver_context_t<i_t, f_t>& context_)
  : context(context_),
    temp_problem(*context.problem_ptr),
    bounds_update(temp_problem, context),
    bounds_repair(context.problem_ptr->handle_ptr),
    unset_vars(context.problem_ptr->n_variables, context.problem_ptr->handle_ptr->get_stream()),
    temp_assignment(context.problem_ptr->n_variables,
                    context.problem_ptr->handle_ptr->get_stream()),
    bounds_restore(2 * context.problem_ptr->n_variables,
                   context.problem_ptr->handle_ptr->get_stream()),
    assignment_restore(context.problem_ptr->n_variables,
                       context.problem_ptr->handle_ptr->get_stream()),
    rng(cuopt::seed_generator::get_seed(), 0, 0)
{
}

constexpr int n_subsections          = 3 * 7;
constexpr size_t size_of_subsections = n_subsections + 1;

template <typename i_t, typename f_t>
__device__ void assign_offsets(
  raft::device_span<i_t> offsets, i_t category, i_t idx, f_t frac_1, f_t frac_2)
{
  if (frac_1 <= 0.02 && frac_2 > 0.02) {
    offsets[category * 7 + 1] = idx + 1;
  } else if (frac_1 <= 0.05 && frac_2 > 0.05) {
    offsets[category * 7 + 2] = idx + 1;
  } else if (frac_1 <= 0.1 && frac_2 > 0.1) {
    offsets[category * 7 + 3] = idx + 1;
  } else if (frac_1 <= 0.2 && frac_2 > 0.2) {
    offsets[category * 7 + 4] = idx + 1;
  } else if (frac_1 <= 0.3 && frac_2 > 0.3) {
    offsets[category * 7 + 5] = idx + 1;
  } else if (frac_1 <= 0.4 && frac_2 > 0.4) {
    offsets[category * 7 + 6] = idx + 1;
  }
}

// TODO verify this logic
template <typename i_t, typename f_t>
__device__ bool round_val_on_singleton_and_crossing(
  f_t& assign, f_t v_lb, f_t v_ub, f_t o_lb, f_t o_ub)
{
  if (v_lb == v_ub) {
    assign = floor(v_lb + 0.5);
    return true;
  } else if (v_ub <= o_lb && v_lb <= o_ub) {
    assign = floor(v_lb + 0.5);
    return true;
  } else if (v_ub <= o_lb && v_lb >= o_ub) {
    if (!isfinite(o_lb)) {
      assign = ceil(o_ub - 0.5);
    } else if (!isfinite(o_ub)) {
      assign = floor(o_lb + 0.5);
    } else {
      assign = round((o_lb + o_ub) / 2);
    }
    return true;
  } else if (v_lb >= o_ub && v_ub >= o_lb) {
    assign = ceil(v_ub - 0.5);
    return true;
  }
  // if all cases fail
  else if (v_lb > v_ub) {
    if (!isfinite(o_lb)) {
      assign = ceil(o_ub - 0.5);
    } else if (!isfinite(o_ub)) {
      assign = floor(o_lb + 0.5);
    } else {
      assign = round((o_lb + o_ub) / 2);
    }
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, typename f_t2>
struct is_bound_fixed_t {
  // This functor should be called only on integer variables
  f_t eps;
  raft::device_span<f_t2> var_bnd;
  raft::device_span<f_t> original_lb;
  raft::device_span<f_t> original_ub;
  raft::device_span<f_t> assignment;
  is_bound_fixed_t(f_t eps_,
                   raft::device_span<f_t2> var_bnd_,
                   raft::device_span<f_t> original_lb_,
                   raft::device_span<f_t> original_ub_,
                   raft::device_span<f_t> assignment_)
    : eps(eps_),
      var_bnd(var_bnd_),
      original_lb(original_lb_),
      original_ub(original_ub_),
      assignment(assignment_)
  {
  }

  HDI bool operator()(i_t idx)
  {
    auto bnd  = var_bnd[idx];
    auto o_lb = original_lb[idx];
    auto o_ub = original_ub[idx];
    bool is_singleton =
      round_val_on_singleton_and_crossing<i_t, f_t>(assignment[idx], bnd.x, bnd.y, o_lb, o_ub);
    return is_singleton;
  }
};

template <typename i_t, typename f_t>
bool lb_constraint_prop_t<i_t, f_t>::probe(
  rmm::device_uvector<f_t>& assignment,
  load_balanced_problem_t<i_t, f_t>* problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
  problem_t<i_t, f_t>* original_problem,
  const std::vector<thrust::pair<i_t, f_t>>& var_probe_val_pairs,
  size_t* set_count_ptr,
  rmm::device_uvector<i_t>& unset_vars)
{
  const f_t int_tol          = original_problem->tolerances.integrality_tolerance;
  auto set_count             = *set_count_ptr;
  const bool use_host_bounds = true;
  lb_bounds_update.solve(var_probe_val_pairs, use_host_bounds);
  // if we are ii at this point, backtrack the number of variables we have set in this given
  // interval then start setting one by one
  // if we determined that the rounding is ii then don't do any recovery and finish ronuding
  // quickly
  bool bounds_update_ii = lb_bounds_update.infeas_constraints_count > 0;
  if (!recovery_mode && !rounding_ii && bounds_update_ii && bounds_prop_interval != 1) {
    CUOPT_LOG_DEBUG("Activating recovery mode in bounds prop: bounds_prop_interval %d n_ii cstr %d",
                    bounds_prop_interval,
                    lb_bounds_update.infeas_constraints_count);
    // do backtracking
    recovery_mode                             = true;
    lb_bounds_update.infeas_constraints_count = 0;
    n_iter_in_recovery                        = 0;
    return false;
  }
  lb_bounds_update.set_updated_bounds(problem);

  using f_t2 = typename type_2<f_t>::type;
  // which other variables were affected?
  auto iter = thrust::stable_partition(
    original_problem->handle_ptr->get_thrust_policy(),
    unset_vars.begin() + set_count,
    unset_vars.end(),
    is_bound_fixed_t<i_t, f_t, f_t2>{int_tol,
                                     make_span_2(problem->variable_bounds),
                                     make_span(original_problem->variable_lower_bounds),
                                     make_span(original_problem->variable_upper_bounds),
                                     make_span(assignment)});
  i_t n_fixed_vars = (iter - (unset_vars.begin() + set_count));
  cuopt_assert(n_fixed_vars >= var_probe_val_pairs.size(), "Error in number of vars fixed!");
  set_count += n_fixed_vars;
  CUOPT_LOG_TRACE("Set var count increased from %d to %d", *set_count_ptr, set_count);
  *set_count_ptr = set_count;
  return lb_bounds_update.infeas_constraints_count == 0;
}

template <typename i_t, typename f_t>
std::tuple<f_t, f_t, f_t> lb_constraint_prop_t<i_t, f_t>::probing_values(
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
  const solution_t<i_t, f_t>& orig_sol,
  i_t idx)
{
  auto v_lb    = lb_bounds_update.host_bounds[2 * idx];
  auto v_ub    = lb_bounds_update.host_bounds[2 * idx + 1];
  auto var_val = curr_host_assignment[idx];

  const f_t int_tol  = orig_sol.problem_ptr->tolerances.integrality_tolerance;
  auto eps           = int_tol;
  auto within_bounds = (v_lb - eps <= var_val) && (var_val <= v_ub + eps);
  // if it is a collapsed var, return immediately one of the bounds
  if (orig_sol.problem_ptr->integer_equal(v_lb, v_ub)) {
    return std::make_tuple(v_lb, var_val, v_lb);
  }
  if (within_bounds) {
    // the value might have been brought within the bounds when it was out of bounds
    f_t first_round_val = round_nearest(var_val, v_lb, v_ub, int_tol, rng);
    f_t second_round_val;
    auto v_f = std::floor(first_round_val - int_tol);
    auto v_c = std::ceil(first_round_val + int_tol);
    if (first_round_val - var_val >= 0) {
      second_round_val = v_f;
      bool floor_within_bounds =
        (v_lb - eps <= second_round_val) && (second_round_val <= v_ub + eps);
      if (!floor_within_bounds) { second_round_val = v_c; }
    } else {
      second_round_val = v_c;
      bool ceil_within_bounds =
        (v_lb - eps <= second_round_val) && (second_round_val <= v_ub + eps);
      if (!ceil_within_bounds) { second_round_val = v_f; }
    }

    cuopt_assert(v_lb <= first_round_val && first_round_val <= v_ub, "probing value out of bounds");
    cuopt_assert(v_lb <= second_round_val && second_round_val <= v_ub,
                 "probing value out of bounds");
    return std::make_tuple(first_round_val, var_val, second_round_val);
  } else {
    auto orig_v_lb =
      orig_sol.problem_ptr->variable_lower_bounds.element(idx, orig_sol.handle_ptr->get_stream());
    auto orig_v_ub =
      orig_sol.problem_ptr->variable_upper_bounds.element(idx, orig_sol.handle_ptr->get_stream());
    cuopt_assert(v_lb >= orig_v_lb, "Current lb should be greater than original lb");
    cuopt_assert(v_ub <= orig_v_ub, "Current ub should be smaller than original ub");
    v_lb = std::max(v_lb, orig_v_lb);
    v_ub = std::min(v_ub, orig_v_ub);
    // the bounds might cross, so correct them here
    if (v_lb > v_ub) {
      v_lb = orig_v_lb;
      v_ub = orig_v_lb;
    }
    auto v_f = std::floor(var_val);
    auto v_c = std::ceil(var_val);
    if (std::ceil(v_lb) == std::floor(v_ub)) {
      return std::make_tuple(std::ceil(v_lb), var_val, std::floor(v_ub));
    } else if (v_f < std::ceil(v_lb)) {
      v_f = std::ceil(v_lb);
      v_c = v_f + 1;
    } else if (v_c > std::floor(v_ub)) {
      v_c = std::floor(v_ub);
      v_f = v_c - 1;
    }
    cuopt_assert(orig_v_lb <= v_f && v_f <= orig_v_ub, "probing value out of bounds");
    cuopt_assert(orig_v_lb <= v_c && v_c <= orig_v_ub, "probing value out of bounds");
    return std::make_tuple(v_f, var_val, v_c);
  }
}

template <typename i_t, typename f_t>
std::vector<thrust::pair<i_t, f_t>> lb_constraint_prop_t<i_t, f_t>::generate_bulk_rounding_vector(
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
  const solution_t<i_t, f_t>& orig_sol,
  const std::vector<i_t>& host_vars_to_set,
  const std::optional<std::vector<thrust::pair<f_t, f_t>>> probing_candidates)
{
  const f_t int_tol = orig_sol.problem_ptr->tolerances.integrality_tolerance;
  std::string log_str{"Setting var:\t"};
  std::vector<thrust::pair<i_t, f_t>> var_val_pairs;
  var_val_pairs.reserve(host_vars_to_set.size());
  for (i_t i = 0; i < (i_t)host_vars_to_set.size(); ++i) {
    // TODO don't read one by one, copy the bulk indices before to a vector
    auto unset_var_idx = host_vars_to_set[i];
    f_t first_probe, second_probe;
    if (probing_candidates.has_value()) {
      // for now get the first one
      thrust::tie(first_probe, second_probe) = probing_candidates.value()[unset_var_idx];
    } else {
      std::tie(first_probe, std::ignore, second_probe) =
        probing_values(lb_bounds_update, orig_sol, unset_var_idx);
    }
    cuopt_assert(orig_sol.problem_ptr->is_integer(first_probe), "Probing value must be an integer");
    cuopt_assert(orig_sol.problem_ptr->is_integer(second_probe),
                 "Probing value must be an integer");
    f_t val_to_round = first_probe;
    // check probing cache if some implied bounds exists
    if (use_probing_cache && lb_bounds_update.probing_cache.contains(unset_var_idx)) {
      // TODO here we can try checking the amount of implied stack.
      // check if there are any conflicting bounds
      val_to_round = lb_bounds_update.probing_cache.get_least_conflicting_rounding(
        lb_bounds_update.host_bounds, unset_var_idx, first_probe, second_probe, int_tol);
    }
    cuopt_assert(orig_sol.problem_ptr->variable_lower_bounds.element(
                   unset_var_idx, orig_sol.handle_ptr->get_stream()) <= val_to_round + int_tol &&
                   val_to_round - int_tol <= orig_sol.problem_ptr->variable_upper_bounds.element(
                                               unset_var_idx, orig_sol.handle_ptr->get_stream()),
                 "Variable out of original bounds!");
    var_val_pairs.emplace_back(unset_var_idx, val_to_round);
    log_str.append(std::to_string(unset_var_idx) + ", ");
  }
  CUOPT_LOG_TRACE("%s", log_str.c_str());
  return var_val_pairs;
}

template <typename i_t, typename f_t, typename f_t2>
__global__ void compute_implied_slack_consumption_per_var(
  typename problem_t<i_t, f_t>::view_t orig_pb,
  raft::device_span<i_t> var_indices,
  raft::device_span<f_t2> cnst_slack,
  raft::device_span<f_t> implied_var_slack_consumption,
  bool is_problem_ii,
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols)
{
  i_t var_idx = var_indices[blockIdx.x];
  cuopt_assert(orig_pb.is_integer_var(var_idx), "Variable must be integer!");
  i_t var_offset                       = orig_pb.reverse_offsets[var_idx];
  i_t var_degree                       = orig_pb.reverse_offsets[var_idx + 1] - var_offset;
  f_t th_var_implied_slack_consumption = 0.;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto a        = orig_pb.reverse_coefficients[var_offset + i];
    auto cnst_idx = orig_pb.reverse_constraints[var_offset + i];
    auto slack    = cnst_slack[cnst_idx];
    //  don't consider constraints that are infeasible
    // FIXME : Being tested/benchmarked. Pick either one depending upon results
    if constexpr (false) {
      if ((-tols.absolute_tolerance >= slack.x) || (tols.absolute_tolerance <= slack.y)) {
        continue;
      }
    } else {
      auto cnstr_ub = orig_pb.constraint_upper_bounds[cnst_idx];
      auto cnstr_lb = orig_pb.constraint_lower_bounds[cnst_idx];
      f_t eps       = get_cstr_tolerance<i_t, f_t>(cnstr_lb,
                                             cnstr_ub,
                                             orig_pb.tolerances.absolute_tolerance,
                                             orig_pb.tolerances.relative_tolerance);
      if ((0 > slack.x + eps) || (eps < slack.y)) { continue; }
    }

#pragma unroll
    for (auto act : {slack.x, slack.y}) {
      f_t slack_consumption_ratio;
      if (is_problem_ii && abs(act) < tols.absolute_tolerance) {
        slack_consumption_ratio = 1000.;
      } else {
        slack_consumption_ratio = (a / act) * (a / act);
      }
      th_var_implied_slack_consumption += slack_consumption_ratio;
    }
  }
  __shared__ f_t shmem[raft::WarpSize];
  f_t block_var_implied_slack_consumption =
    raft::blockReduce(th_var_implied_slack_consumption, (char*)shmem);
  if (threadIdx.x == 0) {
    implied_var_slack_consumption[blockIdx.x] = block_var_implied_slack_consumption;
  }
}

// sort by the implied percent of slack consumption
// across all constraints, sum the square roots of implied slack consumption percent
template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::sort_by_implied_slack_consumption(
  problem_t<i_t, f_t>& original_problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
  raft::device_span<i_t> vars,
  bool problem_ii)
{
  CUOPT_LOG_TRACE("Sorting vars by importance");
  rmm::device_uvector<f_t> implied_slack_consumption_per_var(
    vars.size(), original_problem.handle_ptr->get_stream());
  using f_t2          = typename type_2<f_t>::type;
  const i_t block_dim = 128;
  lb_bounds_update.calculate_constraint_slack(original_problem.handle_ptr);
  compute_implied_slack_consumption_per_var<i_t, f_t, f_t2>
    <<<vars.size(), block_dim, 0, original_problem.handle_ptr->get_stream()>>>(
      original_problem.view(),
      vars,
      make_span_2(lb_bounds_update.cnst_slack),
      make_span(implied_slack_consumption_per_var),
      problem_ii,
      context.settings.get_tolerances());
  thrust::sort_by_key(original_problem.handle_ptr->get_thrust_policy(),
                      implied_slack_consumption_per_var.begin(),
                      implied_slack_consumption_per_var.end(),
                      vars.data(),
                      thrust::greater<f_t>{});
  original_problem.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::collapse_crossing_bounds(
  load_balanced_problem_t<i_t, f_t>* problem,
  problem_t<i_t, f_t>& orig_problem,
  const raft::handle_t* handle_ptr)
{
  using f_t2           = typename type_2<f_t>::type;
  auto variable_bounds = make_span_2(problem->variable_bounds);
  auto original_lb     = make_span(orig_problem.variable_lower_bounds);
  auto original_ub     = make_span(orig_problem.variable_upper_bounds);
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator((i_t)variable_bounds.size()),
    [variable_bounds,
     original_lb,
     original_ub,
     variable_types = make_span(orig_problem.variable_types),
     int_tol        = orig_problem.tolerances.integrality_tolerance] __device__(i_t idx) {
      auto var_bnd = variable_bounds[idx];
      auto v_lb    = var_bnd.x;
      auto v_ub    = var_bnd.y;
      auto o_lb    = original_lb[idx];
      auto o_ub    = original_ub[idx];
      if (v_lb > v_ub) {
        f_t val_to_collapse;
        if (variable_types[idx] == var_t::INTEGER) {
          round_val_on_singleton_and_crossing<i_t, f_t>(val_to_collapse, v_lb, v_ub, o_lb, o_ub);
        } else {
          if (isfinite(o_lb) && isfinite(o_ub)) {
            val_to_collapse = (o_lb + o_ub) / 2;
          } else {
            val_to_collapse = isfinite(o_lb) ? o_lb : o_ub;
          }
        }

        cuopt_assert(o_lb - int_tol <= val_to_collapse && val_to_collapse <= o_ub + int_tol,
                     "Out of original bounds!");
        variable_bounds[idx] = f_t2{val_to_collapse, val_to_collapse};
      }
    });
}

template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::restore_original_bounds_on_unfixed(
  load_balanced_problem_t<i_t, f_t>* problem,
  problem_t<i_t, f_t>& original_problem,
  const raft::handle_t* handle_ptr)
{
  using f_t2           = typename type_2<f_t>::type;
  auto variable_bounds = make_span_2(problem->variable_bounds);
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(problem->n_variables),
    [variable_bounds, op_v = original_problem.view()] __device__(i_t var_idx) {
      auto var_bnd = variable_bounds[var_idx];
      if (!op_v.integer_equal(var_bnd.x, var_bnd.y) || !op_v.is_integer_var(var_idx)) {
        variable_bounds[var_idx] =
          f_t2{op_v.variable_lower_bounds[var_idx], op_v.variable_upper_bounds[var_idx]};
      }
    });
}

template <typename i_t, typename f_t>
bool lb_constraint_prop_t<i_t, f_t>::run_repair_procedure(
  load_balanced_problem_t<i_t, f_t>* problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
  problem_t<i_t, f_t>& original_problem,
  timer_t& timer,
  const raft::handle_t* handle_ptr)
{
  lb_bounds_update.set_updated_bounds(problem);
  repair_attempts++;
  f_t repair_start_time                = timer.remaining_time();
  i_t n_of_repairs_needed_for_feasible = 0;
  do {
    n_of_repairs_needed_for_feasible++;
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Time limit is reached in repair loop!");
      f_t repair_end_time = timer.remaining_time();
      total_time_spent_on_repair += repair_start_time - repair_end_time;
      return false;
    }
    total_repair_loops++;
    collapse_crossing_bounds(problem, original_problem, handle_ptr);
    // infeasible cnst_slack valid
    bool bounds_repaired =
      bounds_repair.repair_problem(problem, lb_bounds_update, original_problem, timer, handle_ptr);
    if (bounds_repaired) {
      intermediate_repair_success++;
      CUOPT_LOG_DEBUG("Bounds repair success, running bounds prop to verify feasibility!");
    }
    f_t bounds_prop_start_time = timer.remaining_time();
    // now the problem is repaired. restore all bounds to the original bounds and run bounds prop on
    // it. note that the number of fixed vars will still be the same, as repair only shifts the vars
    restore_original_bounds_on_unfixed(problem, original_problem, handle_ptr);
    lb_bounds_update.settings.iteration_limit = 100;
    // infeasible cnst_slack invalid
    auto term_crit                            = lb_bounds_update.solve();
    lb_bounds_update.settings.iteration_limit = 20;
    // if time limit is reached, this is needed sometimes we reach time limit and decide that it is
    // not ii but propagation eventually will make it ii
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Time limit is reached in repair loop!");
      f_t repair_end_time = timer.remaining_time();
      total_time_spent_on_repair += repair_start_time - repair_end_time;
      return false;
    }
    if (termination_criterion_t::NO_UPDATE != term_crit) {
      lb_bounds_update.set_updated_bounds(problem);
    }
    f_t bounds_prop_end_time = timer.remaining_time();
    total_time_spent_bounds_prop_after_repair += bounds_prop_start_time - bounds_prop_end_time;
  } while (lb_bounds_update.infeas_constraints_count > 0);
  repair_success++;
  CUOPT_LOG_INFO("Repair success: n_of_repair_calls needed: %d", n_of_repairs_needed_for_feasible);
  f_t repair_end_time = timer.remaining_time();
  total_time_spent_on_repair += repair_start_time - repair_end_time;
  return true;
}

template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::copy_bounds(
  rmm::device_uvector<f_t>& output_variable_bounds,
  rmm::device_uvector<f_t>& output_assignment,
  const rmm::device_uvector<f_t>& input_variable_bounds,
  const rmm::device_uvector<f_t>& input_assignment,
  const raft::handle_t* handle_ptr)
{
  raft::copy(output_variable_bounds.data(),
             input_variable_bounds.data(),
             input_variable_bounds.size(),
             handle_ptr->get_stream());
  raft::copy(output_assignment.data(),
             input_assignment.data(),
             input_assignment.size(),
             handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::restore_bounds(load_balanced_problem_t<i_t, f_t>* problem,
                                                    rmm::device_uvector<f_t>* assignment,
                                                    const raft::handle_t* handle_ptr)
{
  copy_bounds(
    problem->variable_bounds, *assignment, bounds_restore, assignment_restore, handle_ptr);
}

template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::save_bounds(load_balanced_problem_t<i_t, f_t>& problem,
                                                 rmm::device_uvector<f_t>& assignment,
                                                 const raft::handle_t* handle_ptr)
{
  copy_bounds(bounds_restore, assignment_restore, problem.variable_bounds, assignment, handle_ptr);
}

template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::update_host_assignment(
  const rmm::device_uvector<f_t>& assignment, const raft::handle_t* handle_ptr)
{
  raft::copy(
    curr_host_assignment.data(), assignment.data(), assignment.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool lb_constraint_prop_t<i_t, f_t>::is_problem_ii(
  load_balanced_problem_t<i_t, f_t>& problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update)
{
  lb_bounds_update.copy_input_bounds(problem);
  lb_bounds_update.calculate_constraint_slack(problem.handle_ptr);
  lb_bounds_update.calculate_infeasible_redundant_constraints(problem.handle_ptr);
  bool problem_ii = lb_bounds_update.infeas_constraints_count > 0;
  return problem_ii;
}

template <typename i_t, typename f_t>
void sort_subsections(raft::device_span<i_t> vars,
                      rmm::device_uvector<f_t>& random_vector,
                      rmm::device_uvector<i_t>& offsets,
                      const raft::handle_t* handle_ptr)
{
  size_t temp_storage_bytes = 0;
  rmm::device_uvector<std::byte> d_temp_storage(0, handle_ptr->get_stream());
  rmm::device_uvector<f_t> input_random_vec(random_vector, handle_ptr->get_stream());
  rmm::device_uvector<i_t> input_vars(vars.size(), handle_ptr->get_stream());
  raft::copy(input_vars.data(), vars.data(), vars.size(), handle_ptr->get_stream());
  cub::DeviceSegmentedSort::SortPairs(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      input_random_vec.data(),
                                      random_vector.data(),
                                      input_vars.data(),
                                      vars.data(),
                                      vars.size(),
                                      n_subsections,
                                      offsets.data(),
                                      offsets.data() + 1,
                                      handle_ptr->get_stream());

  // Allocate temporary storage
  d_temp_storage.resize(temp_storage_bytes, handle_ptr->get_stream());

  // Run sorting operation
  cub::DeviceSegmentedSort::SortPairs(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      input_random_vec.data(),
                                      random_vector.data(),
                                      input_vars.data(),
                                      vars.data(),
                                      vars.size(),
                                      n_subsections,
                                      offsets.data(),
                                      offsets.data() + 1,
                                      handle_ptr->get_stream());
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void lb_constraint_prop_t<i_t, f_t>::sort_by_interval_and_frac(
  load_balanced_problem_t<i_t, f_t>& problem,
  rmm::device_uvector<f_t>& assignment,
  raft::device_span<i_t> vars,
  std::mt19937 rng)
{
  // we can't call this function when the problem is ii. it causes false offset computations
  // TODO add assert that the problem is not ii
  auto variable_bounds = make_span_2(problem.variable_bounds);
  auto assgn           = make_span(assignment);
  thrust::stable_sort(problem.handle_ptr->get_thrust_policy(),
                      vars.begin(),
                      vars.end(),
                      [variable_bounds, assgn] __device__(i_t v_idx_1, i_t v_idx_2) {
                        auto bnd_idx_1        = variable_bounds[v_idx_1];
                        auto bnd_idx_2        = variable_bounds[v_idx_2];
                        f_t bounds_interval_1 = bnd_idx_1.y - bnd_idx_1.x;
                        f_t bounds_interval_2 = bnd_idx_2.y - bnd_idx_2.x;
                        // if bounds interval are equal (binary and ternary) check fraction
                        // if both bounds intervals are greater than 2. then do fraction
                        if ((bounds_interval_1 == bounds_interval_2) ||
                            (bounds_interval_1 > 2 && bounds_interval_2 > 2)) {
                          f_t frac_1 = get_fractionality_of_val(assgn[v_idx_1]);
                          f_t frac_2 = get_fractionality_of_val(assgn[v_idx_2]);
                          return frac_1 < frac_2;
                        } else {
                          return bounds_interval_1 < bounds_interval_2;
                        }
                      });
  // now do the suffling, for that we need to assign some random values to rnd array
  // we will sort this rnd array and the vars in subsections, so that each subsection will be
  // shuffled in total we will have 3(binary, ternary and rest) x 7 intervals = 21 subsections.
  // first extract these subsections from the data
  rmm::device_uvector<i_t> subsection_offsets(size_of_subsections,
                                              problem.handle_ptr->get_stream());
  thrust::fill(problem.handle_ptr->get_thrust_policy(),
               subsection_offsets.begin(),
               subsection_offsets.end(),
               -1);
  subsection_offsets.set_element(0, 0, problem.handle_ptr->get_stream());
  subsection_offsets.set_element(n_subsections, vars.size(), problem.handle_ptr->get_stream());
  thrust::for_each(
    problem.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator((i_t)vars.size() - 1),
    [offsets = make_span(subsection_offsets), variable_bounds, vars, assgn] __device__(i_t idx) {
      i_t var_1             = vars[idx];
      i_t var_2             = vars[idx + 1];
      auto bnds_1           = variable_bounds[var_1];
      auto bnds_2           = variable_bounds[var_2];
      f_t bounds_interval_1 = bnds_1.y - bnds_1.x;
      f_t bounds_interval_2 = bnds_2.y - bnds_2.x;
      f_t frac_1            = get_fractionality_of_val(assgn[var_1]);
      f_t frac_2            = get_fractionality_of_val(assgn[var_2]);
      if (bounds_interval_1 == 1 && bounds_interval_2 == 1) {
        i_t category = 0;
        assign_offsets<i_t, f_t>(offsets, category, idx, frac_1, frac_2);
      } else if (bounds_interval_1 == 1 && bounds_interval_2 == 2) {
        offsets[7] = idx + 1;
      } else if (bounds_interval_1 == 2 && bounds_interval_2 == 2) {
        i_t category = 1;
        assign_offsets<i_t, f_t>(offsets, category, idx, frac_1, frac_2);
      } else if (bounds_interval_1 == 2 && bounds_interval_2 > 2) {
        offsets[14] = idx + 1;
      } else {
        i_t category = 2;
        assign_offsets<i_t, f_t>(offsets, category, idx, frac_1, frac_2);
      }
    });
  // if there are any empty sections fill their offsets as the previous offset
  thrust::for_each(problem.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   [offsets = subsection_offsets.data()] __device__(i_t idx) {
                     i_t last_existing_offset = 0;
                     for (i_t i = n_subsections; i > 0; --i) {
                       if (offsets[i] == -1) {
                         offsets[i] = last_existing_offset;
                       } else {
                         last_existing_offset = offsets[i];
                       }
                     }
                   });
  auto random_vector = get_random_uniform_vector<i_t, f_t>((i_t)vars.size(), rng);
  rmm::device_uvector<f_t> device_random_vector(random_vector.size(),
                                                problem.handle_ptr->get_stream());
  raft::copy(device_random_vector.data(),
             random_vector.data(),
             random_vector.size(),
             problem.handle_ptr->get_stream());
  sort_subsections<i_t, f_t>(vars, device_random_vector, subsection_offsets, problem.handle_ptr);
}

template <typename i_t, typename f_t>
bool lb_constraint_prop_t<i_t, f_t>::apply_round(
  solution_t<i_t, f_t>& sol,
  f_t lp_run_time_after_feasible,
  timer_t& timer,
  std::optional<std::vector<thrust::pair<f_t, f_t>>> probing_candidates)
{
  raft::common::nvtx::range fun_scope("constraint prop round");

  // this is second timer that can continue but without recovery mode
  const f_t max_time_for_bounds_prop = 5.;
  max_timer                          = timer_t{max_time_for_bounds_prop};
  if (check_brute_force_rounding(sol)) { return true; }
  recovery_mode      = false;
  rounding_ii        = false;
  n_iter_in_recovery = 0;
  sol.compute_constraints();

  temp_problem.setup(*sol.problem_ptr);
  bounds_update.setup(temp_problem);

  expand_device_copy(temp_assignment, sol.assignment, sol.handle_ptr->get_stream());
  f_t bounds_prop_start_time = max_timer.remaining_time();
  bool sol_found             = find_integer(temp_assignment,
                                temp_problem,
                                bounds_update,
                                sol,
                                lp_run_time_after_feasible,
                                timer,
                                probing_candidates);
  f_t bounds_prop_end_time   = max_timer.remaining_time();
  total_time_spent_on_bounds_prop += bounds_prop_start_time - bounds_prop_end_time;

  CUOPT_LOG_DEBUG(
    "repair_success %lu repair_attempts %lu intermediate_repair_success %lu total_repair_loops %lu "
    "total_time_spent_on_repair %f total_time_spent_bounds_prop_after_repair %f "
    "total_time_spent_on_bounds_prop %f",
    repair_success,
    repair_attempts,
    intermediate_repair_success,
    total_repair_loops,
    total_time_spent_on_repair,
    total_time_spent_bounds_prop_after_repair,
    total_time_spent_on_bounds_prop);
  if (!sol_found) {
    sol.compute_feasibility();
    return false;
  }
  return sol.compute_feasibility();
}

template <typename i_t, typename f_t>
bool lb_constraint_prop_t<i_t, f_t>::find_integer(
  rmm::device_uvector<f_t>& assignment,
  load_balanced_problem_t<i_t, f_t>& problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
  solution_t<i_t, f_t>& orig_sol,
  f_t lp_run_time_after_feasible,
  timer_t& timer,
  std::optional<std::vector<thrust::pair<f_t, f_t>>> probing_candidates)
{
  RAFT_CHECK_CUDA(problem.handle_ptr->get_stream());
  if (orig_sol.problem_ptr->n_integer_vars == 0) {
    CUOPT_LOG_ERROR("No integer variables provided in the bounds prop rounding");
    cuopt_func_call(orig_sol.test_variable_bounds());
    return orig_sol.compute_feasibility();
  }

  using crit_t             = termination_criterion_t;
  auto& unset_integer_vars = unset_vars;
  std::mt19937 rng(cuopt::seed_generator::get_seed());

  bounds_restore.resize(2 * orig_sol.problem_ptr->n_variables, orig_sol.handle_ptr->get_stream());
  assignment_restore.resize(orig_sol.problem_ptr->n_variables, orig_sol.handle_ptr->get_stream());
  unset_integer_vars.resize(orig_sol.problem_ptr->n_integer_vars,
                            orig_sol.handle_ptr->get_stream());
  curr_host_assignment.resize(orig_sol.problem_ptr->n_variables);
  temp_assignment.resize(orig_sol.problem_ptr->n_variables, orig_sol.handle_ptr->get_stream());

  lb_bounds_update.settings.time_limit      = max_timer.remaining_time();
  lb_bounds_update.settings.iteration_limit = 20;
  RAFT_CHECK_CUDA(problem.handle_ptr->get_stream());

  if (max_timer.check_time_limit()) {
    CUOPT_LOG_DEBUG("Time limit is reached before bounds prop rounding!");
    orig_sol.round_nearest();
    cuopt_func_call(orig_sol.test_variable_bounds());
    return orig_sol.compute_feasibility();
  }

  raft::copy(unset_integer_vars.data(),
             orig_sol.problem_ptr->integer_indices.data(),
             orig_sol.problem_ptr->n_integer_vars,
             orig_sol.handle_ptr->get_stream());
  CUOPT_LOG_DEBUG("LB Bounds propagation rounding: unset vars %lu", unset_integer_vars.size());
  RAFT_CHECK_CUDA(problem.handle_ptr->get_stream());

  // this is needed for the sort inside of the loop
  // infeasible cnst_slack invalid
  bool problem_ii = is_problem_ii(problem, lb_bounds_update);

  // if the problem is ii, run the bounds prop in the beginning
  if (problem_ii) {
    bool bounds_repaired = bounds_repair.repair_problem(
      &problem, lb_bounds_update, *orig_sol.problem_ptr, timer, orig_sol.handle_ptr);
    if (bounds_repaired) {
      CUOPT_LOG_DEBUG("Initial ii is repaired by bounds repair!");
    } else {
      auto term_crit = lb_bounds_update.solve();
      if (termination_criterion_t::NO_UPDATE != term_crit) {
        lb_bounds_update.set_updated_bounds(&problem);
      }
      rounding_ii = true;
    }
  }
  // do the sort if the problem is not ii. crossing bounds might cause some issues on the sort order
  else {
    // this is a sort to have initial shuffling, so that stable sort within will keep the order and
    // some randomness will be achieved
    sort_by_interval_and_frac(problem, assignment, make_span(unset_integer_vars), rng);
  }

  lb_bounds_update.update_host_bounds(problem);

  size_t set_count      = 0;
  bool timeout_happened = false;
  bool repair_tried     = false;

  while (set_count < unset_integer_vars.size()) {
    update_host_assignment(assignment, orig_sol.handle_ptr);
    if (max_timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Second time limit is reached returning nearest rounding!");
      // sol.round_nearest();
      timeout_happened = true;
      break;
    }
    if (!rounding_ii && timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("First time limit is reached! Continuing without backtracking!");
      rounding_ii = true;
    }
    const i_t n_curr_unset = unset_integer_vars.size() - set_count;
    if (!recovery_mode || rounding_ii) {
      if (n_curr_unset > 36) {
        bounds_prop_interval = sqrt(n_curr_unset);
      } else {
        bounds_prop_interval = 1;
      }
    }
    i_t n_vars_to_set = recovery_mode ? 1 : bounds_prop_interval;
    // if we are not at the last stage or if we are in recovery mode, don't sort
    if (n_vars_to_set != 1) {
      // CHECK - uses bounds_update.cnst_slack, skip infeasible
      sort_by_implied_slack_consumption(
        *orig_sol.problem_ptr,
        lb_bounds_update,
        make_span(unset_integer_vars, set_count, unset_integer_vars.size()),
        problem_ii);
    }
    std::vector<i_t> host_vars_to_set(n_vars_to_set);
    raft::copy(host_vars_to_set.data(),
               unset_integer_vars.data() + set_count,
               n_vars_to_set,
               orig_sol.handle_ptr->get_stream());
    auto var_val_pairs = generate_bulk_rounding_vector(
      lb_bounds_update, orig_sol, host_vars_to_set, probing_candidates);
    probe(assignment,
          &problem,
          lb_bounds_update,
          orig_sol.problem_ptr,
          var_val_pairs,
          &set_count,
          unset_integer_vars);
    if (!repair_tried && rounding_ii && !timeout_happened) {
      timer_t repair_timer{min(timer.remaining_time() / 5, timer.elapsed_time() / 3)};
      save_bounds(problem, assignment, orig_sol.handle_ptr);
      // update bounds and run repair procedure
      // infeasible cnst_slack invalid
      bool bounds_repaired = run_repair_procedure(
        &problem, lb_bounds_update, *orig_sol.problem_ptr, repair_timer, orig_sol.handle_ptr);
      if (!bounds_repaired) {
        restore_bounds(&problem, &assignment, orig_sol.handle_ptr);
        repair_tried = true;
      } else {
        CUOPT_LOG_DEBUG(
          "Bounds are repaired! Deactivating recovery mode in bounds prop. n_curr_unset %d  "
          "bounds_prop_interval %d",
          n_curr_unset,
          bounds_prop_interval);
        recovery_mode      = false;
        rounding_ii        = false;
        n_iter_in_recovery = 0;
        // test that bounds are really repaired and no ii cstr is present
        // infeasible cnst_slack invalid
        cuopt_assert(!is_problem_ii(problem, lb_bounds_update),
                     "Problem must not be ii after repair success");
        // during repair procedure some variables might be collapsed
        using f_t2 = typename type_2<f_t>::type;
        auto iter  = thrust::stable_partition(
          orig_sol.handle_ptr->get_thrust_policy(),
          unset_vars.begin() + set_count,
          unset_vars.end(),
          is_bound_fixed_t<i_t, f_t, f_t2>{orig_sol.problem_ptr->tolerances.integrality_tolerance,
                                            make_span_2(problem.variable_bounds),
                                            make_span(orig_sol.problem_ptr->variable_lower_bounds),
                                            make_span(orig_sol.problem_ptr->variable_upper_bounds),
                                            make_span(assignment)});
        i_t n_fixed_vars = (iter - (unset_vars.begin() + set_count));
        CUOPT_LOG_TRACE("After repair procedure, number of additional fixed vars %d", n_fixed_vars);
        set_count += n_fixed_vars;
      }
    }
    if (recovery_mode && lb_bounds_update.infeas_constraints_count > 0) {
      // if bounds are not repaired, restore previous bounds
      CUOPT_LOG_DEBUG("Problem is ii in constraint prop. n_curr_unset %d  bounds_prop_interval %d",
                      n_curr_unset,
                      bounds_prop_interval);
      rounding_ii   = true;
      recovery_mode = false;
    }
    if (recovery_mode && (++n_iter_in_recovery == bounds_prop_interval)) {
      CUOPT_LOG_DEBUG(
        "Deactivating recovery mode in bounds prop. n_curr_unset %d  bounds_prop_interval %d",
        n_curr_unset,
        bounds_prop_interval);
      recovery_mode      = false;
      n_iter_in_recovery = 0;
    }
    // we use this to utilize the caching
    // we update from the problem bounds and not the final bounds of bounds update
    // because we might be in a recovery mode where we want to continue with the bounds before bulk
    // which is the unchanged problem bounds
    lb_bounds_update.update_host_bounds(problem);
  }

  CUOPT_LOG_DEBUG("LB Bounds propagation rounding end: ii constraint count %d",
                  lb_bounds_update.infeas_constraints_count);

  expand_device_copy(orig_sol.assignment, assignment, orig_sol.handle_ptr->get_stream());
  orig_sol.round_nearest();
  cuopt_assert(orig_sol.test_number_all_integer(), "All integers must be rounded");
  cuopt_func_call(orig_sol.test_variable_bounds());

  // if the constraint is not ii, run LP
  if (lb_bounds_update.infeas_constraints_count == 0 && !timeout_happened) {
    run_lp_with_vars_fixed(*orig_sol.problem_ptr,
                           orig_sol,
                           orig_sol.problem_ptr->integer_indices,
                           context.settings.get_tolerances(),
                           lp_run_time_after_feasible,
                           true);
  }
  bool res_feasible = orig_sol.compute_feasibility();
  return res_feasible;
}

#if MIP_INSTANTIATE_FLOAT
template class lb_constraint_prop_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_constraint_prop_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
