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

#include "feasibility_test.cuh"
#include "solution.cuh"
#include "solution_kernels.cuh"

#include <linear_programming/utils.cuh>
#include <mip/diversity/recombiners/recombiner.cuh>
#include <mip/local_search/rounding/simple_rounding.cuh>
#include <mip/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>

#include <thrust/count.h>
#include <thrust/transform_reduce.h>
#include <cuda/functional>
#include <raft/linalg/binary_op.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
solution_t<i_t, f_t>::solution_t(problem_t<i_t, f_t>& problem_)
  : problem_ptr(&problem_),
    handle_ptr(problem_.handle_ptr),
    assignment(problem_.variable_lower_bounds, handle_ptr->get_stream()),
    lower_excess(problem_.n_constraints, handle_ptr->get_stream()),
    upper_excess(problem_.n_constraints, handle_ptr->get_stream()),
    lower_slack(problem_.n_constraints, handle_ptr->get_stream()),
    upper_slack(problem_.n_constraints, handle_ptr->get_stream()),
    constraint_value(problem_.n_constraints, handle_ptr->get_stream()),
    obj_val(handle_ptr->get_stream()),
    n_feasible_constraints(handle_ptr->get_stream()),
    lp_state(problem_)
{
}

template <typename i_t, typename f_t>
solution_t<i_t, f_t>::solution_t(const solution_t<i_t, f_t>& other)
  : problem_ptr(other.problem_ptr),
    handle_ptr(other.handle_ptr),
    assignment(other.assignment, handle_ptr->get_stream()),
    lower_excess(other.lower_excess, handle_ptr->get_stream()),
    upper_excess(other.upper_excess, handle_ptr->get_stream()),
    lower_slack(other.lower_slack, handle_ptr->get_stream()),
    upper_slack(other.upper_slack, handle_ptr->get_stream()),
    constraint_value(other.constraint_value, handle_ptr->get_stream()),
    obj_val(other.obj_val, handle_ptr->get_stream()),
    n_feasible_constraints(other.n_feasible_constraints, handle_ptr->get_stream()),
    n_assigned_integers(other.n_assigned_integers),
    h_obj(other.h_obj),
    h_user_obj(other.h_user_obj),
    h_infeasibility_cost(other.h_infeasibility_cost),
    is_feasible(other.is_feasible),
    is_problem_fully_reduced(other.is_problem_fully_reduced),
    is_scaled_(other.is_scaled_),
    post_process_completed(other.post_process_completed),
    lp_state(other.lp_state)
{
}

// fill the contents as needed
template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::copy_from(const solution_t<i_t, f_t>& other_sol)
{
  // TODO handle resize
  problem_ptr          = other_sol.problem_ptr;
  handle_ptr           = other_sol.handle_ptr;
  h_obj                = other_sol.h_obj;
  h_user_obj           = other_sol.h_user_obj;
  h_infeasibility_cost = other_sol.h_infeasibility_cost;
  expand_device_copy(assignment, other_sol.assignment, handle_ptr->get_stream());
  expand_device_copy(lower_excess, other_sol.lower_excess, handle_ptr->get_stream());
  expand_device_copy(upper_excess, other_sol.upper_excess, handle_ptr->get_stream());
  expand_device_copy(lower_slack, other_sol.lower_slack, handle_ptr->get_stream());
  expand_device_copy(upper_slack, other_sol.upper_slack, handle_ptr->get_stream());
  expand_device_copy(constraint_value, other_sol.constraint_value, handle_ptr->get_stream());
  raft::copy(obj_val.data(), other_sol.obj_val.data(), 1, handle_ptr->get_stream());
  raft::copy(n_feasible_constraints.data(),
             other_sol.n_feasible_constraints.data(),
             1,
             handle_ptr->get_stream());
  is_feasible              = other_sol.is_feasible;
  is_problem_fully_reduced = other_sol.is_problem_fully_reduced;
  is_scaled_               = other_sol.is_scaled_;
  post_process_completed   = other_sol.post_process_completed;
  expand_device_copy(
    lp_state.prev_primal, other_sol.lp_state.prev_primal, handle_ptr->get_stream());
  expand_device_copy(lp_state.prev_dual, other_sol.lp_state.prev_dual, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::resize_to_problem()
{
  assignment.resize(problem_ptr->n_variables, handle_ptr->get_stream());
  lower_excess.resize(problem_ptr->n_constraints, handle_ptr->get_stream());
  upper_excess.resize(problem_ptr->n_constraints, handle_ptr->get_stream());
  lower_slack.resize(problem_ptr->n_constraints, handle_ptr->get_stream());
  upper_slack.resize(problem_ptr->n_constraints, handle_ptr->get_stream());
  constraint_value.resize(problem_ptr->n_constraints, handle_ptr->get_stream());
  lp_state.prev_primal.resize(problem_ptr->n_variables, handle_ptr->get_stream());
  lp_state.prev_dual.resize(problem_ptr->n_constraints, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::resize_to_original_problem()
{
  assignment.resize(problem_ptr->original_problem_ptr->get_n_variables(), handle_ptr->get_stream());
  lower_excess.resize(problem_ptr->original_problem_ptr->get_n_constraints(),
                      handle_ptr->get_stream());
  upper_excess.resize(problem_ptr->original_problem_ptr->get_n_constraints(),
                      handle_ptr->get_stream());
  lower_slack.resize(problem_ptr->original_problem_ptr->get_n_constraints(),
                     handle_ptr->get_stream());
  upper_slack.resize(problem_ptr->original_problem_ptr->get_n_constraints(),
                     handle_ptr->get_stream());
  constraint_value.resize(problem_ptr->original_problem_ptr->get_n_constraints(),
                          handle_ptr->get_stream());
  lp_state.prev_primal.resize(problem_ptr->original_problem_ptr->get_n_variables(),
                              handle_ptr->get_stream());
  lp_state.prev_dual.resize(problem_ptr->original_problem_ptr->get_n_constraints(),
                            handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::resize_copy(const solution_t<i_t, f_t>& other_sol)
{
  assignment.resize(other_sol.assignment.size(), handle_ptr->get_stream());
  lower_excess.resize(other_sol.lower_excess.size(), handle_ptr->get_stream());
  upper_excess.resize(other_sol.upper_excess.size(), handle_ptr->get_stream());
  lower_slack.resize(other_sol.lower_slack.size(), handle_ptr->get_stream());
  upper_slack.resize(other_sol.upper_slack.size(), handle_ptr->get_stream());
  constraint_value.resize(other_sol.constraint_value.size(), handle_ptr->get_stream());
  lp_state.prev_primal.resize(other_sol.lp_state.prev_primal.size(), handle_ptr->get_stream());
  lp_state.prev_dual.resize(other_sol.lp_state.prev_dual.size(), handle_ptr->get_stream());
  copy_from(other_sol);
}

template <typename i_t, typename f_t>
typename solution_t<i_t, f_t>::view_t solution_t<i_t, f_t>::view()
{
  solution_t<i_t, f_t>::view_t v;
  v.problem          = problem_ptr->view();
  v.assignment       = raft::device_span<f_t>{assignment.data(), assignment.size()};
  v.lower_excess     = raft::device_span<f_t>{lower_excess.data(), lower_excess.size()};
  v.upper_excess     = raft::device_span<f_t>{upper_excess.data(), upper_excess.size()};
  v.lower_slack      = raft::device_span<f_t>{lower_slack.data(), lower_slack.size()};
  v.upper_slack      = raft::device_span<f_t>{upper_slack.data(), upper_slack.size()};
  v.constraint_value = raft::device_span<f_t>{constraint_value.data(), constraint_value.size()};
  v.obj_val          = obj_val.data();
  v.n_feasible_constraints = n_feasible_constraints.data();
  return v;
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::set_feasible()
{
  is_feasible = 1;
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::set_infeasible()
{
  is_feasible = 0;
}

template <typename i_t, typename f_t>
bool solution_t<i_t, f_t>::get_feasible()
{
  return is_feasible;
}

template <typename i_t, typename f_t>
bool solution_t<i_t, f_t>::get_problem_fully_reduced()
{
  return is_problem_fully_reduced;
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::set_problem_fully_reduced()
{
  is_problem_fully_reduced = true;
}

template <typename i_t, typename f_t>
std::vector<f_t> solution_t<i_t, f_t>::get_host_assignment()
{
  // do not use the size of assignment as it might be different than the n_variables
  return cuopt::host_copy(assignment.data(), problem_ptr->n_variables, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::copy_new_assignment(const std::vector<f_t>& h_assignment)
{
  assignment.resize(h_assignment.size(), handle_ptr->get_stream());
  raft::copy(assignment.data(), h_assignment.data(), h_assignment.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::assign_random_within_bounds(f_t ratio_of_vars_to_random_assign,
                                                       bool only_integers)
{
  std::mt19937 rng(cuopt::seed_generator::get_seed());
  std::vector<f_t> h_assignment = host_copy(assignment);
  std::uniform_real_distribution<f_t> unif_prob(0, 1);

  auto variable_lower_bounds = cuopt::host_copy(problem_ptr->variable_lower_bounds);
  auto variable_upper_bounds = cuopt::host_copy(problem_ptr->variable_upper_bounds);
  auto variable_types        = cuopt::host_copy(problem_ptr->variable_types);
  problem_ptr->handle_ptr->sync_stream();
  for (size_t i = 0; i < problem_ptr->variable_lower_bounds.size(); ++i) {
    if (only_integers && variable_types[i] != var_t::INTEGER) { continue; }
    bool skip = unif_prob(rng) > ratio_of_vars_to_random_assign;
    if (skip) { continue; }
    f_t lower_bound = variable_lower_bounds[i];
    f_t upper_bound = variable_upper_bounds[i];
    if (lower_bound == -std::numeric_limits<f_t>::infinity()) {
      h_assignment[i] = upper_bound;
    } else if (upper_bound == std::numeric_limits<f_t>::infinity()) {
      h_assignment[i] = lower_bound;
    } else {
      if (variable_types[i] == var_t::INTEGER) {
        std::uniform_int_distribution<i_t> unif(lower_bound, upper_bound);
        h_assignment[i] = unif(rng);
      } else {
        std::uniform_real_distribution<f_t> unif(lower_bound, upper_bound);
        h_assignment[i] = unif(rng);
      }
    }
    cuopt_assert(!isnan(h_assignment[i]), "Assignment cannot be nan");
    cuopt_assert(isfinite(h_assignment[i]), "Assignment cannot be nan");
  }
  raft::copy(assignment.data(), h_assignment.data(), h_assignment.size(), handle_ptr->get_stream());
  compute_feasibility();
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::set_vars_to_values(
  const std::vector<thrust::pair<i_t, f_t>>& var_val_pairs)
{
  rmm::device_uvector<thrust::pair<i_t, f_t>> d_var_val_pairs(var_val_pairs.size(),
                                                              handle_ptr->get_stream());
  raft::copy(
    d_var_val_pairs.data(), var_val_pairs.data(), var_val_pairs.size(), handle_ptr->get_stream());
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   d_var_val_pairs.begin(),
                   d_var_val_pairs.end(),
                   [assignment = assignment.data()] __device__(auto pair) {
                     assignment[pair.first] = pair.second;
                   });
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::compute_constraints()
{
  if (problem_ptr->n_constraints == 0) { return; }

  i_t TPB = 64;
  compute_constraint_values<i_t, f_t>
    <<<problem_ptr->n_constraints, TPB, 0, handle_ptr->get_stream()>>>(view());
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::compute_l2_residual()
{
  rmm::device_uvector<f_t> combined_excess(problem_ptr->n_constraints, handle_ptr->get_stream());
  rmm::device_scalar<f_t> l2_residual(handle_ptr->get_stream());
  cuopt_assert(combined_excess.size() == lower_excess.size(), "Mismatch at excess sizes");
  cuopt_assert(problem_ptr->n_constraints == lower_excess.size(), "Mismatch at excess sizes");
  raft::linalg::binaryOp(
    combined_excess.data(),
    lower_excess.data(),
    upper_excess.data(),
    problem_ptr->n_constraints,
    [] __device__(f_t lower, f_t upper) -> f_t { return max(abs(lower), abs(upper)); },
    handle_ptr->get_stream());
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  my_l2_norm<i_t, f_t>(combined_excess, l2_residual, handle_ptr);
  return l2_residual.value(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool solution_t<i_t, f_t>::compute_feasibility()
{
  n_feasible_constraints.set_value_to_zero_async(handle_ptr->get_stream());
  compute_constraints();
  compute_objective();
  compute_infeasibility();
  compute_number_of_integers();
  i_t h_n_feas_constraints = n_feasible_constraints.value(handle_ptr->get_stream());
  is_feasible = h_n_feas_constraints == problem_ptr->n_constraints && test_number_all_integer();
  CUOPT_LOG_TRACE("is_feasible %d n_feasible_cstr %d all_cstr %d",
                  is_feasible,
                  h_n_feas_constraints,
                  problem_ptr->n_constraints);
  return is_feasible;
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::compute_objective()
{
  h_obj = compute_objective_from_vec<i_t, f_t>(
    assignment, problem_ptr->objective_coefficients, handle_ptr->get_stream());
  // to save from memory transactions, don't update the device objective
  // when needed we can update the device objective here
  h_user_obj = problem_ptr->get_user_obj_from_solver_obj(h_obj);
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::compute_infeasibility()
{
  auto v = view();
  cuopt_assert(v.lower_excess.size() == v.problem.n_constraints, "Size mismatch");
  cuopt_assert(v.upper_excess.size() == v.problem.n_constraints, "Size mismatch");
  // the constraint values must be valid here
  h_infeasibility_cost = thrust::transform_reduce(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + v.problem.n_constraints,
    [v] __host__ __device__(i_t idx) { return (v.lower_excess[idx] + v.upper_excess[idx]); },
    0.,
    thrust::plus<f_t>());
}

template <typename i_t, typename f_t>
bool solution_t<i_t, f_t>::round_nearest()
{
  invoke_round_nearest(*this);
  return compute_feasibility();
}

template <typename i_t, typename f_t>
bool solution_t<i_t, f_t>::round_random_nearest(i_t n_target_random_rounds)
{
  invoke_random_round_nearest(*this, n_target_random_rounds);
  return compute_feasibility();
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::correct_integer_precision()
{
  invoke_correct_integers(*this, problem_ptr->tolerances.integrality_tolerance);
}

template <typename i_t, typename f_t>
i_t solution_t<i_t, f_t>::compute_number_of_integers()
{
  const f_t* assignment_ptr = assignment.data();
  n_assigned_integers =
    thrust::count_if(handle_ptr->get_thrust_policy(),
                     problem_ptr->integer_indices.begin(),
                     problem_ptr->integer_indices.end(),
                     [pb = problem_ptr->view(), assignment_ptr] __device__(i_t idx) -> bool {
                       return pb.is_integer(assignment_ptr[idx]);
                     });
  return n_assigned_integers;
}

template <typename i_t, typename f_t>
bool solution_t<i_t, f_t>::test_number_all_integer()
{
  i_t n_integers = compute_number_of_integers();
  return n_integers == problem_ptr->n_integer_vars;
}

template <typename i_t, typename f_t>
std::tuple<problem_t<i_t, f_t>, rmm::device_uvector<f_t>, rmm::device_uvector<i_t>>
solution_t<i_t, f_t>::fix_variables(const rmm::device_uvector<i_t>& variable_indices)
{
  rmm::device_uvector<f_t> new_assignment(assignment, handle_ptr->get_stream());
  rmm::device_uvector<i_t> variable_map(assignment.size(), handle_ptr->get_stream());

  problem_t<i_t, f_t> fixed_problem = problem_ptr->get_problem_after_fixing_vars(
    new_assignment, variable_indices, variable_map, handle_ptr);
  fixed_problem.check_problem_representation();
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    new_assignment.begin(),
    new_assignment.end(),
    [] __device__(f_t assgn) { cuopt_assert(isfinite(assgn), "New assignment is not finite"); });
  handle_ptr->sync_stream();
  return {std::move(fixed_problem), std::move(new_assignment), std::move(variable_map)};
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::unfix_variables(rmm::device_uvector<f_t>& fixed_assignment,
                                           const rmm::device_uvector<i_t>& variable_map)
{
  f_t* fixed_assignment_ptr   = fixed_assignment.data();
  f_t* assignment_ptr         = assignment.data();
  const i_t* variable_map_ptr = variable_map.data();
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(0) + variable_map.size(),
                   [fixed_assignment_ptr, assignment_ptr, variable_map_ptr] __device__(i_t idx) {
                     i_t old_idx             = variable_map_ptr[idx];
                     assignment_ptr[old_idx] = fixed_assignment_ptr[idx];
                   });
  handle_ptr->sync_stream();
  compute_feasibility();
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::clamp_within_bounds()
{
  clamp_within_var_bounds(assignment, problem_ptr, handle_ptr);
  handle_ptr->sync_stream();
  compute_feasibility();
}

template <typename i_t, typename f_t>
i_t solution_t<i_t, f_t>::calculate_similarity_radius(solution_t<i_t, f_t>& other_sol) const
{
  cuopt_assert(assignment.size() == other_sol.assignment.size(),
               "Assignment sizes should be equal!");
  const f_t* curr_assignment = assignment.data();
  const f_t* other_ptr       = other_sol.assignment.data();
  i_t n_equal_integers       = thrust::count_if(
    handle_ptr->get_thrust_policy(),
    problem_ptr->integer_indices.begin(),
    problem_ptr->integer_indices.end(),
    cuda::proclaim_return_type<bool>(
      [other_ptr, curr_assignment, p_view = problem_ptr->view()] __device__(i_t idx) -> bool {
        return diverse_equal<f_t>(other_ptr[idx],
                                  curr_assignment[idx],
                                  p_view.variable_lower_bounds[idx],
                                  p_view.variable_upper_bounds[idx],
                                  p_view.is_integer_var(idx),
                                  p_view.tolerances.integrality_tolerance);
      }));
  return n_equal_integers;
}

template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::get_objective()
{
  return h_obj;
}

template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::get_user_objective()
{
  return h_user_obj;
}

template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::get_quality(const weight_t<i_t, f_t>& weights)
{
  return get_quality(weights.cstr_weights, weights.objective_weight);
}

// assumed to be called after compute_feasibility is called
template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::get_quality(const rmm::device_uvector<f_t>& cstr_weights,
                                      const rmm::device_scalar<f_t>& objective_weight)
{
  // TODO we can as well keep the weights in the solution and compute this once
  f_t weighted_infeasibility = thrust::transform_reduce(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem_ptr->n_constraints,
    cuda::proclaim_return_type<f_t>(
      [v = view(), cstr_weights_ptr = cstr_weights.data()] __device__(i_t idx) {
        return (v.lower_excess[idx] + v.upper_excess[idx]) * cstr_weights_ptr[idx];
      }),
    0.,
    thrust::plus<f_t>());
  return weighted_infeasibility + h_obj * objective_weight.value(handle_ptr->get_stream());
}

// assumed to be called after compute_feasibility is called
template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::get_total_excess()
{
  f_t total_excess =
    thrust::transform_reduce(handle_ptr->get_thrust_policy(),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(0) + problem_ptr->n_constraints,
                             cuda::proclaim_return_type<f_t>([v = view()] __device__(i_t idx) {
                               return (v.lower_excess[idx] + v.upper_excess[idx]);
                             }),
                             0.,
                             thrust::plus<f_t>());
  return total_excess;
}

// TODO compute these on unscaled problem
template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::compute_max_constraint_violation()
{
  return thrust::transform_reduce(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem_ptr->n_constraints,
    cuda::proclaim_return_type<f_t>([v = view()] __device__(i_t idx) -> f_t {
      return max(v.lower_excess[idx], v.upper_excess[idx]);
    }),
    0.,
    thrust::maximum<f_t>());
}

template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::compute_max_int_violation()
{
  return thrust::transform_reduce(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem_ptr->n_integer_vars,
    cuda::proclaim_return_type<f_t>([v = view()] __device__(i_t idx) -> f_t {
      if (v.problem.variable_types[idx] == var_t::INTEGER) {
        return abs(v.assignment[idx] - round(v.assignment[idx]));
      }
      return 0.;
    }),
    0.,
    thrust::maximum<f_t>());
}

template <typename i_t, typename f_t>
f_t solution_t<i_t, f_t>::compute_max_variable_violation()
{
  cuopt_assert(problem_ptr->n_variables == assignment.size(), "Size mismatch");
  cuopt_assert(problem_ptr->n_variables == problem_ptr->variable_lower_bounds.size(),
               "Size mismatch");
  cuopt_assert(problem_ptr->n_variables == problem_ptr->variable_upper_bounds.size(),
               "Size mismatch");
  return thrust::transform_reduce(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem_ptr->n_variables,
    cuda::proclaim_return_type<f_t>([v = view()] __device__(i_t idx) -> f_t {
      f_t lower_vio = max(0., v.problem.variable_lower_bounds[idx] - v.assignment[idx]);
      f_t upper_vio = max(0., v.assignment[idx] - v.problem.variable_upper_bounds[idx]);
      return max(lower_vio, upper_vio);
    }),
    0.,
    thrust::maximum<f_t>());
}

// returns the solution after applying the conversions
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solution_t<i_t, f_t>::get_solution(bool output_feasible,
                                                            solver_stats_t<i_t, f_t> stats,
                                                            bool log_stats)
{
  cuopt::default_logger().flush();
  cuopt_expects(
    post_process_completed, error_type_t::RuntimeError, "Post process must be called on solution!");

  if (output_feasible) {
    // TODO we can streamline these info in class
    f_t rel_mip_gap                  = compute_rel_mip_gap(h_user_obj, stats.solution_bound);
    f_t abs_mip_gap                  = fabs(h_user_obj - stats.solution_bound);
    f_t solution_bound               = stats.solution_bound;
    f_t max_constraint_violation     = compute_max_constraint_violation();
    f_t max_int_violation            = compute_max_int_violation();
    f_t max_variable_bound_violation = compute_max_variable_violation();
    f_t total_solve_time             = stats.total_solve_time;
    f_t presolve_time                = stats.presolve_time;
    i_t num_nodes                    = stats.num_nodes;
    i_t num_simplex_iterations       = stats.num_simplex_iterations;
    handle_ptr->sync_stream();
    if (log_stats) {
      CUOPT_LOG_INFO(
        "Solution objective: %f , relative_mip_gap %f solution_bound %f presolve_time %f "
        "total_solve_time %f "
        "max constraint violation %f max int violation %f max var bounds violation %f "
        "nodes %d simplex_iterations %d",
        h_user_obj,
        rel_mip_gap,
        solution_bound,
        presolve_time,
        total_solve_time,
        max_constraint_violation,
        max_int_violation,
        max_variable_bound_violation,
        num_nodes,
        num_simplex_iterations);
    }
    const bool not_optimal = rel_mip_gap > problem_ptr->tolerances.relative_mip_gap &&
                             abs_mip_gap > problem_ptr->tolerances.absolute_mip_gap;
    auto term_reason =
      not_optimal ? mip_termination_status_t::FeasibleFound : mip_termination_status_t::Optimal;
    if (is_problem_fully_reduced) { term_reason = mip_termination_status_t::Optimal; }
    return mip_solution_t<i_t, f_t>(std::move(assignment),
                                    problem_ptr->var_names,
                                    h_user_obj,
                                    rel_mip_gap,
                                    term_reason,
                                    max_constraint_violation,
                                    max_int_violation,
                                    max_variable_bound_violation,
                                    stats);
  } else {
    return mip_solution_t<i_t, f_t>{is_problem_fully_reduced ? mip_termination_status_t::Infeasible
                                                             : mip_termination_status_t::TimeLimit,
                                    stats,
                                    handle_ptr->get_stream()};
  }
}

#if MIP_INSTANTIATE_FLOAT
template class solution_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class solution_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
