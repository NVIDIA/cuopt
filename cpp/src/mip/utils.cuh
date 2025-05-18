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

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>
#include <thrust/transform_reduce.h>
#include <cuopt/error.hpp>
#include <linear_programming/utils.cuh>
#include <raft/random/rng_device.cuh>
#include <random>
#include <utilities/copy_helpers.hpp>

#include <cuopt/linear_programming/mip/solver_settings.hpp>

#pragma once

namespace cuopt::linear_programming::detail {

// TODO templatize as needed
constexpr double default_cont_upper = std::numeric_limits<double>::infinity();
constexpr double default_cont_lower = -std::numeric_limits<double>::infinity();
constexpr int default_int_upper     = std::numeric_limits<int>::max();
constexpr int default_int_lower     = std::numeric_limits<int>::min();
constexpr double zero_bound         = 0.;

template <typename i_t, typename f_t>
HDI f_t get_cstr_tolerance(f_t combined_bound, f_t abs_tol, f_t rel_tol)
{
  f_t tolerance = abs_tol;
  if (USE_REL_TOLERANCE) { tolerance += combined_bound * rel_tol; }
  return tolerance;
}

template <typename i_t, typename f_t>
HDI f_t get_cstr_tolerance(f_t lb, f_t ub, f_t abs_tol, f_t rel_tol)
{
  f_t tolerance = abs_tol;
  // we normally have combined bounds in the problem, but to reduce a memory request we can
  // recompute here
  if (USE_REL_TOLERANCE) {
    f_t max_bound = combine_finite_abs_bounds<f_t>{}(lb, ub);
    tolerance += max_bound * rel_tol;
  }
  return tolerance;
}

template <typename i_t, typename f_t>
HDI bool is_constraint_feasible(f_t cstr_val,
                                f_t lb,
                                f_t ub,
                                typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols)
{
  f_t tolerance =
    get_cstr_tolerance<i_t, f_t>(lb, ub, tols.absolute_tolerance, tols.relative_tolerance);
  return (cstr_val >= lb - tolerance) && (cstr_val <= ub + tolerance);
}

template <typename f_t>
HDI bool integer_equal(f_t val1, f_t val2, f_t tolerance)
{
  return raft::abs(val1 - val2) <= tolerance;
}

template <typename f_t>
HDI bool is_integer(f_t val, f_t tolerance)
{
  return raft::abs(round(val) - (val)) <= tolerance;
}

template <typename f_t>
HDI bool is_binary(f_t val)
{
  return integer_equal<f_t>(val, 1.) || integer_equal<f_t>(val, 0.);
}

// rounds to the nearest integer within bounds
template <typename f_t>
HDI f_t round_nearest(f_t val, f_t lb, f_t ub, f_t int_tol, raft::random::PCGenerator& rng)
{
  if (val > ub) {
    return floor(ub + int_tol);
  } else if (val < lb) {
    return ceil(lb - int_tol);
  } else {
    f_t w = rng.next_float();
    f_t t = 2 * w * (1 - w);
    if (w > 0.5) { t = 1 - t; }
    return floor(val + t);
  }
}

// returns the smallest distance to celing or floor
template <typename f_t>
HDI f_t get_fractionality_of_val(f_t val)
{
  return raft::min(val - floor(val), ceil(val) - val);
}

template <typename i_t, typename f_t>
inline std::vector<f_t> get_random_uniform_vector(i_t size,
                                                  std::mt19937& rng,
                                                  f_t range_start = -1.,
                                                  f_t range_end   = 1.)
{
  std::vector<f_t> vec;
  vec.reserve(size);
  for (i_t i = 0; i < size; ++i) {
    f_t random_val = std::uniform_real_distribution<f_t>(range_start, range_end)(rng);
    vec.push_back(random_val);
  }
  return vec;
}

template <typename f_t>
inline void elementwise_square_root(rmm::device_uvector<f_t>& vals,
                                    const raft::handle_t* handle_ptr)
{
  thrust::transform(handle_ptr->get_thrust_policy(),
                    vals.begin(),
                    vals.end(),
                    vals.begin(),
                    [] __device__(f_t val) { return sqrt(val); });
}

template <typename i_t, typename f_t>
inline f_t compute_l1_distance(const rmm::device_uvector<i_t>& indices,
                               const rmm::device_uvector<f_t>& first_sol,
                               const rmm::device_uvector<f_t>& second_sol,
                               const raft::handle_t* handle_ptr)
{
  cuopt_expects(first_sol.size() == second_sol.size(),
                error_type_t::RuntimeError,
                "Size mismatch at compute_l1_distance!");
  const f_t* first_ptr  = first_sol.data();
  const f_t* second_ptr = second_sol.data();
  f_t diff              = thrust::transform_reduce(
    handle_ptr->get_thrust_policy(),
    indices.begin(),
    indices.end(),
    [first_ptr, second_ptr] __host__ __device__(i_t idx) {
      return abs(first_ptr[idx] - second_ptr[idx]);
    },
    0.,
    thrust::plus<f_t>());
  return diff;
}

// shmem needs to have at least WarpSize
// to use res without race conditions. synchronize after calling the functions
template <typename i_t, typename f_t>
__device__ void compute_l1_distance_block(raft::device_span<f_t>& sol_1,
                                          raft::device_span<f_t>& sol_2,
                                          raft::device_span<i_t>& indices_to_compute,
                                          f_t* shmem,
                                          f_t& res)
{
  cuopt_assert(sol_1.size() == sol_2.size(), "Solutions sizes need to be the same");
  f_t th_l1_dist = 0.;
  for (i_t i = threadIdx.x; i < indices_to_compute.size(); i += blockDim.x) {
    i_t index = indices_to_compute[i];
    th_l1_dist += abs(sol_1[index] - sol_2[index]);
  }
  res = raft::blockReduce(th_l1_dist, (char*)shmem);
}

template <typename i_t, typename f_t>
inline i_t compute_number_of_integer_var_diff(const rmm::device_uvector<i_t>& indices,
                                              const rmm::device_uvector<f_t>& first_sol,
                                              const rmm::device_uvector<f_t>& second_sol,
                                              f_t int_tol,
                                              const raft::handle_t* handle_ptr)
{
  cuopt_expects(
    first_sol.size() == second_sol.size(), error_type_t::RuntimeError, "Size mismatch!");
  const f_t* first_ptr  = first_sol.data();
  const f_t* second_ptr = second_sol.data();
  i_t same_vars =
    thrust::count_if(handle_ptr->get_thrust_policy(),
                     indices.begin(),
                     indices.end(),
                     [first_ptr, second_ptr, int_tol] __host__ __device__(i_t idx) {
                       return integer_equal<f_t>(first_ptr[idx], second_ptr[idx], int_tol);
                     });
  return same_vars;
}

template <typename i_t, typename f_t>
bool check_integer_equal_on_indices(const rmm::device_uvector<i_t>& indices,
                                    const rmm::device_uvector<f_t>& first_sol,
                                    const rmm::device_uvector<f_t>& second_sol,
                                    f_t int_tol,
                                    const raft::handle_t* handle_ptr)
{
  cuopt_assert(first_sol.size() == second_sol.size(), "Size mismatch!");
  auto const first_ptr  = make_span(first_sol);
  auto const second_ptr = make_span(second_sol);
  return thrust::all_of(handle_ptr->get_thrust_policy(),
                        indices.begin(),
                        indices.end(),
                        [first_ptr, second_ptr, int_tol] __host__ __device__(i_t idx) {
                          return integer_equal<f_t>(first_ptr[idx], second_ptr[idx], int_tol);
                        });
}

template <typename i_t, typename f_t>
f_t compute_objective_from_vec(const rmm::device_uvector<f_t>& assignment,
                               const rmm::device_uvector<f_t>& objective_coefficients,
                               rmm::cuda_stream_view stream)
{
  cuopt_assert(assignment.size() == objective_coefficients.size(), "Size mismatch!");
  f_t computed_obj = thrust::inner_product(rmm::exec_policy(stream),
                                           assignment.begin(),
                                           assignment.end(),
                                           objective_coefficients.begin(),
                                           0.);
  return computed_obj;
}

template <typename i_t, typename f_t>
void clamp_within_var_bounds(rmm::device_uvector<f_t>& assignment,
                             const problem_t<i_t, f_t>* problem_ptr,
                             const raft::handle_t* handle_ptr)
{
  cuopt_assert(assignment.size() == problem_ptr->n_variables, "Size mismatch!");
  f_t* assignment_ptr = assignment.data();
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(0) + problem_ptr->n_variables,
                   [assignment_ptr,
                    lower_bound = problem_ptr->variable_lower_bounds.data(),
                    upper_bound = problem_ptr->variable_upper_bounds.data()] __device__(i_t idx) {
                     if (assignment_ptr[idx] < lower_bound[idx]) {
                       assignment_ptr[idx] = lower_bound[idx];
                     } else if (assignment_ptr[idx] > upper_bound[idx]) {
                       assignment_ptr[idx] = upper_bound[idx];
                     }
                   });
}

template <typename i_t, typename f_t>
void clamp_within_constraint_bounds(rmm::device_uvector<f_t>& dual_solution,
                                    const problem_t<i_t, f_t>* problem_ptr,
                                    const raft::handle_t* handle_ptr)
{
  cuopt_assert(dual_solution.size() == problem_ptr->n_constraints, "Size mismatch!");
  f_t* dual_solution_ptr = dual_solution.data();
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(0) + problem_ptr->n_constraints,
                   [dual_solution_ptr,
                    lower_bound = problem_ptr->constraint_lower_bounds.data(),
                    upper_bound = problem_ptr->constraint_upper_bounds.data()] __device__(i_t idx) {
                     if (dual_solution_ptr[idx] < lower_bound[idx]) {
                       dual_solution_ptr[idx] = lower_bound[idx];
                     } else if (dual_solution_ptr[idx] > upper_bound[idx]) {
                       dual_solution_ptr[idx] = upper_bound[idx];
                     }
                   });
}

double inline rand_double(double begin, double end, std::mt19937& gen)
{
  std::uniform_real_distribution<double> dist(begin, end);
  // Generate random number
  double random_double = dist(gen);
  return random_double;
}

template <class F>
static __global__ void run_lambda_kernel(F f)
{
  f();
}

// run a printf statement from the device side, useful for debugging without having to deal with
// explicit memcpys
template <typename Func>
static void inline run_device_lambda(const rmm::cuda_stream_view& stream, Func f)
{
  run_lambda_kernel<<<1, 1, 0, stream.value()>>>(f);
}

template <typename f_t>
f_t compute_rel_mip_gap(f_t user_obj, f_t solution_bound)
{
  if (user_obj == 0.0) {
    return solution_bound == 0.0 ? 0.0 : std::numeric_limits<f_t>::infinity();
  }
  return std::abs(user_obj - solution_bound) / std::abs(user_obj);
}

template <typename f_t>
void print_solution(const raft::handle_t* handle_ptr, const rmm::device_uvector<f_t>& solution)
{
  auto host_solution = cuopt::host_copy(solution, handle_ptr->get_stream());
  std::string log_str{"sol: ["};
  for (int i = 0; i < (int)solution.size(); i++) {
    log_str.append(std::to_string(host_solution[i]) + ", ");
  }
  CUOPT_LOG_DEBUG("%s]", log_str.c_str());
}

template <typename f_t>
bool has_nans(const raft::handle_t* handle_ptr, const rmm::device_uvector<f_t>& vec)
{
  return thrust::any_of(
    handle_ptr->get_thrust_policy(), vec.begin(), vec.end(), [] __device__(f_t val) {
      return isnan(val);
    });
}

template <typename i_t, typename f_t>
bool has_integrality_discrepancy(const raft::handle_t* handle_ptr,
                                 const rmm::device_uvector<i_t>& integer_var_indices,
                                 const rmm::device_uvector<f_t>& assignment,
                                 f_t int_tol)
{
  auto const assignment_span = make_span(assignment);
  return thrust::any_of(handle_ptr->get_thrust_policy(),
                        integer_var_indices.begin(),
                        integer_var_indices.end(),
                        [assignment_span, int_tol] __host__ __device__(i_t idx) {
                          return !is_integer<f_t>(assignment_span[idx], int_tol);
                        });
}

template <typename i_t, typename f_t>
bool has_variable_bounds_violation(const raft::handle_t* handle_ptr,
                                   const rmm::device_uvector<f_t>& assignment,
                                   problem_t<i_t, f_t>* problem_ptr)
{
  auto const assignment_span = make_span(assignment);
  return thrust::any_of(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem_ptr->original_problem_ptr->get_n_variables(),
    [assignment_span, problem_view = problem_ptr->view()] __device__(i_t idx) {
      return !problem_view.check_variable_within_bounds(idx, assignment_span[idx]);
    });
}

}  // namespace cuopt::linear_programming::detail
