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

#pragma once

#include "problem.cuh"

#include <cuopt/error.hpp>

#include <mip/logger.cuh>
#include <raft/linalg/unary_op.cuh>
#include <utilities/copy_helpers.hpp>

#include <cuda_runtime_api.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

namespace cuopt::linear_programming::detail {
template <typename f_t>
struct transform_bounds_functor {
  __device__ thrust::tuple<f_t, f_t> operator()(const thrust::tuple<char, f_t>& input) const
  {
    f_t lower_bound, upper_bound;
    const char row_type = thrust::get<0>(input);
    const f_t b_value   = thrust::get<1>(input);

    if (row_type == 'E') {
      lower_bound = b_value;
      upper_bound = b_value;
    } else if (row_type == 'G') {
      lower_bound = b_value;
      upper_bound = std::numeric_limits<f_t>::infinity();
    } else if (row_type == 'L') {
      lower_bound = -std::numeric_limits<f_t>::infinity();
      upper_bound = b_value;
    } else {
      // Can never happen
      lower_bound = -std::numeric_limits<f_t>::infinity();
      upper_bound = -std::numeric_limits<f_t>::infinity();
    }

    return thrust::make_tuple(lower_bound, upper_bound);
  }
};

template <typename i_t, typename f_t>
static void set_bounds_if_not_set(detail::problem_t<i_t, f_t>& op_problem)
{
  raft::common::nvtx::range scope("set_bounds_if_not_set");

  // If an user gave row type instead of lower/upper bounds
  if (op_problem.constraint_lower_bounds.is_empty() &&
      op_problem.constraint_upper_bounds.is_empty() &&
      !op_problem.original_problem_ptr->get_row_types().is_empty() &&
      !op_problem.original_problem_ptr->get_constraint_bounds().is_empty()) {
    op_problem.constraint_lower_bounds.resize(
      op_problem.original_problem_ptr->get_row_types().size(), op_problem.handle_ptr->get_stream());
    op_problem.constraint_upper_bounds.resize(
      op_problem.original_problem_ptr->get_row_types().size(), op_problem.handle_ptr->get_stream());

    auto first = thrust::make_zip_iterator(
      thrust::make_tuple(op_problem.original_problem_ptr->get_row_types().cbegin(),
                         op_problem.original_problem_ptr->get_constraint_bounds().cbegin()));
    auto last = thrust::make_zip_iterator(
      thrust::make_tuple(op_problem.original_problem_ptr->get_row_types().cend(),
                         op_problem.original_problem_ptr->get_constraint_bounds().cend()));

    auto out_first = thrust::make_zip_iterator(thrust::make_tuple(
      op_problem.constraint_lower_bounds.begin(), op_problem.constraint_upper_bounds.begin()));

    thrust::transform(op_problem.handle_ptr->get_thrust_policy(),
                      first,
                      last,
                      out_first,
                      transform_bounds_functor<f_t>());
  }

  // If variable bound was not set, set it to default value
  if (op_problem.variable_lower_bounds.is_empty() &&
      !op_problem.objective_coefficients.is_empty()) {
    op_problem.variable_lower_bounds.resize(op_problem.objective_coefficients.size(),
                                            op_problem.handle_ptr->get_stream());
    thrust::fill(op_problem.handle_ptr->get_thrust_policy(),
                 op_problem.variable_lower_bounds.begin(),
                 op_problem.variable_lower_bounds.end(),
                 f_t(0));
  }
  if (op_problem.variable_upper_bounds.is_empty() &&
      !op_problem.objective_coefficients.is_empty()) {
    op_problem.variable_upper_bounds.resize(op_problem.objective_coefficients.size(),
                                            op_problem.handle_ptr->get_stream());
    thrust::fill(op_problem.handle_ptr->get_thrust_policy(),
                 op_problem.variable_upper_bounds.begin(),
                 op_problem.variable_upper_bounds.end(),
                 std::numeric_limits<f_t>::infinity());
  }
  if (op_problem.variable_types.is_empty() && !op_problem.objective_coefficients.is_empty()) {
    op_problem.variable_types.resize(op_problem.objective_coefficients.size(),
                                     op_problem.handle_ptr->get_stream());
    thrust::fill(op_problem.handle_ptr->get_thrust_policy(),
                 op_problem.variable_types.begin(),
                 op_problem.variable_types.end(),
                 var_t::CONTINUOUS);
  }
}

template <typename f_t>
struct negate {
  __device__ f_t operator()(f_t value) { return -value; }
};

template <typename i_t, typename f_t>
static void convert_to_maximization_problem(detail::problem_t<i_t, f_t>& op_problem)
{
  raft::common::nvtx::range scope("convert_to_maximization_problem");

  if (op_problem.objective_coefficients.size()) {
    // Negate objective coefficient
    raft::linalg::unaryOp(op_problem.objective_coefficients.data(),
                          op_problem.objective_coefficients.data(),
                          op_problem.objective_coefficients.size(),
                          detail::negate<f_t>(),
                          op_problem.handle_ptr->get_stream());
  }
  // Negate objective scaling factor and objective offset so that primal / dual stay same sign after
  // negating objective coeffs
  op_problem.presolve_data.objective_scaling_factor =
    -op_problem.presolve_data.objective_scaling_factor;

  // Negate objective offset
  op_problem.presolve_data.objective_offset = -op_problem.presolve_data.objective_offset;
}

/*
 * For each constraints in the csr untransposed matrix (one block per constraint)
 * Loop over the variables founds in the considered constraint
 * For each variables found, loop in parallel (block-level) over the transposed matrix
 * Make sure at least one thread finds the corresponding value and at the correct position
 */
template <typename i_t, typename f_t>
__global__ void kernel_check_transpose_validity(raft::device_span<const f_t> coefficients,
                                                raft::device_span<const i_t> offsets,
                                                raft::device_span<const i_t> variables,
                                                raft::device_span<const f_t> reverse_coefficients,
                                                raft::device_span<const i_t> reverse_offsets,
                                                raft::device_span<const i_t> reverse_variables,
                                                bool* failed)
{
  const i_t constraint_id = blockIdx.x;
  __shared__ bool shared_found;
  if (threadIdx.x == 0) shared_found = false;
  __syncthreads();
  for (i_t j = offsets[constraint_id]; j < offsets[constraint_id + 1]; ++j) {
    i_t col   = variables[j];
    f_t value = coefficients[j];

    bool found = false;

    for (i_t k = reverse_offsets[col] + threadIdx.x; k < reverse_offsets[col + 1];
         k += blockDim.x) {
      if (reverse_variables[k] == constraint_id && reverse_coefficients[k] == value) {
        found = true;
        break;
      }
    }
    if (found) shared_found = true;
    __syncthreads();
    // Would want to assert there but no easy way to gtest it, so moved it to the host
    if (!shared_found) {
      DEVICE_LOG_DEBUG(
        "For cstr %d, var %d, value %f was not found in the transpose", constraint_id, col, value);
      *failed = true;
      return;
    }
    __syncthreads();
    if (threadIdx.x == 0) shared_found = false;
    __syncthreads();
  }
}

// Only called in assert mode
template <typename i_t, typename f_t>
static bool check_transpose_validity(const rmm::device_uvector<f_t>& coefficients,
                                     const rmm::device_uvector<i_t>& offsets,
                                     const rmm::device_uvector<i_t>& variables,
                                     const rmm::device_uvector<f_t>& reverse_coefficients,
                                     const rmm::device_uvector<i_t>& reverse_offsets,
                                     const rmm::device_uvector<i_t>& reverse_variables,
                                     const raft::handle_t* handle_ptr)
{
  if (offsets.size() <= 1) { return true; }

  rmm::device_scalar<bool> failed(false, handle_ptr->get_stream());
  kernel_check_transpose_validity<i_t, f_t>
    <<<offsets.size() - 1, 64, 0, handle_ptr->get_stream()>>>(
      raft::device_span<const f_t>(coefficients.data(), coefficients.size()),
      raft::device_span<const i_t>(offsets.data(), offsets.size()),
      raft::device_span<const i_t>(variables.data(), variables.size()),
      raft::device_span<const f_t>(reverse_coefficients.data(), reverse_coefficients.size()),
      raft::device_span<const i_t>(reverse_offsets.data(), reverse_offsets.size()),
      raft::device_span<const i_t>(reverse_variables.data(), reverse_variables.size()),
      failed.data());
  RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  cuopt_assert(!failed.value(handle_ptr->get_stream()),
               "Difference between the matrix and its transpose");
  return true;
}

template <typename i_t, typename f_t>
static void check_csr_representation([[maybe_unused]] const rmm::device_uvector<f_t>& coefficients,
                                     const rmm::device_uvector<i_t>& offsets,
                                     [[maybe_unused]] const rmm::device_uvector<i_t>& variables,
                                     const raft::handle_t* handle_ptr,
                                     [[maybe_unused]] i_t n_variables,
                                     [[maybe_unused]] i_t n_constraints)
{
  raft::common::nvtx::range scope("check_csr_representation");

  cuopt_assert(variables.size() == coefficients.size(),
               "A_index and A_values must have same sizes.");

  // Check offset values
  const i_t first_value = offsets.front_element(handle_ptr->get_stream());
  cuopt_assert(first_value == 0, "A_offsets first value should be 0.");

  cuopt_assert(thrust::is_sorted(handle_ptr->get_thrust_policy(), offsets.cbegin(), offsets.cend()),
               "A_offsets values must in an increasing order.");

  // Check indices
  cuopt_assert(thrust::all_of(handle_ptr->get_thrust_policy(),
                              variables.cbegin(),
                              variables.cend(),
                              [n_variables = n_variables] __device__(i_t val) {
                                return val >= 0 && val < n_variables;
                              }),
               "A_indices values must positive lower than the number of variables (c size).");

  cuopt_assert(offsets.size() == (std::size_t)n_constraints + 1,
               "A_offsets must be exactly equal to number of constraints + 1");
}

template <typename i_t, typename f_t>
static bool check_var_bounds_sanity(const detail::problem_t<i_t, f_t>& problem)
{
  bool crossing_bounds_detected =
    thrust::any_of(problem.handle_ptr->get_thrust_policy(),
                   thrust::counting_iterator(0),
                   thrust::counting_iterator((i_t)problem.variable_lower_bounds.size()),
                   [tolerance = problem.tolerances.presolve_absolute_tolerance,
                    lb        = make_span(problem.variable_lower_bounds),
                    ub        = make_span(problem.variable_upper_bounds)] __device__(i_t index) {
                     return (lb[index] > ub[index] + tolerance);
                   });
  return !crossing_bounds_detected;
}

template <typename i_t, typename f_t>
static bool check_constraint_bounds_sanity(const detail::problem_t<i_t, f_t>& problem)
{
  bool crossing_bounds_detected =
    thrust::any_of(problem.handle_ptr->get_thrust_policy(),
                   thrust::counting_iterator(0),
                   thrust::counting_iterator((i_t)problem.constraint_lower_bounds.size()),
                   [tolerance = problem.tolerances.presolve_absolute_tolerance,
                    lb        = make_span(problem.constraint_lower_bounds),
                    ub        = make_span(problem.constraint_upper_bounds)] __device__(i_t index) {
                     return (lb[index] > ub[index] + tolerance);
                   });
  return !crossing_bounds_detected;
}

template <typename i_t, typename f_t>
static void round_bounds(detail::problem_t<i_t, f_t>& problem)
{
  // round bounds to integer for integer variables
  thrust::for_each(problem.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(problem.n_variables),
                   [lb    = make_span(problem.variable_lower_bounds),
                    ub    = make_span(problem.variable_upper_bounds),
                    types = make_span(problem.variable_types)] __device__(i_t index) {
                     if (types[index] == var_t::INTEGER) {
                       lb[index] = ceil(lb[index]);
                       ub[index] = floor(ub[index]);
                     }
                   });
}

template <typename i_t, typename f_t>
static bool check_bounds_sanity(const detail::problem_t<i_t, f_t>& problem)
{
  return check_var_bounds_sanity<i_t, f_t>(problem) &&
         check_constraint_bounds_sanity<i_t, f_t>(problem);
}

}  // namespace cuopt::linear_programming::detail
