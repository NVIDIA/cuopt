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

#include <mip/solution/solution.cuh>
#include "problem.cuh"
#include "problem_helpers.cuh"
#include "problem_kernels.cuh"

#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>
#include <utilities/macros.cuh>

#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <mip/presolve/trivial_presolve.cuh>
#include <mip/utils.cuh>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <cuda/std/functional>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/logger.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/sparse/linalg/transpose.cuh>

#include <unordered_set>

#include <cuda_profiler_api.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::op_problem_cstr_body(const optimization_problem_t<i_t, f_t>& problem_)
{
  // Mark the problem as empty if the op_problem has an empty matrix.
  if (problem_.get_constraint_matrix_values().is_empty()) {
    cuopt_assert(problem_.get_constraint_matrix_indices().is_empty(),
                 "Problem is empty but constraint matrix indices are not empty.");
    cuopt_assert(problem_.get_constraint_matrix_offsets().size() == 1,
                 "Problem is empty but constraint matrix offsets are not empty.");
    cuopt_assert(problem_.get_constraint_lower_bounds().is_empty(),
                 "Problem is empty but constraint lower bounds are not empty.");
    cuopt_assert(problem_.get_constraint_upper_bounds().is_empty(),
                 "Problem is empty but constraint upper bounds are not empty.");
    empty = true;
  }

  // Set variables bounds to default if not set and constraints bounds if user has set a row type
  set_bounds_if_not_set(*this);

  set_variable_bounds(*this);

  const bool is_mip = original_problem_ptr->get_problem_category() != problem_category_t::LP;
  if (is_mip) {
    variable_types =
      rmm::device_uvector<var_t>(problem_.get_variable_types(), handle_ptr->get_stream());
    // round bounds to integer for integer variables, note: do this before checking sanity
    round_bounds(*this);
  }

  // check bounds sanity before, so that we can throw exceptions before going into asserts
  check_bounds_sanity(*this);

  // Check before any modifications
  check_problem_representation(false, false);
  // If maximization problem, convert the problem
  if (maximize) convert_to_maximization_problem(*this);
  if (is_mip) {
    integer_indices.resize(n_variables, handle_ptr->get_stream());
    is_binary_variable.resize(n_variables, handle_ptr->get_stream());
    compute_n_integer_vars();
    compute_binary_var_table();
    compute_vars_with_objective_coeffs();
  }
  compute_transpose_of_problem();
  // Check after modifications
  check_problem_representation(true, is_mip);
  combine_constraint_bounds<i_t, f_t>(*this, combined_bounds);
}

template <typename i_t, typename f_t>
problem_t<i_t, f_t>::problem_t(
  const optimization_problem_t<i_t, f_t>& problem_,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_)
  : original_problem_ptr(&problem_),
    handle_ptr(problem_.get_handle_ptr()),
    integer_fixed_variable_map(problem_.get_n_variables(), problem_.get_handle_ptr()->get_stream()),
    tolerances(tolerances_),
    n_variables(problem_.get_n_variables()),
    n_constraints(problem_.get_n_constraints()),
    n_binary_vars(0),
    n_integer_vars(0),
    nnz(problem_.get_nnz()),
    maximize(problem_.get_sense()),
    presolve_data(problem_, handle_ptr->get_stream()),
    reverse_coefficients(0, problem_.get_handle_ptr()->get_stream()),
    reverse_constraints(0, problem_.get_handle_ptr()->get_stream()),
    reverse_offsets(0, problem_.get_handle_ptr()->get_stream()),
    coefficients(problem_.get_constraint_matrix_values(), problem_.get_handle_ptr()->get_stream()),
    variables(problem_.get_constraint_matrix_indices(), problem_.get_handle_ptr()->get_stream()),
    offsets(problem_.get_constraint_matrix_offsets(), problem_.get_handle_ptr()->get_stream()),
    objective_coefficients(problem_.get_objective_coefficients(),
                           problem_.get_handle_ptr()->get_stream()),
    variable_bounds(0, problem_.get_handle_ptr()->get_stream()),
    constraint_lower_bounds(problem_.get_constraint_lower_bounds(),
                            problem_.get_handle_ptr()->get_stream()),
    constraint_upper_bounds(problem_.get_constraint_upper_bounds(),
                            problem_.get_handle_ptr()->get_stream()),
    combined_bounds(problem_.get_n_constraints(), problem_.get_handle_ptr()->get_stream()),
    variable_types(0, problem_.get_handle_ptr()->get_stream()),
    integer_indices(0, problem_.get_handle_ptr()->get_stream()),
    binary_indices(0, problem_.get_handle_ptr()->get_stream()),
    nonbinary_indices(0, problem_.get_handle_ptr()->get_stream()),
    is_binary_variable(0, problem_.get_handle_ptr()->get_stream()),
    related_variables(0, problem_.get_handle_ptr()->get_stream()),
    related_variables_offsets(n_variables, problem_.get_handle_ptr()->get_stream()),
    var_names(problem_.get_variable_names()),
    row_names(problem_.get_row_names()),
    objective_name(problem_.get_objective_name()),
    objective_offset(problem_.get_objective_offset()),
    lp_state(*this, problem_.get_handle_ptr()->get_stream()),
    fixing_helpers(n_constraints, n_variables, handle_ptr)
{
  op_problem_cstr_body(problem_);
  branch_and_bound_callback = nullptr;
}

template <typename i_t, typename f_t>
problem_t<i_t, f_t>::problem_t(const problem_t<i_t, f_t>& problem_)
  : original_problem_ptr(problem_.original_problem_ptr),
    tolerances(problem_.tolerances),
    handle_ptr(problem_.handle_ptr),
    integer_fixed_problem(problem_.integer_fixed_problem),
    integer_fixed_variable_map(problem_.integer_fixed_variable_map, handle_ptr->get_stream()),
    branch_and_bound_callback(nullptr),
    n_variables(problem_.n_variables),
    n_constraints(problem_.n_constraints),
    n_binary_vars(problem_.n_binary_vars),
    n_integer_vars(problem_.n_integer_vars),
    nnz(problem_.nnz),
    maximize(problem_.maximize),
    empty(problem_.empty),
    is_binary_pb(problem_.is_binary_pb),
    presolve_data(problem_.presolve_data, handle_ptr->get_stream()),
    original_ids(problem_.original_ids),
    reverse_original_ids(problem_.reverse_original_ids),
    reverse_coefficients(problem_.reverse_coefficients, handle_ptr->get_stream()),
    reverse_constraints(problem_.reverse_constraints, handle_ptr->get_stream()),
    reverse_offsets(problem_.reverse_offsets, handle_ptr->get_stream()),
    coefficients(problem_.coefficients, handle_ptr->get_stream()),
    variables(problem_.variables, handle_ptr->get_stream()),
    offsets(problem_.offsets, handle_ptr->get_stream()),
    objective_coefficients(problem_.objective_coefficients, handle_ptr->get_stream()),
    variable_bounds(problem_.variable_bounds, handle_ptr->get_stream()),
    constraint_lower_bounds(problem_.constraint_lower_bounds, handle_ptr->get_stream()),
    constraint_upper_bounds(problem_.constraint_upper_bounds, handle_ptr->get_stream()),
    combined_bounds(problem_.combined_bounds, handle_ptr->get_stream()),
    variable_types(problem_.variable_types, handle_ptr->get_stream()),
    integer_indices(problem_.integer_indices, handle_ptr->get_stream()),
    binary_indices(problem_.binary_indices, handle_ptr->get_stream()),
    nonbinary_indices(problem_.nonbinary_indices, handle_ptr->get_stream()),
    is_binary_variable(problem_.is_binary_variable, handle_ptr->get_stream()),
    related_variables(problem_.related_variables, handle_ptr->get_stream()),
    related_variables_offsets(problem_.related_variables_offsets, handle_ptr->get_stream()),
    var_names(problem_.var_names),
    row_names(problem_.row_names),
    objective_name(problem_.objective_name),
    is_scaled_(problem_.is_scaled_),
    preprocess_called(problem_.preprocess_called),
    lp_state(problem_.lp_state),
    fixing_helpers(problem_.fixing_helpers, handle_ptr),
    vars_with_objective_coeffs(problem_.vars_with_objective_coeffs)
{
}

template <typename i_t, typename f_t>
problem_t<i_t, f_t>::problem_t(const problem_t<i_t, f_t>& problem_, bool no_deep_copy)
  : original_problem_ptr(problem_.original_problem_ptr),
    tolerances(problem_.tolerances),
    handle_ptr(problem_.handle_ptr),
    integer_fixed_problem(problem_.integer_fixed_problem),
    integer_fixed_variable_map(problem_.n_variables, handle_ptr->get_stream()),
    n_variables(problem_.n_variables),
    n_constraints(problem_.n_constraints),
    n_binary_vars(problem_.n_binary_vars),
    n_integer_vars(problem_.n_integer_vars),
    nnz(problem_.nnz),
    maximize(problem_.maximize),
    empty(problem_.empty),
    is_binary_pb(problem_.is_binary_pb),
    // Copy constructor used by PDLP and MIP
    // PDLP uses the version with no_deep_copy = false which deep copy some fields but doesn't
    // allocate others that are not needed in PDLP
    presolve_data(
      (!no_deep_copy)
        ? std::move(presolve_data_t{*problem_.original_problem_ptr, handle_ptr->get_stream()})
        : std::move(presolve_data_t{problem_.presolve_data, handle_ptr->get_stream()})),
    original_ids(problem_.original_ids),
    reverse_original_ids(problem_.reverse_original_ids),
    reverse_coefficients(
      (!no_deep_copy)
        ? rmm::device_uvector<f_t>(problem_.reverse_coefficients, handle_ptr->get_stream())
        : rmm::device_uvector<f_t>(problem_.reverse_coefficients.size(), handle_ptr->get_stream())),
    reverse_constraints(
      (!no_deep_copy)
        ? rmm::device_uvector<i_t>(problem_.reverse_constraints, handle_ptr->get_stream())
        : rmm::device_uvector<i_t>(problem_.reverse_constraints.size(), handle_ptr->get_stream())),
    reverse_offsets(
      (!no_deep_copy)
        ? rmm::device_uvector<i_t>(problem_.reverse_offsets, handle_ptr->get_stream())
        : rmm::device_uvector<i_t>(problem_.reverse_offsets.size(), handle_ptr->get_stream())),
    coefficients(
      (!no_deep_copy)
        ? rmm::device_uvector<f_t>(problem_.coefficients, handle_ptr->get_stream())
        : rmm::device_uvector<f_t>(problem_.coefficients.size(), handle_ptr->get_stream())),
    variables((!no_deep_copy)
                ? rmm::device_uvector<i_t>(problem_.variables, handle_ptr->get_stream())
                : rmm::device_uvector<i_t>(problem_.variables.size(), handle_ptr->get_stream())),
    offsets((!no_deep_copy)
              ? rmm::device_uvector<i_t>(problem_.offsets, handle_ptr->get_stream())
              : rmm::device_uvector<i_t>(problem_.offsets.size(), handle_ptr->get_stream())),
    objective_coefficients(
      (!no_deep_copy)
        ? rmm::device_uvector<f_t>(problem_.objective_coefficients, handle_ptr->get_stream())
        : rmm::device_uvector<f_t>(problem_.objective_coefficients.size(),
                                   handle_ptr->get_stream())),
    variable_bounds(
      (!no_deep_copy)
        ? rmm::device_uvector<f_t2>(problem_.variable_bounds, handle_ptr->get_stream())
        : rmm::device_uvector<f_t2>(problem_.variable_bounds.size(), handle_ptr->get_stream())),
    constraint_lower_bounds(
      (!no_deep_copy)
        ? rmm::device_uvector<f_t>(problem_.constraint_lower_bounds, handle_ptr->get_stream())
        : rmm::device_uvector<f_t>(problem_.constraint_lower_bounds.size(),
                                   handle_ptr->get_stream())),
    constraint_upper_bounds(
      (!no_deep_copy)
        ? rmm::device_uvector<f_t>(problem_.constraint_upper_bounds, handle_ptr->get_stream())
        : rmm::device_uvector<f_t>(problem_.constraint_upper_bounds.size(),
                                   handle_ptr->get_stream())),
    combined_bounds(
      (!no_deep_copy)
        ? rmm::device_uvector<f_t>(problem_.combined_bounds, handle_ptr->get_stream())
        : rmm::device_uvector<f_t>(problem_.combined_bounds.size(), handle_ptr->get_stream())),
    variable_types(
      (!no_deep_copy)
        ? rmm::device_uvector<var_t>(problem_.variable_types, handle_ptr->get_stream())
        : rmm::device_uvector<var_t>(problem_.variable_types.size(), handle_ptr->get_stream())),
    integer_indices((!no_deep_copy) ? 0 : problem_.integer_indices.size(),
                    handle_ptr->get_stream()),
    binary_indices((!no_deep_copy) ? 0 : problem_.binary_indices.size(), handle_ptr->get_stream()),
    nonbinary_indices((!no_deep_copy) ? 0 : problem_.nonbinary_indices.size(),
                      handle_ptr->get_stream()),
    is_binary_variable((!no_deep_copy) ? 0 : problem_.is_binary_variable.size(),
                       handle_ptr->get_stream()),
    related_variables(problem_.related_variables, handle_ptr->get_stream()),
    related_variables_offsets(problem_.related_variables_offsets, handle_ptr->get_stream()),
    var_names(problem_.var_names),
    row_names(problem_.row_names),
    objective_name(problem_.objective_name),
    is_scaled_(problem_.is_scaled_),
    preprocess_called(problem_.preprocess_called),
    lp_state(problem_.lp_state),
    fixing_helpers(problem_.fixing_helpers, handle_ptr),
    vars_with_objective_coeffs(problem_.vars_with_objective_coeffs)
{
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::compute_transpose_of_problem()
{
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  // Resize what is needed for LP
  reverse_offsets.resize(n_variables + 1, handle_ptr->get_stream());
  reverse_constraints.resize(nnz, handle_ptr->get_stream());
  reverse_coefficients.resize(nnz, handle_ptr->get_stream());

  // Special case if A is empty
  // as cuSparse had a bug up until 12.9 causing cusparseCsr2cscEx2 to return incorrect results
  // for empty matrices (CUSPARSE-2319)
  // In this case, construct it manually
  if (reverse_coefficients.is_empty()) {
    thrust::fill(
      handle_ptr->get_thrust_policy(), reverse_offsets.begin(), reverse_offsets.end(), 0);
    return;
  }

  raft::sparse::linalg::csr_transpose(*handle_ptr,
                                      offsets.data(),
                                      variables.data(),
                                      coefficients.data(),
                                      reverse_offsets.data(),
                                      reverse_constraints.data(),
                                      reverse_coefficients.data(),
                                      n_constraints,
                                      n_variables,
                                      nnz,
                                      handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
i_t problem_t<i_t, f_t>::get_n_binary_variables()
{
  n_binary_vars = thrust::count_if(handle_ptr->get_thrust_policy(),
                                   is_binary_variable.begin(),
                                   is_binary_variable.end(),
                                   cuda::std::identity{});
  return n_binary_vars;
}

// Check all fields
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::check_problem_representation(bool check_transposed,
                                                       bool check_mip_related_data)
{
  raft::common::nvtx::range scope("check_problem_representation");

  cuopt_assert(!offsets.is_empty(), "A_offsets must never be empty.");
  if (check_transposed) {
    cuopt_assert(!reverse_offsets.is_empty(), "A_offsets must never be empty.");
  }
  // Presolve reductions might trivially solve the problem to optimality/infeasibility.
  // In this case, it is exptected that the problem fields are empty.
  if (!empty) {
    // Check for empty fields
    cuopt_assert(!coefficients.is_empty(), "A_values must be set before calling the solver.");
    cuopt_assert(!variables.is_empty(), "A_indices must be set before calling the solver.");
    if (check_transposed) {
      cuopt_assert(!reverse_coefficients.is_empty(),
                   "A_values must be set before calling the solver.");
      cuopt_assert(!reverse_constraints.is_empty(),
                   "A_indices must be set before calling the solver.");
    }
  }
  cuopt_assert(objective_coefficients.size() == n_variables,
               "objective_coefficients size mismatch");

  // Check CSR validity
  check_csr_representation(
    coefficients, offsets, variables, handle_ptr, n_variables, n_constraints);
  if (check_transposed) {
    // Check revere CSR validity
    check_csr_representation(reverse_coefficients,
                             reverse_offsets,
                             reverse_constraints,
                             handle_ptr,
                             n_constraints,
                             n_variables);
    cuopt_assert(check_transpose_validity(this->coefficients,
                                          this->offsets,
                                          this->variables,
                                          this->reverse_coefficients,
                                          this->reverse_offsets,
                                          this->reverse_constraints,
                                          handle_ptr),
                 "Tranpose invalide");
  }

  // Check variable bounds are set and with the correct size
  if (!empty) { cuopt_assert(!variable_bounds.is_empty(), "Variable bounds must be set."); }
  cuopt_assert(variable_bounds.size() == objective_coefficients.size(),
               "Sizes for vectors related to the variables are not the same.");
  cuopt_assert(variable_bounds.size() == (std::size_t)n_variables,
               "Sizes for vectors related to the variables are not the same.");

  cuopt_assert(variable_types.size() == (std::size_t)n_variables,
               "Sizes for vectors related to the variables are not the same.");
  // Check constraints bounds sizes
  if (!empty) {
    cuopt_assert(!constraint_lower_bounds.is_empty() && !constraint_upper_bounds.is_empty(),
                 "Constraints lower bounds and constraints upper bounds must be set.");
  }
  cuopt_assert(constraint_lower_bounds.size() == constraint_upper_bounds.size(),
               "Sizes for vectors related to the constraints are not the same.");
  cuopt_assert(constraint_lower_bounds.size() == (size_t)n_constraints,
               "Sizes for vectors related to the constraints are not the same.");
  cuopt_assert((offsets.size() - 1) == constraint_lower_bounds.size(),
               "Sizes for vectors related to the constraints are not the same.");

  // Check combined bounds
  cuopt_assert(combined_bounds.size() == (size_t)n_constraints,
               "Sizes for vectors related to the constraints are not the same.");

  // Check the validity of bounds
  cuopt_expects(thrust::all_of(handle_ptr->get_thrust_policy(),
                               thrust::make_counting_iterator<i_t>(0),
                               thrust::make_counting_iterator<i_t>(n_variables),
                               [vars_bnd = make_span(variable_bounds)] __device__(i_t idx) {
                                 auto bounds = vars_bnd[idx];
                                 return get_lower(bounds) <= get_upper(bounds);
                               }),
                error_type_t::ValidationError,
                "Variable bounds are invalid");
  cuopt_expects(
    thrust::all_of(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<i_t>(0),
                   thrust::make_counting_iterator<i_t>(n_constraints),
                   [constraint_lower_bounds = constraint_lower_bounds.data(),
                    constraint_upper_bounds = constraint_upper_bounds.data()] __device__(i_t idx) {
                     return constraint_lower_bounds[idx] <= constraint_upper_bounds[idx];
                   }),
    error_type_t::ValidationError,
    "Constraints bounds are invalid");

  if (check_mip_related_data) {
    cuopt_assert(n_integer_vars == integer_indices.size(), "incorrect integer indices structure");
    cuopt_assert(is_binary_variable.size() == n_variables, "incorrect binary variable table size");

    cuopt_assert(thrust::is_sorted(
                   handle_ptr->get_thrust_policy(), binary_indices.begin(), binary_indices.end()),
                 "binary indices are not sorted");
    cuopt_assert(
      thrust::is_sorted(
        handle_ptr->get_thrust_policy(), nonbinary_indices.begin(), nonbinary_indices.end()),
      "nonbinary indices are not sorted");
    cuopt_assert(thrust::is_sorted(
                   handle_ptr->get_thrust_policy(), integer_indices.begin(), integer_indices.end()),
                 "integer indices are not sorted");
    // check precomputed helpers
    cuopt_assert(thrust::all_of(handle_ptr->get_thrust_policy(),
                                integer_indices.cbegin(),
                                integer_indices.cend(),
                                [types = variable_types.data()] __device__(i_t idx) {
                                  return types[idx] == var_t::INTEGER;
                                }),
                 "The integer indices table contains references to non-integer variables.");
    cuopt_assert(thrust::all_of(handle_ptr->get_thrust_policy(),
                                binary_indices.cbegin(),
                                binary_indices.cend(),
                                [bin_table = is_binary_variable.data()] __device__(i_t idx) {
                                  return bin_table[idx];
                                }),
                 "The binary indices table contains references to non-binary variables.");
    cuopt_assert(thrust::all_of(handle_ptr->get_thrust_policy(),
                                nonbinary_indices.cbegin(),
                                nonbinary_indices.cend(),
                                [bin_table = is_binary_variable.data()] __device__(i_t idx) {
                                  return !bin_table[idx];
                                }),
                 "The non-binary indices table contains references to binary variables.");
    cuopt_assert(
      thrust::all_of(
        handle_ptr->get_thrust_policy(),
        thrust::make_counting_iterator<i_t>(0),
        thrust::make_counting_iterator<i_t>(n_variables),
        [types     = variable_types.data(),
         bin_table = is_binary_variable.data(),
         pb_view   = view()] __device__(i_t idx) {
          // ensure the binary variable tables are correct
          if (bin_table[idx]) {
            if (!thrust::binary_search(
                  thrust::seq, pb_view.binary_indices.begin(), pb_view.binary_indices.end(), idx))
              return false;
          } else {
            if (!thrust::binary_search(thrust::seq,
                                       pb_view.nonbinary_indices.begin(),
                                       pb_view.nonbinary_indices.end(),
                                       idx))
              return false;
          }

          // finish by checking the correctness of the integer indices table
          switch (types[idx]) {
            case var_t::INTEGER:
              return thrust::binary_search(
                thrust::seq, pb_view.integer_indices.begin(), pb_view.integer_indices.end(), idx);
            case var_t::CONTINUOUS:
              return !thrust::binary_search(
                thrust::seq, pb_view.integer_indices.begin(), pb_view.integer_indices.end(), idx);
          }
          return true;
        }),
      "Some variables aren't referenced in the appropriate indice tables");
    cuopt_assert(
      thrust::all_of(
        handle_ptr->get_thrust_policy(),
        thrust::make_counting_iterator<i_t>(0),
        thrust::make_counting_iterator<i_t>(n_variables),
        [types     = variable_types.data(),
         bin_table = is_binary_variable.data(),
         pb_view   = view()] __device__(i_t idx) {
          // ensure the binary variable tables are correct
          if (bin_table[idx]) {
            if (!thrust::binary_search(
                  thrust::seq, pb_view.binary_indices.begin(), pb_view.binary_indices.end(), idx))
              return false;
          } else {
            if (!thrust::binary_search(thrust::seq,
                                       pb_view.nonbinary_indices.begin(),
                                       pb_view.nonbinary_indices.end(),
                                       idx))
              return false;
          }

          // finish by checking the correctness of the integer indices table
          switch (types[idx]) {
            case var_t::INTEGER:
              return thrust::binary_search(
                thrust::seq, pb_view.integer_indices.begin(), pb_view.integer_indices.end(), idx);
            case var_t::CONTINUOUS:
              return !thrust::binary_search(
                thrust::seq, pb_view.integer_indices.begin(), pb_view.integer_indices.end(), idx);
          }
          return true;
        }),
      "Some variables aren't referenced in the appropriate indice tables");
    cuopt_assert(
      thrust::all_of(
        handle_ptr->get_thrust_policy(),
        thrust::make_counting_iterator<i_t>(0),
        thrust::make_counting_iterator<i_t>(n_variables),
        [types     = variable_types.data(),
         bin_table = is_binary_variable.data(),
         pb_view   = view()] __device__(i_t idx) {
          // ensure the binary variable tables are correct
          if (bin_table[idx]) {
            if (!thrust::binary_search(
                  thrust::seq, pb_view.binary_indices.begin(), pb_view.binary_indices.end(), idx))
              return false;
          } else {
            if (!thrust::binary_search(thrust::seq,
                                       pb_view.nonbinary_indices.begin(),
                                       pb_view.nonbinary_indices.end(),
                                       idx))
              return false;
          }

          // finish by checking the correctness of the integer indices table
          switch (types[idx]) {
            case var_t::INTEGER:
              return thrust::binary_search(
                thrust::seq, pb_view.integer_indices.begin(), pb_view.integer_indices.end(), idx);
            case var_t::CONTINUOUS:
              return !thrust::binary_search(
                thrust::seq, pb_view.integer_indices.begin(), pb_view.integer_indices.end(), idx);
          }
          return true;
        }),
      "Some variables aren't referenced in the appropriate indice tables");
    cuopt_assert(thrust::all_of(handle_ptr->get_thrust_policy(),
                                thrust::make_zip_iterator(thrust::make_counting_iterator<i_t>(0),
                                                          is_binary_variable.cbegin()),
                                thrust::make_zip_iterator(
                                  thrust::make_counting_iterator<i_t>(is_binary_variable.size()),
                                  is_binary_variable.cend()),
                                [types    = variable_types.data(),
                                 vars_bnd = make_span(variable_bounds),
                                 v = view()] __device__(const thrust::tuple<int, int> tuple) {
                                  i_t idx     = thrust::get<0>(tuple);
                                  i_t pred    = thrust::get<1>(tuple);
                                  auto bounds = vars_bnd[idx];
                                  return pred == (types[idx] != var_t::CONTINUOUS &&
                                                  v.integer_equal(get_lower(bounds), 0.) &&
                                                  v.integer_equal(get_upper(bounds), 1.));
                                }),
                 "The binary variable table is incorrect.");
    if (!empty) {
      cuopt_assert(is_binary_pb == (n_variables == thrust::count(handle_ptr->get_thrust_policy(),
                                                                 is_binary_variable.begin(),
                                                                 is_binary_variable.end(),
                                                                 1)),
                   "is_binary_pb is incorrectly set");
    }
  }
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::pre_process_assignment(rmm::device_uvector<f_t>& assignment)
{
  auto has_nans = cuopt::linear_programming::detail::has_nans(handle_ptr, assignment);
  if (has_nans) {
    CUOPT_LOG_DEBUG("Solution discarded due to nans");
    return false;
  }
  cuopt_assert(assignment.size() == original_problem_ptr->get_n_variables(), "size mismatch");

  // create a temp assignment with the var size after bounds standardization (free vars added)
  rmm::device_uvector<f_t> temp_assignment(presolve_data.additional_var_used.size(),
                                           handle_ptr->get_stream());
  // copy the assignment to the first part(the original variable count) of the temp_assignment
  raft::copy(
    temp_assignment.data(), assignment.data(), assignment.size(), handle_ptr->get_stream());
  auto d_additional_var_used =
    cuopt::device_copy(presolve_data.additional_var_used, handle_ptr->get_stream());
  auto d_additional_var_id_per_var =
    cuopt::device_copy(presolve_data.additional_var_id_per_var, handle_ptr->get_stream());

  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<i_t>(0),
                   thrust::make_counting_iterator<i_t>(original_problem_ptr->get_n_variables()),
                   [additional_var_used       = d_additional_var_used.data(),
                    additional_var_id_per_var = d_additional_var_id_per_var.data(),
                    assgn                     = temp_assignment.data()] __device__(auto idx) {
                     if (additional_var_used[idx]) {
                       cuopt_assert(additional_var_id_per_var[idx] != -1,
                                    "additional_var_id_per_var is not set");
                       // We have two non-negative variables y and z that simulate a free variable
                       // x. If the value of x is negative, we can set z to be something higher than
                       // y. If the value of  x is positive we can set y greater than z
                       assgn[additional_var_id_per_var[idx]] = (assgn[idx] < 0 ? -assgn[idx] : 0.);
                       assgn[idx] += assgn[additional_var_id_per_var[idx]];
                     }
                   });
  assignment.resize(n_variables, handle_ptr->get_stream());
  assignment.shrink_to_fit(handle_ptr->get_stream());
  cuopt_assert(presolve_data.variable_mapping.size() == n_variables, "size mismatch");
  thrust::gather(handle_ptr->get_thrust_policy(),
                 presolve_data.variable_mapping.begin(),
                 presolve_data.variable_mapping.end(),
                 temp_assignment.begin(),
                 assignment.begin());
  handle_ptr->sync_stream();

  auto has_integrality_discrepancy = cuopt::linear_programming::detail::has_integrality_discrepancy(
    handle_ptr, integer_indices, assignment, tolerances.integrality_tolerance);
  if (has_integrality_discrepancy) {
    CUOPT_LOG_DEBUG("Solution discarded due to integrality discrepancy");
    return false;
  }

  auto has_variable_bounds_violation =
    cuopt::linear_programming::detail::has_variable_bounds_violation(handle_ptr, assignment, this);
  if (has_variable_bounds_violation) {
    CUOPT_LOG_DEBUG("Solution discarded due to variable bounds violation");
    return false;
  }
  return true;
}

// this function is used to post process the assignment
// it removes the additional variable for free variables
// and expands the assignment to the original variable dimension
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::post_process_assignment(rmm::device_uvector<f_t>& current_assignment,
                                                  bool resize_to_original_problem)
{
  cuopt_assert(current_assignment.size() == presolve_data.variable_mapping.size(), "size mismatch");
  auto assgn       = make_span(current_assignment);
  auto fixed_assgn = make_span(presolve_data.fixed_var_assignment);
  auto var_map     = make_span(presolve_data.variable_mapping);
  if (current_assignment.size() > 0) {
    thrust::for_each(handle_ptr->get_thrust_policy(),
                     thrust::make_counting_iterator<i_t>(0),
                     thrust::make_counting_iterator<i_t>(current_assignment.size()),
                     [fixed_assgn, var_map, assgn] __device__(auto idx) {
                       fixed_assgn[var_map[idx]] = assgn[idx];
                     });
  }
  expand_device_copy(
    current_assignment, presolve_data.fixed_var_assignment, handle_ptr->get_stream());
  auto h_assignment = cuopt::host_copy(current_assignment, handle_ptr->get_stream());
  cuopt_assert(presolve_data.additional_var_id_per_var.size() == h_assignment.size(),
               "Size mismatch");
  cuopt_assert(presolve_data.additional_var_used.size() == h_assignment.size(), "Size mismatch");
  for (i_t i = 0; i < (i_t)h_assignment.size(); ++i) {
    if (presolve_data.additional_var_used[i]) {
      cuopt_assert(presolve_data.additional_var_id_per_var[i] != -1,
                   "additional_var_id_per_var is not set");
      h_assignment[i] -= h_assignment[presolve_data.additional_var_id_per_var[i]];
    }
  }
  raft::copy(
    current_assignment.data(), h_assignment.data(), h_assignment.size(), handle_ptr->get_stream());
  // this separate resizing is needed because of the callback
  if (resize_to_original_problem) {
    current_assignment.resize(original_problem_ptr->get_n_variables(), handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::post_process_solution(solution_t<i_t, f_t>& solution)
{
  post_process_assignment(solution.assignment);
  // this is for resizing other fields such as excess, slack so that we can compute the feasibility
  solution.resize_to_original_problem();
  handle_ptr->sync_stream();
  solution.post_process_completed = true;
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::recompute_auxilliary_data(bool check_representation)
{
  compute_n_integer_vars();
  compute_binary_var_table();
  compute_vars_with_objective_coeffs();
  // TODO: speedup compute related variables
  const double time_limit = 30.;
  compute_related_variables(time_limit);
  if (check_representation) check_problem_representation(true);
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::compute_n_integer_vars()
{
  cuopt_assert(n_variables == variable_types.size(), "size mismatch");
  integer_indices.resize(n_variables, handle_ptr->get_stream());
  auto end =
    thrust::copy_if(handle_ptr->get_thrust_policy(),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(i_t(variable_types.size())),
                    variable_types.begin(),
                    integer_indices.begin(),
                    [] __host__ __device__(var_t var_type) { return var_type == var_t::INTEGER; });

  n_integer_vars = end - integer_indices.begin();
  // Resize indices vector to the actual number of matching indices
  integer_indices.resize(n_integer_vars, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::is_integer(f_t val) const
{
  return raft::abs(round(val) - (val)) <= tolerances.integrality_tolerance;
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::integer_equal(f_t val1, f_t val2) const
{
  return raft::abs(val1 - val2) <= tolerances.integrality_tolerance;
}

// TODO consider variables that have u - l == 1 as binary
// include that in preprocessing and offset the l to make it true binary
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::compute_binary_var_table()
{
  auto pb_view = view();

  is_binary_variable.resize(n_variables, handle_ptr->get_stream());
  thrust::tabulate(handle_ptr->get_thrust_policy(),
                   is_binary_variable.begin(),
                   is_binary_variable.end(),
                   [pb_view] __device__(i_t i) {
                     auto bounds = pb_view.variable_bounds[i];
                     return pb_view.variable_types[i] != var_t::CONTINUOUS &&
                            (pb_view.integer_equal(get_lower(bounds), 0) &&
                             pb_view.integer_equal(get_upper(bounds), 1));
                   });
  get_n_binary_variables();

  binary_indices.resize(n_variables, handle_ptr->get_stream());
  auto end = thrust::copy_if(handle_ptr->get_thrust_policy(),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(i_t(is_binary_variable.size())),
                             is_binary_variable.begin(),
                             binary_indices.begin(),
                             [] __host__ __device__(i_t is_bin) { return is_bin; });
  binary_indices.resize(end - binary_indices.begin(), handle_ptr->get_stream());

  nonbinary_indices.resize(n_variables, handle_ptr->get_stream());
  end = thrust::copy_if(handle_ptr->get_thrust_policy(),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(i_t(is_binary_variable.size())),
                        is_binary_variable.begin(),
                        nonbinary_indices.begin(),
                        [] __host__ __device__(i_t is_bin) { return !is_bin; });
  nonbinary_indices.resize(end - nonbinary_indices.begin(), handle_ptr->get_stream());

  is_binary_pb =
    n_variables ==
    thrust::count(
      handle_ptr->get_thrust_policy(), is_binary_variable.begin(), is_binary_variable.end(), 1);
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::compute_related_variables(double time_limit)
{
  if (n_variables == 0) {
    related_variables.resize(0, handle_ptr->get_stream());
    related_variables_offsets.resize(0, handle_ptr->get_stream());
    return;
  }
  auto pb_view = view();

  handle_ptr->sync_stream();

  // previously used constants were based on 40GB of memory. Scale accordingly on smaller GPUs
  // We can't rely on querying free memory or allocation try/catch
  // since this would break determinism guarantees (GPU may be shared by other processes)
  f_t size_factor = std::min(1.0, cuopt::get_device_memory_size() / 1e9 / 40.0);

  // TODO: determine optimal number of slices based on available GPU memory? This used to be 2e9 /
  // n_variables
  i_t max_slice_size = 6e8 * size_factor / n_variables;

  rmm::device_uvector<i_t> varmap(max_slice_size * n_variables, handle_ptr->get_stream());
  rmm::device_uvector<i_t> offsets(max_slice_size * n_variables, handle_ptr->get_stream());

  related_variables.resize(0, handle_ptr->get_stream());
  // TODO: this used to be 1e8
  related_variables.reserve(1e8 * size_factor, handle_ptr->get_stream());  // reserve space
  related_variables_offsets.resize(n_variables + 1, handle_ptr->get_stream());
  related_variables_offsets.set_element_to_zero_async(0, handle_ptr->get_stream());

  // compaction operation to get the related variable values
  auto repeating_counting_iterator = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0),
    cuda::proclaim_return_type<i_t>(
      [n_v = n_variables] __device__(i_t x) -> i_t { return x % n_v; }));

  i_t output_offset      = 0;
  i_t related_var_offset = 0;
  auto start_time        = std::chrono::high_resolution_clock::now();
  for (i_t i = 0;; ++i) {
    i_t slice_size = std::min(max_slice_size, n_variables - i * max_slice_size);
    if (slice_size <= 0) break;

    i_t slice_begin = i * max_slice_size;
    i_t slice_end   = slice_begin + slice_size;

    CUOPT_LOG_TRACE("Iter %d: %d [%d %d] alloc'd %gmb",
                    i,
                    slice_size,
                    slice_begin,
                    slice_end,
                    related_variables.size() / (f_t)1e6);

    thrust::fill(handle_ptr->get_thrust_policy(), varmap.begin(), varmap.end(), 0);
    compute_related_vars_unique<i_t, f_t><<<1024, 128, 0, handle_ptr->get_stream()>>>(
      pb_view, slice_begin, slice_end, make_span(varmap));

    // prefix sum to generate offsets
    thrust::inclusive_scan(handle_ptr->get_thrust_policy(),
                           varmap.begin(),
                           varmap.begin() + slice_size * n_variables,
                           offsets.begin());
    // get the required allocation size
    i_t array_size       = offsets.element(slice_size * n_variables - 1, handle_ptr->get_stream());
    i_t related_var_base = related_variables.size();
    related_variables.resize(related_variables.size() + array_size, handle_ptr->get_stream());

    auto current_time = std::chrono::high_resolution_clock::now();
    // if the related variable array would wind up being too large for available memory, abort
    // TODO this used to be 1e9
    if (related_variables.size() > 1e9 * size_factor ||
        std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() >
          time_limit) {
      CUOPT_LOG_DEBUG(
        "Computing the related variable array would use too much memory or time, aborting\n");
      related_variables.resize(0, handle_ptr->get_stream());
      related_variables_offsets.resize(0, handle_ptr->get_stream());
      return;
    }

    auto end =
      thrust::copy_if(handle_ptr->get_thrust_policy(),
                      repeating_counting_iterator,
                      repeating_counting_iterator + varmap.size(),
                      varmap.begin(),
                      related_variables.begin() + related_var_offset,
                      cuda::proclaim_return_type<bool>([] __device__(i_t x) { return x == 1; }));
    related_var_offset = end - related_variables.begin();

    // generate the related var offsets from the prefix sum
    auto offset_it = related_variables_offsets.begin() + 1 + output_offset;
    thrust::tabulate(handle_ptr->get_thrust_policy(),
                     offset_it,
                     offset_it + slice_size,
                     cuda::proclaim_return_type<i_t>(
                       [related_var_base, offsets = offsets.data(), n_v = n_variables] __device__(
                         i_t x) -> i_t { return related_var_base + offsets[(x + 1) * n_v - 1]; }));

    output_offset += slice_size;
  }
  cuopt_assert(related_var_offset == related_variables.size(), "");
  cuopt_assert(output_offset + 1 == related_variables_offsets.size(), "");

  handle_ptr->sync_stream();
  CUOPT_LOG_TRACE("GPU done");
}

template <typename i_t, typename f_t>
typename problem_t<i_t, f_t>::view_t problem_t<i_t, f_t>::view()
{
  problem_t<i_t, f_t>::view_t v;
  v.tolerances     = tolerances;
  v.n_variables    = n_variables;
  v.n_integer_vars = n_integer_vars;
  v.n_constraints  = n_constraints;
  v.nnz            = nnz;
  v.reverse_coefficients =
    raft::device_span<f_t>{reverse_coefficients.data(), reverse_coefficients.size()};
  v.reverse_constraints =
    raft::device_span<i_t>{reverse_constraints.data(), reverse_constraints.size()};
  v.reverse_offsets = raft::device_span<i_t>{reverse_offsets.data(), reverse_offsets.size()};
  v.coefficients    = raft::device_span<f_t>{coefficients.data(), coefficients.size()};
  v.variables       = raft::device_span<i_t>{variables.data(), variables.size()};
  v.offsets         = raft::device_span<i_t>{offsets.data(), offsets.size()};
  v.objective_coefficients =
    raft::device_span<f_t>{objective_coefficients.data(), objective_coefficients.size()};
  v.variable_bounds = make_span(variable_bounds);
  v.constraint_lower_bounds =
    raft::device_span<f_t>{constraint_lower_bounds.data(), constraint_lower_bounds.size()};
  v.constraint_upper_bounds =
    raft::device_span<f_t>{constraint_upper_bounds.data(), constraint_upper_bounds.size()};
  v.variable_types = raft::device_span<var_t>{variable_types.data(), variable_types.size()};
  v.is_binary_variable =
    raft::device_span<i_t>{is_binary_variable.data(), is_binary_variable.size()};
  v.related_variables = raft::device_span<i_t>{related_variables.data(), related_variables.size()};
  v.related_variables_offsets =
    raft::device_span<i_t>{related_variables_offsets.data(), related_variables_offsets.size()};
  v.integer_indices   = raft::device_span<i_t>{integer_indices.data(), integer_indices.size()};
  v.binary_indices    = raft::device_span<i_t>{binary_indices.data(), binary_indices.size()};
  v.nonbinary_indices = raft::device_span<i_t>{nonbinary_indices.data(), nonbinary_indices.size()};
  v.objective_offset  = presolve_data.objective_offset;
  v.objective_scaling_factor = presolve_data.objective_scaling_factor;
  return v;
}

// TODO think about overallocating
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::resize_variables(size_t size)
{
  variable_bounds.resize(size, handle_ptr->get_stream());
  variable_types.resize(size, handle_ptr->get_stream());
  objective_coefficients.resize(size, handle_ptr->get_stream());
  is_binary_variable.resize(size, handle_ptr->get_stream());
  related_variables_offsets.resize(size, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::resize_constraints(size_t matrix_size,
                                             size_t constraint_size,
                                             size_t n_variables)
{
  coefficients.resize(matrix_size, handle_ptr->get_stream());
  variables.resize(matrix_size, handle_ptr->get_stream());
  reverse_constraints.resize(matrix_size, handle_ptr->get_stream());
  reverse_coefficients.resize(matrix_size, handle_ptr->get_stream());
  cuopt_assert(offsets.size() == constraint_lower_bounds.size() + 1, "size mismatch");
  constraint_lower_bounds.resize(constraint_size, handle_ptr->get_stream());
  constraint_upper_bounds.resize(constraint_size, handle_ptr->get_stream());
  combined_bounds.resize(constraint_size, handle_ptr->get_stream());
  offsets.resize(constraint_size + 1, handle_ptr->get_stream());
  reverse_offsets.resize(n_variables + 1, handle_ptr->get_stream());
}

// note that these don't change the reverse structure
// TODO add a boolean value to change the reverse structures
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::insert_variables(variables_delta_t<i_t, f_t>& h_vars)
{
  CUOPT_LOG_DEBUG("problem added variable size %d prev %d", h_vars.size(), n_variables);
  // resize the variable arrays if it can't fit the variables
  resize_variables(n_variables + h_vars.size());
  raft::copy(variable_bounds.data() + n_variables,
             h_vars.variable_bounds.data(),
             h_vars.variable_bounds.size(),
             handle_ptr->get_stream());
  raft::copy(variable_types.data() + n_variables,
             h_vars.variable_types.data(),
             h_vars.variable_types.size(),
             handle_ptr->get_stream());
  raft::copy(is_binary_variable.data() + n_variables,
             h_vars.is_binary_variable.data(),
             h_vars.is_binary_variable.size(),
             handle_ptr->get_stream());
  raft::copy(objective_coefficients.data() + n_variables,
             h_vars.objective_coefficients.data(),
             h_vars.objective_coefficients.size(),
             handle_ptr->get_stream());
  n_variables += h_vars.size();

  compute_n_integer_vars();
  compute_binary_var_table();
  compute_vars_with_objective_coeffs();
}

// note that these don't change the reverse structure
// TODO add a boolean value to change the reverse structures
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::insert_constraints(constraints_delta_t<i_t, f_t>& h_constraints)
{
  CUOPT_LOG_DEBUG(
    "added nnz %d constraints %d  offset size %d prev nnz %d prev cstr %d prev offset size%d ",
    h_constraints.matrix_size(),
    h_constraints.n_constraints(),
    h_constraints.constraint_offsets.size(),
    nnz,
    n_constraints,
    offsets.size());

  resize_constraints(
    h_constraints.matrix_size() + nnz, h_constraints.n_constraints() + n_constraints, n_variables);
  raft::copy(constraint_lower_bounds.data() + n_constraints,
             h_constraints.constraint_lower_bounds.data(),
             h_constraints.constraint_lower_bounds.size(),
             handle_ptr->get_stream());
  raft::copy(constraint_upper_bounds.data() + n_constraints,
             h_constraints.constraint_upper_bounds.data(),
             h_constraints.constraint_upper_bounds.size(),
             handle_ptr->get_stream());
  // get the last offset of the current constraint and append to it
  // avoid using back() or variables.size() because the size of the vectors might be different
  // than the implied problem size as we might be overallocating
  i_t last_offset = offsets.element(n_constraints, handle_ptr->get_stream());
  std::transform(h_constraints.constraint_offsets.begin(),
                 h_constraints.constraint_offsets.end(),
                 h_constraints.constraint_offsets.begin(),
                 [last_offset](int x) { return x + last_offset; });
  raft::copy(offsets.data() + n_constraints + 1,
             // skip the first element
             h_constraints.constraint_offsets.data() + 1,
             h_constraints.constraint_offsets.size() - 1,
             handle_ptr->get_stream());
  raft::copy(variables.data() + nnz,
             h_constraints.constraint_variables.data(),
             h_constraints.constraint_variables.size(),
             handle_ptr->get_stream());
  raft::copy(coefficients.data() + nnz,
             h_constraints.constraint_coefficients.data(),
             h_constraints.constraint_coefficients.size(),
             handle_ptr->get_stream());
  nnz += h_constraints.matrix_size();
  n_constraints += h_constraints.n_constraints();
  cuopt_assert(offsets.element(n_constraints, handle_ptr->get_stream()) == nnz,
               "nnz and offset should match!");
  cuopt_assert(offsets.size() == n_constraints + 1, "offset size should match!");
  combine_constraint_bounds<i_t, f_t>(*this, combined_bounds);
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::fix_given_variables(problem_t<i_t, f_t>& original_problem,
                                              rmm::device_uvector<f_t>& assignment,
                                              const rmm::device_uvector<i_t>& variables_to_fix,
                                              const raft::handle_t* handle_ptr)
{
  fixing_helpers.reduction_in_rhs.resize(n_constraints, handle_ptr->get_stream());
  fixing_helpers.variable_fix_mask.resize(original_problem.n_variables, handle_ptr->get_stream());
  thrust::fill(handle_ptr->get_thrust_policy(),
               fixing_helpers.reduction_in_rhs.begin(),
               fixing_helpers.reduction_in_rhs.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               fixing_helpers.variable_fix_mask.begin(),
               fixing_helpers.variable_fix_mask.end(),
               0);

  thrust::for_each(handle_ptr->get_thrust_policy(),
                   variables_to_fix.begin(),
                   variables_to_fix.end(),
                   [variable_fix_mask = make_span(fixing_helpers.variable_fix_mask)] __device__(
                     i_t x) { variable_fix_mask[x] = 1; });
  const i_t num_segments = original_problem.n_constraints;
  f_t initial_value{0.};

  auto input_transform_it = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [coefficients      = make_span(original_problem.coefficients),
     variables         = make_span(original_problem.variables),
     variable_fix_mask = make_span(fixing_helpers.variable_fix_mask),
     assignment        = make_span(assignment),
     int_tol = original_problem.tolerances.integrality_tolerance] __device__(i_t idx) -> f_t {
      i_t var_idx = variables[idx];
      if (variable_fix_mask[var_idx]) {
        f_t reduction = coefficients[idx] * floor(assignment[var_idx] + int_tol);
        return reduction;
      } else {
        return 0.;
      }
    });
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                     temp_storage_bytes,
                                     input_transform_it,
                                     fixing_helpers.reduction_in_rhs.data(),
                                     num_segments,
                                     original_problem.offsets.data(),
                                     original_problem.offsets.data() + 1,
                                     cuda::std::plus<>{},
                                     initial_value,
                                     handle_ptr->get_stream());

  rmm::device_uvector<std::uint8_t> temp_storage(temp_storage_bytes, handle_ptr->get_stream());
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                     temp_storage_bytes,
                                     input_transform_it,
                                     fixing_helpers.reduction_in_rhs.data(),
                                     num_segments,
                                     original_problem.offsets.data(),
                                     original_problem.offsets.data() + 1,
                                     cuda::std::plus<>{},
                                     initial_value,
                                     handle_ptr->get_stream());
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + n_constraints,
    [lower_bounds          = make_span(constraint_lower_bounds),
     upper_bounds          = make_span(constraint_upper_bounds),
     original_lower_bounds = make_span(original_problem.constraint_lower_bounds),
     original_upper_bounds = make_span(original_problem.constraint_upper_bounds),
     reduction_in_rhs      = make_span(fixing_helpers.reduction_in_rhs)] __device__(i_t cstr_idx) {
      lower_bounds[cstr_idx] = original_lower_bounds[cstr_idx] - reduction_in_rhs[cstr_idx];
      upper_bounds[cstr_idx] = original_upper_bounds[cstr_idx] - reduction_in_rhs[cstr_idx];
    });
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
problem_t<i_t, f_t> problem_t<i_t, f_t>::get_problem_after_fixing_vars(
  rmm::device_uvector<f_t>& assignment,
  const rmm::device_uvector<i_t>& variables_to_fix,
  rmm::device_uvector<i_t>& variable_map,
  const raft::handle_t* handle_ptr)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  cuopt_assert(n_variables == assignment.size(), "Assignment size issue");
  problem_t<i_t, f_t> problem(*this, true);
  CUOPT_LOG_DEBUG("Fixing %d variables", variables_to_fix.size());
  // we will gather from this and scatter back to the original problem
  variable_map.resize(assignment.size() - variables_to_fix.size(), handle_ptr->get_stream());
  // compute variable map to recover the assignment later
  // get the variable indices to gather
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  cuopt_assert(
    (thrust::is_sorted(
      handle_ptr->get_thrust_policy(), variables_to_fix.begin(), variables_to_fix.end())),
    "variables_to_fix should be sorted!");

  i_t* result_end = thrust::set_difference(handle_ptr->get_thrust_policy(),
                                           thrust::make_counting_iterator(0),
                                           thrust::make_counting_iterator(0) + n_variables,
                                           variables_to_fix.begin(),
                                           variables_to_fix.end(),
                                           variable_map.begin());
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  cuopt_assert(result_end - variable_map.data() == variable_map.size(),
               "Size issue in set_difference");
  problem.fix_given_variables(*this, assignment, variables_to_fix, handle_ptr);
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  problem.remove_given_variables(*this, assignment, variable_map, handle_ptr);
  // if we are fixing on the original problem, the variable_map is what we want in
  // problem.original_ids but considering the case that we are fixing some variables multiple times,
  // do an assignment from the original_ids of the current problem
  problem.original_ids.resize(variable_map.size());
  std::fill(problem.reverse_original_ids.begin(), problem.reverse_original_ids.end(), -1);
  auto h_variable_map = cuopt::host_copy(variable_map);
  for (size_t i = 0; i < variable_map.size(); ++i) {
    cuopt_assert(h_variable_map[i] < original_ids.size(), "Variable index out of bounds");
    problem.original_ids[i] = original_ids[h_variable_map[i]];
    cuopt_assert(original_ids[h_variable_map[i]] < reverse_original_ids.size(),
                 "Variable index out of bounds");
    problem.reverse_original_ids[original_ids[h_variable_map[i]]] = i;
  }
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  auto end_time = std::chrono::high_resolution_clock::now();
  double time_taken =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  static double total_time_taken = 0.;
  static int total_calls         = 0;
  total_time_taken += time_taken;
  total_calls++;
  CUOPT_LOG_DEBUG(
    "Time taken to fix variables: %f milliseconds, average: %f milliseconds total time: %f",
    time_taken,
    total_time_taken / total_calls,
    total_time_taken);
  return problem;
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::remove_given_variables(problem_t<i_t, f_t>& original_problem,
                                                 rmm::device_uvector<f_t>& assignment,
                                                 rmm::device_uvector<i_t>& variable_map,
                                                 const raft::handle_t* handle_ptr)
{
  thrust::fill(handle_ptr->get_thrust_policy(), offsets.begin(), offsets.end(), 0);
  cuopt_assert(assignment.size() == n_variables, "Variable size mismatch");
  cuopt_assert(variable_map.size() < n_variables, "Too many variables to fix");
  rmm::device_uvector<f_t> tmp_assignment(assignment, handle_ptr->get_stream());

  // first remove the assignment and variable related vectors
  thrust::gather(handle_ptr->get_thrust_policy(),
                 variable_map.begin(),
                 variable_map.end(),
                 tmp_assignment.begin(),
                 assignment.begin());
  assignment.resize(variable_map.size(), handle_ptr->get_stream());
  thrust::gather(handle_ptr->get_thrust_policy(),
                 variable_map.begin(),
                 variable_map.end(),
                 original_problem.variable_bounds.begin(),
                 variable_bounds.begin());
  variable_bounds.resize(variable_map.size(), handle_ptr->get_stream());
  thrust::gather(handle_ptr->get_thrust_policy(),
                 variable_map.begin(),
                 variable_map.end(),
                 original_problem.objective_coefficients.begin(),
                 objective_coefficients.begin());
  objective_coefficients.resize(variable_map.size(), handle_ptr->get_stream());
  thrust::gather(handle_ptr->get_thrust_policy(),
                 variable_map.begin(),
                 variable_map.end(),
                 original_problem.variable_types.begin(),
                 variable_types.begin());
  variable_types.resize(variable_map.size(), handle_ptr->get_stream());
  const i_t TPB = 64;
  // compute new offsets
  compute_new_offsets<i_t, f_t><<<variable_map.size(), TPB, 0, handle_ptr->get_stream()>>>(
    original_problem.view(), view(), cuopt::make_span(variable_map));
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  thrust::exclusive_scan(handle_ptr->get_thrust_policy(),
                         offsets.data(),
                         offsets.data() + offsets.size(),
                         offsets.data());  // in-place scan
  rmm::device_uvector<i_t> write_pos(n_constraints, handle_ptr->get_stream());
  thrust::fill(handle_ptr->get_thrust_policy(), write_pos.begin(), write_pos.end(), 0);
  // compute new csr
  compute_new_csr<i_t, f_t><<<variable_map.size(), TPB, 0, handle_ptr->get_stream()>>>(
    original_problem.view(), view(), cuopt::make_span(variable_map), cuopt::make_span(write_pos));
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  // assign nnz, number of variables etc.
  nnz         = offsets.back_element(handle_ptr->get_stream());
  n_variables = variable_map.size();
  coefficients.resize(nnz, handle_ptr->get_stream());
  variables.resize(nnz, handle_ptr->get_stream());
  compute_transpose_of_problem();
  combine_constraint_bounds<i_t, f_t>(*this, combined_bounds);
  handle_ptr->sync_stream();
  recompute_auxilliary_data();
  check_problem_representation(true);
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t> problem_t<i_t, f_t>::get_fixed_assignment_from_integer_fixed_problem(
  const rmm::device_uvector<f_t>& assignment)
{
  rmm::device_uvector<f_t> fixed_assignment(integer_fixed_variable_map.size(),
                                            handle_ptr->get_stream());
  // first remove the assignment and variable related vectors
  thrust::gather(handle_ptr->get_thrust_policy(),
                 integer_fixed_variable_map.begin(),
                 integer_fixed_variable_map.end(),
                 assignment.begin(),
                 fixed_assignment.begin());
  return fixed_assignment;
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::compute_integer_fixed_problem()
{
  cuopt_assert(integer_fixed_problem == nullptr, "Integer fixed problem already computed");
  if (n_variables == n_integer_vars) {
    integer_fixed_problem = nullptr;
    return;
  }
  rmm::device_uvector<f_t> assignment(n_variables, handle_ptr->get_stream());
  integer_fixed_problem = std::make_shared<problem_t<i_t, f_t>>(get_problem_after_fixing_vars(
    assignment, integer_indices, integer_fixed_variable_map, handle_ptr));
  integer_fixed_problem->check_problem_representation(true);
  integer_fixed_problem->lp_state.resize(*integer_fixed_problem, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::copy_rhs_from_problem(const raft::handle_t* handle_ptr)
{
  raft::copy(integer_fixed_problem->constraint_lower_bounds.data(),
             constraint_lower_bounds.data(),
             integer_fixed_problem->constraint_lower_bounds.size(),
             handle_ptr->get_stream());
  raft::copy(integer_fixed_problem->constraint_upper_bounds.data(),
             constraint_upper_bounds.data(),
             integer_fixed_problem->constraint_upper_bounds.size(),
             handle_ptr->get_stream());
}
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::fill_integer_fixed_problem(rmm::device_uvector<f_t>& assignment,
                                                     const raft::handle_t* handle_ptr)
{
  cuopt_assert(integer_fixed_problem->n_variables > 0, "Integer fixed problem not computed");
  copy_rhs_from_problem(handle_ptr);
  integer_fixed_problem->fix_given_variables(*this, assignment, integer_indices, handle_ptr);
  combine_constraint_bounds<i_t, f_t>(*integer_fixed_problem,
                                      integer_fixed_problem->combined_bounds);
  integer_fixed_problem->check_problem_representation(true);
}

template <typename i_t, typename f_t>
std::vector<std::vector<std::pair<i_t, f_t>>> compute_var_to_constraint_map(
  const problem_t<i_t, f_t>& pb)
{
  std::vector<std::vector<std::pair<i_t, f_t>>> variable_constraint_map(pb.n_variables);
  auto h_variables    = cuopt::host_copy(pb.variables);
  auto h_coefficients = cuopt::host_copy(pb.coefficients);
  auto h_offsets      = cuopt::host_copy(pb.offsets);
  for (i_t cnst = 0; cnst < pb.n_constraints; ++cnst) {
    for (i_t i = h_offsets[cnst]; i < h_offsets[cnst + 1]; ++i) {
      i_t var   = h_variables[i];
      f_t coeff = h_coefficients[i];
      if (coeff != 0.) { variable_constraint_map[var].emplace_back(cnst, coeff); }
    }
  }

  return variable_constraint_map;
}

template <typename i_t, typename f_t>
void standardize_bounds(std::vector<std::vector<std::pair<i_t, f_t>>>& variable_constraint_map,
                        problem_t<i_t, f_t>& pb)
{
  auto handle_ptr               = pb.handle_ptr;
  auto h_var_bounds             = cuopt::host_copy(pb.variable_bounds);
  auto h_objective_coefficients = cuopt::host_copy(pb.objective_coefficients);
  auto h_variable_types         = cuopt::host_copy(pb.variable_types);
  handle_ptr->sync_stream();

  const i_t n_vars_originally = (i_t)h_var_bounds.size();

  for (i_t i = 0; i < n_vars_originally; ++i) {
    // if variable has free bounds, replace it with two vars
    // but add only one var and use it in all constraints
    // TODO create one var for integrals and one var for continuous
    auto h_var_bound = h_var_bounds[i];
    if (get_lower(h_var_bound) == -std::numeric_limits<f_t>::infinity() &&
        get_upper(h_var_bound) == std::numeric_limits<f_t>::infinity()) {
      // add new variable
      auto var_coeff_vec = variable_constraint_map[i];
      // negate all values in vec
      for (auto& [constr, coeff] : var_coeff_vec) {
        coeff = -coeff;
      }

      h_var_bounds[i].x                             = 0.;
      pb.presolve_data.variable_offsets[i]          = 0.;
      pb.presolve_data.additional_var_used[i]       = true;
      pb.presolve_data.additional_var_id_per_var[i] = pb.n_variables;

      using f_t2 = typename type_2<f_t>::type;
      // new var data
      std::stable_sort(var_coeff_vec.begin(), var_coeff_vec.end());
      variable_constraint_map.push_back(var_coeff_vec);
      h_var_bounds.push_back(f_t2{0., std::numeric_limits<f_t>::infinity()});
      pb.presolve_data.variable_offsets.push_back(0.);
      h_objective_coefficients.push_back(-h_objective_coefficients[i]);
      h_variable_types.push_back(h_variable_types[i]);
      pb.presolve_data.additional_var_used.push_back(false);
      pb.presolve_data.additional_var_id_per_var.push_back(-1);
      pb.n_variables++;
    }
  }

  if (pb.presolve_data.additional_var_id_per_var.size() > (size_t)n_vars_originally) {
    CUOPT_LOG_INFO("Free variable found! Make sure the correct bounds are given.");
  }
  // TODO add some tests

  // resize the device vectors is sizes are smaller
  if (pb.variable_bounds.size() < h_var_bounds.size()) {
    pb.variable_bounds.resize(h_var_bounds.size(), handle_ptr->get_stream());
    pb.objective_coefficients.resize(h_objective_coefficients.size(), handle_ptr->get_stream());
    pb.variable_types.resize(h_variable_types.size(), handle_ptr->get_stream());
  }

  raft::copy(
    pb.variable_bounds.data(), h_var_bounds.data(), h_var_bounds.size(), handle_ptr->get_stream());
  raft::copy(pb.objective_coefficients.data(),
             h_objective_coefficients.data(),
             h_objective_coefficients.size(),
             handle_ptr->get_stream());
  raft::copy(pb.variable_types.data(),
             h_variable_types.data(),
             h_variable_types.size(),
             handle_ptr->get_stream());
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void compute_csr(const std::vector<std::vector<std::pair<i_t, f_t>>>& variable_constraint_map,
                 problem_t<i_t, f_t>& pb)
{
  auto handle_ptr = pb.handle_ptr;
  std::vector<std::vector<i_t>> vars_per_constraint(pb.n_constraints);
  std::vector<std::vector<f_t>> coefficient_per_constraint(pb.n_constraints);
  // fill the reverse vectors
  for (i_t v = 0; v < (i_t)variable_constraint_map.size(); ++v) {
    const auto& vec = variable_constraint_map[v];
    for (auto const& [constr, coeff] : vec) {
      vars_per_constraint[constr].push_back(v);
      coefficient_per_constraint[constr].push_back(coeff);
      cuopt_assert(coeff != 0., "Coeff cannot be zero");
    }
  }
  std::vector<i_t> h_offsets;
  std::vector<i_t> h_variables;
  std::vector<f_t> h_coefficients;
  h_offsets.push_back(0);
  for (i_t c = 0; c < (i_t)vars_per_constraint.size(); ++c) {
    const auto coeff_vec = coefficient_per_constraint[c];
    const auto var_vec   = vars_per_constraint[c];
    h_offsets.push_back(coeff_vec.size() + h_offsets.back());
    h_variables.insert(h_variables.end(), var_vec.begin(), var_vec.end());
    h_coefficients.insert(h_coefficients.end(), coeff_vec.begin(), coeff_vec.end());
  }
  cuopt_assert(h_offsets.back() == h_variables.size(), "Sizes should match!");
  pb.nnz = h_offsets.back();
  // resize the device vectors is sizes are smaller
  pb.coefficients.resize(h_coefficients.size(), handle_ptr->get_stream());
  pb.variables.resize(h_coefficients.size(), handle_ptr->get_stream());
  pb.offsets.resize(h_offsets.size(), handle_ptr->get_stream());
  raft::copy(
    pb.coefficients.data(), h_coefficients.data(), h_coefficients.size(), handle_ptr->get_stream());
  raft::copy(pb.variables.data(), h_variables.data(), h_variables.size(), handle_ptr->get_stream());
  raft::copy(pb.offsets.data(), h_offsets.data(), h_offsets.size(), handle_ptr->get_stream());
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::preprocess_problem()
{
  auto variable_constraint_map = compute_var_to_constraint_map(*this);
  standardize_bounds(variable_constraint_map, *this);
  compute_csr(variable_constraint_map, *this);
  compute_transpose_of_problem();
  check_problem_representation(true, false);
  presolve_data.initialize_var_mapping(*this, handle_ptr);
  integer_indices.resize(n_variables, handle_ptr->get_stream());
  is_binary_variable.resize(n_variables, handle_ptr->get_stream());
  original_ids.resize(n_variables);
  std::iota(original_ids.begin(), original_ids.end(), 0);
  reverse_original_ids.resize(n_variables);
  std::iota(reverse_original_ids.begin(), reverse_original_ids.end(), 0);
  compute_n_integer_vars();
  compute_binary_var_table();
  check_problem_representation(true);
  preprocess_called = true;
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::get_host_user_problem(
  cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t>& user_problem) const
{
  i_t m                  = n_constraints;
  i_t n                  = n_variables;
  i_t nz                 = nnz;
  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = cuopt::host_copy(objective_coefficients);

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A;
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.nz_max    = nz;
  csr_A.x         = cuopt::host_copy(coefficients);
  csr_A.j         = cuopt::host_copy(variables);
  csr_A.row_start = cuopt::host_copy(offsets);

  csr_A.to_compressed_col(user_problem.A);

  user_problem.rhs.resize(m);
  user_problem.row_sense.resize(m);
  user_problem.range_rows.clear();
  user_problem.range_value.clear();

  auto model_constraint_lower_bounds = cuopt::host_copy(constraint_lower_bounds);
  auto model_constraint_upper_bounds = cuopt::host_copy(constraint_upper_bounds);

  // All constraints have lower and upper bounds
  // lr <= a_i^T x <= ur
  for (i_t i = 0; i < m; ++i) {
    const f_t constraint_lower_bound = model_constraint_lower_bounds[i];
    const f_t constraint_upper_bound = model_constraint_upper_bounds[i];
    if (constraint_lower_bound == constraint_upper_bound) {
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_upper_bound == std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'G';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_lower_bound == -std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'L';
      user_problem.rhs[i]       = constraint_upper_bound;
    } else {
      // This is range row
      assert(constraint_lower_bound < constraint_upper_bound);
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
      user_problem.range_rows.push_back(i);
      const double bound_difference = constraint_upper_bound - constraint_lower_bound;
      assert(bound_difference > 0);
      user_problem.range_value.push_back(bound_difference);
    }
  }
  user_problem.num_range_rows = user_problem.range_rows.size();
  std::tie(user_problem.lower, user_problem.upper) =
    extract_host_bounds<f_t>(variable_bounds, handle_ptr);
  user_problem.problem_name = original_problem_ptr->get_problem_name();
  if (static_cast<i_t>(row_names.size()) == m) {
    user_problem.row_names.resize(m);
    for (int i = 0; i < m; ++i) {
      user_problem.row_names[i] = row_names[i];
    }
  } else {
    user_problem.row_names.resize(m);
    for (i_t i = 0; i < m; ++i) {
      std::stringstream ss;
      ss << "c" << i;
      user_problem.row_names[i] = ss.str();
    }
  }
  if (static_cast<i_t>(var_names.size()) == n) {
    user_problem.col_names.resize(n);
    for (i_t j = 0; j < n; ++j) {
      user_problem.col_names[j] = var_names[j];
    }
  } else {
    user_problem.col_names.resize(n);
    for (i_t j = 0; j < n; ++j) {
      std::stringstream ss;
      ss << "x" << j;
      user_problem.col_names[j] = ss.str();
    }
  }
  user_problem.obj_constant = presolve_data.objective_offset;
  user_problem.obj_scale    = presolve_data.objective_scaling_factor;
  user_problem.var_types.resize(n);

  auto model_variable_types = cuopt::host_copy(variable_types);
  for (int j = 0; j < n; ++j) {
    user_problem.var_types[j] =
      model_variable_types[j] == var_t::CONTINUOUS
        ? cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS
        : cuopt::linear_programming::dual_simplex::variable_type_t::INTEGER;
  }
}
template <typename i_t, typename f_t>
f_t problem_t<i_t, f_t>::get_user_obj_from_solver_obj(f_t solver_obj)
{
  return presolve_data.objective_scaling_factor * (solver_obj + presolve_data.objective_offset);
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::compute_vars_with_objective_coeffs()
{
  auto h_objective_coefficients = cuopt::host_copy(objective_coefficients);
  std::vector<i_t> vars_with_objective_coeffs_;
  std::vector<f_t> objective_coeffs_;
  for (i_t i = 0; i < n_variables; ++i) {
    if (h_objective_coefficients[i] != 0) {
      vars_with_objective_coeffs_.push_back(i);
      objective_coeffs_.push_back(h_objective_coefficients[i]);
    }
  }
  vars_with_objective_coeffs = std::make_pair(vars_with_objective_coeffs_, objective_coeffs_);
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::add_cutting_plane_at_objective(f_t objective)
{
  CUOPT_LOG_DEBUG("Adding cutting plane at objective %f", objective);
  if (cutting_plane_added) {
    // modify the RHS
    i_t last_constraint = n_constraints - 1;
    constraint_upper_bounds.set_element_async(last_constraint, objective, handle_ptr->get_stream());
    return;
  }
  cutting_plane_added = true;
  constraints_delta_t<i_t, f_t> h_constraints;
  h_constraints.add_constraint(vars_with_objective_coeffs.first,
                               vars_with_objective_coeffs.second,
                               -std::numeric_limits<f_t>::infinity(),
                               objective);
  insert_constraints(h_constraints);
  compute_transpose_of_problem();
  cuopt_func_call(check_problem_representation(true));
}

#if MIP_INSTANTIATE_FLOAT
template class problem_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class problem_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
