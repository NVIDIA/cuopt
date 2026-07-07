/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <pdlp/cuopt_c_internal.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace cuopt::mathematical_optimization;

namespace {

problem_and_stream_view_t* as_problem(cuOptOptimizationProblem problem)
{
  return static_cast<problem_and_stream_view_t*>(problem);
}

optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>* get_iface(
  cuOptOptimizationProblem problem)
{
  return as_problem(problem)->get_problem();
}

bool is_int_attribute(cuopt_int_t attribute)
{
  switch (attribute) {
    case CUOPT_ATTR_NUM_VARIABLES:
    case CUOPT_ATTR_NUM_CONSTRAINTS:
    case CUOPT_ATTR_NUM_NONZEROS:
    case CUOPT_ATTR_NUM_INTEGERS:
    case CUOPT_ATTR_OBJECTIVE_SENSE:
    case CUOPT_ATTR_PROBLEM_CATEGORY:
    case CUOPT_ATTR_IS_MIP:
    case CUOPT_ATTR_HAS_QUADRATIC_OBJECTIVE:
    case CUOPT_ATTR_HAS_QUADRATIC_CONSTRAINTS: return true;
    default: return false;
  }
}

bool is_float_attribute(cuopt_int_t attribute)
{
  return attribute == CUOPT_ATTR_OBJECTIVE_OFFSET ||
         attribute == CUOPT_ATTR_OBJECTIVE_SCALING_FACTOR;
}

cuopt_int_t get_array_size(optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>* problem,
                           cuopt_int_t attribute)
{
  switch (attribute) {
    case CUOPT_ARRAY_ATTR_OBJECTIVE_COEFFICIENTS:
    case CUOPT_ARRAY_ATTR_VARIABLE_LOWER_BOUNDS:
    case CUOPT_ARRAY_ATTR_VARIABLE_UPPER_BOUNDS:
    case CUOPT_ARRAY_ATTR_VARIABLE_TYPES: return problem->get_n_variables();
    case CUOPT_ARRAY_ATTR_CONSTRAINT_LOWER_BOUNDS:
    case CUOPT_ARRAY_ATTR_CONSTRAINT_UPPER_BOUNDS:
    case CUOPT_ARRAY_ATTR_CONSTRAINT_RHS:
    case CUOPT_ARRAY_ATTR_CONSTRAINT_SENSE: return problem->get_n_constraints();
    default: return -1;
  }
}

}  // namespace

cuopt_int_t cuOptGetProblemIntAttribute(cuOptOptimizationProblem problem,
                                        cuopt_int_t attribute,
                                        cuopt_int_t* value_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (value_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (!is_int_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }

  auto* iface = get_iface(problem);
  switch (attribute) {
    case CUOPT_ATTR_NUM_VARIABLES: *value_out = iface->get_n_variables(); return CUOPT_SUCCESS;
    case CUOPT_ATTR_NUM_CONSTRAINTS: *value_out = iface->get_n_constraints(); return CUOPT_SUCCESS;
    case CUOPT_ATTR_NUM_NONZEROS: *value_out = iface->get_nnz(); return CUOPT_SUCCESS;
    case CUOPT_ATTR_NUM_INTEGERS: *value_out = iface->get_n_integers(); return CUOPT_SUCCESS;
    case CUOPT_ATTR_OBJECTIVE_SENSE:
      *value_out = iface->get_sense() ? CUOPT_MAXIMIZE : CUOPT_MINIMIZE;
      return CUOPT_SUCCESS;
    case CUOPT_ATTR_PROBLEM_CATEGORY:
      *value_out = static_cast<cuopt_int_t>(iface->get_problem_category());
      return CUOPT_SUCCESS;
    case CUOPT_ATTR_IS_MIP: {
      const auto category = iface->get_problem_category();
      *value_out =
        (category == problem_category_t::MIP || category == problem_category_t::IP) ? 1 : 0;
      return CUOPT_SUCCESS;
    }
    case CUOPT_ATTR_HAS_QUADRATIC_OBJECTIVE:
      *value_out = iface->has_quadratic_objective() ? 1 : 0;
      return CUOPT_SUCCESS;
    case CUOPT_ATTR_HAS_QUADRATIC_CONSTRAINTS:
      *value_out = iface->has_quadratic_constraints() ? 1 : 0;
      return CUOPT_SUCCESS;
    default: return CUOPT_INVALID_ARGUMENT;
  }
}

cuopt_int_t cuOptGetProblemFloatAttribute(cuOptOptimizationProblem problem,
                                          cuopt_int_t attribute,
                                          cuopt_float_t* value_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (value_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (!is_float_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }

  auto* iface = get_iface(problem);
  if (attribute == CUOPT_ATTR_OBJECTIVE_OFFSET) {
    *value_out = iface->get_objective_offset();
  } else {
    *value_out = iface->get_objective_scaling_factor();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemFloatArrayAttribute(cuOptOptimizationProblem problem,
                                               cuopt_int_t attribute,
                                               cuopt_float_t* out,
                                               cuopt_int_t count)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (out == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* iface                = get_iface(problem);
  const cuopt_int_t expected = get_array_size(iface, attribute);
  if (expected < 0 || count != expected) { return CUOPT_INVALID_ARGUMENT; }

  std::vector<cuopt_float_t> values;
  switch (attribute) {
    case CUOPT_ARRAY_ATTR_OBJECTIVE_COEFFICIENTS:
      values = iface->get_objective_coefficients_host();
      break;
    case CUOPT_ARRAY_ATTR_VARIABLE_LOWER_BOUNDS:
      values = iface->get_variable_lower_bounds_host();
      break;
    case CUOPT_ARRAY_ATTR_VARIABLE_UPPER_BOUNDS:
      values = iface->get_variable_upper_bounds_host();
      break;
    case CUOPT_ARRAY_ATTR_CONSTRAINT_LOWER_BOUNDS:
      values = iface->get_constraint_lower_bounds_host();
      break;
    case CUOPT_ARRAY_ATTR_CONSTRAINT_UPPER_BOUNDS:
      values = iface->get_constraint_upper_bounds_host();
      break;
    case CUOPT_ARRAY_ATTR_CONSTRAINT_RHS: values = iface->get_constraint_bounds_host(); break;
    default: return CUOPT_INVALID_ARGUMENT;
  }

  if (static_cast<cuopt_int_t>(values.size()) != expected) { return CUOPT_VALIDATION_ERROR; }
  std::copy(values.begin(), values.end(), out);
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemCharArrayAttribute(cuOptOptimizationProblem problem,
                                              cuopt_int_t attribute,
                                              char* out,
                                              cuopt_int_t count)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (out == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* iface                = get_iface(problem);
  const cuopt_int_t expected = get_array_size(iface, attribute);
  if (expected < 0 || count != expected) { return CUOPT_INVALID_ARGUMENT; }

  if (attribute == CUOPT_ARRAY_ATTR_CONSTRAINT_SENSE) {
    const std::vector<char> row_types = iface->get_row_types_host();
    if (static_cast<cuopt_int_t>(row_types.size()) != expected) { return CUOPT_VALIDATION_ERROR; }
    std::copy(row_types.begin(), row_types.end(), out);
  } else if (attribute == CUOPT_ARRAY_ATTR_VARIABLE_TYPES) {
    const std::vector<var_t> var_types = iface->get_variable_types_host();
    if (static_cast<cuopt_int_t>(var_types.size()) != expected) { return CUOPT_VALIDATION_ERROR; }
    for (cuopt_int_t i = 0; i < count; ++i) {
      out[i] = var_type_to_char(var_types[static_cast<std::size_t>(i)]);
    }
  } else {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemStringArrayAttribute(cuOptOptimizationProblem problem,
                                                cuopt_int_t attribute,
                                                const char** strings_out,
                                                cuopt_int_t count)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (strings_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (attribute != CUOPT_STRING_ARRAY_VARIABLE_NAMES && attribute != CUOPT_STRING_ARRAY_ROW_NAMES) {
    return CUOPT_INVALID_ARGUMENT;
  }

  auto* iface       = get_iface(problem);
  const auto& names = (attribute == CUOPT_STRING_ARRAY_VARIABLE_NAMES) ? iface->get_variable_names()
                                                                       : iface->get_row_names();

  if (count != static_cast<cuopt_int_t>(names.size())) { return CUOPT_INVALID_ARGUMENT; }
  for (cuopt_int_t i = 0; i < count; ++i) {
    strings_out[i] = names[static_cast<std::size_t>(i)].c_str();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintMatrixCSC(cuOptOptimizationProblem problem,
                                        cuopt_int_t* column_offsets_ptr,
                                        cuopt_int_t* row_indices_ptr,
                                        cuopt_float_t* values_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (column_offsets_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* iface         = get_iface(problem);
  const cuopt_int_t n = iface->get_n_variables();
  const cuopt_int_t m = iface->get_n_constraints();

  const std::vector<cuopt_int_t> row_offsets = iface->get_constraint_matrix_offsets_host();
  const std::vector<cuopt_int_t> col_indices = iface->get_constraint_matrix_indices_host();
  const std::vector<cuopt_float_t> values    = iface->get_constraint_matrix_values_host();
  const cuopt_int_t nnz                      = static_cast<cuopt_int_t>(values.size());

  // Empty / unset matrix: emit all-zero column offsets and nothing else.
  if (row_offsets.size() < static_cast<std::size_t>(m + 1) || nnz == 0) {
    for (cuopt_int_t c = 0; c <= n; ++c) {
      column_offsets_ptr[c] = 0;
    }
    return CUOPT_SUCCESS;
  }
  if (row_indices_ptr == nullptr || values_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  // Count non-zeros per column, then prefix-sum into column offsets.
  std::vector<cuopt_int_t> col_counts(static_cast<std::size_t>(n), 0);
  for (cuopt_int_t k = 0; k < nnz; ++k) {
    const cuopt_int_t c = col_indices[static_cast<std::size_t>(k)];
    if (c < 0 || c >= n) { return CUOPT_VALIDATION_ERROR; }
    ++col_counts[static_cast<std::size_t>(c)];
  }
  column_offsets_ptr[0] = 0;
  for (cuopt_int_t c = 0; c < n; ++c) {
    column_offsets_ptr[c + 1] = column_offsets_ptr[c] + col_counts[static_cast<std::size_t>(c)];
  }

  // Scatter each CSR entry into its CSC position using a running write cursor per column.
  std::vector<cuopt_int_t> next(column_offsets_ptr, column_offsets_ptr + n);
  for (cuopt_int_t i = 0; i < m; ++i) {
    const cuopt_int_t row_begin = row_offsets[static_cast<std::size_t>(i)];
    const cuopt_int_t row_end   = row_offsets[static_cast<std::size_t>(i + 1)];
    for (cuopt_int_t k = row_begin; k < row_end; ++k) {
      const cuopt_int_t c    = col_indices[static_cast<std::size_t>(k)];
      const cuopt_int_t dest = next[static_cast<std::size_t>(c)]++;
      row_indices_ptr[dest]  = i;
      values_ptr[dest]       = values[static_cast<std::size_t>(k)];
    }
  }
  return CUOPT_SUCCESS;
}
