/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuopt/linear_programming/cuopt_c.h>

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/logger.hpp>

#include <mps_parser/parser.hpp>

#include <cuopt/version_config.hpp>

#include <memory>
#include <string>

using namespace cuopt::mps_parser;
using namespace cuopt::linear_programming;

struct problem_and_stream_view_t {
  problem_and_stream_view_t()
    : op_problem(nullptr), stream_view(rmm::cuda_stream_per_thread), handle(stream_view)
  {
  }
  raft::handle_t* get_handle_ptr() { return &handle; }
  cuopt::linear_programming::optimization_problem_t<cuopt_int_t, cuopt_float_t>* op_problem;
  rmm::cuda_stream_view stream_view;
  raft::handle_t handle;
};

struct solution_and_stream_view_t {
  solution_and_stream_view_t(bool solution_for_mip, rmm::cuda_stream_view stream_view)
    : is_mip(solution_for_mip),
      mip_solution_ptr(nullptr),
      lp_solution_ptr(nullptr),
      stream_view(stream_view)
  {
  }
  bool is_mip;
  mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution_ptr;
  optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* lp_solution_ptr;
  rmm::cuda_stream_view stream_view;
};

int8_t cuOptGetFloatSize() { return sizeof(cuopt_float_t); }

int8_t cuOptGetIntSize() { return sizeof(cuopt_int_t); }

cuopt_int_t cuOptGetVersion(cuopt_int_t* version_major,
                            cuopt_int_t* version_minor,
                            cuopt_int_t* version_patch)
{
  if (version_major == nullptr || version_minor == nullptr || version_patch == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }
  *version_major = CUOPT_VERSION_MAJOR;
  *version_minor = CUOPT_VERSION_MINOR;
  *version_patch = CUOPT_VERSION_PATCH;
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptReadProblem(const char* filename, cuOptOptimizationProblem* problem_ptr)
{
  problem_and_stream_view_t* problem_and_stream = new problem_and_stream_view_t();
  std::string filename_str(filename);
  bool input_mps_strict = false;
  std::unique_ptr<mps_data_model_t<cuopt_int_t, cuopt_float_t>> mps_data_model_ptr;
  try {
    mps_data_model_ptr = std::make_unique<mps_data_model_t<cuopt_int_t, cuopt_float_t>>(
      parse_mps<cuopt_int_t, cuopt_float_t>(filename_str, input_mps_strict));
  } catch (const std::exception& e) {
    CUOPT_LOG_INFO("Error parsing MPS file: %s", e.what());
    *problem_ptr = nullptr;
    if (std::string(e.what()).find("Error opening MPS file") != std::string::npos) {
      return CUOPT_MPS_FILE_ERROR;
    } else {
      return CUOPT_MPS_PARSE_ERROR;
    }
  }
  optimization_problem_t<cuopt_int_t, cuopt_float_t>* op_problem =
    new optimization_problem_t<cuopt_int_t, cuopt_float_t>(mps_data_model_to_optimization_problem(
      problem_and_stream->get_handle_ptr(), *mps_data_model_ptr));
  problem_and_stream->op_problem = op_problem;
  *problem_ptr                   = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptCreateProblem(cuopt_int_t num_constraints,
                               cuopt_int_t num_variables,
                               cuopt_int_t objective_sense,
                               cuopt_float_t objective_offset,
                               const cuopt_float_t* objective_coefficients,
                               const cuopt_int_t* constraint_matrix_row_offsets,
                               const cuopt_int_t* constraint_matrix_column_indices,
                               const cuopt_float_t* constraint_matrix_coefficent_values,
                               const char* constraint_sense,
                               const cuopt_float_t* rhs,
                               const cuopt_float_t* lower_bounds,
                               const cuopt_float_t* upper_bounds,
                               const char* variable_types,
                               cuOptOptimizationProblem* problem_ptr)
{
  if (problem_ptr == nullptr || objective_coefficients == nullptr ||
      constraint_matrix_row_offsets == nullptr || constraint_matrix_column_indices == nullptr ||
      constraint_matrix_coefficent_values == nullptr || constraint_sense == nullptr ||
      rhs == nullptr || lower_bounds == nullptr || upper_bounds == nullptr ||
      variable_types == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_and_stream_view_t* problem_and_stream = new problem_and_stream_view_t();
  problem_and_stream->op_problem =
    new optimization_problem_t<cuopt_int_t, cuopt_float_t>(problem_and_stream->get_handle_ptr());
  try {
    problem_and_stream->op_problem->set_maximize(objective_sense == CUOPT_MAXIMIZE);
    problem_and_stream->op_problem->set_objective_offset(objective_offset);
    problem_and_stream->op_problem->set_objective_coefficients(objective_coefficients,
                                                               num_variables);
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];
    problem_and_stream->op_problem->set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                                              nnz,
                                                              constraint_matrix_column_indices,
                                                              nnz,
                                                              constraint_matrix_row_offsets,
                                                              num_constraints + 1);
    problem_and_stream->op_problem->set_row_types(constraint_sense, num_constraints);
    problem_and_stream->op_problem->set_constraint_bounds(rhs, num_constraints);
    problem_and_stream->op_problem->set_variable_lower_bounds(lower_bounds, num_variables);
    problem_and_stream->op_problem->set_variable_upper_bounds(upper_bounds, num_variables);
    std::vector<var_t> variable_types_host(num_variables);
    for (int j = 0; j < num_variables; j++) {
      variable_types_host[j] =
        variable_types[j] == CUOPT_CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;
    }
    problem_and_stream->op_problem->set_variable_types(variable_types_host.data(), num_variables);
    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const raft::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptCreateRangedProblem(cuopt_int_t num_constraints,
                                     cuopt_int_t num_variables,
                                     cuopt_int_t objective_sense,
                                     cuopt_float_t objective_offset,
                                     const cuopt_float_t* objective_coefficients,
                                     const cuopt_int_t* constraint_matrix_row_offsets,
                                     const cuopt_int_t* constraint_matrix_column_indices,
                                     const cuopt_float_t* constraint_matrix_coefficent_values,
                                     const cuopt_float_t* constraint_lower_bounds,
                                     const cuopt_float_t* constraint_upper_bounds,
                                     const cuopt_float_t* variable_lower_bounds,
                                     const cuopt_float_t* variable_upper_bounds,
                                     const char* variable_types,
                                     cuOptOptimizationProblem* problem_ptr)
{
  if (problem_ptr == nullptr || objective_coefficients == nullptr ||
      constraint_matrix_row_offsets == nullptr || constraint_matrix_column_indices == nullptr ||
      constraint_matrix_coefficent_values == nullptr || constraint_lower_bounds == nullptr ||
      constraint_upper_bounds == nullptr || variable_lower_bounds == nullptr ||
      variable_upper_bounds == nullptr || variable_types == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_and_stream_view_t* problem_and_stream = new problem_and_stream_view_t();
  problem_and_stream->op_problem =
    new optimization_problem_t<cuopt_int_t, cuopt_float_t>(problem_and_stream->get_handle_ptr());
  try {
    problem_and_stream->op_problem->set_maximize(objective_sense == CUOPT_MAXIMIZE);
    problem_and_stream->op_problem->set_objective_offset(objective_offset);
    problem_and_stream->op_problem->set_objective_coefficients(objective_coefficients,
                                                               num_variables);
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];
    problem_and_stream->op_problem->set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                                              nnz,
                                                              constraint_matrix_column_indices,
                                                              nnz,
                                                              constraint_matrix_row_offsets,
                                                              num_constraints + 1);
    problem_and_stream->op_problem->set_constraint_lower_bounds(constraint_lower_bounds,
                                                                num_constraints);
    problem_and_stream->op_problem->set_constraint_upper_bounds(constraint_upper_bounds,
                                                                num_constraints);
    problem_and_stream->op_problem->set_variable_lower_bounds(variable_lower_bounds, num_variables);
    problem_and_stream->op_problem->set_variable_upper_bounds(variable_upper_bounds, num_variables);
    std::vector<var_t> variable_types_host(num_variables);
    for (int j = 0; j < num_variables; j++) {
      variable_types_host[j] =
        variable_types[j] == CUOPT_CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;
    }
    problem_and_stream->op_problem->set_variable_types(variable_types_host.data(), num_variables);
    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const raft::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

void cuOptDestroyProblem(cuOptOptimizationProblem* problem_ptr)
{
  if (problem_ptr == nullptr) { return; }
  if (*problem_ptr == nullptr) { return; }
  delete static_cast<problem_and_stream_view_t*>(*problem_ptr);
  *problem_ptr = nullptr;
}

cuopt_int_t cuOptGetNumConstraints(cuOptOptimizationProblem problem,
                                   cuopt_int_t* num_constraints_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_constraints_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *num_constraints_ptr = problem_and_stream_view->op_problem->get_n_constraints();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetNumVariables(cuOptOptimizationProblem problem, cuopt_int_t* num_variables_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_variables_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *num_variables_ptr = problem_and_stream_view->op_problem->get_n_variables();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveSense(cuOptOptimizationProblem problem,
                                   cuopt_int_t* objective_sense_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_sense_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *objective_sense_ptr =
    problem_and_stream_view->op_problem->get_sense() ? CUOPT_MAXIMIZE : CUOPT_MINIMIZE;
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveOffset(cuOptOptimizationProblem problem,
                                    cuopt_float_t* objective_offset_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_offset_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *objective_offset_ptr = problem_and_stream_view->op_problem->get_objective_offset();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveCoefficients(cuOptOptimizationProblem problem,
                                          cuopt_float_t* objective_coefficients_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_coefficients_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<cuopt_float_t>& objective_coefficients =
    problem_and_stream_view->op_problem->get_objective_coefficients();
  raft::copy(objective_coefficients_ptr,
             objective_coefficients.data(),
             objective_coefficients.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetNumNonZeros(cuOptOptimizationProblem problem,
                                cuopt_int_t* num_non_zero_elements_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_non_zero_elements_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *num_non_zero_elements_ptr = problem_and_stream_view->op_problem->get_nnz();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintMatrix(cuOptOptimizationProblem problem,
                                     cuopt_int_t* constraint_matrix_row_offsets_ptr,
                                     cuopt_int_t* constraint_matrix_column_indices_ptr,
                                     cuopt_float_t* constraint_matrix_coefficients_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (constraint_matrix_row_offsets_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (constraint_matrix_column_indices_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (constraint_matrix_coefficients_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<cuopt_float_t>& constraint_matrix_coefficients =
    problem_and_stream_view->op_problem->get_constraint_matrix_values();
  const rmm::device_uvector<cuopt_int_t>& constraint_matrix_column_indices =
    problem_and_stream_view->op_problem->get_constraint_matrix_indices();
  const rmm::device_uvector<cuopt_int_t>& constraint_matrix_row_offsets =
    problem_and_stream_view->op_problem->get_constraint_matrix_offsets();
  raft::copy(constraint_matrix_coefficients_ptr,
             constraint_matrix_coefficients.data(),
             constraint_matrix_coefficients.size(),
             problem_and_stream_view->stream_view);
  raft::copy(constraint_matrix_column_indices_ptr,
             constraint_matrix_column_indices.data(),
             constraint_matrix_column_indices.size(),
             problem_and_stream_view->stream_view);
  raft::copy(constraint_matrix_row_offsets_ptr,
             constraint_matrix_row_offsets.data(),
             constraint_matrix_row_offsets.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintSense(cuOptOptimizationProblem problem, char* constraint_sense_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (constraint_sense_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<char>& constraint_sense =
    problem_and_stream_view->op_problem->get_row_types();
  raft::copy(constraint_sense_ptr,
             constraint_sense.data(),
             constraint_sense.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintRightHandSide(cuOptOptimizationProblem problem,
                                            cuopt_float_t* rhs_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (rhs_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<cuopt_float_t>& rhs =
    problem_and_stream_view->op_problem->get_constraint_bounds();
  raft::copy(rhs_ptr, rhs.data(), rhs.size(), problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintLowerBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* lower_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (lower_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<cuopt_float_t>& lower_bounds =
    problem_and_stream_view->op_problem->get_constraint_lower_bounds();
  raft::copy(lower_bounds_ptr,
             lower_bounds.data(),
             lower_bounds.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintUpperBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* upper_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (upper_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<cuopt_float_t>& upper_bounds =
    problem_and_stream_view->op_problem->get_constraint_upper_bounds();
  raft::copy(upper_bounds_ptr,
             upper_bounds.data(),
             upper_bounds.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableLowerBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* lower_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (lower_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<cuopt_float_t>& lower_bounds =
    problem_and_stream_view->op_problem->get_variable_lower_bounds();
  raft::copy(lower_bounds_ptr,
             lower_bounds.data(),
             lower_bounds.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableUpperBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* upper_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (upper_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<cuopt_float_t>& upper_bounds =
    problem_and_stream_view->op_problem->get_variable_upper_bounds();
  raft::copy(upper_bounds_ptr,
             upper_bounds.data(),
             upper_bounds.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableTypes(cuOptOptimizationProblem problem, char* variable_types_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (variable_types_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  const rmm::device_uvector<var_t>& variable_types =
    problem_and_stream_view->op_problem->get_variable_types();
  std::vector<cuopt::linear_programming::var_t> variable_types_host(variable_types.size());
  raft::copy(variable_types_host.data(),
             variable_types.data(),
             variable_types.size(),
             problem_and_stream_view->stream_view);
  problem_and_stream_view->stream_view.synchronize();
  for (size_t j = 0; j < variable_types_host.size(); j++) {
    variable_types_ptr[j] =
      variable_types_host[j] == var_t::INTEGER ? CUOPT_INTEGER : CUOPT_CONTINUOUS;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptCreateSolverSettings(cuOptSolverSettings* settings_ptr)
{
  if (settings_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* settings =
    new solver_settings_t<cuopt_int_t, cuopt_float_t>();
  *settings_ptr = static_cast<cuOptSolverSettings>(settings);
  return CUOPT_SUCCESS;
}

void cuOptDestroySolverSettings(cuOptSolverSettings* settings_ptr)
{
  if (settings_ptr == nullptr) { return; }
  delete static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(*settings_ptr);
  *settings_ptr = nullptr;
}

cuopt_int_t cuOptSetParameter(cuOptSolverSettings settings,
                              const char* parameter_name,
                              const char* parameter_value)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_name == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_value == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
  try {
    solver_settings->set_parameter_from_string(parameter_name, parameter_value);
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetParameter(cuOptSolverSettings settings,
                              const char* parameter_name,
                              cuopt_int_t parameter_value_size,
                              char* parameter_value)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_name == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_value == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_value_size <= 0) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
  try {
    std::string parameter_value_str = solver_settings->get_parameter_as_string(parameter_name);
    std::snprintf(parameter_value, parameter_value_size, "%s", parameter_value_str.c_str());
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetIntegerParameter(cuOptSolverSettings settings,
                                     const char* parameter_name,
                                     cuopt_int_t parameter_value)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_name == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
  try {
    solver_settings->set_parameter<cuopt_int_t>(parameter_name, parameter_value);
  } catch (const std::invalid_argument& e) {
    // We could be trying to set a boolean parameter. Try that
    try {
      bool value = static_cast<bool>(parameter_value);
      solver_settings->set_parameter<bool>(parameter_name, value);
    } catch (const std::exception& e) {
      return CUOPT_INVALID_ARGUMENT;
    }
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetIntegerParameter(cuOptSolverSettings settings,
                                     const char* parameter_name,
                                     cuopt_int_t* parameter_value_ptr)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_name == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
  try {
    *parameter_value_ptr = solver_settings->get_parameter<cuopt_int_t>(parameter_name);
  } catch (const std::invalid_argument& e) {
    // We could be trying to get a boolean parameter. Try that
    try {
      *parameter_value_ptr =
        static_cast<cuopt_int_t>(solver_settings->get_parameter<bool>(parameter_name));
    } catch (const std::exception& e) {
      return CUOPT_INVALID_ARGUMENT;
    }
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetFloatParameter(cuOptSolverSettings settings,
                                   const char* parameter_name,
                                   cuopt_float_t parameter_value)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_name == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
  try {
    solver_settings->set_parameter<cuopt_float_t>(parameter_name, parameter_value);
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetFloatParameter(cuOptSolverSettings settings,
                                   const char* parameter_name,
                                   cuopt_float_t* parameter_value_ptr)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_name == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (parameter_value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
  try {
    *parameter_value_ptr = solver_settings->get_parameter<cuopt_float_t>(parameter_name);
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptIsMIP(cuOptOptimizationProblem problem, cuopt_int_t* is_mip_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (is_mip_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  bool is_mip =
    (problem_and_stream_view->op_problem->get_problem_category() == problem_category_t::MIP) ||
    (problem_and_stream_view->op_problem->get_problem_category() == problem_category_t::IP);
  *is_mip_ptr = static_cast<cuopt_int_t>(is_mip);
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSolve(cuOptOptimizationProblem problem,
                       cuOptSolverSettings settings,
                       cuOptSolution* solution_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solution_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  if (problem_and_stream_view->op_problem->get_problem_category() == problem_category_t::MIP ||
      problem_and_stream_view->op_problem->get_problem_category() == problem_category_t::IP) {
    solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
      static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
    mip_solver_settings_t<cuopt_int_t, cuopt_float_t>& mip_settings =
      solver_settings->get_mip_settings();
    optimization_problem_t<cuopt_int_t, cuopt_float_t>* op_problem =
      problem_and_stream_view->op_problem;
    solution_and_stream_view_t* solution_and_stream_view =
      new solution_and_stream_view_t(true, problem_and_stream_view->stream_view);
    solution_and_stream_view->mip_solution_ptr = new mip_solution_t<cuopt_int_t, cuopt_float_t>(
      solve_mip<cuopt_int_t, cuopt_float_t>(*op_problem, mip_settings));
    *solution_ptr = static_cast<cuOptSolution>(solution_and_stream_view);
    return static_cast<cuopt_int_t>(
      solution_and_stream_view->mip_solution_ptr->get_error_status().get_error_type());
  } else {
    solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
      static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);
    pdlp_solver_settings_t<cuopt_int_t, cuopt_float_t>& pdlp_settings =
      solver_settings->get_pdlp_settings();
    optimization_problem_t<cuopt_int_t, cuopt_float_t>* op_problem =
      problem_and_stream_view->op_problem;
    solution_and_stream_view_t* solution_and_stream_view =
      new solution_and_stream_view_t(false, problem_and_stream_view->stream_view);
    solution_and_stream_view->lp_solution_ptr =
      new optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>(
        solve_lp<cuopt_int_t, cuopt_float_t>(*op_problem, pdlp_settings));
    *solution_ptr = static_cast<cuOptSolution>(solution_and_stream_view);
    return static_cast<cuopt_int_t>(
      solution_and_stream_view->lp_solution_ptr->get_error_status().get_error_type());
  }
}

void cuOptDestroySolution(cuOptSolution* solution_ptr)
{
  if (solution_ptr == nullptr) { return; }
  if (*solution_ptr == nullptr) { return; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(*solution_ptr);
  if (solution_and_stream_view->is_mip) {
    mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution =
      static_cast<mip_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->mip_solution_ptr);
    delete mip_solution;
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    delete optimization_problem_solution;
  }
  delete solution_and_stream_view;
  *solution_ptr = nullptr;
}

cuopt_int_t cuOptGetTerminationStatus(cuOptSolution solution, cuopt_int_t* termination_status_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (termination_status_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution =
      static_cast<mip_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->mip_solution_ptr);
    *termination_status_ptr = static_cast<cuopt_int_t>(mip_solution->get_termination_status());
  } else {
    pdlp_termination_status_t termination_status =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr)
        ->get_termination_status();
    *termination_status_ptr = static_cast<cuopt_int_t>(termination_status);
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetErrorStatus(cuOptSolution solution, cuopt_int_t* error_status_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (error_status_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    *error_status_ptr = static_cast<cuopt_int_t>(
      solution_and_stream_view->mip_solution_ptr->get_error_status().get_error_type());
  } else {
    *error_status_ptr = static_cast<cuopt_int_t>(
      solution_and_stream_view->lp_solution_ptr->get_error_status().get_error_type());
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetErrorString(cuOptSolution solution,
                                char* error_string_ptr,
                                cuopt_int_t error_string_size)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (error_string_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    std::string error_string =
      solution_and_stream_view->mip_solution_ptr->get_error_status().what();
    std::snprintf(error_string_ptr, error_string_size, "%s", error_string.c_str());
  } else {
    std::string error_string = solution_and_stream_view->lp_solution_ptr->get_error_status().what();
    std::snprintf(error_string_ptr, error_string_size, "%s", error_string.c_str());
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetPrimalSolution(cuOptSolution solution, cuopt_float_t* solution_values_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solution_values_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution =
      static_cast<mip_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->mip_solution_ptr);
    const rmm::device_uvector<cuopt_float_t>& solution_values = mip_solution->get_solution();
    rmm::cuda_stream_view stream_view{};
    raft::copy(solution_values_ptr,
               solution_values.data(),
               solution_values.size(),
               solution_and_stream_view->stream_view);
    solution_and_stream_view->stream_view.synchronize();
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    const rmm::device_uvector<cuopt_float_t>& solution_values =
      optimization_problem_solution->get_primal_solution();
    raft::copy(solution_values_ptr,
               solution_values.data(),
               solution_values.size(),
               solution_and_stream_view->stream_view);
    solution_and_stream_view->stream_view.synchronize();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveValue(cuOptSolution solution, cuopt_float_t* objective_value_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution =
      static_cast<mip_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->mip_solution_ptr);
    *objective_value_ptr = mip_solution->get_objective_value();
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    *objective_value_ptr = optimization_problem_solution->get_objective_value();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetSolveTime(cuOptSolution solution, cuopt_float_t* solve_time_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solve_time_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution =
      static_cast<mip_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->mip_solution_ptr);
    *solve_time_ptr = mip_solution->get_total_solve_time();
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    *solve_time_ptr = (optimization_problem_solution->get_solve_time());
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetMIPGap(cuOptSolution solution, cuopt_float_t* mip_gap_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (mip_gap_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution =
      static_cast<mip_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->mip_solution_ptr);
    *mip_gap_ptr = mip_solution->get_mip_gap();
  } else {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetSolutionBound(cuOptSolution solution, cuopt_float_t* solution_bound_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solution_bound_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution =
      static_cast<mip_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->mip_solution_ptr);
    *solution_bound_ptr = mip_solution->get_solution_bound();
  } else {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetDualSolution(cuOptSolution solution, cuopt_float_t* dual_solution_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (dual_solution_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    return CUOPT_INVALID_ARGUMENT;
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    const rmm::device_uvector<cuopt_float_t>& dual_solution =
      optimization_problem_solution->get_dual_solution();
    raft::copy(dual_solution_ptr,
               dual_solution.data(),
               dual_solution.size(),
               solution_and_stream_view->stream_view);
    solution_and_stream_view->stream_view.synchronize();
    return CUOPT_SUCCESS;
  }
}

cuopt_int_t cuOptGetDualObjectiveValue(cuOptSolution solution,
                                       cuopt_float_t* dual_objective_value_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (dual_objective_value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    return CUOPT_INVALID_ARGUMENT;
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    *dual_objective_value_ptr = optimization_problem_solution->get_dual_objective_value();
    return CUOPT_SUCCESS;
  }
}

cuopt_int_t cuOptGetReducedCosts(cuOptSolution solution, cuopt_float_t* reduced_cost_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (reduced_cost_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  if (solution_and_stream_view->is_mip) {
    return CUOPT_INVALID_ARGUMENT;
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    const rmm::device_uvector<cuopt_float_t>& reduced_cost =
      optimization_problem_solution->get_reduced_cost();
    raft::copy(reduced_cost_ptr,
               reduced_cost.data(),
               reduced_cost.size(),
               solution_and_stream_view->stream_view);
    solution_and_stream_view->stream_view.synchronize();
    return CUOPT_SUCCESS;
  }
}
