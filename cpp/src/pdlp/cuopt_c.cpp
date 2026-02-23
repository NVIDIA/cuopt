/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cuopt_c.h>

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/utilities/timestamp_utils.hpp>
#include <pdlp/cuopt_c_internal.hpp>
#include <utilities/logger.hpp>

#include <mps_parser/parser.hpp>

#include <cuopt/version_config.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace cuopt::mps_parser;
using namespace cuopt::linear_programming;

class c_get_solution_callback_t : public cuopt::internals::get_solution_callback_t {
 public:
  explicit c_get_solution_callback_t(cuOptMIPGetSolutionCallback callback) : callback_(callback) {}

  void get_solution(void* data,
                    void* objective_value,
                    void* solution_bound,
                    void* user_data) override
  {
    if (callback_ == nullptr) { return; }
    callback_(static_cast<const cuopt_float_t*>(data),
              static_cast<const cuopt_float_t*>(objective_value),
              static_cast<const cuopt_float_t*>(solution_bound),
              user_data);
  }

 private:
  cuOptMIPGetSolutionCallback callback_;
};

class c_set_solution_callback_t : public cuopt::internals::set_solution_callback_t {
 public:
  explicit c_set_solution_callback_t(cuOptMIPSetSolutionCallback callback) : callback_(callback) {}

  void set_solution(void* data,
                    void* objective_value,
                    void* solution_bound,
                    void* user_data) override
  {
    if (callback_ == nullptr) { return; }
    callback_(static_cast<cuopt_float_t*>(data),
              static_cast<cuopt_float_t*>(objective_value),
              static_cast<const cuopt_float_t*>(solution_bound),
              user_data);
  }

 private:
  cuOptMIPSetSolutionCallback callback_;
};

// Owns solver settings and C callback wrappers for C API lifetime.
struct solver_settings_handle_t {
  solver_settings_handle_t() : settings(new solver_settings_t<cuopt_int_t, cuopt_float_t>()) {}
  ~solver_settings_handle_t() { delete settings; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* settings;
  std::vector<std::unique_ptr<cuopt::internals::base_solution_callback_t>> callbacks;
};

solver_settings_handle_t* get_settings_handle(cuOptSolverSettings settings)
{
  return static_cast<solver_settings_handle_t*>(settings);
}

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
  problem_and_stream_view_t* problem_and_stream =
    new problem_and_stream_view_t(get_memory_backend_type());
  std::string filename_str(filename);
  bool input_mps_strict = false;
  std::unique_ptr<mps_data_model_t<cuopt_int_t, cuopt_float_t>> mps_data_model_ptr;
  try {
    mps_data_model_ptr = std::make_unique<mps_data_model_t<cuopt_int_t, cuopt_float_t>>(
      parse_mps<cuopt_int_t, cuopt_float_t>(filename_str, input_mps_strict));
  } catch (const std::exception& e) {
    CUOPT_LOG_INFO("Error parsing MPS file: %s", e.what());
    delete problem_and_stream;
    *problem_ptr = nullptr;
    if (std::string(e.what()).find("Error opening MPS file") != std::string::npos) {
      return CUOPT_MPS_FILE_ERROR;
    } else {
      return CUOPT_MPS_PARSE_ERROR;
    }
  }

  // Populate interface directly from MPS data model (avoids temporary GPU allocation)
  populate_from_mps_data_model(problem_and_stream->get_problem(), *mps_data_model_ptr);

  *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptWriteProblem(cuOptOptimizationProblem problem,
                              const char* filename,
                              cuopt_int_t format)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (filename == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (strlen(filename) == 0) { return CUOPT_INVALID_ARGUMENT; }
  if (format != CUOPT_FILE_FORMAT_MPS) { return CUOPT_INVALID_ARGUMENT; }

  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  try {
    // Use the write_to_mps method from the interface (works for both CPU and GPU)
    problem_and_stream_view->get_problem()->write_to_mps(std::string(filename));
  } catch (const std::exception& e) {
    CUOPT_LOG_INFO("Error writing MPS file: %s", e.what());
    return CUOPT_MPS_FILE_ERROR;
  }
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
  cuopt::utilities::printTimestamp("CUOPT_CREATE_PROBLEM");

  if (problem_ptr == nullptr || objective_coefficients == nullptr ||
      constraint_matrix_row_offsets == nullptr || constraint_matrix_column_indices == nullptr ||
      constraint_matrix_coefficent_values == nullptr || constraint_sense == nullptr ||
      rhs == nullptr || lower_bounds == nullptr || upper_bounds == nullptr ||
      variable_types == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_and_stream_view_t* problem_and_stream =
    new problem_and_stream_view_t(get_memory_backend_type());
  try {
    auto* problem = problem_and_stream->get_problem();
    problem->set_maximize(objective_sense == CUOPT_MAXIMIZE);
    problem->set_objective_offset(objective_offset);
    problem->set_objective_coefficients(objective_coefficients, num_variables);
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];
    problem->set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                       nnz,
                                       constraint_matrix_column_indices,
                                       nnz,
                                       constraint_matrix_row_offsets,
                                       num_constraints + 1);
    problem->set_row_types(constraint_sense, num_constraints);
    problem->set_constraint_bounds(rhs, num_constraints);
    problem->set_variable_lower_bounds(lower_bounds, num_variables);
    problem->set_variable_upper_bounds(upper_bounds, num_variables);

    // Set variable types (problem category is auto-detected)
    std::vector<var_t> variable_types_host(num_variables);
    for (int j = 0; j < num_variables; j++) {
      variable_types_host[j] =
        variable_types[j] == CUOPT_CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;
    }
    problem->set_variable_types(variable_types_host.data(), num_variables);

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const raft::exception& e) {
    delete problem_and_stream;
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
  cuopt::utilities::printTimestamp("CUOPT_CREATE_PROBLEM");

  if (problem_ptr == nullptr || objective_coefficients == nullptr ||
      constraint_matrix_row_offsets == nullptr || constraint_matrix_column_indices == nullptr ||
      constraint_matrix_coefficent_values == nullptr || constraint_lower_bounds == nullptr ||
      constraint_upper_bounds == nullptr || variable_lower_bounds == nullptr ||
      variable_upper_bounds == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_and_stream_view_t* problem_and_stream =
    new problem_and_stream_view_t(get_memory_backend_type());
  try {
    auto* problem = problem_and_stream->get_problem();
    problem->set_maximize(objective_sense == CUOPT_MAXIMIZE);
    problem->set_objective_offset(objective_offset);
    problem->set_objective_coefficients(objective_coefficients, num_variables);
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];
    problem->set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                       nnz,
                                       constraint_matrix_column_indices,
                                       nnz,
                                       constraint_matrix_row_offsets,
                                       num_constraints + 1);
    problem->set_constraint_lower_bounds(constraint_lower_bounds, num_constraints);
    problem->set_constraint_upper_bounds(constraint_upper_bounds, num_constraints);
    problem->set_variable_lower_bounds(variable_lower_bounds, num_variables);
    problem->set_variable_upper_bounds(variable_upper_bounds, num_variables);

    // Set variable types (NULL means all continuous)
    // Problem category (LP/MIP/IP) is auto-detected by set_variable_types
    std::vector<var_t> variable_types_host(num_variables);
    if (variable_types != nullptr) {
      for (int j = 0; j < num_variables; j++) {
        variable_types_host[j] =
          variable_types[j] == CUOPT_CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;
      }
    } else {
      // Default to all continuous
      for (int j = 0; j < num_variables; j++) {
        variable_types_host[j] = var_t::CONTINUOUS;
      }
    }
    problem->set_variable_types(variable_types_host.data(), num_variables);

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const raft::exception& e) {
    delete problem_and_stream;
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptCreateQuadraticProblem(
  cuopt_int_t num_constraints,
  cuopt_int_t num_variables,
  cuopt_int_t objective_sense,
  cuopt_float_t objective_offset,
  const cuopt_float_t* objective_coefficients,
  const cuopt_int_t* quadratic_objective_matrix_row_offsets,
  const cuopt_int_t* quadratic_objective_matrix_column_indices,
  const cuopt_float_t* quadratic_objective_matrix_coefficent_values,
  const cuopt_int_t* constraint_matrix_row_offsets,
  const cuopt_int_t* constraint_matrix_column_indices,
  const cuopt_float_t* constraint_matrix_coefficent_values,
  const char* constraint_sense,
  const cuopt_float_t* rhs,
  const cuopt_float_t* lower_bounds,
  const cuopt_float_t* upper_bounds,
  cuOptOptimizationProblem* problem_ptr)
{
  cuopt::utilities::printTimestamp("CUOPT_CREATE_PROBLEM");

  if (problem_ptr == nullptr || objective_coefficients == nullptr ||
      quadratic_objective_matrix_row_offsets == nullptr ||
      quadratic_objective_matrix_column_indices == nullptr ||
      quadratic_objective_matrix_coefficent_values == nullptr ||
      constraint_matrix_row_offsets == nullptr || constraint_matrix_column_indices == nullptr ||
      constraint_matrix_coefficent_values == nullptr || constraint_sense == nullptr ||
      rhs == nullptr || lower_bounds == nullptr || upper_bounds == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_and_stream_view_t* problem_and_stream =
    new problem_and_stream_view_t(get_memory_backend_type());
  try {
    auto* problem = problem_and_stream->get_problem();
    problem->set_maximize(objective_sense == CUOPT_MAXIMIZE);
    problem->set_objective_offset(objective_offset);
    problem->set_objective_coefficients(objective_coefficients, num_variables);
    cuopt_int_t Q_nnz = quadratic_objective_matrix_row_offsets[num_variables];
    problem->set_quadratic_objective_matrix(quadratic_objective_matrix_coefficent_values,
                                            Q_nnz,
                                            quadratic_objective_matrix_column_indices,
                                            Q_nnz,
                                            quadratic_objective_matrix_row_offsets,
                                            num_variables + 1);
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];
    problem->set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                       nnz,
                                       constraint_matrix_column_indices,
                                       nnz,
                                       constraint_matrix_row_offsets,
                                       num_constraints + 1);
    problem->set_row_types(constraint_sense, num_constraints);
    problem->set_constraint_bounds(rhs, num_constraints);
    problem->set_variable_lower_bounds(lower_bounds, num_variables);
    problem->set_variable_upper_bounds(upper_bounds, num_variables);

    // Quadratic problems default to LP category (no variable types set, so no MIP detection)

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const raft::exception& e) {
    delete problem_and_stream;
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptCreateQuadraticRangedProblem(
  cuopt_int_t num_constraints,
  cuopt_int_t num_variables,
  cuopt_int_t objective_sense,
  cuopt_float_t objective_offset,
  const cuopt_float_t* objective_coefficients,
  const cuopt_int_t* quadratic_objective_matrix_row_offsets,
  const cuopt_int_t* quadratic_objective_matrix_column_indices,
  const cuopt_float_t* quadratic_objective_matrix_coefficent_values,
  const cuopt_int_t* constraint_matrix_row_offsets,
  const cuopt_int_t* constraint_matrix_column_indices,
  const cuopt_float_t* constraint_matrix_coefficent_values,
  const cuopt_float_t* constraint_lower_bounds,
  const cuopt_float_t* constraint_upper_bounds,
  const cuopt_float_t* variable_lower_bounds,
  const cuopt_float_t* variable_upper_bounds,
  cuOptOptimizationProblem* problem_ptr)
{
  cuopt::utilities::printTimestamp("CUOPT_CREATE_QUADRATIC_RANGED_PROBLEM");

  if (problem_ptr == nullptr || objective_coefficients == nullptr ||
      quadratic_objective_matrix_row_offsets == nullptr ||
      quadratic_objective_matrix_column_indices == nullptr ||
      quadratic_objective_matrix_coefficent_values == nullptr ||
      constraint_matrix_row_offsets == nullptr || constraint_matrix_column_indices == nullptr ||
      constraint_matrix_coefficent_values == nullptr || constraint_lower_bounds == nullptr ||
      constraint_upper_bounds == nullptr || variable_lower_bounds == nullptr ||
      variable_upper_bounds == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_and_stream_view_t* problem_and_stream =
    new problem_and_stream_view_t(get_memory_backend_type());
  try {
    auto* problem = problem_and_stream->get_problem();
    problem->set_maximize(objective_sense == CUOPT_MAXIMIZE);
    problem->set_objective_offset(objective_offset);
    problem->set_objective_coefficients(objective_coefficients, num_variables);
    cuopt_int_t Q_nnz = quadratic_objective_matrix_row_offsets[num_variables];
    problem->set_quadratic_objective_matrix(quadratic_objective_matrix_coefficent_values,
                                            Q_nnz,
                                            quadratic_objective_matrix_column_indices,
                                            Q_nnz,
                                            quadratic_objective_matrix_row_offsets,
                                            num_variables + 1);
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];
    problem->set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                       nnz,
                                       constraint_matrix_column_indices,
                                       nnz,
                                       constraint_matrix_row_offsets,
                                       num_constraints + 1);
    problem->set_constraint_lower_bounds(constraint_lower_bounds, num_constraints);
    problem->set_constraint_upper_bounds(constraint_upper_bounds, num_constraints);
    problem->set_variable_lower_bounds(variable_lower_bounds, num_variables);
    problem->set_variable_upper_bounds(variable_upper_bounds, num_variables);

    // Quadratic problems default to LP category (no variable types set, so no MIP detection)

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const raft::exception& e) {
    delete problem_and_stream;
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
  *num_constraints_ptr = problem_and_stream_view->get_problem()->get_n_constraints();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetNumVariables(cuOptOptimizationProblem problem, cuopt_int_t* num_variables_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_variables_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *num_variables_ptr = problem_and_stream_view->get_problem()->get_n_variables();
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
    problem_and_stream_view->get_problem()->get_sense() ? CUOPT_MAXIMIZE : CUOPT_MINIMIZE;
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveOffset(cuOptOptimizationProblem problem,
                                    cuopt_float_t* objective_offset_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_offset_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *objective_offset_ptr = problem_and_stream_view->get_problem()->get_objective_offset();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveCoefficients(cuOptOptimizationProblem problem,
                                          cuopt_float_t* objective_coefficients_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_coefficients_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_variables();
  problem_and_stream_view->get_problem()->copy_objective_coefficients_to_host(
    objective_coefficients_ptr, size);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetNumNonZeros(cuOptOptimizationProblem problem,
                                cuopt_int_t* num_non_zero_elements_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_non_zero_elements_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  *num_non_zero_elements_ptr = problem_and_stream_view->get_problem()->get_nnz();
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

  auto* prob           = problem_and_stream_view->get_problem();
  cuopt_int_t num_nnz  = prob->get_nnz();
  cuopt_int_t num_rows = prob->get_n_constraints();

  prob->copy_constraint_matrix_to_host(constraint_matrix_coefficients_ptr,
                                       constraint_matrix_column_indices_ptr,
                                       constraint_matrix_row_offsets_ptr,
                                       num_nnz,
                                       num_nnz,
                                       num_rows + 1);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintSense(cuOptOptimizationProblem problem, char* constraint_sense_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (constraint_sense_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_constraints();
  problem_and_stream_view->get_problem()->copy_row_types_to_host(constraint_sense_ptr, size);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintRightHandSide(cuOptOptimizationProblem problem,
                                            cuopt_float_t* rhs_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (rhs_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_constraints();
  problem_and_stream_view->get_problem()->copy_constraint_bounds_to_host(rhs_ptr, size);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintLowerBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* lower_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (lower_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_constraints();
  problem_and_stream_view->get_problem()->copy_constraint_lower_bounds_to_host(lower_bounds_ptr,
                                                                               size);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintUpperBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* upper_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (upper_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_constraints();
  problem_and_stream_view->get_problem()->copy_constraint_upper_bounds_to_host(upper_bounds_ptr,
                                                                               size);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableLowerBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* lower_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (lower_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_variables();
  problem_and_stream_view->get_problem()->copy_variable_lower_bounds_to_host(lower_bounds_ptr,
                                                                             size);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableUpperBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* upper_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (upper_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_variables();
  problem_and_stream_view->get_problem()->copy_variable_upper_bounds_to_host(upper_bounds_ptr,
                                                                             size);

  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableTypes(cuOptOptimizationProblem problem, char* variable_types_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (variable_types_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  cuopt_int_t size = problem_and_stream_view->get_problem()->get_n_variables();
  std::vector<cuopt::linear_programming::var_t> variable_types_host(size);
  problem_and_stream_view->get_problem()->copy_variable_types_to_host(variable_types_host.data(),
                                                                      size);

  // Convert var_t enum to C API char values
  for (size_t j = 0; j < variable_types_host.size(); j++) {
    variable_types_ptr[j] =
      variable_types_host[j] == var_t::INTEGER ? CUOPT_INTEGER : CUOPT_CONTINUOUS;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptCreateSolverSettings(cuOptSolverSettings* settings_ptr)
{
  if (settings_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_handle_t* settings_handle = new solver_settings_handle_t();
  *settings_ptr                             = static_cast<cuOptSolverSettings>(settings_handle);
  return CUOPT_SUCCESS;
}

void cuOptDestroySolverSettings(cuOptSolverSettings* settings_ptr)
{
  if (settings_ptr == nullptr) { return; }
  delete get_settings_handle(*settings_ptr);
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
    get_settings_handle(settings)->settings;
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
    get_settings_handle(settings)->settings;
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
    get_settings_handle(settings)->settings;
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
    get_settings_handle(settings)->settings;
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
    get_settings_handle(settings)->settings;
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
    get_settings_handle(settings)->settings;
  try {
    *parameter_value_ptr = solver_settings->get_parameter<cuopt_float_t>(parameter_name);
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetMIPGetSolutionCallback(cuOptSolverSettings settings,
                                           cuOptMIPGetSolutionCallback callback,
                                           void* user_data)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (callback == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_handle_t* settings_handle = get_settings_handle(settings);
  auto callback_wrapper                     = std::make_unique<c_get_solution_callback_t>(callback);
  settings_handle->settings->set_mip_callback(callback_wrapper.get(), user_data);
  settings_handle->callbacks.push_back(std::move(callback_wrapper));
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetMIPSetSolutionCallback(cuOptSolverSettings settings,
                                           cuOptMIPSetSolutionCallback callback,
                                           void* user_data)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (callback == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solver_settings_handle_t* settings_handle = get_settings_handle(settings);
  auto callback_wrapper                     = std::make_unique<c_set_solution_callback_t>(callback);
  settings_handle->settings->set_mip_callback(callback_wrapper.get(), user_data);
  settings_handle->callbacks.push_back(std::move(callback_wrapper));
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetInitialPrimalSolution(cuOptSolverSettings settings,
                                          const cuopt_float_t* primal_solution,
                                          cuopt_int_t num_variables)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (primal_solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_variables <= 0) { return CUOPT_INVALID_ARGUMENT; }

  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    get_settings_handle(settings)->settings;
  try {
    solver_settings->set_initial_pdlp_primal_solution(primal_solution, num_variables);
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetInitialDualSolution(cuOptSolverSettings settings,
                                        const cuopt_float_t* dual_solution,
                                        cuopt_int_t num_constraints)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (dual_solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_constraints <= 0) { return CUOPT_INVALID_ARGUMENT; }

  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    get_settings_handle(settings)->settings;
  try {
    solver_settings->set_initial_pdlp_dual_solution(dual_solution, num_constraints);
  } catch (const std::exception& e) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptAddMIPStart(cuOptSolverSettings settings,
                             const cuopt_float_t* solution,
                             cuopt_int_t num_variables)
{
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_variables <= 0) { return CUOPT_INVALID_ARGUMENT; }

  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    get_settings_handle(settings)->settings;
  try {
    solver_settings->get_mip_settings().add_initial_solution(solution, num_variables);
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
  problem_category_t category = problem_and_stream_view->get_problem()->get_problem_category();
  bool is_mip = (category == problem_category_t::MIP) || (category == problem_category_t::IP);
  *is_mip_ptr = static_cast<cuopt_int_t>(is_mip);
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSolve(cuOptOptimizationProblem problem,
                       cuOptSolverSettings settings,
                       cuOptSolution* solution_ptr)
{
  cuopt::utilities::printTimestamp("CUOPT_SOLVE_START");

  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solution_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  // Get the problem interface (GPU or CPU backed)
  optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>* problem_interface =
    problem_and_stream_view->get_problem();

  try {
    if (problem_interface->get_problem_category() == problem_category_t::MIP ||
        problem_interface->get_problem_category() == problem_category_t::IP) {
      solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
        get_settings_handle(settings)->settings;
      mip_solver_settings_t<cuopt_int_t, cuopt_float_t>& mip_settings =
        solver_settings->get_mip_settings();

      // Solve returns unique_ptr<mip_solution_interface_t>
      auto solution_interface =
        solve_mip<cuopt_int_t, cuopt_float_t>(problem_interface, mip_settings);

      auto solution_holder =
        std::make_unique<solution_and_stream_view_t>(true, problem_and_stream_view->memory_backend);
      solution_holder->mip_solution_interface_ptr = solution_interface.release();

      cuopt::utilities::printTimestamp("CUOPT_SOLVE_RETURN");

      auto err = static_cast<cuopt_int_t>(
        solution_holder->mip_solution_interface_ptr->get_error_status().get_error_type());
      *solution_ptr = static_cast<cuOptSolution>(solution_holder.release());
      return err;
    } else {
      solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
        get_settings_handle(settings)->settings;
      pdlp_solver_settings_t<cuopt_int_t, cuopt_float_t>& pdlp_settings =
        solver_settings->get_pdlp_settings();

      // Solve returns unique_ptr<lp_solution_interface_t>
      auto solution_interface =
        solve_lp<cuopt_int_t, cuopt_float_t>(problem_interface, pdlp_settings);

      auto solution_holder = std::make_unique<solution_and_stream_view_t>(
        false, problem_and_stream_view->memory_backend);
      solution_holder->lp_solution_interface_ptr = solution_interface.release();

      cuopt::utilities::printTimestamp("CUOPT_SOLVE_RETURN");

      auto err = static_cast<cuopt_int_t>(
        solution_holder->lp_solution_interface_ptr->get_error_status().get_error_type());
      *solution_ptr = static_cast<cuOptSolution>(solution_holder.release());
      return err;
    }
  } catch (const cuopt::logic_error& e) {
    // Remote execution not yet implemented or other logic errors
    CUOPT_LOG_ERROR("Solve failed: %s", e.what());
    return static_cast<cuopt_int_t>(e.get_error_type());
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("Solve failed with exception: %s", e.what());
    return CUOPT_RUNTIME_ERROR;
  }
}

void cuOptDestroySolution(cuOptSolution* solution_ptr)
{
  if (solution_ptr == nullptr) { return; }
  if (*solution_ptr == nullptr) { return; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(*solution_ptr);
  // Destructor handles cleanup of interface pointers
  delete solution_and_stream_view;
  *solution_ptr = nullptr;
}

cuopt_int_t cuOptGetTerminationStatus(cuOptSolution solution, cuopt_int_t* termination_status_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (termination_status_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  *termination_status_ptr = static_cast<cuopt_int_t>(
    solution_and_stream_view->get_solution()->get_termination_status_int());
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetErrorStatus(cuOptSolution solution, cuopt_int_t* error_status_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (error_status_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  *error_status_ptr = static_cast<cuopt_int_t>(
    solution_and_stream_view->get_solution()->get_error_status().get_error_type());
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
  std::string error_string = solution_and_stream_view->get_solution()->get_error_status().what();
  std::snprintf(error_string_ptr, error_string_size, "%s", error_string.c_str());
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetPrimalSolution(cuOptSolution solution, cuopt_float_t* solution_values_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solution_values_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);

  const auto solution_host = solution_and_stream_view->get_solution()->get_solution_host();
  std::memcpy(
    solution_values_ptr, solution_host.data(), solution_host.size() * sizeof(cuopt_float_t));
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveValue(cuOptSolution solution, cuopt_float_t* objective_value_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  *objective_value_ptr = solution_and_stream_view->get_solution()->get_objective_value();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetSolveTime(cuOptSolution solution, cuopt_float_t* solve_time_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solve_time_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  *solve_time_ptr = solution_and_stream_view->get_solution()->get_solve_time();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetMIPGap(cuOptSolution solution, cuopt_float_t* mip_gap_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (mip_gap_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  try {
    *mip_gap_ptr = solution_and_stream_view->get_solution()->get_mip_gap();
    return CUOPT_SUCCESS;
  } catch (const std::logic_error&) {
    return CUOPT_INVALID_ARGUMENT;
  }
}

cuopt_int_t cuOptGetSolutionBound(cuOptSolution solution, cuopt_float_t* solution_bound_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (solution_bound_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  try {
    *solution_bound_ptr = solution_and_stream_view->get_solution()->get_solution_bound();
    return CUOPT_SUCCESS;
  } catch (const std::logic_error&) {
    return CUOPT_INVALID_ARGUMENT;
  }
}

cuopt_int_t cuOptGetDualSolution(cuOptSolution solution, cuopt_float_t* dual_solution_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (dual_solution_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  try {
    const auto dual_host = solution_and_stream_view->get_solution()->get_dual_solution();
    std::memcpy(dual_solution_ptr, dual_host.data(), dual_host.size() * sizeof(cuopt_float_t));
    return CUOPT_SUCCESS;
  } catch (const std::logic_error&) {
    return CUOPT_INVALID_ARGUMENT;
  }
}

cuopt_int_t cuOptGetDualObjectiveValue(cuOptSolution solution,
                                       cuopt_float_t* dual_objective_value_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (dual_objective_value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  try {
    *dual_objective_value_ptr =
      solution_and_stream_view->get_solution()->get_dual_objective_value();
    return CUOPT_SUCCESS;
  } catch (const std::logic_error&) {
    return CUOPT_INVALID_ARGUMENT;
  }
}

cuopt_int_t cuOptGetReducedCosts(cuOptSolution solution, cuopt_float_t* reduced_cost_ptr)
{
  if (solution == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (reduced_cost_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  solution_and_stream_view_t* solution_and_stream_view =
    static_cast<solution_and_stream_view_t*>(solution);
  try {
    const auto reduced_cost_host = solution_and_stream_view->get_solution()->get_reduced_costs();
    std::memcpy(
      reduced_cost_ptr, reduced_cost_host.data(), reduced_cost_host.size() * sizeof(cuopt_float_t));
    return CUOPT_SUCCESS;
  } catch (const std::logic_error&) {
    return CUOPT_INVALID_ARGUMENT;
  }
}
