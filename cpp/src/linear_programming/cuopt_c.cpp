/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cuopt_c.h>

#include <cuopt/linear_programming/data_model_view.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <cuopt/utilities/timestamp_utils.hpp>
#include <utilities/logger.hpp>

#include <mps_parser/parser.hpp>

#include <cuopt/version_config.hpp>

#include <raft/core/copy.hpp>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace cuopt::mps_parser;
using namespace cuopt::linear_programming;

/**
 * @brief CPU-side storage for problem data.
 *
 * This struct stores all problem data in CPU memory. At solve time, a data_model_view_t
 * is created pointing to this data, and the solve_lp/solve_mip routines handle
 * local vs remote solve automatically.
 */
struct problem_cpu_data_t {
  // Problem dimensions
  cuopt_int_t num_constraints = 0;
  cuopt_int_t num_variables   = 0;

  // Objective
  bool maximize                  = false;
  cuopt_float_t objective_offset = 0.0;
  std::vector<cuopt_float_t> objective_coefficients;

  // Quadratic objective (optional)
  std::vector<cuopt_float_t> Q_values;
  std::vector<cuopt_int_t> Q_indices;
  std::vector<cuopt_int_t> Q_offsets;

  // Constraint matrix (CSR format)
  std::vector<cuopt_float_t> A_values;
  std::vector<cuopt_int_t> A_indices;
  std::vector<cuopt_int_t> A_offsets;

  // Constraint bounds (two representations)
  std::vector<char> row_types;                         // '<', '>', '=' style
  std::vector<cuopt_float_t> constraint_bounds;        // single RHS for row_types style
  std::vector<cuopt_float_t> constraint_lower_bounds;  // ranged style
  std::vector<cuopt_float_t> constraint_upper_bounds;  // ranged style
  bool uses_ranged_constraints = false;

  // Variable bounds
  std::vector<cuopt_float_t> variable_lower_bounds;
  std::vector<cuopt_float_t> variable_upper_bounds;

  // Variable types
  std::vector<char> variable_types;  // 'C' for continuous, 'I' for integer

  /**
   * @brief Create a data_model_view_t pointing to this CPU data.
   */
  cuopt::linear_programming::data_model_view_t<cuopt_int_t, cuopt_float_t> create_view() const
  {
    cuopt::linear_programming::data_model_view_t<cuopt_int_t, cuopt_float_t> view;

    view.set_maximize(maximize);
    view.set_objective_offset(objective_offset);

    if (!objective_coefficients.empty()) {
      view.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());
    }

    if (!Q_values.empty()) {
      view.set_quadratic_objective_matrix(Q_values.data(),
                                          Q_values.size(),
                                          Q_indices.data(),
                                          Q_indices.size(),
                                          Q_offsets.data(),
                                          Q_offsets.size());
    }

    if (!A_values.empty()) {
      view.set_csr_constraint_matrix(A_values.data(),
                                     A_values.size(),
                                     A_indices.data(),
                                     A_indices.size(),
                                     A_offsets.data(),
                                     A_offsets.size());
    }

    if (uses_ranged_constraints) {
      if (!constraint_lower_bounds.empty()) {
        view.set_constraint_lower_bounds(constraint_lower_bounds.data(),
                                         constraint_lower_bounds.size());
      }
      if (!constraint_upper_bounds.empty()) {
        view.set_constraint_upper_bounds(constraint_upper_bounds.data(),
                                         constraint_upper_bounds.size());
      }
    } else {
      if (!row_types.empty()) { view.set_row_types(row_types.data(), row_types.size()); }
      if (!constraint_bounds.empty()) {
        view.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
      }
    }

    if (!variable_lower_bounds.empty()) {
      view.set_variable_lower_bounds(variable_lower_bounds.data(), variable_lower_bounds.size());
    }

    if (!variable_upper_bounds.empty()) {
      view.set_variable_upper_bounds(variable_upper_bounds.data(), variable_upper_bounds.size());
    }

    if (!variable_types.empty()) {
      view.set_variable_types(variable_types.data(), variable_types.size());
    }

    return view;
  }

  /**
   * @brief Check if this is a MIP (has integer variables).
   */
  bool is_mip() const
  {
    for (char vt : variable_types) {
      if (vt == CUOPT_INTEGER) { return true; }
    }
    return false;
  }
};

struct problem_and_stream_view_t {
  problem_and_stream_view_t() : cpu_data(nullptr), gpu_problem(nullptr), handle(nullptr) {}

  /**
   * @brief Ensure CUDA resources are initialized (lazy initialization).
   * Only call this when local solve is needed.
   */
  void ensure_cuda_initialized()
  {
    if (!handle) { handle = std::make_unique<raft::handle_t>(); }
  }

  raft::handle_t* get_handle_ptr()
  {
    ensure_cuda_initialized();
    return handle.get();
  }

  /**
   * @brief Check if this is a MIP problem.
   */
  bool is_mip() const
  {
    if (view.is_device_memory()) {
      // GPU path: check gpu_problem's problem category
      if (!gpu_problem) return false;
      auto cat = gpu_problem->get_problem_category();
      return (cat == problem_category_t::MIP) || (cat == problem_category_t::IP);
    } else {
      // CPU path: check variable types in cpu_data
      if (!cpu_data) return false;
      return cpu_data->is_mip();
    }
  }

  // Only ONE of these is allocated (optimized memory usage):
  std::unique_ptr<problem_cpu_data_t> cpu_data;  // for remote solve (CPU memory)
  std::unique_ptr<optimization_problem_t<cuopt_int_t, cuopt_float_t>>
    gpu_problem;  // for local solve (GPU memory)

  // Non-owning view pointing to whichever storage is active
  // Use view.is_device_memory() to check if data is on GPU or CPU
  cuopt::linear_programming::data_model_view_t<cuopt_int_t, cuopt_float_t> view;
  std::vector<char> gpu_variable_types;  // host copy for view when GPU data is used

  // Lazy-initialized CUDA handle (only created for local solve)
  std::unique_ptr<raft::handle_t> handle;

  /**
   * @brief Create a view pointing to GPU data from the gpu_problem.
   * Call this after gpu_problem is fully populated.
   */
  void create_view_from_gpu_problem()
  {
    if (!gpu_problem) return;
    auto& gpu = *gpu_problem;

    view.set_maximize(gpu.get_sense());
    view.set_objective_offset(gpu.get_objective_offset());
    view.set_objective_coefficients(gpu.get_objective_coefficients().data(), gpu.get_n_variables());
    view.set_csr_constraint_matrix(gpu.get_constraint_matrix_values().data(),
                                   gpu.get_constraint_matrix_values().size(),
                                   gpu.get_constraint_matrix_indices().data(),
                                   gpu.get_constraint_matrix_indices().size(),
                                   gpu.get_constraint_matrix_offsets().data(),
                                   gpu.get_constraint_matrix_offsets().size());

    if (!gpu.get_constraint_lower_bounds().is_empty()) {
      view.set_constraint_lower_bounds(gpu.get_constraint_lower_bounds().data(),
                                       gpu.get_n_constraints());
      view.set_constraint_upper_bounds(gpu.get_constraint_upper_bounds().data(),
                                       gpu.get_n_constraints());
    } else if (!gpu.get_row_types().is_empty()) {
      view.set_row_types(gpu.get_row_types().data(), gpu.get_n_constraints());
      view.set_constraint_bounds(gpu.get_constraint_bounds().data(), gpu.get_n_constraints());
    }

    view.set_variable_lower_bounds(gpu.get_variable_lower_bounds().data(), gpu.get_n_variables());
    view.set_variable_upper_bounds(gpu.get_variable_upper_bounds().data(), gpu.get_n_variables());

    if (gpu.get_n_variables() > 0) {
      std::vector<var_t> gpu_var_types(gpu.get_n_variables());
      raft::copy(gpu_var_types.data(),
                 gpu.get_variable_types().data(),
                 gpu.get_n_variables(),
                 gpu.get_handle_ptr()->get_stream());
      gpu.get_handle_ptr()->sync_stream();

      gpu_variable_types.resize(gpu.get_n_variables());
      for (cuopt_int_t i = 0; i < gpu.get_n_variables(); ++i) {
        gpu_variable_types[i] = (gpu_var_types[i] == var_t::INTEGER) ? 'I' : 'C';
      }
      view.set_variable_types(gpu_variable_types.data(), gpu.get_n_variables());
    }

    if (gpu.has_quadratic_objective()) {
      view.set_quadratic_objective_matrix(gpu.get_quadratic_objective_values().data(),
                                          gpu.get_quadratic_objective_values().size(),
                                          gpu.get_quadratic_objective_indices().data(),
                                          gpu.get_quadratic_objective_indices().size(),
                                          gpu.get_quadratic_objective_offsets().data(),
                                          gpu.get_quadratic_objective_offsets().size());
    }

    view.set_is_device_memory(true);
  }

  /**
   * @brief Create a view pointing to CPU data from cpu_data.
   * Call this after cpu_data is fully populated.
   */
  void create_view_from_cpu_data()
  {
    if (!cpu_data) return;
    view = cpu_data->create_view();
    view.set_is_device_memory(false);
  }
};

struct solution_and_stream_view_t {
  solution_and_stream_view_t(bool solution_for_mip, raft::handle_t* handle_ptr = nullptr)
    : is_mip(solution_for_mip), mip_solution_ptr(nullptr), lp_solution_ptr(nullptr)
  {
    // Store stream only if we have a handle (local solve)
    if (handle_ptr) { stream_view = handle_ptr->get_stream(); }
  }
  bool is_mip;
  mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution_ptr;
  optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* lp_solution_ptr;
  std::optional<rmm::cuda_stream_view> stream_view;  // Only present for local solve
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
    delete problem_and_stream;
    *problem_ptr = nullptr;
    if (std::string(e.what()).find("Error opening MPS file") != std::string::npos) {
      return CUOPT_MPS_FILE_ERROR;
    } else {
      return CUOPT_MPS_PARSE_ERROR;
    }
  }

  // Check remote solve configuration at creation time
  bool is_remote = is_remote_solve_enabled();

  if (is_remote) {
    // Remote: store in CPU memory
    problem_and_stream->cpu_data = std::make_unique<problem_cpu_data_t>();
    auto& cpu_data               = *problem_and_stream->cpu_data;
    const auto& mps              = *mps_data_model_ptr;

    cpu_data.num_constraints =
      static_cast<cuopt_int_t>(mps.get_constraint_matrix_offsets().size() - 1);
    cpu_data.num_variables    = static_cast<cuopt_int_t>(mps.get_objective_coefficients().size());
    cpu_data.maximize         = mps.get_sense();
    cpu_data.objective_offset = mps.get_objective_offset();

    cpu_data.objective_coefficients = mps.get_objective_coefficients();
    cpu_data.A_values               = mps.get_constraint_matrix_values();
    cpu_data.A_indices              = mps.get_constraint_matrix_indices();
    cpu_data.A_offsets              = mps.get_constraint_matrix_offsets();

    if (!mps.get_constraint_lower_bounds().empty() || !mps.get_constraint_upper_bounds().empty()) {
      cpu_data.uses_ranged_constraints = true;
      cpu_data.constraint_lower_bounds = mps.get_constraint_lower_bounds();
      cpu_data.constraint_upper_bounds = mps.get_constraint_upper_bounds();
    } else {
      cpu_data.uses_ranged_constraints = false;
      cpu_data.constraint_bounds       = mps.get_constraint_bounds();
      const auto& mps_row_types        = mps.get_row_types();
      cpu_data.row_types.resize(mps_row_types.size());
      for (size_t i = 0; i < mps_row_types.size(); ++i) {
        cpu_data.row_types[i] = mps_row_types[i];
      }
    }

    cpu_data.variable_lower_bounds = mps.get_variable_lower_bounds();
    cpu_data.variable_upper_bounds = mps.get_variable_upper_bounds();

    const auto& mps_var_types = mps.get_variable_types();
    cpu_data.variable_types.resize(mps_var_types.size());
    for (size_t i = 0; i < mps_var_types.size(); ++i) {
      cpu_data.variable_types[i] =
        (mps_var_types[i] == 'I' || mps_var_types[i] == 'B') ? CUOPT_INTEGER : CUOPT_CONTINUOUS;
    }

    // Create view pointing to CPU data
    problem_and_stream->create_view_from_cpu_data();
  } else {
    // Local: store in GPU memory using existing mps_data_model_to_optimization_problem
    problem_and_stream->gpu_problem =
      std::make_unique<optimization_problem_t<cuopt_int_t, cuopt_float_t>>(
        mps_data_model_to_optimization_problem(problem_and_stream->get_handle_ptr(),
                                               *mps_data_model_ptr));
    // Create view pointing to GPU data
    problem_and_stream->create_view_from_gpu_problem();
  }

  *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
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

  problem_and_stream_view_t* problem_and_stream = new problem_and_stream_view_t();
  bool is_remote                                = is_remote_solve_enabled();

  try {
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];

    if (is_remote) {
      // Remote: store in CPU memory
      problem_and_stream->cpu_data = std::make_unique<problem_cpu_data_t>();
      auto& cpu_data               = *problem_and_stream->cpu_data;

      cpu_data.num_constraints  = num_constraints;
      cpu_data.num_variables    = num_variables;
      cpu_data.maximize         = (objective_sense == CUOPT_MAXIMIZE);
      cpu_data.objective_offset = objective_offset;

      cpu_data.objective_coefficients.assign(objective_coefficients,
                                             objective_coefficients + num_variables);
      cpu_data.A_values.assign(constraint_matrix_coefficent_values,
                               constraint_matrix_coefficent_values + nnz);
      cpu_data.A_indices.assign(constraint_matrix_column_indices,
                                constraint_matrix_column_indices + nnz);
      cpu_data.A_offsets.assign(constraint_matrix_row_offsets,
                                constraint_matrix_row_offsets + num_constraints + 1);

      cpu_data.uses_ranged_constraints = false;
      cpu_data.row_types.assign(constraint_sense, constraint_sense + num_constraints);
      cpu_data.constraint_bounds.assign(rhs, rhs + num_constraints);

      cpu_data.variable_lower_bounds.assign(lower_bounds, lower_bounds + num_variables);
      cpu_data.variable_upper_bounds.assign(upper_bounds, upper_bounds + num_variables);
      cpu_data.variable_types.assign(variable_types, variable_types + num_variables);

      // Create view pointing to CPU data
      problem_and_stream->create_view_from_cpu_data();
    } else {
      // Local: store in GPU memory
      problem_and_stream->gpu_problem =
        std::make_unique<optimization_problem_t<cuopt_int_t, cuopt_float_t>>(
          problem_and_stream->get_handle_ptr());
      auto& gpu_problem = *problem_and_stream->gpu_problem;

      gpu_problem.set_maximize(objective_sense == CUOPT_MAXIMIZE);
      gpu_problem.set_objective_offset(objective_offset);
      gpu_problem.set_objective_coefficients(objective_coefficients, num_variables);
      gpu_problem.set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                            nnz,
                                            constraint_matrix_column_indices,
                                            nnz,
                                            constraint_matrix_row_offsets,
                                            num_constraints + 1);
      gpu_problem.set_row_types(constraint_sense, num_constraints);
      gpu_problem.set_constraint_bounds(rhs, num_constraints);
      gpu_problem.set_variable_lower_bounds(lower_bounds, num_variables);
      gpu_problem.set_variable_upper_bounds(upper_bounds, num_variables);

      // Convert variable types to enum
      std::vector<var_t> variable_types_host(num_variables);
      for (cuopt_int_t j = 0; j < num_variables; j++) {
        variable_types_host[j] =
          variable_types[j] == CUOPT_CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;
      }
      gpu_problem.set_variable_types(variable_types_host.data(), num_variables);

      // Create view pointing to GPU data
      problem_and_stream->create_view_from_gpu_problem();
    }

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const std::exception& e) {
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
      variable_upper_bounds == nullptr || variable_types == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_and_stream_view_t* problem_and_stream = new problem_and_stream_view_t();
  bool is_remote                                = is_remote_solve_enabled();

  try {
    cuopt_int_t nnz = constraint_matrix_row_offsets[num_constraints];

    if (is_remote) {
      // Remote: store in CPU memory
      problem_and_stream->cpu_data = std::make_unique<problem_cpu_data_t>();
      auto& cpu_data               = *problem_and_stream->cpu_data;

      cpu_data.num_constraints  = num_constraints;
      cpu_data.num_variables    = num_variables;
      cpu_data.maximize         = (objective_sense == CUOPT_MAXIMIZE);
      cpu_data.objective_offset = objective_offset;

      cpu_data.objective_coefficients.assign(objective_coefficients,
                                             objective_coefficients + num_variables);
      cpu_data.A_values.assign(constraint_matrix_coefficent_values,
                               constraint_matrix_coefficent_values + nnz);
      cpu_data.A_indices.assign(constraint_matrix_column_indices,
                                constraint_matrix_column_indices + nnz);
      cpu_data.A_offsets.assign(constraint_matrix_row_offsets,
                                constraint_matrix_row_offsets + num_constraints + 1);

      cpu_data.uses_ranged_constraints = true;
      cpu_data.constraint_lower_bounds.assign(constraint_lower_bounds,
                                              constraint_lower_bounds + num_constraints);
      cpu_data.constraint_upper_bounds.assign(constraint_upper_bounds,
                                              constraint_upper_bounds + num_constraints);

      cpu_data.variable_lower_bounds.assign(variable_lower_bounds,
                                            variable_lower_bounds + num_variables);
      cpu_data.variable_upper_bounds.assign(variable_upper_bounds,
                                            variable_upper_bounds + num_variables);
      cpu_data.variable_types.assign(variable_types, variable_types + num_variables);

      // Create view pointing to CPU data
      problem_and_stream->create_view_from_cpu_data();
    } else {
      // Local: store in GPU memory
      problem_and_stream->gpu_problem =
        std::make_unique<optimization_problem_t<cuopt_int_t, cuopt_float_t>>(
          problem_and_stream->get_handle_ptr());
      auto& gpu_problem = *problem_and_stream->gpu_problem;

      gpu_problem.set_maximize(objective_sense == CUOPT_MAXIMIZE);
      gpu_problem.set_objective_offset(objective_offset);
      gpu_problem.set_objective_coefficients(objective_coefficients, num_variables);
      gpu_problem.set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                            nnz,
                                            constraint_matrix_column_indices,
                                            nnz,
                                            constraint_matrix_row_offsets,
                                            num_constraints + 1);
      gpu_problem.set_constraint_lower_bounds(constraint_lower_bounds, num_constraints);
      gpu_problem.set_constraint_upper_bounds(constraint_upper_bounds, num_constraints);
      gpu_problem.set_variable_lower_bounds(variable_lower_bounds, num_variables);
      gpu_problem.set_variable_upper_bounds(variable_upper_bounds, num_variables);

      std::vector<var_t> variable_types_host(num_variables);
      for (cuopt_int_t j = 0; j < num_variables; j++) {
        variable_types_host[j] =
          variable_types[j] == CUOPT_CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;
      }
      gpu_problem.set_variable_types(variable_types_host.data(), num_variables);

      // Create view pointing to GPU data
      problem_and_stream->create_view_from_gpu_problem();
    }

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const std::exception& e) {
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

  problem_and_stream_view_t* problem_and_stream = new problem_and_stream_view_t();
  bool is_remote                                = is_remote_solve_enabled();

  try {
    cuopt_int_t Q_nnz = quadratic_objective_matrix_row_offsets[num_variables];
    cuopt_int_t nnz   = constraint_matrix_row_offsets[num_constraints];

    if (is_remote) {
      // Remote: store in CPU memory
      problem_and_stream->cpu_data = std::make_unique<problem_cpu_data_t>();
      auto& cpu_data               = *problem_and_stream->cpu_data;

      cpu_data.num_constraints  = num_constraints;
      cpu_data.num_variables    = num_variables;
      cpu_data.maximize         = (objective_sense == CUOPT_MAXIMIZE);
      cpu_data.objective_offset = objective_offset;

      cpu_data.objective_coefficients.assign(objective_coefficients,
                                             objective_coefficients + num_variables);

      cpu_data.Q_values.assign(quadratic_objective_matrix_coefficent_values,
                               quadratic_objective_matrix_coefficent_values + Q_nnz);
      cpu_data.Q_indices.assign(quadratic_objective_matrix_column_indices,
                                quadratic_objective_matrix_column_indices + Q_nnz);
      cpu_data.Q_offsets.assign(quadratic_objective_matrix_row_offsets,
                                quadratic_objective_matrix_row_offsets + num_variables + 1);

      cpu_data.A_values.assign(constraint_matrix_coefficent_values,
                               constraint_matrix_coefficent_values + nnz);
      cpu_data.A_indices.assign(constraint_matrix_column_indices,
                                constraint_matrix_column_indices + nnz);
      cpu_data.A_offsets.assign(constraint_matrix_row_offsets,
                                constraint_matrix_row_offsets + num_constraints + 1);

      cpu_data.uses_ranged_constraints = false;
      cpu_data.row_types.assign(constraint_sense, constraint_sense + num_constraints);
      cpu_data.constraint_bounds.assign(rhs, rhs + num_constraints);

      cpu_data.variable_lower_bounds.assign(lower_bounds, lower_bounds + num_variables);
      cpu_data.variable_upper_bounds.assign(upper_bounds, upper_bounds + num_variables);
      cpu_data.variable_types.assign(num_variables, CUOPT_CONTINUOUS);

      // Create view pointing to CPU data
      problem_and_stream->create_view_from_cpu_data();
    } else {
      // Local: store in GPU memory
      problem_and_stream->gpu_problem =
        std::make_unique<optimization_problem_t<cuopt_int_t, cuopt_float_t>>(
          problem_and_stream->get_handle_ptr());
      auto& gpu_problem = *problem_and_stream->gpu_problem;

      gpu_problem.set_maximize(objective_sense == CUOPT_MAXIMIZE);
      gpu_problem.set_objective_offset(objective_offset);
      gpu_problem.set_objective_coefficients(objective_coefficients, num_variables);
      gpu_problem.set_quadratic_objective_matrix(quadratic_objective_matrix_coefficent_values,
                                                 Q_nnz,
                                                 quadratic_objective_matrix_column_indices,
                                                 Q_nnz,
                                                 quadratic_objective_matrix_row_offsets,
                                                 num_variables + 1);
      gpu_problem.set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                            nnz,
                                            constraint_matrix_column_indices,
                                            nnz,
                                            constraint_matrix_row_offsets,
                                            num_constraints + 1);
      gpu_problem.set_row_types(constraint_sense, num_constraints);
      gpu_problem.set_constraint_bounds(rhs, num_constraints);
      gpu_problem.set_variable_lower_bounds(lower_bounds, num_variables);
      gpu_problem.set_variable_upper_bounds(upper_bounds, num_variables);

      // Create view pointing to GPU data
      problem_and_stream->create_view_from_gpu_problem();
    }

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const std::exception& e) {
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

  problem_and_stream_view_t* problem_and_stream = new problem_and_stream_view_t();
  bool is_remote                                = is_remote_solve_enabled();

  try {
    cuopt_int_t Q_nnz = quadratic_objective_matrix_row_offsets[num_variables];
    cuopt_int_t nnz   = constraint_matrix_row_offsets[num_constraints];

    if (is_remote) {
      // Remote: store in CPU memory
      problem_and_stream->cpu_data = std::make_unique<problem_cpu_data_t>();
      auto& cpu_data               = *problem_and_stream->cpu_data;

      cpu_data.num_constraints  = num_constraints;
      cpu_data.num_variables    = num_variables;
      cpu_data.maximize         = (objective_sense == CUOPT_MAXIMIZE);
      cpu_data.objective_offset = objective_offset;

      cpu_data.objective_coefficients.assign(objective_coefficients,
                                             objective_coefficients + num_variables);

      cpu_data.Q_values.assign(quadratic_objective_matrix_coefficent_values,
                               quadratic_objective_matrix_coefficent_values + Q_nnz);
      cpu_data.Q_indices.assign(quadratic_objective_matrix_column_indices,
                                quadratic_objective_matrix_column_indices + Q_nnz);
      cpu_data.Q_offsets.assign(quadratic_objective_matrix_row_offsets,
                                quadratic_objective_matrix_row_offsets + num_variables + 1);

      cpu_data.A_values.assign(constraint_matrix_coefficent_values,
                               constraint_matrix_coefficent_values + nnz);
      cpu_data.A_indices.assign(constraint_matrix_column_indices,
                                constraint_matrix_column_indices + nnz);
      cpu_data.A_offsets.assign(constraint_matrix_row_offsets,
                                constraint_matrix_row_offsets + num_constraints + 1);

      cpu_data.uses_ranged_constraints = true;
      cpu_data.constraint_lower_bounds.assign(constraint_lower_bounds,
                                              constraint_lower_bounds + num_constraints);
      cpu_data.constraint_upper_bounds.assign(constraint_upper_bounds,
                                              constraint_upper_bounds + num_constraints);

      cpu_data.variable_lower_bounds.assign(variable_lower_bounds,
                                            variable_lower_bounds + num_variables);
      cpu_data.variable_upper_bounds.assign(variable_upper_bounds,
                                            variable_upper_bounds + num_variables);
      cpu_data.variable_types.assign(num_variables, CUOPT_CONTINUOUS);

      // Create view pointing to CPU data
      problem_and_stream->create_view_from_cpu_data();
    } else {
      // Local: store in GPU memory
      problem_and_stream->gpu_problem =
        std::make_unique<optimization_problem_t<cuopt_int_t, cuopt_float_t>>(
          problem_and_stream->get_handle_ptr());
      auto& gpu_problem = *problem_and_stream->gpu_problem;

      gpu_problem.set_maximize(objective_sense == CUOPT_MAXIMIZE);
      gpu_problem.set_objective_offset(objective_offset);
      gpu_problem.set_objective_coefficients(objective_coefficients, num_variables);
      gpu_problem.set_quadratic_objective_matrix(quadratic_objective_matrix_coefficent_values,
                                                 Q_nnz,
                                                 quadratic_objective_matrix_column_indices,
                                                 Q_nnz,
                                                 quadratic_objective_matrix_row_offsets,
                                                 num_variables + 1);
      gpu_problem.set_csr_constraint_matrix(constraint_matrix_coefficent_values,
                                            nnz,
                                            constraint_matrix_column_indices,
                                            nnz,
                                            constraint_matrix_row_offsets,
                                            num_constraints + 1);
      gpu_problem.set_constraint_lower_bounds(constraint_lower_bounds, num_constraints);
      gpu_problem.set_constraint_upper_bounds(constraint_upper_bounds, num_constraints);
      gpu_problem.set_variable_lower_bounds(variable_lower_bounds, num_variables);
      gpu_problem.set_variable_upper_bounds(variable_upper_bounds, num_variables);

      // Create view pointing to GPU data
      problem_and_stream->create_view_from_gpu_problem();
    }

    *problem_ptr = static_cast<cuOptOptimizationProblem>(problem_and_stream);
  } catch (const std::exception& e) {
    delete problem_and_stream;
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

void cuOptDestroyProblem(cuOptOptimizationProblem* problem_ptr)
{
  if (problem_ptr == nullptr) { return; }
  if (*problem_ptr == nullptr) { return; }
  problem_and_stream_view_t* problem_and_stream =
    static_cast<problem_and_stream_view_t*>(*problem_ptr);
  delete problem_and_stream;
  *problem_ptr = nullptr;
}

cuopt_int_t cuOptGetNumConstraints(cuOptOptimizationProblem problem,
                                   cuopt_int_t* num_constraints_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_constraints_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  if (!problem_and_stream_view->view.is_device_memory()) {
    *num_constraints_ptr = problem_and_stream_view->cpu_data->num_constraints;
  } else {
    *num_constraints_ptr = problem_and_stream_view->gpu_problem->get_n_constraints();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetNumVariables(cuOptOptimizationProblem problem, cuopt_int_t* num_variables_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_variables_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  if (!problem_and_stream_view->view.is_device_memory()) {
    *num_variables_ptr = problem_and_stream_view->cpu_data->num_variables;
  } else {
    *num_variables_ptr = problem_and_stream_view->gpu_problem->get_n_variables();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveSense(cuOptOptimizationProblem problem,
                                   cuopt_int_t* objective_sense_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_sense_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  if (!problem_and_stream_view->view.is_device_memory()) {
    *objective_sense_ptr =
      problem_and_stream_view->cpu_data->maximize ? CUOPT_MAXIMIZE : CUOPT_MINIMIZE;
  } else {
    *objective_sense_ptr =
      problem_and_stream_view->gpu_problem->get_sense() ? CUOPT_MAXIMIZE : CUOPT_MINIMIZE;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveOffset(cuOptOptimizationProblem problem,
                                    cuopt_float_t* objective_offset_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_offset_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  if (!problem_and_stream_view->view.is_device_memory()) {
    *objective_offset_ptr = problem_and_stream_view->cpu_data->objective_offset;
  } else {
    *objective_offset_ptr = problem_and_stream_view->gpu_problem->get_objective_offset();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetObjectiveCoefficients(cuOptOptimizationProblem problem,
                                          cuopt_float_t* objective_coefficients_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (objective_coefficients_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& coeffs = problem_and_stream_view->cpu_data->objective_coefficients;
    std::copy(coeffs.begin(), coeffs.end(), objective_coefficients_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    raft::copy(objective_coefficients_ptr,
               gpu_problem.get_objective_coefficients().data(),
               gpu_problem.get_n_variables(),
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetNumNonZeros(cuOptOptimizationProblem problem,
                                cuopt_int_t* num_non_zero_elements_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_non_zero_elements_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);
  if (!problem_and_stream_view->view.is_device_memory()) {
    *num_non_zero_elements_ptr =
      static_cast<cuopt_int_t>(problem_and_stream_view->cpu_data->A_values.size());
  } else {
    *num_non_zero_elements_ptr = static_cast<cuopt_int_t>(
      problem_and_stream_view->gpu_problem->get_constraint_matrix_values().size());
  }
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

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& cpu_data = *problem_and_stream_view->cpu_data;
    std::copy(
      cpu_data.A_values.begin(), cpu_data.A_values.end(), constraint_matrix_coefficients_ptr);
    std::copy(
      cpu_data.A_indices.begin(), cpu_data.A_indices.end(), constraint_matrix_column_indices_ptr);
    std::copy(
      cpu_data.A_offsets.begin(), cpu_data.A_offsets.end(), constraint_matrix_row_offsets_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    auto stream             = gpu_problem.get_handle_ptr()->get_stream();
    raft::copy(constraint_matrix_coefficients_ptr,
               gpu_problem.get_constraint_matrix_values().data(),
               gpu_problem.get_constraint_matrix_values().size(),
               stream);
    raft::copy(constraint_matrix_column_indices_ptr,
               gpu_problem.get_constraint_matrix_indices().data(),
               gpu_problem.get_constraint_matrix_indices().size(),
               stream);
    raft::copy(constraint_matrix_row_offsets_ptr,
               gpu_problem.get_constraint_matrix_offsets().data(),
               gpu_problem.get_constraint_matrix_offsets().size(),
               stream);
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintSense(cuOptOptimizationProblem problem, char* constraint_sense_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (constraint_sense_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& row_types = problem_and_stream_view->cpu_data->row_types;
    std::copy(row_types.begin(), row_types.end(), constraint_sense_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    raft::copy(constraint_sense_ptr,
               gpu_problem.get_row_types().data(),
               gpu_problem.get_row_types().size(),
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintRightHandSide(cuOptOptimizationProblem problem,
                                            cuopt_float_t* rhs_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (rhs_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& bounds = problem_and_stream_view->cpu_data->constraint_bounds;
    std::copy(bounds.begin(), bounds.end(), rhs_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    raft::copy(rhs_ptr,
               gpu_problem.get_constraint_bounds().data(),
               gpu_problem.get_constraint_bounds().size(),
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintLowerBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* lower_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (lower_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& bounds = problem_and_stream_view->cpu_data->constraint_lower_bounds;
    std::copy(bounds.begin(), bounds.end(), lower_bounds_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    raft::copy(lower_bounds_ptr,
               gpu_problem.get_constraint_lower_bounds().data(),
               gpu_problem.get_constraint_lower_bounds().size(),
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetConstraintUpperBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* upper_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (upper_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& bounds = problem_and_stream_view->cpu_data->constraint_upper_bounds;
    std::copy(bounds.begin(), bounds.end(), upper_bounds_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    raft::copy(upper_bounds_ptr,
               gpu_problem.get_constraint_upper_bounds().data(),
               gpu_problem.get_constraint_upper_bounds().size(),
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableLowerBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* lower_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (lower_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& bounds = problem_and_stream_view->cpu_data->variable_lower_bounds;
    std::copy(bounds.begin(), bounds.end(), lower_bounds_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    raft::copy(lower_bounds_ptr,
               gpu_problem.get_variable_lower_bounds().data(),
               gpu_problem.get_variable_lower_bounds().size(),
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableUpperBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* upper_bounds_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (upper_bounds_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& bounds = problem_and_stream_view->cpu_data->variable_upper_bounds;
    std::copy(bounds.begin(), bounds.end(), upper_bounds_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    raft::copy(upper_bounds_ptr,
               gpu_problem.get_variable_upper_bounds().data(),
               gpu_problem.get_variable_upper_bounds().size(),
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetVariableTypes(cuOptOptimizationProblem problem, char* variable_types_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (variable_types_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  problem_and_stream_view_t* problem_and_stream_view =
    static_cast<problem_and_stream_view_t*>(problem);

  if (!problem_and_stream_view->view.is_device_memory()) {
    const auto& var_types = problem_and_stream_view->cpu_data->variable_types;
    std::copy(var_types.begin(), var_types.end(), variable_types_ptr);
  } else {
    const auto& gpu_problem = *problem_and_stream_view->gpu_problem;
    auto num_vars           = gpu_problem.get_n_variables();
    std::vector<var_t> gpu_var_types(num_vars);
    raft::copy(gpu_var_types.data(),
               gpu_problem.get_variable_types().data(),
               num_vars,
               gpu_problem.get_handle_ptr()->get_stream());
    gpu_problem.get_handle_ptr()->sync_stream();
    // Convert from var_t enum to char
    for (cuopt_int_t i = 0; i < num_vars; ++i) {
      variable_types_ptr[i] =
        (gpu_var_types[i] == var_t::CONTINUOUS) ? CUOPT_CONTINUOUS : CUOPT_INTEGER;
    }
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
  *is_mip_ptr = static_cast<cuopt_int_t>(problem_and_stream_view->is_mip());
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
  solver_settings_t<cuopt_int_t, cuopt_float_t>* solver_settings =
    static_cast<solver_settings_t<cuopt_int_t, cuopt_float_t>*>(settings);

  bool is_mip = problem_and_stream_view->is_mip();

  // Use the view - solve_lp/solve_mip will check is_device_memory() to determine path
  const auto& view = problem_and_stream_view->view;

  if (view.is_device_memory()) {
    // Local path: data is already on GPU
    // Use gpu_problem directly for optimal performance (no extra copy)
    auto& gpu_problem = *problem_and_stream_view->gpu_problem;

    if (is_mip) {
      mip_solver_settings_t<cuopt_int_t, cuopt_float_t>& mip_settings =
        solver_settings->get_mip_settings();

      solution_and_stream_view_t* solution_and_stream_view =
        new solution_and_stream_view_t(true, problem_and_stream_view->handle.get());

      solution_and_stream_view->mip_solution_ptr = new mip_solution_t<cuopt_int_t, cuopt_float_t>(
        solve_mip<cuopt_int_t, cuopt_float_t>(gpu_problem, mip_settings));

      *solution_ptr = static_cast<cuOptSolution>(solution_and_stream_view);

      cuopt::utilities::printTimestamp("CUOPT_SOLVE_RETURN");

      return static_cast<cuopt_int_t>(
        solution_and_stream_view->mip_solution_ptr->get_error_status().get_error_type());
    } else {
      pdlp_solver_settings_t<cuopt_int_t, cuopt_float_t>& pdlp_settings =
        solver_settings->get_pdlp_settings();

      solution_and_stream_view_t* solution_and_stream_view =
        new solution_and_stream_view_t(false, problem_and_stream_view->handle.get());

      solution_and_stream_view->lp_solution_ptr =
        new optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>(
          solve_lp<cuopt_int_t, cuopt_float_t>(gpu_problem, pdlp_settings));

      *solution_ptr = static_cast<cuOptSolution>(solution_and_stream_view);

      cuopt::utilities::printTimestamp("CUOPT_SOLVE_RETURN");

      return static_cast<cuopt_int_t>(
        solution_and_stream_view->lp_solution_ptr->get_error_status().get_error_type());
    }
  } else {
    // CPU path: use view directly - solve_lp/solve_mip handle remote vs local conversion
    // For remote solve, handle may be nullptr (no CUDA)
    // For local solve with CPU data, handle will be created lazily
    raft::handle_t* handle_ptr =
      is_remote_solve_enabled() ? nullptr : problem_and_stream_view->get_handle_ptr();

    if (is_mip) {
      mip_solver_settings_t<cuopt_int_t, cuopt_float_t>& mip_settings =
        solver_settings->get_mip_settings();

      solution_and_stream_view_t* solution_and_stream_view =
        new solution_and_stream_view_t(true, handle_ptr);

      solution_and_stream_view->mip_solution_ptr = new mip_solution_t<cuopt_int_t, cuopt_float_t>(
        solve_mip<cuopt_int_t, cuopt_float_t>(handle_ptr, view, mip_settings));

      *solution_ptr = static_cast<cuOptSolution>(solution_and_stream_view);

      cuopt::utilities::printTimestamp("CUOPT_SOLVE_RETURN");

      return static_cast<cuopt_int_t>(
        solution_and_stream_view->mip_solution_ptr->get_error_status().get_error_type());
    } else {
      pdlp_solver_settings_t<cuopt_int_t, cuopt_float_t>& pdlp_settings =
        solver_settings->get_pdlp_settings();

      solution_and_stream_view_t* solution_and_stream_view =
        new solution_and_stream_view_t(false, handle_ptr);

      solution_and_stream_view->lp_solution_ptr =
        new optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>(
          solve_lp<cuopt_int_t, cuopt_float_t>(handle_ptr, view, pdlp_settings));

      *solution_ptr = static_cast<cuOptSolution>(solution_and_stream_view);

      cuopt::utilities::printTimestamp("CUOPT_SOLVE_RETURN");

      return static_cast<cuopt_int_t>(
        solution_and_stream_view->lp_solution_ptr->get_error_status().get_error_type());
    }
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
    if (mip_solution->is_device_memory()) {
      const rmm::device_uvector<cuopt_float_t>& solution_values = mip_solution->get_solution();
      raft::copy(solution_values_ptr,
                 solution_values.data(),
                 solution_values.size(),
                 solution_and_stream_view->stream_view.value());
      solution_and_stream_view->stream_view->synchronize();
    } else {
      const std::vector<cuopt_float_t>& solution_values = mip_solution->get_solution_host();
      std::copy(solution_values.begin(), solution_values.end(), solution_values_ptr);
    }
  } else {
    optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* optimization_problem_solution =
      static_cast<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>*>(
        solution_and_stream_view->lp_solution_ptr);
    if (optimization_problem_solution->is_device_memory()) {
      const rmm::device_uvector<cuopt_float_t>& solution_values =
        optimization_problem_solution->get_primal_solution();
      raft::copy(solution_values_ptr,
                 solution_values.data(),
                 solution_values.size(),
                 solution_and_stream_view->stream_view.value());
      solution_and_stream_view->stream_view->synchronize();
    } else {
      const std::vector<cuopt_float_t>& solution_values =
        optimization_problem_solution->get_primal_solution_host();
      std::copy(solution_values.begin(), solution_values.end(), solution_values_ptr);
    }
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
    if (optimization_problem_solution->is_device_memory()) {
      const rmm::device_uvector<cuopt_float_t>& dual_solution =
        optimization_problem_solution->get_dual_solution();
      raft::copy(dual_solution_ptr,
                 dual_solution.data(),
                 dual_solution.size(),
                 solution_and_stream_view->stream_view.value());
      solution_and_stream_view->stream_view->synchronize();
    } else {
      const std::vector<cuopt_float_t>& dual_solution =
        optimization_problem_solution->get_dual_solution_host();
      std::copy(dual_solution.begin(), dual_solution.end(), dual_solution_ptr);
    }
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
    if (optimization_problem_solution->is_device_memory()) {
      const rmm::device_uvector<cuopt_float_t>& reduced_cost =
        optimization_problem_solution->get_reduced_cost();
      raft::copy(reduced_cost_ptr,
                 reduced_cost.data(),
                 reduced_cost.size(),
                 solution_and_stream_view->stream_view.value());
      solution_and_stream_view->stream_view->synchronize();
    } else {
      const std::vector<cuopt_float_t>& reduced_cost =
        optimization_problem_solution->get_reduced_cost_host();
      std::copy(reduced_cost.begin(), reduced_cost.end(), reduced_cost_ptr);
    }
    return CUOPT_SUCCESS;
  }
}
