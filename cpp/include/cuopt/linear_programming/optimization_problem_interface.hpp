/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/utilities/internals.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

enum class var_t { CONTINUOUS = 0, INTEGER };
enum class problem_category_t : int8_t { LP = 0, MIP = 1, IP = 2 };

template <typename i_t, typename f_t>
class optimization_problem_t;
template <typename i_t, typename f_t>
class cpu_optimization_problem_t;
template <typename i_t, typename f_t>
class pdlp_solver_settings_t;
template <typename i_t, typename f_t>
class mip_solver_settings_t;
template <typename i_t, typename f_t>
class lp_solution_interface_t;
template <typename i_t, typename f_t>
class mip_solution_interface_t;

/**
 * @brief Interface for optimization problem implementations that can store data
 *        in either CPU or GPU memory.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 *
 * This interface provides setters that accept both CPU and GPU pointers,
 * and getters in two forms:
 * - Device getters returning rmm::device_uvector (GPU memory)
 * - Host getters returning std::vector (CPU memory)
 */
template <typename i_t, typename f_t>
class optimization_problem_interface_t {
 public:
  static_assert(std::is_integral<i_t>::value,
                "'optimization_problem_interface_t' accepts only integer types for indexes");
  static_assert(std::is_floating_point<f_t>::value,
                "'optimization_problem_interface_t' accepts only floating point types for weights");

  virtual ~optimization_problem_interface_t() = default;

  // ============================================================================
  // Setters (accept both CPU and GPU pointers)
  // ============================================================================

  /**
   * @brief Set the sense of optimization to maximize.
   * @param[in] maximize true means to maximize the objective function, else minimize.
   */
  virtual void set_maximize(bool maximize) = 0;

  /**
   * @brief Set the constraint matrix (A) in CSR format.
   * @param[in] A_values Values of the CSR representation (device or host pointer)
   * @param size_values Size of the A_values array
   * @param[in] A_indices Indices of the CSR representation (device or host pointer)
   * @param size_indices Size of the A_indices array
   * @param[in] A_offsets Offsets of the CSR representation (device or host pointer)
   * @param size_offsets Size of the A_offsets array
   */
  virtual void set_csr_constraint_matrix(const f_t* A_values,
                                         i_t size_values,
                                         const i_t* A_indices,
                                         i_t size_indices,
                                         const i_t* A_offsets,
                                         i_t size_offsets) = 0;

  /**
   * @brief Set the constraint bounds (b / right-hand side) array.
   * @param[in] b Device or host memory pointer
   * @param size Size of the b array
   */
  virtual void set_constraint_bounds(const f_t* b, i_t size) = 0;

  /**
   * @brief Set the objective coefficients (c) array.
   * @param[in] c Device or host memory pointer
   * @param size Size of the c array
   */
  virtual void set_objective_coefficients(const f_t* c, i_t size) = 0;

  /**
   * @brief Set the scaling factor of the objective function.
   * @param objective_scaling_factor Objective scaling factor value
   */
  virtual void set_objective_scaling_factor(f_t objective_scaling_factor) = 0;

  /**
   * @brief Set the offset of the objective function.
   * @param objective_offset Objective offset value
   */
  virtual void set_objective_offset(f_t objective_offset) = 0;

  /**
   * @brief Set the quadratic objective matrix (Q) in CSR format.
   * @param[in] Q_values Values of the CSR representation
   * @param size_values Size of the Q_values array
   * @param[in] Q_indices Indices of the CSR representation
   * @param size_indices Size of the Q_indices array
   * @param[in] Q_offsets Offsets of the CSR representation
   * @param size_offsets Size of the Q_offsets array
   * @param validate_positive_semi_definite Whether to validate if the matrix is positive semi
   * definite
   */
  virtual void set_quadratic_objective_matrix(const f_t* Q_values,
                                              i_t size_values,
                                              const i_t* Q_indices,
                                              i_t size_indices,
                                              const i_t* Q_offsets,
                                              i_t size_offsets,
                                              bool validate_positive_semi_definite = false) = 0;

  /**
   * @brief Set the variables (x) lower bounds.
   * @param[in] variable_lower_bounds Device or host memory pointer
   * @param size Size of the variable_lower_bounds array
   */
  virtual void set_variable_lower_bounds(const f_t* variable_lower_bounds, i_t size) = 0;

  /**
   * @brief Set the variables (x) upper bounds.
   * @param[in] variable_upper_bounds Device or host memory pointer
   * @param size Size of the variable_upper_bounds array
   */
  virtual void set_variable_upper_bounds(const f_t* variable_upper_bounds, i_t size) = 0;

  /**
   * @brief Set the variables types.
   * @param[in] variable_types Device or host memory pointer to a var_t array
   * @param size Size of the variable_types array
   */
  virtual void set_variable_types(const var_t* variable_types, i_t size) = 0;

  /**
   * @brief Set the problem category.
   * @param[in] category Problem category value
   */
  virtual void set_problem_category(const problem_category_t& category) = 0;

  /**
   * @brief Set the constraints lower bounds.
   * @param[in] constraint_lower_bounds Device or host memory pointer
   * @param size Size of the constraint_lower_bounds array
   */
  virtual void set_constraint_lower_bounds(const f_t* constraint_lower_bounds, i_t size) = 0;

  /**
   * @brief Set the constraints upper bounds.
   * @param[in] constraint_upper_bounds Device or host memory pointer
   * @param size Size of the constraint_upper_bounds array
   */
  virtual void set_constraint_upper_bounds(const f_t* constraint_upper_bounds, i_t size) = 0;

  /**
   * @brief Set the type of each row (constraint).
   * @param[in] row_types Device or host memory pointer to a character array
   * @param size Size of the row_types array
   */
  virtual void set_row_types(const char* row_types, i_t size) = 0;

  /**
   * @brief Set the name of the objective function.
   * @param[in] objective_name Objective name value
   */
  virtual void set_objective_name(const std::string& objective_name) = 0;

  /**
   * @brief Set the problem name.
   * @param[in] problem_name Problem name value
   */
  virtual void set_problem_name(const std::string& problem_name) = 0;

  /**
   * @brief Set the variables names.
   * @param[in] variable_names Variable names values
   */
  virtual void set_variable_names(const std::vector<std::string>& variable_names) = 0;

  /**
   * @brief Set the row names.
   * @param[in] row_names Row names value
   */
  virtual void set_row_names(const std::vector<std::string>& row_names) = 0;

  // ============================================================================
  // Getters - Device memory (GPU)
  // ============================================================================

  virtual i_t get_n_variables() const                                           = 0;
  virtual i_t get_n_constraints() const                                         = 0;
  virtual i_t get_nnz() const                                                   = 0;
  virtual i_t get_n_integers() const                                            = 0;
  virtual const rmm::device_uvector<f_t>& get_constraint_matrix_values() const  = 0;
  virtual rmm::device_uvector<f_t>& get_constraint_matrix_values()              = 0;
  virtual const rmm::device_uvector<i_t>& get_constraint_matrix_indices() const = 0;
  virtual rmm::device_uvector<i_t>& get_constraint_matrix_indices()             = 0;
  virtual const rmm::device_uvector<i_t>& get_constraint_matrix_offsets() const = 0;
  virtual rmm::device_uvector<i_t>& get_constraint_matrix_offsets()             = 0;
  virtual const rmm::device_uvector<f_t>& get_constraint_bounds() const         = 0;
  virtual rmm::device_uvector<f_t>& get_constraint_bounds()                     = 0;
  virtual const rmm::device_uvector<f_t>& get_objective_coefficients() const    = 0;
  virtual rmm::device_uvector<f_t>& get_objective_coefficients()                = 0;
  virtual f_t get_objective_scaling_factor() const                              = 0;
  virtual f_t get_objective_offset() const                                      = 0;
  virtual const rmm::device_uvector<f_t>& get_variable_lower_bounds() const     = 0;
  virtual rmm::device_uvector<f_t>& get_variable_lower_bounds()                 = 0;
  virtual const rmm::device_uvector<f_t>& get_variable_upper_bounds() const     = 0;
  virtual rmm::device_uvector<f_t>& get_variable_upper_bounds()                 = 0;
  virtual const rmm::device_uvector<f_t>& get_constraint_lower_bounds() const   = 0;
  virtual rmm::device_uvector<f_t>& get_constraint_lower_bounds()               = 0;
  virtual const rmm::device_uvector<f_t>& get_constraint_upper_bounds() const   = 0;
  virtual rmm::device_uvector<f_t>& get_constraint_upper_bounds()               = 0;
  virtual const rmm::device_uvector<char>& get_row_types() const                = 0;
  virtual const rmm::device_uvector<var_t>& get_variable_types() const          = 0;
  virtual bool get_sense() const                                                = 0;
  virtual bool empty() const                                                    = 0;
  virtual std::string get_objective_name() const                                = 0;
  virtual std::string get_problem_name() const                                  = 0;
  virtual problem_category_t get_problem_category() const                       = 0;
  virtual const std::vector<std::string>& get_variable_names() const            = 0;
  virtual const std::vector<std::string>& get_row_names() const                 = 0;
  virtual const std::vector<i_t>& get_quadratic_objective_offsets() const       = 0;
  virtual const std::vector<i_t>& get_quadratic_objective_indices() const       = 0;
  virtual const std::vector<f_t>& get_quadratic_objective_values() const        = 0;
  virtual bool has_quadratic_objective() const                                  = 0;

  // ============================================================================
  // Getters - Host memory (CPU)
  // ============================================================================

  /**
   * @brief Get constraint matrix values in host memory.
   * @return std::vector containing the constraint matrix values
   */
  virtual std::vector<f_t> get_constraint_matrix_values_host() const = 0;

  /**
   * @brief Get constraint matrix indices in host memory.
   * @return std::vector containing the constraint matrix indices
   */
  virtual std::vector<i_t> get_constraint_matrix_indices_host() const = 0;

  /**
   * @brief Get constraint matrix offsets in host memory.
   * @return std::vector containing the constraint matrix offsets
   */
  virtual std::vector<i_t> get_constraint_matrix_offsets_host() const = 0;

  /**
   * @brief Get constraint bounds in host memory.
   * @return std::vector containing the constraint bounds
   */
  virtual std::vector<f_t> get_constraint_bounds_host() const = 0;

  /**
   * @brief Get objective coefficients in host memory.
   * @return std::vector containing the objective coefficients
   */
  virtual std::vector<f_t> get_objective_coefficients_host() const = 0;

  /**
   * @brief Get variable lower bounds in host memory.
   * @return std::vector containing the variable lower bounds
   */
  virtual std::vector<f_t> get_variable_lower_bounds_host() const = 0;

  /**
   * @brief Get variable upper bounds in host memory.
   * @return std::vector containing the variable upper bounds
   */
  virtual std::vector<f_t> get_variable_upper_bounds_host() const = 0;

  /**
   * @brief Get constraint lower bounds in host memory.
   * @return std::vector containing the constraint lower bounds
   */
  virtual std::vector<f_t> get_constraint_lower_bounds_host() const = 0;

  /**
   * @brief Get constraint upper bounds in host memory.
   * @return std::vector containing the constraint upper bounds
   */
  virtual std::vector<f_t> get_constraint_upper_bounds_host() const = 0;

  /**
   * @brief Get row types in host memory.
   * @return std::vector containing the row types
   */
  virtual std::vector<char> get_row_types_host() const = 0;

  /**
   * @brief Get variable types in host memory.
   * @return std::vector containing the variable types
   */
  virtual std::vector<var_t> get_variable_types_host() const = 0;

  // ============================================================================
  // File I/O
  // ============================================================================

  /**
   * @brief Write the optimization problem to an MPS file.
   * @param[in] mps_file_path Path to the output MPS file
   */
  virtual void write_to_mps(const std::string& mps_file_path) = 0;

  // ============================================================================
  // Comparison
  // ============================================================================

  /**
   * @brief Check if this problem is equivalent to another problem.
   * @param[in] other The other optimization problem to compare against
   * @return true if the problems are equivalent (up to permutation of variables/constraints)
   */
  virtual bool is_equivalent(const optimization_problem_interface_t<i_t, f_t>& other) const = 0;

  // ============================================================================
  // C API Support: Copy to Host (Polymorphic)
  // ============================================================================

  /**
   * @brief Copy objective coefficients to host memory (polymorphic)
   * GPU implementation: cudaMemcpy from device to host
   * CPU implementation: std::copy from host vector
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of elements to copy
   */
  virtual void copy_objective_coefficients_to_host(f_t* output, i_t size) const = 0;

  /**
   * @brief Copy constraint matrix to host memory (polymorphic)
   * @param[out] values Output buffer for matrix values
   * @param[out] indices Output buffer for column indices
   * @param[out] offsets Output buffer for row offsets
   * @param[in] num_values Number of non-zero values
   * @param[in] num_indices Number of indices (should equal num_values)
   * @param[in] num_offsets Number of row offsets (num_constraints + 1)
   */
  virtual void copy_constraint_matrix_to_host(f_t* values,
                                              i_t* indices,
                                              i_t* offsets,
                                              i_t num_values,
                                              i_t num_indices,
                                              i_t num_offsets) const = 0;

  /**
   * @brief Copy constraint sense/row types to host memory (polymorphic)
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of constraints
   */
  virtual void copy_row_types_to_host(char* output, i_t size) const = 0;

  /**
   * @brief Copy constraint bounds (RHS) to host memory (polymorphic)
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of constraints
   */
  virtual void copy_constraint_bounds_to_host(f_t* output, i_t size) const = 0;

  /**
   * @brief Copy constraint lower bounds to host memory (polymorphic)
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of constraints
   */
  virtual void copy_constraint_lower_bounds_to_host(f_t* output, i_t size) const = 0;

  /**
   * @brief Copy constraint upper bounds to host memory (polymorphic)
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of constraints
   */
  virtual void copy_constraint_upper_bounds_to_host(f_t* output, i_t size) const = 0;

  /**
   * @brief Copy variable lower bounds to host memory (polymorphic)
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of variables
   */
  virtual void copy_variable_lower_bounds_to_host(f_t* output, i_t size) const = 0;

  /**
   * @brief Copy variable upper bounds to host memory (polymorphic)
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of variables
   */
  virtual void copy_variable_upper_bounds_to_host(f_t* output, i_t size) const = 0;

  /**
   * @brief Copy variable types to host memory (polymorphic)
   * @param[out] output Pointer to host memory buffer
   * @param[in] size Number of variables
   */
  virtual void copy_variable_types_to_host(var_t* output, i_t size) const = 0;

  // ============================================================================
  // Conversion
  // ============================================================================

  /**
   * @brief Convert to a GPU-backed optimization_problem_t.
   *
   * For optimization_problem_t (GPU): returns nullptr (already is one).
   * For cpu_optimization_problem_t: creates new GPU problem, copies data, returns owned pointer.
   *
   * Usage pattern:
   *   auto temp = problem_interface->to_optimization_problem(&handle);
   *   optimization_problem_t& op = temp ? *temp : static_cast<optimization_problem_t&>(*this);
   *
   * @param handle_ptr RAFT handle with CUDA resources for GPU memory allocation.
   *                   Required for CPU->GPU conversion. Ignored for GPU problems.
   * @return unique_ptr to new GPU problem, or nullptr if already a GPU problem
   */
  virtual std::unique_ptr<optimization_problem_t<i_t, f_t>> to_optimization_problem(
    raft::handle_t const* handle_ptr = nullptr) = 0;
};

}  // namespace cuopt::linear_programming
