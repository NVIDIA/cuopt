/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

// Forward declarations
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
  // Conversion
  // ============================================================================

  /**
   * @brief Convert to optimization_problem_t (moves data to GPU if needed)
   * @return optimization_problem_t<i_t, f_t> GPU-backed optimization problem
   */
  virtual optimization_problem_t<i_t, f_t> to_optimization_problem() = 0;

  // ============================================================================
  // Getters - Host memory (CPU) - NEW
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
  // Remote Execution (Polymorphic Dispatch)
  // ============================================================================

  /**
   * @brief Solve LP problem using remote execution (polymorphic)
   * This method dispatches to the appropriate solve_lp_remote overload based on
   * the concrete type (GPU or CPU).
   * @param[in] settings PDLP solver settings
   * @return Pointer to solution interface
   */
  virtual std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
    pdlp_solver_settings_t<i_t, f_t> const& settings,
    bool problem_checking     = true,
    bool use_pdlp_solver_mode = true) const = 0;

  /**
   * @brief Solve MIP problem using remote execution (polymorphic)
   * This method dispatches to the appropriate solve_mip_remote overload based on
   * the concrete type (GPU or CPU).
   * @param[in] settings MIP solver settings
   * @return Pointer to solution interface
   */
  virtual std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
    mip_solver_settings_t<i_t, f_t> const& settings) const = 0;

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
};

// ==============================================================================
// GPU Implementation
// ==============================================================================

/**
 * @brief GPU-based implementation of optimization_problem_interface_t.
 *
 * This implementation stores all data in GPU memory using rmm::device_uvector.
 * It implements both device getters (returning rmm::device_uvector references)
 * and host getters (returning std::vector by copying from GPU to CPU).
 */
template <typename i_t, typename f_t>
class gpu_optimization_problem_t : public optimization_problem_interface_t<i_t, f_t> {
 public:
  explicit gpu_optimization_problem_t(raft::handle_t const* handle_ptr);

  // Setters
  void set_maximize(bool maximize) override;
  void set_csr_constraint_matrix(const f_t* A_values,
                                 i_t size_values,
                                 const i_t* A_indices,
                                 i_t size_indices,
                                 const i_t* A_offsets,
                                 i_t size_offsets) override;
  void set_constraint_bounds(const f_t* b, i_t size) override;
  void set_objective_coefficients(const f_t* c, i_t size) override;
  void set_objective_scaling_factor(f_t objective_scaling_factor) override;
  void set_objective_offset(f_t objective_offset) override;
  void set_quadratic_objective_matrix(const f_t* Q_values,
                                      i_t size_values,
                                      const i_t* Q_indices,
                                      i_t size_indices,
                                      const i_t* Q_offsets,
                                      i_t size_offsets,
                                      bool validate_positive_semi_definite = false) override;
  void set_variable_lower_bounds(const f_t* variable_lower_bounds, i_t size) override;
  void set_variable_upper_bounds(const f_t* variable_upper_bounds, i_t size) override;
  void set_variable_types(const var_t* variable_types, i_t size) override;
  void set_problem_category(const problem_category_t& category) override;
  void set_constraint_lower_bounds(const f_t* constraint_lower_bounds, i_t size) override;
  void set_constraint_upper_bounds(const f_t* constraint_upper_bounds, i_t size) override;
  void set_row_types(const char* row_types, i_t size) override;
  void set_objective_name(const std::string& objective_name) override;
  void set_problem_name(const std::string& problem_name) override;
  void set_variable_names(const std::vector<std::string>& variable_names) override;
  void set_row_names(const std::vector<std::string>& row_names) override;

  // Device getters
  i_t get_n_variables() const override;
  i_t get_n_constraints() const override;
  i_t get_nnz() const override;
  i_t get_n_integers() const override;
  const rmm::device_uvector<f_t>& get_constraint_matrix_values() const override;
  rmm::device_uvector<f_t>& get_constraint_matrix_values() override;
  const rmm::device_uvector<i_t>& get_constraint_matrix_indices() const override;
  rmm::device_uvector<i_t>& get_constraint_matrix_indices() override;
  const rmm::device_uvector<i_t>& get_constraint_matrix_offsets() const override;
  rmm::device_uvector<i_t>& get_constraint_matrix_offsets() override;
  const rmm::device_uvector<f_t>& get_constraint_bounds() const override;
  rmm::device_uvector<f_t>& get_constraint_bounds() override;
  const rmm::device_uvector<f_t>& get_objective_coefficients() const override;
  rmm::device_uvector<f_t>& get_objective_coefficients() override;
  f_t get_objective_scaling_factor() const override;
  f_t get_objective_offset() const override;
  const rmm::device_uvector<f_t>& get_variable_lower_bounds() const override;
  rmm::device_uvector<f_t>& get_variable_lower_bounds() override;
  const rmm::device_uvector<f_t>& get_variable_upper_bounds() const override;
  rmm::device_uvector<f_t>& get_variable_upper_bounds() override;
  const rmm::device_uvector<f_t>& get_constraint_lower_bounds() const override;
  rmm::device_uvector<f_t>& get_constraint_lower_bounds() override;
  const rmm::device_uvector<f_t>& get_constraint_upper_bounds() const override;
  rmm::device_uvector<f_t>& get_constraint_upper_bounds() override;
  const rmm::device_uvector<char>& get_row_types() const override;
  const rmm::device_uvector<var_t>& get_variable_types() const override;
  bool get_sense() const override;
  bool empty() const override;
  std::string get_objective_name() const override;
  std::string get_problem_name() const override;
  problem_category_t get_problem_category() const override;
  const std::vector<std::string>& get_variable_names() const override;
  const std::vector<std::string>& get_row_names() const override;
  const std::vector<i_t>& get_quadratic_objective_offsets() const override;
  const std::vector<i_t>& get_quadratic_objective_indices() const override;
  const std::vector<f_t>& get_quadratic_objective_values() const override;
  bool has_quadratic_objective() const override;

  // Host getters
  std::vector<f_t> get_constraint_matrix_values_host() const override;
  std::vector<i_t> get_constraint_matrix_indices_host() const override;
  std::vector<i_t> get_constraint_matrix_offsets_host() const override;
  std::vector<f_t> get_constraint_bounds_host() const override;
  std::vector<f_t> get_objective_coefficients_host() const override;
  std::vector<f_t> get_variable_lower_bounds_host() const override;
  std::vector<f_t> get_variable_upper_bounds_host() const override;
  std::vector<f_t> get_constraint_lower_bounds_host() const override;
  std::vector<f_t> get_constraint_upper_bounds_host() const override;
  std::vector<char> get_row_types_host() const override;
  std::vector<var_t> get_variable_types_host() const override;

  /**
   * @brief Convert this GPU optimization problem to an optimization_problem_t
   *        by moving GPU data ownership (zero-copy transfer).
   * @return optimization_problem_t with ownership of all GPU data
   */
  optimization_problem_t<i_t, f_t> to_optimization_problem();

  /**
   * @brief Write the optimization problem to an MPS file.
   * @param[in] mps_file_path Path to the output MPS file
   */
  void write_to_mps(const std::string& mps_file_path) override;

  /**
   * @brief Check if this problem is equivalent to another problem.
   * @param[in] other The other optimization problem to compare against
   * @return true if the problems are equivalent (up to permutation of variables/constraints)
   */
  bool is_equivalent(const optimization_problem_interface_t<i_t, f_t>& other) const override;

  // Remote execution (polymorphic dispatch)
  std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
    pdlp_solver_settings_t<i_t, f_t> const& settings,
    bool problem_checking     = true,
    bool use_pdlp_solver_mode = true) const override;

  std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
    mip_solver_settings_t<i_t, f_t> const& settings) const override;

  // C API support: Copy to host (polymorphic)
  void copy_objective_coefficients_to_host(f_t* output, i_t size) const override;
  void copy_constraint_matrix_to_host(f_t* values,
                                      i_t* indices,
                                      i_t* offsets,
                                      i_t num_values,
                                      i_t num_indices,
                                      i_t num_offsets) const override;
  void copy_row_types_to_host(char* output, i_t size) const override;
  void copy_constraint_bounds_to_host(f_t* output, i_t size) const override;
  void copy_constraint_lower_bounds_to_host(f_t* output, i_t size) const override;
  void copy_constraint_upper_bounds_to_host(f_t* output, i_t size) const override;
  void copy_variable_lower_bounds_to_host(f_t* output, i_t size) const override;
  void copy_variable_upper_bounds_to_host(f_t* output, i_t size) const override;
  void copy_variable_types_to_host(var_t* output, i_t size) const override;

  raft::handle_t const* get_handle_ptr() const noexcept;

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  problem_category_t problem_category_ = problem_category_t::LP;
  bool maximize_{false};
  i_t n_vars_{0};
  i_t n_constraints_{0};

  // GPU memory storage
  rmm::device_uvector<f_t> A_;
  rmm::device_uvector<i_t> A_indices_;
  rmm::device_uvector<i_t> A_offsets_;
  rmm::device_uvector<f_t> b_;
  rmm::device_uvector<f_t> c_;
  f_t objective_scaling_factor_{1};
  f_t objective_offset_{0};

  std::vector<i_t> Q_offsets_;
  std::vector<i_t> Q_indices_;
  std::vector<f_t> Q_values_;

  rmm::device_uvector<f_t> variable_lower_bounds_;
  rmm::device_uvector<f_t> variable_upper_bounds_;
  rmm::device_uvector<f_t> constraint_lower_bounds_;
  rmm::device_uvector<f_t> constraint_upper_bounds_;
  rmm::device_uvector<char> row_types_;
  rmm::device_uvector<var_t> variable_types_;

  std::string objective_name_;
  std::string problem_name_;
  std::vector<std::string> var_names_{};
  std::vector<std::string> row_names_{};
};

// ==============================================================================
// CPU Implementation
// ==============================================================================

/**
 * @brief CPU-based implementation of optimization_problem_interface_t.
 *
 * This implementation stores all data in CPU memory using std::vector.
 * It only implements host getters (returning std::vector references).
 * Device getters throw exceptions as GPU memory access is not supported.
 */
template <typename i_t, typename f_t>
class cpu_optimization_problem_t : public optimization_problem_interface_t<i_t, f_t> {
 public:
  explicit cpu_optimization_problem_t(raft::handle_t const* handle_ptr = nullptr);

  // Setters
  void set_maximize(bool maximize) override;
  void set_csr_constraint_matrix(const f_t* A_values,
                                 i_t size_values,
                                 const i_t* A_indices,
                                 i_t size_indices,
                                 const i_t* A_offsets,
                                 i_t size_offsets) override;
  void set_constraint_bounds(const f_t* b, i_t size) override;
  void set_objective_coefficients(const f_t* c, i_t size) override;
  void set_objective_scaling_factor(f_t objective_scaling_factor) override;
  void set_objective_offset(f_t objective_offset) override;
  void set_quadratic_objective_matrix(const f_t* Q_values,
                                      i_t size_values,
                                      const i_t* Q_indices,
                                      i_t size_indices,
                                      const i_t* Q_offsets,
                                      i_t size_offsets,
                                      bool validate_positive_semi_definite = false) override;
  void set_variable_lower_bounds(const f_t* variable_lower_bounds, i_t size) override;
  void set_variable_upper_bounds(const f_t* variable_upper_bounds, i_t size) override;
  void set_variable_types(const var_t* variable_types, i_t size) override;
  void set_problem_category(const problem_category_t& category) override;
  void set_constraint_lower_bounds(const f_t* constraint_lower_bounds, i_t size) override;
  void set_constraint_upper_bounds(const f_t* constraint_upper_bounds, i_t size) override;
  void set_row_types(const char* row_types, i_t size) override;
  void set_objective_name(const std::string& objective_name) override;
  void set_problem_name(const std::string& problem_name) override;
  void set_variable_names(const std::vector<std::string>& variable_names) override;
  void set_row_names(const std::vector<std::string>& row_names) override;

  // Scalar and metadata getters (implemented for CPU; do not throw)
  i_t get_n_variables() const override;
  i_t get_n_constraints() const override;
  i_t get_nnz() const override;
  i_t get_n_integers() const override;
  f_t get_objective_scaling_factor() const override;
  f_t get_objective_offset() const override;
  bool get_sense() const override;
  bool empty() const override;
  std::string get_objective_name() const override;
  std::string get_problem_name() const override;
  problem_category_t get_problem_category() const override;
  const std::vector<std::string>& get_variable_names() const override;
  const std::vector<std::string>& get_row_names() const override;
  const std::vector<i_t>& get_quadratic_objective_offsets() const override;
  const std::vector<i_t>& get_quadratic_objective_indices() const override;
  const std::vector<f_t>& get_quadratic_objective_values() const override;
  bool has_quadratic_objective() const override;

  // Device memory accessors - throw on CPU implementation (use host getters below instead)
  const rmm::device_uvector<f_t>& get_constraint_matrix_values() const override;
  rmm::device_uvector<f_t>& get_constraint_matrix_values() override;
  const rmm::device_uvector<i_t>& get_constraint_matrix_indices() const override;
  rmm::device_uvector<i_t>& get_constraint_matrix_indices() override;
  const rmm::device_uvector<i_t>& get_constraint_matrix_offsets() const override;
  rmm::device_uvector<i_t>& get_constraint_matrix_offsets() override;
  const rmm::device_uvector<f_t>& get_constraint_bounds() const override;
  rmm::device_uvector<f_t>& get_constraint_bounds() override;
  const rmm::device_uvector<f_t>& get_objective_coefficients() const override;
  rmm::device_uvector<f_t>& get_objective_coefficients() override;
  const rmm::device_uvector<f_t>& get_variable_lower_bounds() const override;
  rmm::device_uvector<f_t>& get_variable_lower_bounds() override;
  const rmm::device_uvector<f_t>& get_variable_upper_bounds() const override;
  rmm::device_uvector<f_t>& get_variable_upper_bounds() override;
  const rmm::device_uvector<f_t>& get_constraint_lower_bounds() const override;
  rmm::device_uvector<f_t>& get_constraint_lower_bounds() override;
  const rmm::device_uvector<f_t>& get_constraint_upper_bounds() const override;
  rmm::device_uvector<f_t>& get_constraint_upper_bounds() override;
  const rmm::device_uvector<char>& get_row_types() const override;
  const rmm::device_uvector<var_t>& get_variable_types() const override;

  // Host getters - these are the only supported getters for CPU implementation
  std::vector<f_t> get_constraint_matrix_values_host() const override;
  std::vector<i_t> get_constraint_matrix_indices_host() const override;
  std::vector<i_t> get_constraint_matrix_offsets_host() const override;
  std::vector<f_t> get_constraint_bounds_host() const override;
  std::vector<f_t> get_objective_coefficients_host() const override;
  std::vector<f_t> get_variable_lower_bounds_host() const override;
  std::vector<f_t> get_variable_upper_bounds_host() const override;
  std::vector<f_t> get_constraint_lower_bounds_host() const override;
  std::vector<f_t> get_constraint_upper_bounds_host() const override;
  std::vector<char> get_row_types_host() const override;
  std::vector<var_t> get_variable_types_host() const override;

  /**
   * @brief Set the CUDA handle for GPU operations
   *
   * This is used in test mode when a CPU problem needs to be converted to GPU
   * for local solving. The handle must be set before calling to_optimization_problem().
   *
   * @param handle_ptr Pointer to the RAFT handle with CUDA resources
   */
  void set_handle(raft::handle_t const* handle_ptr) { handle_ptr_ = handle_ptr; }

  /**
   * @brief Convert this CPU optimization problem to an optimization_problem_t
   *        by copying CPU data to GPU (requires GPU memory transfer).
   * @return optimization_problem_t with all data copied to GPU
   */
  optimization_problem_t<i_t, f_t> to_optimization_problem();

  /**
   * @brief Write the optimization problem to an MPS file.
   * @param[in] mps_file_path Path to the output MPS file
   */
  void write_to_mps(const std::string& mps_file_path) override;

  /**
   * @brief Check if this problem is equivalent to another problem.
   * @param[in] other The other optimization problem to compare against
   * @return true if the problems are equivalent (up to permutation of variables/constraints)
   */
  bool is_equivalent(const optimization_problem_interface_t<i_t, f_t>& other) const override;

  // Remote execution (polymorphic dispatch)
  std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
    pdlp_solver_settings_t<i_t, f_t> const& settings,
    bool problem_checking     = true,
    bool use_pdlp_solver_mode = true) const override;

  std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
    mip_solver_settings_t<i_t, f_t> const& settings) const override;

  // C API support: Copy to host (polymorphic)
  void copy_objective_coefficients_to_host(f_t* output, i_t size) const override;
  void copy_constraint_matrix_to_host(f_t* values,
                                      i_t* indices,
                                      i_t* offsets,
                                      i_t num_values,
                                      i_t num_indices,
                                      i_t num_offsets) const override;
  void copy_row_types_to_host(char* output, i_t size) const override;
  void copy_constraint_bounds_to_host(f_t* output, i_t size) const override;
  void copy_constraint_lower_bounds_to_host(f_t* output, i_t size) const override;
  void copy_constraint_upper_bounds_to_host(f_t* output, i_t size) const override;
  void copy_variable_lower_bounds_to_host(f_t* output, i_t size) const override;
  void copy_variable_upper_bounds_to_host(f_t* output, i_t size) const override;
  void copy_variable_types_to_host(var_t* output, i_t size) const override;

 private:
  raft::handle_t const* handle_ptr_{nullptr};

  problem_category_t problem_category_ = problem_category_t::LP;
  bool maximize_{false};
  i_t n_vars_{0};
  i_t n_constraints_{0};

  // CPU memory storage
  std::vector<f_t> A_;
  std::vector<i_t> A_indices_;
  std::vector<i_t> A_offsets_;
  std::vector<f_t> b_;
  std::vector<f_t> c_;
  f_t objective_scaling_factor_{1};
  f_t objective_offset_{0};

  std::vector<i_t> Q_offsets_;
  std::vector<i_t> Q_indices_;
  std::vector<f_t> Q_values_;

  std::vector<f_t> variable_lower_bounds_;
  std::vector<f_t> variable_upper_bounds_;
  std::vector<f_t> constraint_lower_bounds_;
  std::vector<f_t> constraint_upper_bounds_;
  std::vector<char> row_types_;
  std::vector<var_t> variable_types_;

  std::string objective_name_;
  std::string problem_name_;
  std::vector<std::string> var_names_{};
  std::vector<std::string> row_names_{};
};

}  // namespace cuopt::linear_programming
