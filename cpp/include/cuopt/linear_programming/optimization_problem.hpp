/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <raft/core/device_span.hpp>
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
 * @brief A representation of a linear programming (LP) optimization problem
 *
 * @tparam i_t  Integer type for indices
 * @tparam f_t  Data type of the variables and their weights in the equations
 *
 * This implementation stores all data in GPU memory using rmm::device_uvector.
 * It implements both device getters (returning rmm::device_uvector references)
 * and host getters (returning std::vector by copying from GPU to CPU).
 *
 * This structure stores all the information necessary to represent the
 * following LP:
 *
 * <pre>
 * Minimize:
 *   dot(c, x)
 * Subject to:
 *   matmul(A, x) (= or >= or)<= b
 * Where:
 *   x = n-dim vector
 *   A = mxn-dim sparse matrix
 *   n = number of variables
 *   m = number of constraints
 *
 * </pre>
 *
 * @note: By default this assumes objective minimization.
 *
 * Objective value can be scaled and offset accordingly:
 * objective_scaling_factor * (dot(c, x) + objective_offset)
 * please refer to the `set_objective_scaling_factor()` and
 * `set_objective_offset()` methods.
 */
template <typename i_t, typename f_t>
class optimization_problem_t : public optimization_problem_interface_t<i_t, f_t> {
 public:
  static_assert(std::is_integral<i_t>::value,
                "'optimization_problem_t' accepts only integer types for indexes");
  static_assert(std::is_floating_point<f_t>::value,
                "'optimization_problem_t' accepts only floating point types for weights");

  /**
   * @brief A device-side view of the `optimization_problem_t` structure with
   * the RAII stuffs stripped out, to make it easy to work inside kernels
   *
   * @note It is assumed that the pointers are NOT owned by this class, but
   * rather by the encompassing `optimization_problem_t` class via RAII
   * abstractions like `rmm::device_uvector`
   */
  struct view_t {
    /** number of variables */
    i_t n_vars;
    /** number of constraints in the LP representation */
    i_t n_constraints;
    /** number of non-zero elements in the constraint matrix */
    i_t nnz;
    /**
     * constraint matrix in the CSR format
     * @{
     */
    raft::device_span<f_t> A;
    raft::device_span<const i_t> A_indices;
    raft::device_span<const i_t> A_offsets;
    /** @} */
    /** RHS of the constraints */
    raft::device_span<const f_t> b;
    /** array of weights used in the objective function */
    raft::device_span<const f_t> c;
    /** array of lower bounds for the variables */
    raft::device_span<const f_t> variable_lower_bounds;
    /** array of upper bounds for the variables */
    raft::device_span<const f_t> variable_upper_bounds;
    /** variable types */
    raft::device_span<const var_t> variable_types;
    /** array of lower bounds for the constraint */
    raft::device_span<const f_t> constraint_lower_bounds;
    /** array of upper bounds for the constraint */
    raft::device_span<const f_t> constraint_upper_bounds;
  };  // struct view_t

  explicit optimization_problem_t(raft::handle_t const* handle_ptr);
  optimization_problem_t(const optimization_problem_t<i_t, f_t>& other);
  optimization_problem_t(optimization_problem_t<i_t, f_t>&&)            = default;
  optimization_problem_t& operator=(optimization_problem_t<i_t, f_t>&&) = default;

  std::vector<internals::base_solution_callback_t*> mip_callbacks_;

  // ============================================================================
  // Setters
  // ============================================================================

  /**
   * @brief Set the sense of optimization to maximize.
   * @note Setting before calling the solver is optional, default is false (minimize).
   * @param[in] maximize true means to maximize the objective function, else minimize.
   */
  void set_maximize(bool maximize) override;

  /**
   * @brief Set the constraint matrix (A) in CSR format.
   * @note Setting before calling the solver is mandatory.
   * Data is copied to GPU memory on the stream of the RAFT handle passed to the problem.
   * @param[in] A_values Values of the CSR representation (device or host pointer)
   * @param size_values Size of the A_values array
   * @param[in] A_indices Indices of the CSR representation (device or host pointer)
   * @param size_indices Size of the A_indices array
   * @param[in] A_offsets Offsets of the CSR representation (device or host pointer)
   * @param size_offsets Size of the A_offsets array
   */
  void set_csr_constraint_matrix(const f_t* A_values,
                                 i_t size_values,
                                 const i_t* A_indices,
                                 i_t size_indices,
                                 const i_t* A_offsets,
                                 i_t size_offsets) override;

  /**
   * @brief Set the constraint bounds (b / right-hand side) array.
   * @note Setting before calling the solver is mandatory.
   * @param[in] b Device or host memory pointer to a floating point array of size size.
   * @param size Size of the b array.
   */
  void set_constraint_bounds(const f_t* b, i_t size) override;

  /**
   * @brief Set the objective coefficients (c) array.
   * @note Setting before calling the solver is mandatory.
   * @param[in] c Device or host memory pointer to a floating point array of size size.
   * @param size Size of the c array.
   */
  void set_objective_coefficients(const f_t* c, i_t size) override;

  /**
   * @brief Set the scaling factor of the objective function (scaling_factor * objective_value).
   * @note Setting before calling the solver is optional, default value is 1.
   * @param objective_scaling_factor Objective scaling factor value.
   */
  void set_objective_scaling_factor(f_t objective_scaling_factor) override;

  /**
   * @brief Set the offset of the objective function (objective_offset + objective_value).
   * @note Setting before calling the solver is optional, default value is 0.
   * @param objective_offset Objective offset value.
   */
  void set_objective_offset(f_t objective_offset) override;

  /**
   * @brief Set the quadratic objective matrix (Q) in CSR format.
   * @note Used for quadratic programming: objective is x^T * Q * x + c^T * x
   * @param[in] Q_values Values of the CSR representation
   * @param size_values Size of the Q_values array
   * @param[in] Q_indices Indices of the CSR representation
   * @param size_indices Size of the Q_indices array
   * @param[in] Q_offsets Offsets of the CSR representation
   * @param size_offsets Size of the Q_offsets array
   * @param validate_positive_semi_definite Whether to validate PSD property
   */
  void set_quadratic_objective_matrix(const f_t* Q_values,
                                      i_t size_values,
                                      const i_t* Q_indices,
                                      i_t size_indices,
                                      const i_t* Q_offsets,
                                      i_t size_offsets,
                                      bool validate_positive_semi_definite = false) override;

  /** @copydoc optimization_problem_interface_t::set_variable_lower_bounds */
  void set_variable_lower_bounds(const f_t* variable_lower_bounds, i_t size) override;
  /** @copydoc optimization_problem_interface_t::set_variable_upper_bounds */
  void set_variable_upper_bounds(const f_t* variable_upper_bounds, i_t size) override;
  /** @copydoc optimization_problem_interface_t::set_variable_types */
  void set_variable_types(const var_t* variable_types, i_t size) override;
  /** @copydoc optimization_problem_interface_t::set_problem_category */
  void set_problem_category(const problem_category_t& category) override;
  /** @copydoc optimization_problem_interface_t::set_constraint_lower_bounds */
  void set_constraint_lower_bounds(const f_t* constraint_lower_bounds, i_t size) override;
  /** @copydoc optimization_problem_interface_t::set_constraint_upper_bounds */
  void set_constraint_upper_bounds(const f_t* constraint_upper_bounds, i_t size) override;
  /** @copydoc optimization_problem_interface_t::set_row_types */
  void set_row_types(const char* row_types, i_t size) override;
  /** @copydoc optimization_problem_interface_t::set_objective_name */
  void set_objective_name(const std::string& objective_name) override;
  /** @copydoc optimization_problem_interface_t::set_problem_name */
  void set_problem_name(const std::string& problem_name) override;
  /** @copydoc optimization_problem_interface_t::set_variable_names */
  void set_variable_names(const std::vector<std::string>& variable_names) override;
  /** @copydoc optimization_problem_interface_t::set_row_names */
  void set_row_names(const std::vector<std::string>& row_names) override;

  // ============================================================================
  // Device getters
  // ============================================================================

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

  // ============================================================================
  // Host getters
  // ============================================================================

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

  // ============================================================================
  // File I/O
  // ============================================================================

  /**
   * @brief Write the optimization problem to an MPS file.
   * @param[in] mps_file_path Path to the output MPS file
   */
  void write_to_mps(const std::string& mps_file_path) override;

  /* Print scaling information */
  void print_scaling_information() const;

  // ============================================================================
  // Comparison
  // ============================================================================

  /**
   * @brief Check if this problem is equivalent to another optimization_problem_t.
   * @param[in] other The other optimization problem to compare against
   * @return true if the problems are equivalent (up to permutation of variables/constraints)
   */
  bool is_equivalent(const optimization_problem_t<i_t, f_t>& other) const;

  /**
   * @brief Check if this problem is equivalent to another problem (via interface).
   * @param[in] other The other optimization problem to compare against
   * @return true if the problems are equivalent (up to permutation of variables/constraints)
   */
  bool is_equivalent(const optimization_problem_interface_t<i_t, f_t>& other) const override;

  // ============================================================================
  // Conversion
  // ============================================================================

  /**
   * @brief Returns nullptr since this is already a GPU problem.
   * @return nullptr
   */
  std::unique_ptr<optimization_problem_t<i_t, f_t>> to_optimization_problem(
    raft::handle_t const* handle_ptr = nullptr) override;

  // ============================================================================
  // C API support: Copy to host (polymorphic)
  // ============================================================================

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

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view() const;

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

}  // namespace cuopt::linear_programming
