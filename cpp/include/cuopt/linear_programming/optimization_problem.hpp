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

  // ============================================================================
  // Move-based setters (zero-copy, transfers ownership)
  // ============================================================================

  /**
   * @brief Move constraint matrix data without copying (transfers ownership).
   * @note This is a zero-copy operation that just moves device pointers.
   * @param[in] A_values rvalue reference to constraint matrix values
   * @param[in] A_indices rvalue reference to constraint matrix column indices
   * @param[in] A_offsets rvalue reference to constraint matrix row offsets
   */
  void set_csr_constraint_matrix_move(rmm::device_uvector<f_t>&& A_values,
                                      rmm::device_uvector<i_t>&& A_indices,
                                      rmm::device_uvector<i_t>&& A_offsets);

  /**
   * @brief Move constraint bounds without copying (transfers ownership).
   * @param[in] b rvalue reference to constraint bounds vector
   */
  void set_constraint_bounds_move(rmm::device_uvector<f_t>&& b);

  /**
   * @brief Move objective coefficients without copying (transfers ownership).
   * @param[in] c rvalue reference to objective coefficients vector
   */
  void set_objective_coefficients_move(rmm::device_uvector<f_t>&& c);

  /**
   * @brief Move variable lower bounds without copying (transfers ownership).
   * @param[in] variable_lower_bounds rvalue reference to lower bounds vector
   */
  void set_variable_lower_bounds_move(rmm::device_uvector<f_t>&& variable_lower_bounds);

  /**
   * @brief Move variable upper bounds without copying (transfers ownership).
   * @param[in] variable_upper_bounds rvalue reference to upper bounds vector
   */
  void set_variable_upper_bounds_move(rmm::device_uvector<f_t>&& variable_upper_bounds);

  /**
   * @brief Move variable types without copying (transfers ownership).
   * @param[in] variable_types rvalue reference to variable types vector
   */
  void set_variable_types_move(rmm::device_uvector<var_t>&& variable_types);

  /**
   * @brief Move constraint lower bounds without copying (transfers ownership).
   * @param[in] constraint_lower_bounds rvalue reference to lower bounds vector
   */
  void set_constraint_lower_bounds_move(rmm::device_uvector<f_t>&& constraint_lower_bounds);

  /**
   * @brief Move constraint upper bounds without copying (transfers ownership).
   * @param[in] constraint_upper_bounds rvalue reference to upper bounds vector
   */
  void set_constraint_upper_bounds_move(rmm::device_uvector<f_t>&& constraint_upper_bounds);

  /**
   * @brief Move row types without copying (transfers ownership).
   * @param[in] row_types rvalue reference to row types vector
   */
  void set_row_types_move(rmm::device_uvector<char>&& row_types);

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
  // Remote Execution (Polymorphic Dispatch)
  // ============================================================================

  std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
    pdlp_solver_settings_t<i_t, f_t> const& settings,
    bool problem_checking     = true,
    bool use_pdlp_solver_mode = true) const override;

  std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
    mip_solver_settings_t<i_t, f_t> const& settings) const override;

  // ============================================================================
  // Conversion
  // ============================================================================

  /**
   * @brief Returns nullptr since this is already a GPU problem.
   * @return nullptr
   */
  std::unique_ptr<optimization_problem_t<i_t, f_t>> to_optimization_problem() override;

  /**
   * @brief Creates a new CPU problem by copying data from GPU to host memory.
   * @return unique_ptr to new cpu_optimization_problem_t with copied data
   */
  std::unique_ptr<cpu_optimization_problem_t<i_t, f_t>> to_cpu_optimization_problem()
    const override;

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
