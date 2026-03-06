/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem_interface.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

// Forward declarations
template <typename i_t, typename f_t>
class optimization_problem_t;
template <typename i_t, typename f_t>
class pdlp_solver_settings_t;
template <typename i_t, typename f_t>
class mip_solver_settings_t;
template <typename i_t, typename f_t>
class lp_solution_interface_t;
template <typename i_t, typename f_t>
class mip_solution_interface_t;

/**
 * @brief CPU-based implementation of optimization_problem_interface_t.
 *
 * This implementation stores all data in CPU memory using std::vector.
 * It only implements host getters (returning std::vector by value).
 * Device getters throw exceptions as GPU memory access is not supported.
 */
template <typename i_t, typename f_t>
class cpu_optimization_problem_t : public optimization_problem_interface_t<i_t, f_t> {
 public:
  cpu_optimization_problem_t();

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

  // Device getters - throw exceptions (not supported for CPU implementation)
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
   * @brief Convert this CPU optimization problem to an optimization_problem_t
   *        by copying CPU data to GPU (requires GPU memory transfer).
   *
   * @param handle_ptr RAFT handle with CUDA resources for GPU memory allocation.
   * @return unique_ptr to new optimization_problem_t with all data copied to GPU
   * @throws std::runtime_error if handle_ptr is null
   */
  std::unique_ptr<optimization_problem_t<i_t, f_t>> to_optimization_problem(
    raft::handle_t const* handle_ptr = nullptr) override;

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
