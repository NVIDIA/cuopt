/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <mps_parser/utilities/span.hpp>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace cuopt::mps_parser {

/**
 * @brief A representation of a linear programming (LP) optimization problem
 *
 * @tparam f_t  Data type of the variables and their weights in the equations
 *
 * A linear programming optimization problem is defined as follows:
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
 * please refeto to the `set_objective_scaling_factor()` and `set_objective_offset()` method.
 */
template <typename i_t, typename f_t>
class data_model_view_t {
 public:
  static_assert(std::is_integral<i_t>::value,
                "'data_model_view_t' accepts only integer types for indexes");
  static_assert(std::is_floating_point<f_t>::value,
                "'data_model_view_t' accepts only floating point types for weights");

  /**
   * @brief Set the sense of optimization to maximize.
   * @note Setting before calling the solver is optional, default value if false (minimize).
   *
   * @param[in] maximize true means to maximize the objective function, else minimize.
   */
  void set_maximize(bool maximize);

  /**
   * @brief Set the constraint matrix (A) in CSR format. For more information about CSR checkout:
   * https://docs.nvidia.com/cuda/cusparse/index.html#compressed-sparse-row-csr

   * @note Setting before calling the solver is mandatory.
   *
   * @throws cuopt::logic_error when an error occurs.
   *
   * @param[in] A_values Values of the CSR representation of the constraint matrix as a device
   memory pointer to a floating point array of size size_values.
   * cuOpt does not own or copy this data.
   * @param size_values Size of the A_values array.
   * @param[in] A_indices Indices of the CSR representation of the constraint matrix as a device
   memory pointer to an integer array of size size_indices.
   * cuOpt does not own or copy this data.
   * @param size_indices Size of the A_indices array.
   * @param[in] A_offsets Offsets of the CSR representation of the constraint matrix as a device
   memory pointer to a integer array of size size_offsets.
   * cuOpt does not own or copy this data.
   * @param size_offsets Size of the A_offsets array.
   */
  void set_csr_constraint_matrix(const f_t* A_values,
                                 i_t size_values,
                                 const i_t* A_indices,
                                 i_t size_indices,
                                 const i_t* A_offsets,
                                 i_t size_offsets);

  /**
   * @brief Set the constraint bounds (b / right-hand side) array.
   * @note Setting before calling the solver is mandatory.
   *
   * @param[in] b Device memory pointer to a floating point array of size size.
   * cuOpt does not own or copy this data.
   * @param size Size of the b array.
   */
  void set_constraint_bounds(const f_t* b, i_t size);
  /**
   * @brief Set the objective coefficients (c) array.
   * @note Setting before calling the solver is mandatory.
   *
   * @param[in] c Device memory pointer to a floating point array of size size.
   * cuOpt does not own or copy this data.
   * @param size Size of the c array.
   */
  void set_objective_coefficients(const f_t* c, i_t size);
  /**
   * @brief Set the scaling factor of the objective function (scaling_factor * objective_value).
   * @note Setting before calling the solver is optional, default value if 1.
   *
   * @param objective_scaling_factor Objective scaling factor value.
   */
  void set_objective_scaling_factor(f_t objective_scaling_factor);
  /**
   * @brief Set the offset of the objective function (objective_offset + objective_value).
   * @note Setting before calling the solver is optional, default value if 0.
   *
   * @param objective_offset Objective offset value.
   */
  void set_objective_offset(f_t objective_offset);
  /**
   * @brief Set the variables (x) lower bounds.
   * @note Setting before calling the solver is optional, default value for all is 0.
   *
   * @param[in] variable_lower_bounds Device memory pointer to a floating point array of size size.
   * cuOpt does not own or copy this data.
   * @param size Size of the variable_lower_bounds array
   */
  void set_variable_lower_bounds(const f_t* variable_lower_bounds, i_t size);
  /**
   * @brief Set the variables (x) upper bounds.
   * @note Setting before calling the solver is optional, default value for all is +infinity.
   *
   *
   * @param[in] variable_upper_bounds Device memory pointer to a floating point array of size size.
   * cuOpt does not own or copy this data.
   * @param size Size of the variable_upper_bounds array.
   */
  void set_variable_upper_bounds(const f_t* variable_upper_bounds, i_t size);
  /**
   * @brief Set the variables (x) types.
   * @note Setting before calling the solver is optional, default value for all is 'C' meaning
   * continuous.
   *
   *
   * @param[in] variable_types Device memory pointer to a char array of size size. Can be 'C' or
   * 'I'. cuOpt does not own or copy this data.
   * @param size Size of the variable_types array.
   */
  void set_variable_types(const char* variable_types, i_t size);
  /**
   * @brief Set the type of each row (constraint). Possible values are:
   * 'E' for equality ( = ),
   * 'L' for less-than ( <= )
   * 'G' for greater-than ( >= ),
   * 'N' for non-constraining rows (objective)
   * @note Setting before calling the solver is optional if you set the constraint lower and upper
   * bounds, else it's mandatory
   *
   * @param[in] row_types Device memory pointer to a character array of size size.
   * cuOpt does not own or copy this data.
   * @param size Size of the row_types array
   */
  void set_row_types(const char* row_types, i_t size);
  /**
   * @brief Set the name of the objective function.
   * @note Setting before calling the solver is optional. Value is only used for file generation of
   * the solution.
   *
   * @param[in] objective_name Objective name value.
   */
  void set_objective_name(const std::string& objective_name);
  /**
   * @brief Set the problem name.
   * @note Setting before calling the solver is optional.
   *
   * @param[in] problem_name Problem name value.
   */
  void set_problem_name(const std::string& problem_name);
  /**
   * @brief Set the constraints lower bounds.
   * @note Setting before calling the solver is optional if you set the row type, else it's
   * mandatory along with the upper bounds.
   *
   * @param[in] constraint_lower_bounds Device memory pointer to a floating point array of size
   * size.
   * cuOpt does not own or copy this data.
   * @param size Size of the row_types array
   */
  void set_constraint_lower_bounds(const f_t* constraint_lower_bounds, i_t size);
  /**
   * @brief Set the constraints upper bounds.
   * @note Setting before calling the solver is optional if you set the row type, else it's
   * mandatory along with the lower bounds.
   *
   * @param[in] constraint_upper_bounds Device memory pointer to a floating point array of size
   * size.
   * cuOpt does not own or copy this data.
   * @param size Size of the row_types array
   */
  void set_constraint_upper_bounds(const f_t* constraint_upper_bounds, i_t size);
  /**
   * @brief Set an initial primal solution.
   * @note Setting before calling the solver is optional, default value is all 0.
   *
   * @param[in] initial_primal_solution Device memory pointer to a floating point array of size
   * size.
   * cuOpt does not own or copy this data.
   * @param size Size of the initial_primal_solution array.
   */
  void set_initial_primal_solution(const f_t* initial_primal_solution, i_t size);
  /**
   * @brief Set an initial dual solution.
   * @note Setting before calling the solver is optional, default value is all 0.
   *
   * @param[in] initial_dual_solution Device memory pointer to a floating point array of size
   * size.
   * cuOpt does not own or copy this data.
   * @param size Size of the initial_dual_solution array.
   */
  void set_initial_dual_solution(const f_t* initial_dual_solution, i_t size);

  /**
   * @brief Set the quadratic objective matrix (Q) in CSR format for QPS files.
   *
   * @note This is used for quadratic programming problems where the objective
   * function contains quadratic terms: (1/2) * x^T * Q * x + c^T * x
   * cuOpt does not own or copy this data.
   *
   * @param[in] Q_values Device memory pointer to values of the CSR representation of the quadratic
   * objective matrix
   * @param size_values Size of the Q_values array
   * @param[in] Q_indices Device memory pointer to indices of the CSR representation of the
   * quadratic objective matrix
   * @param size_indices Size of the Q_indices array
   * @param[in] Q_offsets Device memory pointer to offsets of the CSR representation of the
   * quadratic objective matrix
   * @param size_offsets Size of the Q_offsets array
   */
  void set_quadratic_objective_matrix(const f_t* Q_values,
                                      i_t size_values,
                                      const i_t* Q_indices,
                                      i_t size_indices,
                                      const i_t* Q_offsets,
                                      i_t size_offsets);

  /**
   * @brief Get the sense value (false:minimize, true:maximize)
   *
   * @return Sense value
   */
  bool get_sense() const noexcept;
  /**
   * @brief Get the CSR constraint matrix values
   *
   * @return span<f_t const>
   */
  span<f_t const> get_constraint_matrix_values() const noexcept;
  /**
   * @brief Get the CSR constraint matrix indices
   *
   * @return span<i_t const>
   */
  span<i_t const> get_constraint_matrix_indices() const noexcept;
  /**
   * @brief Get the CSR constraint matrix offsets
   *
   * @return span<i_t const>
   */
  span<i_t const> get_constraint_matrix_offsets() const noexcept;
  /**
   * @brief Get the b (right-hand side) constraints array
   *
   * @return span<f_t const>
   */
  span<f_t const> get_constraint_bounds() const noexcept;
  /**
   * @brief Get the c vector (weights of each x variable).
   *
   * @return span<f_t const>
   */
  span<f_t const> get_objective_coefficients() const noexcept;
  /**
   * @brief Get the objective scaling factor
   *
   * @return Objective scaling factor value
   */
  f_t get_objective_scaling_factor() const noexcept;
  /**
   * @brief Get the objective offset
   *
   * @return Objective offset value
   */
  f_t get_objective_offset() const noexcept;
  /**
   * @brief Get the variables (x) lower bounds
   *
   * @return span<f_t const>
   */
  span<f_t const> get_variable_lower_bounds() const noexcept;
  /**
   * @brief Get the variables (x) upper bounds
   *
   * @return span<f_t const>
   */
  span<f_t const> get_variable_upper_bounds() const noexcept;
  /**
   * @brief Get the variables (x) types
   *
   * @return span<char const>
   */
  span<char const> get_variable_types() const noexcept;
  /**
   * @brief Get the row types
   *
   * @return span<char const>
   */
  span<char const> get_row_types() const noexcept;
  /**
   * @brief Get the constraints lower bounds
   *
   * @return span<f_t const>
   */
  span<f_t const> get_constraint_lower_bounds() const noexcept;
  /**
   * @brief Get the constraints upper bounds
   *
   * @return span<f_t const>
   */
  span<f_t const> get_constraint_upper_bounds() const noexcept;
  /**
   * @brief Get the initial primal solution
   *
   * @return span<f_t const>
   */
  span<f_t const> get_initial_primal_solution() const noexcept;
  /**
   * @brief Get the initial dual solution
   *
   * @return span<f_t const>
   */
  span<f_t const> get_initial_dual_solution() const noexcept;

  /**
   * @brief Get the problem name
   *
   * @return std::string
   */
  std::string get_problem_name() const noexcept;
  /**
   * @brief Get the objective name
   *
   * @return std::string
   */
  std::string get_objective_name() const noexcept;

  // QPS-specific getters
  /**
   * @brief Get the quadratic objective matrix values
   *
   * @return span<f_t const>
   */
  span<f_t const> get_quadratic_objective_values() const noexcept;
  /**
   * @brief Get the quadratic objective matrix indices
   *
   * @return span<i_t const>
   */
  span<i_t const> get_quadratic_objective_indices() const noexcept;
  /**
   * @brief Get the quadratic objective matrix offsets
   *
   * @return span<i_t const>
   */
  span<i_t const> get_quadratic_objective_offsets() const noexcept;
  /**
   * @brief Check if the problem has quadratic objective terms
   *
   * @return bool
   */
  bool has_quadratic_objective() const noexcept;

 private:
  bool maximize_{false};
  span<f_t const> A_;
  span<i_t const> A_indices_;
  span<i_t const> A_offsets_;
  span<f_t const> b_;
  span<f_t const> c_;
  f_t objective_scaling_factor_{1};
  f_t objective_offset_{0};
  span<f_t const> variable_lower_bounds_;
  span<f_t const> variable_upper_bounds_;
  span<char const> variable_types_;
  span<char const> row_types_;
  std::string objective_name_;
  std::string problem_name_;
  span<f_t const> constraint_lower_bounds_;
  span<f_t const> constraint_upper_bounds_;

  // TODO move to solver_settings in next release
  span<f_t const> initial_primal_solution_;
  span<f_t const> initial_dual_solution_;

  // QPS-specific data members for quadratic programming support
  span<f_t const> Q_objective_;
  span<i_t const> Q_objective_indices_;
  span<i_t const> Q_objective_offsets_;

};  // class data_model_view_t

}  // namespace cuopt::mps_parser
