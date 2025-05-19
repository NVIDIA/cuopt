/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
 * please refer to the `set_objective_scaling_factor()` and `set_objective_offset()` methods.
 */
template <typename i_t, typename f_t>
class mps_data_model_t {
 public:
  static_assert(std::is_integral<i_t>::value,
                "'mps_data_model_t' accepts only integer types for indexes");
  static_assert(std::is_floating_point<f_t>::value,
                "'mps_data_model_t' accepts only floating point types for weights");

  mps_data_model_t() = default;

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
   * @throws std::logic_error when an error occurs.
   * @param[in] A_values Values of the CSR representation of the constraint matrix as a host memory
   pointer to a floating point array of size size_values.
   * MPS Parser copies this data.
   * @param size_values Size of the A_values array.
   * @param[in] A_indices Indices of the CSR representation of the constraint matrix as a host
   memory pointer to an integer array of size size_indices.
   * MPS Parser copies this data.
   * @param size_indices Size of the A_indices array.
   * @param[in] A_offsets Offsets of the CSR representation of the constraint matrix as a host
   memory pointer to a integer array of size size_offsets.
   * MPS Parser copies this data.
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
   * @param[in] b Host memory pointer to a floating point array of size size.
   * MPS Parser copies this data.
   * @param size Size of the b array.
   */
  void set_constraint_bounds(const f_t* b, i_t size);
  /**
   * @brief Set the objective coefficients (c) array.
   * @note Setting before calling the solver is mandatory.
   *
   * @param[in] c Host memory pointer to a floating point array of size size.
   * MPS Parser copies this data.
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
   * @param[in] variable_lower_bounds Host memory pointer to a floating point array of
   * size size.
   * MPS Parser copies this data.
   * @param size Size of the variable_lower_bounds array
   */
  void set_variable_lower_bounds(const f_t* variable_lower_bounds, i_t size);
  /**
   * @brief Set the variables (x) upper bounds.
   * @note Setting before calling the solver is optional, default value for all is +infinity.
   *
   * @param[in] variable_upper_bounds Host memory pointer to a floating point array of
   * size size.
   * MPS Parser copies this data.
   * @param size Size of the variable_upper_bounds array.
   */
  void set_variable_upper_bounds(const f_t* variable_upper_bounds, i_t size);
  /**
   * @brief Set the constraints lower bounds.
   * @note Setting before calling the solver is optional if you set the row type, else it's
   * mandatory along with the upper bounds.
   *
   * @param[in] constraint_lower_bounds Host memory pointer to a floating point array of
   * size size.
   * MPS Parser copies this data.
   * @param size Size of the constraint_lower_bounds array
   */
  void set_constraint_lower_bounds(const f_t* constraint_lower_bounds, i_t size);
  /**
   * @brief Set the constraints upper bounds.
   * @note Setting before calling the solver is optional if you set the row type, else it's
   * mandatory along with the lower bounds.
   * If both are set, priority goes to set_constraints.
   *
   * @param[in] constraint_upper_bounds Host memory pointer to a floating point array of
   * size size.
   * MPS Parser copies this data.
   * @param size Size of the constraint_upper_bounds array
   */
  void set_constraint_upper_bounds(const f_t* constraint_upper_bounds, i_t size);

  /**
   * @brief Set the type of each row (constraint). Possible values are:
   * 'E' for equality ( = ): lower & upper constrains bound equal to b
   * 'L' for less-than ( <= ): lower constrains bound equal to -infinity, upper constrains bound
   * equal to b
   * 'G' for greater-than ( >= ): lower constrains bound equal to b, upper constrains
   * bound equal to +infinity
   * @note Setting before calling the solver is optional if you set the constraint lower and upper
   * bounds, else it's mandatory
   * If both are set, priority goes to set_constraints.
   *
   * @param[in] row_types Host memory pointer to a character array of
   * size size.
   * MPS Parser copies this data.
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
   * @brief Set the variables names.
   * @note Setting before calling the solver is optional. Value is only used for file generation of
   * the solution.
   *
   * @param[in] variable_names Variable names values.
   */
  void set_variable_names(const std::vector<std::string>& variables_names);
  /**
   * @brief Set the variables types.
   * @note Setting before calling the solver is optional. Value is only used for file generation of
   * the solution.
   *
   * @param[in] variable_types Variable type values.
   */
  void set_variable_types(const std::vector<char>& variables_types);
  /**
   * @brief Set the row names.
   * @note Setting before calling the solver is optional. Value is only used for file generation of
   * the solution.
   *
   * @param[in] row_names Row names value.
   */
  void set_row_names(const std::vector<std::string>& row_names);

  /**
   * @brief Set an initial primal solution.
   *
   * @note Default value is all 0.
   *
   * @param[in] initial_primal_solution Host memory pointer to a floating point array of
   * size size.
   * MPS Parser copies this data.
   * @param size Size of the initial_primal_solution array.
   */
  void set_initial_primal_solution(const f_t* initial_primal_solution, i_t size);

  /**
   * @brief Set an initial dual solution.
   *
   * @note Default value is all 0.
   *
   * @param[in] initial_dual_solution Host memory pointer to a floating point array of
   * size size.
   * MPS Parser copies this data.
   * @param size Size of the initial_dual_solution array.
   */
  void set_initial_dual_solution(const f_t* initial_dual_solution, i_t size);

  i_t get_n_variables() const;
  i_t get_n_constraints() const;
  i_t get_nnz() const;
  const std::vector<f_t>& get_constraint_matrix_values() const;
  std::vector<f_t>& get_constraint_matrix_values();
  const std::vector<i_t>& get_constraint_matrix_indices() const;
  std::vector<i_t>& get_constraint_matrix_indices();
  const std::vector<i_t>& get_constraint_matrix_offsets() const;
  std::vector<i_t>& get_constraint_matrix_offsets();
  const std::vector<f_t>& get_constraint_bounds() const;
  std::vector<f_t>& get_constraint_bounds();
  const std::vector<f_t>& get_objective_coefficients() const;
  std::vector<f_t>& get_objective_coefficients();
  f_t get_objective_scaling_factor() const;
  f_t get_objective_offset() const;
  const std::vector<f_t>& get_variable_lower_bounds() const;
  const std::vector<f_t>& get_variable_upper_bounds() const;
  std::vector<f_t>& get_variable_lower_bounds();
  std::vector<f_t>& get_variable_upper_bounds();
  const std::vector<char>& get_variable_types() const;
  const std::vector<f_t>& get_constraint_lower_bounds() const;
  const std::vector<f_t>& get_constraint_upper_bounds() const;
  std::vector<f_t>& get_constraint_lower_bounds();
  std::vector<f_t>& get_constraint_upper_bounds();
  const std::vector<char>& get_row_types() const;
  bool get_sense() const;
  const std::vector<f_t>& get_initial_primal_solution() const;
  const std::vector<f_t>& get_initial_dual_solution() const;

  std::string get_objective_name() const;
  std::string get_problem_name() const;
  const std::vector<std::string>& get_variable_names() const;
  const std::vector<std::string>& get_row_names() const;

  /** whether to maximize or minimize the objective function */
  bool maximize_;
  /**
   * the constraint matrix itself in the CSR format
   * @{
   */
  std::vector<f_t> A_;
  std::vector<i_t> A_indices_;
  std::vector<i_t> A_offsets_;
  /** @} */
  /** RHS of the constraints */
  std::vector<f_t> b_;
  /** weights in the objective function */
  std::vector<f_t> c_;
  /** scale factor of the objective function */
  f_t objective_scaling_factor_{1};
  /** offset of the objective function */
  f_t objective_offset_{0};
  /** lower bounds of the variables (primal part) */
  std::vector<f_t> variable_lower_bounds_;
  /** upper bounds of the variables (primal part) */
  std::vector<f_t> variable_upper_bounds_;
  /** types of variables can be 'C' or 'I' */
  std::vector<char> var_types_;
  /** lower bounds of the constraint (dual part) */
  std::vector<f_t> constraint_lower_bounds_;
  /** upper bounds of the constraint (dual part) */
  std::vector<f_t> constraint_upper_bounds_;
  /** Type of each constraint */
  std::vector<char> row_types_;
  /** name of the objective (only a single objective is currently allowed) */
  std::string objective_name_;
  /** name of the problem  */
  std::string problem_name_;
  /** names of each of the variables in the OP */
  std::vector<std::string> var_names_{};
  /** names of each of the rows (aka constraints or objective) in the OP */
  std::vector<std::string> row_names_{};
  /** number of variables */
  i_t n_vars_{0};
  /** number of constraints in the LP representation */
  i_t n_constraints_{0};
  /** number of non-zero elements in the constraint matrix */
  i_t nnz_{0};
  /** Initial primal solution */
  std::vector<f_t> initial_primal_solution_;
  /** Initial dual solution */
  std::vector<f_t> initial_dual_solution_;
};  // class mps_data_model_t

}  // namespace cuopt::mps_parser
