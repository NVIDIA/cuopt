/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>

#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mps_parser/data_model_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace cuopt::linear_programming {

enum class var_t { CONTINUOUS = 0, INTEGER };
enum class problem_category_t : int8_t { LP = 0, MIP = 1, IP = 2 };

/**
 * @brief A representation of a linear programming (LP) optimization problem
 *
 * @tparam f_t  Data type of the variables and their weights in the equations
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
class optimization_problem_t {
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

  optimization_problem_t(raft::handle_t const* handle_ptr);
  optimization_problem_t(const optimization_problem_t<i_t, f_t>& other);

  std::vector<internals::base_solution_callback_t*> mip_callbacks_;

  /**
   * @brief Set the sense of optimization to maximize.
   * @note Setting before calling the solver is optional, default value if false
   * (minimize).
   *
   * @param[in] maximize true means to maximize the objective function, else
   * minimize.
   */
  void set_maximize(bool maximize);
  /**
   * @brief Set the constraint matrix (A) in CSR format. For more information
   about CSR checkout:
   * https://docs.nvidia.com/cuda/cusparse/index.html#compressed-sparse-row-csr

   * @note Setting before calling the solver is mandatory.
   *
   * @throws cuopt::logic_error when an error occurs.
   * @param[in] A_values Values of the CSR representation of the constraint
   matrix as a device or host memory pointer to a floating point array of size
   size_values.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler
   passed to the problem.
   * @param size_values Size of the A_values array.
   * @param[in] A_indices Indices of the CSR representation of the constraint
   matrix as a device or host memory pointer to an integer array of size
   size_indices.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler
   passed to the problem.
   * @param size_indices Size of the A_indices array.
   * @param[in] A_offsets Offsets of the CSR representation of the constraint
   matrix as a device or host memory pointer to a integer array of size
   size_offsets.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler
   passed to the problem.
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
   * @param[in] b Device or host memory pointer to a floating point array of
   * size size. cuOpt copies this data. Copy happens on the stream of the
   * raft:handler passed to the problem.
   * @param size Size of the b array.
   */
  void set_constraint_bounds(const f_t* b, i_t size);
  /**
   * @brief Set the objective coefficients (c) array.
   * @note Setting before calling the solver is mandatory.
   *
   * @param[in] c Device or host memory pointer to a floating point array of
   * size size. cuOpt copies this data. Copy happens on the stream of the
   * raft:handler passed to the problem.
   * @param size Size of the c array.
   */
  void set_objective_coefficients(const f_t* c, i_t size);
  /**
   * @brief Set the scaling factor of the objective function (scaling_factor *
   * objective_value).
   * @note Setting before calling the solver is optional, default value if 1.
   *
   * @param objective_scaling_factor Objective scaling factor value.
   */
  void set_objective_scaling_factor(f_t objective_scaling_factor);
  /**
   * @brief Set the offset of the objective function (objective_offset +
   * objective_value).
   * @note Setting before calling the solver is optional, default value if 0.
   *
   * @param objective_offset Objective offset value.
   */
  void set_objective_offset(f_t objective_offset);
  /**
   * @brief Set the variables (x) lower bounds.
   * @note Setting before calling the solver is optional, default value for all
   * is 0.
   *
   * @param[in] variable_lower_bounds Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
   * @param size Size of the variable_lower_bounds array
   */
  void set_variable_lower_bounds(const f_t* variable_lower_bounds, i_t size);
  /**
   * @brief Set the variables (x) upper bounds.
   * @note Setting before calling the solver is optional, default value for all
   * is +infinity.
   *
   * @param[in] variable_upper_bounds Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
   * @param size Size of the variable_upper_bounds array.
   */
  void set_variable_upper_bounds(const f_t* variable_upper_bounds, i_t size);
  /**
   * @brief Set the variables types.
   * @note Setting before calling the solver is optional, default value for all
   * is CONTINUOUS.
   *
   * @param[in] variable_types Device or host memory pointer to a var_t array.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler
   * passed to the problem.
   * @param size Size of the variable_types array.
   */
  void set_variable_types(const var_t* variable_types, i_t size);
  void set_problem_category(const problem_category_t& category);
  /**
   * @brief Set the constraints lower bounds.
   * @note Setting before calling the solver is optional if you set the row
   * type, else it's mandatory along with the upper bounds.
   *
   * @param[in] constraint_lower_bounds Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
   * @param size Size of the constraint_lower_bounds array
   */
  void set_constraint_lower_bounds(const f_t* constraint_lower_bounds, i_t size);
  /**
   * @brief Set the constraints upper bounds.
   * @note Setting before calling the solver is optional if you set the row
   * type, else it's mandatory along with the lower bounds. If both are set,
   * priority goes to set_constraints.
   *
   * @param[in] constraint_upper_bounds Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
   * @param size Size of the constraint_upper_bounds array
   */
  void set_constraint_upper_bounds(const f_t* constraint_upper_bounds, i_t size);

  /**
   * @brief Set the type of each row (constraint). Possible values are:
   * 'E' for equality ( = ): lower & upper constrains bound equal to b
   * 'L' for less-than ( <= ): lower constrains bound equal to -infinity, upper
   * constrains bound equal to b 'G' for greater-than ( >= ): lower constrains
   * bound equal to b, upper constrains bound equal to +infinity
   * @note Setting before calling the solver is optional if you set the
   * constraint lower and upper bounds, else it's mandatory If both are set,
   * priority goes to set_constraints.
   *
   * @param[in] row_types Device or host memory pointer to a character array of
   * size size.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler
   * passed to the problem.
   * @param size Size of the row_types array
   */
  void set_row_types(const char* row_types, i_t size);

  /**
   * @brief Set the name of the objective function.
   * @note Setting before calling the solver is optional. Value is only used for
   * file generation of the solution.
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
   * @note Setting before calling the solver is optional. Value is only used for
   * file generation of the solution.
   *
   * @param[in] variable_names Variable names values.
   */
  void set_variable_names(const std::vector<std::string>& variables_names);
  /**
   * @brief Set the row names.
   * @note Setting before calling the solver is optional. Value is only used for
   * file generation of the solution.
   *
   * @param[in] row_names Row names value.
   */
  void set_row_names(const std::vector<std::string>& row_names);

  /**
   * @brief Write the problem to an MPS formatted file
   *
   * @param[in] mps_file_path Path to the MPS file to write
   */
  void write_to_mps(const std::string& mps_file_path);

  /* Print scaling information */
  void print_scaling_information() const;

  i_t get_n_variables() const;
  i_t get_n_constraints() const;
  i_t get_nnz() const;
  i_t get_n_integers() const;
  raft::handle_t const* get_handle_ptr() const noexcept;
  const rmm::device_uvector<f_t>& get_constraint_matrix_values() const;
  rmm::device_uvector<f_t>& get_constraint_matrix_values();
  const rmm::device_uvector<i_t>& get_constraint_matrix_indices() const;
  rmm::device_uvector<i_t>& get_constraint_matrix_indices();
  const rmm::device_uvector<i_t>& get_constraint_matrix_offsets() const;
  rmm::device_uvector<i_t>& get_constraint_matrix_offsets();
  const rmm::device_uvector<f_t>& get_constraint_bounds() const;
  rmm::device_uvector<f_t>& get_constraint_bounds();
  const rmm::device_uvector<f_t>& get_objective_coefficients() const;
  rmm::device_uvector<f_t>& get_objective_coefficients();
  f_t get_objective_scaling_factor() const;
  f_t get_objective_offset() const;
  const rmm::device_uvector<f_t>& get_variable_lower_bounds() const;
  const rmm::device_uvector<f_t>& get_variable_upper_bounds() const;
  rmm::device_uvector<f_t>& get_variable_lower_bounds();
  rmm::device_uvector<f_t>& get_variable_upper_bounds();
  const rmm::device_uvector<f_t>& get_constraint_lower_bounds() const;
  const rmm::device_uvector<f_t>& get_constraint_upper_bounds() const;
  rmm::device_uvector<f_t>& get_constraint_lower_bounds();
  rmm::device_uvector<f_t>& get_constraint_upper_bounds();
  const rmm::device_uvector<char>& get_row_types() const;
  const rmm::device_uvector<var_t>& get_variable_types() const;
  bool get_sense() const;
  bool empty() const;

  std::string get_objective_name() const;
  std::string get_problem_name() const;
  // Unless an integer variable is added, by default it is LP
  problem_category_t get_problem_category() const;
  const std::vector<std::string>& get_variable_names() const;
  const std::vector<std::string>& get_row_names() const;

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view() const;

 private:
  void add_row_related_vars_to_row(std::vector<i_t>& indices,
                                   std::vector<f_t>& values,
                                   std::vector<i_t>& A_offsets,
                                   std::vector<i_t>& A_indices,
                                   std::vector<f_t>& A_values);

  // Pointer to library handle (RAFT) containing hardware resources information
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  /** problem classification */
  problem_category_t problem_category_ = problem_category_t::LP;
  /** whether to maximize or minimize the objective function */
  bool maximize_;
  /** number of variables */
  i_t n_vars_;
  /** number of constraints in the LP representation */
  i_t n_constraints_;
  /**
   * the constraint matrix itself in the CSR format
   * @{
   */
  rmm::device_uvector<f_t> A_;
  rmm::device_uvector<i_t> A_indices_;
  rmm::device_uvector<i_t> A_offsets_;
  /** @} */
  /** RHS of the constraints */
  rmm::device_uvector<f_t> b_;
  /** weights in the objective function */
  rmm::device_uvector<f_t> c_;
  /** scale factor of the objective function */
  f_t objective_scaling_factor_{1};
  /** offset of the objective function */
  f_t objective_offset_{0};
  /** lower bounds of the variables (primal part) */
  rmm::device_uvector<f_t> variable_lower_bounds_;
  /** upper bounds of the variables (primal part) */
  rmm::device_uvector<f_t> variable_upper_bounds_;
  /** lower bounds of the constraint (dual part) */
  rmm::device_uvector<f_t> constraint_lower_bounds_;
  /** upper bounds of the constraint (dual part) */
  rmm::device_uvector<f_t> constraint_upper_bounds_;
  /** Type of each constraint */
  rmm::device_uvector<char> row_types_;
  /** Type of each variable */
  rmm::device_uvector<var_t> variable_types_;
  /** name of the objective (only a single objective is currently allowed) */
  std::string objective_name_;
  /** name of the problem  */
  std::string problem_name_;
  /** names of each of the variables in the OP */
  std::vector<std::string> var_names_{};
  /** names of each of the rows (aka constraints or objective) in the OP */
  std::vector<std::string> row_names_{};
};  // class optimization_problem_t

}  // namespace cuopt::linear_programming
