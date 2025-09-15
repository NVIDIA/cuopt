/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mps_parser/mps_data_model.hpp>
#include <utilities/error.hpp>

#include <algorithm>

namespace cuopt::mps_parser {

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_csr_constraint_matrix(const f_t* A_values,
                                                           i_t size_values,
                                                           const i_t* A_indices,
                                                           i_t size_indices,
                                                           const i_t* A_offsets,
                                                           i_t size_offsets)
{
  if (size_values != 0) {
    mps_parser_expects(
      A_values != nullptr, error_type_t::ValidationError, "A_values cannot be null");
  }
  A_.resize(size_values);
  std::copy(A_values, A_values + size_values, A_.data());

  if (size_indices != 0) {
    mps_parser_expects(
      A_indices != nullptr, error_type_t::ValidationError, "A_indices cannot be null");
  }
  A_indices_.resize(size_indices);
  std::copy(A_indices, A_indices + size_indices, A_indices_.data());

  mps_parser_expects(
    A_offsets != nullptr, error_type_t::ValidationError, "A_offsets cannot be null");
  mps_parser_expects(
    size_offsets > 0, error_type_t::ValidationError, "size_offsets cannot be empty");
  A_offsets_.resize(size_offsets);
  std::copy(A_offsets, A_offsets + size_offsets, A_offsets_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_constraint_bounds(const f_t* b, i_t size)
{
  if (size != 0) {
    mps_parser_expects(b != nullptr, error_type_t::ValidationError, "b cannot be null");
  }
  b_.resize(size);
  n_constraints_ = size;
  std::copy(b, b + size, b_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_objective_coefficients(const f_t* c, i_t size)
{
  if (size != 0) {
    mps_parser_expects(c != nullptr, error_type_t::ValidationError, "c cannot be null");
  }
  c_.resize(size);
  n_vars_ = size;
  std::copy(c, c + size, c_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_objective_scaling_factor(f_t objective_scaling_factor)
{
  objective_scaling_factor_ = objective_scaling_factor;
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_objective_offset(f_t objective_offset)
{
  objective_offset_ = objective_offset;
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_variable_lower_bounds(const f_t* variable_lower_bounds,
                                                           i_t size)
{
  if (size != 0) {
    mps_parser_expects(variable_lower_bounds != nullptr,
                       error_type_t::ValidationError,
                       "variable_lower_bounds cannot be null");
  }
  variable_lower_bounds_.resize(size);
  std::copy(variable_lower_bounds, variable_lower_bounds + size, variable_lower_bounds_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_variable_upper_bounds(const f_t* variable_upper_bounds,
                                                           i_t size)
{
  if (size != 0) {
    mps_parser_expects(variable_upper_bounds != nullptr,
                       error_type_t::ValidationError,
                       "variable_upper_bounds cannot be null");
  }
  variable_upper_bounds_.resize(size);
  std::copy(variable_upper_bounds, variable_upper_bounds + size, variable_upper_bounds_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_constraint_lower_bounds(const f_t* constraint_lower_bounds,
                                                             i_t size)
{
  if (size != 0) {
    mps_parser_expects(constraint_lower_bounds != nullptr,
                       error_type_t::ValidationError,
                       "constraint_lower_bounds cannot be null");
  }
  constraint_lower_bounds_.resize(size);
  n_constraints_ = size;
  std::copy(
    constraint_lower_bounds, constraint_lower_bounds + size, constraint_lower_bounds_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_constraint_upper_bounds(const f_t* constraint_upper_bounds,
                                                             i_t size)
{
  if (size != 0) {
    mps_parser_expects(constraint_upper_bounds != nullptr,
                       error_type_t::ValidationError,
                       "constraint_upper_bounds cannot be null");
  }
  constraint_upper_bounds_.resize(size);
  std::copy(
    constraint_upper_bounds, constraint_upper_bounds + size, constraint_upper_bounds_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_row_types(const char* row_types, i_t size)
{
  mps_parser_expects(
    row_types != nullptr, error_type_t::ValidationError, "row_types cannot be null");
  row_types_.resize(size);
  std::copy(row_types, row_types + size, row_types_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_objective_name(const std::string& objective_name)
{
  objective_name_ = objective_name;
}
template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_problem_name(const std::string& problem_name)
{
  problem_name_ = problem_name;
}
template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_variable_names(const std::vector<std::string>& variable_names)
{
  var_names_ = variable_names;
}
template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_variable_types(const std::vector<char>& variable_types)
{
  var_types_ = variable_types;
}
template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_row_names(const std::vector<std::string>& row_names)
{
  row_names_ = row_names;
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_initial_primal_solution(const f_t* initial_primal_solution,
                                                             i_t size)
{
  mps_parser_expects(initial_primal_solution != nullptr,
                     error_type_t::ValidationError,
                     "initial_primal_solution cannot be null");
  initial_primal_solution_.resize(size);
  std::copy(
    initial_primal_solution, initial_primal_solution + size, initial_primal_solution_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_initial_dual_solution(const f_t* initial_dual_solution,
                                                           i_t size)
{
  mps_parser_expects(initial_dual_solution != nullptr,
                     error_type_t::ValidationError,
                     "initial_dual_solution cannot be null");
  initial_dual_solution_.resize(size);
  std::copy(initial_dual_solution, initial_dual_solution + size, initial_dual_solution_.data());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_quadratic_objective_matrix(const f_t* Q_values,
                                                                i_t size_values,
                                                                const i_t* Q_indices,
                                                                i_t size_indices,
                                                                const i_t* Q_offsets,
                                                                i_t size_offsets)
{
  if (size_values != 0) {
    mps_parser_expects(
      Q_values != nullptr, error_type_t::ValidationError, "Q_values cannot be null");
  }
  Q_objective_.resize(size_values);
  std::copy(Q_values, Q_values + size_values, Q_objective_.data());

  if (size_indices != 0) {
    mps_parser_expects(
      Q_indices != nullptr, error_type_t::ValidationError, "Q_indices cannot be null");
  }
  Q_objective_indices_.resize(size_indices);
  std::copy(Q_indices, Q_indices + size_indices, Q_objective_indices_.data());

  mps_parser_expects(
    Q_offsets != nullptr, error_type_t::ValidationError, "Q_offsets cannot be null");
  mps_parser_expects(
    size_offsets > 0, error_type_t::ValidationError, "size_offsets cannot be empty");
  Q_objective_offsets_.resize(size_offsets);
  std::copy(Q_offsets, Q_offsets + size_offsets, Q_objective_offsets_.data());
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_matrix_values() const
{
  return A_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_matrix_values()
{
  return A_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_constraint_matrix_indices() const
{
  return A_indices_;
}

template <typename i_t, typename f_t>
std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_constraint_matrix_indices()
{
  return A_indices_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_constraint_matrix_offsets() const
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_constraint_matrix_offsets()
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_bounds() const
{
  return b_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_bounds()
{
  return b_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_objective_coefficients() const
{
  return c_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_objective_coefficients()
{
  return c_;
}

template <typename i_t, typename f_t>
f_t mps_data_model_t<i_t, f_t>::get_objective_scaling_factor() const
{
  return objective_scaling_factor_;
}

template <typename i_t, typename f_t>
f_t mps_data_model_t<i_t, f_t>::get_objective_offset() const
{
  return objective_offset_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_variable_lower_bounds() const
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_variable_upper_bounds() const
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_variable_lower_bounds()
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_variable_upper_bounds()
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_lower_bounds() const
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_upper_bounds() const
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_lower_bounds()
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_constraint_upper_bounds()
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
const std::vector<char>& mps_data_model_t<i_t, f_t>::get_row_types() const
{
  return row_types_;
}

template <typename i_t, typename f_t>
std::string mps_data_model_t<i_t, f_t>::get_objective_name() const
{
  return objective_name_;
}

template <typename i_t, typename f_t>
std::string mps_data_model_t<i_t, f_t>::get_problem_name() const
{
  return problem_name_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& mps_data_model_t<i_t, f_t>::get_variable_names() const
{
  return var_names_;
}

template <typename i_t, typename f_t>
const std::vector<char>& mps_data_model_t<i_t, f_t>::get_variable_types() const
{
  return var_types_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& mps_data_model_t<i_t, f_t>::get_row_names() const
{
  return row_names_;
}

template <typename i_t, typename f_t>
bool mps_data_model_t<i_t, f_t>::get_sense() const
{
  return maximize_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_initial_primal_solution() const
{
  return initial_primal_solution_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_initial_dual_solution() const
{
  return initial_dual_solution_;
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_maximize(bool _maximize)
{
  maximize_ = _maximize;
}

template <typename i_t, typename f_t>
i_t mps_data_model_t<i_t, f_t>::get_n_variables() const
{
  return n_vars_;
}

template <typename i_t, typename f_t>
i_t mps_data_model_t<i_t, f_t>::get_n_constraints() const
{
  return n_constraints_;
}

template <typename i_t, typename f_t>
i_t mps_data_model_t<i_t, f_t>::get_nnz() const
{
  return A_.size();
}

// QPS-specific getter implementations
template <typename i_t, typename f_t>
const std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_quadratic_objective_values() const
{
  return Q_objective_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_quadratic_objective_values()
{
  return Q_objective_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_quadratic_objective_indices() const
{
  return Q_objective_indices_;
}

template <typename i_t, typename f_t>
std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_quadratic_objective_indices()
{
  return Q_objective_indices_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_quadratic_objective_offsets() const
{
  return Q_objective_offsets_;
}

template <typename i_t, typename f_t>
std::vector<i_t>& mps_data_model_t<i_t, f_t>::get_quadratic_objective_offsets()
{
  return Q_objective_offsets_;
}

template <typename i_t, typename f_t>
bool mps_data_model_t<i_t, f_t>::has_quadratic_objective() const noexcept
{
  return !Q_objective_.empty();
}

// NOTE: Explicitly instantiate all types here in order to avoid linker error
template class mps_data_model_t<int, float>;

template class mps_data_model_t<int, double>;
//  TODO current raft to cusparse wrappers only support int64_t
//  can be CUSPARSE_INDEX_16U, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_64I

}  // namespace cuopt::mps_parser
