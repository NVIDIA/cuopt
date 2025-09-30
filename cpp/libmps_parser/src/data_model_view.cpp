/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mps_parser/data_model_view.hpp>
#include <mps_parser/utilities/span.hpp>
#include <utilities/error.hpp>

namespace cuopt::mps_parser {

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_maximize(bool maximize)
{
  maximize_ = maximize;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_csr_constraint_matrix(const f_t* A_values,
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
  A_ = span<f_t const>(A_values, size_values);

  if (size_indices != 0) {
    mps_parser_expects(
      A_indices != nullptr, error_type_t::ValidationError, "A_indices cannot be null");
  }
  A_indices_ = span<i_t const>(A_indices, size_indices);

  mps_parser_expects(
    A_offsets != nullptr, error_type_t::ValidationError, "A_offsets cannot be null");
  mps_parser_expects(
    size_offsets > 0, error_type_t::ValidationError, "size_offsets cannot be empty");
  A_offsets_ = span<i_t const>(A_offsets, size_offsets);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_constraint_bounds(const f_t* b, i_t size)
{
  if (size != 0) {
    mps_parser_expects(b != nullptr, error_type_t::ValidationError, "b cannot be null");
  }
  b_ = span<f_t const>(b, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_objective_coefficients(const f_t* c, i_t size)
{
  if (size != 0) {
    mps_parser_expects(c != nullptr, error_type_t::ValidationError, "c cannot be null");
  }
  c_ = span<f_t const>(c, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_objective_scaling_factor(f_t objective_scaling_factor)
{
  objective_scaling_factor_ = objective_scaling_factor;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_objective_offset(f_t objective_offset)
{
  objective_offset_ = objective_offset;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_variable_lower_bounds(const f_t* variable_lower_bounds,
                                                            i_t size)
{
  mps_parser_expects(variable_lower_bounds != nullptr,
                     error_type_t::ValidationError,
                     "data model variable_lower_bounds cannot be null");
  variable_lower_bounds_ = span<f_t const>(variable_lower_bounds, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_variable_upper_bounds(const f_t* variable_upper_bounds,
                                                            i_t size)
{
  mps_parser_expects(variable_upper_bounds != nullptr,
                     error_type_t::ValidationError,
                     "variable_upper_bounds cannot be null");
  variable_upper_bounds_ = span<f_t const>(variable_upper_bounds, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_variable_types(const char* variable_types, i_t size)
{
  mps_parser_expects(
    variable_types != nullptr, error_type_t::ValidationError, "variable_types cannot be null");
  variable_types_ = span<char const>(variable_types, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_constraint_lower_bounds(const f_t* constraint_lower_bounds,
                                                              i_t size)
{
  mps_parser_expects(constraint_lower_bounds != nullptr,
                     error_type_t::ValidationError,
                     "constraint_lower_bounds cannot be null");
  constraint_lower_bounds_ = span<f_t const>(constraint_lower_bounds, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_constraint_upper_bounds(const f_t* constraint_upper_bounds,
                                                              i_t size)
{
  mps_parser_expects(constraint_upper_bounds != nullptr,
                     error_type_t::ValidationError,
                     "constraint_upper_bounds cannot be null");
  constraint_upper_bounds_ = span<f_t const>(constraint_upper_bounds, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_initial_primal_solution(const f_t* initial_primal_solution,
                                                              i_t size)
{
  mps_parser_expects(initial_primal_solution != nullptr,
                     error_type_t::ValidationError,
                     "initial_primal_solution cannot be null");
  initial_primal_solution_ = span<f_t const>(initial_primal_solution, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_initial_dual_solution(const f_t* initial_dual_solution,
                                                            i_t size)
{
  mps_parser_expects(initial_dual_solution != nullptr,
                     error_type_t::ValidationError,
                     "initial_dual_solution cannot be null");
  initial_dual_solution_ = span<f_t const>(initial_dual_solution, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_quadratic_objective_matrix(const f_t* Q_values,
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
  Q_objective_ = span<f_t const>(Q_values, size_values);

  if (size_indices != 0) {
    mps_parser_expects(
      Q_indices != nullptr, error_type_t::ValidationError, "Q_indices cannot be null");
  }
  Q_objective_indices_ = span<i_t const>(Q_indices, size_indices);

  mps_parser_expects(
    Q_offsets != nullptr, error_type_t::ValidationError, "Q_offsets cannot be null");
  mps_parser_expects(
    size_offsets > 0, error_type_t::ValidationError, "size_offsets cannot be empty");
  Q_objective_offsets_ = span<i_t const>(Q_offsets, size_offsets);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_row_types(const char* row_types, i_t size)
{
  mps_parser_expects(
    row_types != nullptr, error_type_t::ValidationError, "row_types cannot be null");
  row_types_ = span<char const>(row_types, size);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_objective_name(const std::string& objective_name)
{
  objective_name_ = objective_name;
}
template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_problem_name(const std::string& problem_name)
{
  problem_name_ = problem_name;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_variable_names(
  const std::vector<std::string>& variables_names)
{
  variable_names_ = variables_names;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_row_names(const std::vector<std::string>& row_names)
{
  row_names_ = row_names;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_constraint_matrix_values() const noexcept
{
  return A_;
}

template <typename i_t, typename f_t>
span<const i_t> data_model_view_t<i_t, f_t>::get_constraint_matrix_indices() const noexcept
{
  return A_indices_;
}

template <typename i_t, typename f_t>
span<const i_t> data_model_view_t<i_t, f_t>::get_constraint_matrix_offsets() const noexcept
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_constraint_bounds() const noexcept
{
  return b_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_objective_coefficients() const noexcept
{
  return c_;
}

template <typename i_t, typename f_t>
f_t data_model_view_t<i_t, f_t>::get_objective_scaling_factor() const noexcept
{
  return objective_scaling_factor_;
}

template <typename i_t, typename f_t>
f_t data_model_view_t<i_t, f_t>::get_objective_offset() const noexcept
{
  return objective_offset_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_variable_lower_bounds() const noexcept
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_variable_upper_bounds() const noexcept
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
span<const char> data_model_view_t<i_t, f_t>::get_variable_types() const noexcept
{
  return variable_types_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_constraint_lower_bounds() const noexcept
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_constraint_upper_bounds() const noexcept
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_initial_primal_solution() const noexcept
{
  return initial_primal_solution_;
}

template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_initial_dual_solution() const noexcept
{
  return initial_dual_solution_;
}

template <typename i_t, typename f_t>
span<const char> data_model_view_t<i_t, f_t>::get_row_types() const noexcept
{
  return row_types_;
}

template <typename i_t, typename f_t>
std::string data_model_view_t<i_t, f_t>::get_objective_name() const noexcept
{
  return objective_name_;
}

template <typename i_t, typename f_t>
std::string data_model_view_t<i_t, f_t>::get_problem_name() const noexcept
{
  return problem_name_;
}

template <typename i_t, typename f_t>
bool data_model_view_t<i_t, f_t>::get_sense() const noexcept
{
  return maximize_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& data_model_view_t<i_t, f_t>::get_variable_names() const noexcept
{
  return variable_names_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& data_model_view_t<i_t, f_t>::get_row_names() const noexcept
{
  return row_names_;
}

// QPS-specific getter implementations
template <typename i_t, typename f_t>
span<const f_t> data_model_view_t<i_t, f_t>::get_quadratic_objective_values() const noexcept
{
  return Q_objective_;
}

template <typename i_t, typename f_t>
span<const i_t> data_model_view_t<i_t, f_t>::get_quadratic_objective_indices() const noexcept
{
  return Q_objective_indices_;
}

template <typename i_t, typename f_t>
span<const i_t> data_model_view_t<i_t, f_t>::get_quadratic_objective_offsets() const noexcept
{
  return Q_objective_offsets_;
}

template <typename i_t, typename f_t>
bool data_model_view_t<i_t, f_t>::has_quadratic_objective() const noexcept
{
  return Q_objective_.size() > 0;
}

// NOTE: Explicitly instantiate all types here in order to avoid linker error
template class data_model_view_t<int, float>;

template class data_model_view_t<int, double>;

}  // namespace cuopt::mps_parser
