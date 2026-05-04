/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mps_parser/mps_data_model.hpp>
#include <utilities/error.hpp>

#include <algorithm>
#include <utility>

namespace cuopt::mps_parser {

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_csr_constraint_matrix(std::span<const f_t> A_values,
                                                           std::span<const i_t> A_indices,
                                                           std::span<const i_t> A_offsets)
{
  mps_parser_expects(
    !A_offsets.empty(), error_type_t::ValidationError, "A_offsets cannot be empty");
  A_.assign(A_values.begin(), A_values.end());
  A_indices_.assign(A_indices.begin(), A_indices.end());
  A_offsets_.assign(A_offsets.begin(), A_offsets.end());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_constraint_bounds(std::span<const f_t> b)
{
  b_.assign(b.begin(), b.end());
  n_constraints_ = static_cast<i_t>(b.size());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_objective_coefficients(std::span<const f_t> c)
{
  c_.assign(c.begin(), c.end());
  n_vars_ = static_cast<i_t>(c.size());
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
void mps_data_model_t<i_t, f_t>::set_variable_lower_bounds(
  std::span<const f_t> variable_lower_bounds)
{
  variable_lower_bounds_.assign(variable_lower_bounds.begin(), variable_lower_bounds.end());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_variable_upper_bounds(
  std::span<const f_t> variable_upper_bounds)
{
  variable_upper_bounds_.assign(variable_upper_bounds.begin(), variable_upper_bounds.end());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_constraint_lower_bounds(
  std::span<const f_t> constraint_lower_bounds)
{
  constraint_lower_bounds_.assign(constraint_lower_bounds.begin(), constraint_lower_bounds.end());
  n_constraints_ = static_cast<i_t>(constraint_lower_bounds.size());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_constraint_upper_bounds(
  std::span<const f_t> constraint_upper_bounds)
{
  constraint_upper_bounds_.assign(constraint_upper_bounds.begin(), constraint_upper_bounds.end());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_row_types(std::span<const char> row_types)
{
  row_types_.assign(row_types.begin(), row_types.end());
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
void mps_data_model_t<i_t, f_t>::set_initial_primal_solution(
  std::span<const f_t> initial_primal_solution)
{
  initial_primal_solution_.assign(initial_primal_solution.begin(), initial_primal_solution.end());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_initial_dual_solution(
  std::span<const f_t> initial_dual_solution)
{
  initial_dual_solution_.assign(initial_dual_solution.begin(), initial_dual_solution.end());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::set_quadratic_objective_matrix(std::span<const f_t> Q_values,
                                                                std::span<const i_t> Q_indices,
                                                                std::span<const i_t> Q_offsets)
{
  mps_parser_expects(
    !Q_offsets.empty(), error_type_t::ValidationError, "Q_offsets cannot be empty");
  Q_objective_values_.assign(Q_values.begin(), Q_values.end());
  Q_objective_indices_.assign(Q_indices.begin(), Q_indices.end());
  Q_objective_offsets_.assign(Q_offsets.begin(), Q_offsets.end());
}

template <typename i_t, typename f_t>
void mps_data_model_t<i_t, f_t>::append_quadratic_constraint(i_t constraint_row_index,
                                                             const std::string& constraint_row_name,
                                                             char constraint_row_type,
                                                             std::span<const f_t> linear_values,
                                                             std::span<const i_t> linear_indices,
                                                             f_t rhs_value,
                                                             std::span<const f_t> quadratic_values,
                                                             std::span<const i_t> quadratic_indices,
                                                             std::span<const i_t> quadratic_offsets)
{
  mps_parser_expects(constraint_row_index >= 0,
                     error_type_t::ValidationError,
                     "constraint_row_index must be non-negative");

  mps_parser_expects(constraint_row_type == 'L',
                     error_type_t::ValidationError,
                     "Quadratic constraint ROWS type must be 'L' (less-or-equal); got '%c'. "
                     "Only 'L' is supported for convex quadratic constraints.",
                     constraint_row_type);

  mps_parser_expects(linear_values.size() == linear_indices.size(),
                     error_type_t::ValidationError,
                     "linear_values and linear_indices must have the same nnz count");

  mps_parser_expects(
    !quadratic_offsets.empty(), error_type_t::ValidationError, "quadratic_offsets cannot be empty");

  quadratic_constraint_t qc;
  qc.constraint_row_index = constraint_row_index;
  qc.constraint_row_name  = constraint_row_name;
  qc.constraint_row_type  = constraint_row_type;
  qc.rhs_value            = rhs_value;
  qc.linear_values.assign(linear_values.begin(), linear_values.end());
  qc.linear_indices.assign(linear_indices.begin(), linear_indices.end());
  qc.quadratic_values.assign(quadratic_values.begin(), quadratic_values.end());
  qc.quadratic_indices.assign(quadratic_indices.begin(), quadratic_indices.end());
  qc.quadratic_offsets.assign(quadratic_offsets.begin(), quadratic_offsets.end());

  quadratic_constraints_.push_back(std::move(qc));
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
  return Q_objective_values_;
}

template <typename i_t, typename f_t>
std::vector<f_t>& mps_data_model_t<i_t, f_t>::get_quadratic_objective_values()
{
  return Q_objective_values_;
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
auto mps_data_model_t<i_t, f_t>::get_quadratic_constraints() const
  -> const std::vector<quadratic_constraint_t>&
{
  return quadratic_constraints_;
}

template <typename i_t, typename f_t>
bool mps_data_model_t<i_t, f_t>::has_quadratic_objective() const noexcept
{
  return !Q_objective_values_.empty();
}

template <typename i_t, typename f_t>
bool mps_data_model_t<i_t, f_t>::has_quadratic_constraints() const noexcept
{
  return !quadratic_constraints_.empty();
}

// NOTE: Explicitly instantiate all types here in order to avoid linker error
template class mps_data_model_t<int, float>;

template class mps_data_model_t<int, double>;
//  TODO current raft to cusparse wrappers only support int64_t
//  can be CUSPARSE_INDEX_16U, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_64I

}  // namespace cuopt::mps_parser
