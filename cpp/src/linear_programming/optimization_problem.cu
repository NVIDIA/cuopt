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

#include <cuopt/error.hpp>
#include <mps_parser/data_model_view.hpp>

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <mip/mip_constants.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/count.h>

#include <cuda_profiler_api.h>

#include <algorithm>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t>::optimization_problem_t(raft::handle_t const* handle_ptr)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    maximize_{false},
    n_vars_{0},
    n_constraints_{0},
    A_{0, stream_view_},
    A_indices_{0, stream_view_},
    A_offsets_{0, stream_view_},
    b_{0, stream_view_},
    c_{0, stream_view_},
    variable_lower_bounds_{0, stream_view_},
    variable_upper_bounds_{0, stream_view_},
    constraint_lower_bounds_{0, stream_view_},
    constraint_upper_bounds_{0, stream_view_},
    row_types_{0, stream_view_},
    variable_types_{0, stream_view_},
    var_names_{},
    row_names_{}
{
  raft::common::nvtx::range fun_scope("optimization problem construction");
}

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t>::optimization_problem_t(
  const optimization_problem_t<i_t, f_t>& other)
  : handle_ptr_(other.get_handle_ptr()),
    stream_view_(handle_ptr_->get_stream()),
    maximize_{other.get_sense()},
    n_vars_{other.get_n_variables()},
    n_constraints_{other.get_n_constraints()},
    A_{other.get_constraint_matrix_values(), stream_view_},
    A_indices_{other.get_constraint_matrix_indices(), stream_view_},
    A_offsets_{other.get_constraint_matrix_offsets(), stream_view_},
    b_{other.get_constraint_bounds(), stream_view_},
    c_{other.get_objective_coefficients(), stream_view_},
    objective_scaling_factor_{other.get_objective_scaling_factor()},
    objective_offset_{other.get_objective_offset()},
    variable_lower_bounds_{other.get_variable_lower_bounds(), stream_view_},
    variable_upper_bounds_{other.get_variable_upper_bounds(), stream_view_},
    constraint_lower_bounds_{other.get_constraint_lower_bounds(), stream_view_},
    constraint_upper_bounds_{other.get_constraint_upper_bounds(), stream_view_},
    row_types_{other.get_row_types(), stream_view_},
    variable_types_{other.get_variable_types(), stream_view_},
    objective_name_{other.get_objective_name()},
    problem_name_{other.get_problem_name()},
    problem_category_{other.get_problem_category()},
    var_names_{other.get_variable_names()},
    row_names_{other.get_row_names()}
{
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_csr_constraint_matrix(const f_t* A_values,
                                                                 i_t size_values,
                                                                 const i_t* A_indices,
                                                                 i_t size_indices,
                                                                 const i_t* A_offsets,
                                                                 i_t size_offsets)
{
  if (size_values != 0) {
    cuopt_expects(A_values != nullptr, error_type_t::ValidationError, "A_values cannot be null");
  }
  A_.resize(size_values, stream_view_);
  raft::copy(A_.data(), A_values, size_values, stream_view_);

  if (size_indices != 0) {
    cuopt_expects(A_indices != nullptr, error_type_t::ValidationError, "A_indices cannot be null");
  }
  A_indices_.resize(size_indices, stream_view_);
  raft::copy(A_indices_.data(), A_indices, size_indices, stream_view_);

  cuopt_expects(A_offsets != nullptr, error_type_t::ValidationError, "A_offsets cannot be null");
  A_offsets_.resize(size_offsets, stream_view_);
  raft::copy(A_offsets_.data(), A_offsets, size_offsets, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_constraint_bounds(const f_t* b, i_t size)
{
  cuopt_expects(b != nullptr, error_type_t::ValidationError, "b cannot be null");
  b_.resize(size, stream_view_);
  n_constraints_ = size;
  raft::copy(b_.data(), b, size, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_objective_coefficients(const f_t* c, i_t size)
{
  cuopt_expects(c != nullptr, error_type_t::ValidationError, "c cannot be null");
  c_.resize(size, stream_view_);
  n_vars_ = size;
  raft::copy(c_.data(), c, size, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_objective_scaling_factor(f_t objective_scaling_factor)
{
  objective_scaling_factor_ = objective_scaling_factor;
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_objective_offset(f_t objective_offset)
{
  objective_offset_ = objective_offset;
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_variable_lower_bounds(const f_t* variable_lower_bounds,
                                                                 i_t size)
{
  if (size != 0) {
    cuopt_expects(variable_lower_bounds != nullptr,
                  error_type_t::ValidationError,
                  "variable_lower_bounds cannot be null");
  }
  n_vars_ = size;
  variable_lower_bounds_.resize(size, stream_view_);
  raft::copy(variable_lower_bounds_.data(), variable_lower_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_variable_upper_bounds(const f_t* variable_upper_bounds,
                                                                 i_t size)
{
  if (size != 0) {
    cuopt_expects(variable_upper_bounds != nullptr,
                  error_type_t::ValidationError,
                  "variable_upper_bounds cannot be null");
  }
  n_vars_ = size;
  variable_upper_bounds_.resize(size, stream_view_);
  raft::copy(variable_upper_bounds_.data(), variable_upper_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_constraint_lower_bounds(
  const f_t* constraint_lower_bounds, i_t size)
{
  if (size != 0) {
    cuopt_expects(constraint_lower_bounds != nullptr,
                  error_type_t::ValidationError,
                  "constraint_lower_bounds cannot be null");
  }
  n_constraints_ = size;
  constraint_lower_bounds_.resize(size, stream_view_);
  raft::copy(constraint_lower_bounds_.data(), constraint_lower_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_constraint_upper_bounds(
  const f_t* constraint_upper_bounds, i_t size)
{
  if (size != 0) {
    cuopt_expects(constraint_upper_bounds != nullptr,
                  error_type_t::ValidationError,
                  "constraint_upper_bounds cannot be null");
  }
  n_constraints_ = size;
  constraint_upper_bounds_.resize(size, stream_view_);
  raft::copy(constraint_upper_bounds_.data(), constraint_upper_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_row_types(const char* row_types, i_t size)
{
  cuopt_expects(row_types != nullptr, error_type_t::ValidationError, "row_types cannot be null");
  n_constraints_ = size;
  row_types_.resize(size, stream_view_);
  raft::copy(row_types_.data(), row_types, size, stream_view_);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_variable_types(const var_t* var_types, i_t size)
{
  cuopt_expects(var_types != nullptr, error_type_t::ValidationError, "var_types cannot be null");
  variable_types_.resize(size, stream_view_);
  raft::copy(variable_types_.data(), var_types, size, stream_view_);
  // TODO when having a unified problem representation
  // compute this in a single places (currently also in problem.cu)
  i_t n_integer = thrust::count_if(handle_ptr_->get_thrust_policy(),
                                   variable_types_.begin(),
                                   variable_types_.end(),
                                   [] __device__(auto val) { return val == var_t::INTEGER; });
  // by default it is LP
  if (n_integer == size) {
    problem_category_ = problem_category_t::IP;
  } else if (n_integer > 0) {
    problem_category_ = problem_category_t::MIP;
  }
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_problem_category(const problem_category_t& category)
{
  problem_category_ = category;
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_objective_name(const std::string& objective_name)
{
  objective_name_ = objective_name;
}
template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_problem_name(const std::string& problem_name)
{
  problem_name_ = problem_name;
}
template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_variable_names(
  const std::vector<std::string>& variable_names)
{
  var_names_ = variable_names;
}
template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_row_names(const std::vector<std::string>& row_names)
{
  row_names_ = row_names;
}

template <typename i_t, typename f_t>
i_t optimization_problem_t<i_t, f_t>::get_n_variables() const
{
  return n_vars_;
}

template <typename i_t, typename f_t>
i_t optimization_problem_t<i_t, f_t>::get_n_constraints() const
{
  return n_constraints_;
}

template <typename i_t, typename f_t>
i_t optimization_problem_t<i_t, f_t>::get_nnz() const
{
  return A_.size();
}

template <typename i_t, typename f_t>
raft::handle_t const* optimization_problem_t<i_t, f_t>::get_handle_ptr() const noexcept
{
  return handle_ptr_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_matrix_values()
  const
{
  return A_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_matrix_values()
{
  return A_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<i_t>& optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices()
  const
{
  return A_indices_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<i_t>& optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices()
{
  return A_indices_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<i_t>& optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets()
  const
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<i_t>& optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets()
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_bounds() const
{
  return b_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_bounds()
{
  return b_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_objective_coefficients() const
{
  return c_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_objective_coefficients()
{
  return c_;
}

template <typename i_t, typename f_t>
f_t optimization_problem_t<i_t, f_t>::get_objective_scaling_factor() const
{
  return objective_scaling_factor_;
}

template <typename i_t, typename f_t>
f_t optimization_problem_t<i_t, f_t>::get_objective_offset() const
{
  return objective_offset_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_variable_lower_bounds() const
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_variable_upper_bounds() const
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_variable_lower_bounds()
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_variable_upper_bounds()
{
  return variable_upper_bounds_;
}
template <typename i_t, typename f_t>
const rmm::device_uvector<var_t>& optimization_problem_t<i_t, f_t>::get_variable_types() const
{
  return variable_types_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
  const
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds()
  const
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds()
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<char>& optimization_problem_t<i_t, f_t>::get_row_types() const
{
  return row_types_;
}

template <typename i_t, typename f_t>
std::string optimization_problem_t<i_t, f_t>::get_objective_name() const
{
  return objective_name_;
}

template <typename i_t, typename f_t>
std::string optimization_problem_t<i_t, f_t>::get_problem_name() const
{
  return problem_name_;
}

template <typename i_t, typename f_t>
problem_category_t optimization_problem_t<i_t, f_t>::get_problem_category() const
{
  return problem_category_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& optimization_problem_t<i_t, f_t>::get_variable_names() const
{
  return var_names_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& optimization_problem_t<i_t, f_t>::get_row_names() const
{
  return row_names_;
}

template <typename i_t, typename f_t>
bool optimization_problem_t<i_t, f_t>::get_sense() const
{
  return maximize_;
}

template <typename i_t, typename f_t>
typename optimization_problem_t<i_t, f_t>::view_t optimization_problem_t<i_t, f_t>::view() const
{
  optimization_problem_t<i_t, f_t>::view_t v;
  v.n_vars        = get_n_variables();
  v.n_constraints = get_n_constraints();
  v.nnz           = get_nnz();
  v.A             = raft::device_span<f_t>{const_cast<f_t*>(get_constraint_matrix_values().data()),
                                           get_constraint_matrix_values().size()};
  v.A_indices     = raft::device_span<const i_t>{get_constraint_matrix_indices().data(),
                                                 get_constraint_matrix_indices().size()};
  v.A_offsets     = raft::device_span<const i_t>{get_constraint_matrix_offsets().data(),
                                                 get_constraint_matrix_offsets().size()};
  v.b =
    raft::device_span<const f_t>{get_constraint_bounds().data(), get_constraint_bounds().size()};
  v.c                       = raft::device_span<const f_t>{get_objective_coefficients().data(),
                                                           get_objective_coefficients().size()};
  v.variable_lower_bounds   = raft::device_span<const f_t>{get_variable_lower_bounds().data(),
                                                           get_variable_lower_bounds().size()};
  v.variable_upper_bounds   = raft::device_span<const f_t>{get_variable_upper_bounds().data(),
                                                           get_variable_upper_bounds().size()};
  v.constraint_lower_bounds = raft::device_span<const f_t>{get_constraint_lower_bounds().data(),
                                                           get_constraint_lower_bounds().size()};
  v.constraint_upper_bounds = raft::device_span<const f_t>{get_constraint_upper_bounds().data(),
                                                           get_constraint_upper_bounds().size()};
  return v;
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_maximize(bool _maximize)
{
  maximize_ = _maximize;
}

// NOTE: Explicitly instantiate all types here in order to avoid linker error
#if MIP_INSTANTIATE_FLOAT
template class optimization_problem_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class optimization_problem_t<int, double>;
#endif

// TODO current raft to cusparse wrappers only support int64_t
// can be CUSPARSE_INDEX_16U, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_64I

}  // namespace cuopt::linear_programming
