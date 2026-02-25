/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/csr_matrix_utils.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/solve_remote.hpp>

#include <mip_heuristics/mip_constants.hpp>
#include <mps_parser/writer.hpp>
#include <utilities/logger.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/cuda_support.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
gpu_optimization_problem_t<i_t, f_t>::gpu_optimization_problem_t(raft::handle_t const* handle_ptr)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr->get_stream()),
    A_(0, stream_view_),
    A_indices_(0, stream_view_),
    A_offsets_(0, stream_view_),
    b_(0, stream_view_),
    c_(0, stream_view_),
    variable_lower_bounds_(0, stream_view_),
    variable_upper_bounds_(0, stream_view_),
    constraint_lower_bounds_(0, stream_view_),
    constraint_upper_bounds_(0, stream_view_),
    row_types_(0, stream_view_),
    variable_types_(0, stream_view_)
{
  CUOPT_LOG_DEBUG("gpu_optimization_problem_t constructor: Using GPU backend");
}

// ==============================================================================
// Setters
// ==============================================================================

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_maximize(bool maximize)
{
  maximize_ = maximize;
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_csr_constraint_matrix(const f_t* A_values,
                                                                     i_t size_values,
                                                                     const i_t* A_indices,
                                                                     i_t size_indices,
                                                                     const i_t* A_offsets,
                                                                     i_t size_offsets)
{
  cuopt_expects(size_offsets > 0,
                error_type_t::ValidationError,
                "CSR offsets array must have at least one element");
  n_constraints_ = size_offsets - 1;

  A_.resize(size_values, stream_view_);
  A_indices_.resize(size_indices, stream_view_);
  A_offsets_.resize(size_offsets, stream_view_);

  raft::copy(A_.data(), A_values, size_values, stream_view_);
  raft::copy(A_indices_.data(), A_indices, size_indices, stream_view_);
  raft::copy(A_offsets_.data(), A_offsets, size_offsets, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_constraint_bounds(const f_t* b, i_t size)
{
  b_.resize(size, stream_view_);
  raft::copy(b_.data(), b, size, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_objective_coefficients(const f_t* c, i_t size)
{
  n_vars_ = size;
  c_.resize(size, stream_view_);
  raft::copy(c_.data(), c, size, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_objective_scaling_factor(
  f_t objective_scaling_factor)
{
  objective_scaling_factor_ = objective_scaling_factor;
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_objective_offset(f_t objective_offset)
{
  objective_offset_ = objective_offset;
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_quadratic_objective_matrix(
  const f_t* Q_values,
  i_t size_values,
  const i_t* Q_indices,
  i_t size_indices,
  const i_t* Q_offsets,
  i_t size_offsets,
  bool validate_positive_semi_definite)
{
  // TODO: If validate_positive_semi_definite is true, verify the matrix is PSD
  // (e.g. attempt a Cholesky factorization and reject if it fails).

  Q_values_.resize(size_values);
  Q_indices_.resize(size_indices);
  Q_offsets_.resize(size_offsets);

  std::copy(Q_values, Q_values + size_values, Q_values_.begin());
  std::copy(Q_indices, Q_indices + size_indices, Q_indices_.begin());
  std::copy(Q_offsets, Q_offsets + size_offsets, Q_offsets_.begin());
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_variable_lower_bounds(
  const f_t* variable_lower_bounds, i_t size)
{
  variable_lower_bounds_.resize(size, stream_view_);
  raft::copy(variable_lower_bounds_.data(), variable_lower_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_variable_upper_bounds(
  const f_t* variable_upper_bounds, i_t size)
{
  variable_upper_bounds_.resize(size, stream_view_);
  raft::copy(variable_upper_bounds_.data(), variable_upper_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_variable_types(const var_t* variable_types, i_t size)
{
  variable_types_.resize(size, stream_view_);
  raft::copy(variable_types_.data(), variable_types, size, stream_view_);

  // Auto-detect problem category based on variable types (matching original optimization_problem_t)
  i_t n_integer = thrust::count_if(handle_ptr_->get_thrust_policy(),
                                   variable_types_.begin(),
                                   variable_types_.end(),
                                   [] __device__(auto val) { return val == var_t::INTEGER; });
  // By default it is LP
  if (n_integer == size) {
    problem_category_ = problem_category_t::IP;
  } else if (n_integer > 0) {
    problem_category_ = problem_category_t::MIP;
  }
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_problem_category(const problem_category_t& category)
{
  problem_category_ = category;
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_constraint_lower_bounds(
  const f_t* constraint_lower_bounds, i_t size)
{
  constraint_lower_bounds_.resize(size, stream_view_);
  raft::copy(constraint_lower_bounds_.data(), constraint_lower_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_constraint_upper_bounds(
  const f_t* constraint_upper_bounds, i_t size)
{
  constraint_upper_bounds_.resize(size, stream_view_);
  raft::copy(constraint_upper_bounds_.data(), constraint_upper_bounds, size, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_row_types(const char* row_types, i_t size)
{
  row_types_.resize(size, stream_view_);
  raft::copy(row_types_.data(), row_types, size, stream_view_);
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_objective_name(const std::string& objective_name)
{
  objective_name_ = objective_name;
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_problem_name(const std::string& problem_name)
{
  problem_name_ = problem_name;
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_variable_names(
  const std::vector<std::string>& variable_names)
{
  var_names_ = variable_names;
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::set_row_names(const std::vector<std::string>& row_names)
{
  row_names_ = row_names;
}

// ==============================================================================
// Device Getters
// ==============================================================================

template <typename i_t, typename f_t>
i_t gpu_optimization_problem_t<i_t, f_t>::get_n_variables() const
{
  return n_vars_;
}

template <typename i_t, typename f_t>
i_t gpu_optimization_problem_t<i_t, f_t>::get_n_constraints() const
{
  return n_constraints_;
}

template <typename i_t, typename f_t>
i_t gpu_optimization_problem_t<i_t, f_t>::get_nnz() const
{
  return A_.size();
}

template <typename i_t, typename f_t>
i_t gpu_optimization_problem_t<i_t, f_t>::get_n_integers() const
{
  if (variable_types_.size() == 0) return 0;

  return thrust::count(
    rmm::exec_policy(stream_view_), variable_types_.begin(), variable_types_.end(), var_t::INTEGER);
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_values()
  const
{
  return A_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_values()
{
  return A_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<i_t>&
gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices() const
{
  return A_indices_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<i_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices()
{
  return A_indices_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<i_t>&
gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets() const
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<i_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets()
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_bounds() const
{
  return b_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_bounds()
{
  return b_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_objective_coefficients()
  const
{
  return c_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_objective_coefficients()
{
  return c_;
}

template <typename i_t, typename f_t>
f_t gpu_optimization_problem_t<i_t, f_t>::get_objective_scaling_factor() const
{
  return objective_scaling_factor_;
}

template <typename i_t, typename f_t>
f_t gpu_optimization_problem_t<i_t, f_t>::get_objective_offset() const
{
  return objective_offset_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_variable_lower_bounds()
  const
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_variable_lower_bounds()
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_variable_upper_bounds()
  const
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_variable_upper_bounds()
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
  const
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds()
  const
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds()
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<char>& gpu_optimization_problem_t<i_t, f_t>::get_row_types() const
{
  return row_types_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<var_t>& gpu_optimization_problem_t<i_t, f_t>::get_variable_types() const
{
  return variable_types_;
}

template <typename i_t, typename f_t>
bool gpu_optimization_problem_t<i_t, f_t>::get_sense() const
{
  return maximize_;
}

template <typename i_t, typename f_t>
bool gpu_optimization_problem_t<i_t, f_t>::empty() const
{
  return n_vars_ == 0 || n_constraints_ == 0;
}

template <typename i_t, typename f_t>
std::string gpu_optimization_problem_t<i_t, f_t>::get_objective_name() const
{
  return objective_name_;
}

template <typename i_t, typename f_t>
std::string gpu_optimization_problem_t<i_t, f_t>::get_problem_name() const
{
  return problem_name_;
}

template <typename i_t, typename f_t>
problem_category_t gpu_optimization_problem_t<i_t, f_t>::get_problem_category() const
{
  return problem_category_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& gpu_optimization_problem_t<i_t, f_t>::get_variable_names() const
{
  return var_names_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& gpu_optimization_problem_t<i_t, f_t>::get_row_names() const
{
  return row_names_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& gpu_optimization_problem_t<i_t, f_t>::get_quadratic_objective_offsets()
  const
{
  return Q_offsets_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& gpu_optimization_problem_t<i_t, f_t>::get_quadratic_objective_indices()
  const
{
  return Q_indices_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& gpu_optimization_problem_t<i_t, f_t>::get_quadratic_objective_values() const
{
  return Q_values_;
}

template <typename i_t, typename f_t>
bool gpu_optimization_problem_t<i_t, f_t>::has_quadratic_objective() const
{
  return !Q_values_.empty();
}

template <typename i_t, typename f_t>
raft::handle_t const* gpu_optimization_problem_t<i_t, f_t>::get_handle_ptr() const noexcept
{
  return handle_ptr_;
}

// ==============================================================================
// Host Getters (copy from GPU to CPU)
// ==============================================================================

template <typename i_t, typename f_t>
std::vector<f_t> gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_values_host() const
{
  std::vector<f_t> host_data(A_.size());
  raft::copy(host_data.data(), A_.data(), A_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<i_t> gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices_host() const
{
  std::vector<i_t> host_data(A_indices_.size());
  raft::copy(host_data.data(), A_indices_.data(), A_indices_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<i_t> gpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets_host() const
{
  std::vector<i_t> host_data(A_offsets_.size());
  raft::copy(host_data.data(), A_offsets_.data(), A_offsets_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> gpu_optimization_problem_t<i_t, f_t>::get_constraint_bounds_host() const
{
  std::vector<f_t> host_data(b_.size());
  raft::copy(host_data.data(), b_.data(), b_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> gpu_optimization_problem_t<i_t, f_t>::get_objective_coefficients_host() const
{
  std::vector<f_t> host_data(c_.size());
  raft::copy(host_data.data(), c_.data(), c_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> gpu_optimization_problem_t<i_t, f_t>::get_variable_lower_bounds_host() const
{
  std::vector<f_t> host_data(variable_lower_bounds_.size());
  raft::copy(
    host_data.data(), variable_lower_bounds_.data(), variable_lower_bounds_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> gpu_optimization_problem_t<i_t, f_t>::get_variable_upper_bounds_host() const
{
  std::vector<f_t> host_data(variable_upper_bounds_.size());
  raft::copy(
    host_data.data(), variable_upper_bounds_.data(), variable_upper_bounds_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> gpu_optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds_host() const
{
  std::vector<f_t> host_data(constraint_lower_bounds_.size());
  raft::copy(host_data.data(),
             constraint_lower_bounds_.data(),
             constraint_lower_bounds_.size(),
             stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> gpu_optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds_host() const
{
  std::vector<f_t> host_data(constraint_upper_bounds_.size());
  raft::copy(host_data.data(),
             constraint_upper_bounds_.data(),
             constraint_upper_bounds_.size(),
             stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<char> gpu_optimization_problem_t<i_t, f_t>::get_row_types_host() const
{
  std::vector<char> host_data(row_types_.size());
  raft::copy(host_data.data(), row_types_.data(), row_types_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<var_t> gpu_optimization_problem_t<i_t, f_t>::get_variable_types_host() const
{
  std::vector<var_t> host_data(variable_types_.size());
  raft::copy(host_data.data(), variable_types_.data(), variable_types_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

// ==============================================================================
// Conversion to optimization_problem_t
// ==============================================================================

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> gpu_optimization_problem_t<i_t, f_t>::to_optimization_problem()
{
  // NOTE: This method copies device vectors rather than moving them so that the
  // gpu_optimization_problem_t remains valid and the problem can be solved multiple
  // times. The copies are device-to-device at full GPU memory bandwidth and are
  // negligible compared to solve time. A subsequent refactor will make
  // optimization_problem_t inherit from the interface directly, eliminating the
  // need for this conversion entirely.

  optimization_problem_t<i_t, f_t> problem(handle_ptr_);

  // Set scalar values
  problem.set_maximize(maximize_);
  problem.set_objective_scaling_factor(objective_scaling_factor_);
  problem.set_objective_offset(objective_offset_);
  problem.set_problem_category(problem_category_);

  // Set string values
  if (!objective_name_.empty()) problem.set_objective_name(objective_name_);
  if (!problem_name_.empty()) problem.set_problem_name(problem_name_);
  if (!var_names_.empty()) problem.set_variable_names(var_names_);
  if (!row_names_.empty()) problem.set_row_names(row_names_);

  // Copy all device vectors (raft::copy handles device-to-device automatically)

  // Copy CSR constraint matrix
  if (A_.size() > 0) {
    problem.set_csr_constraint_matrix(A_.data(),
                                      A_.size(),
                                      A_indices_.data(),
                                      A_indices_.size(),
                                      A_offsets_.data(),
                                      A_offsets_.size());
  }

  // Copy constraint bounds
  if (b_.size() > 0) { problem.set_constraint_bounds(b_.data(), b_.size()); }

  // Copy objective coefficients
  if (c_.size() > 0) { problem.set_objective_coefficients(c_.data(), c_.size()); }

  // Set quadratic objective if present (stored in std::vector, not device_uvector)
  if (!Q_values_.empty()) {
    problem.set_quadratic_objective_matrix(Q_values_.data(),
                                           Q_values_.size(),
                                           Q_indices_.data(),
                                           Q_indices_.size(),
                                           Q_offsets_.data(),
                                           Q_offsets_.size());
  }

  // Copy variable bounds
  if (variable_lower_bounds_.size() > 0) {
    problem.set_variable_lower_bounds(variable_lower_bounds_.data(), variable_lower_bounds_.size());
  }
  if (variable_upper_bounds_.size() > 0) {
    problem.set_variable_upper_bounds(variable_upper_bounds_.data(), variable_upper_bounds_.size());
  }

  // Copy variable types
  if (variable_types_.size() > 0) {
    problem.set_variable_types(variable_types_.data(), variable_types_.size());
  }

  // Copy constraint lower/upper bounds
  if (constraint_lower_bounds_.size() > 0) {
    problem.set_constraint_lower_bounds(constraint_lower_bounds_.data(),
                                        constraint_lower_bounds_.size());
  }
  if (constraint_upper_bounds_.size() > 0) {
    problem.set_constraint_upper_bounds(constraint_upper_bounds_.data(),
                                        constraint_upper_bounds_.size());
  }

  // Copy row types
  if (row_types_.size() > 0) { problem.set_row_types(row_types_.data(), row_types_.size()); }

  return problem;
}

// ==============================================================================
// File I/O
// ==============================================================================

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::write_to_mps(const std::string& mps_file_path)
{
  // Use the existing host getters to get data, then write to MPS
  cuopt::mps_parser::data_model_view_t<i_t, f_t> data_model_view;

  // Set optimization sense
  data_model_view.set_maximize(get_sense());

  // Copy to host using host getters
  auto constraint_matrix_values  = get_constraint_matrix_values_host();
  auto constraint_matrix_indices = get_constraint_matrix_indices_host();
  auto constraint_matrix_offsets = get_constraint_matrix_offsets_host();
  auto constraint_bounds         = get_constraint_bounds_host();
  auto objective_coefficients    = get_objective_coefficients_host();
  auto variable_lower_bounds     = get_variable_lower_bounds_host();
  auto variable_upper_bounds     = get_variable_upper_bounds_host();
  auto constraint_lower_bounds   = get_constraint_lower_bounds_host();
  auto constraint_upper_bounds   = get_constraint_upper_bounds_host();
  auto row_types                 = get_row_types_host();

  // Set constraint matrix in CSR format
  if (!constraint_matrix_values.empty()) {
    data_model_view.set_csr_constraint_matrix(constraint_matrix_values.data(),
                                              constraint_matrix_values.size(),
                                              constraint_matrix_indices.data(),
                                              constraint_matrix_indices.size(),
                                              constraint_matrix_offsets.data(),
                                              constraint_matrix_offsets.size());
  }

  // Set constraint bounds (RHS)
  if (!constraint_bounds.empty()) {
    data_model_view.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
  }

  // Set objective coefficients
  if (!objective_coefficients.empty()) {
    data_model_view.set_objective_coefficients(objective_coefficients.data(),
                                               objective_coefficients.size());
  }

  // Set objective scaling and offset
  data_model_view.set_objective_scaling_factor(objective_scaling_factor_);
  data_model_view.set_objective_offset(objective_offset_);

  // Set variable bounds
  if (!variable_lower_bounds.empty()) {
    data_model_view.set_variable_lower_bounds(variable_lower_bounds.data(),
                                              variable_lower_bounds.size());
  }
  if (!variable_upper_bounds.empty()) {
    data_model_view.set_variable_upper_bounds(variable_upper_bounds.data(),
                                              variable_upper_bounds.size());
  }

  // Set row types (constraint types)
  if (!row_types.empty()) { data_model_view.set_row_types(row_types.data(), row_types.size()); }

  // Set constraint bounds (independently, a problem may have only one side)
  if (!constraint_lower_bounds.empty()) {
    data_model_view.set_constraint_lower_bounds(constraint_lower_bounds.data(),
                                                constraint_lower_bounds.size());
  }
  if (!constraint_upper_bounds.empty()) {
    data_model_view.set_constraint_upper_bounds(constraint_upper_bounds.data(),
                                                constraint_upper_bounds.size());
  }

  // Set problem and variable names FIRST (before variable types)
  if (!problem_name_.empty()) { data_model_view.set_problem_name(problem_name_); }
  if (!objective_name_.empty()) { data_model_view.set_objective_name(objective_name_); }
  if (!var_names_.empty()) { data_model_view.set_variable_names(var_names_); }
  if (!row_names_.empty()) { data_model_view.set_row_names(row_names_); }

  // Set variable types AFTER names (convert from enum to char)
  // CRITICAL: Declare variable_types OUTSIDE the if block so it stays alive
  // until after write_mps() is called, since data_model_view stores a span (pointer) to it
  std::vector<char> variable_types;
  if (n_vars_ > 0) {
    auto enum_variable_types = get_variable_types_host();
    variable_types.resize(enum_variable_types.size());

    for (size_t i = 0; i < variable_types.size(); ++i) {
      variable_types[i] = (enum_variable_types[i] == var_t::INTEGER) ? 'I' : 'C';
    }

    data_model_view.set_variable_types(variable_types.data(), variable_types.size());
  }

  cuopt::mps_parser::write_mps(data_model_view, mps_file_path);
}

// ==============================================================================
// Comparison
// ==============================================================================

template <typename i_t, typename f_t>
bool gpu_optimization_problem_t<i_t, f_t>::is_equivalent(
  const optimization_problem_interface_t<i_t, f_t>& other) const
{
  // Compare scalar properties
  if (maximize_ != other.get_sense()) return false;
  if (n_vars_ != other.get_n_variables()) return false;
  if (n_constraints_ != other.get_n_constraints()) return false;
  if (std::abs(objective_scaling_factor_ - other.get_objective_scaling_factor()) > 1e-9)
    return false;
  if (std::abs(objective_offset_ - other.get_objective_offset()) > 1e-9) return false;
  if (problem_category_ != other.get_problem_category()) return false;

  // Get host data from both problems
  auto this_c  = get_objective_coefficients_host();
  auto other_c = other.get_objective_coefficients_host();
  if (this_c.size() != other_c.size()) return false;

  auto this_var_lb  = get_variable_lower_bounds_host();
  auto other_var_lb = other.get_variable_lower_bounds_host();
  if (this_var_lb.size() != other_var_lb.size()) return false;

  auto this_var_ub  = get_variable_upper_bounds_host();
  auto other_var_ub = other.get_variable_upper_bounds_host();
  if (this_var_ub.size() != other_var_ub.size()) return false;

  auto this_var_types  = get_variable_types_host();
  auto other_var_types = other.get_variable_types_host();
  if (this_var_types.size() != other_var_types.size()) return false;

  auto this_b  = get_constraint_bounds_host();
  auto other_b = other.get_constraint_bounds_host();
  if (this_b.size() != other_b.size()) return false;

  auto this_A_values  = get_constraint_matrix_values_host();
  auto other_A_values = other.get_constraint_matrix_values_host();
  if (this_A_values.size() != other_A_values.size()) return false;

  // Check if we have variable and row names for permutation matching
  const auto& this_var_names  = get_variable_names();
  const auto& other_var_names = other.get_variable_names();
  const auto& this_row_names  = get_row_names();
  const auto& other_row_names = other.get_row_names();

  bool has_names = !this_var_names.empty() && !other_var_names.empty() && !this_row_names.empty() &&
                   !other_row_names.empty();

  // If no names, fall back to direct-order comparison
  if (!has_names) {
    for (i_t i = 0; i < n_vars_; ++i) {
      if (std::abs(this_c[i] - other_c[i]) > 1e-9) return false;
      if (std::abs(this_var_lb[i] - other_var_lb[i]) > 1e-9) return false;
      if (std::abs(this_var_ub[i] - other_var_ub[i]) > 1e-9) return false;
      if (this_var_types[i] != other_var_types[i]) return false;
    }
    for (i_t i = 0; i < n_constraints_; ++i) {
      if (std::abs(this_b[i] - other_b[i]) > 1e-9) return false;
    }
    // Direct CSR comparison without permutation
    auto this_A_indices  = get_constraint_matrix_indices_host();
    auto other_A_indices = other.get_constraint_matrix_indices_host();
    auto this_A_offsets  = get_constraint_matrix_offsets_host();
    auto other_A_offsets = other.get_constraint_matrix_offsets_host();
    if (this_A_indices.size() != other_A_indices.size()) return false;
    if (this_A_offsets.size() != other_A_offsets.size()) return false;
    for (size_t i = 0; i < this_A_values.size(); ++i) {
      if (std::abs(this_A_values[i] - other_A_values[i]) > 1e-9) return false;
    }
    for (size_t i = 0; i < this_A_indices.size(); ++i) {
      if (this_A_indices[i] != other_A_indices[i]) return false;
    }
    for (size_t i = 0; i < this_A_offsets.size(); ++i) {
      if (this_A_offsets[i] != other_A_offsets[i]) return false;
    }
    return true;
  }

  // Build variable permutation map
  std::unordered_map<std::string, i_t> other_var_idx;
  for (size_t j = 0; j < other_var_names.size(); ++j) {
    other_var_idx[other_var_names[j]] = static_cast<i_t>(j);
  }

  std::vector<i_t> var_perm(n_vars_);
  for (i_t i = 0; i < n_vars_; ++i) {
    auto it = other_var_idx.find(this_var_names[i]);
    if (it == other_var_idx.end()) return false;
    var_perm[i] = it->second;
  }

  // Build row permutation map
  std::unordered_map<std::string, i_t> other_row_idx;
  for (size_t j = 0; j < other_row_names.size(); ++j) {
    other_row_idx[other_row_names[j]] = static_cast<i_t>(j);
  }

  std::vector<i_t> row_perm(n_constraints_);
  for (i_t i = 0; i < n_constraints_; ++i) {
    auto it = other_row_idx.find(this_row_names[i]);
    if (it == other_row_idx.end()) return false;
    row_perm[i] = it->second;
  }

  // Compare variable-indexed arrays with permutation
  for (i_t i = 0; i < n_vars_; ++i) {
    i_t j = var_perm[i];
    if (std::abs(this_c[i] - other_c[j]) > 1e-9) return false;
    if (std::abs(this_var_lb[i] - other_var_lb[j]) > 1e-9) return false;
    if (std::abs(this_var_ub[i] - other_var_ub[j]) > 1e-9) return false;
    if (this_var_types[i] != other_var_types[j]) return false;
  }

  // Compare constraint-indexed arrays with permutation
  for (i_t i = 0; i < n_constraints_; ++i) {
    i_t j = row_perm[i];
    if (std::abs(this_b[i] - other_b[j]) > 1e-9) return false;
  }

  // Compare constraint lower/upper bounds with permutation
  auto this_clb  = get_constraint_lower_bounds_host();
  auto other_clb = other.get_constraint_lower_bounds_host();
  if (this_clb.size() != other_clb.size()) return false;
  for (i_t i = 0; i < n_constraints_ && i < static_cast<i_t>(this_clb.size()); ++i) {
    i_t j = row_perm[i];
    if (std::abs(this_clb[i] - other_clb[j]) > 1e-9) return false;
  }

  auto this_cub  = get_constraint_upper_bounds_host();
  auto other_cub = other.get_constraint_upper_bounds_host();
  if (this_cub.size() != other_cub.size()) return false;
  for (i_t i = 0; i < n_constraints_ && i < static_cast<i_t>(this_cub.size()); ++i) {
    i_t j = row_perm[i];
    if (std::abs(this_cub[i] - other_cub[j]) > 1e-9) return false;
  }

  // Compare row types with permutation
  auto this_rt  = get_row_types_host();
  auto other_rt = other.get_row_types_host();
  if (this_rt.size() != other_rt.size()) return false;
  for (i_t i = 0; i < n_constraints_ && i < static_cast<i_t>(this_rt.size()); ++i) {
    i_t j = row_perm[i];
    if (this_rt[i] != other_rt[j]) return false;
  }

  // Compare CSR constraint matrix with row/column permutations
  auto this_A_indices  = get_constraint_matrix_indices_host();
  auto other_A_indices = other.get_constraint_matrix_indices_host();
  auto this_A_offsets  = get_constraint_matrix_offsets_host();
  auto other_A_offsets = other.get_constraint_matrix_offsets_host();

  if (!csr_matrices_equivalent_with_permutation_host(this_A_offsets,
                                                     this_A_indices,
                                                     this_A_values,
                                                     other_A_offsets,
                                                     other_A_indices,
                                                     other_A_values,
                                                     row_perm,
                                                     var_perm)) {
    return false;
  }

  return true;
}

// ==============================================================================
// Remote Execution (Polymorphic Dispatch)
// ==============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>>
gpu_optimization_problem_t<i_t, f_t>::solve_lp_remote(
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode) const
{
  return ::cuopt::linear_programming::solve_lp_remote(
    *this, settings, problem_checking, use_pdlp_solver_mode);
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>>
gpu_optimization_problem_t<i_t, f_t>::solve_mip_remote(
  mip_solver_settings_t<i_t, f_t> const& settings) const
{
  return ::cuopt::linear_programming::solve_mip_remote(*this, settings);
}

// ==============================================================================
// C API Support: Copy to Host (GPU Implementation)
// ==============================================================================

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_objective_coefficients_to_host(f_t* output,
                                                                               i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(c_.size()),
                error_type_t::ValidationError,
                "copy_objective_coefficients_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(cudaMemcpy(output, c_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_constraint_matrix_to_host(
  f_t* values, i_t* indices, i_t* offsets, i_t num_values, i_t num_indices, i_t num_offsets) const
{
  cuopt_expects(values != nullptr && num_values <= static_cast<i_t>(A_.size()),
                error_type_t::ValidationError,
                "copy_constraint_matrix_to_host: null values or size exceeds buffer");
  cuopt_expects(indices != nullptr && num_indices <= static_cast<i_t>(A_indices_.size()),
                error_type_t::ValidationError,
                "copy_constraint_matrix_to_host: null indices or size exceeds buffer");
  cuopt_expects(offsets != nullptr && num_offsets <= static_cast<i_t>(A_offsets_.size()),
                error_type_t::ValidationError,
                "copy_constraint_matrix_to_host: null offsets or size exceeds buffer");
  RAFT_CUDA_TRY(cudaMemcpy(values, A_.data(), num_values * sizeof(f_t), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(
    cudaMemcpy(indices, A_indices_.data(), num_indices * sizeof(i_t), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(
    cudaMemcpy(offsets, A_offsets_.data(), num_offsets * sizeof(i_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_row_types_to_host(char* output, i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(row_types_.size()),
                error_type_t::ValidationError,
                "copy_row_types_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(cudaMemcpy(output, row_types_.data(), size * sizeof(char), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_constraint_bounds_to_host(f_t* output,
                                                                          i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(b_.size()),
                error_type_t::ValidationError,
                "copy_constraint_bounds_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(cudaMemcpy(output, b_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_constraint_lower_bounds_to_host(f_t* output,
                                                                                i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(constraint_lower_bounds_.size()),
                error_type_t::ValidationError,
                "copy_constraint_lower_bounds_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(cudaMemcpy(
    output, constraint_lower_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_constraint_upper_bounds_to_host(f_t* output,
                                                                                i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(constraint_upper_bounds_.size()),
                error_type_t::ValidationError,
                "copy_constraint_upper_bounds_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(cudaMemcpy(
    output, constraint_upper_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_variable_lower_bounds_to_host(f_t* output,
                                                                              i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(variable_lower_bounds_.size()),
                error_type_t::ValidationError,
                "copy_variable_lower_bounds_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(
    cudaMemcpy(output, variable_lower_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_variable_upper_bounds_to_host(f_t* output,
                                                                              i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(variable_upper_bounds_.size()),
                error_type_t::ValidationError,
                "copy_variable_upper_bounds_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(
    cudaMemcpy(output, variable_upper_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void gpu_optimization_problem_t<i_t, f_t>::copy_variable_types_to_host(var_t* output,
                                                                       i_t size) const
{
  cuopt_expects(output != nullptr && size <= static_cast<i_t>(variable_types_.size()),
                error_type_t::ValidationError,
                "copy_variable_types_to_host: null output or size exceeds buffer");
  RAFT_CUDA_TRY(
    cudaMemcpy(output, variable_types_.data(), size * sizeof(var_t), cudaMemcpyDeviceToHost));
}

// ==============================================================================
// Template instantiations
// ==============================================================================
// Explicit template instantiations matching optimization_problem_t
#if MIP_INSTANTIATE_FLOAT
template class gpu_optimization_problem_t<int32_t, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class gpu_optimization_problem_t<int32_t, double>;
#endif

}  // namespace cuopt::linear_programming
