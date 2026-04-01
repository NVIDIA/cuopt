/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve_remote.hpp>

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/csr_matrix_utils.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mps_parser/writer.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/cuda_support.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <cuda_profiler_api.h>
#include <cub/device/device_transform.cuh>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t>::optimization_problem_t(raft::handle_t const* handle_ptr)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr != nullptr ? handle_ptr->get_stream() : rmm::cuda_stream_view{}),
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
    Q_offsets_{other.get_quadratic_objective_offsets()},
    Q_indices_{other.get_quadratic_objective_indices()},
    Q_values_{other.get_quadratic_objective_values()},
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

// ==============================================================================
// Setters
// ==============================================================================

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_maximize(bool maximize)
{
  maximize_ = maximize;
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
  n_constraints_ = size_offsets == 0 ? 0 : size_offsets - 1;
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
  n_vars_ = size;
  c_.resize(size, stream_view_);
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
void optimization_problem_t<i_t, f_t>::set_quadratic_objective_matrix(
  const f_t* Q_values,
  i_t size_values,
  const i_t* Q_indices,
  i_t size_indices,
  const i_t* Q_offsets,
  i_t size_offsets,
  bool validate_positive_semi_definite)
{
  cuopt_expects(Q_values != nullptr, error_type_t::ValidationError, "Q_values cannot be null");
  cuopt_expects(
    size_values > 0, error_type_t::ValidationError, "size_values must be greater than 0");

  if (size_indices != 0) {
    cuopt_expects(Q_indices != nullptr, error_type_t::ValidationError, "Q_indices cannot be null");
  }

  cuopt_expects(
    size_offsets >= 1, error_type_t::ValidationError, "Q_offsets must have at least 1 element");
  cuopt_expects(Q_offsets != nullptr, error_type_t::ValidationError, "Q_offsets cannot be null");

  // Replace Q with Q + Q^T
  i_t qn    = size_offsets - 1;  // Number of variables
  i_t q_nnz = size_indices;
  // Construct H = Q + Q^T in triplet form first
  std::vector<i_t> H_i;
  std::vector<i_t> H_j;
  std::vector<f_t> H_x;

  H_i.reserve(2 * q_nnz);
  H_j.reserve(2 * q_nnz);
  H_x.reserve(2 * q_nnz);

  for (i_t i = 0; i < qn; ++i) {
    i_t row_start = Q_offsets[i];
    i_t row_end   = Q_offsets[i + 1];
    for (i_t p = row_start; p < row_end; ++p) {
      i_t j = Q_indices[p];
      f_t x = Q_values[p];
      // Add H(i,j)
      H_i.push_back(i);
      H_j.push_back(j);
      if (i == j) { H_x.push_back(2 * x); }
      if (i != j) {
        H_x.push_back(x);
        // Add H(j,i)
        H_i.push_back(j);
        H_j.push_back(i);
        H_x.push_back(x);
      }
    }
  }
  // Convert H to CSR format
  // Get row counts
  i_t H_nz = H_x.size();
  std::vector<i_t> H_row_counts(qn, 0);
  for (i_t k = 0; k < H_nz; ++k) {
    H_row_counts[H_i[k]]++;
  }
  std::vector<i_t> H_cumulative_counts(qn + 1, 0);
  for (i_t k = 0; k < qn; ++k) {
    H_cumulative_counts[k + 1] = H_cumulative_counts[k] + H_row_counts[k];
  }
  std::vector<i_t> H_row_starts = H_cumulative_counts;
  std::vector<i_t> H_indices(H_nz);
  std::vector<f_t> H_values(H_nz);
  for (i_t k = 0; k < H_nz; ++k) {
    i_t p        = H_cumulative_counts[H_i[k]]++;
    H_indices[p] = H_j[k];
    H_values[p]  = H_x[k];
  }

  // H_row_starts, H_indices, H_values are the CSR representation of H
  // But this contains duplicate entries

  std::vector<i_t> workspace(qn, -1);
  Q_offsets_.resize(qn + 1);
  std::fill(Q_offsets_.begin(), Q_offsets_.end(), 0);
  Q_indices_.resize(H_nz);
  Q_values_.resize(H_nz);
  i_t nz = 0;
  for (i_t i = 0; i < qn; ++i) {
    i_t q               = nz;  // row i will start at q
    const i_t row_start = H_row_starts[i];
    const i_t row_end   = H_row_starts[i + 1];
    for (i_t p = row_start; p < row_end; ++p) {
      i_t j = H_indices[p];
      if (workspace[j] >= q) {
        Q_values_[workspace[j]] += H_values[p];  // H(i,j) is duplicate
      } else {
        workspace[j]   = nz;  // record where column j occurs
        Q_indices_[nz] = j;   // keep H(i,j)
        Q_values_[nz]  = H_values[p];
        nz++;
      }
    }
    Q_offsets_[i] = q;  // record start of row i
  }

  Q_offsets_[qn] = nz;  // finalize Q
  Q_indices_.resize(nz);
  Q_values_.resize(nz);
  // FIX ME:: check for positive semi definite matrix
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
void optimization_problem_t<i_t, f_t>::set_variable_types(const var_t* variable_types, i_t size)
{
  cuopt_expects(
    variable_types != nullptr, error_type_t::ValidationError, "variable_types cannot be null");
  variable_types_.resize(size, stream_view_);
  raft::copy(variable_types_.data(), variable_types, size, stream_view_);

  // Auto-detect problem category based on variable types
  i_t n_integer = thrust::count_if(handle_ptr_->get_thrust_policy(),
                                   variable_types_.begin(),
                                   variable_types_.end(),
                                   [] __device__(auto val) { return val == var_t::INTEGER; });
  if (n_integer == size) {
    problem_category_ = problem_category_t::IP;
  } else if (n_integer > 0) {
    problem_category_ = problem_category_t::MIP;
  } else {
    problem_category_ = problem_category_t::LP;
  }
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::set_problem_category(const problem_category_t& category)
{
  problem_category_ = category;
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

// ==============================================================================
// Device Getters
// ==============================================================================

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
i_t optimization_problem_t<i_t, f_t>::get_n_integers() const
{
  if (variable_types_.size() == 0) return 0;
  return thrust::count(
    rmm::exec_policy(stream_view_), variable_types_.begin(), variable_types_.end(), var_t::INTEGER);
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
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_variable_lower_bounds()
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_variable_upper_bounds() const
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_variable_upper_bounds()
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
  const
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
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
const rmm::device_uvector<var_t>& optimization_problem_t<i_t, f_t>::get_variable_types() const
{
  return variable_types_;
}

template <typename i_t, typename f_t>
bool optimization_problem_t<i_t, f_t>::get_sense() const
{
  return maximize_;
}

template <typename i_t, typename f_t>
bool optimization_problem_t<i_t, f_t>::empty() const
{
  return n_vars_ == 0 || n_constraints_ == 0;
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
const std::vector<i_t>& optimization_problem_t<i_t, f_t>::get_quadratic_objective_offsets() const
{
  return Q_offsets_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& optimization_problem_t<i_t, f_t>::get_quadratic_objective_indices() const
{
  return Q_indices_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& optimization_problem_t<i_t, f_t>::get_quadratic_objective_values() const
{
  return Q_values_;
}

template <typename i_t, typename f_t>
bool optimization_problem_t<i_t, f_t>::has_quadratic_objective() const
{
  return !Q_values_.empty();
}

template <typename i_t, typename f_t>
raft::handle_t const* optimization_problem_t<i_t, f_t>::get_handle_ptr() const noexcept
{
  return handle_ptr_;
}

// ==============================================================================
// Conversion
// ==============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<optimization_problem_t<i_t, f_t>>
optimization_problem_t<i_t, f_t>::to_optimization_problem(raft::handle_t const* /*handle_ptr*/)
{
  // Already a GPU problem, return nullptr
  return nullptr;
}

// ==============================================================================
// Host Getters (copy from GPU to CPU)
// ==============================================================================

template <typename i_t, typename f_t>
std::vector<f_t> optimization_problem_t<i_t, f_t>::get_constraint_matrix_values_host() const
{
  std::vector<f_t> host_data(A_.size());
  raft::copy(host_data.data(), A_.data(), A_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<i_t> optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices_host() const
{
  std::vector<i_t> host_data(A_indices_.size());
  raft::copy(host_data.data(), A_indices_.data(), A_indices_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<i_t> optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets_host() const
{
  std::vector<i_t> host_data(A_offsets_.size());
  raft::copy(host_data.data(), A_offsets_.data(), A_offsets_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> optimization_problem_t<i_t, f_t>::get_constraint_bounds_host() const
{
  std::vector<f_t> host_data(b_.size());
  raft::copy(host_data.data(), b_.data(), b_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> optimization_problem_t<i_t, f_t>::get_objective_coefficients_host() const
{
  std::vector<f_t> host_data(c_.size());
  raft::copy(host_data.data(), c_.data(), c_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> optimization_problem_t<i_t, f_t>::get_variable_lower_bounds_host() const
{
  std::vector<f_t> host_data(variable_lower_bounds_.size());
  raft::copy(
    host_data.data(), variable_lower_bounds_.data(), variable_lower_bounds_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> optimization_problem_t<i_t, f_t>::get_variable_upper_bounds_host() const
{
  std::vector<f_t> host_data(variable_upper_bounds_.size());
  raft::copy(
    host_data.data(), variable_upper_bounds_.data(), variable_upper_bounds_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<f_t> optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds_host() const
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
std::vector<f_t> optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds_host() const
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
std::vector<char> optimization_problem_t<i_t, f_t>::get_row_types_host() const
{
  std::vector<char> host_data(row_types_.size());
  raft::copy(host_data.data(), row_types_.data(), row_types_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

template <typename i_t, typename f_t>
std::vector<var_t> optimization_problem_t<i_t, f_t>::get_variable_types_host() const
{
  std::vector<var_t> host_data(variable_types_.size());
  raft::copy(host_data.data(), variable_types_.data(), variable_types_.size(), stream_view_);
  handle_ptr_->sync_stream();
  return host_data;
}

// ==============================================================================
// View
// ==============================================================================

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
  v.c                     = raft::device_span<const f_t>{get_objective_coefficients().data(),
                                                         get_objective_coefficients().size()};
  v.variable_lower_bounds = raft::device_span<const f_t>{get_variable_lower_bounds().data(),
                                                         get_variable_lower_bounds().size()};
  v.variable_upper_bounds = raft::device_span<const f_t>{get_variable_upper_bounds().data(),
                                                         get_variable_upper_bounds().size()};
  v.variable_types =
    raft::device_span<const var_t>{get_variable_types().data(), get_variable_types().size()};
  v.constraint_lower_bounds = raft::device_span<const f_t>{get_constraint_lower_bounds().data(),
                                                           get_constraint_lower_bounds().size()};
  v.constraint_upper_bounds = raft::device_span<const f_t>{get_constraint_upper_bounds().data(),
                                                           get_constraint_upper_bounds().size()};
  return v;
}

// ==============================================================================
// File I/O
// ==============================================================================

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::write_to_mps(const std::string& mps_file_path)
{
  cuopt::mps_parser::data_model_view_t<i_t, f_t> data_model_view;

  // Set optimization sense
  data_model_view.set_maximize(get_sense());

  // Copy to host
  auto stream                    = handle_ptr_->get_stream();
  auto constraint_matrix_values  = cuopt::host_copy(get_constraint_matrix_values(), stream);
  auto constraint_matrix_indices = cuopt::host_copy(get_constraint_matrix_indices(), stream);
  auto constraint_matrix_offsets = cuopt::host_copy(get_constraint_matrix_offsets(), stream);
  auto constraint_bounds         = cuopt::host_copy(get_constraint_bounds(), stream);
  auto objective_coefficients    = cuopt::host_copy(get_objective_coefficients(), stream);
  auto variable_lower_bounds     = cuopt::host_copy(get_variable_lower_bounds(), stream);
  auto variable_upper_bounds     = cuopt::host_copy(get_variable_upper_bounds(), stream);
  auto constraint_lower_bounds   = cuopt::host_copy(get_constraint_lower_bounds(), stream);
  auto constraint_upper_bounds   = cuopt::host_copy(get_constraint_upper_bounds(), stream);
  auto row_types                 = cuopt::host_copy(get_row_types(), stream);

  // Set constraint matrix in CSR format (guard on offsets, not nnz, to preserve
  // zero-nnz but structurally valid matrices)
  if (!constraint_matrix_offsets.empty()) {
    data_model_view.set_csr_constraint_matrix(constraint_matrix_values.data(),
                                              constraint_matrix_values.size(),
                                              constraint_matrix_indices.data(),
                                              constraint_matrix_indices.size(),
                                              constraint_matrix_offsets.data(),
                                              constraint_matrix_offsets.size());
  }

  // Set constraint bounds (RHS)
  if (get_n_constraints() != 0) {
    data_model_view.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
  }

  // Set objective coefficients
  if (get_n_variables() != 0) {
    data_model_view.set_objective_coefficients(objective_coefficients.data(),
                                               objective_coefficients.size());
  }

  // Set objective scaling and offset
  data_model_view.set_objective_scaling_factor(get_objective_scaling_factor());
  data_model_view.set_objective_offset(get_objective_offset());

  // Set variable bounds
  if (get_n_variables() != 0) {
    data_model_view.set_variable_lower_bounds(variable_lower_bounds.data(),
                                              variable_lower_bounds.size());
  }
  if (!variable_upper_bounds.empty()) {
    data_model_view.set_variable_upper_bounds(variable_upper_bounds.data(),
                                              variable_upper_bounds.size());
  }

  // Set row types (constraint types)
  if (get_row_types().size() != 0) {
    data_model_view.set_row_types(row_types.data(), row_types.size());
  }

  if (!constraint_lower_bounds.empty()) {
    data_model_view.set_constraint_lower_bounds(constraint_lower_bounds.data(),
                                                constraint_lower_bounds.size());
  }
  if (!constraint_upper_bounds.empty()) {
    data_model_view.set_constraint_upper_bounds(constraint_upper_bounds.data(),
                                                constraint_upper_bounds.size());
  }

  // Set problem and variable names FIRST (before variable types)
  if (!get_problem_name().empty()) { data_model_view.set_problem_name(get_problem_name()); }
  if (!get_objective_name().empty()) { data_model_view.set_objective_name(get_objective_name()); }
  if (!get_variable_names().empty()) { data_model_view.set_variable_names(get_variable_names()); }
  if (!get_row_names().empty()) { data_model_view.set_row_names(get_row_names()); }

  // Set variable types AFTER names (convert from enum to char)
  // CRITICAL: Declare variable_types OUTSIDE the if block so it stays alive
  // until after write_mps() is called, since data_model_view stores a span (pointer) to it
  std::vector<char> variable_types;
  if (get_n_variables() != 0) {
    auto enum_variable_types = cuopt::host_copy(get_variable_types(), stream);
    variable_types.resize(enum_variable_types.size());

    // Convert enum types to char types
    for (size_t i = 0; i < variable_types.size(); ++i) {
      variable_types[i] = (enum_variable_types[i] == var_t::INTEGER) ? 'I' : 'C';
    }

    data_model_view.set_variable_types(variable_types.data(), variable_types.size());
  }

  cuopt::mps_parser::write_mps(data_model_view, mps_file_path);
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::print_scaling_information() const
{
  auto stream = handle_ptr_->get_stream();
  std::vector<f_t> constraint_matrix_values =
    cuopt::host_copy(get_constraint_matrix_values(), stream);
  std::vector<f_t> constraint_rhs         = cuopt::host_copy(get_constraint_bounds(), stream);
  std::vector<f_t> objective_coefficients = cuopt::host_copy(get_objective_coefficients(), stream);
  std::vector<f_t> variable_lower_bounds  = cuopt::host_copy(get_variable_lower_bounds(), stream);
  std::vector<f_t> variable_upper_bounds  = cuopt::host_copy(get_variable_upper_bounds(), stream);
  std::vector<f_t> constraint_lower_bounds =
    cuopt::host_copy(get_constraint_lower_bounds(), stream);
  std::vector<f_t> constraint_upper_bounds =
    cuopt::host_copy(get_constraint_upper_bounds(), stream);

  auto findMaxAbs = [](const std::vector<f_t>& vec) -> f_t {
    if (vec.empty()) { return 0.0; }
    const f_t inf = std::numeric_limits<f_t>::infinity();

    const size_t sz = vec.size();
    f_t max_abs_val = f_t(0.0);
    for (size_t i = 0; i < sz; ++i) {
      const f_t val = std::abs(vec[i]);
      if (val < inf) { max_abs_val = std::max(max_abs_val, val); }
    }
    return max_abs_val;
  };

  auto findMinAbs = [](const std::vector<f_t>& vec) -> f_t {
    if (vec.empty()) { return f_t(0.0); }
    const size_t sz = vec.size();
    const f_t inf   = std::numeric_limits<f_t>::infinity();
    f_t min_abs_val = inf;
    for (size_t i = 0; i < sz; ++i) {
      const f_t val = std::abs(vec[i]);
      if (val > f_t(0.0)) { min_abs_val = std::min(min_abs_val, val); }
    }
    return min_abs_val < inf ? min_abs_val : f_t(0.0);
  };

  f_t A_max          = findMaxAbs(constraint_matrix_values);
  f_t A_min          = findMinAbs(constraint_matrix_values);
  f_t b_max          = findMaxAbs(constraint_rhs);
  f_t b_min          = findMinAbs(constraint_rhs);
  f_t c_max          = findMaxAbs(objective_coefficients);
  f_t c_min          = findMinAbs(objective_coefficients);
  f_t x_lower_max    = findMaxAbs(variable_lower_bounds);
  f_t x_lower_min    = findMinAbs(variable_lower_bounds);
  f_t x_upper_max    = findMaxAbs(variable_upper_bounds);
  f_t x_upper_min    = findMinAbs(variable_upper_bounds);
  f_t cstr_lower_max = findMaxAbs(constraint_lower_bounds);
  f_t cstr_lower_min = findMinAbs(constraint_lower_bounds);
  f_t cstr_upper_max = findMaxAbs(constraint_upper_bounds);
  f_t cstr_upper_min = findMinAbs(constraint_upper_bounds);

  f_t rhs_max = std::max(b_max, std::max(cstr_lower_max, cstr_upper_max));
  f_t rhs_min = std::min(b_min, std::min(cstr_lower_min, cstr_upper_min));

  f_t bound_max = std::max(x_upper_max, x_lower_max);
  f_t bound_min = std::min(x_upper_min, x_lower_min);

  CUOPT_LOG_INFO("Problem scaling:");
  CUOPT_LOG_INFO("Objective coefficents range:          [%.0e, %.0e]", c_min, c_max);
  CUOPT_LOG_INFO("Constraint matrix coefficients range: [%.0e, %.0e]", A_min, A_max);
  CUOPT_LOG_INFO("Constraint rhs / bounds range:        [%.0e, %.0e]", rhs_min, rhs_max);
  CUOPT_LOG_INFO("Variable bounds range:                [%.0e, %.0e]", bound_min, bound_max);

  auto safelog10 = [](f_t x) { return x > 0 ? std::log10(x) : 0.0; };

  f_t obj_range   = safelog10(c_max) - safelog10(c_min);
  f_t A_range     = safelog10(A_max) - safelog10(A_min);
  f_t rhs_range   = safelog10(rhs_max) - safelog10(rhs_min);
  f_t bound_range = safelog10(bound_max) - safelog10(bound_min);

  if (obj_range >= 6.0 || A_range >= 6.0 || rhs_range >= 6.0 || bound_range >= 6.0) {
    CUOPT_LOG_INFO(
      "Warning: input problem contains a large range of coefficients: consider reformulating to "
      "avoid numerical difficulties.");
  }
  CUOPT_LOG_INFO("");
}

// ==============================================================================
// Comparison
// ==============================================================================

/**
 * @brief Compare two CSR matrices for equivalence under row and column permutations.
 *
 * @param this_offsets Row offsets of first matrix
 * @param this_indices Column indices of first matrix
 * @param this_values Values of first matrix
 * @param other_offsets Row offsets of second matrix
 * @param other_indices Column indices of second matrix
 * @param other_values Values of second matrix
 * @param d_row_perm_inv Inverse row permutation (maps other's row indices to this's)
 * @param d_col_perm_inv Inverse column permutation (maps other's col indices to this's)
 * @param n_cols Number of columns (used for sort key computation)
 * @param stream CUDA stream
 * @return true if matrices are equivalent under the given permutations
 */
template <typename i_t, typename f_t>
static bool csr_matrices_equivalent_with_permutation(const rmm::device_uvector<i_t>& this_offsets,
                                                     const rmm::device_uvector<i_t>& this_indices,
                                                     const rmm::device_uvector<f_t>& this_values,
                                                     const rmm::device_uvector<i_t>& other_offsets,
                                                     const rmm::device_uvector<i_t>& other_indices,
                                                     const rmm::device_uvector<f_t>& other_values,
                                                     const rmm::device_uvector<i_t>& d_row_perm_inv,
                                                     const rmm::device_uvector<i_t>& d_col_perm_inv,
                                                     i_t n_cols,
                                                     rmm::cuda_stream_view stream)
{
  const i_t nnz = static_cast<i_t>(this_values.size());
  if (nnz != static_cast<i_t>(other_values.size())) { return false; }
  if (nnz == 0) { return true; }

  auto policy = rmm::exec_policy(stream);

  // Expand CSR row offsets to row indices for 'this'
  rmm::device_uvector<i_t> this_rows(nnz, stream);
  rmm::device_uvector<i_t> this_cols(nnz, stream);
  rmm::device_uvector<f_t> this_vals(nnz, stream);

  // upper_bound returns 1-based indices; convert to 0-based
  thrust::upper_bound(policy,
                      this_offsets.begin(),
                      this_offsets.end(),
                      thrust::make_counting_iterator<i_t>(0),
                      thrust::make_counting_iterator<i_t>(nnz),
                      this_rows.begin());
  thrust::transform(
    policy, this_rows.begin(), this_rows.end(), this_rows.begin(), [] __device__(i_t r) {
      return r - 1;
    });

  thrust::copy(policy, this_indices.begin(), this_indices.end(), this_cols.begin());
  thrust::copy(policy, this_values.begin(), this_values.end(), this_vals.begin());

  // For 'other': expand and apply inverse permutations to map to 'this' coordinate system
  rmm::device_uvector<i_t> other_rows(nnz, stream);
  rmm::device_uvector<i_t> other_cols(nnz, stream);
  rmm::device_uvector<f_t> other_vals(nnz, stream);

  thrust::upper_bound(policy,
                      other_offsets.begin(),
                      other_offsets.end(),
                      thrust::make_counting_iterator<i_t>(0),
                      thrust::make_counting_iterator<i_t>(nnz),
                      other_rows.begin());
  thrust::transform(
    policy, other_rows.begin(), other_rows.end(), other_rows.begin(), [] __device__(i_t r) {
      return r - 1;
    });

  thrust::gather(
    policy, other_rows.begin(), other_rows.end(), d_row_perm_inv.begin(), other_rows.begin());

  thrust::gather(
    policy, other_indices.begin(), other_indices.end(), d_col_perm_inv.begin(), other_cols.begin());

  thrust::copy(policy, other_values.begin(), other_values.end(), other_vals.begin());

  // Create sort keys: row * n_cols + col (to sort by row then column)
  rmm::device_uvector<int64_t> this_keys(nnz, stream);
  rmm::device_uvector<int64_t> other_keys(nnz, stream);

  const int64_t n_cols_64 = n_cols;
  thrust::transform(policy,
                    thrust::make_zip_iterator(this_rows.begin(), this_cols.begin()),
                    thrust::make_zip_iterator(this_rows.end(), this_cols.end()),
                    this_keys.begin(),
                    [n_cols_64] __device__(thrust::tuple<i_t, i_t> rc) {
                      return static_cast<int64_t>(thrust::get<0>(rc)) * n_cols_64 +
                             static_cast<int64_t>(thrust::get<1>(rc));
                    });

  thrust::transform(policy,
                    thrust::make_zip_iterator(other_rows.begin(), other_cols.begin()),
                    thrust::make_zip_iterator(other_rows.end(), other_cols.end()),
                    other_keys.begin(),
                    [n_cols_64] __device__(thrust::tuple<i_t, i_t> rc) {
                      return static_cast<int64_t>(thrust::get<0>(rc)) * n_cols_64 +
                             static_cast<int64_t>(thrust::get<1>(rc));
                    });

  thrust::sort_by_key(policy, this_keys.begin(), this_keys.end(), this_vals.begin());
  thrust::sort_by_key(policy, other_keys.begin(), other_keys.end(), other_vals.begin());

  if (!thrust::equal(policy, this_keys.begin(), this_keys.end(), other_keys.begin())) {
    return false;
  }

  if (!thrust::equal(policy, this_vals.begin(), this_vals.end(), other_vals.begin())) {
    return false;
  }

  return true;
}

template <typename i_t, typename f_t>
bool optimization_problem_t<i_t, f_t>::is_equivalent(
  const optimization_problem_t<i_t, f_t>& other) const
{
  if (maximize_ != other.maximize_) { return false; }
  if (n_vars_ != other.n_vars_) { return false; }
  if (n_constraints_ != other.n_constraints_) { return false; }
  if (objective_scaling_factor_ != other.objective_scaling_factor_) { return false; }
  if (objective_offset_ != other.objective_offset_) { return false; }
  if (problem_category_ != other.problem_category_) { return false; }
  if (A_.size() != other.A_.size()) { return false; }

  if (var_names_.size() != static_cast<size_t>(n_vars_) ||
      other.var_names_.size() != static_cast<size_t>(other.n_vars_)) {
    return false;
  }
  if (row_names_.size() != static_cast<size_t>(n_constraints_) ||
      other.row_names_.size() != static_cast<size_t>(other.n_constraints_)) {
    return false;
  }

  // Build variable permutation: var_perm[i] = index j in other where var_names_[i] ==
  // other.var_names_[j]
  std::unordered_map<std::string, i_t> other_var_idx;
  for (size_t j = 0; j < other.var_names_.size(); ++j) {
    other_var_idx[other.var_names_[j]] = static_cast<i_t>(j);
  }
  std::vector<i_t> var_perm(n_vars_);
  for (i_t i = 0; i < n_vars_; ++i) {
    auto it = other_var_idx.find(var_names_[i]);
    if (it == other_var_idx.end()) { return false; }
    var_perm[i] = it->second;
  }

  // Build row permutation: row_perm[i] = index j in other where row_names_[i] ==
  // other.row_names_[j]
  std::unordered_map<std::string, i_t> other_row_idx;
  for (size_t j = 0; j < other.row_names_.size(); ++j) {
    other_row_idx[other.row_names_[j]] = static_cast<i_t>(j);
  }
  std::vector<i_t> row_perm(n_constraints_);
  for (i_t i = 0; i < n_constraints_; ++i) {
    auto it = other_row_idx.find(row_names_[i]);
    if (it == other_row_idx.end()) { return false; }
    row_perm[i] = it->second;
  }

  // Upload permutations to GPU
  rmm::device_uvector<i_t> d_var_perm(n_vars_, stream_view_);
  rmm::device_uvector<i_t> d_row_perm(n_constraints_, stream_view_);
  raft::copy(d_var_perm.data(), var_perm.data(), n_vars_, stream_view_);
  raft::copy(d_row_perm.data(), row_perm.data(), n_constraints_, stream_view_);

  auto policy = rmm::exec_policy(stream_view_);

  auto permuted_eq = [&](auto this_begin, auto this_end, auto other_begin, auto perm_begin) {
    auto other_perm = thrust::make_permutation_iterator(other_begin, perm_begin);
    return thrust::equal(policy, this_begin, this_end, other_perm);
  };

  // Compare variable-indexed arrays
  if (c_.size() != other.c_.size()) { return false; }
  if (!permuted_eq(c_.begin(), c_.end(), other.c_.begin(), d_var_perm.begin())) { return false; }
  if (variable_lower_bounds_.size() != other.variable_lower_bounds_.size()) { return false; }
  if (!permuted_eq(variable_lower_bounds_.begin(),
                   variable_lower_bounds_.end(),
                   other.variable_lower_bounds_.begin(),
                   d_var_perm.begin())) {
    return false;
  }
  if (variable_upper_bounds_.size() != other.variable_upper_bounds_.size()) { return false; }
  if (!permuted_eq(variable_upper_bounds_.begin(),
                   variable_upper_bounds_.end(),
                   other.variable_upper_bounds_.begin(),
                   d_var_perm.begin())) {
    return false;
  }
  if (variable_types_.size() != other.variable_types_.size()) { return false; }
  if (!permuted_eq(variable_types_.begin(),
                   variable_types_.end(),
                   other.variable_types_.begin(),
                   d_var_perm.begin())) {
    return false;
  }

  // Compare constraint-indexed arrays
  if (b_.size() != other.b_.size()) { return false; }
  if (!permuted_eq(b_.begin(), b_.end(), other.b_.begin(), d_row_perm.begin())) { return false; }
  if (constraint_lower_bounds_.size() != other.constraint_lower_bounds_.size()) { return false; }
  if (!permuted_eq(constraint_lower_bounds_.begin(),
                   constraint_lower_bounds_.end(),
                   other.constraint_lower_bounds_.begin(),
                   d_row_perm.begin())) {
    return false;
  }
  if (constraint_upper_bounds_.size() != other.constraint_upper_bounds_.size()) { return false; }
  if (!permuted_eq(constraint_upper_bounds_.begin(),
                   constraint_upper_bounds_.end(),
                   other.constraint_upper_bounds_.begin(),
                   d_row_perm.begin())) {
    return false;
  }
  if (row_types_.size() != other.row_types_.size()) { return false; }
  if (!permuted_eq(
        row_types_.begin(), row_types_.end(), other.row_types_.begin(), d_row_perm.begin())) {
    return false;
  }

  // Build inverse permutations on CPU (needed for CSR comparisons)
  std::vector<i_t> var_perm_inv(n_vars_);
  for (i_t i = 0; i < n_vars_; ++i) {
    var_perm_inv[var_perm[i]] = i;
  }
  std::vector<i_t> row_perm_inv(n_constraints_);
  for (i_t i = 0; i < n_constraints_; ++i) {
    row_perm_inv[row_perm[i]] = i;
  }

  // Upload inverse permutations to GPU
  rmm::device_uvector<i_t> d_var_perm_inv(n_vars_, stream_view_);
  rmm::device_uvector<i_t> d_row_perm_inv(n_constraints_, stream_view_);
  raft::copy(d_var_perm_inv.data(), var_perm_inv.data(), n_vars_, stream_view_);
  raft::copy(d_row_perm_inv.data(), row_perm_inv.data(), n_constraints_, stream_view_);

  // Constraint matrix (A) comparison with row and column permutations
  if (!csr_matrices_equivalent_with_permutation(A_offsets_,
                                                A_indices_,
                                                A_,
                                                other.A_offsets_,
                                                other.A_indices_,
                                                other.A_,
                                                d_row_perm_inv,
                                                d_var_perm_inv,
                                                n_vars_,
                                                stream_view_)) {
    return false;
  }

  // Q matrix writing to MPS not supported yet. Don't check for equivalence here

  return true;
}

template <typename i_t, typename f_t>
bool optimization_problem_t<i_t, f_t>::is_equivalent(
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
// C API Support: Copy to Host (GPU Implementation)
// ==============================================================================

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_objective_coefficients_to_host(f_t* output,
                                                                           i_t size) const
{
  cuopt_expects(output != nullptr && size >= 0 && size <= static_cast<i_t>(c_.size()),
                error_type_t::ValidationError,
                "copy_objective_coefficients_to_host: null output or invalid size");
  RAFT_CUDA_TRY(cudaMemcpy(output, c_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_constraint_matrix_to_host(
  f_t* values, i_t* indices, i_t* offsets, i_t num_values, i_t num_indices, i_t num_offsets) const
{
  cuopt_expects(values != nullptr && num_values >= 0 && num_values <= static_cast<i_t>(A_.size()),
                error_type_t::ValidationError,
                "copy_constraint_matrix_to_host: null values or invalid size");
  cuopt_expects(
    indices != nullptr && num_indices >= 0 && num_indices <= static_cast<i_t>(A_indices_.size()),
    error_type_t::ValidationError,
    "copy_constraint_matrix_to_host: null indices or invalid size");
  cuopt_expects(
    offsets != nullptr && num_offsets >= 0 && num_offsets <= static_cast<i_t>(A_offsets_.size()),
    error_type_t::ValidationError,
    "copy_constraint_matrix_to_host: null offsets or invalid size");
  RAFT_CUDA_TRY(cudaMemcpy(values, A_.data(), num_values * sizeof(f_t), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(
    cudaMemcpy(indices, A_indices_.data(), num_indices * sizeof(i_t), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(
    cudaMemcpy(offsets, A_offsets_.data(), num_offsets * sizeof(i_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_row_types_to_host(char* output, i_t size) const
{
  cuopt_expects(output != nullptr && size >= 0 && size <= static_cast<i_t>(row_types_.size()),
                error_type_t::ValidationError,
                "copy_row_types_to_host: null output or invalid size");
  RAFT_CUDA_TRY(cudaMemcpy(output, row_types_.data(), size * sizeof(char), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_constraint_bounds_to_host(f_t* output, i_t size) const
{
  cuopt_expects(output != nullptr && size >= 0 && size <= static_cast<i_t>(b_.size()),
                error_type_t::ValidationError,
                "copy_constraint_bounds_to_host: null output or invalid size");
  RAFT_CUDA_TRY(cudaMemcpy(output, b_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_constraint_lower_bounds_to_host(f_t* output,
                                                                            i_t size) const
{
  cuopt_expects(
    output != nullptr && size >= 0 && size <= static_cast<i_t>(constraint_lower_bounds_.size()),
    error_type_t::ValidationError,
    "copy_constraint_lower_bounds_to_host: null output or invalid size");
  RAFT_CUDA_TRY(cudaMemcpy(
    output, constraint_lower_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_constraint_upper_bounds_to_host(f_t* output,
                                                                            i_t size) const
{
  cuopt_expects(
    output != nullptr && size >= 0 && size <= static_cast<i_t>(constraint_upper_bounds_.size()),
    error_type_t::ValidationError,
    "copy_constraint_upper_bounds_to_host: null output or invalid size");
  RAFT_CUDA_TRY(cudaMemcpy(
    output, constraint_upper_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_variable_lower_bounds_to_host(f_t* output,
                                                                          i_t size) const
{
  cuopt_expects(
    output != nullptr && size >= 0 && size <= static_cast<i_t>(variable_lower_bounds_.size()),
    error_type_t::ValidationError,
    "copy_variable_lower_bounds_to_host: null output or invalid size");
  RAFT_CUDA_TRY(
    cudaMemcpy(output, variable_lower_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_variable_upper_bounds_to_host(f_t* output,
                                                                          i_t size) const
{
  cuopt_expects(
    output != nullptr && size >= 0 && size <= static_cast<i_t>(variable_upper_bounds_.size()),
    error_type_t::ValidationError,
    "copy_variable_upper_bounds_to_host: null output or invalid size");
  RAFT_CUDA_TRY(
    cudaMemcpy(output, variable_upper_bounds_.data(), size * sizeof(f_t), cudaMemcpyDeviceToHost));
}

template <typename i_t, typename f_t>
void optimization_problem_t<i_t, f_t>::copy_variable_types_to_host(var_t* output, i_t size) const
{
  cuopt_expects(output != nullptr && size >= 0 && size <= static_cast<i_t>(variable_types_.size()),
                error_type_t::ValidationError,
                "copy_variable_types_to_host: null output or invalid size");
  RAFT_CUDA_TRY(
    cudaMemcpy(output, variable_types_.data(), size * sizeof(var_t), cudaMemcpyDeviceToHost));
}

template <typename From, typename To>
struct cast_op {
  HDI To operator()(From val) const { return static_cast<To>(val); }
};

template <typename From, typename To>
rmm::device_uvector<To> gpu_cast(const rmm::device_uvector<From>& src, rmm::cuda_stream_view stream)
{
  rmm::device_uvector<To> dst(src.size(), stream);
  if (src.size() > 0) {
    RAFT_CUDA_TRY(cub::DeviceTransform::Transform(
      src.data(), dst.data(), src.size(), cast_op<From, To>{}, stream.value()));
  }
  return dst;
}

template rmm::device_uvector<float> gpu_cast<double, float>(const rmm::device_uvector<double>&,
                                                            rmm::cuda_stream_view);
template rmm::device_uvector<double> gpu_cast<float, double>(const rmm::device_uvector<float>&,
                                                             rmm::cuda_stream_view);

template <typename i_t, typename f_t>
template <typename other_f_t>
optimization_problem_t<i_t, other_f_t> optimization_problem_t<i_t, f_t>::convert_to_other_prec(
  rmm::cuda_stream_view stream) const
{
  optimization_problem_t<i_t, other_f_t> other(handle_ptr_);

  other.set_maximize(maximize_);
  other.set_objective_offset(static_cast<other_f_t>(objective_offset_));
  other.set_objective_scaling_factor(static_cast<other_f_t>(objective_scaling_factor_));

  if (A_.size() > 0) {
    auto other_A = gpu_cast<f_t, other_f_t>(A_, stream);
    other.set_csr_constraint_matrix(other_A.data(),
                                    static_cast<i_t>(other_A.size()),
                                    A_indices_.data(),
                                    static_cast<i_t>(A_indices_.size()),
                                    A_offsets_.data(),
                                    static_cast<i_t>(A_offsets_.size()));
  }

  if (c_.size() > 0) {
    auto other_c = gpu_cast<f_t, other_f_t>(c_, stream);
    other.set_objective_coefficients(other_c.data(), static_cast<i_t>(other_c.size()));
  }

  if (b_.size() > 0) {
    auto other_b = gpu_cast<f_t, other_f_t>(b_, stream);
    other.set_constraint_bounds(other_b.data(), static_cast<i_t>(other_b.size()));
  }

  if (constraint_lower_bounds_.size() > 0) {
    auto other_clb = gpu_cast<f_t, other_f_t>(constraint_lower_bounds_, stream);
    other.set_constraint_lower_bounds(other_clb.data(), static_cast<i_t>(other_clb.size()));
  }

  if (constraint_upper_bounds_.size() > 0) {
    auto other_cub = gpu_cast<f_t, other_f_t>(constraint_upper_bounds_, stream);
    other.set_constraint_upper_bounds(other_cub.data(), static_cast<i_t>(other_cub.size()));
  }

  if (variable_lower_bounds_.size() > 0) {
    auto other_vlb = gpu_cast<f_t, other_f_t>(variable_lower_bounds_, stream);
    other.set_variable_lower_bounds(other_vlb.data(), static_cast<i_t>(other_vlb.size()));
  }

  if (variable_upper_bounds_.size() > 0) {
    auto other_vub = gpu_cast<f_t, other_f_t>(variable_upper_bounds_, stream);
    other.set_variable_upper_bounds(other_vub.data(), static_cast<i_t>(other_vub.size()));
  }

  if (variable_types_.size() > 0) {
    other.set_variable_types(variable_types_.data(), static_cast<i_t>(variable_types_.size()));
  }

  other.set_variable_names(var_names_);
  other.set_row_names(row_names_);
  other.set_objective_name(objective_name_);
  other.set_problem_category(problem_category_);

  return other;
}

// ==============================================================================
// Template instantiations
// ==============================================================================
// Explicit template instantiations matching MIP constants
#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template class optimization_problem_t<int32_t, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class optimization_problem_t<int32_t, double>;
#endif

#if PDLP_INSTANTIATE_FLOAT || MIP_INSTANTIATE_FLOAT
template optimization_problem_t<int32_t, float>
  optimization_problem_t<int32_t, double>::convert_to_other_prec<float>(
    rmm::cuda_stream_view) const;
#endif

}  // namespace cuopt::linear_programming
