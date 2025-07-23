/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// A sparse vector stored as a list of nonzero coefficients and their indices
template <typename i_t, typename f_t>
class sparse_vector_t {
 public:
  // Construct a sparse vector of dimension n with nz nonzero coefficients
  sparse_vector_t(i_t n, i_t nz) : n(n), i(nz), x(nz) {}
  // Construct a sparse vector from a dense vector.
  sparse_vector_t(const std::vector<f_t>& in) { from_dense(in); }
  // Construct a sparse vector from a column of a CSC matrix
  sparse_vector_t(const csc_matrix_t<i_t, f_t>& A, i_t col);
  // gather a dense vector into a sparse vector
  void from_dense(const std::vector<f_t>& in);
  // convert a sparse vector into a CSC matrix with a single column
  void to_csc(csc_matrix_t<i_t, f_t>& A) const;
  // convert a sparse vector into a dense vector. Dense vector is cleared and resized.
  void to_dense(std::vector<f_t>& x_dense) const;
  // scatter a sparse vector into a dense vector. Assumes x_dense is already cleared or
  // preinitialized
  void scatter(std::vector<f_t>& x_dense) const;
  // inverse permute the current sparse vector
  void inverse_permute_vector(const std::vector<i_t>& p);
  // inverse permute a sparse vector into another sparse vector
  void inverse_permute_vector(const std::vector<i_t>& p, sparse_vector_t<i_t, f_t>& y) const;
  // compute the dot product of a sparse vector with a column of a CSC matrix
  f_t sparse_dot(const csc_matrix_t<i_t, f_t>& Y, i_t y_col) const;
  // ensure the coefficients in the sparse vectory are sorted in terms of increasing index
  void sort();
  // compute the squared 2-norm of the sparse vector
  f_t norm2_squared() const;
  void negate();
  f_t find_coefficient(i_t index) const;

  i_t n;
  std::vector<i_t> i;
  std::vector<f_t> x;
};

}  // namespace cuopt::linear_programming::dual_simplex
