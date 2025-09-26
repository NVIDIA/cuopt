/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/device_sparse_matrix.cuh>
#include <dual_simplex/pinned_host_allocator.hpp>
#include <dual_simplex/sparse_matrix.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
template <typename Allocator>
void csc_matrix_t<i_t, f_t>::scale_columns(const std::vector<f_t, Allocator>& scale)
{
  const i_t n = this->n;
  assert(scale.size() == n);
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = this->col_start[j];
    const i_t col_end   = this->col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      this->x[p] *= scale[j];
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template int
matrix_vector_multiply<int, double, PinnedHostAllocator<double>, PinnedHostAllocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template int
matrix_vector_multiply<int, double, PinnedHostAllocator<double>, std::allocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, std::allocator<double>>& y);

template int
matrix_vector_multiply<int, double, std::allocator<double>, PinnedHostAllocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, std::allocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template int matrix_transpose_vector_multiply<int,
                                              double,
                                              PinnedHostAllocator<double>,
                                              PinnedHostAllocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template int
matrix_transpose_vector_multiply<int, double, PinnedHostAllocator<double>, std::allocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, std::allocator<double>>& y);

template int
matrix_transpose_vector_multiply<int, double, std::allocator<double>, PinnedHostAllocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, std::allocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template void csc_matrix_t<int, double>::scale_columns<std::allocator<double>>(
  const std::vector<double, std::allocator<double>>& scale);
template void csc_matrix_t<int, double>::scale_columns<PinnedHostAllocator<double>>(
  const std::vector<double, PinnedHostAllocator<double>>& scale);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
