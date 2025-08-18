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

#include <dual_simplex/sparse_vector.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
sparse_vector_t<i_t, f_t>::sparse_vector_t(const csc_matrix_t<i_t, f_t>& A, i_t col)
{
  const i_t col_start = A.col_start[col];
  const i_t col_end   = A.col_start[col + 1];
  n                   = A.m;
  const i_t nz        = col_end - col_start;
  i.reserve(nz);
  x.reserve(nz);
  for (i_t k = col_start; k < col_end; ++k) {
    i.push_back(A.i[k]);
    x.push_back(A.x[k]);
  }
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::from_dense(const std::vector<f_t>& in)
{
  i.clear();
  x.clear();
  n = in.size();
  i.reserve(n);
  x.reserve(n);
  for (i_t k = 0; k < n; ++k) {
    if (in[k] != 0) {
      i.push_back(k);
      x.push_back(in[k]);
    }
  }
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::to_csc(csc_matrix_t<i_t, f_t>& A) const
{
  A.m      = n;
  A.n      = 1;
  A.nz_max = i.size();
  A.col_start.resize(2);
  A.col_start[0] = 0;
  A.col_start[1] = i.size();
  A.i            = i;
  A.x            = x;
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::to_dense(std::vector<f_t>& x_dense) const
{
  x_dense.clear();
  x_dense.resize(n, 0.0);
  const i_t nz = i.size();
  for (i_t k = 0; k < nz; ++k) {
    x_dense[i[k]] = x[k];
  }
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::scatter(std::vector<f_t>& x_dense) const
{
  // Assumes x_dense is already cleared
  const i_t nz = i.size();
  for (i_t k = 0; k < nz; ++k) {
    x_dense[i[k]] += x[k];
  }
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::inverse_permute_vector(const std::vector<i_t>& p)
{
  assert(p.size() == n);
  i_t nz = i.size();
  std::vector<i_t> i_perm(nz);
  for (i_t k = 0; k < nz; ++k) {
    i_perm[k] = p[i[k]];
  }
  i = i_perm;
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::inverse_permute_vector(const std::vector<i_t>& p,
                                                       sparse_vector_t<i_t, f_t>& y) const
{
  i_t m = p.size();
  assert(n == m);
  i_t nz = i.size();
  y.n    = n;
  y.x    = x;
  std::vector<i_t> i_perm(nz);
  for (i_t k = 0; k < nz; ++k) {
    i_perm[k] = p[i[k]];
  }
  y.i = i_perm;
}

template <typename i_t, typename f_t>
f_t sparse_vector_t<i_t, f_t>::sparse_dot(const csc_matrix_t<i_t, f_t>& Y, i_t y_col) const
{
  const i_t col_start = Y.col_start[y_col];
  const i_t col_end   = Y.col_start[y_col + 1];
  const i_t ny        = col_end - col_start;
  const i_t nx        = i.size();
  f_t dot             = 0.0;
  for (i_t h = 0, k = col_start; h < nx && k < col_end;) {
    const i_t p = i[h];
    const i_t q = Y.i[k];
    if (p == q) {
      dot += Y.x[k] * x[h];
      h++;
      k++;
    } else if (p < q) {
      h++;
    } else if (q < p) {
      k++;
    }
  }
  return dot;
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::sort()
{
  if (i.size() == 1) { return; }
  // If the number of nonzeros is large, use a O(n) bucket sort
  if (i.size() > 0.3 * n) {
    std::vector<f_t> bucket(n, 0.0);
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      bucket[i[k]] = x[k];
    }
    i.clear();
    i.reserve(nz);
    x.clear();
    x.reserve(nz);
    for (i_t k = 0; k < n; ++k) {
      if (bucket[k] != 0.0) {
        i.push_back(k);
        x.push_back(bucket[k]);
      }
    }
  } else {
    // Use a n log n sort
    const i_t nz = i.size();
    std::vector<i_t> i_sorted(nz);
    std::vector<f_t> x_sorted(nz);
    std::vector<i_t> perm(nz);
    for (i_t k = 0; k < nz; ++k) {
      perm[k] = k;
    }
    std::vector<i_t>& iunsorted = i;
    std::sort(
      perm.begin(), perm.end(), [&iunsorted](i_t a, i_t b) { return iunsorted[a] < iunsorted[b]; });
    for (i_t k = 0; k < nz; ++k) {
      i_sorted[k] = i[perm[k]];
      x_sorted[k] = x[perm[k]];
    }
    i = i_sorted;
    x = x_sorted;
  }

  // Check
#ifdef CHECK_SORT
  if (!std::is_sorted(i.begin(), i.end())) { printf("Sort error\n"); }
#endif
}

template <typename i_t, typename f_t>
f_t sparse_vector_t<i_t, f_t>::norm2_squared() const
{
  f_t dot      = 0.0;
  const i_t nz = i.size();
  for (i_t k = 0; k < nz; ++k) {
    dot += x[k] * x[k];
  }
  return dot;
}

template <typename i_t, typename f_t>
void sparse_vector_t<i_t, f_t>::negate()
{
  const i_t nz = x.size();
  for (i_t k = 0; k < nz; ++k) {
    x[k] *= -1.0;
  }
}

template <typename i_t, typename f_t>
f_t sparse_vector_t<i_t, f_t>::find_coefficient(i_t index) const
{
  const i_t nz = i.size();
  for (i_t k = 0; k < nz; ++k) {
    if (i[k] == index) { return x[k]; }
  }
  return std::numeric_limits<f_t>::quiet_NaN();
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class sparse_vector_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::dual_simplex
