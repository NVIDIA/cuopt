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

#include <dual_simplex/pinned_host_allocator.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/vector_math.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm2_squared(const std::vector<f_t, Allocator>& x)
{
  i_t n   = x.size();
  f_t sum = 0.0;
  for (i_t j = 0; j < n; ++j) {
    sum += x[j] * x[j];
  }
  return sum;
}

template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm2(const std::vector<f_t, Allocator>& x)
{
  return std::sqrt(vector_norm2_squared<i_t, f_t, Allocator>(x));
}

template <typename i_t, typename f_t>
f_t vector_norm1(const std::vector<f_t>& x)
{
  i_t n   = x.size();
  f_t sum = 0.0;
  for (i_t j = 0; j < n; ++j) {
    sum += std::abs(x[j]);
  }
  return sum;
}

template <typename i_t, typename f_t>
f_t dot(const std::vector<f_t>& x, const std::vector<f_t>& y)
{
  assert(x.size() == y.size());
  const i_t n = x.size();
  f_t dot     = 0.0;
  for (i_t k = 0; k < n; ++k) {
    dot += x[k] * y[k];
  }
  return dot;
}

template <typename i_t, typename f_t>
f_t sparse_dot(
  i_t const* xind, f_t const* xval, i_t nx, i_t const* yind, i_t ny, f_t const* y_scatter_val)
{
  f_t dot = 0.0;
  for (i_t i = 0, j = 0; i < nx && j < ny;) {
    const i_t p = xind[i];
    const i_t q = yind[j];
    if (p == q) {
      dot += xval[i] * y_scatter_val[q];
      i++;
      j++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      j++;
    }
  }
  return dot;
}

template <typename i_t, typename f_t>
f_t sparse_dot(i_t* xind, f_t* xval, i_t nx, i_t* yind, f_t* yval, i_t ny)
{
  f_t dot = 0.0;
  for (i_t i = 0, j = 0; i < nx && j < ny;) {
    const i_t p = xind[i];
    const i_t q = yind[j];
    if (p == q) {
      dot += xval[i] * yval[j];
      i++;
      j++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      j++;
    }
  }
  return dot;
}

template <typename i_t, typename f_t>
f_t sparse_dot(const std::vector<i_t>& xind,
               const std::vector<f_t>& xval,
               const std::vector<i_t>& yind,
               const std::vector<f_t>& yval)
{
  const i_t nx = xind.size();
  const i_t ny = yind.size();
  f_t dot      = 0.0;
  for (i_t i = 0, j = 0; i < nx && j < ny;) {
    const i_t p = xind[i];
    const i_t q = yind[j];
    if (p == q) {
      dot += xval[i] * yval[j];
      i++;
      j++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      j++;
    }
  }
  return dot;
}

// x = b(p)
template <typename i_t, typename f_t>
i_t permute_vector(const std::vector<i_t>& p, const std::vector<f_t>& b, std::vector<f_t>& x)
{
  i_t n = p.size();
  assert(x.size() == n);
  assert(b.size() == n);
  for (i_t k = 0; k < n; ++k) {
    x[k] = b[p[k]];
  }
  return 0;
}

// x(p) = b
template <typename i_t, typename f_t>
i_t inverse_permute_vector(const std::vector<i_t>& p,
                           const std::vector<f_t>& b,
                           std::vector<f_t>& x)
{
  i_t n = p.size();
  assert(x.size() == n);
  assert(b.size() == n);
  for (i_t k = 0; k < n; ++k) {
    x[p[k]] = b[k];
  }
  return 0;
}

template <typename i_t>
i_t inverse_permutation(const std::vector<i_t>& p, std::vector<i_t>& pinv)
{
  i_t n = p.size();
  if (pinv.size() != n) { pinv.resize(n); }
  for (i_t k = 0; k < n; ++k) {
    pinv[p[k]] = k;
  }
  return 0;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template double vector_norm_inf<int, double, std::allocator<double>>(const std::vector<double>& x);

template double vector_norm2_squared<int, double, std::allocator<double>>(
  const std::vector<double, std::allocator<double>>& x);

template double vector_norm2<int, double, std::allocator<double>>(
  const std::vector<double, std::allocator<double>>& x);

template double vector_norm2_squared<int, double, PinnedHostAllocator<double>>(
  const std::vector<double, PinnedHostAllocator<double>>&);
template double vector_norm2<int, double, PinnedHostAllocator<double>>(
  const std::vector<double, PinnedHostAllocator<double>>&);

template double vector_norm1<int, double>(const std::vector<double>& x);

template double dot<int, double>(const std::vector<double>& x, const std::vector<double>& y);

template double sparse_dot<int, double>(const std::vector<int>& xind,
                                        const std::vector<double>& xval,
                                        const std::vector<int>& yind,
                                        const std::vector<double>& yval);

template double sparse_dot<int, double>(int const* xind,
                                        double const* xval,
                                        int nx,
                                        int const* yind,
                                        int ny,
                                        double const* y_scatter_val);

template double sparse_dot<int, double>(
  int* xind, double* xval, int nx, int* yind, double* yval, int ny);

template int permute_vector<int, double>(const std::vector<int>& p,
                                         const std::vector<double>& b,
                                         std::vector<double>& x);

template int inverse_permute_vector<int, double>(const std::vector<int>& p,
                                                 const std::vector<double>& b,
                                                 std::vector<double>& x);

template int inverse_permutation<int>(const std::vector<int>& p, std::vector<int>& pinv);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
