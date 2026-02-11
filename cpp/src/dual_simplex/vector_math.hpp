/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cassert>
#include <cmath>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Computes || x ||_inf = max_j | x |_j
template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm_inf(const std::vector<f_t, Allocator>& x)
{
  i_t n = x.size();
  f_t a = 0.0;
  for (i_t j = 0; j < n; ++j) {
    f_t t = std::abs(x[j]);
    if (t > a) { a = t; }
  }
  return a;
}

// Computes || x ||_2^2
template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm2_squared(const std::vector<f_t, Allocator>& x);

// Computes || x ||_2
template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm2(const std::vector<f_t, Allocator>& x);

// Computes || x ||_1
template <typename i_t, typename f_t>
f_t vector_norm1(const std::vector<f_t>& x);

// Computes x'*y
template <typename i_t, typename f_t>
f_t dot(const std::vector<f_t>& x, const std::vector<f_t>& y);

// Computes x'*y when x and y are sparse
template <typename i_t, typename f_t>
f_t sparse_dot(const std::vector<i_t>& xind,
               const std::vector<f_t>& xval,
               const std::vector<i_t>& yind,
               const std::vector<f_t>& yval);

template <typename i_t, typename f_t>
f_t sparse_dot(
  i_t const* xind, f_t const* xval, i_t nx, i_t const* yind, i_t ny, f_t const* y_scatter_val);

template <typename i_t, typename f_t>
f_t sparse_dot(i_t* xind, f_t* xval, i_t nx, i_t* yind, f_t* yval, i_t ny);

// Computes x = P*b or x=b(p) in MATLAB.
template <typename VectorI, typename VectorF_in, typename VectorF_out>
int permute_vector(const VectorI& p, const VectorF_in& b, VectorF_out& x)
{
  auto n = p.size();
  assert(x.size() == n);
  assert(b.size() == n);
  for (decltype(n) k = 0; k < n; ++k) {
    x[k] = b[p[k]];
  }
  return 0;
}

// Computes x = P'*b or x(p) = b in MATLAB.
template <typename VectorI, typename VectorF_in, typename VectorF_out>
int inverse_permute_vector(const VectorI& p, const VectorF_in& b, VectorF_out& x)
{
  auto n = p.size();
  assert(x.size() == n);
  assert(b.size() == n);
  for (decltype(n) k = 0; k < n; ++k) {
    x[p[k]] = b[k];
  }
  return 0;
}

// Computes pinv from p. Or pinv(p) = 1:n in MATLAB
template <typename VectorI_in, typename VectorI_out>
int inverse_permutation(const VectorI_in& p, VectorI_out& pinv)
{
  auto n = p.size();
  if (pinv.size() != n) { pinv.resize(n); }
  for (decltype(n) k = 0; k < n; ++k) {
    pinv[p[k]] = k;
  }
  return 0;
}

}  // namespace cuopt::linear_programming::dual_simplex
