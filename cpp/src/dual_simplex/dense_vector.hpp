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
#pragma once

#include "dual_simplex/types.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t, typename Allocator = std::allocator<f_t>>
class dense_vector_t : public std::vector<f_t, Allocator> {
 public:
  dense_vector_t(i_t n) { this->resize(n, 0.0); }
  dense_vector_t(const std::vector<f_t>& in)
  {
    this->resize(in.size());
    const i_t n = static_cast<i_t>(in.size());
    for (i_t i = 0; i < n; i++) {
      (*this)[i] = in[i];
    }
  }

  template <typename OtherAlloc>
  dense_vector_t& operator=(const std::vector<f_t, OtherAlloc>& rhs)
  {
    this->assign(rhs.begin(), rhs.end());
    return *this;
  }

  template <typename OtherAlloc>
  dense_vector_t& operator=(const dense_vector_t<i_t, f_t, OtherAlloc>& rhs)
  {
    this->assign(rhs.begin(), rhs.end());
    return *this;
  }

  template <typename OtherAlloc>
  dense_vector_t(const std::vector<f_t, OtherAlloc>& rhs)
  {
    this->assign(rhs.begin(), rhs.end());
  }

  template <typename OtherAlloc>
  dense_vector_t(const dense_vector_t<i_t, f_t, OtherAlloc>& rhs)
  {
    this->assign(rhs.begin(), rhs.end());
  }

  f_t minimum() const
  {
    const i_t n = this->size();
    f_t min_x   = inf;
    for (i_t i = 0; i < n; i++) {
      min_x = std::min(min_x, (*this)[i]);
    }
    return min_x;
  }

  f_t maximum() const
  {
    const i_t n = this->size();
    f_t max_x   = -inf;
    for (i_t i = 0; i < n; i++) {
      max_x = std::max(max_x, (*this)[i]);
    }
    return max_x;
  }

  // b <- sqrt(a)
  template <typename OtherAlloc>
  void sqrt(dense_vector_t<i_t, f_t, OtherAlloc>& b) const
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      b[i] = std::sqrt((*this)[i]);
    }
  }
  // a <- a + alpha
  void add_scalar(f_t alpha)
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      (*this)[i] += alpha;
    }
  }
  // a <- alpha * e, e = (1, 1, ..., 1)
  void set_scalar(f_t alpha)
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      (*this)[i] = alpha;
    }
  }
  // a <- alpha * a
  void multiply_scalar(f_t alpha)
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      (*this)[i] *= alpha;
    }
  }
  f_t sum() const
  {
    f_t sum     = 0.0;
    const i_t n = this->size();
    for (i_t i = 0; i < n; ++i) {
      sum += (*this)[i];
    }
    return sum;
  }

  template <typename AllocatorB>
  f_t inner_product(dense_vector_t<i_t, f_t, AllocatorB>& b) const
  {
    f_t dot     = 0.0;
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      dot += (*this)[i] * b[i];
    }
    return dot;
  }

  // c <- a .* b
  template <typename AllocatorA, typename AllocatorB>
  void pairwise_product(const std::vector<f_t, AllocatorA>& b,
                        std::vector<f_t, AllocatorB>& c) const
  {
    const i_t n = this->size();
    if (static_cast<i_t>(b.size()) != n) {
      printf("Error: b.size() %d != n %d\n", static_cast<i_t>(b.size()), n);
      exit(1);
    }
    if (static_cast<i_t>(c.size()) != n) {
      printf("Error: c.size() %d != n %d\n", static_cast<i_t>(c.size()), n);
      exit(1);
    }
    for (i_t i = 0; i < n; i++) {
      c[i] = (*this)[i] * b[i];
    }
  }
  // c <- a ./ b
  void pairwise_divide(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& c) const
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      c[i] = (*this)[i] / b[i];
    }
  }

  // c <- a - b
  template <typename AllocatorA, typename AllocatorB>
  void pairwise_subtract(const dense_vector_t<i_t, f_t, AllocatorA>& b,
                         dense_vector_t<i_t, f_t, AllocatorB>& c) const
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      c[i] = (*this)[i] - b[i];
    }
  }

  // y <- alpha * x + beta * y
  template <typename InputAllocator>
  void axpy(f_t alpha, const std::vector<f_t, InputAllocator>& x, f_t beta)
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      (*this)[i] = alpha * x[i] + beta * (*this)[i];
    }
  }

  void ensure_positive(f_t epsilon_adjust)
  {
    const f_t mix_x = minimum();
    if (mix_x <= 0.0) {
      const f_t delta_x = -mix_x + epsilon_adjust;
      add_scalar(delta_x);
    }
  }

  void bound_away_from_zero(f_t epsilon_adjust)
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      if ((*this)[i] < epsilon_adjust) { (*this)[i] = epsilon_adjust; }
    }
  }

  // b <- 1.0 /a
  template <typename InputAllocator>
  void inverse(dense_vector_t<i_t, f_t, InputAllocator>& b) const
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      b[i] = 1.0 / (*this)[i];
    }
  }

  dense_vector_t<i_t, f_t> head(i_t p) const
  {
    dense_vector_t<i_t, f_t> y(p);
    const i_t N = std::min(p, static_cast<i_t>(this->size()));
    for (i_t i = 0; i < p; i++) {
      y[i] = (*this)[i];
    }
    return y;
  }

  dense_vector_t<i_t, f_t> tail(i_t p) const
  {
    dense_vector_t<i_t, f_t> y(p);
    const i_t n = this->size();
    const i_t N = std::max(n - p, 0);
    i_t j       = 0;
    for (i_t i = N; i < N + p; i++) {
      y[j++] = (*this)[i];
    }
    return y;
  }
};

template <typename T, typename Alloc>
std::vector<T> copy(const std::vector<T, Alloc>& src)
{
  return std::vector<T>(src.begin(), src.end());
}

}  // namespace cuopt::linear_programming::dual_simplex
