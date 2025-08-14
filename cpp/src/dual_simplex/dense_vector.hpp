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

#include "dual_simplex/types.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class dense_vector_t : public std::vector<f_t> {
 public:
  dense_vector_t(i_t n) { this->resize(n, 0.0); }
  dense_vector_t(const std::vector<f_t>& in)
  {
    this->resize(in.size());
    for (i_t i = 0; i < in.size(); i++) {
      (*this)[i] = in[i];
    }
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
  void sqrt(dense_vector_t<i_t, f_t>& b) const
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

  f_t inner_product(dense_vector_t<i_t, f_t>& b) const
  {
    f_t dot     = 0.0;
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      dot += (*this)[i] * b[i];
    }
    return dot;
  }

  // c <- a .* b
  void pairwise_product(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& c) const
  {
    const i_t n = this->size();
    if (b.size() != n) {
      printf("Error: b.size() %d != n %d\n", b.size(), n);
      exit(1);
    }
    if (c.size() != n) {
      printf("Error: c.size() %d != n %d\n", c.size(), n);
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
  void pairwise_subtract(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& c) const
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      c[i] = (*this)[i] - b[i];
    }
  }

  // y <- alpha * x + beta * y
  void axpy(f_t alpha, const dense_vector_t<i_t, f_t>& x, f_t beta)
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
  void inverse(dense_vector_t<i_t, f_t>& b) const
  {
    const i_t n = this->size();
    for (i_t i = 0; i < n; i++) {
      b[i] = 1.0 / (*this)[i];
    }
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
