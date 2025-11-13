/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <random>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class random_t {
 public:
  random_t(i_t seed) : gen(seed) {}

  template <typename T>
  T random_index(T n)
  {
    std::uniform_int_distribution<> distrib(
      0, n - 1);  // Generate random number in the range [min, max]
    return distrib(gen);
  }

  f_t random()
  {
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    return distrib(gen);
  }

 private:
  std::mt19937 gen;
};

}  // namespace cuopt::linear_programming::dual_simplex
