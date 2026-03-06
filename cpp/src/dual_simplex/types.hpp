/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

#define DUAL_SIMPLEX_INSTANTIATE_DOUBLE

using float32_t = float;
using float64_t = double;

constexpr float64_t inf = std::numeric_limits<float64_t>::infinity();

// First LP solution from either PDLP/Barrier or dual simplex; used to notify diversity manager
// without B&B depending on PDLP types.
template <typename i_t, typename f_t>
struct root_relaxation_first_solution_t {
  std::vector<f_t> primal;
  std::vector<f_t> dual;
  std::vector<f_t> reduced_costs;
  f_t objective{0};
  f_t user_objective{0};
  i_t iterations{0};
};

// We return this constant to signal that a concurrent halt has occurred
#define CONCURRENT_HALT_RETURN -2
// We return this constant to signal that a time limit has occurred
#define TIME_LIMIT_RETURN -3

}  // namespace cuopt::linear_programming::dual_simplex
