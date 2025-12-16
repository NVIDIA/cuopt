/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <limits>

namespace cuopt::linear_programming::dual_simplex {

#define DUAL_SIMPLEX_INSTANTIATE_DOUBLE

using float32_t = float;
using float64_t = double;

constexpr float64_t inf = std::numeric_limits<float64_t>::infinity();

}  // namespace cuopt::linear_programming::dual_simplex
