/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/types.hpp>

namespace cuopt::linear_programming::dual_simplex {
double tic();
double toc(double start);

}  // namespace cuopt::linear_programming::dual_simplex
