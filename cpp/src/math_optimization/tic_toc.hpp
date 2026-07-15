/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
#include <math_optimization/types.hpp>

namespace cuopt::mathematical_optimization {
CUOPT_INTERNAL_EXPORT double tic();
CUOPT_INTERNAL_EXPORT double toc(double start);

}  // namespace cuopt::mathematical_optimization
