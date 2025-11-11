/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::linear_programming::detail {

enum class termination_criterion_t {
  TIME_LIMIT,
  ITERATION_LIMIT,
  CONVERGENCE,
  INFEASIBLE,
  NO_UPDATE
};

}  // namespace cuopt::linear_programming::detail
