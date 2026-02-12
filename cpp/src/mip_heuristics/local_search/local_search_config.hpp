/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::linear_programming::detail {

struct ls_config_t {
  static constexpr bool use_line_segment                     = true;
  static constexpr bool use_fj                               = true;
  static constexpr bool use_fp_ls                            = false;
  static constexpr bool use_cutting_plane_from_best_solution = false;
};

}  // namespace cuopt::linear_programming::detail
