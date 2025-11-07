/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once
namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
struct solver_stats_t {
  f_t total_solve_time       = 0.;
  f_t presolve_time          = 0.;
  f_t solution_bound         = std::numeric_limits<f_t>::min();
  i_t num_nodes              = 0;
  i_t num_simplex_iterations = 0;
};

}  // namespace cuopt::linear_programming
