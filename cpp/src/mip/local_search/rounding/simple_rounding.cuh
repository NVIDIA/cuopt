/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip/solution/solution.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void invoke_round_nearest(solution_t<i_t, f_t>& solution);

template <typename i_t, typename f_t>
bool invoke_simple_rounding(solution_t<i_t, f_t>& solution);

template <typename i_t, typename f_t>
void invoke_random_round_nearest(solution_t<i_t, f_t>& solution, i_t n_target_random_rounds);

template <typename i_t, typename f_t>
void invoke_correct_integers(solution_t<i_t, f_t>& solution, f_t tol);

template <typename i_t, typename f_t>
bool check_brute_force_rounding(solution_t<i_t, f_t>& solution);

}  // namespace cuopt::linear_programming::detail
