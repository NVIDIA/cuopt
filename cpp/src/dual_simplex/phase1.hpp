/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>

#include <limits>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t create_phase1_problem(const lp_problem_t<i_t, f_t>& lp, lp_problem_t<i_t, f_t>& out);

}  // namespace cuopt::linear_programming::dual_simplex
