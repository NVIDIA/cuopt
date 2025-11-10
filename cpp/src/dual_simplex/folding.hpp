/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
void folding(lp_problem_t<i_t, f_t>& problem,
             const simplex_solver_settings_t<i_t, f_t>& settings,
             presolve_info_t<i_t, f_t>& presolve_info);

}  // namespace cuopt::linear_programming::dual_simplex
