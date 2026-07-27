/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <math_optimization/types.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
i_t scaling(const lp_problem_t<i_t, f_t>& unscaled,
            const simplex_solver_settings_t<i_t, f_t>& settings,
            lp_problem_t<i_t, f_t>& scaled,
            std::vector<f_t>& column_scaling,
            std::vector<f_t>& row_scaling);

template <typename i_t, typename f_t>
void unscale_solution(const std::vector<f_t>& column_scaling,
                      const std::vector<f_t>& row_scaling,
                      const std::vector<f_t>& scaled_x,
                      const std::vector<f_t>& scaled_y,
                      const std::vector<f_t>& scaled_z,
                      std::vector<f_t>& unscaled_x,
                      std::vector<f_t>& unscaled_y,
                      std::vector<f_t>& unscaled_z);

}  // namespace cuopt::mathematical_optimization::simplex
