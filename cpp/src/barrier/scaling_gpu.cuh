/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::barrier {
template <typename i_t, typename f_t>
class device_csc_matrix_t;
}  // namespace cuopt::mathematical_optimization::barrier

namespace cuopt::mathematical_optimization::simplex {

// GPU-based Ruiz scaling
template <typename i_t, typename f_t>
i_t scaling_ruiz_gpu(const lp_problem_t<i_t, f_t>& unscaled,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     lp_problem_t<i_t, f_t>& scaled,
                     std::vector<f_t>& column_scaling,
                     std::vector<f_t>& row_scaling,
                     std::unique_ptr<barrier::device_csc_matrix_t<i_t, f_t>>& device_A,
                     std::unique_ptr<barrier::device_csc_matrix_t<i_t, f_t>>& device_Q);

}  // namespace cuopt::mathematical_optimization::simplex
