/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <barrier/barrier.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::simplex {

// GPU-based Ruiz scaling
template <typename i_t, typename f_t>
i_t scaling_ruiz_gpu(const lp_problem_t<i_t, f_t>& unscaled,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     lp_problem_t<i_t, f_t>& scaled,
                     std::vector<f_t>& column_scaling,
                     std::vector<f_t>& row_scaling,
                     barrier::device_csc_matrix_ptr_t<i_t, f_t>& device_A,
                     barrier::device_csc_matrix_ptr_t<i_t, f_t>& device_Q);

}  // namespace cuopt::mathematical_optimization::simplex
