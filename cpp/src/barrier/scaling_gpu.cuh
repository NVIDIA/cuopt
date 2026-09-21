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
// Only ever held by shared_ptr below, so the definition (which needs nvcc) stays out of here.
template <typename i_t, typename f_t>
class device_csc_matrix_t;
}  // namespace cuopt::mathematical_optimization::barrier

namespace cuopt::mathematical_optimization::simplex {

// GPU-based Ruiz scaling.
//
// `device_A` receives the scaled A when it is left on device (second-order cones only, where
// `scaled.A` then keeps its sparsity pattern but has an empty `x` and this is the only copy of the
// values); `device_Q` receives the scaled Q. Both are null when there is nothing to hand over, and
// the barrier uploads from `scaled` instead.
template <typename i_t, typename f_t>
i_t scaling_ruiz_gpu(const lp_problem_t<i_t, f_t>& unscaled,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     lp_problem_t<i_t, f_t>& scaled,
                     std::vector<f_t>& column_scaling,
                     std::vector<f_t>& row_scaling,
                     std::shared_ptr<barrier::device_csc_matrix_t<i_t, f_t>>& device_A,
                     std::shared_ptr<barrier::device_csc_matrix_t<i_t, f_t>>& device_Q);

}  // namespace cuopt::mathematical_optimization::simplex
