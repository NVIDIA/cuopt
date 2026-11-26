/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
bool compute_farkas_certificate(const lp_problem_t<i_t, f_t>& lp,
                                const simplex_solver_settings_t<i_t, f_t>& settings,
                                const std::vector<f_t>& delta_y,
                                const std::vector<f_t>& delta_z,
                                std::vector<f_t>& farkas_y,
                                std::vector<f_t>& farkas_zl,
                                std::vector<f_t>& farkas_zu,
                                f_t& farkas_constant,
                                f_t& work_estimate);

}
