/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../state.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void report_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c,
                          f_t objective,
                          const std::vector<f_t>& assignment,
                          double work_units);

template <typename i_t, typename f_t>
void report_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void share_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c,
                         f_t objective,
                         const std::vector<f_t>& assignment);

template <typename i_t, typename f_t>
void share_cpu_incumbent(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void recompute_lhs(fj_cpu_climber_t<i_t, f_t>& fj_cpu);

template <typename i_t, typename f_t>
void recompute_slack(fj_cpu_climber_t<i_t, f_t>& fj_cpu);

template <typename i_t, typename f_t>
void invalidate_mtm_cache(fj_cpu_climber_t<i_t, f_t>& fj_cpu);

template <typename i_t, typename f_t>
void compute_variable_coloring(fj_cpu_climber_t<i_t, f_t>& fj_cpu);

template <typename i_t, typename f_t>
void retire_var_best_moves(fj_cpu_climber_t<i_t, f_t>& fj_cpu);

}  // namespace cuopt::mathematical_optimization::mip
