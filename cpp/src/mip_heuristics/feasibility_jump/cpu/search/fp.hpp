/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
struct user_problem_t;

}

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;

template <typename i_t, typename f_t>
bool solve_lp_relaxation(const simplex::user_problem_t<i_t, f_t>& relaxation,
                         double time_limit,
                         std::vector<f_t>& assignment,
                         double& solve_seconds);

template <typename i_t, typename f_t>
bool solve_lp_with_fixed_variables(const simplex::user_problem_t<i_t, f_t>& problem,
                                   const std::vector<i_t>& fixed_variables,
                                   const std::vector<f_t>& fixed_values,
                                   double time_limit,
                                   std::vector<f_t>& assignment,
                                   double& solve_seconds);

template <typename i_t, typename f_t>
void run_cpu_feasibility_pump(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                              const simplex::user_problem_t<i_t, f_t>& base,
                              double budget,
                              bool monotone_integer_equalities);

}  // namespace cuopt::mathematical_optimization::mip
