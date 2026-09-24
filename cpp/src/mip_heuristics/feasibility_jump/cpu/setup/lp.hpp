/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../state.hpp"

namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
class csr_matrix_t;

namespace simplex {

template <typename i_t, typename f_t>
struct lp_problem_t;

}  // namespace simplex
}  // namespace cuopt::mathematical_optimization

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void eliminate_slacks(const simplex::lp_problem_t<i_t, f_t>&,
                      i_t,
                      csr_matrix_t<i_t, f_t>&,
                      std::vector<f_t>&,
                      std::vector<f_t>&);

template <typename i_t, typename f_t>
void apply_lp_rounded_start(fj_cpu_climber_t<i_t, f_t>&, f_t);
template <typename i_t, typename f_t>
bool apply_lp_polish(fj_cpu_climber_t<i_t, f_t>&, double);
}  // namespace cuopt::mathematical_optimization::mip
