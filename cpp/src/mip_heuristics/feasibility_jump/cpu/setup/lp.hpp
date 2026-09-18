/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

}  // namespace cuopt::mathematical_optimization::mip
