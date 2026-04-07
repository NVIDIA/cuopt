/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>

namespace cuopt::linear_programming::detail {

/**
 * @brief Reformulate semi-continuous variables in-place inside the MIP solver.
 *
 * A semi-continuous variable x satisfies: x = 0  OR  L <= x <= U  (0 < L < U).
 * Reformulation introduces a binary variable b and two linking constraints:
 *   x - L * b >= 0      (forces x >= L when b=1; allows x=0 when b=0)
 *   x - U * b <= 0      (forces x <= U when b=1; forces x=0 when b=0)
 *   b in {0, 1},  x in [0, U]
 *
 * GPU bounds propagation (bound_presolve_t) is used to derive tight upper bounds
 * for SC variables that have infinite original upper bounds.  If propagation cannot
 * derive a finite bound, settings.sc_big_m is used as a fallback.
 *
 * This must be called before problem_t construction and Papilo presolve.
 *
 * @tparam i_t  Integer index type
 * @tparam f_t  Floating-point value type
 * @param[in,out] op_problem  The optimization problem (modified in-place)
 * @param[in]     settings    MIP solver settings (provides sc_big_m and tolerances)
 * @returns true if any semi-continuous variables were found and reformulated.
 */
template <typename i_t, typename f_t>
bool reformulate_semi_continuous(optimization_problem_t<i_t, f_t>& op_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::linear_programming::detail
