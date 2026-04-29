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
 * A semi-continuous variable x satisfies: x = 0  OR  L <= x <= U  (0 < L <= U).
 * Reformulation adds a binary variable b and two linking constraints when needed.
 * Added binaries are appended at the end of the variable arrays, and their linking
 * constraints are appended at the end of the CSR row arrays in the same order.
 *   x - L * b >= 0      (forces x >= L when b=1; allows x=0 when b=0)
 *   x - U * b <= 0      (forces x <= U when b=1; forces x=0 when b=0)
 *   b in {0, 1},  x in [0, U]
 *
 * Deterministic CPU bounds strengthening is seeded only from SC variables to derive tight upper
 * bounds for SC variables that have infinite original upper bounds. If strengthening cannot
 * derive a finite bound, settings.semi_continuous_big_m is used as a fallback.
 *
 * This must be called before problem_t construction and Papilo presolve.
 *
 * @tparam i_t  Integer index type
 * @tparam f_t  Floating-point value type
 * @param[in,out] op_problem  The optimization problem (modified in-place)
 * @param[in]     settings    MIP solver settings (provides semi_continuous_big_m and tolerances)
 * @param[out]    used_fallback_big_m Per-original-variable flags. Entry i is set to 1
 *                                    when variable i uses settings.semi_continuous_big_m as a
 * fallback upper bound during reformulation. Used to reject the final solution if its upper bound
 * lands on big-m within integrality tolerance.
 * @param[out]    semi_continuous_binary_to_original_indices Optional mapping for appended
 *                                    auxiliary
 *                                    binaries. Entry k stores the original semi-continuous
 *                                    variable index that produced appended binary k, in append
 *                                    order.
 * @returns true if any semi-continuous variables were found and reformulated.
 */
template <typename i_t, typename f_t>
bool reformulate_semi_continuous(
  optimization_problem_t<i_t, f_t>& op_problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  std::vector<uint8_t>* used_fallback_big_m,
  std::vector<i_t>* semi_continuous_binary_to_original_indices = nullptr);

template <typename i_t, typename f_t>
void expand_initial_solutions_for_semi_continuous(
  mip_solver_settings_t<i_t, f_t>& settings,
  const std::vector<i_t>& semi_continuous_binary_to_original_indices,
  rmm::cuda_stream_view stream);

template <typename i_t, typename f_t>
void append_semi_continuous_auxiliaries_to_assignment(
  std::vector<f_t>& assignment,
  const std::vector<i_t>& semi_continuous_binary_to_original_indices,
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances);

template <typename i_t, typename f_t>
void strip_semi_continuous_auxiliaries_from_assignment(std::vector<f_t>& assignment,
                                                       i_t original_num_variables);

}  // namespace cuopt::linear_programming::detail
