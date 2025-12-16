/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

namespace rmm {
template <typename T>
class device_uvector;
}  // namespace rmm

namespace cuopt::linear_programming {

namespace detail {
template <typename i_t, typename f_t>
class problem_t;
}  // namespace detail

template <typename i_t, typename f_t>
class problem_checking_t {
 public:
  static void check_csr_representation(const optimization_problem_t<i_t, f_t>& op_problem);
  // Check all fields and convert row_types to constraints lower/upper bounds if needed
  static void check_problem_representation(const optimization_problem_t<i_t, f_t>& op_problem);
  static bool has_crossing_bounds(const optimization_problem_t<i_t, f_t>& op_problem);

  static void check_scaled_problem(detail::problem_t<i_t, f_t> const& scaled_problem,
                                   detail::problem_t<i_t, f_t> const& op_problem);
  static void check_unscaled_solution(detail::problem_t<i_t, f_t>& op_problem,
                                      rmm::device_uvector<f_t> const& assignment);
  static void check_initial_primal_representation(
    const optimization_problem_t<i_t, f_t>& op_problem,
    const rmm::device_uvector<f_t>& primal_initial_solution);
  static void check_initial_dual_representation(
    const optimization_problem_t<i_t, f_t>& op_problem,
    const rmm::device_uvector<f_t>& dual_initial_solution);
  static void check_initial_solution_representation(
    const optimization_problem_t<i_t, f_t>& op_problem,
    const pdlp_solver_settings_t<i_t, f_t>& settings);
  static void check_initial_solution_representation(
    const optimization_problem_t<i_t, f_t>& op_problem,
    const mip_solver_settings_t<i_t, f_t>& settings);
};

}  // namespace cuopt::linear_programming
