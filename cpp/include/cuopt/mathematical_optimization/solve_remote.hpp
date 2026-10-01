/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
// Include the solution interface definitions so unique_ptr can properly delete them
#include <cuopt/mathematical_optimization/optimization_problem_solution_interface.hpp>

#include <memory>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

// Forward declarations (only declaration needed, not definition)
template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

template <typename i_t, typename f_t>
class pdlp_solver_settings_t;

template <typename i_t, typename f_t>
class mip_solver_settings_t;

template <typename i_t, typename f_t>
class solver_settings_t;

// ============================================================================
// Remote Execution Functions
// ============================================================================

/**
 * @brief Solve LP problem remotely (CPU backend)
 */
template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings);

/**
 * @brief Solve an LP remotely from a solver_settings_t.
 *
 * Sends the deprecated typed fields and the parameter map. Callers that only
 * have a pdlp_solver_settings_t keep the overload above and send typed fields.
 */
template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem, solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Solve MIP problem remotely (CPU backend)
 */
template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings);

/**
 * @brief Solve a MIP remotely from a solver_settings_t.
 *
 * Sends the deprecated typed fields and the parameter map. Callers that only
 * have a mip_solver_settings_t keep the overload above and send typed fields.
 */
template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem, solver_settings_t<i_t, f_t>& settings);

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
