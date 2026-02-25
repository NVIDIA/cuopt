/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Include the solution interface definitions so unique_ptr can properly delete them
#include <cuopt/linear_programming/optimization_problem_solution_interface.hpp>

#include <memory>

namespace cuopt::linear_programming {

// Forward declarations for optimization problems (only declaration needed, not definition)
template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

template <typename i_t, typename f_t>
class gpu_optimization_problem_t;

template <typename i_t, typename f_t>
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

// ============================================================================
// Remote Execution Functions
// ============================================================================

/**
 * @brief Solve LP problem remotely (CPU backend)
 */
template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking     = true,
  bool use_pdlp_solver_mode = true);

/**
 * @brief Solve LP problem remotely (GPU backend)
 */
template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  gpu_optimization_problem_t<i_t, f_t> const& gpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking     = true,
  bool use_pdlp_solver_mode = true);

/**
 * @brief Solve MIP problem remotely (CPU backend)
 */
template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings);

/**
 * @brief Solve MIP problem remotely (GPU backend)
 */
template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  gpu_optimization_problem_t<i_t, f_t> const& gpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings);

}  // namespace cuopt::linear_programming
