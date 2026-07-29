/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/optimization_problem_solution_interface.hpp>
#include <memory>

// Forward declarations — full types live in libcuopt_lp / libcuopt_grpc headers.
namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

template <typename i_t, typename f_t>
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

// Function pointer types — only the <int,double> instantiation is supported.
using solve_lp_remote_fn_t = std::unique_ptr<lp_solution_interface_t<int, double>> (*)(
  cpu_optimization_problem_t<int, double> const&, pdlp_solver_settings_t<int, double> const&);

using solve_mip_remote_fn_t = std::unique_ptr<mip_solution_interface_t<int, double>> (*)(
  cpu_optimization_problem_t<int, double> const&, mip_solver_settings_t<int, double> const&);

// Defined in libcuopt_lp.so (remote_solve_registry.cpp).
// Set to nullptr until libcuopt_grpc.so is loaded and calls register_remote_solvers().
extern solve_lp_remote_fn_t g_solve_lp_remote_fn;
extern solve_mip_remote_fn_t g_solve_mip_remote_fn;

// Called by libcuopt_grpc.so's constructor to wire up the real implementations.
void register_remote_solvers(solve_lp_remote_fn_t lp_fn, solve_mip_remote_fn_t mip_fn);

}  // namespace cuopt::mathematical_optimization
