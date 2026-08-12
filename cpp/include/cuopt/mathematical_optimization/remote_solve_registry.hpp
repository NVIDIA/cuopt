/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>

#include <cuopt/mathematical_optimization/optimization_problem_solution_interface.hpp>

#include <atomic>
#include <memory>

// Forward declarations — full types live in libcuopt_mathopt / libcuopt_grpc
// headers.
namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

template <typename i_t, typename f_t>
class pdlp_solver_settings_t;

template <typename i_t, typename f_t>
class mip_solver_settings_t;

/**
 * @brief Remote LP solve entry point implemented by libcuopt_grpc.so.
 *
 * The returned solution is owned by the caller. The callback must not propagate
 * exceptions across the component boundary. Only the `<int, double>`
 * instantiation is supported.
 */
using solve_lp_remote_fn_t = std::unique_ptr<lp_solution_interface_t<int, double>> (*)(
  cpu_optimization_problem_t<int, double> const&, pdlp_solver_settings_t<int, double> const&);

/**
 * @brief Remote MIP solve entry point implemented by libcuopt_grpc.so.
 *
 * Same ownership and exception contract as @ref solve_lp_remote_fn_t.
 */
using solve_mip_remote_fn_t = std::unique_ptr<mip_solution_interface_t<int, double>> (*)(
  cpu_optimization_problem_t<int, double> const&, mip_solver_settings_t<int, double> const&);

/**
 * @brief Registry slots defined in libcuopt_mathopt.so
 * (remote_solve_registry.cpp).
 *
 * Null until libcuopt_grpc.so is loaded and calls register_remote_solvers(). Atomic
 * because the registering ELF constructor runs on whichever thread triggers the lazy
 * dlopen while other threads may be reading the slots.
 */
extern std::atomic<solve_lp_remote_fn_t> g_solve_lp_remote_fn;
extern std::atomic<solve_mip_remote_fn_t> g_solve_mip_remote_fn;

/**
 * @brief Readiness flag, published after both callbacks are stored.
 *
 * Readers must observe this as true before trusting either slot.
 */
extern std::atomic<bool> g_remote_solvers_ready;

/**
 * @brief Wire up the real remote-solve implementations.
 *
 * Called by libcuopt_grpc.so's ELF constructor. Thread-safe.
 */
CUOPT_EXPORT void register_remote_solvers(solve_lp_remote_fn_t lp_fn, solve_mip_remote_fn_t mip_fn);

/**
 * @brief Load libcuopt_grpc.so on demand so its constructor populates the registry.
 *
 * Idempotent and thread-safe; a failed load leaves the registry slots null so callers
 * can report the failure themselves.
 */
void ensure_remote_solvers_loaded();

}  // namespace cuopt::mathematical_optimization
