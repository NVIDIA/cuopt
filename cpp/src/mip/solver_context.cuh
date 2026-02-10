/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <mip/problem/problem.cuh>
#include <mip/relaxed_lp/lp_state.cuh>
#include <utilities/work_limit_context.hpp>
#include <utilities/work_unit_scheduler.hpp>

#include <limits>

#pragma once

// Forward declare
namespace cuopt::linear_programming::dual_simplex {
template <typename i_t, typename f_t>
class branch_and_bound_t;
}

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class diversity_manager_t;

// Aggregate structure containing the global context of the solving process for convenience:
// The current problem, user settings, raft handle and statistics objects
template <typename i_t, typename f_t>
struct mip_solver_context_t {
  explicit mip_solver_context_t(raft::handle_t const* handle_ptr_,
                                problem_t<i_t, f_t>* problem_ptr_,
                                mip_solver_settings_t<i_t, f_t> settings_,
                                pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling)
    : handle_ptr(handle_ptr_), problem_ptr(problem_ptr_), settings(settings_), scaling(scaling)
  {
    cuopt_assert(problem_ptr != nullptr, "problem_ptr is nullptr");
    stats.set_solution_bound(problem_ptr->maximize ? std::numeric_limits<f_t>::infinity()
                                                   : -std::numeric_limits<f_t>::infinity());
    gpu_heur_loop.deterministic = settings.determinism_mode == CUOPT_MODE_DETERMINISTIC;
  }

  mip_solver_context_t(const mip_solver_context_t&)            = delete;
  mip_solver_context_t& operator=(const mip_solver_context_t&) = delete;

  raft::handle_t const* const handle_ptr;
  problem_t<i_t, f_t>* problem_ptr;
  dual_simplex::branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr{nullptr};
  diversity_manager_t<i_t, f_t>* diversity_manager_ptr{nullptr};
  std::atomic<bool> preempt_heuristic_solver_ = false;
  const mip_solver_settings_t<i_t, f_t> settings;
  pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling;
  solver_stats_t<i_t, f_t> stats;
  // Work limit context for tracking work units in deterministic mode (shared across all timers in
  // GPU heuristic loop)
  work_limit_context_t gpu_heur_loop{"GPUHeur"};

  // synchronization every 5 seconds for deterministic mode
  work_unit_scheduler_t work_unit_scheduler_{5.0};
};

}  // namespace cuopt::linear_programming::detail
