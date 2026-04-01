/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/relaxed_lp/lp_state.cuh>
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

template <typename i_t, typename f_t>
class early_cpufj_t;

// Aggregate structure containing the global context of the solving process for convenience:
// The current problem, user settings, raft handle and statistics objects
template <typename i_t, typename f_t>
struct mip_solver_context_t {
  explicit mip_solver_context_t(raft::handle_t const* handle_ptr_,
                                problem_t<i_t, f_t>* problem_ptr_,
                                mip_solver_settings_t<i_t, f_t> settings_)
    : handle_ptr(handle_ptr_), problem_ptr(problem_ptr_), settings(settings_)
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
  solver_stats_t<i_t, f_t> stats;
  // Work limit context for tracking work units in deterministic mode (shared across all timers in
  // GPU heuristic loop)
  work_limit_context_t gpu_heur_loop{"GPUHeur"};

  // synchronization every 5 seconds for deterministic mode
  work_unit_scheduler_t work_unit_scheduler_{5.0};

  early_cpufj_t<i_t, f_t>* early_cpufj_ptr{nullptr};
  // Best objective from early heuristics, in user-space.
  // Must be converted to the target solver-space before use:
  //   - B&B: problem_ptr->get_solver_obj_from_user_obj(initial_cutoff)
  //   - CPUFJ: papilo_problem.get_solver_obj_from_user_obj(initial_cutoff)
  // Use std::isfinite() to check whether a valid cutoff exists.
  f_t initial_cutoff{std::numeric_limits<f_t>::infinity()};
};

}  // namespace cuopt::linear_programming::detail
