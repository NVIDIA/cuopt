/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>

#include <mps_parser/mps_data_model.hpp>

#include <raft/core/handle.hpp>

namespace cuopt::linear_programming {

namespace detail {
template <typename i_t, typename f_t>
class problem_t;
}  // namespace detail

template <typename i_t, typename f_t>
cuopt::linear_programming::optimization_problem_t<i_t, f_t> mps_data_model_to_optimization_problem(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& data_model);

template <typename i_t, typename f_t>
cuopt::linear_programming::optimization_problem_solution_t<i_t, f_t> solve_lp_with_method(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer,
  bool is_batch_mode = false);

// Entry point for batch PDLP. Two call contexts:
//
//   1. Strong branching: caller passes an un-expanded
//      optimization_problem_t plus per-climber variable-bound in settings.new_bounds.
//      run_batch_pdlp auto-picks the optimal batch size and potentially loops over sub-batches, managing
//      memory pressure internally.
//
//   2. Fixed path (fixed path, settings.fixed_batch_size > 0): caller has already sized
//      the batch (using compute_optimal_batch_size below), pre-expanded the per-climber
//      problem fields directly on the optimization_problem_t (objective_coefficients,
//      constraint_lower_bounds, constraint_upper_bounds, batch_objective_offsets_) and set settings.fixed_batch_size.
//      run_batch_pdlp runs a single solve_lp with no memory-aware sub-batching.
template <typename i_t, typename f_t>
cuopt::linear_programming::optimization_problem_solution_t<i_t, f_t> run_batch_pdlp(
  cuopt::linear_programming::optimization_problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings);

/**
  @brief Compute the optimal batch size for the problem.
  @param problem The problem to compute the optimal batch size for.
  @param per_climber_objectives Whether the problem will per-climber objectives (resulting in a larger memory footprint).
  @param per_climber_constraint_bounds Whether the problem will have per-climber constraint bounds (resulting in a larger memory footprint).
  @param collect_solutions Whether the problem has per-climber solutions (only for testing, by default we don't need to collect solution vectors).
  @return The optimal batch size for the problem.
  @note At this stage, the problem shouldn't already be expanded. The results of this function should be used as the fixed_batch_size to expand the problem and call run_batch_pdlp.
*/
template <typename i_t, typename f_t>
size_t compute_optimal_batch_size(
  const cuopt::linear_programming::optimization_problem_t<i_t, f_t>& problem,
  bool per_climber_objectives,
  bool per_climber_constraint_bounds,
  bool collect_solutions = false); // Only for testing

template <typename i_t, typename f_t>
void set_pdlp_solver_mode(pdlp_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::linear_programming
