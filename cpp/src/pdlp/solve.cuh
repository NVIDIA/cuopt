/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/optimization_problem.hpp>

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <raft/core/handle.hpp>

namespace cuopt::mathematical_optimization {

namespace mip {
template <typename i_t, typename f_t>
class problem_t;
}  // namespace mip

template <typename i_t, typename f_t>
cuopt::mathematical_optimization::optimization_problem_t<i_t, f_t>
mps_data_model_to_optimization_problem(
  raft::handle_t const* handle_ptr,
  const cuopt::mathematical_optimization::io::mps_data_model_t<i_t, f_t>& data_model);

template <typename i_t, typename f_t>
cuopt::mathematical_optimization::optimization_problem_solution_t<i_t, f_t> solve_lp_with_method(
  mip::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer,
  bool is_batch_mode = false);

/**
 * @brief Distributed-PDLP entry point that consumes the host MPS data model
 *        directly, partitioning it across GPUs without ever materializing the
 *        full problem on a single (master) GPU.
 *
 * Intended for problems whose `nnz` exceeds the memory of a single device. The
 * master `pdlp_solver_t` is constructed from a shape-0 placeholder problem; the
 * real work happens inside it, built straight from the host `mps_data_model`:
 *   1. host-side graph partitioning off the MPS CSR,
 *   2. per-shard host CSR slicing,
 *   3. one shard pdlp_solver_t per GPU, while master holds only scalar metadata
 *      + gather buffers (no full A / A^T / scaled copies).
 * It then runs the solver, gathers the solution to master, applies the
 * maximization sign-flip on the dual / reduced cost when the sense is maximize,
 * and returns the gathered solution.
 *
 * Resolves the `distributed_pdlp_num_gpus == -1` sentinel against the
 * visible-device count. Several configurations are rejected up front
 * (see @pre).
 *
 * @param handle_ptr  Master raft handle (its stream owns the gather buffers and
 *                    any master-side aggregator allocations). Must be non-null.
 * @param mps_data_model  Host-resident MPS data (CPU vectors only).
 * @param settings    User-supplied PDLP solver settings; the
 *                    `distributed_pdlp_num_gpus == -1` sentinel is resolved
 *                    here against the visible-device count.
 * @param problem_checking      Currently unused on this path (accepted for
 *                    signature parity with the device-side entry points).
 * @param use_pdlp_solver_mode  When true, applies set_pdlp_solver_mode() to the
 *                    resolved settings before solving.
 *
 * @pre `settings.use_distributed_pdlp == true`, `method == PDLP`,
 *      `pdlp_precision == DefaultPrecision`, not inside MIP, and no initial
 *      primal/dual or warm-start data.
 */
template <typename i_t, typename f_t>
cuopt::mathematical_optimization::optimization_problem_solution_t<i_t, f_t>
solve_lp_distributed_from_mps(
  raft::handle_t const* handle_ptr,
  const cuopt::mathematical_optimization::io::mps_data_model_t<i_t, f_t>& mps_data_model,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode);

/**
 * @brief Entry point for batch PDLP. Solves multiple LPs sharing the same constraint
 *        matrix structure in a single batched GPU run.
 *
 * Two call contexts are supported:
 *
 *   1. Strong-branching path:
 *      The caller passes an un-expanded optimization_problem_t plus per-climber
 *      variable bounds in settings.new_bounds. Each bound entry has shape
 *      (climber_id, variable_index, lower, upper); several entries may target
 *      the same climber. The batch size is max(climber_id) + 1. run_batch_pdlp
 *      auto-picks the optimal sub-batch size and may loop over sub-batches,
 *      managing memory pressure internally.
 *      See pdlp_test.cu:strong_branching_user_api for a full example.
 *
 *   2. Fixed-batch path (settings.fixed_batch_size > 0):
 *      The caller has already sized the batch (typically via
 *      compute_optimal_batch_size below) and pre-expanded the per-climber problem
 *      fields directly on the optimization_problem_t (objective_coefficients,
 *      constraint_lower_bounds, constraint_upper_bounds, batch_objective_offsets_).
 *      run_batch_pdlp performs a single solve_lp with no memory-aware sub-batching.
 *      See pdlp_test.cu:big_batch_fixed_path for a full example.
 *
 * @param problem  The optimization problem (un-expanded for case 1, pre-expanded for case 2).
 * @param settings Solver settings
 * @return The batched solution.
 *
 * @code
 * // Case 1: Strong branching (auto batch sizing)
 * pdlp_solver_settings_t<i_t, f_t> settings;
 * // Per-climber variable bounds: (climber_id, variable_index, lower, upper).
 * settings.new_bounds.push_back({0, branch_var, lower_bound, down_bound});
 * settings.new_bounds.push_back({1, branch_var, up_bound, upper_bound});
 * auto solution = run_batch_pdlp(problem, settings);
 * @endcode
 *
 * @code
 * // Case 2: Fixed batch (caller-managed expansion)
 * size_t batch_size = compute_optimal_batch_size(problem,
 *                                                per_climber_objectives,
 *                                                per_climber_constraint_bounds);
 * expand_problem_in_place(problem, batch_size);     // caller fills the per-climber fields
 * // Shouldn't use the set_X API as it will change the problem n_variables and n_constraints
 * // Instead, directly use get_X() = X to set the values
 * pdlp_solver_settings_t<i_t, f_t> settings;
 * settings.fixed_batch_size = batch_size;
 * auto solution = run_batch_pdlp(problem, settings);
 * @endcode
 */
template <typename i_t, typename f_t>
cuopt::mathematical_optimization::optimization_problem_solution_t<i_t, f_t> run_batch_pdlp(
  cuopt::mathematical_optimization::optimization_problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings);

/**
  @brief Compute the optimal batch size for the problem.
  @param problem The problem to compute the optimal batch size for.
  @param per_climber_objectives Whether the problem will per-climber objectives (resulting in a
  larger memory footprint).
  @param per_climber_constraint_bounds Whether the problem will have per-climber constraint bounds
  (resulting in a larger memory footprint).
  @param collect_solutions Whether the problem has per-climber solutions (only for testing, by
  default we don't need to collect solution vectors).
  @return The optimal batch size for the problem.
  @note At this stage, the problem shouldn't already be expanded. The results of this function
  should be used as the fixed_batch_size to expand the problem and call run_batch_pdlp.
*/
template <typename i_t, typename f_t>
size_t compute_optimal_batch_size(
  const cuopt::mathematical_optimization::optimization_problem_t<i_t, f_t>& problem,
  bool per_climber_objectives,
  bool per_climber_constraint_bounds,
  bool collect_solutions = false);  // Only for testing

template <typename i_t, typename f_t>
void set_pdlp_solver_mode(pdlp_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::mathematical_optimization
