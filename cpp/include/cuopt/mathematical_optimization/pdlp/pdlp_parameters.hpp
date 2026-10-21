/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <cuopt/export.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

// Plain, user-facing PDLP/LP parameters. This header must not include CUDA, rmm or raft: it is
// what pdlp_solver_settings_t derives from, and what a CPU-only client uses without a GPU stack.

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

/**
 * @brief Enum representing the different solver modes under which PDLP can
 * operate.
 *
 * Stable3: Best overall mode from experiments; balances speed and convergence
 * success. If you want to use the legacy version, use Stable2.
 * Methodical1: Usually leads to slower individual steps but fewer are needed to
 * converge. It uses from 1.3x up to 1.7x times more memory.
 * Fast1: Less convergence success but usually yields the highest speed
 *
 * @note Default mode is Stable3.
 */
// Forced to use an enum instead of an enum class for compatibility with the
// Cython layer
enum pdlp_solver_mode_t : int {
  Stable1     = CUOPT_PDLP_SOLVER_MODE_STABLE1,
  Stable2     = CUOPT_PDLP_SOLVER_MODE_STABLE2,
  Methodical1 = CUOPT_PDLP_SOLVER_MODE_METHODICAL1,
  Fast1       = CUOPT_PDLP_SOLVER_MODE_FAST1,
  Stable3     = CUOPT_PDLP_SOLVER_MODE_STABLE3
};

/**
 * @brief Enum representing the different methods that can be used to solve the
 * linear programming problem.
 *
 * Concurrent: Use PDLP, Barrier and DualSimplex in parallel.
 * PDLP: Use the PDLP method.
 * DualSimplex: Use the dual simplex method.
 * Barrier: Use the barrier method
 * PrimalSimplex: Use the primal simplex method.
 * Unset: The value was not set.
 *
 * @note Default method is Concurrent.
 */
enum method_t : int {
  Concurrent    = CUOPT_METHOD_CONCURRENT,
  PDLP          = CUOPT_METHOD_PDLP,
  DualSimplex   = CUOPT_METHOD_DUAL_SIMPLEX,
  Barrier       = CUOPT_METHOD_BARRIER,
  PrimalSimplex = CUOPT_METHOD_PRIMAL_SIMPLEX,
  Unset         = CUOPT_METHOD_UNSET
};

/// Returns the corresponding string from the enum `method_t`.
inline std::string method_to_string(method_t method)
{
  switch (method) {
    case method_t::DualSimplex: return "Dual Simplex";
    case method_t::PDLP: return "PDLP";
    case method_t::Barrier: return "Barrier";
    case method_t::Concurrent: return "Concurrent";
    case method_t::PrimalSimplex: return "Primal Simplex";
    default: return "Unset";
  }
}

/**
 * @brief Enum representing the PDLP precision modes.
 *
 * DefaultPrecision: Use the type of the problem (FP64 for double problems).
 * SinglePrecision:  Run PDLP internally in FP32, converting inputs and outputs.
 * DoublePrecision:  Explicitly run in FP64 (same as default for double problems).
 * MixedPrecision:   Use mixed precision SpMV (FP32 matrix with FP64 vectors/compute).
 */
enum pdlp_precision_t : int {
  DefaultPrecision = CUOPT_PDLP_DEFAULT_PRECISION,
  SinglePrecision  = CUOPT_PDLP_SINGLE_PRECISION,
  DoublePrecision  = CUOPT_PDLP_DOUBLE_PRECISION,
  MixedPrecision   = CUOPT_PDLP_MIXED_PRECISION
};

/**
 * @brief Which graph partitioner multi-GPU PDLP uses.
 *
 * Auto: pick automatically (RoundRobin on 1 GPU, KaMinPar otherwise).
 * KaMinPar: multi-threaded KaMinPar graph partitioner.
 * RoundRobin: round-robin assignment, no graph.
 */
enum multigpu_pdlp_partitioner_t : int {
  Auto       = CUOPT_MULTIGPU_PDLP_PARTITIONER_AUTO,
  KaMinPar   = CUOPT_MULTIGPU_PDLP_PARTITIONER_KAMINPAR,
  RoundRobin = CUOPT_MULTIGPU_PDLP_PARTITIONER_ROUND_ROBIN,
};

/**
 * @brief The PDLP/LP parameters, with no GPU-backed members.
 *
 * pdlp_solver_settings_t derives from this and adds initial solutions, warm start and the
 * solver-internal plumbing. Every member here stays reachable by the same name on
 * pdlp_solver_settings_t.
 */
template <typename i_t, typename f_t>
struct pdlp_parameters_t {
  struct tolerances_t {
    f_t absolute_dual_tolerance     = 1.0e-4;
    f_t relative_dual_tolerance     = 1.0e-4;
    f_t absolute_primal_tolerance   = 1.0e-4;
    f_t relative_primal_tolerance   = 1.0e-4;
    f_t absolute_gap_tolerance      = 1.0e-4;
    f_t relative_gap_tolerance      = 1.0e-4;
    f_t primal_infeasible_tolerance = 1.0e-10;
    f_t dual_infeasible_tolerance   = 1.0e-10;
  };

  tolerances_t get_tolerances() const noexcept { return tolerances; }

  /**
   * @brief Set both absolute and relative tolerance on the primal feasibility,
   dual feasibility and gap.
   * Changing this value has a significant impact on accuracy and runtime.
   *
   * Optimality is computed as follows:
   * - dual_feasiblity < absolute_dual_tolerance + relative_dual_tolerance *
   norm_objective_coefficient (l2_norm(c))
   * - primal_feasiblity < absolute_primal_tolerance + relative_primal_tolerance
   * norm_constraint_bounds (l2_norm(b))
   * - duality_gap < absolute_gap_tolerance + relative_gap_tolerance *
   (|primal_objective| + |dual_objective|)
   *
   * If all three conditions hold, optimality is reached.
   *
   * @note Default value is 1e-4.
   *
   * To set each absolute and relative tolerance, use the provided setters.
   *
   * @param eps_optimal Tolerance to optimality
   */
  void set_optimality_tolerance(f_t eps_optimal)
  {
    tolerances.absolute_dual_tolerance   = eps_optimal;
    tolerances.relative_dual_tolerance   = eps_optimal;
    tolerances.absolute_primal_tolerance = eps_optimal;
    tolerances.relative_primal_tolerance = eps_optimal;
    tolerances.absolute_gap_tolerance    = eps_optimal;
    tolerances.relative_gap_tolerance    = eps_optimal;
  }

  tolerances_t tolerances;
  bool detect_infeasibility{false};
  bool strict_infeasibility{false};
  i_t iteration_limit{std::numeric_limits<i_t>::max()};
  f_t time_limit{std::numeric_limits<f_t>::infinity()};
  pdlp_solver_mode_t pdlp_solver_mode{pdlp_solver_mode_t::Stable3};
  bool log_to_console{true};
  std::string log_file{""};
  std::string sol_file{""};
  std::string user_problem_file{""};
  std::string presolve_file{""};
  bool per_constraint_residual{false};
  bool crossover{false};
  bool cudss_deterministic{false};
  i_t project_initial_primal_override{-1};
  i_t folding{-1};
  i_t augmented{-1};
  i_t dualize{-1};
  i_t ordering{-1};
  i_t initial_perturbation{-1};
  i_t remove_perturbation{-1};
  i_t primal_pricing{1};
  barrier_dual_initial_point_t barrier_dual_initial_point{barrier_dual_initial_point_t::Automatic};
  i_t postsolve_info{-1};
  i_t barrier_presolve_bound_free_variables{-1};  // -1 automatic, 0 disabled, 1 enabled
  // Ruiz equilibration for QCQP (barrier) scaling: -1 automatic (row/column
  // imbalance heuristic), 0 disabled, 1 enabled. Distinct from PDLP's own Ruiz
  // scaling in pdlp_hyper_params_t.
  i_t qcqp_ruiz_equilibration{-1};
  // nnz(A)+nnz(Q) at or above which the barrier path runs Ruiz equilibration on GPU instead
  // of CPU. Below it the upload costs more than the scaling saves.
  i_t gpu_ruiz_nnz_threshold{500000};
  // Margin used to push the barrier method's initial iterate into the interior of the
  // nonnegative orthant / SOC (values are shifted to be at least this far from the boundary).
  f_t barrier_initial_point_safeguard{10.0};
  bool eliminate_dense_columns{true};
  pdlp_precision_t pdlp_precision{pdlp_precision_t::DefaultPrecision};
  bool barrier_iterative_refinement{true};
  i_t barrier_adaptive_regularization{-1};  // -1 automatic, 0 disabled, 1 enabled
  // Initial regularization for the barrier method's augmented KKT system, applied to the first
  // factorization only (adaptive regularization, if enabled, still scales it up/down on later
  // iterations). -1 automatic (uses the built-in heuristic), else the literal starting value.
  f_t barrier_primal_regularization{-1.0};
  f_t barrier_dual_regularization{-1.0};
  i_t barrier_soc_threshold{100};
  f_t barrier_step_scale{0.9};
  bool save_best_primal_so_far{false};
  /**
   * @brief Stop the solver as soon as a primal feasible iterate is encountered.
   *
   * In non-batch mode the solver returns the first primal feasible iterate (without waiting for
   * optimality / dual feasibility). In batch mode the whole batch stops the moment any climber
   * reaches primal feasibility; every climber returns its current iterate with its current
   * termination status. Can be composed with `per_constraint_residual`.
   * Mutually exclusive with `all_primal_feasible`.
   */
  bool first_primal_feasible{false};
  /**
   * @brief Batch-only: stop only once every climber has reached (at least) primal feasibility.
   *
   * Each climber is individually ejected from the batch the first time it becomes primal
   * feasible and its per-climber solution is captured. The solver returns when all climbers
   * have been captured. Setting this in non-batch mode is a validation error. Setting it
   * together with `first_primal_feasible` is a validation error.
   */
  bool all_primal_feasible{false};
  presolver_t presolver{presolver_t::Default};
  bool dual_postsolve{true};
  // Concurrent LP/MIP: 1–2 GPUs. Multi-GPU PDLP (method=PDLP): up to the visible device
  // count; -1 selects all visible GPUs, which dispatches to the multi-GPU PDLP engine
  // whenever num_gpus == -1 or num_gpus > 1.
  int num_gpus{1};
  // Which graph partitioner multi-GPU PDLP uses. See
  // multigpu_pdlp_partitioner_t for the meaning of each value.
  multigpu_pdlp_partitioner_t multigpu_pdlp_partitioner{multigpu_pdlp_partitioner_t::Auto};
  method_t method{method_t::Concurrent};
  // At or above this reduced NNZ, skip concurrent CPU solvers and enable DEFAULT column reduction
  // on compatible PDLP paths. -1 disables both automatic decisions.
  // TODO: Remove the concurrent CPU solver bypass once their memory usage and cuDSS long running
  // kernels are resolved; retain the size threshold for automatic column reduction.
  i_t large_nnz_threshold{50'000'000};
  pdlp::pdlp_hyper_params_t hyper_params;
  // Holds per-climber variable-bound overrides in the format:
  // (climber id, variable index, new lower bound, new upper bound).
  // Per-climber objective coefficients / offsets / constraint bounds must be pre-expanded directly
  // on the optimization_problem_t instead.
  std::vector<std::tuple<i_t, i_t, f_t, f_t>> new_bounds;
  // By default to save memory and speed we don't store and copy each climber's primal and dual
  // solutions We only retrieve termination statistics and the objective values
  bool generate_batch_primal_dual_solution{false};
  // Used to force batch PDLP to solve a subbatch of the problems at a time
  // The 0 default value will make the solver use its heuristic to determine the subbatch size
  i_t fixed_batch_size{0};
  /** When true, the first GPU barrier/QCQP solve retains cache state for later reuse. */
  bool sequence_solve{false};
};

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
