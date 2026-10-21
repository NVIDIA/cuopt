/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/mip/diving_hyper_params.hpp>
#include <cuopt/mathematical_optimization/mip/heuristics_hyper_params.hpp>
#include <cuopt/mathematical_optimization/mip/submip_hyper_params.hpp>
#include <cuopt/mathematical_optimization/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/mathematical_optimization/pdlp/pdlp_parameters.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <cuopt/export.hpp>

#include <limits>
#include <optional>
#include <string>
#include <vector>

// Plain, user-facing MIP parameters. This header must not include CUDA, rmm or raft; see
// pdlp_parameters.hpp.

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

struct benchmark_info_t {
  double last_improvement_of_best_feasible    = 0;
  double last_improvement_after_recombination = 0;
  double objective_of_initial_population      = std::numeric_limits<double>::max();
  // LP relaxation objective at the root node, BEFORE any cuts have been
  // added. quiet_NaN() means "B&B did not run cut passes / value was
  // never written" — distinguishes it from a legitimate 0.0.
  double root_lp_no_cuts = std::numeric_limits<double>::quiet_NaN();
  // LP relaxation objective at the root node, AFTER the full cut loop
  // (final pass result). The dual gap "by cuts at the root" is then
  //   gap_after_cuts = opt - root_lp_with_cuts        (in B&B's solver
  //                                                    objective sense)
  // and the classical "gap closed by cuts" metric is
  //   gap_closed_pct = 100 * (root_lp_with_cuts - root_lp_no_cuts)
  //                          / (opt - root_lp_no_cuts).
  // quiet_NaN() means "B&B did not finish the cut loop / value not written".
  double root_lp_with_cuts = std::numeric_limits<double>::quiet_NaN();

  // Wall-clock time spent inside the root-node cut generation loop
  // (sum of generate_cuts + score_cuts + check_for_duplicate_cuts +
  // get_best_cuts + add_cuts + post-cut LP resolves), in seconds.
  // Published by branch_and_bound.cpp::solve() at the same point that
  // root_lp_with_cuts is finalised. quiet_NaN() means "cut loop did
  // not run / value never written".
  double cut_generation_time_sec = std::numeric_limits<double>::quiet_NaN();
};

/**
 * @brief The MIP parameters, with no GPU-backed members.
 *
 * mip_solver_settings_t derives from this and adds initial solutions and callbacks.
 */
template <typename i_t, typename f_t>
struct mip_parameters_t {
  struct tolerances_t {
    f_t presolve_absolute_tolerance = 1.0e-6;
    f_t absolute_tolerance          = 1.0e-6;
    f_t relative_tolerance          = 1.0e-12;
    f_t integrality_tolerance       = 1.0e-5;
    f_t absolute_mip_gap            = 1.0e-10;
    f_t relative_mip_gap            = 1.0e-4;
  };

  tolerances_t get_tolerances() const noexcept { return tolerances; }

  tolerances_t tolerances;

  f_t time_limit                = std::numeric_limits<f_t>::infinity();
  f_t work_limit                = std::numeric_limits<f_t>::infinity();
  f_t semi_continuous_big_m     = f_t(1e10);
  i_t node_limit                = std::numeric_limits<i_t>::max();
  bool heuristics_only          = false;
  i_t reliability_branching     = -1;
  i_t num_cpu_threads           = -1;  // -1 means use default number of threads in branch and bound
  i_t symmetry                  = -1;
  i_t max_cut_passes            = 10;  // number of cut passes to make
  i_t mir_cuts                  = -1;
  i_t mixed_integer_gomory_cuts = -1;
  i_t knapsack_cuts             = -1;
  i_t flow_cover_cuts           = -1;
  i_t clique_cuts               = -1;
  i_t zero_half_cuts            = -1;
  i_t implied_bound_cuts        = -1;
  i_t strong_chvatal_gomory_cuts       = -1;
  i_t reduced_cost_strengthening       = -1;
  i_t dual_degenerate_feasibility_pump = -1;  // -1 = automatic (off), 0 = off, 1 = on
  i_t primal_degenerate_pivots         = -1;  // -1 = automatic (on), 0 = off, 1 = on
  i_t dual_degenerate_pivots = -1;  // -1 = automatic (off), 0 = off, 1 = root/dives, 2 = everywhere
  i_t objective_step         = 1;   // 0 = disable objective step tightening, 1 = enable
  f_t cut_change_threshold   = -1.0;
  f_t cut_min_orthogonality  = 0.5;
  i_t mip_batch_pdlp_strong_branching{
    0};  // 0 = DS only, 1 = cooperative DS + PDLP, 2 = batch PDLP only
  i_t mip_batch_pdlp_reliability_branching{
    0};  // 0 = DS only, 1 = cooperative DS + PDLP, 2 = batch PDLP only
  i_t strong_branching_simplex_iteration_limit = -1;
  i_t num_gpus                                 = 1;
  method_t method{method_t::Concurrent};
  i_t large_nnz_threshold{50'000'000};
  bool log_to_console = true;

  std::string log_file;
  std::string sol_file;
  std::string user_problem_file;
  std::string presolve_file;

  int mip_scaling = CUOPT_MIP_SCALING_NO_OBJECTIVE;
  presolver_t presolver{presolver_t::Default};
  /**
   * @brief Enable the cuOpt internal probing-cache step of presolve (MIP only).
   *
   * Probing is part of cuOpt's internal MIP presolve and runs only when the
   * higher-level presolve is enabled (i.e. `presolver != presolver_t::None`).
   * When this is `false`, probing is skipped even if presolve is otherwise on.
   */
  bool probing{true};
  /**
   * @brief Enable the block bounded-variable-elimination step of cuOpt's MIP presolve.
   *
   * Runs after trivial_presolve and eliminates blocks of functionally-determined binary auxiliary
   * variables discovered via the probing-cache implication closure, re-encoding each block's
   * projected relation as certified prime-implicate clauses. Requires the probing-cache step; a
   * no-op when no certified reduction exists.
   */
  bool block_bve{true};
  /**
   * @brief Enable the indicator-strengthening step of presolve (MIP only).
   *
   * Runs before Papilo and only when the higher-level presolve is enabled. It appends implied
   * indicator rows and lifts capacity rows by the indicator that bounds all of their members.
   */
  bool indicator_strengthening{true};
  /**
   * @brief Determinism mode for MIP solver.
   *
   * Controls the determinism behavior of the MIP solver:
   * - CUOPT_MODE_OPPORTUNISTIC (0): Default mode, allows non-deterministic
   *   parallelism for better performance
   * - CUOPT_MODE_DETERMINISTIC (1): Ensures deterministic results across runs
   *   at potential cost of performance
   */
  int determinism_mode = CUOPT_MODE_OPPORTUNISTIC;
  /**
   * @brief Random seed for the MIP solver.
   *
   * Controls the initial seed for random number generation in the solver.
   * Use -1 to generate a random seed.
   */
  i_t seed = -1;
  // this is for extracting info from different places of the solver during
  // benchmarks
  benchmark_info_t* benchmark_info_ptr = nullptr;

  // TODO check with Akif and Alice
  pdlp::pdlp_hyper_params_t hyper_params;

  mip_heuristics_hyper_params_t<i_t, f_t> heuristic_params;
  mip_diving_hyper_params_t<i_t, f_t> diving_params;
  mip_submip_hyper_params_t<i_t, f_t> submip_params;
};

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
