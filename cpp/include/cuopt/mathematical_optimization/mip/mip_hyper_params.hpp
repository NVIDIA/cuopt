/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <cstdint>
#include <limits>

namespace cuopt::mathematical_optimization::mip {

/**
 * @brief Tuning knobs for the MIP solver.
 *
 * Each member is prefixed with the part of the solver that reads it. All fields carry their
 * actual defaults. A config file only needs to list the knobs being changed; omitted keys keep
 * the values shown here. These are registered in the unified parameter framework via
 * solver_settings_t and can be loaded from a config file with load_parameters_from_file().
 */
template <typename i_t, typename f_t>
struct mip_hyper_params_t {
  // ---------------------------------------------------------------------------------------------
  // Presolve
  // ---------------------------------------------------------------------------------------------

  // Presolve budgeting. Both are derived from the problem's dimensions and structure by default
  // (see presolve_budget_policy.hpp); a negative value asks for that rule, and any other value
  // overrides it, where <=0 removes the cap entirely.
  i_t presolve_max_rounds                   = -1;  // Papilo presolve rounds cap
  i_t presolve_papilo_probing_max_badgesize = -1;  // ceiling on Papilo's probing.minbadgesize

  // Eliminate blocks of functionally-determined binary auxiliary variables discovered via the
  // probing-cache implication closure. Requires the probing-cache step.
  bool presolve_block_bve = true;

  // Append implied indicator rows and lift capacity rows by the indicator that bounds all of
  // their members. Runs before Papilo and only when presolve is enabled.
  bool presolve_indicator_strengthening = true;

  // ---------------------------------------------------------------------------------------------
  // GPU heuristics
  // ---------------------------------------------------------------------------------------------
  i_t heuristic_population_size                    = 32;      // max solutions in pool
  i_t heuristic_num_cpufj_threads                  = 8;       // parallel CPU FJ climbers
  f_t heuristic_root_lp_time_ratio                 = 0.1;     // fraction of total time for root LP
  f_t heuristic_root_lp_max_time                   = 15.0;    // hard cap on root LP seconds
  i_t heuristic_stagnation_trigger                 = 3;       // FP loops w/o improvement
  i_t heuristic_max_iterations_without_improvement = 8;       // diversity step depth
  f_t heuristic_initial_infeasibility_weight       = 1000.0;  // constraint violation penalty seed
  i_t heuristic_n_of_minimums_for_exit             = 7000;    // FJ local-minima exit threshold
  i_t heuristic_enabled_recombiners                = 15;      // bitmask: 1=BP 2=FP 4=LS 8=SubMIP
  i_t heuristic_cycle_detection_length             = 30;      // FP assignment cycle ring buffer
  f_t heuristic_relaxed_lp_time_limit              = 1.0;     // base relaxed LP time cap
  f_t heuristic_related_vars_time_limit            = 2.0;     // related-variable structure build

  // ---------------------------------------------------------------------------------------------
  // Branch-and-bound
  // ---------------------------------------------------------------------------------------------

  // Set the number of workers for best-first search, diving and sub-MIP. This is given in terms
  // of the total number of threads allocated to B&B and do not need to sum to 1 as
  // the OpenMP will automatically switch between them at every sync point. The only exception is
  // that BFS workers are long-lived so setting a high number here may prevent the other
  // workers to execute.
  f_t bnb_bfs_workers_ratio    = 0.5;
  f_t bnb_diving_workers_ratio = 0.5;
  f_t bnb_submip_workers_ratio = 0.125;

  // Indicate the chance in which a worker can steal a node from another worker.
  f_t bnb_steal_chance = 0.05;

  // Indicate the number of nodes to steal in each attempt
  i_t bnb_nodes_per_steal = 10;

  // Set the maximum number of steal attempts before giving up
  i_t bnb_max_steal_attempts = 3;

  // Root heuristics: -1 (automatic) / 0 (disable) / 1 (enable)
  i_t bnb_root_heuristics = -1;

  // Limit of the total number of simplex iterations in B&B
  std::int64_t bnb_simplex_iteration_limit = std::numeric_limits<std::int64_t>::max();

  // ---------------------------------------------------------------------------------------------
  // Diving
  // ---------------------------------------------------------------------------------------------

  // -1 automatic, 0 disabled, 1 enabled
  i_t diving_line_search   = -1;
  i_t diving_pseudocost    = -1;
  i_t diving_guided        = -1;
  i_t diving_coefficient   = -1;
  i_t diving_farkas        = -1;
  i_t diving_vector_length = -1;

  // The maximum number of nodes when performing a dive.
  i_t diving_node_limit = 500;

  // The maximum number of dual simplex iteration allowed
  // in a single dive.
  f_t diving_iteration_limit_factor          = 0.05;
  std::int64_t diving_iteration_limit_offset = 10000;

  // The maximum backtracking allowed.
  i_t diving_backtrack_limit = 5;

  // For the Farkas diving to be effective, the coefficients in the objective function
  // must have distinct values. The low tolerance here disables Farkas diving for
  // set covering/partitioning, where all coefficients have the same value.
  f_t diving_farkas_obj_dynamism_tol = 1E-4;

  // If a given diving heuristic found a new incumbent, show the corresponding
  // symbol in the first column of the log row. When false, every dive collapses
  // to 'D'. Otherwise,
  //   B = best-first
  //   H = heuristics
  //   C = coefficient diving
  //   L = line-search diving
  //   P = pseudocost diving
  //   G = guided diving
  //   U = unknown
  bool diving_show_type = false;

  // ---------------------------------------------------------------------------------------------
  // Recursive sub-MIP
  // ---------------------------------------------------------------------------------------------

  // Enable or disable (recursive) RINS/RENS: -1 automatic, 0 disabled, 1 enabled
  i_t submip_rins     = -1;
  i_t submip_rens     = -1;
  i_t submip_mutation = -1;

  // Base for calculating the target fix rate for the neighbourhood. Actual target value is
  // determined automatically according to the success and infeasible rate.
  f_t submip_base_target_fixrate = 0.6;

  // Minimum fix rate for accepting the neighbourhood.
  f_t submip_min_fixrate = 0.25;

  // Hard cap for the minimum fix rate for solving a sub-MIP.
  f_t submip_min_fixrate_cap = 0.1;

  // MIP gap for the sub-MIP (unless the MIP gap from the B&B is lower)
  f_t submip_target_mip_gap = 0.01;

  // The base node limit for the sub-MIP
  i_t submip_node_limit_offset = 200;

  // The base iteration limit for the sub-MIP
  i_t submip_iteration_limit_offset = 10000;

  // The current level in the recursion. This is an internal parameter and will set automatically.
  i_t submip_level = 0;

  // Maximum recursion level
  i_t submip_max_level = 10;

  // Limit the number of simplex iterations spent in the submip. Set as a factor of the total
  // number of simplex iteration from the parent B&B.
  f_t submip_iteration_limit_ratio = 0.8;

  // If there is not enough variables fixed or we already found an improving solution,
  // perform a short DFS to quickly find a feasible solution. This setting controls
  // the maximum number of nodes allow for backtracking.
  i_t submip_dfs_max_backtrack = 5;

  // How many variables a single round can fix. Set in terms of ratio of
  // (1 - current fixrate).
  f_t submip_round_close_ratio = 0.8;

  // Run CPU FJ over the sub-MIP
  bool submip_enable_cpufj = true;
};

}  // namespace cuopt::mathematical_optimization::mip
