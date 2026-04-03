/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::linear_programming {

/**
 * @brief Tuning knobs for MIP GPU heuristics.
 *
 * All fields carry their actual defaults. A config file only needs to list
 * the knobs being changed; omitted keys keep the values shown here.
 * These are registered in the unified parameter framework via solver_settings_t
 * and can be loaded from a config file with load_parameters_from_file().
 */
struct mip_heuristics_hyper_params_t {
  int population_size                    = 32;      // max solutions in pool
  int num_cpufj_threads                  = 8;       // parallel CPU FJ climbers
  double presolve_time_ratio             = 0.1;     // fraction of total time for presolve
  double presolve_max_time               = 60.0;    // hard cap on presolve seconds
  double root_lp_time_ratio              = 0.1;     // fraction of total time for root LP
  double root_lp_max_time                = 15.0;    // hard cap on root LP seconds
  double rins_time_limit                 = 3.0;     // per-call RINS sub-MIP time
  double rins_max_time_limit             = 20.0;    // ceiling for RINS adaptive time budget
  double rins_fix_rate                   = 0.5;     // RINS variable fix rate
  int stagnation_trigger                 = 3;       // FP loops w/o improvement before recombination
  int max_iterations_without_improvement = 8;       // diversity step depth after stagnation
  double initial_infeasibility_weight    = 1000.0;  // constraint violation penalty seed
  int n_of_minimums_for_exit             = 7000;    // FJ baseline local-minima exit threshold
  int enabled_recombiners                = 15;      // bitmask: 1=BP 2=FP 4=LS 8=SubMIP
  int cycle_detection_length             = 30;      // FP assignment cycle ring buffer
  double relaxed_lp_time_limit           = 1.0;     // base relaxed LP time cap in heuristics
  double related_vars_time_limit         = 30.0;    // time for related-variable structure build
};

}  // namespace cuopt::linear_programming
