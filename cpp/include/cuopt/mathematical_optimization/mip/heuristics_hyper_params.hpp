/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::mathematical_optimization {

/**
 * @brief Tuning knobs for MIP GPU heuristics.
 *
 * All fields carry their actual defaults. A config file only needs to list
 * the knobs being changed; omitted keys keep the values shown here.
 * These are registered in the unified parameter framework via solver_settings_t
 * and can be loaded from a config file with load_parameters_from_file().
 */
template <typename i_t, typename f_t>
struct mip_heuristics_hyper_params_t {
  i_t population_size   = 32;  // max solutions in pool
  i_t num_cpufj_threads = 8;   // parallel CPU FJ climbers

  // Presolve budgeting. Both are derived from the problem's dimensions and structure by default
  // (see presolve_budget_policy.hpp); a negative value asks for that rule, and any other value
  // overrides it, where <=0 removes the cap entirely.
  i_t presolve_max_rounds          = -1;  // Papilo presolve rounds cap
  i_t papilo_probing_max_badgesize = -1;  // ceiling on Papilo's probing.minbadgesize

  f_t root_lp_time_ratio                 = 0.1;     // fraction of total time for root LP
  f_t root_lp_max_time                   = 15.0;    // hard cap on root LP seconds
  f_t rins_time_limit                    = 3.0;     // per-call RINS sub-MIP time
  f_t rins_max_time_limit                = 20.0;    // ceiling for RINS adaptive time budget
  f_t rins_fix_rate                      = 0.5;     // RINS variable fix rate
  i_t stagnation_trigger                 = 3;       // FP loops w/o improvement before recombination
  i_t max_iterations_without_improvement = 8;       // diversity step depth after stagnation
  f_t initial_infeasibility_weight       = 1000.0;  // constraint violation penalty seed
  i_t n_of_minimums_for_exit             = 7000;    // FJ baseline local-minima exit threshold
  i_t enabled_recombiners                = 15;      // bitmask: 1=BP 2=FP 4=LS 8=SubMIP
  i_t cycle_detection_length             = 30;      // FP assignment cycle ring buffer
  f_t relaxed_lp_time_limit              = 1.0;     // base relaxed LP time cap in heuristics
  f_t related_vars_time_limit            = 30.0;    // time for related-variable structure build
};

}  // namespace cuopt::mathematical_optimization
