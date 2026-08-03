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
  i_t population_size     = 32;    // max solutions in pool
  i_t num_cpufj_threads   = 8;     // parallel CPU FJ climbers
  f_t presolve_time_ratio = 0.1;   // fraction of total time for presolve
  f_t presolve_max_time   = 60.0;  // hard cap on presolve seconds

  // Presolve budgeting. presolve_budget_policy selects how the four knobs below are derived from
  // the problem's dimensions and structure (see presolve_budget_policy.hpp); the values here are
  // the defaults the policy starts from and the literal values used by the `manual` policy.
  i_t presolve_budget_policy       = 7;     // presolve_budget_policy_t (cost)
  i_t presolve_max_rounds          = 30;    // Papilo presolve rounds cap (<=0 = Papilo default)
  i_t papilo_probing_max_badgesize = 1024;  // ceiling on Papilo's probing.minbadgesize
  f_t cuopt_presolve_work_limit    = 30.0;  // probing-cache budget, work units
  i_t probing_step_size            = 512;   // probed vars between work-budget checks
  // Weights of the probing-cache work model. Work units measure probing effort reproducibly; they
  // are not an estimate of elapsed time, so the effort-per-second they correspond to legitimately
  // differs between instances.
  f_t probe_host_overhead_work = 0.02;  // charged per probed variable
  f_t probe_iter_work          = 0.01;  // charged per multi-probe propagation iteration
  // Numerator of the probing work ceiling, divided by the cost proxy (nnz + n_cand * avg_col_len).
  // This is the bound on probing, not a backstop: measured over 240 instances it stopped every run
  // before the wall cap could fire, with a worst case of 42.8s, while the arms at 4e8 and 1e9
  // needed the wall on 2 and 8 instances respectively. It also spends the least time to do it --
  // 582s of probing across the set against 1136s at 4e8 -- and instances whose solve overran the
  // limit fell from 17 to 7.
  //
  // Tight enough to bound time is also tight enough to truncate: 34 instances end below 5% coverage
  // here against 20 under the older loose scale. That trade is deliberate and measured neutral on
  // solution quality (11.78 mean error against 12.01, inside the 0.53 run-to-run noise), because
  // probing takes its time from branch and bound.
  //
  // A larger scale is not the way to buy coverage back. The proxy predicts throughput only to
  // within ~340x and one instance (nw04) pins the scale under every reshaping tried, including
  // refitting the exponent to the measured nnz^0.65. Giving wide-row problems (avg row length above
  // ~100) their own scale is worth 2.3x on this one, and beyond that the residual is not explained
  // by any structural feature -- it needs throughput measured during probing rather than predicted
  // from the problem.
  f_t probing_work_time_scale = 1.5e8;

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
