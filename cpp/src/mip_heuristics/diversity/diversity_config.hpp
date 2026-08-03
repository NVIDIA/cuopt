/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstddef>

namespace cuopt::mathematical_optimization::mip {

struct diversity_config_t {
  double time_ratio_of_probing_cache = 0.1;
  // Backstop only. probing_work_time_scale is what bounds probing -- across 240 instances it
  // stopped every run before this could fire, worst case 42.8s -- so this exists for an instance
  // whose cost the proxy misses badly, not for routine use. Kept loose deliberately: tightening it
  // would start shaping the common case again, which is what the work ceiling replaced.
  //
  // At the default presolve share (a tenth of the solve) this cannot bind below a 1200s limit,
  // since the share itself is the smaller bound before then.
  double max_time_on_probing         = 120.0;
  int max_var_diff                   = 256;
  double default_time_limit          = 10.;
  int initial_island_size            = 3;
  int maximum_island_size            = 8;
  bool use_avg_diversity             = false;
  double generation_time_limit_ratio = 0.6;
  double max_island_gen_time         = 600;
  size_t n_sol_for_skip_init_gen     = 3;
  double max_fast_sol_time           = 10;
  double lp_run_time_if_feasible     = 2.;
  double lp_run_time_if_infeasible   = 1.;
  bool halve_population              = false;
};

}  // namespace cuopt::mathematical_optimization::mip
