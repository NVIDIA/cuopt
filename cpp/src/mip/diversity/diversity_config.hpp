/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>

namespace cuopt::linear_programming::detail {

struct diversity_config_t {
  double time_ratio_on_init_lp        = 0.1;
  double max_time_on_lp               = 15.0;
  double time_ratio_of_probing_cache  = 0.04;
  double max_time_on_probing          = 15.0;
  int max_var_diff                    = 256;
  size_t max_solutions                = 32;
  double initial_infeasibility_weight = 1000.;
  double default_time_limit           = 10.;
  int initial_island_size             = 3;
  int maximum_island_size             = 8;
  bool use_avg_diversity              = false;
  double generation_time_limit_ratio  = 0.6;
  double max_island_gen_time          = 600;
  size_t n_sol_for_skip_init_gen      = 3;
  double max_fast_sol_time            = 10;
  double lp_run_time_if_feasible      = 2.;
  double lp_run_time_if_infeasible    = 1.;
  bool halve_population               = false;
};

}  // namespace cuopt::linear_programming::detail
