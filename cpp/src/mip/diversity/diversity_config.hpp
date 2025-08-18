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

namespace cuopt::linear_programming::detail {

struct diversity_config_t {
  static constexpr double time_ratio_on_init_lp              = 0.1;
  static constexpr double max_time_on_lp                     = 30;
  static constexpr double time_ratio_of_probing_cache        = 0.10;
  static constexpr double max_time_on_probing                = 60;
  static constexpr size_t max_iterations_without_improvement = 15;
  static constexpr int max_var_diff                          = 256;
  static constexpr size_t max_solutions                      = 32;
  static constexpr double initial_infeasibility_weight       = 1000.;
  static constexpr double default_time_limit                 = 10.;
  static constexpr int initial_island_size                   = 3;
  static constexpr int maximum_island_size                   = 8;
  static constexpr bool use_avg_diversity                    = false;
  static constexpr double generation_time_limit_ratio        = 0.6;
  static constexpr double max_island_gen_time                = 600;
  static constexpr size_t n_sol_for_skip_init_gen            = 3;
  static constexpr double max_fast_sol_time                  = 10;
  static constexpr double lp_run_time_if_feasible            = 15.;
  static constexpr double lp_run_time_if_infeasible          = 1;
  static constexpr double close_to_parents_ratio             = 0.1;
  static constexpr bool halve_population                     = false;
};

}  // namespace cuopt::linear_programming::detail
