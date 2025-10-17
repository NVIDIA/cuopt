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

struct bp_recombiner_config_t {
  static constexpr double bounds_prop_time_limit          = 2.;
  static constexpr double lp_after_bounds_prop_time_limit = 2.;
  // number of repair iterations even if it fails during the repair
  static constexpr size_t n_repair_iterations          = 10;
  static constexpr size_t initial_n_of_vars_from_other = 200;
  static constexpr size_t max_different_var_limit      = 10000;
  static constexpr size_t min_different_var_limit      = 20;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in BP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in BP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

struct ls_recombiner_config_t {
  // line segment related configs
  // FIXME: not implemented yet
  static constexpr bool use_fj_for_rounding            = false;
  static constexpr int n_points_to_search              = 20;
  static constexpr double time_limit                   = 2.;
  static constexpr size_t initial_n_of_vars_from_other = 200;
  static constexpr size_t max_different_var_limit      = 10000;
  static constexpr size_t min_different_var_limit      = 20;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in LS recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in LS recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

struct fp_recombiner_config_t {
  static constexpr double infeasibility_detection_time_limit = 0.05;
  static constexpr double fp_time_limit                      = 2.;
  static constexpr double alpha                              = 0.99;
  static constexpr double alpha_decrease_factor              = 0.9;
  static constexpr size_t initial_n_of_vars_from_other       = 200;
  static constexpr size_t max_different_var_limit            = 10000;
  static constexpr size_t min_different_var_limit            = 20;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in FP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in FP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

struct sub_mip_recombiner_config_t {
  static constexpr size_t max_continuous_vars                = 5000;
  static constexpr double sub_mip_time_limit                 = 2.;
  static constexpr double infeasibility_detection_time_limit = 0.05;
  static constexpr size_t initial_n_of_vars_from_other       = 40;
  static constexpr size_t max_different_var_limit            = 500;
  static constexpr size_t min_different_var_limit            = 10;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in SUB_MIP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in SUB_MIP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

}  // namespace cuopt::linear_programming::detail
