/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

/**
 * @brief Tuning knobs for the recursive sub-MIP.
 */
template <typename i_t, typename f_t>
struct mip_submip_hyper_params_t {
  // Enable or disable (recursive) RINS: -1 automatic, 0 disabled, 1 enabled
  i_t rins = -1;

  // Base for calculating the target fix rate for the neighbourhood. Actual target value is
  // determined automatically according to the success and infeasible rate.
  f_t base_target_fixrate = 0.6;

  // Minimum fix rate for accepting the neighbourhood.
  f_t min_fixrate = 0.25;

  // Hard cap for the minimum fix rate for solving a sub-MIP.
  f_t min_fixrate_cap = 0.1;

  // MIP gap for the sub-MIP (unless the MIP gap from the B&B is lower)
  f_t target_mip_gap = 0.01;

  // The base node limit for the sub-MIP
  i_t node_limit_base = 200;

  // The current level in the recursion. This is an internal parameter and will set automatically.
  i_t level = 0;

  // Maximum recursion level
  i_t max_level = 10;

  // Limit the number of simplex iterations spent in the submip. Set as a factor of the total
  // number of simplex iteration from the parent B&B.
  f_t iteration_limit_ratio = 0.8;

  // Run CPU FJ over the sub-MIP
  bool enable_cpufj = true;
};
