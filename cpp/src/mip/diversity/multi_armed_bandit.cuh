/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/local_search/local_search.cuh>

#include <algorithm>
#include <random>
#include <vector>

namespace cuopt::linear_programming::detail {

constexpr double recombiner_alpha = 0.05;
constexpr double ls_alpha         = 0.03;

template <typename i_t, typename f_t>
struct mab_ls_config_t {
  static constexpr i_t n_of_arms = static_cast<i_t>(ls_method_t::LS_METHODS_SIZE);

  static void get_local_search_and_lm_from_config(i_t config_id, ls_config_t<i_t, f_t>& ls_config)
  {
    ls_method_t local_search                      = ls_method_t(config_id % n_of_arms);
    ls_config.ls_method                           = local_search;
    mab_ls_config_t<i_t, f_t>::last_ls_mab_option = config_id;
  }
  static i_t last_lm_config;
  static i_t last_ls_mab_option;
};

template <typename i_t, typename f_t>
i_t mab_ls_config_t<i_t, f_t>::last_lm_config = 0;
template <typename i_t, typename f_t>
i_t mab_ls_config_t<i_t, f_t>::last_ls_mab_option = 0;

struct ls_work_normalized_reward_t {
  int option_id;
  ls_work_normalized_reward_t(int option_id) : option_id(option_id) {}

  double operator()(double factor) const { return factor; }
};

struct recombiner_work_normalized_reward_t {
  double time_in_miliseconds;
  recombiner_work_normalized_reward_t(double time_in_miliseconds)
    : time_in_miliseconds(time_in_miliseconds)
  {
  }

  double operator()(double factor) const
  {
    // normal recombiners take 2000 ms
    return factor * (std::max(0.1, 4.0 - (time_in_miliseconds / 2000)));
  }
};

struct mab_t {
  mab_t(int n_arms, int seed, double alpha, std::string bandit_name);
  // Enhanced statistics structure for UCB with exponential recency weighting
  struct mab_arm_stats_t {
    int num_pulls      = 0;    // Number of times this arm was selected
    double q_value     = 0.5;  // Exponential recency-weighted average estimate
    double last_reward = 0.0;  // Last reward received (for debugging)
  };

  // --- MAB Helper Methods ---
  void resize_mab_arm_stats(int n_arms);
  int select_mab_option();
  template <typename Func>
  void add_mab_reward(int option_id,
                      double best_of_parents_quality,
                      double best_feasible_quality,
                      double offspring_quality,
                      Func work_normalized_reward);
  int select_ucb_arm();
  int select_epsilon_greedy_arm();
  void set_last_chosen_option(int option_id);

  std::vector<mab_arm_stats_t> mab_arm_stats_;
  double mab_epsilon_ = 0.15;   // Probability of exploration in Epsilon-Greedy.
  std::mt19937 mab_rng_;        // RNG dedicated to MAB decisions.
  double mab_alpha_    = 0.05;  // Step size for exponential recency weighting
  int mab_total_steps_ = 0;     // Total number of action selections (for UCB)
  bool use_ucb_        = true;  // Flag to enable UCB vs epsilon-greedy
  std::string bandit_name;
  int last_chosen_option = -1;
};

}  // namespace cuopt::linear_programming::detail
