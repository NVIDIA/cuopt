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

#include <mip/diversity/recombiners/recombiner_stats.hpp>
#include <mip/mip_constants.hpp>
#include <mip/problem/problem.cuh>
#include "multi_armed_bandit.cuh"

#include <cuopt/error.hpp>

namespace cuopt::linear_programming::detail {

mab_t::mab_t(int n_arms, int seed, double alpha, std::string bandit_name)
  : mab_arm_stats_(n_arms), mab_rng_(seed), bandit_name(bandit_name), mab_alpha_(alpha)
{
}

int mab_t::select_mab_option()
{
  if (mab_arm_stats_.size() == 0) {
    CUOPT_LOG_ERROR("select_mab_recombiner called with no recombiners defined.");
    cuopt_expects(false, error_type_t::RuntimeError, "No recombiners available to select in MAB.");
  }

  mab_total_steps_++;

  // Phase 1: Initial exploration - ensure each arm is tried at least once
  for (int i = 0; i < static_cast<int>(mab_arm_stats_.size()); ++i) {
    if (mab_arm_stats_[i].num_pulls == 0) {
      CUOPT_LOG_DEBUG("MAB " + bandit_name + ": Initial Pull: Arm " + std::to_string(i));
      return i;
    }
  }

  if (use_ucb_) {
    // Phase 2: UCB Action Selection
    return select_ucb_arm();
  } else {
    // Fallback to epsilon-greedy if desired
    return select_epsilon_greedy_arm();
  }
}

// UCB arm selection with confidence bounds
int mab_t::select_ucb_arm()
{
  double max_ucb_value = -std::numeric_limits<double>::infinity();
  std::vector<int> best_arms;

  for (int i = 0; i < static_cast<int>(mab_arm_stats_.size()); ++i) {
    // Calculate UCB value: Q(a) + 2*sqrt(ln(t)/N(a))
    double confidence_bound = std::sqrt((2.0 * std::log(mab_total_steps_)) /
                                        static_cast<double>(mab_arm_stats_[i].num_pulls));
    double ucb_value        = mab_arm_stats_[i].q_value + confidence_bound;

    CUOPT_LOG_DEBUG("MAB " + bandit_name + ": UCB: Arm " + std::to_string(i) +
                    ", Q=" + std::to_string(mab_arm_stats_[i].q_value) + ", CB=" +
                    std::to_string(confidence_bound) + ", UCB=" + std::to_string(ucb_value));

    constexpr double tolerance = 1e-9;
    if (ucb_value > max_ucb_value + tolerance) {
      max_ucb_value = ucb_value;
      best_arms.clear();
      best_arms.push_back(i);
    } else if (std::abs(ucb_value - max_ucb_value) < tolerance) {
      best_arms.push_back(i);
    }
  }

  if (!best_arms.empty()) {
    std::uniform_int_distribution<int> dist_tie(0, best_arms.size() - 1);
    int chosen_arm = best_arms[dist_tie(mab_rng_)];
    CUOPT_LOG_DEBUG("MAB " + bandit_name + ": UCB Selected: Arm " + std::to_string(chosen_arm) +
                    " (UCB Value: " + std::to_string(max_ucb_value) + ")");
    return chosen_arm;
  } else {
    CUOPT_LOG_ERROR("MAB " + bandit_name + ": UCB: No best arm found, falling back to random.");
    std::uniform_int_distribution<int> dist_arm(0, mab_arm_stats_.size() - 1);
    return dist_arm(mab_rng_);
  }
}

// Fallback epsilon-greedy method (preserved for compatibility)
int mab_t::select_epsilon_greedy_arm()
{
  std::uniform_real_distribution<double> dist_epsilon(0.0, 1.0);
  if (dist_epsilon(mab_rng_) < mab_epsilon_) {
    // Explore: Choose a random arm
    std::uniform_int_distribution<int> dist_arm(0, mab_arm_stats_.size() - 1);
    int random_arm = dist_arm(mab_rng_);
    CUOPT_LOG_DEBUG("MAB " + bandit_name + ": Explore: Arm " +
                    std::to_string(static_cast<int>(random_arm)));
    return random_arm;
  } else {
    // Exploit: Choose the arm with the highest Q value
    double max_q_value = -std::numeric_limits<double>::infinity();
    std::vector<int> best_arms;

    for (int i = 0; i < static_cast<int>(mab_arm_stats_.size()); ++i) {
      constexpr double tolerance = 1e-9;
      if (mab_arm_stats_[i].q_value > max_q_value + tolerance) {
        max_q_value = mab_arm_stats_[i].q_value;
        best_arms.clear();
        best_arms.push_back(i);
      } else if (std::abs(mab_arm_stats_[i].q_value - max_q_value) < tolerance) {
        best_arms.push_back(i);
      }
    }

    if (!best_arms.empty()) {
      std::uniform_int_distribution<int> dist_tie(0, best_arms.size() - 1);
      int chosen_arm = best_arms[dist_tie(mab_rng_)];
      CUOPT_LOG_DEBUG("MAB " + bandit_name + ": Exploit: Arm " +
                      std::to_string(static_cast<int>(chosen_arm)) +
                      " (Q Value: " + std::to_string(max_q_value) + ")");
      return chosen_arm;
    }
  }

  // Fallback
  std::uniform_int_distribution<int> dist_arm(0, mab_arm_stats_.size() - 1);
  return dist_arm(mab_rng_);
}

template <typename Func>
void mab_t::add_mab_reward(int option_id,
                           double best_of_parents_quality,
                           double best_feasible_quality,
                           double offspring_quality,
                           Func work_normalized_reward)
{
  double epsilon                      = max(1e-6, 1e-4 * fabs(best_feasible_quality));
  bool is_better_than_best_feasible   = offspring_quality + epsilon < best_feasible_quality;
  bool is_better_than_best_of_parents = offspring_quality + epsilon < best_of_parents_quality;
  if (option_id >= 0 && option_id < static_cast<int>(mab_arm_stats_.size())) {
    // Calculate reward based on your existing logic
    double reward = 0.0;
    if (is_better_than_best_feasible) {
      reward = 8.0;
    } else if (is_better_than_best_of_parents) {
      double factor = 0.;
      if (fabs(offspring_quality - best_feasible_quality) / (fabs(best_feasible_quality) + 1.0) <
          0.2) {
        factor = 1.;
      }
      reward = work_normalized_reward(factor);
    }

    // Update statistics
    mab_arm_stats_[option_id].num_pulls++;
    mab_arm_stats_[option_id].last_reward = reward;

    // Exponential recency-weighted average update:  Q_new = Q_old + α(R - Q_old)
    double prediction_error = reward - mab_arm_stats_[option_id].q_value;
    mab_arm_stats_[option_id].q_value += mab_alpha_ * prediction_error;

    CUOPT_LOG_DEBUG("MAB " + bandit_name + ": Reward Update: Arm " + std::to_string(option_id) +
                    ", Reward: " + std::to_string(reward) + ", is_better_than_best_of_parents: " +
                    (is_better_than_best_of_parents ? "Yes" : "No") +
                    ", Better than best: " + (is_better_than_best_feasible ? "Yes" : "No") +
                    ", Pulls: " + std::to_string(mab_arm_stats_[option_id].num_pulls) +
                    ", Q Value: " + std::to_string(mab_arm_stats_[option_id].q_value));
  } else {
    CUOPT_LOG_ERROR("MAB " + bandit_name + ": Attempted to add reward for invalid option_id: " +
                    std::to_string(option_id));
  }
}

#if MIP_INSTANTIATE_FLOAT
template struct mab_ls_config_t<int, float>;
template void mab_t::add_mab_reward<ls_work_normalized_reward_t>(
  int, double, double, double, ls_work_normalized_reward_t);
template void mab_t::add_mab_reward<recombiner_work_normalized_reward_t>(
  int, double, double, double, recombiner_work_normalized_reward_t);
#endif

#if MIP_INSTANTIATE_DOUBLE
template struct mab_ls_config_t<int, double>;
template void mab_t::add_mab_reward<ls_work_normalized_reward_t>(
  int, double, double, double, ls_work_normalized_reward_t);
template void mab_t::add_mab_reward<recombiner_work_normalized_reward_t>(
  int, double, double, double, recombiner_work_normalized_reward_t);
#endif

}  // namespace cuopt::linear_programming::detail