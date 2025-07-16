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

#include "population.cuh"

#include "recombiners/bound_prop_recombiner.cuh"
#include "recombiners/fp_recombiner.cuh"
#include "recombiners/line_segment_recombiner.cuh"
#include "recombiners/recombiner_stats.hpp"

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <mip/local_search/local_search.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class diversity_manager_t {
 public:
  diversity_manager_t(mip_solver_context_t<i_t, f_t>& context);
  bool run_presolve(f_t time_limit);
  solution_t<i_t, f_t> run_solver();
  solution_t<i_t, f_t> generate_solution(f_t time_limit, bool random_start = true);
  // generates initial solutions
  void generate_initial_solutions();
  void run_fj_alone(solution_t<i_t, f_t>& solution);
  // main loop of diversity improvements
  void main_loop();
  // randomly chooses a recombiner and returns the offspring
  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a, solution_t<i_t, f_t>& b);
  bool regenerate_solutions();
  void generate_add_solution(std::vector<solution_t<i_t, f_t>>& initial_sol_vector,
                             f_t time_limit,
                             bool random_start = true);
  void average_fj_weights(i_t i);
  void diversity_step();
  std::vector<solution_t<i_t, f_t>> generate_more_solutions();
  void add_user_given_solutions(std::vector<solution_t<i_t, f_t>>& initial_sol_vector);
  population_t<i_t, f_t>* get_population_pointer() { return &population; }
  void recombine_and_ls_with_all(std::vector<solution_t<i_t, f_t>>& solutions);
  void recombine_and_ls_with_all(solution_t<i_t, f_t>& solution);
  std::pair<solution_t<i_t, f_t>, solution_t<i_t, f_t>> recombine_and_local_search(
    solution_t<i_t, f_t>& a, solution_t<i_t, f_t>& b);
  void set_new_user_bound(f_t new_user_bound);
  void generate_quick_feasible_solution();
  bool check_b_b_preemption();
  void check_better_than_both(solution_t<i_t, f_t>& offspring,
                              solution_t<i_t, f_t>& sol1,
                              solution_t<i_t, f_t>& sol2);

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  population_t<i_t, f_t> population;
  rmm::device_uvector<f_t> lp_optimal_solution;
  local_search_t<i_t, f_t> ls;
  cuopt::timer_t timer;
  bound_prop_recombiner_t<i_t, f_t> bound_prop_recombiner;
  fp_recombiner_t<i_t, f_t> fp_recombiner;
  line_segment_recombiner_t<i_t, f_t> line_segment_recombiner;
  all_recombine_stats recombine_stats;
  std::mt19937 rng;
  i_t current_step{0};
  solver_stats_t<i_t, f_t>& stats;
  std::vector<solution_t<i_t, f_t>> initial_sol_vector;

  // Enhanced statistics structure for UCB with exponential recency weighting
  struct mab_arm_stats_t {
    int num_pulls      = 0;    // Number of times this arm was selected
    double q_value     = 0.5;  // Exponential recency-weighted average estimate
    double last_reward = 0.0;  // Last reward received (for debugging)
  };
  std::vector<mab_arm_stats_t> mab_arm_stats_;
  double mab_epsilon_ = 0.15;   // Probability of exploration in Epsilon-Greedy.
  std::mt19937 mab_rng_;        // RNG dedicated to MAB decisions.
  double mab_alpha_    = 0.05;  // Step size for exponential recency weighting
  int mab_total_steps_ = 0;     // Total number of action selections (for UCB)
  bool use_ucb_        = true;  // Flag to enable UCB vs epsilon-greedy

  // --- MAB Helper Methods ---
  recombiner_enum_t select_mab_recombiner();
  void add_mab_reward(recombiner_enum_t recombiner_id,
                      double best_of_parents_quality,
                      double best_feasible_quality,
                      double offspring_quality,
                      double recombination_time_in_miliseconds);
  recombiner_enum_t select_ucb_arm();
  recombiner_enum_t select_epsilon_greedy_arm();
  bool run_only_ls_recombiner{false};
  bool run_only_bp_recombiner{false};
  bool run_only_fp_recombiner{false};
};

}  // namespace cuopt::linear_programming::detail
