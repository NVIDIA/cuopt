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

#include "assignment_hash_map.cuh"
#include "diversity_config.hpp"
#include "multi_armed_bandit.cuh"
#include "population.cuh"

#include "recombiners/bound_prop_recombiner.cuh"
#include "recombiners/fp_recombiner.cuh"
#include "recombiners/line_segment_recombiner.cuh"
#include "recombiners/recombiner_stats.hpp"
#include "recombiners/sub_mip.cuh"

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
  void generate_solution(f_t time_limit, bool random_start = true);
  void run_fj_alone(solution_t<i_t, f_t>& solution);
  void run_fp_alone();
  // main loop of diversity improvements
  void main_loop();
  // randomly chooses a recombiner and returns the offspring
  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a,
                                                  solution_t<i_t, f_t>& b,
                                                  recombiner_enum_t recombiner_type);
  void average_fj_weights(i_t i);
  void diversity_step(i_t max_iterations_without_improvement);
  void add_user_given_solutions(std::vector<solution_t<i_t, f_t>>& initial_sol_vector);
  population_t<i_t, f_t>* get_population_pointer() { return &population; }
  void recombine_and_ls_with_all(std::vector<solution_t<i_t, f_t>>& solutions,
                                 bool add_only_feasible = false);
  void recombine_and_ls_with_all(solution_t<i_t, f_t>& solution, bool add_only_feasible = false);
  std::pair<solution_t<i_t, f_t>, solution_t<i_t, f_t>> recombine_and_local_search(
    solution_t<i_t, f_t>& a,
    solution_t<i_t, f_t>& b,
    recombiner_enum_t recombiner_type = recombiner_enum_t::SIZE);
  void set_new_user_bound(f_t new_user_bound);
  void generate_quick_feasible_solution();
  bool check_b_b_preemption();
  void check_better_than_both(solution_t<i_t, f_t>& offspring,
                              solution_t<i_t, f_t>& sol1,
                              solution_t<i_t, f_t>& sol2);
  bool run_local_search(solution_t<i_t, f_t>& solution,
                        const weight_t<i_t, f_t>& weights,
                        timer_t& timer,
                        ls_config_t<i_t, f_t>& ls_config);

  void set_simplex_solution(const std::vector<f_t>& solution,
                            const std::vector<f_t>& dual_solution,
                            f_t objective);

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  diversity_config_t diversity_config;
  population_t<i_t, f_t> population;
  rmm::device_uvector<f_t> lp_optimal_solution;
  rmm::device_uvector<f_t> lp_dual_optimal_solution;
  std::atomic<bool> simplex_solution_exists{false};
  local_search_t<i_t, f_t> ls;
  cuopt::timer_t timer;
  bound_prop_recombiner_t<i_t, f_t> bound_prop_recombiner;
  fp_recombiner_t<i_t, f_t> fp_recombiner;
  line_segment_recombiner_t<i_t, f_t> line_segment_recombiner;
  sub_mip_recombiner_t<i_t, f_t> sub_mip_recombiner;
  all_recombine_stats recombine_stats;
  std::mt19937 rng;
  i_t current_step{0};
  solver_stats_t<i_t, f_t>& stats;
  std::vector<solution_t<i_t, f_t>> initial_sol_vector;
  mab_t mab_recombiner;
  mab_t mab_ls;
  assignment_hash_map_t<i_t, f_t> ls_hash_map;
  // mutex for the simplex solution update
  std::mutex relaxed_solution_mutex;
  // atomic for signalling pdlp to stop
  volatile int global_concurrent_halt{0};

  bool run_only_ls_recombiner{false};
  bool run_only_bp_recombiner{false};
  bool run_only_fp_recombiner{false};
  bool run_only_sub_mip_recombiner{false};
};

}  // namespace cuopt::linear_programming::detail
