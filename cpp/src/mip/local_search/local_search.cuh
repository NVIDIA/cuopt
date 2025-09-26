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

#include <mip/diversity/population.cuh>
#include <mip/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip/local_search/line_segment_search/line_segment_search.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <utilities/timer.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace cuopt::linear_programming::detail {

// make sure RANDOM is always the last
enum class ls_method_t : int {
  FJ_ANNEALING = 0,
  FJ_LINE_SEGMENT,
  RANDOM,
  LS_METHODS_SIZE = RANDOM
};

template <typename i_t, typename f_t>
struct ls_config_t {
  bool at_least_one_parent_feasible{true};
  f_t best_objective_of_parents{std::numeric_limits<f_t>::lowest()};
  i_t n_local_mins_for_line_segment       = 50;
  i_t n_points_to_search_for_line_segment = 5;
  i_t n_local_mins                        = 2500;
  i_t iteration_limit_for_line_segment    = 20 * n_local_mins_for_line_segment;
  i_t iteration_limit                     = 20 * n_local_mins;
  ls_method_t ls_method                   = ls_method_t::RANDOM;
};

template <typename i_t, typename f_t>
struct cpu_fj_thread_t {
  cpu_fj_thread_t();
  ~cpu_fj_thread_t();

  void cpu_worker_thread();
  void start_cpu_solver();
  void stop_cpu_solver();
  bool wait_for_cpu_solver();  // return feasibility
  void kill_cpu_solver();

  std::thread cpu_worker;
  std::mutex cpu_mutex;
  std::condition_variable cpu_cv;
  std::atomic<bool> should_stop{false};
  std::atomic<bool> cpu_thread_should_start{false};
  std::atomic<bool> cpu_thread_done{false};
  std::atomic<bool> cpu_thread_terminate{false};
  bool cpu_fj_solution_found{false};
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> fj_cpu;
  fj_t<i_t, f_t>* fj_ptr{nullptr};
};

template <typename i_t, typename f_t>
class local_search_t {
 public:
  local_search_t() = delete;
  local_search_t(mip_solver_context_t<i_t, f_t>& context,
                 rmm::device_uvector<f_t>& lp_optimal_solution_);

  void start_cpufj_scratch_threads(population_t<i_t, f_t>& population);
  void stop_cpufj_scratch_threads();
  void generate_fast_solution(solution_t<i_t, f_t>& solution, timer_t timer);
  bool generate_solution(solution_t<i_t, f_t>& solution,
                         bool perturb,
                         population_t<i_t, f_t>* population_ptr,
                         f_t time_limit = 300.);
  bool run_fj_until_timer(solution_t<i_t, f_t>& solution,
                          const weight_t<i_t, f_t>& weights,
                          timer_t timer);
  bool run_local_search(solution_t<i_t, f_t>& solution,
                        const weight_t<i_t, f_t>& weights,
                        timer_t timer,
                        const ls_config_t<i_t, f_t>& ls_config);
  bool run_fj_annealing(solution_t<i_t, f_t>& solution,
                        timer_t timer,
                        const ls_config_t<i_t, f_t>& ls_config);
  bool run_fj_line_segment(solution_t<i_t, f_t>& solution,
                           timer_t timer,
                           const ls_config_t<i_t, f_t>& ls_config);
  bool run_fj_on_zero(solution_t<i_t, f_t>& solution, timer_t timer);
  bool check_fj_on_lp_optimal(solution_t<i_t, f_t>& solution, bool perturb, timer_t timer);
  bool run_staged_fp(solution_t<i_t, f_t>& solution,
                     timer_t timer,
                     population_t<i_t, f_t>* population_ptr);
  bool run_fp(solution_t<i_t, f_t>& solution,
              timer_t timer,
              population_t<i_t, f_t>* population_ptr = nullptr);
  void resize_vectors(problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr);

  bool do_fj_solve(solution_t<i_t, f_t>& solution, fj_t<i_t, f_t>& fj, const std::string& source);

  i_t ls_threads() const { return ls_cpu_fj.size() + scratch_cpu_fj.size(); }
  void save_solution_and_add_cutting_plane(solution_t<i_t, f_t>& solution,
                                           rmm::device_uvector<f_t>& best_solution,
                                           f_t& best_objective);
  void resize_to_new_problem();
  void resize_to_old_problem(problem_t<i_t, f_t>* old_problem_ptr);
  void reset_alpha_and_run_recombiners(solution_t<i_t, f_t>& solution,
                                       problem_t<i_t, f_t>* old_problem_ptr,
                                       population_t<i_t, f_t>* population_ptr,
                                       i_t i,
                                       i_t last_unimproved_iteration,
                                       rmm::device_uvector<f_t>& best_solution,
                                       f_t& best_objective);

  mip_solver_context_t<i_t, f_t>& context;
  rmm::device_uvector<f_t>& lp_optimal_solution;
  bool lp_optimal_exists{false};
  rmm::device_uvector<f_t> fj_sol_on_lp_opt;
  fj_t<i_t, f_t> fj;
  constraint_prop_t<i_t, f_t> constraint_prop;
  line_segment_search_t<i_t, f_t> line_segment_search;
  feasibility_pump_t<i_t, f_t> fp;
  std::mt19937 rng;

  std::array<cpu_fj_thread_t<i_t, f_t>, 8> ls_cpu_fj;
  std::array<cpu_fj_thread_t<i_t, f_t>, 1> scratch_cpu_fj;
  cpu_fj_thread_t<i_t, f_t> scratch_cpu_fj_on_lp_opt;
  problem_t<i_t, f_t> problem_with_objective_cut;
  bool cutting_plane_added_for_active_run{false};
};

}  // namespace cuopt::linear_programming::detail
