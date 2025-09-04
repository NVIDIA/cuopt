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

#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/local_search/line_segment_search/line_segment_search.cuh>
#include <mip/local_search/rounding/constraint_prop.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/timer.hpp>

#include <thrust/count.h>

#include <deque>

namespace cuopt::linear_programming::detail {

constexpr double default_alpha                  = 0.99;
constexpr double distance_to_check_for_feasible = 0.01;

template <typename i_t, typename f_t>
struct cycle_queue_t {
  cycle_queue_t(problem_t<i_t, f_t>& problem) : curr_recent_sol(cycle_detection_length - 1)
  {
    for (i_t i = 0; i < cycle_detection_length; ++i) {
      recent_solutions.emplace_back(
        rmm::device_uvector<f_t>(problem.n_variables, problem.handle_ptr->get_stream()));
    }
  }

  bool check_same_solution(solution_t<i_t, f_t>& solution,
                           rmm::device_uvector<f_t>& recent_solution)
  {
    const f_t* other_ptr        = recent_solution.data();
    const f_t* curr_assignement = solution.assignment.data();
    i_t n_equal_integers        = thrust::count_if(
      solution.handle_ptr->get_thrust_policy(),
      solution.problem_ptr->integer_indices.begin(),
      solution.problem_ptr->integer_indices.end(),
      [other_ptr, curr_assignement, pb = solution.problem_ptr->view()] __device__(i_t idx) {
        return pb.integer_equal(other_ptr[idx], curr_assignement[idx]);
      });
    return n_equal_integers == solution.problem_ptr->n_integer_vars;
  }

  bool check_cycle(solution_t<i_t, f_t>& solution)
  {
    for (i_t i = 0; i < cycle_detection_length; ++i) {
      i_t sol_idx  = (curr_recent_sol + cycle_detection_length - i) % cycle_detection_length;
      bool is_same = check_same_solution(solution, recent_solutions[sol_idx]);
      if (is_same) {
        CUOPT_LOG_DEBUG(
          "Detected same solution recent order %d",
          (curr_recent_sol + cycle_detection_length - sol_idx) % cycle_detection_length);
        return true;
      }
    }
    return false;
  }

  void update_recent_solutions(solution_t<i_t, f_t>& solution)
  {
    curr_recent_sol++;
    if (curr_recent_sol == cycle_detection_length) { curr_recent_sol = 0; }
    // update current integer solution
    raft::copy(recent_solutions[curr_recent_sol].data(),
               solution.assignment.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
  }

  void reset(solution_t<i_t, f_t>& solution)
  {
    for (i_t i = 0; i < cycle_detection_length; ++i) {
      recent_solutions[i].resize(solution.problem_ptr->n_variables,
                                 solution.handle_ptr->get_stream());
      thrust::fill(solution.handle_ptr->get_thrust_policy(),
                   recent_solutions[i].begin(),
                   recent_solutions[i].end(),
                   NAN);
    }
  }

  std::vector<rmm::device_uvector<f_t>> recent_solutions;
  const i_t cycle_detection_length = 30;
  i_t curr_recent_sol;
  i_t n_iterations_without_cycle = 0;
};

struct fp_config_t {
  double alpha                           = default_alpha;
  double alpha_decrease_factor           = 0.9;
  bool check_distance_cycle              = true;
  int first_stage_kk                     = 70;
  double cycle_distance_reduction_ration = 0.1;
};

template <typename i_t, typename f_t>
class feasibility_pump_t {
 public:
  feasibility_pump_t() = delete;
  feasibility_pump_t(mip_solver_context_t<i_t, f_t>& context,
                     fj_t<i_t, f_t>& fj,
                     //                     fj_tree_t<i_t, f_t>& fj_tree_,
                     constraint_prop_t<i_t, f_t>& constraint_prop_,
                     line_segment_search_t<i_t, f_t>& line_segment_search_,
                     rmm::device_uvector<f_t>& lp_optimal_solution_);

  void adjust_objective_with_original(solution_t<i_t, f_t>& solution,
                                      std::vector<f_t>& dist_objective,
                                      bool longer_lp_run = false);
  bool linear_project_onto_polytope(solution_t<i_t, f_t>& solution,
                                    f_t proximity_to_polytope,
                                    bool longer_lp_run = false);

  void perturbate(solution_t<i_t, f_t>& solution);
  bool run_fj_cycle_escape(solution_t<i_t, f_t>& solution);
  bool run_single_fp_descent(solution_t<i_t, f_t>& solution);
  bool round(solution_t<i_t, f_t>& solution);
  bool handle_cycle(solution_t<i_t, f_t>& solution);
  bool restart_fp(solution_t<i_t, f_t>& solution);
  bool test_number_all_integer(solution_t<i_t, f_t>& solution);
  bool check_distance_cycle(solution_t<i_t, f_t>& solution);
  void reset();
  void resize_vectors(problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr);
  void save_best_excess_solution(solution_t<i_t, f_t>& solution);
  bool random_round_with_fj(solution_t<i_t, f_t>& solution, timer_t& round_timer);
  bool round_multiple_points(solution_t<i_t, f_t>& solution);
  void relax_general_integers(solution_t<i_t, f_t>& solution);
  void revert_relaxation(solution_t<i_t, f_t>& solution);
  bool test_fj_feasible(solution_t<i_t, f_t>& solution, f_t time_limit);

  mip_solver_context_t<i_t, f_t>& context;
  // keep a reference from upstream local search
  fj_t<i_t, f_t>& fj;
  // fj_tree_t<i_t, f_t>& fj_tree;
  line_segment_search_t<i_t, f_t>& line_segment_search;
  cycle_queue_t<i_t, f_t> cycle_queue;
  constraint_prop_t<i_t, f_t>& constraint_prop;
  fp_config_t config;
  rmm::device_uvector<f_t> last_rounding;
  rmm::device_uvector<f_t> last_projection;
  rmm::device_uvector<var_t> orig_variable_types;
  f_t best_excess;
  rmm::device_uvector<f_t> best_excess_solution;
  rmm::device_uvector<f_t>& lp_optimal_solution;
  std::mt19937 rng;
  std::deque<f_t> last_distances;
  f_t last_lp_time;
  f_t total_fp_time_until_cycle;
  f_t fp_fj_cycle_time_begin;
  f_t proj_and_round_time;
  f_t proj_begin;
  i_t n_fj_single_descents;
  i_t max_n_of_integers = 0;
  cuopt::timer_t timer;
};

}  // namespace cuopt::linear_programming::detail
