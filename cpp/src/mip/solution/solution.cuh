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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/mip/solver_stats.hpp>
#include <mip/diversity/weights.cuh>
#include <mip/problem/problem.cuh>
#include <mip/relaxed_lp/lp_state.cuh>

#include <raft/util/cuda_dev_essentials.cuh>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class solution_t {
 public:
  solution_t(problem_t<i_t, f_t>& problem);
  solution_t(const solution_t<i_t, f_t>& other);
  solution_t& operator=(solution_t<i_t, f_t>&& other) noexcept = default;
  solution_t(solution_t<i_t, f_t>&& other)                     = default;

  void copy_from(const solution_t<i_t, f_t>& other_sol);

  void resize_copy(const solution_t<i_t, f_t>& other_sol);
  void resize_to_problem();
  void resize_to_original_problem();
  // computes the constraint values, later we can add excess computation
  void compute_constraints();
  // computes the objective and stores it in a host variable
  void compute_objective();
  // computes the total excess on constraints
  void compute_infeasibility();
  // returns the host assignment as a vector
  std::vector<f_t> get_host_assignment();
  // assigns random within bounds
  void assign_random_within_bounds(f_t ratio_of_vars_to_random_assign = 1.0,
                                   bool only_integers                 = false);
  // sets given pairs of var/value to the assignment
  void set_vars_to_values(const std::vector<thrust::pair<i_t, f_t>>& var_val_pairs);
  // copy new assignments
  void copy_new_assignment(const std::vector<f_t>& h_assignment);
  // rounds integer variables to the nearest integer val, returns whether the rounding is feasible
  bool round_nearest();
  // rounds integers to random if fractionality is between 0.25 and 0.75. otherwise, to nearest
  bool round_random_nearest(i_t n_target_random_rounds);
  // makes the approximate integer values up to INTEGRALITY TOLERANCE whole integers
  void correct_integer_precision();
  // does a reduction and returns if the current solution is feasible
  bool compute_feasibility();
  // sets the is_feasible flag to 1
  void set_feasible();
  // sets the is_feasible flag to 0
  void set_infeasible();
  // gets the is_feasible
  bool get_feasible();
  // gets the is_problem_fully_reduced
  bool get_problem_fully_reduced();
  // sets the is_problem_fully_reduced flag to 1
  void set_problem_fully_reduced();
  // computes the number of integral variables that have integral value
  i_t compute_number_of_integers();
  // computes the l2 residual value from the excess values
  f_t compute_l2_residual();
  // returns whether all integer vars are integral
  bool test_number_all_integer();
  // fixes variables and returns a problem and assignment
  std::tuple<problem_t<i_t, f_t>, rmm::device_uvector<f_t>, rmm::device_uvector<i_t>> fix_variables(
    const rmm::device_uvector<i_t>& variable_indices);
  // unfixes the variables and assigns the given assignment into the current assignment
  void unfix_variables(rmm::device_uvector<f_t>& fixed_assignment,
                       const rmm::device_uvector<i_t>& variable_map);
  // calculates the similarity radius: number of equal integers
  i_t calculate_similarity_radius(solution_t<i_t, f_t>& other_sol) const;
  // calculates the weighted quality where objective function is taken as 1.
  f_t get_quality(const weight_t<i_t, f_t>& weights);
  f_t get_quality(const rmm::device_uvector<f_t>& cstr_weights,
                  const rmm::device_scalar<f_t>& objective_weight);
  f_t get_objective();
  // use this function for logging purposes
  f_t get_user_objective();
  f_t get_total_excess();
  // brings all vars within bounds
  void clamp_within_bounds();
  mip_solution_t<i_t, f_t> get_solution(bool output_feasible,
                                        solver_stats_t<i_t, f_t> stats,
                                        bool log_stats = true);
  f_t compute_max_constraint_violation();
  f_t compute_max_int_violation();
  f_t compute_max_variable_violation();

  struct view_t {
    // let's not bloat the class for every simple getter and setters
    // if there is an index calculation or some other logic then use getters
    // void compute_all_constraints();
    // void compute_constraints_with_delta(i_t var_id, f_t delta);
    // bool is_constraint_feasible(i_t index);
    // f_t get_excess_of_constraint(i_t index);

    typename problem_t<i_t, f_t>::view_t problem;
    raft::device_span<f_t> assignment;
    raft::device_span<f_t> lower_excess;
    raft::device_span<f_t> upper_excess;
    raft::device_span<f_t> lower_slack;
    raft::device_span<f_t> upper_slack;
    raft::device_span<f_t> constraint_value;
    f_t* obj_val;
    i_t* n_feasible_constraints;
  };

  view_t view();

  // we might need to change it later as we tighten the bounds
  // and run lp on fixed parts
  problem_t<i_t, f_t>* problem_ptr;
  const raft::handle_t* handle_ptr;
  rmm::device_uvector<f_t> assignment;
  rmm::device_uvector<f_t> lower_excess;
  rmm::device_uvector<f_t> upper_excess;
  rmm::device_uvector<f_t> lower_slack;
  rmm::device_uvector<f_t> upper_slack;
  rmm::device_uvector<f_t> constraint_value;
  rmm::device_scalar<f_t> obj_val;
  rmm::device_scalar<i_t> n_feasible_constraints;

  i_t n_assigned_integers = 0;
  f_t h_obj               = 0.;
  // user objective is different than internal objective.
  // internal objective represents the minimzation problem, without any offsets or scaling factor
  // computations. user objective is the final objective we report and the objective we use for
  // logging
  f_t h_user_obj           = 0.;
  f_t h_infeasibility_cost = 0.;
  bool is_feasible         = false;
  bool is_problem_fully_reduced{false};
  bool is_scaled_{false};
  bool post_process_completed{false};
  lp_state_t<i_t, f_t> lp_state;

  // runtime TEST functions
  void test_feasibility(bool check_integer = true);
  void test_absolute_feasibility();
  void test_variable_bounds(bool check_integer = true, i_t* is_feasible = nullptr);
};

}  // namespace cuopt::linear_programming::detail
