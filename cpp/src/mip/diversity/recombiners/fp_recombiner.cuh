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

#include "recombiner.cuh"

#include <mip/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip/local_search/rounding/constraint_prop.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/seed_generator.cuh>

#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class fp_recombiner_t : public recombiner_t<i_t, f_t> {
 public:
  fp_recombiner_t(mip_solver_context_t<i_t, f_t>& context,
                  i_t n_vars,
                  feasibility_pump_t<i_t, f_t>& fp_,
                  const raft::handle_t* handle_ptr)
    : recombiner_t<i_t, f_t>(context, n_vars, handle_ptr),
      vars_to_fix(n_vars, handle_ptr->get_stream()),
      fp(fp_)
  {
  }

  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a, solution_t<i_t, f_t>& b)
  {
    raft::common::nvtx::range fun_scope("FP recombiner");

    // copy the solution from A
    solution_t<i_t, f_t> offspring(a);
    // find same values and populate it to offspring
    this->assign_same_integer_values(a, b, offspring);
    const i_t n_remaining_vars = this->n_remaining.value(a.handle_ptr->get_stream());
    // partition to get only integer uncommon
    auto iter                  = thrust::stable_partition(a.handle_ptr->get_thrust_policy(),
                                         this->remaining_indices.begin(),
                                         this->remaining_indices.begin() + n_remaining_vars,
                                         [a_view = a.view()] __device__(i_t idx) {
                                           if (a_view.problem.is_integer_var(idx)) { return true; }
                                           return false;
                                         });
    i_t h_n_remaining_integers = iter - this->remaining_indices.begin();
    i_t n_common_integers      = a.problem_ptr->n_integer_vars - h_n_remaining_integers;
    if (h_n_remaining_integers == 0 || n_common_integers == 0) {
      CUOPT_LOG_DEBUG("All integers are common or different in FP recombiner, returning A!");
      return std::make_pair(offspring, false);
    }
    CUOPT_LOG_DEBUG("n_integer_vars from A/B %d n_integer_vars from B %d",
                    n_common_integers,
                    h_n_remaining_integers);
    vars_to_fix.resize(n_common_integers, a.handle_ptr->get_stream());
    // set difference needs two sorted arrays
    thrust::sort(a.handle_ptr->get_thrust_policy(),
                 this->remaining_indices.data(),
                 this->remaining_indices.data() + h_n_remaining_integers);
    cuopt_assert((thrust::is_sorted(a.handle_ptr->get_thrust_policy(),
                                    a.problem_ptr->integer_indices.begin(),
                                    a.problem_ptr->integer_indices.end())),
                 "vars_to_fix should be sorted!");
    // get the variables to fix (common variables)
    iter = thrust::set_difference(a.handle_ptr->get_thrust_policy(),
                                  a.problem_ptr->integer_indices.begin(),
                                  a.problem_ptr->integer_indices.end(),
                                  this->remaining_indices.data(),
                                  this->remaining_indices.data() + h_n_remaining_integers,
                                  vars_to_fix.begin());
    cuopt_assert(iter - vars_to_fix.begin() == n_common_integers, "The size should match!");
    cuopt_assert((thrust::is_sorted(a.handle_ptr->get_thrust_policy(),
                                    vars_to_fix.data(),
                                    vars_to_fix.data() + n_common_integers)),
                 "vars_to_fix should be sorted!");
    const f_t tolerance                                  = 1e-2;
    const f_t lp_time                                    = 0.5;
    auto [fixed_problem, fixed_assignment, variable_map] = offspring.fix_variables(vars_to_fix);
    fixed_problem.check_problem_representation(true);
    const bool check_feas            = false;
    const bool return_first_feasible = false;
    const bool save_state            = false;
    // every sub problem is different,so it is very hard to find a valid initial solution
    lp_state_t<i_t, f_t> lp_state = this->context.lp_state;
    auto solver_response          = get_relaxed_lp_solution(fixed_problem,
                                                   fixed_assignment,
                                                   lp_state,
                                                   tolerance,
                                                   lp_time,
                                                   check_feas,
                                                   return_first_feasible,
                                                   save_state);
    // brute force rounding threshold is 8
    const bool run_fp = fixed_problem.n_integer_vars > 8;
    if (run_fp) {
      problem_t<i_t, f_t>* orig_problem_ptr = offspring.problem_ptr;
      offspring.problem_ptr                 = &fixed_problem;
      rmm::device_uvector<f_t> old_assignment(offspring.assignment,
                                              offspring.handle_ptr->get_stream());
      offspring.handle_ptr->sync_stream();
      offspring.assignment = std::move(fixed_assignment);
      offspring.lp_state   = std::move(lp_state);
      cuopt_func_call(offspring.test_variable_bounds(false));
      timer_t timer((f_t)2.);
      fp.timer = timer;
      fp.cycle_queue.reset(offspring);
      fp.reset();
      fp.resize_vectors(*offspring.problem_ptr, offspring.handle_ptr);
      bool is_feasible = fp.run_single_fp_descent(offspring);
      if (is_feasible) { CUOPT_LOG_DEBUG("FP after recombiner, found feasible!"); }
      CUOPT_LOG_DEBUG("FP completed after recombiner!");
      offspring.handle_ptr->sync_stream();
      offspring.problem_ptr = orig_problem_ptr;
      fixed_assignment      = std::move(offspring.assignment);
      offspring.assignment  = std::move(old_assignment);
      offspring.handle_ptr->sync_stream();
    }
    // unfix the assignment on given result no matter if it is feasible
    offspring.unfix_variables(fixed_assignment, variable_map);
    if (!run_fp) { offspring.round_nearest(); }
    cuopt_assert(offspring.test_number_all_integer(), "All must be integers after offspring");
    offspring.compute_feasibility();
    bool same_as_parents = this->check_if_offspring_is_same_as_parents(offspring, a, b);
    return std::make_pair(offspring, !same_as_parents);
  }
  rmm::device_uvector<i_t> vars_to_fix;
  feasibility_pump_t<i_t, f_t>& fp;
};

}  // namespace cuopt::linear_programming::detail
