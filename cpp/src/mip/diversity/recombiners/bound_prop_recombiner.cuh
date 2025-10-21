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

#include <mip/local_search/rounding/constraint_prop.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/seed_generator.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class bound_prop_recombiner_t : public recombiner_t<i_t, f_t> {
 public:
  bound_prop_recombiner_t(mip_solver_context_t<i_t, f_t>& context,
                          i_t n_vars,
                          constraint_prop_t<i_t, f_t>& constraint_prop_,
                          const raft::handle_t* handle_ptr)
    : recombiner_t<i_t, f_t>(context, n_vars, handle_ptr),
      constraint_prop(constraint_prop_),
      rng(cuopt::seed_generator::get_seed()),
      vars_to_fix(n_vars, handle_ptr->get_stream())
  {
  }

  void get_probing_values_for_infeasible(
    solution_t<i_t, f_t>& guiding,
    solution_t<i_t, f_t>& other,
    solution_t<i_t, f_t>& offspring,
    rmm::device_uvector<thrust::pair<f_t, f_t>>& probing_values,
    i_t n_vars_from_other)
  {
    raft::common::nvtx::range fun_scope("get_probing_values_for_infeasible");
    auto guiding_view   = guiding.view();
    auto other_view     = other.view();
    auto offspring_view = offspring.view();
    const f_t int_tol   = guiding.problem_ptr->tolerances.integrality_tolerance;
    // this is to give two possibilities to round in case of conflict
    thrust::for_each(
      guiding.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(guiding.problem_ptr->n_variables),
      [guiding_view, other_view, probing_values = probing_values.data()] __device__(i_t idx) {
        f_t guiding_val = guiding_view.assignment[idx];
        f_t other_val   = other_view.assignment[idx];
        cuopt_assert(guiding_view.problem.check_variable_within_bounds(idx, guiding_val), "");
        cuopt_assert(other_view.problem.check_variable_within_bounds(idx, other_val), "");
        probing_values[idx] = thrust::make_pair(guiding_val, other_val);
      });
    // populate remaining N integers randomly/average from each solution
    thrust::for_each(
      guiding.handle_ptr->get_thrust_policy(),
      this->remaining_indices.data(),
      this->remaining_indices.data() + n_vars_from_other,
      [guiding_view,
       other_view,
       offspring_view,
       int_tol,
       probing_values = probing_values.data(),
       seed           = cuopt::seed_generator::get_seed()] __device__(i_t idx) {
        f_t guiding_val = guiding_view.assignment[idx];
        f_t other_val   = other_view.assignment[idx];
        cuopt_assert(guiding_view.problem.check_variable_within_bounds(idx, guiding_val), "");
        cuopt_assert(other_view.problem.check_variable_within_bounds(idx, other_val), "");
        f_t avg_val = (other_val + guiding_val) / 2;
        if (guiding_view.problem.is_integer_var(idx)) {
          raft::random::PCGenerator rng(seed, idx, 0);
          if (rng.next_u32() % 2) { cuda::std::swap(other_val, guiding_val); }
          cuopt_assert(is_integer<f_t>(other_val, int_tol), "The value must be integer");
          f_t second_val      = round(avg_val) == other_val ? guiding_val : round(avg_val);
          probing_values[idx] = thrust::make_pair(other_val, second_val);
          // assign some floating value, so that they can be rounded by bounds prop
          auto bounds = guiding_view.problem.variable_bounds[idx];
          f_t lb      = get_lower(bounds);
          f_t ub      = get_upper(bounds);
          if (integer_equal<f_t>(lb, ub, int_tol)) {
            cuopt_assert(false, "The var values must be different in A and B!");
          } else if (isfinite(lb)) {
            offspring_view.assignment[idx] = lb + 0.1;
          } else {
            offspring_view.assignment[idx] = ub - 0.1;
          }
        } else {
          // if the var is continuous, take the average
          offspring_view.assignment[idx] = avg_val;
        }
      });
  }

  void get_probing_values_for_feasible(solution_t<i_t, f_t>& guiding,
                                       solution_t<i_t, f_t>& other,
                                       solution_t<i_t, f_t>& offspring,
                                       rmm::device_uvector<thrust::pair<f_t, f_t>>& probing_values,
                                       i_t n_vars_from_other,
                                       rmm::device_uvector<i_t>& variable_map)
  {
    raft::common::nvtx::range fun_scope("get_probing_values_for_feasible");
    cuopt_assert(n_vars_from_other == offspring.problem_ptr->n_integer_vars,
                 "The number of vars from other should match!");
    auto guiding_view   = guiding.view();
    auto other_view     = other.view();
    auto offspring_view = offspring.view();
    const f_t int_tol   = guiding.problem_ptr->tolerances.integrality_tolerance;
    thrust::for_each(
      guiding.handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator(0lu),
      thrust::make_counting_iterator(variable_map.size()),
      [guiding_view,
       other_view,
       offspring_view,
       int_tol,
       probing_values = make_span(probing_values),
       variable_map   = make_span(variable_map)] __device__(size_t idx) {
        f_t other_val   = other_view.assignment[variable_map[idx]];
        f_t guiding_val = guiding_view.assignment[variable_map[idx]];
        cuopt_assert(other_view.problem.check_variable_within_bounds(variable_map[idx], other_val),
                     "");
        cuopt_assert(
          guiding_view.problem.check_variable_within_bounds(variable_map[idx], guiding_val), "");
        f_t avg_val                    = (other_val + guiding_val) / 2;
        probing_values[idx]            = thrust::make_pair(guiding_val, other_val);
        offspring_view.assignment[idx] = avg_val;
      });
  }

  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a,
                                                  solution_t<i_t, f_t>& b,
                                                  const weight_t<i_t, f_t>& weights)
  {
    raft::common::nvtx::range fun_scope("bound_prop_recombiner");
    auto& guiding_solution = a.get_feasible() ? a : b;
    auto& other_solution   = a.get_feasible() ? b : a;
    // copy the solution from guiding
    solution_t<i_t, f_t> offspring(guiding_solution);
    // find same values and populate it to offspring
    i_t n_different_vars = this->assign_same_integer_values(a, b, offspring);
    CUOPT_LOG_DEBUG("BP rec: Number of different variables %d MAX_VARS %d",
                    n_different_vars,
                    bp_recombiner_config_t::max_n_of_vars_from_other);
    i_t n_vars_from_other  = n_different_vars;
    i_t fixed_from_guiding = 0;
    i_t fixed_from_other   = 0;
    if (n_different_vars > (i_t)bp_recombiner_config_t::max_n_of_vars_from_other) {
      fixed_from_guiding = n_vars_from_other - bp_recombiner_config_t::max_n_of_vars_from_other;
      n_vars_from_other  = bp_recombiner_config_t::max_n_of_vars_from_other;
      thrust::default_random_engine g{(unsigned int)cuopt::seed_generator::get_seed()};
      thrust::shuffle(a.handle_ptr->get_thrust_policy(),
                      this->remaining_indices.data(),
                      this->remaining_indices.data() + n_different_vars,
                      g);
    }
    i_t n_vars_from_guiding = a.problem_ptr->n_integer_vars - n_vars_from_other;
    CUOPT_LOG_DEBUG(
      "n_vars_from_guiding %d n_vars_from_other %d", n_vars_from_guiding, n_vars_from_other);
    // if either all integers are from A(meaning all are common) or all integers are from B(meaning
    // all are different), return
    if (n_vars_from_guiding == 0 || n_vars_from_other == 0) {
      CUOPT_LOG_DEBUG("Returning false because all vars are common or different");
      return std::make_pair(offspring, false);
    }

    cuopt_assert(a.problem_ptr == b.problem_ptr,
                 "The two solutions should not refer to different problems");
    const f_t lp_run_time_after_feasible = bp_recombiner_config_t::lp_after_bounds_prop_time_limit;
    constraint_prop.max_n_failed_repair_iterations = bp_recombiner_config_t::n_repair_iterations;
    rmm::device_uvector<thrust::pair<f_t, f_t>> probing_values(a.problem_ptr->n_variables,
                                                               a.handle_ptr->get_stream());
    probing_config_t<i_t, f_t> probing_config(a.problem_ptr->n_variables, a.handle_ptr);
    if (guiding_solution.get_feasible() && !a.problem_ptr->expensive_to_fix_vars) {
      this->compute_vars_to_fix(offspring, vars_to_fix, n_vars_from_other, n_vars_from_guiding);
      auto [fixed_problem, fixed_assignment, variable_map] = offspring.fix_variables(vars_to_fix);
      timer_t timer(bp_recombiner_config_t::bounds_prop_time_limit);
      rmm::device_uvector<f_t> old_assignment(offspring.assignment,
                                              offspring.handle_ptr->get_stream());
      offspring.handle_ptr->sync_stream();
      offspring.assignment  = std::move(fixed_assignment);
      offspring.problem_ptr = &fixed_problem;
      cuopt_func_call(offspring.test_variable_bounds(false));
      get_probing_values_for_feasible(guiding_solution,
                                      other_solution,
                                      offspring,
                                      probing_values,
                                      n_vars_from_other,
                                      variable_map);
      probing_config.probing_values         = host_copy(probing_values);
      probing_config.n_of_fixed_from_first  = fixed_from_guiding;
      probing_config.n_of_fixed_from_second = fixed_from_other;
      probing_config.use_balanced_probing   = true;
      constraint_prop.single_rounding_only  = true;
      constraint_prop.apply_round(offspring, lp_run_time_after_feasible, timer, probing_config);
      constraint_prop.single_rounding_only = false;
      cuopt_func_call(bool feasible_after_bounds_prop = offspring.get_feasible());
      offspring.handle_ptr->sync_stream();
      offspring.problem_ptr = a.problem_ptr;
      fixed_assignment      = std::move(offspring.assignment);
      offspring.assignment  = std::move(old_assignment);
      offspring.handle_ptr->sync_stream();
      offspring.unfix_variables(fixed_assignment, variable_map);
      cuopt_func_call(bool feasible_after_unfix = offspring.get_feasible());
      cuopt_assert(feasible_after_unfix == feasible_after_bounds_prop,
                   "Feasible after unfix should be same as feasible after bounds prop!");
      a.handle_ptr->sync_stream();
    } else {
      timer_t timer(bp_recombiner_config_t::bounds_prop_time_limit);
      get_probing_values_for_infeasible(
        guiding_solution, other_solution, offspring, probing_values, n_vars_from_other);
      probing_config.probing_values = host_copy(probing_values);
      constraint_prop.apply_round(offspring, lp_run_time_after_feasible, timer, probing_config);
    }
    constraint_prop.max_n_failed_repair_iterations = 1;
    cuopt_func_call(offspring.test_number_all_integer());
    bool better_cost_than_parents =
      offspring.get_quality(weights) <
      std::min(other_solution.get_quality(weights), guiding_solution.get_quality(weights));
    bool better_feasibility_than_parents = offspring.get_feasible() &&
                                           !other_solution.get_feasible() &&
                                           !guiding_solution.get_feasible();

    bool same_as_parents =
      this->check_if_offspring_is_same_as_parents(offspring, guiding_solution, other_solution);
    // adjust the max_n_of_vars_from_other
    if (n_different_vars > (i_t)bp_recombiner_config_t::max_n_of_vars_from_other) {
      if (same_as_parents) {
        bp_recombiner_config_t::increase_max_n_of_vars_from_other();
      } else {
        bp_recombiner_config_t::decrease_max_n_of_vars_from_other();
      }
    }
    if (better_cost_than_parents || better_feasibility_than_parents) {
      CUOPT_LOG_DEBUG("Offspring is feasible or better than both parents");
      return std::make_pair(offspring, true);
    }
    return std::make_pair(offspring, !same_as_parents);
  }

  rmm::device_uvector<i_t> vars_to_fix;
  constraint_prop_t<i_t, f_t>& constraint_prop;
  thrust::default_random_engine rng;
};

}  // namespace cuopt::linear_programming::detail
