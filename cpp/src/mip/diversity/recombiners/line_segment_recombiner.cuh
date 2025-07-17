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

#include <mip/local_search/line_segment_search/line_segment_search.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/seed_generator.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class line_segment_recombiner_t : public recombiner_t<i_t, f_t> {
 public:
  line_segment_recombiner_t(mip_solver_context_t<i_t, f_t> context,
                            i_t n_vars,
                            line_segment_search_t<i_t, f_t>& line_segment_search_,
                            const raft::handle_t* handle_ptr)
    : recombiner_t<i_t, f_t>(context, n_vars, handle_ptr), line_segment_search(line_segment_search_)
  {
  }

  rmm::device_uvector<f_t> generate_delta_vector(solution_t<i_t, f_t>& guiding_solution,
                                                 solution_t<i_t, f_t>& other_solution,
                                                 solution_t<i_t, f_t>& offspring,
                                                 i_t n_points_to_search,
                                                 i_t remaining_variables)
  {
    CUOPT_LOG_DEBUG("LS rec: Number of different variables %d MAX_VARS %d",
                    remaining_variables,
                    ls_recombiner_config_t::max_n_of_vars_from_other);
    i_t n_vars_from_other = remaining_variables;
    if (n_vars_from_other > (i_t)ls_recombiner_config_t::max_n_of_vars_from_other) {
      n_vars_from_other = ls_recombiner_config_t::max_n_of_vars_from_other;
      thrust::default_random_engine g{(unsigned int)cuopt::seed_generator::get_seed()};
      thrust::shuffle(guiding_solution.handle_ptr->get_thrust_policy(),
                      this->remaining_indices.data(),
                      this->remaining_indices.data() + remaining_variables,
                      g);
    }
    i_t n_vars_from_guiding = guiding_solution.problem_ptr->n_integer_vars - n_vars_from_other;
    rmm::device_uvector<f_t> delta_vector(offspring.problem_ptr->n_variables,
                                          offspring.handle_ptr->get_stream());
    thrust::fill(
      offspring.handle_ptr->get_thrust_policy(), delta_vector.begin(), delta_vector.end(), (f_t)0);
    // generate delta vector only for the ones we want to search
    thrust::for_each(offspring.handle_ptr->get_thrust_policy(),
                     this->remaining_indices.data(),
                     this->remaining_indices.data() + n_vars_from_other,
                     [guiding_solution = guiding_solution.view(),
                      other_solution   = other_solution.view(),
                      delta_vector     = make_span(delta_vector),
                      n_points_to_search] __device__(i_t idx) {
                       f_t guiding_val   = guiding_solution.assignment[idx];
                       f_t other_val     = other_solution.assignment[idx];
                       delta_vector[idx] = (other_val - guiding_val) / (n_points_to_search + 1);
                     });
    return delta_vector;
  }

  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a,
                                                  solution_t<i_t, f_t>& b,
                                                  const weight_t<i_t, f_t>& weights)
  {
    raft::common::nvtx::range fun_scope("line_segment_recombiner");
    auto& guiding_solution = a.get_feasible() ? a : b;
    auto& other_solution   = a.get_feasible() ? b : a;
    // copy the solution from A
    solution_t<i_t, f_t> offspring(guiding_solution);
    timer_t line_segment_timer{ls_recombiner_config_t::time_limit};
    // TODO after we have the conic combination, detect the lambda change
    // (i.e. the integral variables flip on line segment)
    i_t n_points_to_search        = ls_recombiner_config_t::n_points_to_search;
    const bool is_feasibility_run = false;
    i_t n_different_vars =
      this->assign_same_integer_values(guiding_solution, other_solution, offspring);
    rmm::device_uvector<f_t> delta_vector = generate_delta_vector(
      guiding_solution, other_solution, offspring, n_points_to_search, n_different_vars);
    line_segment_search.fj.copy_weights(weights, offspring.handle_ptr);
    // return infeasible results as well
    line_segment_search.settings.recombiner_mode = true;
    line_segment_search.settings.best_of_parents_cost =
      std::min(guiding_solution.get_quality(weights), other_solution.get_quality(weights));
    line_segment_search.settings.parents_infeasible =
      !guiding_solution.get_feasible() && !other_solution.get_feasible();
    line_segment_search.settings.n_points_to_search = ls_recombiner_config_t::n_points_to_search;
    line_segment_search.search_line_segment(offspring,
                                            guiding_solution.assignment,
                                            other_solution.assignment,
                                            delta_vector,
                                            is_feasibility_run,
                                            line_segment_timer);
    line_segment_search.settings = {};
    bool better_cost_than_parents =
      offspring.get_quality(weights) <
      std::min(other_solution.get_quality(weights), guiding_solution.get_quality(weights));
    bool better_feasibility_than_parents = offspring.get_feasible() &&
                                           !other_solution.get_feasible() &&
                                           !guiding_solution.get_feasible();
    bool same_as_parents =
      this->check_if_offspring_is_same_as_parents(offspring, guiding_solution, other_solution);
    // adjust the max_n_of_vars_from_other
    if (n_different_vars > (i_t)ls_recombiner_config_t::max_n_of_vars_from_other) {
      if (same_as_parents) {
        ls_recombiner_config_t::increase_max_n_of_vars_from_other();
      } else {
        ls_recombiner_config_t::decrease_max_n_of_vars_from_other();
      }
    }
    if (better_cost_than_parents || better_feasibility_than_parents) {
      CUOPT_LOG_DEBUG("Offspring is feasible or better than both parents");
      return std::make_pair(offspring, true);
    }
    return std::make_pair(offspring, !same_as_parents);
  }

  line_segment_search_t<i_t, f_t>& line_segment_search;
};

}  // namespace cuopt::linear_programming::detail
