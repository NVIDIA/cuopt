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

#include <mip/mip_constants.hpp>
#include "line_segment_search.cuh"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
line_segment_search_t<i_t, f_t>::line_segment_search_t(fj_t<i_t, f_t>& fj_) : fj(fj_)
{
}

template <typename i_t, typename f_t>
void test_point_is_within_bounds(solution_t<i_t, f_t>& solution,
                                 const rmm::device_uvector<f_t>& point)
{
  rmm::device_uvector<f_t> original_assignment(solution.assignment,
                                               solution.handle_ptr->get_stream());
  raft::copy(solution.assignment.data(),
             point.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  solution.test_variable_bounds(false);
  raft::copy(solution.assignment.data(),
             original_assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool line_segment_search_t<i_t, f_t>::search_line_segment(solution_t<i_t, f_t>& solution,
                                                          const rmm::device_uvector<f_t>& point_1,
                                                          const rmm::device_uvector<f_t>& point_2,
                                                          i_t n_points_to_search,
                                                          bool is_feasibility_run,
                                                          cuopt::timer_t& timer)
{
  CUOPT_LOG_DEBUG("Running line segment search");
  cuopt_assert(point_1.size() == point_2.size(), "size mismatch");
  cuopt_assert(point_1.size() == solution.assignment.size(), "size mismatch");
  cuopt_func_call(test_point_is_within_bounds(solution, point_1));
  cuopt_func_call(test_point_is_within_bounds(solution, point_2));
  rmm::device_uvector<f_t> delta_vector(solution.problem_ptr->n_variables,
                                        solution.handle_ptr->get_stream());
  rmm::device_uvector<f_t> best_assignment(solution.assignment, solution.handle_ptr->get_stream());
  rmm::device_uvector<f_t> previous_rounding(solution.assignment,
                                             solution.handle_ptr->get_stream());

  thrust::transform(solution.handle_ptr->get_thrust_policy(),
                    point_2.data(),
                    point_2.data() + solution.problem_ptr->n_variables,
                    point_1.data(),
                    delta_vector.begin(),
                    [] __device__(const f_t& a, const f_t& b) { return a - b; });

  thrust::transform(
    solution.handle_ptr->get_thrust_policy(),
    delta_vector.begin(),
    delta_vector.end(),
    delta_vector.begin(),
    [n_points_to_search] __device__(const f_t& x) { return x / (n_points_to_search + 1); });

  f_t best_cost            = solution.get_quality(fj.cstr_weights, fj.objective_weight);
  bool initial_is_feasible = solution.get_feasible();
  // TODO start from middle and increase the resolution later
  for (i_t i = 1; i <= n_points_to_search; ++i) {
    CUOPT_LOG_TRACE("Line segment point %d", i);
    thrust::tabulate(solution.handle_ptr->get_thrust_policy(),
                     solution.assignment.begin(),
                     solution.assignment.end(),
                     [i, delta_ptr = delta_vector.data(), point_1_ptr = point_1.data()] __device__(
                       const i_t index) { return point_1_ptr[index] + delta_ptr[index] * i; });
    cuopt_func_call(solution.test_variable_bounds(false));
    bool is_feasible = solution.round_nearest();
    if (is_feasibility_run && is_feasible) {
      CUOPT_LOG_DEBUG("Feasible found after line segment");
      return true;
    }
    i_t number_of_integer_var_diff = compute_number_of_integer_var_diff<i_t, f_t>(
      solution.problem_ptr->integer_indices,
      solution.assignment,
      previous_rounding,
      solution.problem_ptr->tolerances.integrality_tolerance,
      solution.handle_ptr);
    raft::copy(previous_rounding.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    const i_t min_n_integer_diffs = 5;
    if (number_of_integer_var_diff <= min_n_integer_diffs) { continue; }
    cuopt_func_call(solution.test_variable_bounds(false));
    // do the search here
    fj.settings.mode            = fj_mode_t::GREEDY_DESCENT;
    fj.settings.update_weights  = false;
    fj.settings.feasibility_run = is_feasibility_run;
    fj.settings.time_limit      = std::min(0.5, timer.remaining_time());
    is_feasible                 = fj.solve(solution);
    if (is_feasibility_run) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Line segment found feasible");
        return true;
      }
    } else {
      f_t curr_cost = solution.get_quality(fj.cstr_weights, fj.objective_weight);
      if (curr_cost < best_cost && !(initial_is_feasible && !solution.get_feasible())) {
        best_cost = curr_cost;
        raft::copy(best_assignment.data(),
                   solution.assignment.data(),
                   solution.assignment.size(),
                   solution.handle_ptr->get_stream());
      }
    }
    if (timer.check_time_limit()) { break; }
  }
  raft::copy(solution.assignment.data(),
             best_assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  return solution.compute_feasibility();
}

#if MIP_INSTANTIATE_FLOAT
template class line_segment_search_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class line_segment_search_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
