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
#include <queue>
namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
line_segment_search_t<i_t, f_t>::line_segment_search_t(
  fj_t<i_t, f_t>& fj_, constraint_prop_t<i_t, f_t>& constraint_prop_)
  : fj(fj_), constraint_prop(constraint_prop_)
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

class middle_first_iterator_t {
 public:
  middle_first_iterator_t(int n)
  {
    if (n > 0) queue.push({0, n - 1});
  }

  // Returns true if there is a next index
  bool next(int& idx)
  {
    while (!queue.empty()) {
      auto range = queue.front();
      queue.pop();
      int left  = range.first;
      int right = range.second;
      if (left > right) continue;
      int mid = left + (right - left) / 2;
      idx     = mid;
      queue.push({left, mid - 1});
      queue.push({mid + 1, right});
      return true;
    }
    return false;
  }

 private:
  std::queue<std::pair<int, int>> queue;
};

template <typename i_t, typename f_t>
void line_segment_search_t<i_t, f_t>::save_solution_if_better(
  solution_t<i_t, f_t>& solution,
  const rmm::device_uvector<f_t>& point_1,
  const rmm::device_uvector<f_t>& point_2,
  rmm::device_uvector<f_t>& best_assignment,
  rmm::device_uvector<f_t>& best_feasible_assignment,
  f_t& best_cost,
  f_t& best_feasible_cost,
  f_t curr_cost)
{
  if (curr_cost < best_cost) {
    bool save_solution = true;
    // don't check if it is the same as parents if it is better than the parents
    if (settings.recombiner_mode) {
      bool is_feasible_and_better_than_parents =
        solution.get_feasible() && curr_cost < settings.best_of_parents_cost;
      is_feasible_and_better_than_parents =
        is_feasible_and_better_than_parents ||
        (solution.get_feasible() && settings.parents_infeasible);
      if (!is_feasible_and_better_than_parents) {
        // check if the integer part is the same as one of the parents
        save_solution = save_solution && !check_integer_equal_on_indices(
                                           solution.problem_ptr->integer_indices,
                                           solution.assignment,
                                           point_1,
                                           solution.problem_ptr->tolerances.integrality_tolerance,
                                           solution.handle_ptr);
        save_solution = save_solution && !check_integer_equal_on_indices(
                                           solution.problem_ptr->integer_indices,
                                           solution.assignment,
                                           point_2,
                                           solution.problem_ptr->tolerances.integrality_tolerance,
                                           solution.handle_ptr);
      }
    }
    if (save_solution) {
      best_cost = curr_cost;
      raft::copy(best_assignment.data(),
                 solution.assignment.data(),
                 solution.assignment.size(),
                 solution.handle_ptr->get_stream());
      if (solution.get_feasible()) {
        best_feasible_cost = curr_cost;
        raft::copy(best_feasible_assignment.data(),
                   solution.assignment.data(),
                   solution.assignment.size(),
                   solution.handle_ptr->get_stream());
      }
    }
  }
}

template <typename i_t, typename f_t>
bool line_segment_search_t<i_t, f_t>::search_line_segment(
  solution_t<i_t, f_t>& solution,
  const rmm::device_uvector<f_t>& point_1,
  const rmm::device_uvector<f_t>& point_2,
  i_t n_points_to_search,
  const rmm::device_uvector<f_t>& delta_vector,
  bool is_feasibility_run,
  cuopt::timer_t& timer)
{
  CUOPT_LOG_DEBUG("Running line segment search with a given delta vector");
  cuopt_assert(point_1.size() == point_2.size(), "size mismatch");
  cuopt_assert(point_1.size() == solution.assignment.size(), "size mismatch");
  cuopt_assert(delta_vector.size() == solution.assignment.size(), "size mismatch");
  rmm::device_uvector<f_t> best_assignment(solution.assignment, solution.handle_ptr->get_stream());
  rmm::device_uvector<f_t> best_feasible_assignment(solution.assignment,
                                                    solution.handle_ptr->get_stream());
  rmm::device_uvector<f_t> previous_rounding(solution.assignment,
                                             solution.handle_ptr->get_stream());
  // we want to allow solutions that might be worse but different than parents in recombiner mode
  f_t best_cost = std::numeric_limits<f_t>::max();
  if (!settings.recombiner_mode) {
    best_cost = solution.get_quality(fj.cstr_weights, fj.objective_weight);
  }
  f_t best_feasible_cost   = std::numeric_limits<f_t>::max();
  bool initial_is_feasible = solution.get_feasible();
  middle_first_iterator_t it(n_points_to_search);
  int i;
  while (it.next(i)) {
    // make it one indexed
    i++;
    CUOPT_LOG_DEBUG("Line segment point %d", i);
    thrust::tabulate(solution.handle_ptr->get_thrust_policy(),
                     solution.assignment.begin(),
                     solution.assignment.end(),
                     [i, delta_ptr = delta_vector.data(), point_1_ptr = point_1.data()] __device__(
                       const i_t index) { return point_1_ptr[index] + delta_ptr[index] * i; });
    cuopt_func_call(solution.test_variable_bounds(false));
    bool is_feasible = false;
    // if (!settings.recombiner_mode) {
    if (true) {
      is_feasible = solution.round_nearest();
    } else {
      fj.settings.mode            = fj_mode_t::ROUNDING;
      fj.settings.update_weights  = false;
      fj.settings.feasibility_run = is_feasibility_run;
      fj.settings.time_limit      = std::min(0.1, timer.remaining_time());
      is_feasible                 = fj.solve(solution);
    }
    cuopt_func_call(solution.test_number_all_integer());
    if (is_feasibility_run) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Feasible found during line segment round");
        return true;
      }
    } else {
      f_t curr_cost = solution.get_quality(fj.cstr_weights, fj.objective_weight);
      save_solution_if_better(solution,
                              point_1,
                              point_2,
                              best_assignment,
                              best_feasible_assignment,
                              best_cost,
                              best_feasible_cost,
                              curr_cost);
    }
    if (timer.check_time_limit()) { break; }
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
    fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    fj.settings.termination            = fj_termination_flags_t::FJ_TERMINATION_ITERATION_LIMIT;
    fj.settings.n_of_minimums_for_exit = 50;
    fj.settings.iteration_limit =
      std::max(20 * fj.settings.n_of_minimums_for_exit, solution.problem_ptr->n_constraints / 50);
    fj.settings.update_weights  = false;
    fj.settings.feasibility_run = is_feasibility_run;
    fj.settings.time_limit      = std::min(1., timer.remaining_time());
    is_feasible                 = fj.solve(solution);
    if (is_feasibility_run) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Line segment found feasible");
        return true;
      }
    } else {
      f_t curr_cost = solution.get_quality(fj.cstr_weights, fj.objective_weight);
      save_solution_if_better(solution,
                              point_1,
                              point_2,
                              best_assignment,
                              best_feasible_assignment,
                              best_cost,
                              best_feasible_cost,
                              curr_cost);
    }
    if (timer.check_time_limit()) { break; }
  }
  // if not recombiner mode but local search mode
  if (!settings.recombiner_mode) {
    if (initial_is_feasible && best_feasible_cost != std::numeric_limits<f_t>::max()) {
      raft::copy(solution.assignment.data(),
                 best_feasible_assignment.data(),
                 solution.assignment.size(),
                 solution.handle_ptr->get_stream());
    } else {
      raft::copy(solution.assignment.data(),
                 best_assignment.data(),
                 solution.assignment.size(),
                 solution.handle_ptr->get_stream());
    }
  } else {
    if (best_feasible_cost != std::numeric_limits<f_t>::max()) {
      CUOPT_LOG_DEBUG("Returning best feasible solution");
      // return best feasible solution that is different than parents
      raft::copy(solution.assignment.data(),
                 best_feasible_assignment.data(),
                 solution.assignment.size(),
                 solution.handle_ptr->get_stream());
    } else {
      CUOPT_LOG_DEBUG("Returning best solution");
      raft::copy(solution.assignment.data(),
                 best_assignment.data(),
                 solution.assignment.size(),
                 solution.handle_ptr->get_stream());
    }
  }
  return solution.compute_feasibility();
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

  thrust::transform(solution.handle_ptr->get_thrust_policy(),
                    point_2.data(),
                    point_2.data() + solution.problem_ptr->n_variables,
                    point_1.data(),
                    delta_vector.begin(),
                    [] __device__(const f_t a, const f_t b) { return a - b; });

  thrust::transform(
    solution.handle_ptr->get_thrust_policy(),
    delta_vector.begin(),
    delta_vector.end(),
    delta_vector.begin(),
    [n_points_to_search] __device__(const f_t x) { return x / (n_points_to_search + 1); });
  return search_line_segment(
    solution, point_1, point_2, n_points_to_search, delta_vector, is_feasibility_run, timer);
}

#if MIP_INSTANTIATE_FLOAT
template class line_segment_search_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class line_segment_search_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
