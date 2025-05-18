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

#include "population.cuh"

#include <thrust/for_each.h>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>

#include <mutex>

namespace cuopt::linear_programming::detail {

constexpr double weight_increase_ratio    = 2.;
constexpr double weight_decrease_ratio    = 0.9;
constexpr double max_infeasibility_weight = 10000000.;
constexpr double min_infeasibility_weight = 1.;

template <typename i_t, typename f_t>
population_t<i_t, f_t>::population_t(std::string const& name_,
                                     mip_solver_context_t<i_t, f_t>& context_,
                                     int var_threshold_,
                                     size_t max_solutions_,
                                     f_t infeasibility_weight_)
  : name(name_),
    context(context_),
    problem_ptr(context.problem_ptr),
    var_threshold(var_threshold_),
    max_solutions(max_solutions_),
    infeasibility_importance(infeasibility_weight_),
    weights(0, context.problem_ptr->handle_ptr),
    rng(cuopt::seed_generator::get_seed()),
    early_exit_primal_generation(false)
{
  best_feasible_objective =
    problem_ptr->maximize ? -std::numeric_limits<f_t>::max() : std::numeric_limits<f_t>::max();
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::allocate_solutions()
{
  for (size_t i = 0; i < max_solutions; ++i) {
    bool occupied = false;
    solutions.emplace_back(occupied, solution_t<i_t, f_t>(*problem_ptr));
  }
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::initialize_population()
{
  var_threshold =
    max(problem_ptr->n_variables - var_threshold, (problem_ptr->n_variables / 10) * 8);
  initial_threshold_ratio = (f_t)var_threshold / problem_ptr->n_variables;
  solutions.reserve(max_solutions);
  indices.reserve(max_solutions);
  // indices[0] always points to solutions[0] - a special place for feasible solution
  indices.emplace_back(0, std::numeric_limits<f_t>::max());
  constexpr f_t ten = 10.;
  weights.cstr_weights.resize(problem_ptr->n_constraints, problem_ptr->handle_ptr->get_stream());
  thrust::uninitialized_fill(problem_ptr->handle_ptr->get_thrust_policy(),
                             weights.cstr_weights.begin(),
                             weights.cstr_weights.end(),
                             ten);
}

template <typename i_t, typename f_t>
std::pair<solution_t<i_t, f_t>, solution_t<i_t, f_t>> population_t<i_t, f_t>::get_two_random(
  bool tournament)
{
  raft::common::nvtx::range fun_scope("get_two_random");
  cuopt_assert(indices.size() > 2, "There should be enough solutions");
  size_t add = (size_t)(!solutions[0].first || solutions[indices[1].first].second.get_feasible());
  size_t i   = add + std::uniform_int_distribution<size_t>(0, (indices.size() - 2))(rng);
  size_t j   = add + std::uniform_int_distribution<size_t>(0, (indices.size() - 3))(rng);
  if (tournament) {
    size_t i1 = add + std::uniform_int_distribution<size_t>(0, (indices.size() - 2))(rng);
    size_t j1 = add + std::uniform_int_distribution<size_t>(0, (indices.size() - 3))(rng);
    i         = std::min<size_t>(i, i1);
    j         = std::min<size_t>(j, j1);
  }
  if (j >= i) j++;
  auto first_solution  = solutions[indices[i].first].second;
  auto second_solution = solutions[indices[j].first].second;
  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
  return std::make_pair(std::move(first_solution), std::move(second_solution));
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::add_solutions_from_vec(std::vector<solution_t<i_t, f_t>>&& solutions)
{
  raft::common::nvtx::range fun_scope("add_solution_from_vec");
  for (auto&& sol : solutions) {
    add_solution(std::move(sol));
  }
}

template <typename i_t, typename f_t>
size_t population_t<i_t, f_t>::get_external_solution_size()
{
  std::lock_guard<std::mutex> lock(solution_mutex);
  return external_solution_queue.size();
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::add_external_solution(std::vector<f_t>& solution, f_t objective)
{
  std::lock_guard<std::mutex> lock(solution_mutex);
  CUOPT_LOG_INFO("B&B added a solution to population, solution queue size %lu with objective %g",
                 external_solution_queue.size(),
                 problem_ptr->get_user_obj_from_solver_obj(objective));
  external_solution_queue.emplace_back(solution);
  if (external_solution_queue.size() >= 5) { early_exit_primal_generation = true; }
}

// normally we would need a lock here but these are boolean types and race conditions are not
// possible
template <typename i_t, typename f_t>
void population_t<i_t, f_t>::preempt_heuristic_solver()
{
  preempt_heuristic_solver_    = true;
  early_exit_primal_generation = true;
}

template <typename i_t, typename f_t>
std::vector<solution_t<i_t, f_t>> population_t<i_t, f_t>::get_external_solutions()
{
  std::lock_guard<std::mutex> lock(solution_mutex);
  std::vector<solution_t<i_t, f_t>> return_vector;
  for (auto h_solution_vec : external_solution_queue) {
    solution_t<i_t, f_t> sol(*problem_ptr);
    sol.copy_new_assignment(h_solution_vec);
    sol.compute_feasibility();
    return_vector.emplace_back(std::move(sol));
  }
  if (external_solution_queue.size() > 0) {
    CUOPT_LOG_INFO("Consuming B&B solutions, solution queue size %lu",
                   external_solution_queue.size());
    external_solution_queue.clear();
  }
  return return_vector;
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::run_solution_callbacks(solution_t<i_t, f_t>& sol)
{
  bool better_solution_found = problem_ptr->maximize
                                 ? sol.get_user_objective() > best_feasible_objective
                                 : sol.get_user_objective() < best_feasible_objective;
  auto user_callbacks        = context.settings.get_mip_callbacks();

  if (better_solution_found && sol.get_feasible()) {
    CUOPT_LOG_DEBUG("Population: Found new best solution %g", sol.get_user_objective());
    best_feasible_objective = sol.get_user_objective();
    if (problem_ptr->branch_and_bound_callback != nullptr) {
      problem_ptr->branch_and_bound_callback(sol.get_host_assignment());
    }

    for (auto callback : user_callbacks) {
      if (callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
        auto get_sol_callback = static_cast<internals::get_solution_callback_t*>(callback);
        rmm::device_uvector<f_t> incumbent_assignment(sol.assignment, sol.handle_ptr->get_stream());
        problem_ptr->post_process_assignment(incumbent_assignment);
        rmm::device_uvector<f_t> dummy(0, sol.handle_ptr->get_stream());
        if (context.settings.mip_scaling) {
          context.scaling.unscale_solutions(incumbent_assignment, dummy);
        }

        rmm::device_uvector<f_t> user_objective_vec(1, sol.handle_ptr->get_stream());

        f_t user_objective = sol.problem_ptr->get_user_obj_from_solver_obj(sol.get_objective());
        user_objective_vec.set_element_async(0, user_objective, sol.handle_ptr->get_stream());
        CUOPT_LOG_DEBUG("Returning incumbent solution with objective %g", user_objective);
        get_sol_callback->get_solution(incumbent_assignment.data(), user_objective_vec.data());
      }
    }
  }

  for (auto callback : user_callbacks) {
    if (callback->get_type() == internals::base_solution_callback_type::SET_SOLUTION) {
      auto set_sol_callback = static_cast<internals::set_solution_callback_t*>(callback);
      rmm::device_uvector<f_t> incumbent_assignment(
        problem_ptr->original_problem_ptr->get_n_variables(), sol.handle_ptr->get_stream());
      rmm::device_uvector<f_t> dummy(0, sol.handle_ptr->get_stream());
      solution_t<i_t, f_t> outside_sol(sol);
      rmm::device_scalar<f_t> d_outside_sol_objective(sol.handle_ptr->get_stream());
      set_sol_callback->set_solution(incumbent_assignment.data(), d_outside_sol_objective.data());

      f_t outside_sol_objective = d_outside_sol_objective.value(sol.handle_ptr->get_stream());

      CUOPT_LOG_DEBUG("Injecting external solution with objective %g", outside_sol_objective);

      if (context.settings.mip_scaling) {
        context.scaling.scale_solutions(incumbent_assignment, dummy);
      }
      bool is_valid = problem_ptr->pre_process_assignment(incumbent_assignment);
      if (!is_valid) { return; }

      cuopt_assert(outside_sol.assignment.size() == incumbent_assignment.size(),
                   "Incumbent assignment size mismatch");
      raft::copy(outside_sol.assignment.data(),
                 incumbent_assignment.data(),
                 incumbent_assignment.size(),
                 sol.handle_ptr->get_stream());

      outside_sol.compute_feasibility();

      CUOPT_LOG_DEBUG("Injected solution feasibility =  %d", outside_sol.get_feasible());

      cuopt_assert(std::abs(outside_sol.problem_ptr->get_user_obj_from_solver_obj(
                              outside_sol.get_objective()) -
                            outside_sol_objective) <= 1e-6,
                   "External solution objective mismatch");
      auto h_outside_sol = outside_sol.get_host_assignment();
      add_external_solution(h_outside_sol, outside_sol.get_objective());
    }
  }
}

template <typename i_t, typename f_t>
i_t population_t<i_t, f_t>::add_solution(solution_t<i_t, f_t>&& sol)
{
  raft::common::nvtx::range fun_scope("add_solution");
  double sol_cost = sol.get_quality(weights);
  CUOPT_LOG_TRACE("Adding solution with quality %f and objective %f n_integers %d!",
                  sol_cost,
                  sol.get_user_objective(),
                  sol.n_assigned_integers);
  // We store the best feasible found so far at index 0.
  if (sol.get_feasible() &&
      (solutions[0].first == false || sol_cost + OBJECTIVE_EPSILON < indices[0].second)) {
    run_solution_callbacks(sol);
    solutions[0].first = true;
    // we only have move assignment operator
    solution_t<i_t, f_t> temp_sol(sol);
    solutions[0].second = std::move(temp_sol);
    indices[0].second   = sol_cost;
  }

  // Fast reject
  if (indices.size() == max_solutions && indices.back().second <= sol_cost + OBJECTIVE_EPSILON) {
    CUOPT_LOG_TRACE("Rejecting solution objective is not better!");
    return -1;
  }

  // Find index best solution similar to sol (within the threshold radius) in the indices array
  size_t index = best_similar_index(sol);

  // check if any solution below this index is feasible and the current solution is infeasible

  // No similar was found and added solution is better then worse in population (if the population
  // is full)
  if (index == max_solutions) {
    CUOPT_LOG_TRACE("No similar was found in population!");
    // Place in the solutions vector:
    int hint = -1;
    // If the population is full eject the worse solution
    if (indices.size() == max_solutions) {
      CUOPT_LOG_TRACE("Ejecting worst solution");
      hint = (int)indices.back().first;
      indices.pop_back();
      solutions[hint].first = false;
    }

    if (hint == -1) hint = find_free_solution_index();

    solutions[hint].first  = true;
    solutions[hint].second = std::move(sol);

    int inserted_pos = insert_index(std::pair<size_t, double>((size_t)hint, sol_cost));
    cuopt_assert(test_invariant(), "Population invariant doesn't hold");
    return inserted_pos;

  } else if (sol_cost + OBJECTIVE_EPSILON < indices[index].second) {
    CUOPT_LOG_TRACE("Better than similar solution, eradicating similar solutions!");
    eradicate_similar(index, sol);

    size_t free = find_free_solution_index();

    solutions[free].first  = true;
    solutions[free].second = std::move(sol);

    int inserted_pos = insert_index(std::pair<size_t, double>((size_t)free, sol_cost));
    cuopt_assert(test_invariant(), "Population invariant doesn't hold");
    return inserted_pos;
  }
  CUOPT_LOG_TRACE("Adding solution failed!");
  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
  return -1;
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::normalize_weights()
{
  CUOPT_LOG_DEBUG("Normalizing weights");

  rmm::device_scalar<f_t> l2_norm(problem_ptr->handle_ptr->get_stream());
  my_l2_norm<i_t, f_t>(weights.cstr_weights, l2_norm, problem_ptr->handle_ptr);
  thrust::transform(
    problem_ptr->handle_ptr->get_thrust_policy(),
    weights.cstr_weights.begin(),
    weights.cstr_weights.end(),
    weights.cstr_weights.begin(),
    [l2_norm_ptr = l2_norm.data(), inf_weight = infeasibility_importance] __device__(f_t weight) {
      f_t new_weight = max((weight * inf_weight) / *l2_norm_ptr, 10.);
      new_weight     = (weight * inf_weight) / *l2_norm_ptr;
      cuopt_assert(isfinite(new_weight), "");
      return new_weight;
    });

  thrust::for_each(
    problem_ptr->handle_ptr->get_thrust_policy(),
    thrust::counting_iterator<i_t>(0),
    thrust::counting_iterator<i_t>(problem_ptr->n_constraints),
    [pb = problem_ptr->view(), weights = weights.cstr_weights.data()] __device__(i_t cstr_idx) {
      auto [offset_begin, offset_end] = pb.range_for_constraint(cstr_idx);

      f_t min_weight = 0.0;
      for (i_t j = offset_begin; j < offset_end; ++j) {
        i_t var = pb.variables[j];

        f_t cstr_coeff = pb.coefficients[j];
        f_t obj_coeff  = pb.objective_coefficients[var];

        min_weight = max(min_weight, obj_coeff / cstr_coeff);
      }

      weights[cstr_idx] = max(weights[cstr_idx], min_weight);
    });

  problem_ptr->handle_ptr->sync_stream();
}

// adjust the cstr weights according to the best solution's excess
template <typename i_t, typename f_t>
void population_t<i_t, f_t>::compute_new_weights()
{
  auto& best_sol = best();
  auto settings  = context.settings;

  rmm::device_scalar<f_t> l2_norm(problem_ptr->handle_ptr->get_stream());
  my_l2_norm<i_t, f_t>(weights.cstr_weights, l2_norm, problem_ptr->handle_ptr);

  if (!best_sol.get_feasible()) {
    CUOPT_LOG_DEBUG("Increasing weights!");
    // in the first two rounds, do more agressive updates
    if (update_iter < 2) {
      infeasibility_importance *= 5;
    } else {
      infeasibility_importance *= weight_increase_ratio;
    }

    infeasibility_importance = min(max_infeasibility_weight, infeasibility_importance);
    thrust::for_each(best_sol.handle_ptr->get_thrust_policy(),
                     thrust::counting_iterator(0),
                     thrust::counting_iterator(0) + weights.cstr_weights.size(),
                     [v            = best_sol.view(),
                      cstr_weights = weights.cstr_weights.data(),
                      l2_norm_ptr  = l2_norm.data(),
                      rel_tol      = settings.tolerances.relative_tolerance] __device__(i_t idx) {
                       if ((v.lower_excess[idx] + v.upper_excess[idx]) > rel_tol) {
                         cstr_weights[idx] *= weight_increase_ratio;
                         cstr_weights[idx] = min(cstr_weights[idx], 100000.);
                       }
                     });
  } else {
    CUOPT_LOG_DEBUG("Decreasing weights!");
    infeasibility_importance *= weight_decrease_ratio;
    infeasibility_importance = max(min_infeasibility_weight, infeasibility_importance);

    thrust::for_each(
      best_sol.handle_ptr->get_thrust_policy(),
      thrust::counting_iterator(0),
      thrust::counting_iterator(0) + weights.cstr_weights.size(),
      [v = best_sol.view(), cstr_weights = weights.cstr_weights.data()] __device__(i_t idx) {
        cstr_weights[idx] *= weight_decrease_ratio;
        cstr_weights[idx] = max(cstr_weights[idx], 10.);
      });
  }
  best_sol.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::adjust_threshold(cuopt::timer_t timer)
{
  const double max_diversity_threshold = 0.99;
  double time_ratio =
    (timer.elapsed_time() - diversity_start_time) / (timer.get_time_limit() - diversity_start_time);
  f_t threshold_ratio =
    initial_threshold_ratio + time_ratio * (max_diversity_threshold - initial_threshold_ratio);
  var_threshold = threshold_ratio * problem_ptr->n_variables;
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::update_qualities()
{
  if (indices.size() == 1) return;
  using pr = std::pair<size_t, double>;
  for (size_t i = !is_feasible(); i < indices.size(); i++)
    indices[i].second = solutions[indices[i].first].second.get_quality(weights);

  std::sort(indices.begin() + 1, indices.end(), [](const pr& a, const pr& b) {
    return a.second < b.second;
  });

  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::update_weights()
{
  raft::common::nvtx::range fun_scope("adjust_weight_changes");
  CUOPT_LOG_DEBUG("Changing the weights");
  // TODO activate this if we have a reserve and a diverse initial population
  // by adding new solutions at every diversity step, it doesn't make sense to add
  // adjust_threshold(timer);
  compute_new_weights();
  normalize_weights();
  update_qualities();
  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
  update_iter++;
}

// returns true if solutions are similar, false if different
template <typename i_t, typename f_t>
bool population_t<i_t, f_t>::check_sols_similar(solution_t<i_t, f_t>& sol1,
                                                solution_t<i_t, f_t>& sol2) const
{
  return sol1.calculate_similarity_radius(sol2) > var_threshold;
}

template <typename i_t, typename f_t>
size_t population_t<i_t, f_t>::best_similar_index(solution_t<i_t, f_t>& sol)
{
  raft::common::nvtx::range fun_scope("best_similar_index");
  if (indices.size() == 1) return max_solutions;
  for (size_t i = 1; i < indices.size(); i++) {
    if (check_sols_similar(sol, solutions[indices[i].first].second)) { return i; }
  }

  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
  return max_solutions;
}

template <typename i_t, typename f_t>
i_t population_t<i_t, f_t>::insert_index(std::pair<i_t, f_t> to_insert)
{
  raft::common::nvtx::range fun_scope("insert_index");
  // Assert free index is available
  indices.emplace_back(0, 0.0);
  size_t start = indices.size() - 1;
  while (start > 1 && indices[start - 1].second > to_insert.second) {
    indices[start] = indices[start - 1];
    start--;
  }
  indices[start] = to_insert;
  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
  return start;
}

template <typename i_t, typename f_t>
bool population_t<i_t, f_t>::check_if_feasible_similar_exists(size_t start_index,
                                                              solution_t<i_t, f_t>& sol)
{
  raft::common::nvtx::range fun_scope("check_if_feasible_similar_exists");
  for (size_t i = start_index; i < indices.size(); i++) {
    if (check_sols_similar(sol, solutions[indices[i].first].second)) {
      if (solutions[indices[i].first].second.get_feasible()) { return true; }
    }
  }
  return false;
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::eradicate_similar(size_t start_index, solution_t<i_t, f_t>& sol)
{
  raft::common::nvtx::range fun_scope("eradicate_similar");
  for (size_t i = start_index; i < indices.size(); i++) {
    if (check_sols_similar(sol, solutions[indices[i].first].second)) {
      solutions[indices[i].first].first = false;              // mark place as available
      indices[i].first = std::numeric_limits<size_t>::max();  // mark as deleted in indices
    }
  }

  // Copy all element == std::numeric_limits<size_t>::max() to the right part of the indices array
  size_t count = start_index;
  for (size_t i = start_index; i < indices.size(); i++)
    if (indices[i].first != std::numeric_limits<size_t>::max())
      indices[count++] = indices[i];  // here count is incremented

  indices.erase(indices.begin() + count, indices.end());
  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
}

template <typename i_t, typename f_t>
std::vector<solution_t<i_t, f_t>> population_t<i_t, f_t>::population_to_vector()
{
  std::vector<solution_t<i_t, f_t>> sol_vec;
  bool population_feasible = is_feasible();
  for (size_t i = !population_feasible; i < indices.size(); i++) {
    sol_vec.emplace_back(solution_t<i_t, f_t>{solutions[indices[i].first].second});
  }
  return sol_vec;
}

template <typename i_t>
i_t get_max_var_threshold(i_t n_vars)
{
  if (n_vars < 50) {
    return max(1, n_vars - 1);
  } else if (n_vars < 80) {
    return n_vars - 2;
  } else if (n_vars < 200) {
    return n_vars - 4;
  } else if (n_vars < 1000) {
    return n_vars - 8;
  }
  return n_vars - 10;
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::halve_the_population()
{
  raft::common::nvtx::range fun_scope("halve_the_population");
  if (current_size() <= max_solutions / 2) { return; }
  CUOPT_LOG_DEBUG("Halving the population, current size: %lu", current_size());
  // put population into a vector
  auto sol_vec                  = population_to_vector();
  i_t counter                   = 0;
  constexpr i_t max_adjustments = 4;
  size_t max_var_threshold      = get_max_var_threshold(problem_ptr->n_integer_vars);
  while (current_size() > max_solutions / 2) {
    clear_except_best_feasible();
    var_threshold = max(var_threshold * 0.97, 0.5 * problem_ptr->n_integer_vars);
    for (auto& sol : sol_vec) {
      add_solution(solution_t<i_t, f_t>(sol));
    }
    if (counter++ > max_adjustments) break;
  }
  counter = 0;
  // if we removed too many decrease the diversity a little
  while (current_size() < max_solutions / 4) {
    clear_except_best_feasible();
    var_threshold =
      min(max_var_threshold,
          min((size_t)(var_threshold * 0.97), (size_t)(0.995 * problem_ptr->n_integer_vars)));
    for (auto& sol : sol_vec) {
      add_solution(solution_t<i_t, f_t>(sol));
    }
    if (counter++ > max_adjustments) break;
  }
}

template <typename i_t, typename f_t>
size_t population_t<i_t, f_t>::find_free_solution_index()
{
  raft::common::nvtx::range fun_scope("find_free_solution_index");
  // ASSERT such index exists
  for (size_t i = 1; i < solutions.size(); i++)
    if (solutions[i].first == false) return i;

  cuopt_assert(test_invariant(), "Population invariant doesn't hold");
  return std::numeric_limits<size_t>::max();
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::find_diversity(std::vector<solution_t<i_t, f_t>>& initial_sol_vector,
                                            bool avg)
{
  raft::common::nvtx::range fun_scope("find_diversity");
  i_t n_feasible = 0;
  size_t average = 0;
  size_t max     = 0;
  size_t sum     = 0;
  for (size_t i = 0; i < initial_sol_vector.size(); i++) {
    for (size_t j = i + 1; j < initial_sol_vector.size(); j++) {
      sum++;
      size_t similarity = initial_sol_vector[i].calculate_similarity_radius(initial_sol_vector[j]);
      average += similarity;
      max = std::max(max, similarity);
    }
  }

  if (sum > 0) {
    if (avg)
      average /= (double)sum;
    else
      average = max;
  }
  size_t max_var_threshold = get_max_var_threshold(problem_ptr->n_integer_vars);
  var_threshold            = std::min(average, max_var_threshold);
}

template <typename i_t, typename f_t>
bool population_t<i_t, f_t>::test_invariant()
{
  // Indices size >= 1
  for (size_t i = 1; i < indices.size(); i++) {
    // Every index should point valid solution. Number should match each other
    if (solutions[indices[i].first].first == false) {
      CUOPT_LOG_ERROR("Solution %d empty\n", (int)i);
      return false;
    }
    // Quality in index should match the quality of solution
    if (std::fabs(solutions[indices[i].first].second.get_quality(weights) - indices[i].second) >
        OBJECTIVE_EPSILON) {
      CUOPT_LOG_ERROR("Solution %d quality does not match: %f %f \n",
                      (int)i,
                      solutions[indices[i].first].second.get_quality(weights),
                      indices[i].second);
      return false;
    }
    // Indices should be sorted
    if (i + 1 < indices.size() && indices[i].second > indices[i + 1].second) {
      CUOPT_LOG_ERROR("Indices not sorted: %d \n", (int)i);
      return false;
    }
    // Each two solutions radius should be lower then threshold
    for (size_t j = i + 1; j < indices.size(); j++) {
      if (check_sols_similar(solutions[indices[i].first].second,
                             solutions[indices[j].first].second)) {
        CUOPT_LOG_ERROR("Solutions radius greater then threshold: %d %d\n",
                        (int)indices[i].first,
                        (int)indices[j].first);
        return false;
      }
    }
  }

  // solutions[0] should be feasible
  if (solutions[0].first && !solutions[0].second.get_feasible()) {
    CUOPT_LOG_ERROR(" Non feasible marked as feasible: \n");
    return false;
  }
  if (indices.size() > max_solutions) {
    CUOPT_LOG_ERROR(" Size excess \n");
    return false;
  }

  return true;
}

template <typename i_t, typename f_t>
void population_t<i_t, f_t>::print()
{
  CUOPT_LOG_DEBUG(" -------------- ");
  CUOPT_LOG_DEBUG("%s infeas weight %f threshold %d/%d:",
                  name.c_str(),
                  infeasibility_importance,
                  var_threshold,
                  problem_ptr->n_integer_vars);
  i_t i = 0;
  for (auto& index : indices) {
    if (index.first == 0 && solutions[0].first) {
      CUOPT_LOG_DEBUG(" Best feasible: %f", solutions[index.first].second.get_user_objective());
    }
    CUOPT_LOG_DEBUG("%d :  %f\t%f\t%f",
                    i,
                    index.second,
                    solutions[index.first].second.get_total_excess(),
                    solutions[index.first].second.get_user_objective());
    i++;
  }
  CUOPT_LOG_DEBUG(" -------------- ");
}

#if MIP_INSTANTIATE_FLOAT
template class population_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class population_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
