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

#include "feasibility_pump.cuh"

#include <cuopt/error.hpp>
#include <mip/mip_constants.hpp>
#include <mip/problem/host_helper.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/utils.cuh>

#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <linear_programming/pdlp.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/linalg/binary_op.cuh>
#include <utilities/seed_generator.cuh>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
feasibility_pump_t<i_t, f_t>::feasibility_pump_t(
  mip_solver_context_t<i_t, f_t>& context_,
  fj_t<i_t, f_t>& fj_,
  //  fj_tree_t<i_t, f_t>& fj_tree_,
  constraint_prop_t<i_t, f_t>& constraint_prop_,
  lb_constraint_prop_t<i_t, f_t>& lb_constraint_prop_,
  line_segment_search_t<i_t, f_t>& line_segment_search_,
  rmm::device_uvector<f_t>& lp_optimal_solution_)
  : context(context_),
    fj(fj_),
    // fj_tree(fj_tree_),
    line_segment_search(line_segment_search_),
    cycle_queue(*context.problem_ptr),
    constraint_prop(constraint_prop_),
    lb_constraint_prop(lb_constraint_prop_),
    last_rounding(context.problem_ptr->n_variables, context.problem_ptr->handle_ptr->get_stream()),
    last_projection(context.problem_ptr->n_variables,
                    context.problem_ptr->handle_ptr->get_stream()),
    orig_variable_types(context.problem_ptr->n_variables,
                        context.problem_ptr->handle_ptr->get_stream()),
    best_excess_solution(context.problem_ptr->n_variables,
                         context.problem_ptr->handle_ptr->get_stream()),
    lp_optimal_solution(lp_optimal_solution_),
    rng(cuopt::seed_generator::get_seed()),
    timer(20.)
{
}

template <typename Iter_T>
long double vector_norm(Iter_T first, Iter_T last)
{
  return sqrt(inner_product(first, last, first, 0.0));
}

// this function creates a weighted objective between the distance to the polytope and the original
// objective in the beginning the solution will favor the original objective but later it favors the
// feasibility(distance)
template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::adjust_objective_with_original(solution_t<i_t, f_t>& solution,
                                                                  std::vector<f_t>& dist_objective,
                                                                  bool longer_fp_run)
{
  // TODO set alpha to zero after some point
  if (!longer_fp_run) {
    CUOPT_LOG_TRACE(
      "changing alpha from %f to %f", config.alpha, config.alpha * config.alpha_decrease_factor);
    config.alpha = config.alpha * config.alpha_decrease_factor;
  }
  f_t distance_weight         = 1. - config.alpha;
  std::vector<f_t> obj_vector = cuopt::host_copy(solution.problem_ptr->objective_coefficients,
                                                 solution.handle_ptr->get_stream());
  solution.handle_ptr->sync_stream();
  const f_t l2_norm_of_original_obj = vector_norm(obj_vector.begin(), obj_vector.end());
  const f_t l2_norm_of_distance_obj = vector_norm(dist_objective.begin(), dist_objective.end());
  CUOPT_LOG_TRACE("l2_norm_of_original_obj %f l2_norm_of_distance_obj %f",
                  l2_norm_of_original_obj,
                  l2_norm_of_distance_obj);
  // f_t orig_obj_weight = config.alpha * (l2_norm_of_distance_obj / l2_norm_of_original_obj);
  f_t orig_obj_weight = config.alpha / l2_norm_of_original_obj;
  distance_weight     = distance_weight / l2_norm_of_distance_obj;
  if (!isfinite(orig_obj_weight)) {
    CUOPT_LOG_TRACE("orig_obj_weight is not finite, setting to zero");
    orig_obj_weight = 0.;
  }
  cuopt_expects(isfinite(orig_obj_weight), error_type_t::RuntimeError, "Weight should be finite!");
  CUOPT_LOG_TRACE("dist weight %f obj weight %f", distance_weight, orig_obj_weight);
  for (i_t i = 0; i < (i_t)dist_objective.size(); ++i) {
    f_t orig_obj      = i < (i_t)obj_vector.size() ? obj_vector[i] : 0.;
    dist_objective[i] = dist_objective[i] * distance_weight + orig_obj_weight * orig_obj;
    cuopt_expects(
      isfinite(dist_objective[i]), error_type_t::RuntimeError, "Weight should be finite!");
  }
}

// TODO adjust this tolerance for runs of lower prec(10-8)
double get_tolerance_from_ratio(double ratio_integer, double absolute_tol)
{
  if (ratio_integer < 0.80) {
    return 0.1;
  } else if (ratio_integer < 0.93) {
    return 0.01;
  } else if (ratio_integer < 0.97) {
    return 0.001;
  } else {
    return absolute_tol;
  }
}

// projects the current integer solution to the polytope.
// the epsilon can be larger here maybe 10-1,10-2.
// finding the projection requires running LP that minimizes the distance of the current solution to
// the polytope. the distance to polytope is integrated into the linear programming constraint.
// following is done if current integer value is within the bounds:
// the distance is added as an additional variable for each original variable.
// minimize the distance where distance is at least |x_j-val(x_j)|.
// two constraints are added to handle abs value.
// if we are at the end of the interval.(i.e x_j is u_j or l_j)
// we can get rid of the additional variables and constraints. because the distance can only be to a
// single direction. we won't need a variable and two constraints for the abs value.

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::linear_project_onto_polytope(solution_t<i_t, f_t>& solution,
                                                                f_t ratio_of_set_integers,
                                                                bool longer_lp_run)
{
  CUOPT_LOG_DEBUG("linear projection of fp");
  auto h_assignment            = solution.get_host_assignment();
  auto h_variable_upper_bounds = cuopt::host_copy(solution.problem_ptr->variable_upper_bounds,
                                                  solution.handle_ptr->get_stream());
  auto h_variable_lower_bounds = cuopt::host_copy(solution.problem_ptr->variable_lower_bounds,
                                                  solution.handle_ptr->get_stream());

  const f_t int_tol = context.settings.tolerances.integrality_tolerance;
  constraints_delta_t<i_t, f_t> h_constraints;
  variables_delta_t<i_t, f_t> h_variables;
  h_variables.n_vars = solution.problem_ptr->n_variables;
  std::vector<f_t> obj_coefficients(solution.problem_ptr->n_variables, 0.);
  problem_t<i_t, f_t> temp_p(*solution.problem_ptr);
  auto h_integer_indices =
    cuopt::host_copy(solution.problem_ptr->integer_indices, solution.handle_ptr->get_stream());
  f_t obj_offset = 0;
  // for each integer add the variable and the distance constraints
  for (auto i : h_integer_indices) {
    if (solution.problem_ptr->integer_equal(h_assignment[i], h_variable_upper_bounds[i])) {
      obj_offset += h_variable_upper_bounds[i];
      // set the objective weight to -1,  u - x
      obj_coefficients[i] = -1;
    } else if (solution.problem_ptr->integer_equal(h_assignment[i], h_variable_lower_bounds[i])) {
      obj_offset -= h_variable_lower_bounds[i];
      // set the objective weight to +1,  x - l
      obj_coefficients[i] = 1;
    } else {
      // objective weight is 1
      const f_t obj_weight = 1.;
      // the distance should always be positive
      i_t var_id = h_variables.add_variable(
        0,
        (h_variable_upper_bounds[i] - h_variable_lower_bounds[i]) + int_tol,
        obj_weight,
        var_t::CONTINUOUS);
      obj_coefficients.push_back(obj_weight);
      h_assignment.push_back(0.);
      std::vector<i_t> constr_indices{var_id, i};
      // d_j - x_j >= -val(x_j)
      std::vector<f_t> constr_coeffs_1{1, -1};
      h_constraints.add_constraint(
        constr_indices, constr_coeffs_1, -h_assignment[i], (f_t)default_cont_upper);
      // d_j + x_j >= val(x_j)
      std::vector<f_t> constr_coeffs_2{1, 1};
      h_constraints.add_constraint(
        constr_indices, constr_coeffs_2, h_assignment[i], (f_t)default_cont_upper);
    }
  }
  adjust_objective_with_original(solution, obj_coefficients, longer_lp_run);
  // commit all the changes that were done by the host
  if (h_variables.size() > 0) { temp_p.insert_variables(h_variables); }
  if (h_constraints.n_constraints() > 0) { temp_p.insert_constraints(h_constraints); }
  if (h_constraints.n_constraints() > 0 || h_variables.size() > 0) {
    temp_p.compute_transpose_of_problem();
  }
  cuopt_assert(h_assignment.size() == temp_p.n_variables, "Var count mismatch!");
  cuopt_assert(temp_p.objective_coefficients.size() == temp_p.n_variables, "Var count mismatch!");
  solution.copy_new_assignment(h_assignment);
  cuopt_assert(solution.assignment.size() == temp_p.n_variables, "Var count mismatch!");
  // copy new objective coefficients
  raft::copy(temp_p.objective_coefficients.data(),
             obj_coefficients.data(),
             obj_coefficients.size(),
             solution.handle_ptr->get_stream());
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream());
  temp_p.presolve_data.objective_offset = obj_offset;
  // change the precision between 1. and 10-4 depending on the integer ratio
  // the lp tolerance can be pretty high
  const double lp_tolerance =
    get_tolerance_from_ratio(ratio_of_set_integers, context.settings.tolerances.absolute_tolerance);
  temp_p.check_problem_representation(true);
  f_t time_limit                 = longer_lp_run ? 5. : 1.;
  time_limit                     = min(time_limit, timer.remaining_time());
  static f_t lp_time             = 0;
  static i_t n_calls             = 0;
  f_t old_remaining              = timer.remaining_time();
  const bool check_infeasibility = false;
  cuopt_func_call(solution.test_variable_bounds(false));
  auto solver_response =
    get_relaxed_lp_solution(temp_p, solution, f_t(lp_tolerance), time_limit, check_infeasibility);
  cuopt_func_call(solution.test_variable_bounds(false));
  last_lp_time = old_remaining - timer.remaining_time();
  lp_time += last_lp_time;
  n_calls++;
  CUOPT_LOG_DEBUG("lp_time %f average lp_time %f", last_lp_time, lp_time / n_calls);
  solution.assignment.resize(solution.problem_ptr->n_variables, solution.handle_ptr->get_stream());
  raft::copy(last_projection.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  // Projection result might be feasible but not optimal, due to time limits
  bool is_feasible = solution.compute_feasibility();
  cuopt_func_call(solution.test_variable_bounds(false));
  if (!is_feasible) {
    CUOPT_LOG_DEBUG("LP is infeasible returning the current PDLP solution! Code %d",
                    (int)solver_response.get_termination_status());
    return false;
  }
  // normal feasible return
  return true;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::random_round_with_fj(solution_t<i_t, f_t>& solution,
                                                        timer_t& round_timer)
{
  const i_t n_tries = 200;
  bool is_feasible  = false;
  rmm::device_uvector<f_t> original_assign(solution.assignment, solution.handle_ptr->get_stream());
  rmm::device_uvector<f_t> best_rounding(solution.assignment, solution.handle_ptr->get_stream());
  f_t best_fj_qual = std::numeric_limits<f_t>::max();
  i_t i;
  for (i = 0; i < n_tries; ++i) {
    if (round_timer.check_time_limit()) { break; }
    CUOPT_LOG_DEBUG("Trying random with FJ");
    is_feasible = solution.round_nearest();
    if (is_feasible) {
      CUOPT_LOG_DEBUG("Feasible found after random round");
      return true;
    }
    // run fj_single descent
    fj.settings.mode = fj_mode_t::GREEDY_DESCENT;
    // fj.settings.n_of_minimums_for_exit = 1;
    fj.settings.update_weights  = false;
    fj.settings.feasibility_run = true;
    fj.settings.time_limit      = round_timer.remaining_time();
    i_t original_sync_period    = fj.settings.parameters.sync_period;
    // iterations_per_graph is 10
    fj.settings.parameters.sync_period = 500;
    is_feasible                        = fj.solve(solution);
    fj.settings.parameters.sync_period = original_sync_period;
    cuopt_assert(solution.test_number_all_integer(), "All integers must be rounded");
    if (is_feasible) {
      CUOPT_LOG_DEBUG("Feasible found after random round FJ run");
      return true;
    }
    f_t current_qual = solution.get_quality(fj.cstr_weights, fj.objective_weight);
    if (current_qual < best_fj_qual) {
      CUOPT_LOG_DEBUG("Updating best quality of roundings %f", current_qual);
      best_fj_qual = current_qual;
      raft::copy(best_rounding.data(),
                 solution.assignment.data(),
                 solution.assignment.size(),
                 solution.handle_ptr->get_stream());
    }
    raft::copy(solution.assignment.data(),
               original_assign.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
  }
  n_fj_single_descents                       = i;
  const i_t min_sols_to_run_fj               = 1;
  const i_t non_improving_local_minima_count = 200;
  // if at least 5 solutions are explored, run longer fj
  if (i >= min_sols_to_run_fj) {
    // run longer fj on best sol
    raft::copy(solution.assignment.data(),
               best_rounding.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    rmm::device_uvector<f_t> original_weights(fj.cstr_weights, solution.handle_ptr->get_stream());
    fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    fj.settings.update_weights         = true;
    fj.settings.feasibility_run        = true;
    fj.settings.time_limit             = 0.5;
    fj.settings.n_of_minimums_for_exit = non_improving_local_minima_count;
    is_feasible                        = fj.solve(solution);
    // restore the weights
    raft::copy(fj.cstr_weights.data(),
               original_weights.data(),
               fj.cstr_weights.size(),
               solution.handle_ptr->get_stream());
  }
  if (is_feasible) {
    CUOPT_LOG_DEBUG("Feasible found after %d minima FJ run", non_improving_local_minima_count);
    return true;
  }
  raft::copy(solution.assignment.data(),
             original_assign.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  solution.handle_ptr->sync_stream();
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::round_multiple_points(solution_t<i_t, f_t>& solution)
{
  n_fj_single_descents     = 0;
  const f_t max_time_limit = last_lp_time * 0.1;
  timer_t round_timer{min(max_time_limit, timer.remaining_time())};
  bool is_feasible = random_round_with_fj(solution, round_timer);
  if (is_feasible) {
    CUOPT_LOG_DEBUG("Feasible found after random round with fj");
    return true;
  }
  timer_t line_segment_timer{min(1., timer.remaining_time())};
  i_t n_points_to_search  = n_fj_single_descents;
  bool is_feasibility_run = true;
  // create a copy, because assignment is changing within kernel and we want a separate point_1
  rmm::device_uvector<f_t> starting_point(solution.assignment, solution.handle_ptr->get_stream());
  is_feasible = line_segment_search.search_line_segment(solution,
                                                        starting_point,
                                                        lp_optimal_solution,
                                                        n_points_to_search,
                                                        is_feasibility_run,
                                                        line_segment_timer);
  if (is_feasible) {
    CUOPT_LOG_DEBUG("Feasible found after line segment");
    return true;
  }
  // lns.config.run_lp_and_loop = false;
  // timer_t lns_timer(min(last_lp_time * 0.1, timer.remaining_time()));
  // is_feasible = lns.do_lns(solution, lns_timer);
  // if (is_feasible) {
  //   CUOPT_LOG_DEBUG("Feasible found after inevitable feasibility");
  //   return true;
  // }
  // TODO add the solution with the min distance to the population, if population is given
  is_feasible = solution.round_nearest();
  if (is_feasible) {
    CUOPT_LOG_DEBUG("Feasible found after nearest rounding");
    return true;
  }
  return is_feasible;
}

// round will use inevitable infeasibility while propagating the bounds
template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::round(solution_t<i_t, f_t>& solution)
{
  bool result;
  CUOPT_LOG_DEBUG("Rounding the point");
  timer_t bounds_prop_timer(min(2., timer.remaining_time()));
  const f_t lp_run_time_after_feasible = min(3., timer.remaining_time() / 20.);
  result = lb_constraint_prop.apply_round(solution, lp_run_time_after_feasible, bounds_prop_timer);
  cuopt_func_call(solution.test_variable_bounds(true));
  // copy the last rounding
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  if (result) {
    CUOPT_LOG_DEBUG("New feasible solution with objective %g", solution.get_user_objective());
  }
  return result;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::perturbate(solution_t<i_t, f_t>& solution)
{
  constexpr f_t change_ratio = 0.1;
  solution.assign_random_within_bounds(change_ratio, true);
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_fj_cycle_escape(solution_t<i_t, f_t>& solution)
{
  bool is_feasible;
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.n_of_minimums_for_exit = 5000;
  fj.settings.time_limit             = min(3., timer.remaining_time());
  fj.settings.termination            = fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT;
  is_feasible                        = fj.solve(solution);
  // if FJ didn't change the solution, take last incumbent solution
  if (!is_feasible && cycle_queue.check_cycle(solution)) {
    CUOPT_LOG_DEBUG("cycle detected after FJ, taking last incumbent of fj");
    raft::copy(solution.assignment.data(),
               fj.climbers[0]->incumbent_assignment.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::test_fj_feasible(solution_t<i_t, f_t>& solution, f_t time_limit)
{
  CUOPT_LOG_DEBUG("Running 20%% with %f time limit", time_limit);
  bool is_feasible;
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.n_of_minimums_for_exit = 5000;
  fj.settings.time_limit             = min(time_limit, timer.remaining_time());
  fj.settings.termination            = fj_termination_flags_t::FJ_TERMINATION_TIME_LIMIT;
  cuopt_func_call(solution.test_variable_bounds(true));
  is_feasible = fj.solve(solution);
  cuopt_func_call(solution.test_variable_bounds(true));
  // if FJ didn't change the solution, take last incumbent solution
  if (!is_feasible) {
    raft::copy(solution.assignment.data(),
               last_rounding.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
    cuopt_func_call(solution.test_variable_bounds(true));
  } else {
    CUOPT_LOG_DEBUG("20%% FJ run found feasible!");
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::handle_cycle(solution_t<i_t, f_t>& solution)
{
  CUOPT_LOG_DEBUG("running handle cycle");
  bool is_feasible       = false;
  fp_fj_cycle_time_begin = timer.remaining_time();
  CUOPT_LOG_DEBUG("Running longer FJ on last rounding");
  raft::copy(solution.assignment.data(),
             last_rounding.data(),
             last_rounding.size(),
             solution.handle_ptr->get_stream());
  cuopt_func_call(solution.test_variable_bounds(true));
  cuopt_assert(solution.test_number_all_integer(), "All must be integers before fj");
  is_feasible = run_fj_cycle_escape(solution);
  cuopt_assert(solution.test_number_all_integer(), "All must be integers after fj");
  if (!is_feasible && cycle_queue.check_cycle(solution)) {
    CUOPT_LOG_DEBUG("Cycle couldn't be broken. Perturbating FP");
    perturbate(solution);
  }
  cycle_queue.n_iterations_without_cycle = 0;
  cycle_queue.update_recent_solutions(solution);
  if (is_feasible) {
    solution.test_feasibility();
    CUOPT_LOG_DEBUG("Feasible found cycle breaking long FJ");
  }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::restart_fp(solution_t<i_t, f_t>& solution)
{
  bool is_feasible = handle_cycle(solution);
  // reset the distance
  last_distances.resize(0);
  thrust::transform(solution.handle_ptr->get_thrust_policy(),
                    fj.cstr_weights.begin(),
                    fj.cstr_weights.end(),
                    fj.cstr_weights.begin(),
                    [] __device__(f_t val) {
                      constexpr f_t weight_divisor = 10.;
                      return max(f_t(10.), std::round(val / weight_divisor));
                    });
  return is_feasible;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::reset()
{
  best_excess               = std::numeric_limits<f_t>::infinity();
  total_fp_time_until_cycle = 0;
  fp_fj_cycle_time_begin    = timer.remaining_time();
  max_n_of_integers         = 0;
  run_intensive_restart     = false;
  config.alpha              = default_alpha;
  last_distances.resize(0);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::resize_vectors(problem_t<i_t, f_t>& problem,
                                                  const raft::handle_t* handle_ptr)
{
  last_rounding.resize(problem.n_variables, handle_ptr->get_stream());
  last_projection.resize(problem.n_variables, handle_ptr->get_stream());
  best_excess_solution.resize(problem.n_variables, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::check_distance_cycle(solution_t<i_t, f_t>& solution)
{
  f_t distance_to_last_rounding = compute_l1_distance<i_t, f_t>(
    solution.problem_ptr->integer_indices, last_rounding, solution.assignment, solution.handle_ptr);

  bool is_cycle = false;
  if (last_distances.size() == (size_t)config.first_stage_kk) {
    // perf is not important, very small array
    f_t avg_distance =
      std::accumulate(last_distances.begin(), last_distances.end(), 0.0) / last_distances.size();
    if (avg_distance - distance_to_last_rounding <
        config.cycle_distance_reduction_ration * avg_distance) {
      CUOPT_LOG_DEBUG("Distance cycle detected curr %f avg %f for last %d iter",
                      distance_to_last_rounding,
                      avg_distance,
                      last_distances.size());
      is_cycle = true;
    }
    last_distances.pop_back();
  } else {
    CUOPT_LOG_DEBUG("Distance of projection: %f", distance_to_last_rounding);
  }
  last_distances.push_front(distance_to_last_rounding);
  return is_cycle;
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::save_best_excess_solution(solution_t<i_t, f_t>& solution)
{
  f_t sol_excess = solution.get_total_excess();
  if (sol_excess < best_excess) {
    CUOPT_LOG_DEBUG("FP: updating excess from %f to %f", best_excess, sol_excess);
    best_excess = sol_excess;
    raft::copy(best_excess_solution.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::relax_general_integers(solution_t<i_t, f_t>& solution)
{
  orig_variable_types.resize(solution.problem_ptr->n_variables, solution.handle_ptr->get_stream());

  auto var_types  = make_span(solution.problem_ptr->variable_types);
  auto var_lb     = make_span(solution.problem_ptr->variable_lower_bounds);
  auto var_ub     = make_span(solution.problem_ptr->variable_upper_bounds);
  auto copy_types = make_span(orig_variable_types);

  raft::copy(orig_variable_types.data(),
             solution.problem_ptr->variable_types.data(),
             orig_variable_types.size(),
             solution.handle_ptr->get_stream());
  thrust::for_each(
    solution.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator<i_t>(0),
    thrust::make_counting_iterator<i_t>(solution.problem_ptr->n_variables),
    [var_types, var_lb, var_ub, copy_types, pb = solution.problem_ptr->view()] __device__(
      auto v_idx) {
      auto orig_v_type = var_types[v_idx];
      auto lb          = var_lb[v_idx];
      auto ub          = var_ub[v_idx];
      bool var_binary  = (pb.integer_equal(lb, 0) && pb.integer_equal(ub, 1));
      auto copy_type =
        (orig_v_type == var_t::INTEGER) && var_binary ? var_t::INTEGER : var_t::CONTINUOUS;
      var_types[v_idx] = copy_type;
    });
  solution.handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream());
  solution.problem_ptr->compute_n_integer_vars();
  solution.problem_ptr->compute_binary_var_table();
  CUOPT_LOG_DEBUG("Integers are relaxed n_int vars %d n_binary vars %d n_vars %d",
                  solution.problem_ptr->n_integer_vars,
                  solution.problem_ptr->n_binary_vars,
                  solution.problem_ptr->n_variables);
}

template <typename i_t, typename f_t>
void feasibility_pump_t<i_t, f_t>::revert_relaxation(solution_t<i_t, f_t>& solution)
{
  cuopt_assert(orig_variable_types.size() == solution.problem_ptr->variable_types.size(),
               "variable size mismatch");
  std::swap(orig_variable_types, solution.problem_ptr->variable_types);
  solution.problem_ptr->compute_n_integer_vars();
  solution.problem_ptr->compute_binary_var_table();
  solution.compute_feasibility();
}

template <typename i_t, typename f_t>
bool feasibility_pump_t<i_t, f_t>::run_single_fp_descent(solution_t<i_t, f_t>& solution)
{
  // start by doing nearest rounding
  solution.round_nearest();
  raft::copy(last_rounding.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  while (true) {
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      round(solution);
      return false;
    }
    proj_begin = timer.remaining_time();
    // pass n_assigned_integers from the previous iteration
    f_t ratio_of_assigned_integers =
      f_t(solution.n_assigned_integers) / solution.problem_ptr->n_integer_vars;
    bool is_feasible = linear_project_onto_polytope(solution, ratio_of_assigned_integers);
    i_t n_integers   = solution.compute_number_of_integers();
    CUOPT_LOG_DEBUG("after fp projection n_integers %d total n_integes %d",
                    n_integers,
                    solution.problem_ptr->n_integer_vars);
    bool is_cycle = true;
    // temp comment for presolve run
    if (config.check_distance_cycle) {
      // use distance cycle if we are running ii or objective FP
      is_cycle = check_distance_cycle(solution);
      if (is_cycle) {
        is_feasible = round(solution);
        save_best_excess_solution(solution);
        cuopt_func_call(solution.test_variable_bounds(true));
        if (is_feasible) {
          bool res = solution.compute_feasibility();
          cuopt_assert(res, "Feasibility issue");
          return true;
        }
        cuopt::default_logger().flush();
        f_t remaining_time_end_fp = timer.remaining_time();
        total_fp_time_until_cycle = fp_fj_cycle_time_begin - remaining_time_end_fp;
        CUOPT_LOG_DEBUG("total_fp_time_until_cycle: %f", total_fp_time_until_cycle);
        return false;
      }
    }
    // if it is feasible check if all are still integer
    if (n_integers == solution.problem_ptr->n_integer_vars) {
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Feasible solution found after LP with relative tolerance");
        save_best_excess_solution(solution);
        return true;
      }
      // if the solution is almost on polytope
      else if (last_distances[0] < distance_to_check_for_feasible) {
        // run the linear projection with full precision to check if it actually is feasible
        const bool return_first_feasible = true;
        const f_t lp_verify_time_limit   = 5.;
        run_lp_with_vars_fixed(*solution.problem_ptr,
                               solution,
                               solution.problem_ptr->integer_indices,
                               context.settings.get_tolerances(),
                               context.lp_state,
                               lp_verify_time_limit,
                               return_first_feasible,
                               &constraint_prop.bounds_update);
        is_feasible = solution.get_feasible();
        n_integers  = solution.compute_number_of_integers();
        if (is_feasible && n_integers == solution.problem_ptr->n_integer_vars) {
          CUOPT_LOG_DEBUG("Feasible solution verified with lower precision!");
          save_best_excess_solution(solution);
          return true;
        }
      }
    }
    cuopt_func_call(solution.test_variable_bounds(false));
    is_feasible = round(solution);
    cuopt_func_call(solution.test_variable_bounds(true));
    proj_and_round_time = proj_begin - timer.remaining_time();
    if (!is_feasible) {
      const f_t time_ratio = 0.2;
      is_feasible          = test_fj_feasible(solution, time_ratio * proj_and_round_time);
    }
    save_best_excess_solution(solution);
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("FP time limit reached!");
      return false;
    }
    if (is_feasible) {
      bool res = solution.compute_feasibility();
      cuopt_assert(res, "Feasibility issue");
      return true;
    }
    // do the cycle check if alpha diff is small enough
    f_t alpha_at_earlier_iter = config.alpha / config.alpha_decrease_factor;
    if (alpha_at_earlier_iter - config.alpha < 0.005) {
      is_cycle = cycle_queue.check_cycle(solution);
    }
    cycle_queue.update_recent_solutions(solution);
    if (is_cycle) {
      CUOPT_LOG_DEBUG("FP cycle encountered");
      f_t remaining_time_end_fp = timer.remaining_time();
      total_fp_time_until_cycle = fp_fj_cycle_time_begin - remaining_time_end_fp;
      CUOPT_LOG_DEBUG(
        "remaining_time_end_fp %f fp_fj_cycle_time_begin %f total_fp_time_until_cycle: %f",
        remaining_time_end_fp,
        fp_fj_cycle_time_begin,
        total_fp_time_until_cycle);
      return false;
    }
    cycle_queue.n_iterations_without_cycle++;
  }
  // unreachable
  return false;
}

#if MIP_INSTANTIATE_FLOAT
template class feasibility_pump_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class feasibility_pump_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
