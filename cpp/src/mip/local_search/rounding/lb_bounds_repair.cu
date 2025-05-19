/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "lb_bounds_repair.cuh"

#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <mip/logger.cuh>
#include <mip/mip_constants.hpp>
#include <utilities/seed_generator.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_bounds_repair_t<i_t, f_t>::lb_bounds_repair_t(const raft::handle_t* handle_ptr)
  : candidates(handle_ptr),
    best_bounds(handle_ptr),
    cstr_violations_up(0, handle_ptr->get_stream()),
    cstr_violations_down(0, handle_ptr->get_stream()),
    violated_constraints(0, handle_ptr->get_stream()),
    violated_cstr_map(0, handle_ptr->get_stream()),
    total_vio(handle_ptr->get_stream()),
    gen(cuopt::seed_generator::get_seed()),
    cycle_vector(MAX_CYCLE_SEQUENCE, -1)
{
}

template <typename i_t, typename f_t>
void lb_bounds_repair_t<i_t, f_t>::resize(const load_balanced_problem_t<i_t, f_t>& problem)
{
  candidates.resize(problem.n_variables, handle_ptr);
  best_bounds.resize(problem.n_variables, handle_ptr);
  cstr_violations_up.resize(problem.n_constraints, handle_ptr->get_stream());
  cstr_violations_down.resize(problem.n_constraints, handle_ptr->get_stream());
  violated_constraints.resize(problem.n_constraints, handle_ptr->get_stream());
  violated_cstr_map.resize(problem.n_constraints, handle_ptr->get_stream());
  cycle_vector.assign(MAX_CYCLE_SEQUENCE, -1);
  cycle_write_pos = 0;
}

template <typename i_t, typename f_t>
void lb_bounds_repair_t<i_t, f_t>::reset()
{
  candidates.n_candidates.set_value_to_zero_async(handle_ptr->get_stream());
  total_vio.set_value_to_zero_async(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
std::tuple<f_t, i_t> lb_bounds_repair_t<i_t, f_t>::get_ii_violation(
  load_balanced_problem_t<i_t, f_t>& problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bound_presolve)
{
  using f_t2 = typename type_2<f_t>::type;
  lb_bound_presolve.copy_input_bounds(problem);
  lb_bound_presolve.calculate_constraint_slack(problem.handle_ptr);
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem.n_constraints,
    [tolerances              = problem.tolerances,
     violated_cstr_map       = violated_cstr_map.data(),
     constraint_lower_bounds = problem.constraint_lower_bounds,
     constraint_upper_bounds = problem.constraint_upper_bounds,
     cnst_slack              = make_span_2(lb_bound_presolve.cnst_slack),
     cstr_violations_up      = cstr_violations_up.data(),
     cstr_violations_down    = cstr_violations_down.data(),
     total_vio               = total_vio.data()] __device__(i_t cstr_idx) {
      f_t cnst_lb = constraint_lower_bounds[cstr_idx];
      f_t cnst_ub = constraint_upper_bounds[cstr_idx];
      f_t2 slack  = cnst_slack[cstr_idx];
      f_t eps     = get_cstr_tolerance<i_t, f_t>(
        cnst_lb, cnst_ub, tolerances.absolute_tolerance, tolerances.relative_tolerance);
      f_t curr_cstr_violation_up   = max(0., -slack.x - eps);
      f_t curr_cstr_violation_down = max(0., slack.y - eps);
      f_t violation                = max(curr_cstr_violation_up, curr_cstr_violation_down);
      if (violation >= ROUNDOFF_TOLERANCE) {
        violated_cstr_map[cstr_idx] = 1;
        atomicAdd(total_vio, violation);
      } else {
        violated_cstr_map[cstr_idx] = 0;
      }
      cstr_violations_up[cstr_idx]   = curr_cstr_violation_up;
      cstr_violations_down[cstr_idx] = curr_cstr_violation_down;
    });
  auto iter           = thrust::copy_if(handle_ptr->get_thrust_policy(),
                              thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(0) + problem.n_constraints,
                              violated_cstr_map.data(),
                              violated_constraints.data(),
                              cuda::std::identity{});
  i_t n_violated_cstr = iter - violated_constraints.data();
  f_t total_violation = total_vio.value(handle_ptr->get_stream());
  CUOPT_LOG_TRACE(
    "Repair: n_violated_cstr %d total_violation %f", n_violated_cstr, total_violation);
  return std::make_tuple(total_violation, n_violated_cstr);
}

template <typename i_t, typename f_t>
i_t lb_bounds_repair_t<i_t, f_t>::get_random_cstr()
{
  std::uniform_int_distribution<> dist(0, h_n_violated_cstr - 1);
  // Generate random number
  i_t random_number = dist(gen);
  i_t cstr_idx      = violated_constraints.element(random_number, handle_ptr->get_stream());
  CUOPT_LOG_TRACE("Repair: selected random cstr %d", cstr_idx);
  return cstr_idx;
}

template <typename i_t, typename f_t>
i_t lb_bounds_repair_t<i_t, f_t>::compute_best_shift(load_balanced_problem_t<i_t, f_t>& problem,
                                                     problem_t<i_t, f_t>& original_problem,
                                                     i_t curr_cstr)
{
  // for each variable in the constraint, compute the best shift value.
  // if the shift value doesn't change the violation at all, set it to 0
  i_t cstr_offset      = original_problem.offsets.element(curr_cstr, handle_ptr->get_stream());
  i_t cstr_offset_next = original_problem.offsets.element(curr_cstr + 1, handle_ptr->get_stream());
  i_t cstr_size        = cstr_offset_next - cstr_offset;
  CUOPT_LOG_TRACE(
    "Computing best shift for the vars in cstr %d cstr size %d", curr_cstr, cstr_size);
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(cstr_offset),
    thrust::make_counting_iterator(cstr_offset + cstr_size),
    [candidates           = candidates.view(),
     cstr_violations_up   = cstr_violations_up.data(),
     cstr_violations_down = cstr_violations_down.data(),
     variable_bounds      = make_span_2(problem.variable_bounds),
     o_pb_v               = original_problem.view(),
     curr_cstr] __device__(i_t idx) {
      i_t var_idx      = o_pb_v.variables[idx];
      f_t shift_amount = 0.;
      f_t var_coeff    = o_pb_v.coefficients[idx];
      cuopt_assert(var_coeff != 0., "Var coeff can't be zero");
      if (f_t up_vio = cstr_violations_up[curr_cstr]; up_vio > 0) {
        shift_amount = -(up_vio / var_coeff);
      } else if (f_t down_vio = cstr_violations_down[curr_cstr]; down_vio > 0) {
        shift_amount = (down_vio / var_coeff);
      }
      if (shift_amount != 0.) {
        auto var_bnds = variable_bounds[var_idx];
        auto var_lb   = var_bnds.x;
        auto var_ub   = var_bnds.y;
        f_t o_var_lb  = o_pb_v.variable_lower_bounds[var_idx];
        f_t o_var_ub  = o_pb_v.variable_upper_bounds[var_idx];
        cuopt_assert(var_lb + o_pb_v.tolerances.integrality_tolerance >= o_var_lb, "");
        cuopt_assert(o_var_ub + o_pb_v.tolerances.integrality_tolerance >= var_ub, "");
        // round the shift amount of integer
        if (o_pb_v.is_integer_var(var_idx)) {
          shift_amount = shift_amount > 0 ? ceil(shift_amount) : floor(shift_amount);
        }
        // clip the shift such that the bounds are within original bounds
        // TODO check whether shifting only one side works better instead of both sides
        if (var_lb + shift_amount < o_var_lb) {
          DEVICE_LOG_TRACE(
            "Changing shift value of var %d from %f to %f since var_lb %f o_var_lb %f\n",
            var_idx,
            shift_amount,
            var_lb - o_var_lb,
            var_lb,
            o_var_lb);
          shift_amount = var_lb - o_var_lb;
        }
        if (var_ub + shift_amount > o_var_ub) {
          DEVICE_LOG_TRACE(
            "Changing shift value of var %d from %f to %f since var_ub %f o_var_ub %f\n",
            var_idx,
            shift_amount,
            o_var_ub - var_ub,
            var_ub,
            o_var_ub);
          shift_amount = o_var_ub - var_ub;
        }
        // if the var is not a singleton, don't consider the candidate unless at least one singleton
        // has moved
        bool check_for_singleton_move =
          *candidates.at_least_one_singleton_moved || o_pb_v.integer_equal(var_lb, var_ub);
        // shift amount can be zero most of the time
        if (shift_amount != 0. && check_for_singleton_move) {
          // TODO check if atomics are heavy, if so implement a map and compact outside
          i_t cand_idx                        = atomicAdd(candidates.n_candidates, 1);
          candidates.variable_index[cand_idx] = var_idx;
          candidates.bound_shift[cand_idx]    = shift_amount;
        }
      }
    });
  handle_ptr->sync_stream();
  return candidates.n_candidates.value(handle_ptr->get_stream());
}

template <typename i_t, typename f_t, typename f_t2>
__global__ void compute_damages_kernel(typename problem_t<i_t, f_t>::view_t original_problem,
                                       typename candidates_t<i_t, f_t>::view_t candidates,
                                       raft::device_span<f_t2> variable_bounds,
                                       raft::device_span<f_t> cstr_violations_up,
                                       raft::device_span<f_t> cstr_violations_down,
                                       raft::device_span<f_t2> cnst_slack)
{
  i_t var_idx                     = candidates.variable_index[blockIdx.x];
  f_t shift_amount                = candidates.bound_shift[blockIdx.x];
  auto var_bnd                    = variable_bounds[var_idx];
  f_t v_lb                        = var_bnd.x;
  f_t v_ub                        = var_bnd.y;
  f_t th_damage                   = 0.;
  i_t n_infeasible_cstr_delta     = 0;
  auto [offset_begin, offset_end] = original_problem.reverse_range_for_var(var_idx);
  // loop over all constraints that the variable appears in
  for (i_t c_idx = threadIdx.x + offset_begin; c_idx < offset_end; c_idx += blockDim.x) {
    // compute the "damage": the delta between the current violation and the violation after the
    // shift
    i_t c             = original_problem.reverse_constraints[c_idx];
    f_t coeff         = original_problem.reverse_coefficients[c_idx];
    f_t curr_up_vio   = cstr_violations_up[c];
    f_t curr_down_vio = cstr_violations_down[c];
    // in an infeasible constraint both might have a value, the definition in the paper is max
    f_t curr_vio = max(curr_up_vio, curr_down_vio);
    // now compute the new vio
    f_t cnst_lb             = original_problem.constraint_lower_bounds[c];
    f_t cnst_ub             = original_problem.constraint_upper_bounds[c];
    f_t shift_in_activities = shift_amount * coeff;
    auto slack              = cnst_slack[c];
    f_t eps                 = get_cstr_tolerance<i_t, f_t>(cnst_lb,
                                           cnst_ub,
                                           original_problem.tolerances.absolute_tolerance,
                                           original_problem.tolerances.relative_tolerance);
    // Given
    // f_t new_min_act         = minimum_activity[c] + shift_in_activities;
    // f_t new_max_act         = maximum_activity[c] + shift_in_activities;
    //
    // f_t new_violations_up   = max(0., new_min_act - (cnst_ub + eps));
    // f_t new_violations_down = max(0., cnst_lb - eps - new_max_act);
    //  becomes
    // f_t new_violations_up   = max(0., shift_in_activities - (cnst_ub - min_act) - eps);
    // f_t new_violations_down = max(0., (cnst_lb - max_act) - shift_in_activities - eps);
    //  becomes
    f_t new_violations_up   = max(0., shift_in_activities - slack.x - eps);
    f_t new_violations_down = max(0., slack.y - shift_in_activities - eps);

    f_t new_vio         = max(new_violations_up, new_violations_down);
    i_t curr_cstr_delta = i_t(curr_vio < ROUNDOFF_TOLERANCE) - i_t(new_vio < ROUNDOFF_TOLERANCE);
    n_infeasible_cstr_delta += curr_cstr_delta;
    th_damage += max(0., new_vio - curr_vio);
  }
  __shared__ f_t shmem[raft::WarpSize];
  f_t block_damage = raft::blockReduce(th_damage, (char*)shmem);
  __syncthreads();
  i_t block_infeasible_cstr_delta = raft::blockReduce(n_infeasible_cstr_delta, (char*)shmem);
  if (threadIdx.x == 0) {
    candidates.damage[blockIdx.x]     = block_damage;
    candidates.cstr_delta[blockIdx.x] = block_infeasible_cstr_delta;
  }
}

template <typename i_t, typename f_t>
void lb_bounds_repair_t<i_t, f_t>::compute_damages(
  load_balanced_problem_t<i_t, f_t>& problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bound_presolve,
  problem_t<i_t, f_t>& original_problem,
  i_t n_candidates)
{
  CUOPT_LOG_TRACE("Bounds repair: Computing damanges!");
  // TODO check performance, we can apply load balancing here
  const i_t TPB = 256;
  using f_t2    = typename type_2<f_t>::type;
  compute_damages_kernel<i_t, f_t, f_t2>
    <<<n_candidates, TPB, 0, handle_ptr->get_stream()>>>(original_problem.view(),
                                                         candidates.view(),
                                                         make_span_2(problem.variable_bounds),
                                                         make_span(cstr_violations_up),
                                                         make_span(cstr_violations_down),
                                                         make_span_2(lb_bound_presolve.cnst_slack));
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  auto sort_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(candidates.cstr_delta.data(), candidates.damage.data()));
  // sort the best moves so that we can filter
  thrust::sort_by_key(handle_ptr->get_thrust_policy(),
                      sort_iterator,
                      sort_iterator + n_candidates,
                      thrust::make_zip_iterator(thrust::make_tuple(
                        candidates.bound_shift.data(), candidates.variable_index.data())),
                      [] __device__(auto tuple1, auto tuple2) {
                        if (thrust::get<0>(tuple1) < thrust::get<0>(tuple2)) {
                          return true;
                        } else if (thrust::get<0>(tuple1) == thrust::get<0>(tuple2) &&
                                   thrust::get<1>(tuple1) < thrust::get<1>(tuple2)) {
                          return true;
                        }
                        return false;
                      });
}

template <typename i_t, typename f_t>
i_t lb_bounds_repair_t<i_t, f_t>::find_cutoff_index(const candidates_t<i_t, f_t>& candidates,
                                                    i_t best_cstr_delta,
                                                    f_t best_damage,
                                                    i_t n_candidates)
{
  auto iterator = thrust::make_zip_iterator(
    thrust::make_tuple(candidates.cstr_delta.data(), candidates.damage.data()));
  auto out_iter = thrust::partition_point(
    handle_ptr->get_thrust_policy(),
    iterator,
    iterator + n_candidates,
    [best_cstr_delta, best_damage] __device__(auto tuple) {
      if (thrust::get<0>(tuple) == best_cstr_delta && thrust::get<1>(tuple) <= best_damage) {
        return true;
      }
      return false;
    });
  return out_iter - iterator;
}

template <typename i_t, typename f_t>
i_t lb_bounds_repair_t<i_t, f_t>::get_random_idx(i_t size)
{
  std::uniform_int_distribution<> dist(0, size - 1);
  // Generate random number
  i_t random_number = dist(gen);
  return random_number;
}

// TODO convert this to var and test it.
template <typename i_t, typename f_t>
bool lb_bounds_repair_t<i_t, f_t>::detect_cycle(i_t cstr_idx)
{
  cycle_vector[cycle_write_pos] = cstr_idx;
  bool cycle_found              = false;
  for (i_t seq_length = cycle_vector.size() / 2; seq_length > 1; seq_length--) {
    // only check the two sliding windows, backward of cycle_write_pos
    i_t i = 0;
    for (; i < seq_length; i++) {
      if (cycle_vector[(cycle_write_pos - i + cycle_vector.size()) % cycle_vector.size()] !=
          cycle_vector[(cycle_write_pos - seq_length - i + cycle_vector.size()) %
                       cycle_vector.size()]) {
        break;
      }
    }
    // all sequence have equal length
    if (i == seq_length) {
      cycle_found = true;
      break;
    }
  }
  cycle_write_pos++;
  cycle_write_pos = cycle_write_pos % cycle_vector.size();
  return cycle_found;
}

template <typename i_t, typename f_t>
void lb_bounds_repair_t<i_t, f_t>::apply_move(load_balanced_problem_t<i_t, f_t>* problem,
                                              problem_t<i_t, f_t>& original_problem,
                                              i_t move_idx)
{
  run_device_lambda(handle_ptr->get_stream(),
                    [move_idx,
                     candidates = candidates.view(),
                     // problem          = problem.view(),
                     variable_bounds  = make_span_2(problem->variable_bounds),
                     original_problem = original_problem.view()] __device__() {
                      i_t var_idx     = candidates.variable_index[move_idx];
                      f_t shift_value = candidates.bound_shift[move_idx];
                      auto var_bnd    = variable_bounds[var_idx];
                      DEVICE_LOG_TRACE(
                        "Applying move on var %d with shift %f lb %f ub %f o_lb %f o_ub %f \n",
                        var_idx,
                        shift_value,
                        var_bnd.x,
                        var_bnd.y,
                        original_problem.variable_lower_bounds[var_idx],
                        original_problem.variable_upper_bounds[var_idx]);
                      if (original_problem.integer_equal(var_bnd.x, var_bnd.y)) {
                        *candidates.at_least_one_singleton_moved = 1;
                      }

                      // problem.variable_lower_bounds[var_idx] += shift_value;
                      // problem.variable_upper_bounds[var_idx] += shift_value;
                      var_bnd.x += shift_value;
                      var_bnd.y += shift_value;
                      variable_bounds[var_idx] = var_bnd;
                      cuopt_assert(original_problem.variable_lower_bounds[var_idx] <=
                                     var_bnd.x + original_problem.tolerances.integrality_tolerance,
                                   "");
                      cuopt_assert(original_problem.variable_upper_bounds[var_idx] +
                                       original_problem.tolerances.integrality_tolerance >=
                                     var_bnd.y,
                                   "");
                    });
}

template <typename i_t, typename f_t>
bool lb_bounds_repair_t<i_t, f_t>::repair_problem(
  load_balanced_problem_t<i_t, f_t>* problem,
  load_balanced_bounds_presolve_t<i_t, f_t>& lb_bound_presolve,
  problem_t<i_t, f_t>& original_problem,
  timer_t timer_,
  const raft::handle_t* handle_ptr_)
{
  CUOPT_LOG_DEBUG("Running bounds repair");
  handle_ptr = handle_ptr_;
  timer      = timer_;
  resize(*problem);
  reset();
  std::tie(best_violation, h_n_violated_cstr) = get_ii_violation(*problem, lb_bound_presolve);
  curr_violation                              = best_violation;
  best_bounds.update_from(*problem, handle_ptr);
  i_t no_candidate_in_a_row = 0;
  while (h_n_violated_cstr > 0) {
    CUOPT_LOG_TRACE("Bounds repair loop: n_violated %d best_violation %f curr_violation %f",
                    h_n_violated_cstr,
                    best_violation,
                    curr_violation);
    if (timer.check_time_limit()) { break; }
    i_t curr_cstr = get_random_cstr();
    // best way would be to check a variable cycle, but this is easier and more performant
    bool is_cycle = detect_cycle(curr_cstr);
    if (is_cycle) { CUOPT_LOG_DEBUG("Repair: cycle detected at cstr %d", curr_cstr); }
    // in parallel compute the best shift and best respective damage
    i_t n_candidates = compute_best_shift(*problem, original_problem, curr_cstr);
    // if no candidate is there continue with another constraint
    if (n_candidates == 0) {
      CUOPT_LOG_DEBUG("Repair: no candidate var found for cstr %d", curr_cstr);
      if (no_candidate_in_a_row++ == 10 || h_n_violated_cstr == 1) {
        CUOPT_LOG_DEBUG("Repair: no candidate var found on last violated constraint %d. Exiting...",
                        curr_cstr);
        break;
      }
      continue;
    }
    no_candidate_in_a_row = 0;
    CUOPT_LOG_TRACE("Repair: number of candidates %d", n_candidates);
    // among the ones that have a valid shift value, compute the damage
    compute_damages(*problem, lb_bound_presolve, original_problem, n_candidates);
    // get the best damage
    i_t best_cstr_delta = candidates.cstr_delta.front_element(handle_ptr->get_stream());
    f_t best_damage     = candidates.damage.front_element(handle_ptr->get_stream());
    CUOPT_LOG_TRACE(
      "Repair: best_cstr_delta value %d best_damage %f", best_cstr_delta, best_damage);
    i_t best_move_idx;
    // if the best damage is positive and we are within the prop (paper uses 0.75)
    if ((best_cstr_delta > 0 && rand_double(0, 1, gen) < p) || is_cycle) {
      // pick a random move from the candidate list
      best_move_idx = get_random_idx(n_candidates);
    } else {
      // filter the moves with best_damage(it can be zero or not) and then pick a candidate among
      // them
      i_t n_of_eligible_candidates =
        find_cutoff_index(candidates, best_cstr_delta, best_damage, n_candidates);
      cuopt_assert(n_of_eligible_candidates > 0, "");
      CUOPT_LOG_TRACE("n_of_eligible_candidates %d", n_of_eligible_candidates);
      best_move_idx = get_random_idx(n_of_eligible_candidates);
    }
    CUOPT_LOG_TRACE("Repair: selected best_move_idx %d var id %d shift %f cstr_delta %d damage %f",
                    best_move_idx,
                    candidates.variable_index.element(best_move_idx, handle_ptr->get_stream()),
                    candidates.bound_shift.element(best_move_idx, handle_ptr->get_stream()),
                    candidates.cstr_delta.element(best_move_idx, handle_ptr->get_stream()),
                    candidates.damage.element(best_move_idx, handle_ptr->get_stream()));
    apply_move(problem, original_problem, best_move_idx);
    reset();
    // TODO we might optimize this to only calculate the changed constraints
    std::tie(curr_violation, h_n_violated_cstr) = get_ii_violation(*problem, lb_bound_presolve);

    if (curr_violation < best_violation) {
      best_violation = curr_violation;
      // update best bounds
      best_bounds.update_from(*problem, handle_ptr);
    }
  }
  // fill the problem with the best bounds
  bool feasible = h_n_violated_cstr == 0;
  // copy best bounds into problem
  best_bounds.update_to(*problem, handle_ptr);
  CUOPT_LOG_DEBUG("Repair: returning with feas: %d vio %f", feasible, best_violation);
  return feasible;
}

#if MIP_INSTANTIATE_FLOAT
template class lb_bounds_repair_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_bounds_repair_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
