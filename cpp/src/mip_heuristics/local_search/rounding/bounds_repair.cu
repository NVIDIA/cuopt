/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "bounds_repair.cuh"

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cuda/std/functional>
#include <mip_heuristics/logger.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>

#include <cmath>

// enable to activate detailed determinism logs
#if 0
#undef CUOPT_DETERMINISM_LOG
#define CUOPT_DETERMINISM_LOG(...) \
  do {                             \
    CUOPT_LOG_INFO(__VA_ARGS__);   \
  } while (0)
#endif

namespace cuopt::linear_programming::detail {

namespace {

constexpr double bounds_repair_setup_base_work            = 5e-4;
constexpr double bounds_repair_violation_base_work        = 4e-4;
constexpr double bounds_repair_violation_nnz_work         = 2e-6;
constexpr double bounds_repair_violation_constraint_work  = 3e-6;
constexpr double bounds_repair_best_bounds_variable_work  = 2e-6;
constexpr double bounds_repair_shift_base_work            = 3e-4;
constexpr double bounds_repair_shift_row_entry_work       = 3e-6;
constexpr double bounds_repair_shift_candidate_work       = 8e-6;
constexpr double bounds_repair_shift_neighbor_entry_work  = 3e-6;
constexpr double bounds_repair_shift_sort_work            = 5e-6;
constexpr double bounds_repair_damage_base_work           = 3e-4;
constexpr double bounds_repair_damage_neighbor_entry_work = 8e-6;
constexpr double bounds_repair_damage_sort_work           = 5e-6;
constexpr double bounds_repair_move_base_work             = 5e-5;
constexpr double bounds_repair_no_candidate_base_work     = 4e-4;
constexpr double bounds_repair_cycle_penalty_work         = 3e-4;

template <typename i_t, typename f_t>
double estimate_bounds_repair_violation_refresh_work(const problem_t<i_t, f_t>& problem,
                                                     bool update_best_bounds)
{
  double estimate = bounds_repair_violation_base_work +
                    bounds_repair_violation_nnz_work * (double)problem.nnz +
                    bounds_repair_violation_constraint_work * (double)problem.n_constraints;
  if (update_best_bounds) {
    estimate += bounds_repair_best_bounds_variable_work * (double)problem.n_variables;
  }
  return estimate;
}

template <typename i_t, typename f_t>
double estimate_bounds_repair_setup_work(const problem_t<i_t, f_t>& problem)
{
  return bounds_repair_setup_base_work +
         estimate_bounds_repair_violation_refresh_work(problem, true);
}

template <typename i_t, typename f_t>
double estimate_bounds_repair_shift_work(const problem_t<i_t, f_t>& problem,
                                         i_t curr_cstr,
                                         i_t n_candidates,
                                         bool is_cycle)
{
  const auto stream    = problem.handle_ptr->get_stream();
  const i_t cstr_begin = problem.offsets.element(curr_cstr, stream);
  const i_t cstr_end   = problem.offsets.element(curr_cstr + 1, stream);
  const double row_nnz = cstr_end - cstr_begin;
  const double avg_rev_degree =
    problem.n_variables > 0 ? ((double)problem.nnz / (double)problem.n_variables) : 0.0;
  const double sort_work =
    n_candidates > 1 ? (double)n_candidates * std::log2((double)n_candidates) : 0.0;
  double estimate = bounds_repair_shift_base_work + bounds_repair_shift_row_entry_work * row_nnz;
  if (n_candidates == 0) { estimate = bounds_repair_no_candidate_base_work + estimate; }
  estimate += bounds_repair_shift_candidate_work * (double)n_candidates;
  estimate += bounds_repair_shift_neighbor_entry_work * (double)n_candidates * avg_rev_degree;
  estimate += bounds_repair_shift_sort_work * sort_work;
  if (is_cycle) { estimate += bounds_repair_cycle_penalty_work; }
  return estimate;
}

template <typename i_t, typename f_t>
double estimate_bounds_repair_damage_work(const problem_t<i_t, f_t>& problem, i_t n_candidates)
{
  if (n_candidates == 0) { return 0.0; }
  const double avg_rev_degree =
    problem.n_variables > 0 ? ((double)problem.nnz / (double)problem.n_variables) : 0.0;
  const double sort_work =
    n_candidates > 1 ? (double)n_candidates * std::log2((double)n_candidates) : 0.0;
  return bounds_repair_damage_base_work +
         bounds_repair_damage_neighbor_entry_work * (double)n_candidates * avg_rev_degree +
         bounds_repair_damage_sort_work * sort_work;
}

template <typename timer_t>
void record_estimated_work(timer_t& timer, double* total_estimated_work, double work)
{
  cuopt_assert(std::isfinite(work) && work >= 0.0, "Bounds repair work estimate must be finite");
  timer.record_work(work);
  *total_estimated_work += work;
}

}  // namespace

template <typename i_t, typename f_t>
bounds_repair_t<i_t, f_t>::bounds_repair_t(const problem_t<i_t, f_t>& pb,
                                           bound_presolve_t<i_t, f_t>& bound_presolve_)
  : bound_presolve(bound_presolve_),
    candidates(pb.handle_ptr),
    best_bounds(pb.handle_ptr),
    cstr_violations_up(0, pb.handle_ptr->get_stream()),
    cstr_violations_down(0, pb.handle_ptr->get_stream()),
    violated_constraints(0, pb.handle_ptr->get_stream()),
    violated_cstr_map(0, pb.handle_ptr->get_stream()),
    total_vio(pb.handle_ptr->get_stream()),
    gen(cuopt::seed_generator::get_seed()),
    cycle_vector(MAX_CYCLE_SEQUENCE, -1),
    timer(0.0, cuopt::termination_checker_t::root_tag_t{})
{
}

template <typename i_t, typename f_t>
void bounds_repair_t<i_t, f_t>::resize(const problem_t<i_t, f_t>& problem)
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
void bounds_repair_t<i_t, f_t>::reset()
{
  candidates.n_candidates.set_value_to_zero_async(handle_ptr->get_stream());
  total_vio.set_value_to_zero_async(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
f_t bounds_repair_t<i_t, f_t>::get_ii_violation(problem_t<i_t, f_t>& problem)
{
  bound_presolve.calculate_activity_on_problem_bounds(problem);
  // calculate the violation and mark of violated constraints
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem.n_constraints,
    [pb_v                 = problem.view(),
     violated_cstr_map    = violated_cstr_map.data(),
     min_act              = bound_presolve.upd.min_activity.data(),
     max_act              = bound_presolve.upd.max_activity.data(),
     cstr_violations_up   = cstr_violations_up.data(),
     cstr_violations_down = cstr_violations_down.data()] __device__(i_t cstr_idx) {
      f_t cnst_lb = pb_v.constraint_lower_bounds[cstr_idx];
      f_t cnst_ub = pb_v.constraint_upper_bounds[cstr_idx];
      f_t eps     = get_cstr_tolerance<i_t, f_t>(
        cnst_lb, cnst_ub, pb_v.tolerances.absolute_tolerance, pb_v.tolerances.relative_tolerance);
      f_t curr_cstr_violation_up   = max(0., min_act[cstr_idx] - (cnst_ub + eps));
      f_t curr_cstr_violation_down = max(0., cnst_lb - eps - max_act[cstr_idx]);
      f_t violation                = max(curr_cstr_violation_up, curr_cstr_violation_down);
      if (violation >= ROUNDOFF_TOLERANCE) {
        violated_cstr_map[cstr_idx] = 1;
      } else {
        violated_cstr_map[cstr_idx] = 0;
      }
      cstr_violations_up[cstr_idx]   = curr_cstr_violation_up;
      cstr_violations_down[cstr_idx] = curr_cstr_violation_down;
    });
  auto iter         = thrust::copy_if(handle_ptr->get_thrust_policy(),
                              thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(0) + problem.n_constraints,
                              violated_cstr_map.data(),
                              violated_constraints.data(),
                              cuda::std::identity{});
  h_n_violated_cstr = iter - violated_constraints.data();
  // Use deterministic reduction instead of non-deterministic atomicAdd
  f_t total_violation = thrust::transform_reduce(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + problem.n_constraints,
    [cstr_violations_up   = cstr_violations_up.data(),
     cstr_violations_down = cstr_violations_down.data()] __device__(i_t cstr_idx) -> f_t {
      auto violation = max(cstr_violations_up[cstr_idx], cstr_violations_down[cstr_idx]);
      return violation >= ROUNDOFF_TOLERANCE ? violation : 0.;
    },
    (f_t)0,
    thrust::plus<f_t>());
  CUOPT_LOG_TRACE(
    "Repair: n_violated_cstr %d total_violation %f", h_n_violated_cstr, total_violation);
  return total_violation;
}

template <typename i_t, typename f_t>
i_t bounds_repair_t<i_t, f_t>::get_random_cstr()
{
  std::uniform_int_distribution<> dist(0, h_n_violated_cstr - 1);
  i_t random_index = dist(gen);
  i_t cstr_idx     = violated_constraints.element(random_index, handle_ptr->get_stream());
  CUOPT_LOG_TRACE("Repair: selected random cstr %d", cstr_idx);
  CUOPT_DETERMINISM_LOG("Repair cstr select: random_index=%d cstr=%d n_violated=%d",
                        random_index,
                        cstr_idx,
                        h_n_violated_cstr);
  return cstr_idx;
}

template <typename i_t, typename f_t>
i_t bounds_repair_t<i_t, f_t>::compute_best_shift(problem_t<i_t, f_t>& problem,
                                                  problem_t<i_t, f_t>& original_problem,
                                                  i_t curr_cstr)
{
  // for each variable in the constraint, compute the best shift value.
  // if the shift value doesn't change the violation at all, set it to 0
  i_t cstr_offset      = problem.offsets.element(curr_cstr, handle_ptr->get_stream());
  i_t cstr_offset_next = problem.offsets.element(curr_cstr + 1, handle_ptr->get_stream());
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
     pb_v                 = problem.view(),
     o_pb_v               = original_problem.view(),
     curr_cstr] __device__(i_t idx) {
      i_t var_idx      = pb_v.variables[idx];
      f_t shift_amount = 0.;
      f_t var_coeff    = pb_v.coefficients[idx];
      cuopt_assert(var_coeff != 0., "Var coeff can't be zero");
      if (f_t up_vio = cstr_violations_up[curr_cstr]; up_vio > 0) {
        shift_amount = -(up_vio / var_coeff);
      } else if (f_t down_vio = cstr_violations_down[curr_cstr]; down_vio > 0) {
        shift_amount = (down_vio / var_coeff);
      }
      if (shift_amount != 0.) {
        auto var_bnd   = pb_v.variable_bounds[var_idx];
        auto o_var_bnd = o_pb_v.variable_bounds[var_idx];
        f_t var_lb     = get_lower(var_bnd);
        f_t var_ub     = get_upper(var_bnd);
        f_t o_var_lb   = get_lower(o_var_bnd);
        f_t o_var_ub   = get_upper(o_var_bnd);
        cuopt_assert(var_lb + pb_v.tolerances.integrality_tolerance >= o_var_lb, "");
        cuopt_assert(o_var_ub + pb_v.tolerances.integrality_tolerance >= var_ub, "");
        // round the shift amount of integer
        if (pb_v.is_integer_var(var_idx)) {
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
          shift_amount = o_var_lb - var_lb;
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
          *candidates.at_least_one_singleton_moved || pb_v.integer_equal(var_lb, var_ub);
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
  i_t n_candidates = candidates.n_candidates.value(handle_ptr->get_stream());

  // Sort by (variable_index, bound_shift) to ensure fully deterministic ordering
  auto key_iter = thrust::make_zip_iterator(
    thrust::make_tuple(candidates.variable_index.begin(), candidates.bound_shift.begin()));
  thrust::sort(handle_ptr->get_thrust_policy(), key_iter, key_iter + n_candidates);

  return n_candidates;
}

template <typename i_t, typename f_t>
__global__ void compute_damages_kernel(typename problem_t<i_t, f_t>::view_t problem,
                                       typename candidates_t<i_t, f_t>::view_t candidates,
                                       raft::device_span<f_t> cstr_violations_up,
                                       raft::device_span<f_t> cstr_violations_down,
                                       raft::device_span<f_t> minimum_activity,
                                       raft::device_span<f_t> maximum_activity)
{
  i_t var_idx                     = candidates.variable_index[blockIdx.x];
  f_t shift_amount                = candidates.bound_shift[blockIdx.x];
  auto v_bnd                      = problem.variable_bounds[var_idx];
  f_t v_lb                        = get_lower(v_bnd);
  f_t v_ub                        = get_upper(v_bnd);
  f_t th_damage                   = 0.;
  i_t n_infeasible_cstr_delta     = 0;
  auto [offset_begin, offset_end] = problem.reverse_range_for_var(var_idx);
  // loop over all constraints that the variable appears in
  for (i_t c_idx = threadIdx.x + offset_begin; c_idx < offset_end; c_idx += blockDim.x) {
    // compute the "damage": the delta between the current violation and the violation after the
    // shift
    i_t c             = problem.reverse_constraints[c_idx];
    f_t coeff         = problem.reverse_coefficients[c_idx];
    f_t curr_up_vio   = cstr_violations_up[c];
    f_t curr_down_vio = cstr_violations_down[c];
    // in an infeasible constraint both might have a value, the definition in the paper is max
    f_t curr_vio = max(curr_up_vio, curr_down_vio);
    // now compute the new vio
    f_t cnst_lb             = problem.constraint_lower_bounds[c];
    f_t cnst_ub             = problem.constraint_upper_bounds[c];
    f_t shift_in_activities = shift_amount * coeff;
    f_t new_min_act         = minimum_activity[c] + shift_in_activities;
    f_t new_max_act         = maximum_activity[c] + shift_in_activities;
    f_t eps                 = get_cstr_tolerance<i_t, f_t>(cnst_lb,
                                           cnst_ub,
                                           problem.tolerances.absolute_tolerance,
                                           problem.tolerances.relative_tolerance);
    f_t new_violations_up   = max(0., new_min_act - (cnst_ub + eps));
    f_t new_violations_down = max(0., cnst_lb - eps - new_max_act);
    f_t new_vio             = max(new_violations_up, new_violations_down);
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
void bounds_repair_t<i_t, f_t>::compute_damages(problem_t<i_t, f_t>& problem, i_t n_candidates)
{
  CUOPT_LOG_TRACE("Bounds repair: Computing damanges!");
  // TODO check performance, we can apply load balancing here
  const i_t TPB = 256;
  compute_damages_kernel<i_t, f_t><<<n_candidates, TPB, 0, handle_ptr->get_stream()>>>(
    problem.view(),
    candidates.view(),
    make_span(cstr_violations_up),
    make_span(cstr_violations_down),
    make_span(bound_presolve.upd.min_activity),
    make_span(bound_presolve.upd.max_activity));
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  auto sort_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(candidates.cstr_delta.data(), candidates.damage.data()));
  // sort the best moves so that we can filter
  thrust::sort_by_key(handle_ptr->get_thrust_policy(),
                      sort_iterator,
                      sort_iterator + n_candidates,
                      thrust::make_zip_iterator(thrust::make_tuple(
                        candidates.bound_shift.data(), candidates.variable_index.data())),
                      [] __device__(auto tuple1, auto tuple2) -> bool {
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
i_t bounds_repair_t<i_t, f_t>::find_cutoff_index(const candidates_t<i_t, f_t>& candidates,
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
    [best_cstr_delta, best_damage] __device__(auto tuple) -> bool {
      if (thrust::get<0>(tuple) == best_cstr_delta && thrust::get<1>(tuple) <= best_damage) {
        return true;
      }
      return false;
    });
  return out_iter - iterator;
}

template <typename i_t, typename f_t>
i_t bounds_repair_t<i_t, f_t>::get_random_idx(i_t size)
{
  std::uniform_int_distribution<> dist(0, size - 1);
  // Generate random number
  i_t random_number = dist(gen);
  return random_number;
}

// TODO convert this to var and test it.
template <typename i_t, typename f_t>
bool bounds_repair_t<i_t, f_t>::detect_cycle(i_t cstr_idx)
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
void bounds_repair_t<i_t, f_t>::apply_move(problem_t<i_t, f_t>& problem,
                                           problem_t<i_t, f_t>& original_problem,
                                           i_t move_idx)
{
  run_device_lambda(handle_ptr->get_stream(),
                    [move_idx,
                     candidates       = candidates.view(),
                     problem          = problem.view(),
                     original_problem = original_problem.view()] __device__() {
                      i_t var_idx     = candidates.variable_index[move_idx];
                      f_t shift_value = candidates.bound_shift[move_idx];
                      auto bounds     = problem.variable_bounds[var_idx];
                      DEVICE_LOG_TRACE(
                        "Applying move on var %d with shift %f lb %f ub %f o_lb %f o_ub %f \n",
                        var_idx,
                        shift_value,
                        get_lower(bounds),
                        get_upper(bounds),
                        get_lower(original_problem.variable_bounds[var_idx]),
                        get_upper(original_problem.variable_bounds[var_idx]));
                      if (problem.integer_equal(get_lower(bounds), get_upper(bounds))) {
                        *candidates.at_least_one_singleton_moved = 1;
                      }

                      get_lower(bounds) += shift_value;
                      get_upper(bounds) += shift_value;
                      problem.variable_bounds[var_idx] = bounds;
                      cuopt_assert(get_lower(original_problem.variable_bounds[var_idx]) <=
                                     get_lower(bounds) + problem.tolerances.integrality_tolerance,
                                   "");
                      cuopt_assert(get_upper(original_problem.variable_bounds[var_idx]) +
                                       problem.tolerances.integrality_tolerance >=
                                     get_upper(bounds),
                                   "");
                    });
}

template <typename i_t, typename f_t>
bool bounds_repair_t<i_t, f_t>::repair_problem(problem_t<i_t, f_t>& problem,
                                               problem_t<i_t, f_t>& original_problem,
                                               termination_checker_t& timer_,
                                               const raft::handle_t* handle_ptr_)
{
  CUOPT_LOG_DEBUG("Running bounds repair");
  handle_ptr = handle_ptr_;
  timer      = timer_;
  cuopt_assert(timer.deterministic == problem.deterministic,
               "Bounds repair timer/problem determinism mismatch");
  resize(problem);
  reset();
  best_violation = get_ii_violation(problem);
  curr_violation = best_violation;
  best_bounds.update_from(problem, handle_ptr);
  double total_estimated_work = 0.0;
  i_t repair_iterations       = 0;
  if (timer.deterministic) {
    const double setup_work = estimate_bounds_repair_setup_work(problem);
    record_estimated_work(timer, &total_estimated_work, setup_work);
    CUOPT_DETERMINISM_LOG(
      "Repair entry: pb_hash=0x%x bounds_hash=0x%x violated_hash=0x%x n_violated=%d "
      "best_violation=%.6f timer_rem=%.6f total_work=%.6f setup_work=%.6f",
      problem.get_fingerprint(),
      detail::compute_hash(make_span(problem.variable_bounds), handle_ptr->get_stream()),
      detail::compute_hash(make_span(violated_constraints, 0, h_n_violated_cstr),
                           handle_ptr->get_stream()),
      h_n_violated_cstr,
      best_violation,
      timer.remaining_time(),
      total_estimated_work,
      setup_work);
  }
  i_t no_candidate_in_a_row                = 0;
  [[maybe_unused]] const char* exit_reason = "FEASIBLE";
  // TODO: do this better
  i_t iter_limit = std::numeric_limits<i_t>::max();
  if (timer.deterministic) { iter_limit = 20; }
  while (h_n_violated_cstr > 0 && iter_limit-- > 0) {
    repair_iterations++;
    CUOPT_LOG_TRACE("Bounds repair loop: n_violated %d best_violation %f curr_violation %f",
                    h_n_violated_cstr,
                    best_violation,
                    curr_violation);
    if (timer.deterministic) {
      CUOPT_DETERMINISM_LOG(
        "Repair iter entry: iter=%d pb_hash=0x%x bounds_hash=0x%x violated_hash=0x%x "
        "n_violated=%d best_violation=%.6f curr_violation=%.6f timer_rem=%.6f total_work=%.6f",
        repair_iterations,
        problem.get_fingerprint(),
        detail::compute_hash(make_span(problem.variable_bounds), handle_ptr->get_stream()),
        detail::compute_hash(make_span(violated_constraints, 0, h_n_violated_cstr),
                             handle_ptr->get_stream()),
        h_n_violated_cstr,
        best_violation,
        curr_violation,
        timer.remaining_time(),
        total_estimated_work);
    }
    if (timer.check_time_limit()) {
      exit_reason = "TIME_LIMIT";
      break;
    }
    i_t curr_cstr = get_random_cstr();
    // best way would be to check a variable cycle, but this is easier and more performant
    bool is_cycle = detect_cycle(curr_cstr);
    if (is_cycle) { CUOPT_LOG_DEBUG("Repair: cycle detected at cstr %d", curr_cstr); }
    // in parallel compute the best shift and best respective damage
    i_t n_candidates  = compute_best_shift(problem, original_problem, curr_cstr);
    double shift_work = 0.0;
    if (timer.deterministic) {
      shift_work = estimate_bounds_repair_shift_work(problem, curr_cstr, n_candidates, is_cycle);
      record_estimated_work(timer, &total_estimated_work, shift_work);
      CUOPT_DETERMINISM_LOG(
        "Repair iter shift: iter=%d curr_cstr=%d cycle=%d n_candidates=%d cand_var_hash=0x%x "
        "cand_shift_hash=0x%x singleton_moved=%d shift_work=%.6f timer_rem=%.6f total_work=%.6f",
        repair_iterations,
        curr_cstr,
        (int)is_cycle,
        n_candidates,
        detail::compute_hash(make_span(candidates.variable_index, 0, n_candidates),
                             handle_ptr->get_stream()),
        detail::compute_hash(make_span(candidates.bound_shift, 0, n_candidates),
                             handle_ptr->get_stream()),
        (int)candidates.at_least_one_singleton_moved.value(handle_ptr->get_stream()),
        shift_work,
        timer.remaining_time(),
        total_estimated_work);
    }
    // if no candidate is there continue with another constraint
    if (n_candidates == 0) {
      CUOPT_LOG_DEBUG("Repair: no candidate var found for cstr %d", curr_cstr);
      if (no_candidate_in_a_row++ == 10 || h_n_violated_cstr == 1) {
        CUOPT_LOG_DEBUG("Repair: no candidate var found on last violated constraint %d. Exiting...",
                        curr_cstr);
        exit_reason = "NO_CANDIDATE";
        break;
      }
      continue;
    }
    no_candidate_in_a_row = 0;
    CUOPT_LOG_TRACE("Repair: number of candidates %d", n_candidates);
    // among the ones that have a valid shift value, compute the damage
    compute_damages(problem, n_candidates);
    // get the best damage
    i_t best_cstr_delta = candidates.cstr_delta.front_element(handle_ptr->get_stream());
    f_t best_damage     = candidates.damage.front_element(handle_ptr->get_stream());
    double damage_work  = 0.0;
    if (timer.deterministic) {
      damage_work = estimate_bounds_repair_damage_work(problem, n_candidates);
      record_estimated_work(timer, &total_estimated_work, damage_work);
      CUOPT_DETERMINISM_LOG(
        "Repair iter damage: iter=%d curr_cstr=%d cand_cdelta_hash=0x%x cand_damage_hash=0x%x "
        "best_cstr_delta=%d best_damage=%.6f damage_work=%.6f timer_rem=%.6f total_work=%.6f",
        repair_iterations,
        curr_cstr,
        detail::compute_hash(make_span(candidates.cstr_delta, 0, n_candidates),
                             handle_ptr->get_stream()),
        detail::compute_hash(make_span(candidates.damage, 0, n_candidates),
                             handle_ptr->get_stream()),
        best_cstr_delta,
        best_damage,
        damage_work,
        timer.remaining_time(),
        total_estimated_work);
    }
    CUOPT_LOG_TRACE(
      "Repair: best_cstr_delta value %d best_damage %f", best_cstr_delta, best_damage);
    i_t best_move_idx;
    i_t n_of_eligible_candidates = -1;

    const double rand_u01         = rand_double(0, 1, gen);
    const bool took_random_branch = (best_cstr_delta > 0 && rand_u01 < p) || is_cycle;
    if (took_random_branch) {
      best_move_idx = get_random_idx(n_candidates);
    } else {
      n_of_eligible_candidates =
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
    if (timer.deterministic) {
      CUOPT_DETERMINISM_LOG(
        "Repair iter select: iter=%d cycle=%d rand_u01=%.12f took_random=%d "
        "cutoff_idx=%d n_eligible=%d chosen_idx=%d chosen_var=%d chosen_shift=%.6f "
        "chosen_cdelta=%d chosen_damage=%.6f",
        repair_iterations,
        (int)is_cycle,
        rand_u01,
        (int)took_random_branch,
        (int)(took_random_branch ? -1 : n_of_eligible_candidates),
        (int)(took_random_branch ? n_candidates : n_of_eligible_candidates),
        best_move_idx,
        candidates.variable_index.element(best_move_idx, handle_ptr->get_stream()),
        candidates.bound_shift.element(best_move_idx, handle_ptr->get_stream()),
        candidates.cstr_delta.element(best_move_idx, handle_ptr->get_stream()),
        candidates.damage.element(best_move_idx, handle_ptr->get_stream()));
    }
    apply_move(problem, original_problem, best_move_idx);
    reset();
    // TODO we might optimize this to only calculate the changed constraints
    curr_violation                = get_ii_violation(problem);
    const bool improved_violation = curr_violation < best_violation;
    double refresh_work           = 0.0;
    if (timer.deterministic) {
      refresh_work = bounds_repair_move_base_work +
                     estimate_bounds_repair_violation_refresh_work(problem, improved_violation);
      record_estimated_work(timer, &total_estimated_work, refresh_work);
      CUOPT_DETERMINISM_LOG(
        "Repair iter post: iter=%d pb_hash=0x%x bounds_hash=0x%x violated_hash=0x%x "
        "n_violated=%d curr_violation=%.6f improved=%d refresh_work=%.6f total_work=%.6f "
        "timer_rem=%.6f",
        repair_iterations,
        problem.get_fingerprint(),
        detail::compute_hash(make_span(problem.variable_bounds), handle_ptr->get_stream()),
        detail::compute_hash(make_span(violated_constraints, 0, h_n_violated_cstr),
                             handle_ptr->get_stream()),
        h_n_violated_cstr,
        curr_violation,
        (int)improved_violation,
        refresh_work,
        total_estimated_work,
        timer.remaining_time());
      CUOPT_DETERMINISM_LOG(
        "Repair iter work: cstr=%d candidates=%d cycle=%d improved=%d total=%.6f",
        curr_cstr,
        n_candidates,
        (int)is_cycle,
        (int)improved_violation,
        total_estimated_work);
    }

    if (improved_violation) {
      best_violation = curr_violation;
      // update best bounds
      best_bounds.update_from(problem, handle_ptr);
    }
  }
  if (h_n_violated_cstr > 0 && iter_limit <= 0) { exit_reason = "ITER_LIMIT"; }
  bool feasible = h_n_violated_cstr == 0;
  best_bounds.update_to(problem, handle_ptr);
  CUOPT_LOG_DEBUG("Repair: returning with feas: %d vio %f", feasible, best_violation);
  if (timer.deterministic) {
    CUOPT_DETERMINISM_LOG(
      "Repair exit: reason=%s iters=%d feasible=%d n_violated=%d best_violation=%.6f "
      "total_work=%.6f timer_rem=%.6f",
      exit_reason,
      repair_iterations,
      (int)feasible,
      h_n_violated_cstr,
      best_violation,
      total_estimated_work,
      timer.remaining_time());
  }
  return feasible;
}

#if MIP_INSTANTIATE_FLOAT
template class bounds_repair_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class bounds_repair_t<int, double>;
#endif

};  // namespace cuopt::linear_programming::detail
