/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <mip/mip_constants.hpp>

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <utilities/copy_helpers.hpp>
#include <utilities/device_utils.cuh>

#include <cub/cub.cuh>
#include "bounds_presolve.cuh"
#include "bounds_presolve_helpers.cuh"
#include "bounds_update_helpers.cuh"

namespace cuopt::linear_programming::detail {

// Tobias Achterberg, Robert E. Bixby, Zonghao Gu, Edward Rothberg, Dieter Weninger (2019) Presolve
// Reductions in Mixed Integer Programming. INFORMS Journal on Computing 32(2):473-506.
// https://doi.org/10.1287/ijoc.2018.0857

// This code follows the paper mentioned above, section 3.2
// The solve function runs for a set number of iterations or until the expiry
// of the time limit.
// In each iteration, the minimal activity of all the constraints are calculated
// In infeasbility is not found, then a variable is selected and its bounds are
// updated. This update will invalidate minimal activity which is recalculated
// in the next iteration.
// If no updates to the bounds are detected then the loop is broken and the new
// bounds (if found) are applied to the problem.

template <typename i_t, typename f_t>
struct detect_infeas_redun_t {
  __device__ __forceinline__ thrust::tuple<i_t, i_t> operator()(
    thrust::tuple<f_t, f_t, f_t, f_t> t) const
  {
    auto min_act = thrust::get<0>(t);
    auto max_act = thrust::get<1>(t);
    auto cnst_lb = thrust::get<2>(t);
    auto cnst_ub = thrust::get<3>(t);
    auto infeas  = check_infeasibility<i_t, f_t>(min_act,
                                                max_act,
                                                cnst_lb,
                                                cnst_ub,
                                                tolerances.absolute_tolerance,
                                                tolerances.relative_tolerance);
    auto redund  = check_redundancy<i_t, f_t>(min_act,
                                             max_act,
                                             cnst_lb,
                                             cnst_ub,
                                             tolerances.absolute_tolerance,
                                             tolerances.relative_tolerance);
    return thrust::make_tuple(infeas, redund);
  }

 public:
  detect_infeas_redun_t()                                       = delete;
  detect_infeas_redun_t(const detect_infeas_redun_t<i_t, f_t>&) = default;
  detect_infeas_redun_t(const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tols)
    : tolerances(tols)
  {
  }

 private:
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
};

template <typename i_t, typename f_t>
bound_presolve_t<i_t, f_t>::bound_presolve_t(mip_solver_context_t<i_t, f_t>& context_,
                                             settings_t in_settings)
  : context(context_), upd(*context.problem_ptr), settings(in_settings)
{
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::resize(problem_t<i_t, f_t>& problem)
{
  upd.resize(problem);
  host_lb.resize(problem.n_variables);
  host_ub.resize(problem.n_variables);
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::calculate_activity(problem_t<i_t, f_t>& pb)
{
  cuopt_assert(pb.n_variables == upd.lb.size(), "bounds array size inconsistent");
  cuopt_assert(pb.n_variables == upd.ub.size(), "bounds array size inconsistent");
  cuopt_assert(pb.n_constraints == upd.min_activity.size(), "activity array size inconsistent");
  cuopt_assert(pb.n_constraints == upd.max_activity.size(), "activity array size inconsistent");

  constexpr auto n_threads = 256;
  calc_activity_kernel<i_t, f_t, n_threads>
    <<<pb.n_constraints, n_threads, 0, pb.handle_ptr->get_stream()>>>(pb.view(), upd.view());
}

template <typename i_t, typename f_t>
bool bound_presolve_t<i_t, f_t>::calculate_bounds_update(problem_t<i_t, f_t>& pb)
{
  // update lower bound for variable k : l_k = max(l_k, (b_i - l_iS)/a_i_k
  // where l_iS = sum(a_i_j*l_j)    a_i_j < 0, j != k
  //              + sum(a_i_j*u_j)    a_i_j > 0, j != k
  // here
  // a_i_j is the coefficient of constraint i wrt variable j
  // l_j is lower bound of variable j
  // u_j is upper bound of variable j
  // b_i is constraint upper bound

  constexpr i_t zero       = 0;
  constexpr auto n_threads = 256;

  upd.bounds_changed.set_value_async(zero, pb.handle_ptr->get_stream());
  update_bounds_kernel<i_t, f_t, n_threads>
    <<<pb.n_variables, n_threads, 0, pb.handle_ptr->get_stream()>>>(pb.view(), upd.view());
  RAFT_CHECK_CUDA(pb.handle_ptr->get_stream());
  i_t h_bounds_changed = upd.bounds_changed.value(pb.handle_ptr->get_stream());
  return h_bounds_changed != zero;
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::update_device_bounds(const raft::handle_t* handle_ptr)
{
  cuopt_assert(upd.lb.size() == host_lb.size(), "size of variable lower bound mismatch");
  raft::copy(upd.lb.data(), host_lb.data(), upd.lb.size(), handle_ptr->get_stream());
  cuopt_assert(upd.ub.size() == host_ub.size(), "size of variable upper bound mismatch");
  raft::copy(upd.ub.data(), host_ub.data(), upd.ub.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::update_host_bounds(const raft::handle_t* handle_ptr,
                                                    const raft::device_span<f_t> variable_lb,
                                                    const raft::device_span<f_t> variable_ub)
{
  cuopt_assert(variable_lb.size() == host_lb.size(), "size of variable lower bound mismatch");
  raft::copy(host_lb.data(), variable_lb.data(), variable_lb.size(), handle_ptr->get_stream());
  cuopt_assert(variable_ub.size() == host_ub.size(), "size of variable upper bound mismatch");
  raft::copy(host_ub.data(), variable_ub.data(), variable_ub.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::set_bounds(
  raft::device_span<f_t> var_lb,
  raft::device_span<f_t> var_ub,
  const std::vector<thrust::pair<i_t, f_t>>& var_probe_vals,
  const raft::handle_t* handle_ptr)
{
  auto d_var_probe_vals = device_copy(var_probe_vals, handle_ptr->get_stream());

  thrust::for_each(handle_ptr->get_thrust_policy(),
                   d_var_probe_vals.begin(),
                   d_var_probe_vals.end(),
                   [var_lb, var_ub] __device__(auto pair) {
                     var_lb[pair.first] = pair.second;
                     var_ub[pair.first] = pair.second;
                   });
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t bound_presolve_t<i_t, f_t>::bound_update_loop(problem_t<i_t, f_t>& pb,
                                                                      timer_t timer)
{
  termination_criterion_t criteria = termination_criterion_t::ITERATION_LIMIT;

  i_t iter;
  upd.init_changed_constraints(pb.handle_ptr);
  for (iter = 0; iter < settings.iteration_limit; ++iter) {
    calculate_activity(pb);
    if (timer.check_time_limit()) {
      criteria = termination_criterion_t::TIME_LIMIT;
      CUOPT_LOG_TRACE("Exiting bounds prop because of time limit at iter %d", iter);
      break;
    }
    if (!calculate_bounds_update(pb)) {
      if (iter == 0) {
        criteria = termination_criterion_t::NO_UPDATE;
      } else {
        criteria = termination_criterion_t::CONVERGENCE;
      }
      break;
    }
    upd.prepare_for_next_iteration(pb.handle_ptr);
  }
  pb.handle_ptr->sync_stream();
  calculate_infeasible_redundant_constraints(pb);
  solve_iter = iter;

  return criteria;
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::calculate_activity_on_problem_bounds(problem_t<i_t, f_t>& pb)
{
  auto& handle_ptr = pb.handle_ptr;
  upd.init_changed_constraints(handle_ptr);
  cuopt_assert(upd.lb.size() == pb.variable_lower_bounds.size(),
               "size of variable lower bound mismatch");
  raft::copy(
    upd.lb.data(), pb.variable_lower_bounds.data(), upd.lb.size(), handle_ptr->get_stream());
  cuopt_assert(upd.ub.size() == pb.variable_upper_bounds.size(),
               "size of variable upper bound mismatch");
  raft::copy(
    upd.ub.data(), pb.variable_upper_bounds.data(), upd.ub.size(), handle_ptr->get_stream());
  calculate_activity(pb);
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::copy_input_bounds(problem_t<i_t, f_t>& pb)
{
  auto& handle_ptr = pb.handle_ptr;

  cuopt_assert(upd.lb.size() == pb.variable_lower_bounds.size(),
               "size of variable lower bound mismatch");
  raft::copy(
    upd.lb.data(), pb.variable_lower_bounds.data(), upd.lb.size(), handle_ptr->get_stream());
  cuopt_assert(upd.ub.size() == pb.variable_upper_bounds.size(),
               "size of variable upper bound mismatch");
  raft::copy(
    upd.ub.data(), pb.variable_upper_bounds.data(), upd.ub.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t bound_presolve_t<i_t, f_t>::solve(problem_t<i_t, f_t>& pb,
                                                          f_t var_lb,
                                                          f_t var_ub,
                                                          i_t var_idx)
{
  auto& handle_ptr = pb.handle_ptr;
  timer_t timer(settings.time_limit);
  copy_input_bounds(pb);
  upd.lb.set_element_async(var_idx, var_lb, handle_ptr->get_stream());
  upd.ub.set_element_async(var_idx, var_ub, handle_ptr->get_stream());
  return bound_update_loop(pb, timer);
}

template <typename i_t, typename f_t>
termination_criterion_t bound_presolve_t<i_t, f_t>::solve(
  problem_t<i_t, f_t>& pb,
  const std::vector<thrust::pair<i_t, f_t>>& var_probe_val_pairs,
  bool use_host_bounds)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb.handle_ptr;
  if (use_host_bounds) {
    update_device_bounds(handle_ptr);
  } else {
    copy_input_bounds(pb);
  }
  set_bounds(make_span(upd.lb), make_span(upd.ub), var_probe_val_pairs, handle_ptr);

  return bound_update_loop(pb, timer);
}

template <typename i_t, typename f_t>
termination_criterion_t bound_presolve_t<i_t, f_t>::solve(problem_t<i_t, f_t>& pb,
                                                          raft::device_span<f_t> input_lb,
                                                          raft::device_span<f_t> input_ub)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb.handle_ptr;
  if (input_lb.size() == 0) {
    cuopt_assert(upd.lb.size() == pb.variable_lower_bounds.size(),
                 "size of variable lower bound mismatch");
    raft::copy(
      upd.lb.data(), pb.variable_lower_bounds.data(), upd.lb.size(), handle_ptr->get_stream());
  } else {
    cuopt_assert(input_lb.size() == upd.lb.size(), "size of variable lower bound mismatch");
    raft::copy(upd.lb.data(), input_lb.data(), input_lb.size(), handle_ptr->get_stream());
  }
  if (input_ub.size() == 0) {
    cuopt_assert(upd.ub.size() == pb.variable_upper_bounds.size(),
                 "size of variable upper bound mismatch");
    raft::copy(
      upd.ub.data(), pb.variable_upper_bounds.data(), upd.ub.size(), handle_ptr->get_stream());
  } else {
    cuopt_assert(input_ub.size() == upd.ub.size(), "size of variable lower bound mismatch");
    raft::copy(upd.ub.data(), input_ub.data(), upd.ub.size(), handle_ptr->get_stream());
  }
  return bound_update_loop(pb, timer);
}

template <typename i_t, typename f_t>
bool bound_presolve_t<i_t, f_t>::calculate_infeasible_redundant_constraints(problem_t<i_t, f_t>& pb)
{
  auto detect_iter = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(upd.min_activity.begin(),
                                                 upd.max_activity.begin(),
                                                 pb.constraint_lower_bounds.begin(),
                                                 pb.constraint_upper_bounds.begin())),
    detect_infeas_redun_t<i_t, f_t>{pb.tolerances});

  thrust::tie(infeas_constraints_count, redund_constraints_count) =
    thrust::reduce(pb.handle_ptr->get_thrust_policy(),
                   detect_iter,
                   detect_iter + pb.n_constraints,
                   thrust::make_tuple<i_t, i_t>(0, 0),
                   tuple_plus_t<i_t>{});

  RAFT_CHECK_CUDA(pb.handle_ptr->get_stream());

  if (redund_constraints_count > 0) {
    CUOPT_LOG_TRACE("Redundant constraint count %d", redund_constraints_count);
  }
  if (infeas_constraints_count > 0) {
    CUOPT_LOG_TRACE("Infeasible constraint count %d", infeas_constraints_count);
  }
  return (infeas_constraints_count == 0);
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::set_updated_bounds(problem_t<i_t, f_t>& pb)
{
  set_updated_bounds(pb.handle_ptr,
                     cuopt::make_span(pb.variable_lower_bounds),
                     cuopt::make_span(pb.variable_upper_bounds));
  pb.compute_n_integer_vars();
  pb.compute_binary_var_table();
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::set_updated_bounds(const raft::handle_t* handle_ptr,
                                                    raft::device_span<f_t> output_lb,
                                                    raft::device_span<f_t> output_ub)
{
  cuopt_assert(upd.ub.size() == output_ub.size(), "size of variable upper bound mismatch");
  cuopt_assert(upd.lb.size() == output_lb.size(), "size of variable lower bound mismatch");
  raft::copy(output_lb.data(), upd.lb.data(), upd.lb.size(), handle_ptr->get_stream());
  raft::copy(output_ub.data(), upd.ub.data(), upd.ub.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void bound_presolve_t<i_t, f_t>::calc_and_set_updated_constraint_bounds(problem_t<i_t, f_t>& pb)
{
  calculate_activity_on_problem_bounds(pb);

  thrust::for_each(pb.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(pb.n_constraints),
                   [pb      = pb.view(),
                    min_act = make_span(upd.min_activity),
                    max_act = make_span(upd.max_activity),
                    cnst_lb = make_span(pb.constraint_lower_bounds),
                    cnst_ub = make_span(pb.constraint_upper_bounds)] __device__(i_t idx) {
                     auto min_a    = min_act[idx];
                     auto max_a    = max_act[idx];
                     auto c_lb     = cnst_lb[idx];
                     auto c_ub     = cnst_ub[idx];
                     auto new_c_lb = max(c_lb, min_a);
                     auto new_c_ub = min(c_ub, max_a);
                     i_t infeas    = check_infeasibility<i_t, f_t>(
                       min_a, max_a, new_c_lb, new_c_ub, pb.tolerances.presolve_absolute_tolerance);
                     if (!infeas && (new_c_lb > new_c_ub)) {
                       new_c_lb = (new_c_lb + new_c_ub) / 2;
                       new_c_ub = new_c_lb;
                     }
                     cnst_lb[idx] = new_c_lb;
                     cnst_ub[idx] = new_c_ub;
                   });
}

#if MIP_INSTANTIATE_FLOAT
template class bound_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class bound_presolve_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
