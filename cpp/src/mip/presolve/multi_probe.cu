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

#include <mip/mip_constants.hpp>

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <utilities/copy_helpers.hpp>
#include <utilities/device_utils.cuh>

#include <cub/cub.cuh>
#include "bounds_presolve_helpers.cuh"
#include "bounds_update_helpers.cuh"
#include "multi_probe.cuh"

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
  __device__ __forceinline__ thrust::tuple<i_t, i_t, i_t, i_t> operator()(
    thrust::tuple<f_t, f_t, f_t, f_t, f_t, f_t> t) const
  {
    auto min_act_0 = thrust::get<0>(t);
    auto max_act_0 = thrust::get<1>(t);
    auto min_act_1 = thrust::get<2>(t);
    auto max_act_1 = thrust::get<3>(t);
    auto cnst_ub   = thrust::get<4>(t);
    auto cnst_lb   = thrust::get<5>(t);
    f_t eps        = get_cstr_tolerance<i_t, f_t>(
      cnst_lb, cnst_ub, tolerances.absolute_tolerance, tolerances.relative_tolerance);
    auto infeas_0 = (min_act_0 > cnst_ub + eps) || (max_act_0 < cnst_lb - eps);
    auto redund_0 = (min_act_0 > cnst_lb + eps) && (max_act_0 < cnst_ub - eps);
    auto infeas_1 = (min_act_1 > cnst_ub + eps) || (max_act_1 < cnst_lb - eps);
    auto redund_1 = (min_act_1 > cnst_lb + eps) && (max_act_1 < cnst_ub - eps);
    return thrust::make_tuple(infeas_0, redund_0, infeas_1, redund_1);
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
multi_probe_t<i_t, f_t>::multi_probe_t(mip_solver_context_t<i_t, f_t>& context_,
                                       settings_t in_settings)
  : context(context_),
    upd_0(*context.problem_ptr),
    upd_1(*context.problem_ptr),
    settings(in_settings)
{
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::resize(problem_t<i_t, f_t>& problem)
{
  upd_0.resize(problem);
  upd_1.resize(problem);
  host_lb.resize(problem.n_variables);
  host_ub.resize(problem.n_variables);
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::calculate_activity(problem_t<i_t, f_t>& pb,
                                                 const raft::handle_t* handle_ptr)
{
  cuopt_assert(pb.n_variables == upd_0.lb.size(), "bounds array size inconsistent");
  cuopt_assert(pb.n_variables == upd_0.ub.size(), "bounds array size inconsistent");
  cuopt_assert(pb.n_variables == upd_1.lb.size(), "bounds array size inconsistent");
  cuopt_assert(pb.n_variables == upd_1.ub.size(), "bounds array size inconsistent");
  cuopt_assert(pb.n_constraints == upd_0.min_activity.size(), "activity array size inconsistent");
  cuopt_assert(pb.n_constraints == upd_1.min_activity.size(), "activity array size inconsistent");
  cuopt_assert(pb.n_constraints == upd_0.max_activity.size(), "activity array size inconsistent");
  cuopt_assert(pb.n_constraints == upd_1.max_activity.size(), "activity array size inconsistent");

  cuopt_assert(pb.n_constraints == pb.constraint_lower_bounds.size(),
               "activity array size inconsistent");

  // Calculate minimal activity for constraint i l_iS
  // where l_iS = sum(a_i_j*l_j)    a_i_j < 0
  //              + sum(a_i_j*u_j)    a_i_j > 0
  // here
  // a_i_j is the coefficient of constraint i wrt variable j
  // l_j is lower bound of variable j
  // u_j is upper bound of variable j

  if (skip_0 ^ skip_1) {
    auto& upd                = skip_0 ? upd_1 : upd_0;
    constexpr auto n_threads = 256;
    calc_activity_kernel<i_t, f_t, n_threads>
      <<<pb.n_constraints, n_threads, 0, handle_ptr->get_stream()>>>(pb.view(), upd.view());
  } else {
    constexpr auto n_threads = 256;
    calc_activity_kernel<i_t, f_t, n_threads>
      <<<pb.n_constraints, n_threads, 0, handle_ptr->get_stream()>>>(
        pb.view(), upd_0.view(), upd_1.view());
  }
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool multi_probe_t<i_t, f_t>::calculate_bounds_update(problem_t<i_t, f_t>& pb,
                                                      const raft::handle_t* handle_ptr)
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

  if (skip_0 && skip_1) {
    return false;
  } else if (skip_0) {
    upd_1.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    update_bounds_kernel<i_t, f_t, n_threads>
      <<<pb.n_variables, n_threads, 0, handle_ptr->get_stream()>>>(pb.view(), upd_1.view());
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
    i_t h_bounds_changed_1 = upd_1.bounds_changed.value(handle_ptr->get_stream());
    CUOPT_LOG_TRACE("Bounds changed upd 1 %d", h_bounds_changed_1);
    skip_1 = (h_bounds_changed_1 == zero);
  } else if (skip_1) {
    upd_0.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    update_bounds_kernel<i_t, f_t, n_threads>
      <<<pb.n_variables, n_threads, 0, handle_ptr->get_stream()>>>(pb.view(), upd_0.view());
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
    i_t h_bounds_changed_0 = upd_0.bounds_changed.value(handle_ptr->get_stream());
    CUOPT_LOG_TRACE("Bounds changed upd 0 %d", h_bounds_changed_0);
    skip_0 = (h_bounds_changed_0 == zero);
  } else {
    upd_0.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    upd_1.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    update_bounds_kernel<i_t, f_t, n_threads>
      <<<pb.n_variables, n_threads, 0, handle_ptr->get_stream()>>>(
        pb.view(), upd_0.view(), upd_1.view());
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
    i_t h_bounds_changed_0 = upd_0.bounds_changed.value(handle_ptr->get_stream());
    CUOPT_LOG_TRACE("Bounds changed upd 0 %d", h_bounds_changed_0);
    i_t h_bounds_changed_1 = upd_1.bounds_changed.value(handle_ptr->get_stream());
    CUOPT_LOG_TRACE("Bounds changed upd 1 %d", h_bounds_changed_1);

    skip_0 = (h_bounds_changed_0 == zero);
    skip_1 = (h_bounds_changed_1 == zero);
  }

  return (!skip_0 || !skip_1);
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::set_interval_bounds(
  const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
  problem_t<i_t, f_t>& pb,
  const raft::handle_t* handle_ptr)
{
  const i_t& probe_var                    = std::get<0>(var_interval_vals);
  const std::pair<f_t, f_t>& probe_vals_0 = std::get<1>(var_interval_vals);
  const std::pair<f_t, f_t>& probe_vals_1 = std::get<2>(var_interval_vals);
  run_device_lambda(handle_ptr->get_stream(),
                    [probe_var = probe_var,
                     lb_0      = probe_vals_0.first,
                     ub_0      = probe_vals_0.second,
                     lb_1      = probe_vals_1.first,
                     ub_1      = probe_vals_1.second,
                     upd_0_v   = upd_0.view(),
                     upd_1_v   = upd_1.view()] __device__() {
                      upd_0_v.lb[probe_var] = lb_0;
                      upd_0_v.ub[probe_var] = ub_0;
                      upd_1_v.lb[probe_var] = lb_1;
                      upd_1_v.ub[probe_var] = ub_1;
                    });
  // init changed constraints
  i_t var_offset_begin = pb.reverse_offsets.element(probe_var, handle_ptr->get_stream());
  i_t var_offset_end   = pb.reverse_offsets.element(probe_var + 1, handle_ptr->get_stream());
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_0.changed_constraints.begin(),
               upd_0.changed_constraints.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_1.changed_constraints.begin(),
               upd_1.changed_constraints.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_0.next_changed_constraints.begin(),
               upd_0.next_changed_constraints.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               upd_1.next_changed_constraints.begin(),
               upd_1.next_changed_constraints.end(),
               0);
  // set changed constraints from the vars
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   pb.reverse_constraints.begin() + var_offset_begin,
                   pb.reverse_constraints.begin() + var_offset_end,
                   [upd_0_v = upd_0.view(), upd_1_v = upd_1.view()] __device__(auto i) {
                     upd_0_v.changed_constraints[i] = 1;
                     upd_1_v.changed_constraints[i] = 1;
                   });
  init_changed_constraints = false;
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::set_bounds(
  const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
  const raft::handle_t* handle_ptr)
{
  const std::vector<i_t>& probe_vars   = std::get<0>(var_probe_vals);
  const std::vector<f_t>& probe_vals_0 = std::get<1>(var_probe_vals);
  const std::vector<f_t>& probe_vals_1 = std::get<2>(var_probe_vals);
  auto d_vars                          = device_copy(probe_vars, handle_ptr->get_stream());
  auto d_vals_0                        = device_copy(probe_vals_0, handle_ptr->get_stream());
  auto d_vals_1                        = device_copy(probe_vals_1, handle_ptr->get_stream());

  auto upd_0_v = upd_0.view();
  auto upd_1_v = upd_1.view();
  auto z_iter  = thrust::make_zip_iterator(
    thrust::make_tuple(d_vars.begin(), d_vals_0.begin(), d_vals_1.begin()));
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   z_iter,
                   z_iter + d_vars.size(),
                   [upd_0_v, upd_1_v] __device__(auto t) {
                     upd_0_v.lb[thrust::get<0>(t)] = thrust::get<1>(t);
                     upd_0_v.ub[thrust::get<0>(t)] = thrust::get<1>(t);
                     upd_1_v.lb[thrust::get<0>(t)] = thrust::get<2>(t);
                     upd_1_v.ub[thrust::get<0>(t)] = thrust::get<2>(t);
                   });
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t multi_probe_t<i_t, f_t>::bound_update_loop(problem_t<i_t, f_t>& pb,
                                                                   const raft::handle_t* handle_ptr,
                                                                   timer_t timer)
{
  termination_criterion_t criteria = termination_criterion_t::ITERATION_LIMIT;
  skip_0                           = false;
  skip_1                           = false;

  i_t iter_0 = 0;
  i_t iter_1 = 0;
  if (init_changed_constraints) {
    // all changed constraints are 1, next are zero
    upd_0.init_changed_constraints(handle_ptr);
    upd_1.init_changed_constraints(handle_ptr);
  } else {
    // reset for the next calls on the same object
    init_changed_constraints = true;
  }
  for (i_t iter = 0; iter < settings.iteration_limit; ++iter) {
    if (timer.check_time_limit()) {
      criteria = termination_criterion_t::TIME_LIMIT;
      break;
    }
    // calculate activity for both probes
    calculate_activity(pb, handle_ptr);
    if (!calculate_bounds_update(pb, handle_ptr)) {
      if (iter == 0) {
        criteria = termination_criterion_t::NO_UPDATE;
      } else {
        criteria = termination_criterion_t::CONVERGENCE;
      }
      break;
    }
    // next_changed are updated, fill current changed with zero and swap
    // swap next and current changed constraints
    if (!skip_0) { upd_0.prepare_for_next_iteration(handle_ptr); }
    if (!skip_1) { upd_1.prepare_for_next_iteration(handle_ptr); }
    iter_0 += !skip_0;
    iter_1 += !skip_1;
  }
  handle_ptr->sync_stream();
  if (compute_stats) { constraint_stats(pb, handle_ptr); }

  return criteria;
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::update_device_bounds(const raft::handle_t* handle_ptr)
{
  cuopt_assert(upd_0.lb.size() == host_lb.size(), "size of variable lower bound mismatch");
  raft::copy(upd_0.lb.data(), host_lb.data(), upd_0.lb.size(), handle_ptr->get_stream());
  cuopt_assert(upd_0.ub.size() == host_ub.size(), "size of variable upper bound mismatch");
  raft::copy(upd_0.ub.data(), host_ub.data(), upd_0.ub.size(), handle_ptr->get_stream());
  cuopt_assert(upd_1.lb.size() == host_lb.size(), "size of variable lower bound mismatch");
  raft::copy(upd_1.lb.data(), host_lb.data(), upd_1.lb.size(), handle_ptr->get_stream());
  cuopt_assert(upd_1.ub.size() == host_ub.size(), "size of variable upper bound mismatch");
  raft::copy(upd_1.ub.data(), host_ub.data(), upd_1.ub.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::update_host_bounds(
  const raft::handle_t* handle_ptr,
  const raft::device_span<typename type_2<f_t>::type> variable_bounds)
{
  cuopt_assert(variable_bounds.size() == host_lb.size(), "size of variable lower bound mismatch");
  cuopt_assert(variable_bounds.size() == host_ub.size(), "size of variable upper bound mismatch");

  rmm::device_uvector<f_t> var_lb(variable_bounds.size(), handle_ptr->get_stream());
  rmm::device_uvector<f_t> var_ub(variable_bounds.size(), handle_ptr->get_stream());
  thrust::transform(
    handle_ptr->get_thrust_policy(),
    variable_bounds.begin(),
    variable_bounds.end(),
    thrust::make_zip_iterator(thrust::make_tuple(var_lb.begin(), var_ub.begin())),
    [] __device__(auto i) { return thrust::make_tuple(get_lower(i), get_upper(i)); });
  raft::copy(host_lb.data(), var_lb.data(), var_lb.size(), handle_ptr->get_stream());
  raft::copy(host_ub.data(), var_ub.data(), var_ub.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::copy_problem_into_probing_buffers(problem_t<i_t, f_t>& pb,
                                                                const raft::handle_t* handle_ptr)
{
  cuopt_assert(upd_0.lb.size() == pb.variable_bounds.size(),
               "size of variable lower bound mismatch");
  cuopt_assert(upd_1.lb.size() == pb.variable_bounds.size(),
               "size of variable lower bound mismatch");
  cuopt_assert(upd_0.ub.size() == pb.variable_bounds.size(),
               "size of variable upper bound mismatch");
  cuopt_assert(upd_1.ub.size() == pb.variable_bounds.size(),
               "size of variable upper bound mismatch");

  thrust::transform(
    handle_ptr->get_thrust_policy(),
    pb.variable_bounds.begin(),
    pb.variable_bounds.end(),
    thrust::make_zip_iterator(
      thrust::make_tuple(upd_0.lb.begin(), upd_0.ub.begin(), upd_1.lb.begin(), upd_1.ub.begin())),
    [] __device__(auto i) {
      return thrust::make_tuple(get_lower(i), get_upper(i), get_lower(i), get_upper(i));
    });
}

template <typename i_t, typename f_t>
termination_criterion_t multi_probe_t<i_t, f_t>::solve_for_interval(
  problem_t<i_t, f_t>& pb,
  const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
  const raft::handle_t* handle_ptr)
{
  timer_t timer(settings.time_limit);

  copy_problem_into_probing_buffers(pb, handle_ptr);
  set_interval_bounds(var_interval_vals, pb, handle_ptr);

  return bound_update_loop(pb, handle_ptr, timer);
}

template <typename i_t, typename f_t>
termination_criterion_t multi_probe_t<i_t, f_t>::solve(
  problem_t<i_t, f_t>& pb,
  const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
  bool use_host_bounds)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb.handle_ptr;
  if (use_host_bounds) {
    update_device_bounds(handle_ptr);
  } else {
    copy_problem_into_probing_buffers(pb, handle_ptr);
  }
  set_bounds(var_probe_vals, handle_ptr);

  return bound_update_loop(pb, handle_ptr, timer);
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::set_updated_bounds(const raft::handle_t* handle_ptr,
                                                 raft::device_span<f_t> output_lb,
                                                 raft::device_span<f_t> output_ub,
                                                 i_t select_update)
{
  auto& lb = select_update ? upd_1.lb : upd_0.lb;
  auto& ub = select_update ? upd_1.ub : upd_0.ub;

  cuopt_assert(ub.size() == output_ub.size(), "size of variable upper bound mismatch");
  cuopt_assert(lb.size() == output_lb.size(), "size of variable lower bound mismatch");
  raft::copy(output_ub.data(), ub.data(), ub.size(), handle_ptr->get_stream());
  raft::copy(output_lb.data(), lb.data(), lb.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::set_updated_bounds(
  const raft::handle_t* handle_ptr,
  raft::device_span<typename type_2<f_t>::type> output_bounds,
  i_t select_update)
{
  auto& lb = select_update ? upd_1.lb : upd_0.lb;
  auto& ub = select_update ? upd_1.ub : upd_0.ub;

  cuopt_assert(ub.size() == output_bounds.size(), "size of variable upper bound mismatch");
  cuopt_assert(lb.size() == output_bounds.size(), "size of variable lower bound mismatch");
  thrust::transform(handle_ptr->get_thrust_policy(),
                    thrust::make_zip_iterator(thrust::make_tuple(lb.begin(), ub.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(lb.end(), ub.end())),
                    output_bounds.begin(),
                    [] __device__(auto i) {
                      return typename type_2<f_t>::type{thrust::get<0>(i), thrust::get<1>(i)};
                    });
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::constraint_stats(problem_t<i_t, f_t>& pb,
                                               const raft::handle_t* handle_ptr)
{
  auto detect_iter = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(upd_0.min_activity.begin(),
                                                 upd_0.max_activity.begin(),
                                                 upd_1.min_activity.begin(),
                                                 upd_1.max_activity.begin(),
                                                 pb.constraint_upper_bounds.begin(),
                                                 pb.constraint_lower_bounds.begin())),
    detect_infeas_redun_t<i_t, f_t>(pb.tolerances));

  thrust::tie(infeas_constraints_count_0,
              redund_constraints_count_0,
              infeas_constraints_count_1,
              redund_constraints_count_1) =
    thrust::reduce(handle_ptr->get_thrust_policy(),
                   detect_iter,
                   detect_iter + pb.n_constraints,
                   thrust::make_tuple<i_t, i_t, i_t, i_t>(0, 0, 0, 0),
                   tuple_plus_t<i_t>{});

  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  if (redund_constraints_count_0 > 0) {
    CUOPT_LOG_TRACE("First probe: Redundant constraint count %d", redund_constraints_count_0);
  }
  if (infeas_constraints_count_0 > 0) {
    CUOPT_LOG_TRACE("First probe: Infeasible constraint count %d", infeas_constraints_count_0);
  }
  if (redund_constraints_count_1 > 0) {
    CUOPT_LOG_TRACE("Second probe: Redundant constraint count %d", redund_constraints_count_1);
  }
  if (infeas_constraints_count_1 > 0) {
    CUOPT_LOG_TRACE("Second probe: Infeasible constraint count %d", infeas_constraints_count_1);
  }
}

template <typename i_t, typename f_t>
void multi_probe_t<i_t, f_t>::set_updated_bounds(problem_t<i_t, f_t>& pb,
                                                 i_t select_update,
                                                 const raft::handle_t* handle_ptr)
{
  set_updated_bounds(handle_ptr, make_span(pb.variable_bounds), select_update);
}

#if MIP_INSTANTIATE_FLOAT
template class multi_probe_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class multi_probe_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
