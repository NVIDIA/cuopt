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
#include "lb_bounds_presolve.cuh"
#include "lb_kernels/lb_bounds_update_kernels.cuh"
#include "lb_kernels/lb_constraint_slack_kernels.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_bound_presolve_t<i_t, f_t>::lb_bound_presolve_t(mip_solver_context_t<i_t, f_t>& context_,
                                                   lb_problem_t<i_t, f_t>& problem,
                                                   settings_t in_settings)
  : context(context_), upd(problem), settings(in_settings)
{
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::resize(lb_problem_t<i_t, f_t>& problem)
{
  upd.resize(problem);
  host_bounds.resize(problem.n_variables);
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::calculate_constraint_slack_iter(
  lb_problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr)
{
  auto num_blocks = problem.cnst_csr.sub_warp_block_count + problem.cnst_csr.med_block_count +
                    problem.cnst_csr.num_blocks_heavy;
  // std::cout << "call_cnst_slack sub_warp_block_count " << problem.cnst_csr.sub_warp_block_count
  //           << "\n";
  // std::cout << "call_cnst_slack med_block_count " << problem.cnst_csr.med_block_count << "\n";
  // std::cout << "call_cnst_slack num_blocks_heavy " << problem.cnst_csr.num_blocks_heavy << "\n";

  // std::cout << "call_cnst_slack sub_warp+med "
  //           << problem.cnst_csr.sub_warp_block_count + problem.cnst_csr.med_block_count << "\n";

  // std::cout << "num_heavy_items " << problem.n_constraints - problem.cnst_csr.heavy_beg_id <<
  // "\n";
  constexpr bool erase_inf_cnst = false;
  nvtxRangePush("lb_multi_act");
  call_cnst_slack<erase_inf_cnst, i_t, f_t, 512>
    <<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(problem.cnst_csr.view(), upd.view());
  if (problem.cnst_csr.num_blocks_heavy != 0) {
    auto num_heavy_items = problem.n_constraints - problem.cnst_csr.heavy_beg_id;
    finalize_cnst_heavy<erase_inf_cnst, i_t, f_t, 32>
      <<<num_heavy_items, 32, 0, handle_ptr->get_stream()>>>(problem.cnst_csr.view(), upd.view());
  }
  nvtxRangePop();
}

template <typename i_t, typename f_t>
bool lb_bound_presolve_t<i_t, f_t>::calculate_bounds_update(lb_problem_t<i_t, f_t>& problem,
                                                            const raft::handle_t* handle_ptr)
{
  constexpr i_t zero = 0;
  upd.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
  auto num_blocks = problem.vars_csr.sub_warp_block_count + problem.vars_csr.med_block_count +
                    problem.vars_csr.num_blocks_heavy;
  // std::cout << "call_bnd_update sub_warp_block_count " << problem.vars_csr.sub_warp_block_count
  //           << "\n";
  // std::cout << "call_bnd_update med_block_count " << problem.vars_csr.med_block_count << "\n";
  // std::cout << "call_bnd_update num_blocks_heavy " << problem.vars_csr.num_blocks_heavy << "\n";

  // std::cout << "call_bnd_update sub_warp+med "
  //           << problem.vars_csr.sub_warp_block_count + problem.vars_csr.med_block_count << "\n";

  // std::cout << "num_heavy_items " << problem.n_variables - problem.vars_csr.heavy_beg_id << "\n";
  nvtxRangePush("lb_multi_bnd");
  call_bnd_update<i_t, f_t, 512>
    <<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(problem.vars_csr.view(), upd.view());
  if (problem.vars_csr.num_blocks_heavy != 0) {
    bnd_heavy_update_next_changed_constraints<i_t, f_t, 512>
      <<<problem.vars_csr.num_blocks_heavy, 512, 0, handle_ptr->get_stream()>>>(
        problem.vars_csr.view(), upd.view());
  }
  nvtxRangePop();
  i_t h_bounds_changed = upd.bounds_changed.value(handle_ptr->get_stream());

  return (h_bounds_changed != 0);
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::update_device_bounds(const raft::handle_t* handle_ptr)
{
  cuopt_assert(upd.vars_bnd.size() == host_bounds.size(), "size of variable bound mismatch");
  raft::copy(
    upd.vars_bnd.data(), host_bounds.data(), upd.vars_bnd.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::update_host_bounds(
  const raft::handle_t* handle_ptr,
  const raft::device_span<typename type_2<f_t>::type> variable_bounds)
{
  cuopt_assert(variable_bounds.size() == host_bounds.size(), "size of variable bound mismatch");
  raft::copy(
    host_bounds.data(), variable_bounds.data(), variable_bounds.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::set_bounds(
  raft::device_span<typename type_2<f_t>::type> variable_bounds,
  const std::vector<thrust::pair<i_t, f_t>>& var_probe_vals,
  const raft::handle_t* handle_ptr)
{
  using f_t2            = typename type_2<f_t>::type;
  auto d_var_probe_vals = device_copy(var_probe_vals, handle_ptr->get_stream());

  thrust::for_each(handle_ptr->get_thrust_policy(),
                   d_var_probe_vals.begin(),
                   d_var_probe_vals.end(),
                   [vars_bnd = variable_bounds] __device__(auto t) {
                     auto idx      = thrust::get<0>(t);
                     auto upd_val  = thrust::get<1>(t);
                     vars_bnd[idx] = f_t2{upd_val, upd_val};
                   });
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t lb_bound_presolve_t<i_t, f_t>::bound_update_loop(
  lb_problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr, timer_t timer)
{
  termination_criterion_t criteria = termination_criterion_t::ITERATION_LIMIT;

  i_t iter;
  upd.init_changed_constraints(handle_ptr);
  for (iter = 0; iter < settings.iteration_limit; ++iter) {
    calculate_constraint_slack_iter(pb, handle_ptr);
    if (timer.check_time_limit()) {
      criteria = termination_criterion_t::TIME_LIMIT;
      CUOPT_LOG_TRACE("Exiting bounds prop because of time limit at iter %d", iter);
      break;
    }
    if (!calculate_bounds_update(pb, handle_ptr)) {
      if (iter == 0) {
        criteria = termination_criterion_t::NO_UPDATE;
      } else {
        criteria = termination_criterion_t::CONVERGENCE;
      }
      break;
    }
    upd.prepare_for_next_iteration(handle_ptr);
  }
  handle_ptr->sync_stream();
  calculate_infeasible_redundant_constraints(pb, handle_ptr);
  solve_iter = iter;

  return criteria;
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::copy_input_bounds(lb_problem_t<i_t, f_t>& pb,
                                                      const raft::handle_t* handle_ptr)
{
  cuopt_assert(upd.vars_bnd.size() == pb.vars_bnd.size(), "size of variable bound mismatch");
  raft::copy(
    upd.vars_bnd.data(), pb.vars_bnd.data(), upd.vars_bnd.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t lb_bound_presolve_t<i_t, f_t>::solve(lb_problem_t<i_t, f_t>& pb,
                                                             f_t var_lb,
                                                             f_t var_ub,
                                                             i_t var_idx)
{
  auto& handle_ptr = pb.handle_ptr;
  timer_t timer(settings.time_limit);
  copy_input_bounds(pb, handle_ptr);
  using f_t2   = typename type_2<f_t>::type;
  f_t2 var_val = f_t2{var_lb, var_ub};
  upd.vars_bnd.set_element_async(var_idx, var_val, handle_ptr->get_stream());
  return bound_update_loop(pb, handle_ptr, timer);
}

template <typename i_t, typename f_t>
termination_criterion_t lb_bound_presolve_t<i_t, f_t>::solve(
  lb_problem_t<i_t, f_t>& pb,
  const std::vector<thrust::pair<i_t, f_t>>& var_probe_val_pairs,
  bool use_host_bounds)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb.handle_ptr;
  if (use_host_bounds) {
    update_device_bounds(handle_ptr);
  } else {
    copy_input_bounds(pb, handle_ptr);
  }
  set_bounds(make_span(upd.vars_bnd), var_probe_val_pairs, handle_ptr);

  return bound_update_loop(pb, handle_ptr, timer);
}

template <typename i_t, typename f_t>
termination_criterion_t lb_bound_presolve_t<i_t, f_t>::solve(
  lb_problem_t<i_t, f_t>& pb, raft::device_span<typename type_2<f_t>::type> input_bounds)
{
  timer_t timer(settings.time_limit);
  auto& handle_ptr = pb.handle_ptr;
  if (input_bounds.size() == 0) {
    cuopt_assert(upd.vars_bnd.size() == pb.vars_bnd.size(), "size of variable bound mismatch");
    raft::copy(
      upd.vars_bnd.data(), pb.vars_bnd.data(), upd.vars_bnd.size(), handle_ptr->get_stream());
  } else {
    cuopt_assert(input_bounds.size() == upd.vars_bnd.size(), "size of variable bound mismatch");
    raft::copy(
      upd.vars_bnd.data(), input_bounds.data(), input_bounds.size(), handle_ptr->get_stream());
  }
  return bound_update_loop(pb, handle_ptr, timer);
}

template <typename i_t, typename f_t, typename f_t2>
struct detect_infeas_t {
  __device__ __forceinline__ i_t operator()(thrust::tuple<f_t2, f_t, f_t> t) const
  {
    auto cnst_slack = thrust::get<0>(t);
    auto cnst_ub    = thrust::get<1>(t);
    auto cnst_lb    = thrust::get<2>(t);
    f_t eps         = get_cstr_tolerance<i_t, f_t>(
      cnst_lb, cnst_ub, tolerances.absolute_tolerance, tolerances.relative_tolerance);
    auto infeas = (0 > cnst_slack.x + eps) || (0 < cnst_slack.y - eps);
    return infeas;
  }

 public:
  detect_infeas_t()                                       = delete;
  detect_infeas_t(const detect_infeas_t<i_t, f_t, f_t2>&) = default;
  detect_infeas_t(const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tols)
    : tolerances(tols)
  {
  }

 private:
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
};

template <typename i_t, typename f_t>
bool lb_bound_presolve_t<i_t, f_t>::calculate_infeasible_redundant_constraints(
  lb_problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr)
{
  using f_t2          = typename type_2<f_t>::type;
  auto* orig_prob_ptr = pb.pb;
  auto upd_cnst_slack = upd.view().cnst_slack;
  auto detect_iter    = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(upd_cnst_slack.begin(),
                                                 orig_prob_ptr->constraint_lower_bounds.begin(),
                                                 orig_prob_ptr->constraint_upper_bounds.begin())),
    detect_infeas_t<i_t, f_t, f_t2>{orig_prob_ptr->tolerances});

  infeas_constraints_count =
    thrust::reduce(handle_ptr->get_thrust_policy(), detect_iter, detect_iter + pb.n_constraints);

  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  if (infeas_constraints_count > 0) {
    CUOPT_LOG_TRACE("Infeasible constraint count %d", infeas_constraints_count);
  }
  return (infeas_constraints_count == 0);
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::set_updated_bounds(lb_problem_t<i_t, f_t>& pb)
{
  set_updated_bounds(pb.handle_ptr, cuopt::make_span(pb.vars_bnd));
  // TODO?
  // auto* orig_prob_ptr = pb.pb;
  // orig_prob_ptr->compute_n_integer_vars();
  // orig_prob_ptr->compute_binary_var_table();
}

template <typename i_t, typename f_t>
void lb_bound_presolve_t<i_t, f_t>::set_updated_bounds(
  const raft::handle_t* handle_ptr, raft::device_span<typename type_2<f_t>::type> output_bounds)
{
  cuopt_assert(upd.vars_bnd.size() == output_bounds.size(),
               "size of variable upper bound mismatch");
  raft::copy(
    output_bounds.data(), upd.vars_bnd.data(), upd.vars_bnd.size(), handle_ptr->get_stream());
}

#if MIP_INSTANTIATE_FLOAT
template class lb_bound_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_bound_presolve_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
