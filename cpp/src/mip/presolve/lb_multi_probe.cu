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
#include <nvtx3/nvtx3.hpp>

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <utilities/copy_helpers.hpp>
#include <utilities/device_utils.cuh>

#include <cub/cub.cuh>
#include "lb_kernels/lb_multi_probe_bounds_update_kernels.cuh"
#include "lb_kernels/lb_multi_probe_constraint_kernels.cuh"
#include "lb_multi_probe.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t>
i_t get_id(const std::vector<i_t>& bin_offsets, i_t degree_beg)
{
  return bin_offsets[ceil_log_2(degree_beg)];
}

template <typename i_t>
std::pair<i_t, i_t> get_id_range(const std::vector<i_t>& bin_offsets,
                                 i_t degree_beg,
                                 i_t degree_end)
{
  return std::make_pair(bin_offsets[ceil_log_2(degree_beg)],
                        bin_offsets[ceil_log_2(degree_end) + 1]);
}

template <typename i_t, typename f_t>
lb_multi_probe_t<i_t, f_t>::lb_multi_probe_t(mip_solver_context_t<i_t, f_t>& context_,
                                             lb_problem_t<i_t, f_t>& problem,
                                             settings_t in_settings)
  : context(context_), upd_0(problem), upd_1(problem), settings(in_settings)
{
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::resize(lb_problem_t<i_t, f_t>& problem)
{
  upd_0.resize(problem);
  upd_1.resize(problem);
  host_bounds.resize(problem.n_variables);
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::copy_problem_into_probing_buffers(
  lb_problem_t<i_t, f_t>& lb_problem, const raft::handle_t* handle_ptr)
{
  upd_0.copy(lb_problem);
  upd_1.copy(lb_problem);
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::calculate_constraint_slack_iter(lb_problem_t<i_t, f_t>& problem,
                                                                 const raft::handle_t* handle_ptr)
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
  nvtxRangePush("lb_multi_act");
  constexpr bool erase_inf_cnst = false;
  if (skip_0 ^ skip_1) {
    // handle_ptr->sync_stream();
    //  raft::common::nvtx::range fun_scope("lb_multi_act");
    auto& upd = skip_0 ? upd_1 : upd_0;
    call_cnst_slack<erase_inf_cnst, i_t, f_t, 512>
      <<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(problem.cnst_csr.view(), upd.view());
    if (problem.cnst_csr.num_blocks_heavy != 0) {
      auto num_heavy_items = problem.n_constraints - problem.cnst_csr.heavy_beg_id;
      finalize_cnst_heavy<erase_inf_cnst, i_t, f_t, 32>
        <<<num_heavy_items, 32, 0, handle_ptr->get_stream()>>>(problem.cnst_csr.view(), upd.view());
    }
  } else {
    call_cnst_slack<erase_inf_cnst, i_t, f_t, 512>
      <<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(
        problem.cnst_csr.view(), upd_0.view(), upd_1.view());
    if (problem.cnst_csr.num_blocks_heavy != 0) {
      auto num_heavy_items = problem.n_constraints - problem.cnst_csr.heavy_beg_id;
      finalize_cnst_heavy<erase_inf_cnst, i_t, f_t, 32>
        <<<num_heavy_items, 32, 0, handle_ptr->get_stream()>>>(
          problem.cnst_csr.view(), upd_0.view(), upd_1.view());
    }
  }
  nvtxRangePop();
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::calculate_bounds_update_call(lb_problem_t<i_t, f_t>& problem,
                                                              const raft::handle_t* handle_ptr)
{
  handle_ptr->sync_stream();
  nvtxRangePush("lb_multi_bnd");

  auto num_blocks = problem.vars_csr.sub_warp_block_count + problem.vars_csr.med_block_count +
                    problem.vars_csr.num_blocks_heavy;
  call_bnd_update<i_t, f_t, 512><<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(
    problem.vars_csr.view(), upd_0.view(), upd_1.view());
  if (problem.vars_csr.num_blocks_heavy != 0) {
    bnd_heavy_update_next_changed_constraints<i_t, f_t, 512>
      <<<problem.vars_csr.num_blocks_heavy, 512, 0, handle_ptr->get_stream()>>>(
        problem.vars_csr.view(), upd_0.view(), upd_1.view());
  }

  handle_ptr->sync_stream();
  nvtxRangePop();
}

template <typename i_t, typename f_t>
bool lb_multi_probe_t<i_t, f_t>::calculate_bounds_update(lb_problem_t<i_t, f_t>& problem,
                                                         const raft::handle_t* handle_ptr)
{
  nvtxRangePush("lb_multi_bnd");
  constexpr i_t zero = 0;
  auto num_blocks    = problem.vars_csr.sub_warp_block_count + problem.vars_csr.med_block_count +
                    problem.vars_csr.num_blocks_heavy;
  // std::cout << "call_bnd_update sub_warp_block_count " << problem.vars_csr.sub_warp_block_count
  //           << "\n";
  // std::cout << "call_bnd_update med_block_count " << problem.vars_csr.med_block_count << "\n";
  // std::cout << "call_bnd_update num_blocks_heavy " << problem.vars_csr.num_blocks_heavy << "\n";

  // std::cout << "call_bnd_update sub_warp+med "
  //           << problem.vars_csr.sub_warp_block_count + problem.vars_csr.med_block_count << "\n";

  // std::cout << "num_heavy_items " << problem.n_variables - problem.vars_csr.heavy_beg_id << "\n";
  if (skip_0 && skip_1) {
    return false;
  } else if (skip_0) {
    upd_1.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    call_bnd_update<i_t, f_t, 512>
      <<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(problem.vars_csr.view(), upd_1.view());
    if (problem.vars_csr.num_blocks_heavy != 0) {
      bnd_heavy_update_next_changed_constraints<i_t, f_t, 512>
        <<<problem.vars_csr.num_blocks_heavy, 512, 0, handle_ptr->get_stream()>>>(
          problem.vars_csr.view(), upd_1.view());
    }
    i_t h_bounds_changed_1 = upd_1.bounds_changed.value(handle_ptr->get_stream());
    skip_1                 = (h_bounds_changed_1 == zero);

  } else if (skip_1) {
    upd_0.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    call_bnd_update<i_t, f_t, 512>
      <<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(problem.vars_csr.view(), upd_0.view());
    if (problem.vars_csr.num_blocks_heavy != 0) {
      bnd_heavy_update_next_changed_constraints<i_t, f_t, 512>
        <<<problem.vars_csr.num_blocks_heavy, 512, 0, handle_ptr->get_stream()>>>(
          problem.vars_csr.view(), upd_0.view());
    }
    i_t h_bounds_changed_0 = upd_0.bounds_changed.value(handle_ptr->get_stream());
    skip_0                 = (h_bounds_changed_0 == zero);

  } else {
    upd_0.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    upd_1.bounds_changed.set_value_async(zero, handle_ptr->get_stream());
    call_bnd_update<i_t, f_t, 512><<<num_blocks, 512, 0, handle_ptr->get_stream()>>>(
      problem.vars_csr.view(), upd_0.view(), upd_1.view());
    if (problem.vars_csr.num_blocks_heavy != 0) {
      bnd_heavy_update_next_changed_constraints<i_t, f_t, 512>
        <<<problem.vars_csr.num_blocks_heavy, 512, 0, handle_ptr->get_stream()>>>(
          problem.vars_csr.view(), upd_0.view(), upd_1.view());
    }
    i_t h_bounds_changed_0 = upd_0.bounds_changed.value(handle_ptr->get_stream());
    i_t h_bounds_changed_1 = upd_1.bounds_changed.value(handle_ptr->get_stream());
    skip_0                 = (h_bounds_changed_0 == zero);
    skip_1                 = (h_bounds_changed_1 == zero);
  }
  // i_t h_bounds_changed_0 = upd_0.bounds_changed.value(handle_ptr->get_stream());
  // i_t h_bounds_changed_1 = upd_1.bounds_changed.value(handle_ptr->get_stream());

  nvtxRangePop();
  // std::cout<<"lb_mlt h_bounds_changed_0 "<<h_bounds_changed_0<<" h_bounds_changed_1 "<<"
  // "<<h_bounds_changed_1<<"\n";
  //  return (h_bounds_changed_0 != 0) || (h_bounds_changed_1 != 0);
  // return std::make_tuple((h_bounds_changed_0 == 0), (h_bounds_changed_1 == 0));
  return (!skip_0 || !skip_1);
}

template <typename i_t, typename f_t>
termination_criterion_t lb_multi_probe_t<i_t, f_t>::bound_update_loop(
  lb_problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr, timer_t timer)
{
  skip_0                           = false;
  skip_1                           = false;
  termination_criterion_t criteria = termination_criterion_t::ITERATION_LIMIT;
  if (init_changed_constraints) {
    // all changed constraints are 1, next are zero
    upd_0.init_changed_constraints(handle_ptr);
    upd_1.init_changed_constraints(handle_ptr);
  } else {
    // reset for the next calls on the same object
    init_changed_constraints = true;
  }
  i_t iter;
  for (iter = 0; iter < settings.iteration_limit; ++iter) {
    if (timer.check_time_limit()) {
      criteria = termination_criterion_t::TIME_LIMIT;
      break;
    }
    // calculate activity for both probes
    calculate_constraint_slack_iter(pb, handle_ptr);
    auto proceed = calculate_bounds_update(pb, handle_ptr);
    if (!proceed) {
      if (iter == 0) {
        criteria = termination_criterion_t::NO_UPDATE;
      } else {
        criteria = termination_criterion_t::CONVERGENCE;
      }
      break;
    }
    // next_changed are updated, fill current changed with zero and swap
    // swap next and current changed constraints
    if (iter != settings.iteration_limit - 1) {
      if (!skip_0) { upd_0.prepare_for_next_iteration(handle_ptr); }
      if (!skip_1) { upd_1.prepare_for_next_iteration(handle_ptr); }
    }
  }
  if (compute_stats) { constraint_stats(pb, handle_ptr); }
  return criteria;
}

// template <typename i_t, typename f_t>
// void lb_multi_probe_t<i_t, f_t>::foo(const raft::handle_t* handle_ptr)
//{
//   cudaGraphNode_t cnst_slack_node;
//
//   {
//     auto num_blocks = problem.cnst_csr.sub_warp_block_count + problem.cnst_csr.med_block_count +
//                       problem.cnst_csr.num_blocks_heavy;
//     auto csr_view = problem.cnst_csr.view();
//     auto upd_0_view = upd_0.view();
//     auto upd_1_view = upd_1.view();
//
//     void* kernelArgs[] = {&csr_view, &upd_0_view, &upd_1_view};
//     cudaKernelNodeParams kernelNodeParams = {0};
//     kernelNodeParams.func =
//       (void*)call_cnst_slack<false, i_t, f_t, 512, csr_data_view_t<i_t, f_t>,
//       lb_bounds_update_data_t<i_t, f_t>::view_t>;
//     kernelNodeParams.gridDim        = dim3(num_blocks, 1, 1);
//     kernelNodeParams.blockDim       = dim3(512, 1, 1);
//     kernelNodeParams.sharedMemBytes = 0;
//     kernelNodeParams.kernelParams   = (void**)kernelArgs;
//     kernelNodeParams.extra          = NULL;
//
//     //add
//   }
//
//   if (problem.cnst_csr.num_blocks_heavy != 0) {
//     auto csr_view = problem.cnst_csr.view();
//     auto upd_0_view = upd_0.view();
//     auto upd_1_view = upd_1.view();
//
//     void* kernelArgs[] = {&csr_view, &upd_0_view, &upd_1_view};
//     cudaKernelNodeParams kernelNodeParams = {0};
//     kernelNodeParams.func =
//       (void*)finalize_cnst_heavy<false, i_t, f_t, 32, csr_data_view_t<i_t, f_t>,
//       lb_bounds_update_data_t<i_t, f_t>::view_t>;
//     kernelNodeParams.gridDim        = dim3(problem.n_constraints - problem.cnst_csr.heavy_beg_id,
//     1, 1); kernelNodeParams.blockDim       = dim3(32, 1, 1); kernelNodeParams.sharedMemBytes = 0;
//     kernelNodeParams.kernelParams   = (void**)kernelArgs;
//     kernelNodeParams.extra          = NULL;
//
//     //add
//   }
// }

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::update_device_bounds(const raft::handle_t* handle_ptr)
{
  cuopt_assert(upd_0.vars_bnd.size() == host_bounds.size(), "size of variable bound mismatch");
  cuopt_assert(upd_1.vars_bnd.size() == host_bounds.size(), "size of variable bound mismatch");

  raft::copy(
    upd_0.vars_bnd.data(), host_bounds.data(), host_bounds.size(), handle_ptr->get_stream());
  raft::copy(
    upd_1.vars_bnd.data(), host_bounds.data(), host_bounds.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::update_host_bounds(
  const raft::handle_t* handle_ptr,
  const raft::device_span<typename type_2<f_t>::type> variable_bounds)
{
  cuopt_assert(variable_bounds.size() == host_bounds.size(), "size of variable bound mismatch");
  raft::copy(
    host_bounds.data(), variable_bounds.data(), variable_bounds.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_bounds(
  const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
  const raft::handle_t* handle_ptr)
{
  using f_t2                           = typename type_2<f_t>::type;
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
                     auto idx              = thrust::get<0>(t);
                     auto upd_0_val        = thrust::get<1>(t);
                     auto upd_1_val        = thrust::get<2>(t);
                     upd_0_v.vars_bnd[idx] = f_t2{upd_0_val, upd_0_val};
                     upd_1_v.vars_bnd[idx] = f_t2{upd_1_val, upd_1_val};
                   });
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_interval_bounds(
  const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
  lb_problem_t<i_t, f_t>& pb,
  const raft::handle_t* handle_ptr)
{
  using f_t2                              = typename type_2<f_t>::type;
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
                      upd_0_v.vars_bnd[probe_var] = f_t2{lb_0, ub_0};
                      upd_1_v.vars_bnd[probe_var] = f_t2{lb_1, ub_1};
                    });
  // init changed constraints
  auto* orig_prob_ptr = pb.pb;
  i_t var_offset_begin =
    orig_prob_ptr->reverse_offsets.element(probe_var, handle_ptr->get_stream());
  i_t var_offset_end =
    orig_prob_ptr->reverse_offsets.element(probe_var + 1, handle_ptr->get_stream());
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
                   orig_prob_ptr->reverse_constraints.begin() + var_offset_begin,
                   orig_prob_ptr->reverse_constraints.begin() + var_offset_end,
                   [upd_0_v = upd_0.view(), upd_1_v = upd_1.view()] __device__(auto i) {
                     upd_0_v.changed_constraints[i] = 1;
                     upd_1_v.changed_constraints[i] = 1;
                   });
  init_changed_constraints = false;
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
termination_criterion_t lb_multi_probe_t<i_t, f_t>::solve(
  lb_problem_t<i_t, f_t>& pb,
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
termination_criterion_t lb_multi_probe_t<i_t, f_t>::solve_for_interval(
  lb_problem_t<i_t, f_t>& pb,
  const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
  const raft::handle_t* handle_ptr)
{
  timer_t timer(settings.time_limit);

  copy_problem_into_probing_buffers(pb, handle_ptr);
  set_interval_bounds(var_interval_vals, pb, handle_ptr);

  return bound_update_loop(pb, handle_ptr, timer);
}

template <typename i_t, typename f_t, typename f_t2>
struct detect_infeas_redun_t {
  __device__ __forceinline__ thrust::tuple<i_t, i_t> operator()(
    thrust::tuple<f_t2, f_t2, f_t, f_t> t) const
  {
    auto cnst_slack_0 = thrust::get<0>(t);
    auto cnst_slack_1 = thrust::get<1>(t);
    auto cnst_ub      = thrust::get<2>(t);
    auto cnst_lb      = thrust::get<3>(t);
    f_t eps           = get_cstr_tolerance<i_t, f_t>(
      cnst_lb, cnst_ub, tolerances.absolute_tolerance, tolerances.relative_tolerance);
    auto infeas_0 = (0 > cnst_slack_0.x + eps) || (0 < cnst_slack_0.y - eps);
    auto infeas_1 = (0 > cnst_slack_1.x + eps) || (0 < cnst_slack_1.y - eps);
    return thrust::make_tuple(infeas_0, infeas_1);
  }

 public:
  detect_infeas_redun_t()                                             = delete;
  detect_infeas_redun_t(const detect_infeas_redun_t<i_t, f_t, f_t2>&) = default;
  detect_infeas_redun_t(const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tols)
    : tolerances(tols)
  {
  }

 private:
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
};

template <typename f_t>
struct tuple_plus_t {
  __device__ thrust::tuple<f_t, f_t> operator()(thrust::tuple<f_t, f_t> t0,
                                                thrust::tuple<f_t, f_t> t1)
  {
    return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1),
                              thrust::get<1>(t0) + thrust::get<1>(t1));
  }
  __device__ thrust::tuple<f_t, f_t, f_t, f_t> operator()(thrust::tuple<f_t, f_t, f_t, f_t> t0,
                                                          thrust::tuple<f_t, f_t, f_t, f_t> t1)
  {
    return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1),
                              thrust::get<1>(t0) + thrust::get<1>(t1),
                              thrust::get<2>(t0) + thrust::get<2>(t1),
                              thrust::get<3>(t0) + thrust::get<3>(t1));
  }
};

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::constraint_stats(lb_problem_t<i_t, f_t>& pb,
                                                  const raft::handle_t* handle_ptr)
{
  using f_t2            = typename type_2<f_t>::type;
  auto* orig_prob_ptr   = pb.pb;
  auto upd_0_cnst_slack = upd_0.view().cnst_slack;
  auto upd_1_cnst_slack = upd_1.view().cnst_slack;
  auto detect_iter      = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(upd_0_cnst_slack.begin(),
                                                 upd_1_cnst_slack.begin(),
                                                 orig_prob_ptr->constraint_upper_bounds.begin(),
                                                 orig_prob_ptr->constraint_lower_bounds.begin())),
    detect_infeas_redun_t<i_t, f_t, f_t2>(orig_prob_ptr->tolerances));

  thrust::tie(infeas_constraints_count_0, infeas_constraints_count_1) =
    thrust::reduce(handle_ptr->get_thrust_policy(),
                   detect_iter,
                   detect_iter + pb.n_constraints,
                   thrust::make_tuple<i_t, i_t>(0, 0),
                   tuple_plus_t<i_t>{});

  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  if (infeas_constraints_count_0 > 0) {
    CUOPT_LOG_TRACE("First probe: Infeasible constraint count %d", infeas_constraints_count_0);
  }
  if (infeas_constraints_count_1 > 0) {
    CUOPT_LOG_TRACE("Second probe: Infeasible constraint count %d", infeas_constraints_count_1);
  }
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_updated_bounds(
  const raft::handle_t* handle_ptr,
  raft::device_span<typename type_2<f_t>::type> output_bounds,
  i_t select_update)
{
  auto& bounds = select_update ? upd_1.vars_bnd : upd_0.vars_bnd;

  cuopt_assert(bounds.size() == output_bounds.size(), "size of variable upper bound mismatch");
  raft::copy(output_bounds.data(), bounds.data(), bounds.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::set_updated_bounds(lb_problem_t<i_t, f_t>& pb,
                                                    i_t select_update,
                                                    const raft::handle_t* handle_ptr)
{
  set_updated_bounds(handle_ptr, make_span(pb.vars_bnd), select_update);
}

#if MIP_INSTANTIATE_FLOAT
template class lb_multi_probe_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_multi_probe_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
