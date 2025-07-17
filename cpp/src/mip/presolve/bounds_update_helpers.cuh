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

#include <mip/problem/problem.cuh>
#include <mip/utils.cuh>
#include "bounds_update_data.cuh"

namespace cuopt::linear_programming::detail {

// Activity calculation

template <typename f_t>
inline __device__ f_t min_act_of_var(f_t coeff, f_t var_lb, f_t var_ub)
{
  return (coeff < 0.) ? coeff * var_ub : coeff * var_lb;
}

template <typename f_t>
inline __device__ f_t max_act_of_var(f_t coeff, f_t var_lb, f_t var_ub)
{
  return (coeff < 0.) ? coeff * var_lb : coeff * var_ub;
}

template <typename f_t>
inline __device__ f_t update_lb(f_t curr_lb, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_min_act / coeff : delta_max_act / coeff;
  return max(curr_lb, comp_bnd);
}

template <typename f_t>
inline __device__ f_t update_ub(f_t curr_ub, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_max_act / coeff : delta_min_act / coeff;
  return min(curr_ub, comp_bnd);
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void calc_activity_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_0,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_1)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t cnst_idx    = blockIdx.x;
  i_t cnst_offset = pb.offsets[cnst_idx];
  i_t cnst_degree = pb.offsets[cnst_idx + 1] - cnst_offset;
  f_t min_act_0 = 0, max_act_0 = 0;
  f_t min_act_1 = 0, max_act_1 = 0;
  bool changed_0 = upd_0.changed_constraints[cnst_idx] == 1;
  bool changed_1 = upd_1.changed_constraints[cnst_idx] == 1;
  if (!changed_0 && !changed_1) { return; }

  for (i_t i = threadIdx.x; i < cnst_degree; i += blockDim.x) {
    auto coeff   = pb.coefficients[cnst_offset + i];
    auto var_idx = pb.variables[cnst_offset + i];
    if (changed_0) {
      auto var_lb_0 = upd_0.lb[var_idx];
      auto var_ub_0 = upd_0.ub[var_idx];
      min_act_0 += min_act_of_var(coeff, var_lb_0, var_ub_0);
      max_act_0 += max_act_of_var(coeff, var_lb_0, var_ub_0);
      atomicExch(&upd_0.changed_variables[var_idx], 1);
    }
    if (changed_1) {
      auto var_lb_1 = upd_1.lb[var_idx];
      auto var_ub_1 = upd_1.ub[var_idx];
      min_act_1 += min_act_of_var(coeff, var_lb_1, var_ub_1);
      max_act_1 += max_act_of_var(coeff, var_lb_1, var_ub_1);
      atomicExch(&upd_1.changed_variables[var_idx], 1);
    }
  }
  if (changed_0) {
    min_act_0 = BlockReduce(temp_storage).Sum(min_act_0);
    __syncthreads();
    max_act_0 = BlockReduce(temp_storage).Sum(max_act_0);
    __syncthreads();
  }
  if (changed_1) {
    min_act_1 = BlockReduce(temp_storage).Sum(min_act_1);
    __syncthreads();
    max_act_1 = BlockReduce(temp_storage).Sum(max_act_1);
  }

  if (threadIdx.x == 0) {
    if (changed_0) {
      upd_0.min_activity[cnst_idx] = min_act_0;
      upd_0.max_activity[cnst_idx] = max_act_0;
    }
    if (changed_1) {
      upd_1.min_activity[cnst_idx] = min_act_1;
      upd_1.max_activity[cnst_idx] = max_act_1;
    }
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void calc_activity_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t cnst_idx    = blockIdx.x;
  i_t cnst_offset = pb.offsets[cnst_idx];
  i_t cnst_degree = pb.offsets[cnst_idx + 1] - cnst_offset;
  f_t min_act = 0, max_act = 0;
  if (upd.changed_constraints[cnst_idx] == 0) { return; }

  for (i_t i = threadIdx.x; i < cnst_degree; i += blockDim.x) {
    auto coeff   = pb.coefficients[cnst_offset + i];
    auto var_idx = pb.variables[cnst_offset + i];
    auto var_lb  = upd.lb[var_idx];
    auto var_ub  = upd.ub[var_idx];
    min_act += min_act_of_var(coeff, var_lb, var_ub);
    max_act += max_act_of_var(coeff, var_lb, var_ub);
    atomicExch(&upd.changed_variables[var_idx], 1);
  }
  min_act = BlockReduce(temp_storage).Sum(min_act);
  __syncthreads();
  max_act = BlockReduce(temp_storage).Sum(max_act);

  if (threadIdx.x == 0) {
    upd.min_activity[cnst_idx] = min_act;
    upd.max_activity[cnst_idx] = max_act;
  }
}

// Update bounds

template <typename i_t, typename f_t>
inline __device__ bool check_infeasibility(f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t eps)
{
  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
}

template <typename i_t, typename f_t>
inline __device__ bool check_infeasibility(
  f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t abs_tol, f_t rel_tol)
{
  auto eps = get_cstr_tolerance<i_t, f_t>(cnst_lb, cnst_ub, abs_tol, rel_tol);
  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
}

template <typename i_t, typename f_t>
inline __device__ bool check_redundancy(
  f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t abs_tol, f_t rel_tol)
{
  auto eps = get_cstr_tolerance<i_t, f_t>(cnst_lb, cnst_ub, abs_tol, rel_tol);
  return (min_a > cnst_lb + eps) && (max_a < cnst_ub - eps);
}

template <typename f_t>
inline __device__ bool skip_update(thrust::pair<f_t, f_t> bnd, f_t int_tol)
{
  return (thrust::get<0>(bnd) + int_tol >= thrust::get<1>(bnd));
}

template <typename i_t, typename f_t>
inline __device__ thrust::pair<f_t, f_t> update_bounds_per_cnst(
  typename problem_t<i_t, f_t>::view_t pb,
  f_t coeff,
  i_t cnst_idx,
  f_t cnst_lb,
  f_t cnst_ub,
  typename bounds_update_data_t<i_t, f_t>::view_t upd,
  thrust::pair<f_t, f_t> bnd,
  thrust::pair<f_t, f_t> old_bnd)
{
  auto min_a = upd.min_activity[cnst_idx];
  auto max_a = upd.max_activity[cnst_idx];
  // don't propagate over constraints that are infeasible
  if (check_infeasibility<i_t, f_t>(min_a,
                                    max_a,
                                    cnst_lb,
                                    cnst_ub,
                                    pb.tolerances.absolute_tolerance,
                                    pb.tolerances.relative_tolerance)) {
    return bnd;
  }
  min_a -= (coeff < 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  max_a -= (coeff > 0) ? coeff * thrust::get<1>(old_bnd) : coeff * thrust::get<0>(old_bnd);
  auto delta_min_act  = cnst_ub - min_a;
  auto delta_max_act  = cnst_lb - max_a;
  thrust::get<0>(bnd) = update_lb(thrust::get<0>(bnd), coeff, delta_min_act, delta_max_act);
  thrust::get<1>(bnd) = update_ub(thrust::get<1>(bnd), coeff, delta_min_act, delta_max_act);
  return bnd;
}

template <typename i_t, typename f_t>
inline __device__ bool write_updated_bounds(typename problem_t<i_t, f_t>::view_t pb,
                                            i_t var_idx,
                                            bool is_int,
                                            typename bounds_update_data_t<i_t, f_t>::view_t upd,
                                            thrust::pair<f_t, f_t> bnd,
                                            thrust::pair<f_t, f_t> old_bnd)
{
  auto new_lb = thrust::get<0>(bnd);
  auto new_ub = thrust::get<1>(bnd);
  new_lb      = is_int ? ceil(new_lb - pb.tolerances.integrality_tolerance) : new_lb;
  new_ub      = is_int ? floor(new_ub + pb.tolerances.integrality_tolerance) : new_ub;

  auto lb_updated =
    (abs(new_lb - thrust::get<0>(old_bnd)) > 1e3 * pb.tolerances.absolute_tolerance);
  auto ub_updated =
    (abs(new_ub - thrust::get<1>(old_bnd)) > 1e3 * pb.tolerances.absolute_tolerance);

  if (lb_updated) { upd.lb[var_idx] = new_lb; }
  if (ub_updated) { upd.ub[var_idx] = new_ub; }

  if (lb_updated || ub_updated) { atomicAdd(upd.bounds_changed, 1); }
  // bounds_changed tracks the number of significantly changed bounds, we want any small change to
  // be detected
  if (new_lb != old_bnd.first || new_ub != old_bnd.second) { return true; }
  return false;
}

template <typename i_t, typename f_t, i_t BDIM>
__device__ void update_bounds(typename problem_t<i_t, f_t>::view_t pb,
                              i_t var_idx,
                              i_t var_offset,
                              i_t var_degree,
                              bool is_int,
                              typename bounds_update_data_t<i_t, f_t>::view_t upd_0,
                              thrust::pair<f_t, f_t> old_bnd_0,
                              typename bounds_update_data_t<i_t, f_t>::view_t upd_1,
                              thrust::pair<f_t, f_t> old_bnd_1)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto bnd_0        = old_bnd_0;
  auto bnd_1        = old_bnd_1;
  i_t var_changed_0 = upd_0.changed_variables[var_idx];
  i_t var_changed_1 = upd_1.changed_variables[var_idx];
  if (!var_changed_0 && !var_changed_1) { return; }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    auto a        = pb.reverse_coefficients[var_offset + i];
    auto cnst_ub  = pb.constraint_upper_bounds[cnst_idx];
    auto cnst_lb  = pb.constraint_lower_bounds[cnst_idx];
    if (var_changed_0) {
      bool cstr_changed_0 = upd_0.changed_constraints[cnst_idx] == 1;
      if (cstr_changed_0) {
        bnd_0 = update_bounds_per_cnst(pb, a, cnst_idx, cnst_lb, cnst_ub, upd_0, bnd_0, old_bnd_0);
      }
    }
    if (var_changed_1) {
      bool cstr_changed_1 = upd_1.changed_constraints[cnst_idx] == 1;
      if (cstr_changed_1) {
        bnd_1 = update_bounds_per_cnst(pb, a, cnst_idx, cnst_lb, cnst_ub, upd_1, bnd_1, old_bnd_1);
      }
    }
  }
  __syncthreads();
  if (var_changed_0) {
    thrust::get<0>(bnd_0) = BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd_0), cub::Max());
    __syncthreads();
    thrust::get<1>(bnd_0) = BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd_0), cub::Min());
    __syncthreads();
  }
  if (var_changed_1) {
    thrust::get<0>(bnd_1) = BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd_1), cub::Max());
    __syncthreads();
    thrust::get<1>(bnd_1) = BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd_1), cub::Min());
  }
  __shared__ bool changed_0, changed_1;
  if (threadIdx.x == 0) {
    changed_0 = write_updated_bounds(pb, var_idx, is_int, upd_0, bnd_0, old_bnd_0);
    changed_1 = write_updated_bounds(pb, var_idx, is_int, upd_1, bnd_1, old_bnd_1);
  }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    if (changed_0) { atomicExch(&upd_0.next_changed_constraints[cnst_idx], 1); }
    if (changed_1) { atomicExch(&upd_1.next_changed_constraints[cnst_idx], 1); }
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__device__ void update_bounds(typename problem_t<i_t, f_t>::view_t pb,
                              i_t var_idx,
                              i_t var_offset,
                              i_t var_degree,
                              bool is_int,
                              typename bounds_update_data_t<i_t, f_t>::view_t upd,
                              thrust::pair<f_t, f_t> old_bnd)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t var_changed = upd.changed_variables[var_idx];
  if (!var_changed) { return; }

  auto bnd = old_bnd;
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    bool changed  = upd.changed_constraints[cnst_idx] == 1;
    if (!changed) { continue; }
    auto a       = pb.reverse_coefficients[var_offset + i];
    auto cnst_ub = pb.constraint_upper_bounds[cnst_idx];
    auto cnst_lb = pb.constraint_lower_bounds[cnst_idx];
    bnd          = update_bounds_per_cnst(pb, a, cnst_idx, cnst_lb, cnst_ub, upd, bnd, old_bnd);
  }

  thrust::get<0>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<0>(bnd), cub::Max());
  __syncthreads();
  thrust::get<1>(bnd) = BlockReduce(temp_storage).Reduce(thrust::get<1>(bnd), cub::Min());
  __shared__ bool changed;
  if (threadIdx.x == 0) { changed = write_updated_bounds(pb, var_idx, is_int, upd, bnd, old_bnd); }
  __syncthreads();
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    if (changed) { atomicExch(&upd.next_changed_constraints[cnst_idx], 1); }
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void update_bounds_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t var_idx    = blockIdx.x;
  i_t var_offset = pb.reverse_offsets[var_idx];
  i_t var_degree = pb.reverse_offsets[var_idx + 1] - var_offset;
  bool is_int    = (pb.variable_types[var_idx] == var_t::INTEGER);

  auto old_bnd = thrust::make_pair(upd.lb[var_idx], upd.ub[var_idx]);

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  auto skip = skip_update(old_bnd, pb.tolerances.integrality_tolerance);
  if (skip) {
    return;
  } else {
    update_bounds<i_t, f_t, BDIM>(pb, var_idx, var_offset, var_degree, is_int, upd, old_bnd);
  }
}

template <typename i_t, typename f_t, i_t BDIM>
__global__ void update_bounds_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_0,
                                     typename bounds_update_data_t<i_t, f_t>::view_t upd_1)
{
  using BlockReduce = cub::BlockReduce<f_t, BDIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  i_t var_idx    = blockIdx.x;
  i_t var_offset = pb.reverse_offsets[var_idx];
  i_t var_degree = pb.reverse_offsets[var_idx + 1] - var_offset;
  bool is_int    = (pb.variable_types[var_idx] == var_t::INTEGER);

  auto old_bnd_0 = thrust::make_pair(upd_0.lb[var_idx], upd_0.ub[var_idx]);
  auto old_bnd_1 = thrust::make_pair(upd_1.lb[var_idx], upd_1.ub[var_idx]);

  // if it is a set variable then don't propagate the bound
  // consider continuous vars as set if their bounds cross or equal
  auto skip_0 = skip_update(old_bnd_0, pb.tolerances.integrality_tolerance);
  auto skip_1 = skip_update(old_bnd_1, pb.tolerances.integrality_tolerance);
  if (skip_0 && skip_1) {
    return;
  } else if (skip_0) {
    update_bounds<i_t, f_t, BDIM>(pb, var_idx, var_offset, var_degree, is_int, upd_1, old_bnd_1);
  } else if (skip_1) {
    update_bounds<i_t, f_t, BDIM>(pb, var_idx, var_offset, var_degree, is_int, upd_0, old_bnd_0);
  } else {
    update_bounds<i_t, f_t, BDIM>(
      pb, var_idx, var_offset, var_degree, is_int, upd_0, old_bnd_0, upd_1, old_bnd_1);
  }
}

}  // namespace cuopt::linear_programming::detail
