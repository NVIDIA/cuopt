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

#pragma once

#include "simple_rounding.cuh"

#include <mip/utils.cuh>
#include <utilities/cuda_helpers.cuh>
#include <utilities/device_utils.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
__global__ void simple_rounding_kernel(typename solution_t<i_t, f_t>::view_t solution,
                                       bool* successful)
{
  if (TH_ID_X >= solution.problem.n_integer_vars) { return; }
  i_t var_id   = solution.problem.integer_indices[TH_ID_X];
  f_t curr_val = solution.assignment[var_id];
  if (solution.problem.is_integer(curr_val)) { return; }

  i_t up_locks   = 0;
  i_t down_locks = 0;

  auto [offset_begin, offset_end] = solution.problem.reverse_range_for_var(var_id);
  for (i_t i = offset_begin; i < offset_end; i += 1) {
    auto cstr_idx   = solution.problem.reverse_constraints[i];
    auto cstr_coeff = solution.problem.reverse_coefficients[i];

    // boxed constraint. can't be rounded safely
    if (std::isfinite(solution.problem.constraint_lower_bounds[cstr_idx]) &&
        std::isfinite(solution.problem.constraint_upper_bounds[cstr_idx])) {
      up_locks += 1;
      down_locks += 1;
      continue;
    }

    f_t sign = std::isfinite(solution.problem.constraint_upper_bounds[cstr_idx]) ? 1 : -1;

    if (cstr_coeff * sign > 0) {
      up_locks += 1;
    } else {
      down_locks += 1;
    }
  }

  bool can_round_up   = up_locks == 0;
  bool can_round_down = down_locks == 0;

  if (can_round_up && can_round_down) {
    if (solution.problem.objective_coefficients[var_id] > 0) {
      solution.assignment[var_id] = floor(curr_val);
    } else {
      solution.assignment[var_id] = ceil(curr_val);
    }
  } else if (can_round_up) {
    solution.assignment[var_id] = ceil(curr_val);
  } else if (can_round_down) {
    solution.assignment[var_id] = floor(curr_val);
  } else {
    *successful = false;
  }
}

// rounds each integer variable to the nearest integer value that doesn't violate the bounds
template <typename i_t, typename f_t>
__global__ void nearest_rounding_kernel(typename solution_t<i_t, f_t>::view_t solution,
                                        uint64_t seed)
{
  if (TH_ID_X >= solution.problem.n_integer_vars) { return; }
  i_t var_id = solution.problem.integer_indices[TH_ID_X];
  raft::random::PCGenerator rng(seed, var_id, 0);
  f_t curr_val = solution.assignment[var_id];
  if (solution.problem.is_integer(curr_val)) { return; }
  const f_t int_tol           = solution.problem.tolerances.integrality_tolerance;
  auto var_bnd                = solution.problem.variable_bounds[var_id];
  f_t lb                      = get_lower(var_bnd);
  f_t ub                      = get_upper(var_bnd);
  f_t nearest_val             = round_nearest(curr_val, lb, ub, int_tol, rng);
  solution.assignment[var_id] = nearest_val;
}

template <typename i_t, typename f_t>
__global__ void brute_force_check_kernel(typename solution_t<i_t, f_t>::view_t solution,
                                         i_t n_integers_to_round,
                                         raft::device_span<i_t> var_map,
                                         raft::device_span<f_t> constraint_buf,
                                         i_t* best_config)
{
  auto constraint_buf_block = constraint_buf.subspan(blockIdx.x * solution.constraint_value.size());
  block_copy(constraint_buf_block.data(),
             solution.constraint_value.data(),
             solution.constraint_value.size());
  __syncthreads();
  i_t config = blockIdx.x;
  for (i_t var_idx = 0; var_idx < n_integers_to_round; ++var_idx) {
    bool up_round                   = (config >> var_idx) & 0x1;
    i_t var                         = var_map[var_idx];
    auto [offset_begin, offset_end] = solution.problem.reverse_range_for_var(var);
    f_t val                         = solution.assignment[var];
    f_t val_round                   = up_round ? ceil(val) : floor(val);
    f_t delta                       = val_round - val;
    for (i_t c_idx = threadIdx.x + offset_begin; c_idx < offset_end; c_idx += blockDim.x) {
      i_t c     = solution.problem.reverse_constraints[c_idx];
      f_t coeff = solution.problem.reverse_coefficients[c_idx];
      constraint_buf_block[c] += delta * coeff;
    }
    __syncthreads();
  }
  __syncthreads();
  i_t th_feasible_count = 0.;
  for (i_t c = threadIdx.x; c < solution.problem.n_constraints; c += blockDim.x) {
    f_t constr_val = constraint_buf_block[c];
    th_feasible_count +=
      (i_t)is_constraint_feasible<i_t, f_t>(constr_val,
                                            solution.problem.constraint_lower_bounds[c],
                                            solution.problem.constraint_upper_bounds[c],
                                            solution.problem.tolerances);
  }
  __shared__ i_t shbuf[raft::WarpSize];
  i_t total_feasible = raft::blockReduce(th_feasible_count, (char*)shbuf);
  if (threadIdx.x == 0) {
    if (total_feasible == solution.problem.n_constraints) { atomicExch(best_config, config); }
  }
}

template <typename i_t, typename f_t>
__global__ void apply_feasible_rounding_kernel(typename solution_t<i_t, f_t>::view_t solution,
                                               i_t n_integers_to_round,
                                               raft::device_span<i_t> var_map,
                                               i_t* best_config)
{
  i_t round_config = *best_config;
  for (i_t var_idx = threadIdx.x; var_idx < n_integers_to_round; var_idx += blockDim.x) {
    bool up_round            = (round_config >> var_idx) & 0x1;
    i_t var                  = var_map[var_idx];
    f_t val                  = solution.assignment[var];
    f_t val_round            = up_round ? ceil(val) : floor(val);
    solution.assignment[var] = val_round;
  }
}

template <typename i_t, typename f_t>
__global__ void random_nearest_rounding_kernel(typename solution_t<i_t, f_t>::view_t solution,
                                               uint64_t seed,
                                               i_t* n_randomly_rounded)
{
  i_t var_id = TH_ID_X;
  if (var_id >= solution.problem.n_variables) { return; }
  raft::random::PCGenerator rng(seed, var_id, 0);
  if (solution.problem.is_integer_var(var_id)) {
    // already integer values will be rounded to exact integer
    f_t curr_val = solution.assignment[var_id];
    f_t fraction = get_fractionality_of_val(curr_val);
    f_t rounding_val;
    if (fraction > 0.25 && fraction < 0.75) {
      bool round_up = rng.next_u32() % 2;
      rounding_val  = round_up ? ceil(curr_val) : floor(curr_val);
      atomicAdd(n_randomly_rounded, 1);
      solution.assignment[var_id] = rounding_val;
    }
  }
}

template <typename i_t, typename f_t>
__global__ void random_rounding_kernel(typename solution_t<i_t, f_t>::view_t solution,
                                       uint64_t seed,
                                       i_t* shuffled_indices,
                                       i_t* n_randomly_rounded,
                                       i_t n_random_rounds)
{
  if (TH_ID_X != 0) return;
  i_t curr_n_random = 0;
  raft::random::PCGenerator rng(seed, 0, 0);
  for (i_t i = 0; i < solution.problem.n_integer_vars; ++i) {
    i_t var_id   = shuffled_indices[i];
    f_t curr_val = solution.assignment[var_id];
    if (solution.problem.is_integer(curr_val)) { continue; }
    bool round_up    = rng.next_u32() % 2;
    f_t rounding_val = round_up ? ceil(curr_val) : floor(curr_val);
    cuopt_assert(solution.problem.check_variable_within_bounds(var_id, rounding_val),
                 "Var must be within bounds");
    solution.assignment[var_id] = rounding_val;
    *n_randomly_rounded += 1;
    if (++curr_n_random == n_random_rounds) { break; }
  }
}

}  // namespace cuopt::linear_programming::detail
