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

#include <cub/cub.cuh>

#include "problem.cuh"

namespace cuopt::linear_programming::detail {

// TODO replace this kernel with actually removing the variables from the assignment
template <typename i_t, typename f_t>
__global__ void fix_given_variables_kernel(const typename problem_t<i_t, f_t>::view_t problem,
                                           typename problem_t<i_t, f_t>::view_t new_problem,
                                           raft::device_span<f_t> assignment,
                                           raft::device_span<i_t> vars_to_fix)
{
  // use a threadblock for each constraint
  // loop over variables sequentially (assuming fixations occur with few variables)
  i_t c                  = blockIdx.x;
  const f_t int_tol      = problem.tolerances.integrality_tolerance;
  f_t th_total_reduction = 0.;
  for (i_t v_idx = 0; v_idx < vars_to_fix.size(); ++v_idx) {
    i_t var_to_fix                  = vars_to_fix[v_idx];
    f_t var_val                     = assignment[var_to_fix];
    auto [offset_begin, offset_end] = problem.range_for_constraint(c);
    for (i_t i = threadIdx.x + offset_begin; i < offset_end; i += blockDim.x) {
      i_t v = problem.variables[i];
      // search for var
      if (v == var_to_fix) {
        f_t coeff    = problem.coefficients[i];
        f_t curr_val = floor(var_val + int_tol) * coeff;
        // there should be only one var that is contained within the constraint
        th_total_reduction += curr_val;
      }
    }
    // wait until the var is fixed if any, so that at a time single var modifies the constraint
    __syncthreads();
  }
  __shared__ f_t shmem[raft::WarpSize];
  f_t total_reduction = raft::blockReduce(th_total_reduction, (char*)shmem);
  // the new problem constraint values are not initialized, so we are basically initializng them
  // here even if the total_reduction is zero, do the initiailizrion here
  if (threadIdx.x == 0) {
    new_problem.constraint_lower_bounds[c] = problem.constraint_lower_bounds[c] - total_reduction;
    new_problem.constraint_upper_bounds[c] = problem.constraint_upper_bounds[c] - total_reduction;
  }
}

template <typename i_t, typename f_t>
__global__ void compute_new_offsets(const typename problem_t<i_t, f_t>::view_t orig_problem,
                                    typename problem_t<i_t, f_t>::view_t new_problem,
                                    raft::device_span<i_t> remaining_vars)
{
  i_t var_id                      = remaining_vars[blockIdx.x];
  auto [offset_begin, offset_end] = orig_problem.reverse_range_for_var(var_id);
  for (i_t i = threadIdx.x + offset_begin; i < offset_end; i += blockDim.x) {
    i_t cstr = orig_problem.reverse_constraints[i];
    atomicAdd(new_problem.offsets.data() + cstr, 1);
  }
}

template <typename i_t, typename f_t>
__global__ void compute_new_csr(const typename problem_t<i_t, f_t>::view_t orig_problem,
                                typename problem_t<i_t, f_t>::view_t new_problem,
                                raft::device_span<i_t> remaining_vars,
                                raft::device_span<i_t> write_pos)
{
  i_t var_id                      = remaining_vars[blockIdx.x];
  i_t new_var_id                  = blockIdx.x;
  auto [offset_begin, offset_end] = orig_problem.reverse_range_for_var(var_id);
  for (i_t i = threadIdx.x + offset_begin; i < offset_end; i += blockDim.x) {
    i_t cstr                                              = orig_problem.reverse_constraints[i];
    f_t coeff                                             = orig_problem.reverse_coefficients[i];
    i_t write_offset                                      = atomicAdd(write_pos.data() + cstr, 1);
    i_t new_cstr_offset                                   = new_problem.offsets[cstr];
    new_problem.variables[write_offset + new_cstr_offset] = new_var_id;
    new_problem.coefficients[write_offset + new_cstr_offset] = coeff;
  }
}

// Used to compute the offsets and required memory for computing related vars
template <typename i_t, typename f_t>
__global__ void compute_related_vars_unique(const typename problem_t<i_t, f_t>::view_t problem,
                                            i_t slice_begin,
                                            i_t slice_end,
                                            raft::device_span<i_t> output_varmap)
{
  i_t num_slices = slice_end - slice_begin;
  cuopt_assert(output_varmap.size() >= num_slices * problem.n_variables, "invalid size");

  for (i_t v = slice_begin; v < slice_end; v += 1) {
    auto [cstr_range_begin, cstr_range_end] =
      std::make_pair(problem.reverse_offsets[v], problem.reverse_offsets[v + 1]);

    for (auto i = cstr_range_begin + blockIdx.x; i < cstr_range_end; i += gridDim.x) {
      auto cstr_idx = problem.reverse_constraints[i];
      auto [var_range_begin, var_range_end] =
        std::make_pair(problem.offsets[cstr_idx], problem.offsets[cstr_idx + 1]);

      for (auto j = threadIdx.x; j < var_range_end - var_range_begin; j += blockDim.x) {
        auto var_idx = problem.variables[var_range_begin + j];

        i_t varmap_idx    = var_idx;
        i_t varmap_offset = v - slice_begin;

        // atomicOr? is this necessary for constant byte stores?
        output_varmap[varmap_offset * problem.n_variables + varmap_idx] = 1;
      }
    }
  }
}

}  // namespace cuopt::linear_programming::detail
