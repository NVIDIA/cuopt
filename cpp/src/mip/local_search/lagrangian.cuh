/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/device_utils.cuh>

#include <raft/util/reduction.cuh>
namespace cuopt::linear_programming::detail {

// computes the lagrangian objectives with the given weights of the constraints
// we need both left and right weights because constraints might have had excess from either lower
// or upper bound usually a constraint has an excess from a single side but if it has excess from
// both sides, we simply sum them up (one of them is negative)
template <typename i_t, typename f_t>
__global__ void compute_lagrangian_weights_kernel(typename problem_t<i_t, f_t>::view_t problem,
                                                  raft::device_span<f_t> cstr_left_weights,
                                                  raft::device_span<f_t> cstr_right_weights,
                                                  raft::device_span<f_t> out_weights)
{
  __shared__ f_t shbuf[raft::WarpSize];
  i_t var_id                      = blockIdx.x;
  auto [offset_begin, offset_end] = problem.reverse_range_for_var(var_id);
  f_t th_weight                   = 0.;
  for (i_t i = offset_begin + threadIdx.x; i < offset_end; i += blockDim.x) {
    auto cstr_idx   = problem.reverse_constraints[i];
    auto cstr_coeff = problem.reverse_coefficients[i];
    th_weight += cstr_left_weights[cstr_idx] * cstr_coeff;
    th_weight -= cstr_right_weights[cstr_idx] * cstr_coeff;
  }
  f_t var_weight = raft::blockReduce(th_weight, (char*)shbuf);
  cuopt_assert(isfinite(var_weight), "var_weight should be finite!");
  if (threadIdx.x == 0) { out_weights[var_id] = var_weight; }
}

// TODO make sure in the caller context the left and right cstr weights are updated correctly
// for example diversity manager case
template <typename i_t, typename f_t>
inline rmm::device_uvector<f_t> get_weighted_lagrangian_weights(
  solution_t<i_t, f_t>& solution,
  problem_t<i_t, f_t>& problem,
  rmm::device_uvector<f_t>& cstr_left_weights,
  rmm::device_uvector<f_t>& cstr_right_weights)
{
  rmm::device_uvector<f_t> out_weights(problem.n_variables, solution.handle_ptr->get_stream());
  const i_t TPB      = 128;
  const i_t n_blocks = problem.n_variables;
  compute_lagrangian_weights_kernel<i_t, f_t>
    <<<n_blocks, TPB, 0, solution.handle_ptr->get_stream()>>>(
      problem.view(),
      raft::device_span<f_t>{cstr_left_weights.data(), cstr_left_weights.size()},
      raft::device_span<f_t>{cstr_right_weights.data(), cstr_right_weights.size()},
      raft::device_span<f_t>{out_weights.data(), out_weights.size()});
  solution.handle_ptr->sync_stream();
  return out_weights;
}

template <typename i_t, typename f_t>
inline rmm::device_uvector<f_t> get_lagrangian_weights(solution_t<i_t, f_t>& solution,
                                                       const problem_t<i_t, f_t>& problem)
{
  rmm::device_uvector<f_t> cstr_weights(problem.n_constraints, solution.handle_ptr->get_stream());
  thrust::uninitialized_fill(
    solution.handle_ptr->get_thrust_policy(), cstr_weights.begin(), cstr_weights.end(), 1.);
  return get_weighted_lagrangian_weights(solution, problem, cstr_weights, cstr_weights);
}

}  // namespace cuopt::linear_programming::detail
