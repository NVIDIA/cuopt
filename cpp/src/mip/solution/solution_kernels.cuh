/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "solution.cuh"

#include <raft/util/reduction.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
__global__ void compute_constraint_values(typename solution_t<i_t, f_t>::view_t sol)
{
  __shared__ f_t shmem[raft::WarpSize];
  i_t c                           = blockIdx.x;
  auto [offset_begin, offset_end] = sol.problem.range_for_constraint(c);
  f_t th_val                      = 0.;
  for (i_t i = threadIdx.x + offset_begin; i < offset_end; i += blockDim.x) {
    i_t var   = sol.problem.variables[i];
    f_t coeff = sol.problem.coefficients[i];
    th_val += coeff * sol.assignment[var];
  }
  f_t constr_val = raft::blockReduce(th_val, (char*)shmem);
  if (threadIdx.x == 0) {
    sol.constraint_value[c] = constr_val;
    sol.lower_excess[c]     = max(0., sol.problem.constraint_lower_bounds[c] - constr_val);
    sol.upper_excess[c]     = max(0., constr_val - sol.problem.constraint_upper_bounds[c]);
    i_t feasible            = is_constraint_feasible<i_t, f_t>(constr_val,
                                                    sol.problem.constraint_lower_bounds[c],
                                                    sol.problem.constraint_upper_bounds[c],
                                                    sol.problem.tolerances);
    atomicAdd(sol.n_feasible_constraints, feasible);
  }
}

template <typename i_t, typename f_t>
__global__ void compute_feasibility_kernel(
  typename solution_t<i_t, f_t>::view_t sol,
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols)
{
  i_t c         = threadIdx.x + blockIdx.x * blockDim.x;
  bool feasible = true;
  if (c < sol.problem.n_constraints) {
    f_t val  = sol.constraint_value[c];
    feasible = is_constraint_feasible(
      val, sol.problem.constraint_lower_bounds[c], sol.problem.constraint_upper_bounds[c]);
  }
  __shared__ i_t shmem[raft::WarpSize];
  i_t is_block_feasible = raft::blockReduce((i_t)feasible, (char*)shmem, raft::min_op{});
  if (threadIdx.x == 0) {
    atomicAnd(sol.is_feasible, bool(is_block_feasible));
    atomicAdd(sol.n_feasible_constraints, is_block_feasible);
  }
}

}  // namespace cuopt::linear_programming::detail
