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

#include "solution.cuh"

#include <mip/utils.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
__global__ void test_feasibility_kernel(typename solution_t<i_t, f_t>::view_t sol)
{
  i_t c = threadIdx.x + blockIdx.x * blockDim.x;
  if (c >= sol.problem.n_constraints) { return; }
  f_t abs_tolerance = sol.problem.tolerances.absolute_tolerance;
  f_t val           = sol.constraint_value[c];
  bool feasible     = val > sol.problem.constraint_lower_bounds[c] - abs_tolerance &&
                  val < sol.problem.constraint_upper_bounds[c] + abs_tolerance;
  if (!feasible) {
    printf("c %d is infeasible val %f lower %f upper %f\n",
           c,
           val,
           sol.problem.constraint_lower_bounds[c],
           sol.problem.constraint_upper_bounds[c]);
  }
  cuopt_assert(feasible, "Constraints should be feasible");
}

template <typename i_t, typename f_t>
__global__ void test_variable_bounds_kernel(typename solution_t<i_t, f_t>::view_t sol,
                                            bool check_integer,
                                            i_t* is_feasible)
{
  i_t v = threadIdx.x + blockIdx.x * blockDim.x;
  if (v >= sol.problem.n_variables) { return; }
  f_t val       = sol.assignment[v];
  bool feasible = true;
  if (!isfinite(val)) {
    printf("inf var %d val %f l %f u %f integer %d\n",
           v,
           val,
           get_lower(sol.problem.variable_bounds[v]),
           get_upper(sol.problem.variable_bounds[v]),
           sol.problem.is_integer_var(v));
  }
  cuopt_assert(isfinite(val), "assignment should be finite!");
  if (check_integer && sol.problem.is_integer_var(v)) {
    if (!sol.problem.is_integer(val)) {
      feasible = false;
      printf("var %d val %f\n", v, val);
    }
    cuopt_assert(sol.problem.is_integer(val), "The variable must be integer");
  }
  if (!sol.problem.check_variable_within_bounds(v, val)) {
    feasible = false;
    printf("oob var %d val %f l %f u %f integer %d\n",
           v,
           val,
           get_lower(sol.problem.variable_bounds[v]),
           get_upper(sol.problem.variable_bounds[v]),
           sol.problem.is_integer_var(v));
  }
  cuopt_assert(feasible, "Variables should be feasible");
  if (is_feasible != nullptr) { *is_feasible = feasible; }
}

// test feasibility on
template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::test_feasibility(bool check_integer)
{
  cuopt_assert(compute_feasibility(), "Solution is not feasible!");
  test_variable_bounds(check_integer);
  handle_ptr->sync_stream();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

// test feasibility on
template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::test_absolute_feasibility()
{
  i_t TPB      = 64;
  i_t n_blocks = (problem_ptr->n_constraints + TPB - 1) / TPB;
  test_feasibility_kernel<i_t, f_t><<<n_blocks, TPB, 0, handle_ptr->get_stream()>>>(view());
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void solution_t<i_t, f_t>::test_variable_bounds(bool check_integer, i_t* is_feasible)
{
  i_t TPB      = 64;
  i_t n_blocks = (problem_ptr->n_variables + TPB - 1) / TPB;
  test_variable_bounds_kernel<i_t, f_t>
    <<<n_blocks, TPB, 0, handle_ptr->get_stream()>>>(view(), check_integer, is_feasible);
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

}  // namespace cuopt::linear_programming::detail
