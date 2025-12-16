/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <mip/utils.cuh>
#include "bounds_presolve.cuh"

#include <utilities/timer.hpp>

#include "utils.cuh"

// Tobias Achterberg, Robert E. Bixby, Zonghao Gu, Edward Rothberg, Dieter Weninger (2019) Presolve
// Reductions in Mixed Integer Programming. INFORMS Journal on Computing 32(2):473-506.
// https://doi.org/10.1287/ijoc.2018.0857

// This implementation loosly follows section 5.4 in the paper. For now, we only update the
// constraint bounds which should help reduce the gap between LP relaxation and optimal integer
// solutions We need to implement variable bounds improvement as well

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class conditional_bound_strengthening_t {
 public:
  conditional_bound_strengthening_t(problem_t<i_t, f_t>& problem);

  // FIXME:: For now, just update constraint bounds
  // Implement parameterization logic to improve the variable bounds
  void update_constraint_bounds(problem_t<i_t, f_t>& problem,
                                bound_presolve_t<i_t, f_t>& bounds_update);

  void resize(problem_t<i_t, f_t>& problem);

  void select_constraint_pairs_host(problem_t<i_t, f_t>& problem);
  void select_constraint_pairs_device(problem_t<i_t, f_t>& problem);

  void solve(problem_t<i_t, f_t>& problem);

  rmm::device_uvector<int2> constraint_pairs;

  rmm::device_uvector<i_t> locks_per_constraint;
};

}  // namespace cuopt::linear_programming::detail
