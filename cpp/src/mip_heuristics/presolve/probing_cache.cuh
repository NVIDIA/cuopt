/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "bounds_presolve.cuh"
#include "probing_cache.hpp"

#include <mip_heuristics/utils.cuh>

#include <mip_heuristics/presolve/presolve_budget_policy.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class bound_presolve_t;

/*
  Probing cache is a set of implied bounds when we set a variable to some value.
  We keep two sets of changed bounds for each interval:
  For binary: 0 and 1
  For integer: finite_bound and finite_bound<> if it is unbounded on one side.
  Notice that we keep an interval here.
  Else if both sides are bounded, we do interval/2 > and interval/2 <.
  We can use this cache, for infeasibility detection, implied bounds, fast bounds setting and bulk
  rounding. To save from memory, we will keep the the results in host map.
*/

template <typename i_t, typename f_t>
presolve_features_t probing_presolve_features(problem_t<i_t, f_t> const& problem)
{
  presolve_features_t f{};
  f.n_vars = problem.n_variables;
  f.n_cons = problem.n_constraints;
  f.nnz    = problem.nnz;
  f.n_int  = problem.n_integer_vars;
  f.n_bin  = problem.n_binary_vars;

  auto h_offsets = cuopt::host_copy(problem.offsets, problem.handle_ptr->get_stream());
  for (size_t i = 0; i + 1 < h_offsets.size(); ++i) {
    f.max_row_len = std::max<double>(f.max_row_len, h_offsets[i + 1] - h_offsets[i]);
  }
  return f;
}

template <typename i_t, typename f_t>
bool compute_probing_cache(bound_presolve_t<i_t, f_t>& bound_presolve,
                           problem_t<i_t, f_t>& problem,
                           timer_t timer,
                           double work_limit     = std::numeric_limits<double>::infinity(),
                           size_t step_size_hint = 2048);

}  // namespace cuopt::mathematical_optimization::mip
