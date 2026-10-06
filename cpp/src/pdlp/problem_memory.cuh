/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/problem/problem.cuh>

namespace cuopt::mathematical_optimization::pdlp {

template <typename T>
void release_workspace(rmm::device_uvector<T>& workspace)
{
  workspace.resize(0, workspace.stream());
  workspace.shrink_to_fit(workspace.stream());
}

// Only for a problem owned by an LP solve, or PDLP's private scaled copy. Never
// release these buffers on a caller-owned MIP problem: B&B and fixing use them.
template <typename i_t, typename f_t>
void release_mip_only_workspace(mip::problem_t<i_t, f_t>& problem)
{
  release_workspace(problem.integer_fixed_variable_map);
  release_workspace(problem.related_variables);
  release_workspace(problem.related_variables_offsets);
  release_workspace(problem.lp_state.prev_primal);
  release_workspace(problem.lp_state.prev_dual);
  release_workspace(problem.fixing_helpers.reduction_in_rhs);
  release_workspace(problem.fixing_helpers.variable_fix_mask);
}

template <typename i_t, typename f_t>
mip::problem_t<i_t, f_t> make_scaled_pdlp_problem(const mip::problem_t<i_t, f_t>& problem)
{
  mip::problem_t<i_t, f_t> scaled(problem, false);
  release_mip_only_workspace(scaled);
  return scaled;
}

}  // namespace cuopt::mathematical_optimization::pdlp
