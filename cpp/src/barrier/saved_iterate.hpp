/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <algorithm>
#include <cmath>

namespace cuopt::mathematical_optimization::barrier {

template <typename f_t>
struct iterate_residuals_t {
  f_t primal;
  f_t dual;
  f_t complementarity;
  f_t objective_gap;
};

template <typename f_t>
bool should_save_iterate(const iterate_residuals_t<f_t>& current,
                         const iterate_residuals_t<f_t>& saved,
                         const iterate_residuals_t<f_t>& tolerance,
                         bool check_objective_gap)
{
  const f_t current_residuals[]{
    current.primal, current.dual, current.complementarity, current.objective_gap};
  const f_t saved_residuals[]{saved.primal, saved.dual, saved.complementarity, saved.objective_gap};
  const f_t tolerances[]{
    tolerance.primal, tolerance.dual, tolerance.complementarity, tolerance.objective_gap};

  const int num_metrics = check_objective_gap ? 4 : 3;

  // Only consider candidates that satisfy every enabled relaxed tolerance.
  for (int i = 0; i < num_metrics; ++i) {
    if (!std::isfinite(current_residuals[i]) || !std::isfinite(tolerances[i]) ||
        !(tolerances[i] > 0) || !(current_residuals[i] < tolerances[i])) {
      return false;
    }
  }

  // Rank eligible candidates by their worst tolerance-normalized residual.
  f_t current_merit = 0;
  f_t saved_merit   = 0;
  for (int i = 0; i < num_metrics; ++i) {
    current_merit = std::max(current_merit, current_residuals[i] / tolerances[i]);
    saved_merit   = std::max(saved_merit, saved_residuals[i] / tolerances[i]);
  }

  return current_merit < saved_merit;
}

}  // namespace cuopt::mathematical_optimization::barrier
