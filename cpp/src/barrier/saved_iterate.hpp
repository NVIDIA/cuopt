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
  const auto satisfies_tolerance = [](f_t residual, f_t tolerance) {
    return std::isfinite(residual) && std::isfinite(tolerance) && tolerance > 0 &&
           residual < tolerance;
  };

  // Only consider candidates that satisfy every enabled relaxed tolerance.
  if (!satisfies_tolerance(current.primal, tolerance.primal) ||
      !satisfies_tolerance(current.dual, tolerance.dual) ||
      !satisfies_tolerance(current.complementarity, tolerance.complementarity) ||
      (check_objective_gap &&
       !satisfies_tolerance(current.objective_gap, tolerance.objective_gap))) {
    return false;
  }

  // Rank eligible candidates by their worst residual.
  f_t current_merit = std::max({current.primal, current.dual, current.complementarity});
  f_t saved_merit   = std::max({saved.primal, saved.dual, saved.complementarity});
  if (check_objective_gap) {
    current_merit = std::max(current_merit, current.objective_gap);
    saved_merit   = std::max(saved_merit, saved.objective_gap);
  }

  return current_merit < saved_merit;
}

}  // namespace cuopt::mathematical_optimization::barrier
