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
  if (!(current.primal < tolerance.primal && current.dual < tolerance.dual &&
        current.complementarity < tolerance.complementarity)) {
    return false;
  }
  if (!std::isfinite(current.primal) || !std::isfinite(current.dual) ||
      !std::isfinite(current.complementarity)) {
    return false;
  }
  if (check_objective_gap &&
      (!std::isfinite(current.objective_gap) || !std::isfinite(tolerance.objective_gap) ||
       !(tolerance.objective_gap > 0))) {
    return false;
  }

  // Rank by the worst fraction of a relaxed tolerance. Once the merit is below one,
  // a strict improvement cannot replace a recoverable iterate with an unacceptable one.
  f_t current_merit = std::max({current.primal / tolerance.primal,
                                current.dual / tolerance.dual,
                                current.complementarity / tolerance.complementarity});
  f_t saved_merit   = std::max({saved.primal / tolerance.primal,
                                saved.dual / tolerance.dual,
                                saved.complementarity / tolerance.complementarity});
  if (check_objective_gap) {
    current_merit = std::max(current_merit, current.objective_gap / tolerance.objective_gap);
    saved_merit   = std::max(saved_merit, saved.objective_gap / tolerance.objective_gap);
  }
  return current_merit < saved_merit;
}

}  // namespace cuopt::mathematical_optimization::barrier
