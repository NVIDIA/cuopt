/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

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
  if (check_objective_gap && !std::isfinite(current.objective_gap)) { return false; }

  const bool current_gap_pass =
    !check_objective_gap || current.objective_gap < tolerance.objective_gap;
  const bool saved_gap_pass = !check_objective_gap || saved.objective_gap < tolerance.objective_gap;

  // Preserve a recoverable iterate even when better feasibility comes with a worse objective gap.
  if (current_gap_pass != saved_gap_pass) { return current_gap_pass; }

  return current.primal < saved.primal && current.dual < saved.dual &&
         current.complementarity < saved.complementarity;
}

}  // namespace cuopt::mathematical_optimization::barrier
