/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../audit.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"
#include "starts.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void apply_precedence_completion_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  phase_timer_t timer(fj_cpu.t_start);
  const i_t n_constraints = fj_cpu.problem->n_constraints;

  i_t lower_only = 0;
  for (i_t row = 0; row < n_constraints; ++row)
    lower_only +=
      std::isfinite(fj_cpu.problem->cstr_lb[row]) && !std::isfinite(fj_cpu.problem->cstr_ub[row]);
  if (lower_only * fj_cpu.hp.precedence_lower_den < n_constraints * fj_cpu.hp.precedence_lower_num)
    return;

  const auto started = std::chrono::steady_clock::now();
  auto timed_out     = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
           fj_cpu.hp.precedence_budget_s;
  };

  recompute_lhs(fj_cpu);
  const auto anchor         = fj_cpu.h_assignment;
  auto best                 = anchor;
  const i_t anchor_count    = fj_cpu.violated_constraints.size();
  const f_t anchor_severity = -fj_cpu.total_violations;
  i_t best_count            = anchor_count;
  f_t best_severity         = anchor_severity;

  for (i_t pass = 0; pass < fj_cpu.hp.precedence_passes; ++pass) {
    bool changed = false;
    for (i_t row = 0; row < n_constraints; ++row) {
      if ((row & 0x1FF) == 0 && timed_out()) break;
      const f_t lb = fj_cpu.problem->cstr_lb[row];
      if (!std::isfinite(lb) || std::isfinite(fj_cpu.problem->cstr_ub[row])) continue;
      const f_t lhs     = fj_cpu.h_lhs[row];
      const f_t deficit = lb - lhs;
      if (deficit <= fj_cpu.row_tolerance) continue;

      // The head is the row's only positive continuous coefficient. A row with none, or with
      // several, is not a precedence row and is left to the search.
      i_t head                = -1;
      f_t head_coeff          = 0;
      const auto [begin, end] = model_range_for_row<i_t, f_t>(fj_cpu, row);
      for (i_t p = begin; p < end; ++p) {
        const i_t var   = fj_cpu.problem->variables[p];
        const f_t coeff = fj_cpu.problem->coefficients[p];
        if (coeff <= 0 || is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
        if (head >= 0 && head != var) {
          head = -2;
          break;
        }
        head       = var;
        head_coeff = coeff;
      }
      if (head < 0 || head_coeff == 0) continue;

      const auto bounds = fj_cpu.h_var_bounds[head].get();
      const f_t old_val = fj_cpu.h_assignment[head];
      const f_t value   = std::min(get_upper(bounds), old_val + deficit / head_coeff);
      const f_t delta   = value - old_val;
      if (!(delta > 0) || !std::isfinite(value)) continue;

      fj_cpu.h_assignment[head]       = value;
      const auto [col_begin, col_end] = model_range_for_var<i_t, f_t>(fj_cpu, head);
      for (i_t q = col_begin; q < col_end; ++q) {
        const i_t touched     = fj_cpu.problem->reverse_constraints[q];
        const f_t patched     = fj_cpu.h_lhs[touched];
        fj_cpu.h_lhs[touched] = patched + fj_cpu.problem->reverse_coefficients[q] * delta;
      }
      changed = true;

      cuopt_assert(
        (f_t)fj_cpu.h_lhs[row] >= lb - fj_cpu.row_tolerance || value >= get_upper(bounds),
        "precedence step neither repaired the row nor saturated its head");
    }

    recompute_lhs(fj_cpu);
    const i_t count    = fj_cpu.violated_constraints.size();
    const f_t severity = -fj_cpu.total_violations;
    if (count < best_count || (count == best_count && severity < best_severity)) {
      best_count    = count;
      best_severity = severity;
      best          = fj_cpu.h_assignment;
      if (count == 0) break;
    }
    if (!changed || timed_out()) break;
  }

  cuopt_assert(fj_cpu.h_assignment.size() == anchor.size(),
               "incumbent_assignment span would be invalidated");
  const bool keep =
    best_count < anchor_count || (best_count == anchor_count && best_severity < anchor_severity);
  if (keep) {
    fj_cpu.h_assignment = best;
  } else {
    fj_cpu.h_assignment = anchor;
  }
  recompute_lhs(fj_cpu);
  cuopt_func_call(audit_assignment_bounds(fj_cpu, "precedence start"));
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

#if MIP_INSTANTIATE_FLOAT
template void apply_precedence_completion_start<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void apply_precedence_completion_start<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
