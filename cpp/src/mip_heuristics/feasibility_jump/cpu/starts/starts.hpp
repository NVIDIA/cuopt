/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../audit.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct row_repair_move_t {
  f_t effect;
  i_t var;
  f_t coeff;
  f_t new_val;
};

template <typename i_t, typename f_t>
static bool try_commit_start(fj_cpu_climber_t<i_t, f_t>& c, const std::vector<f_t>& candidate)
{
  const auto& problem = *c.problem;
  cuopt_assert(candidate.size() == (size_t)problem.n_variables, "start size mismatch");
  for (i_t variable = 0; variable < problem.n_variables; ++variable) {
    const auto bounds = c.h_var_bounds[variable].get();
    if (!std::isfinite(candidate[variable]) || candidate[variable] < get_lower(bounds) ||
        candidate[variable] > get_upper(bounds) ||
        (is_integer_var(c, variable) && candidate[variable] != std::round(candidate[variable])))
      return false;
  }
  const f_t tolerance = 1e-7;
  for (i_t row = 0; row < problem.n_constraints; ++row) {
    const f_t activity = compensated_dot2_csr(problem, candidate, row);
    if (!std::isfinite(activity) || activity < problem.cstr_lb[row] - tolerance ||
        activity > problem.cstr_ub[row] + tolerance)
      return false;
  }

  const auto anchor = c.h_assignment;
  std::copy(candidate.begin(), candidate.end(), c.h_assignment.begin());
  recompute_lhs(c);
  if (!c.violated_constraints.empty() || !check_variable_feasibility<i_t, f_t>(c) ||
      (c.feasible_found && c.h_incumbent_objective >= c.h_best_objective)) {
    c.h_assignment = anchor;
    recompute_lhs(c);
    return false;
  }
  c.h_best_assignment = c.h_assignment;
  c.h_best_objective  = c.h_incumbent_objective - c.settings.parameters.breakthrough_move_epsilon;
  c.feasible_found    = true;
  report_cpu_incumbent(c);
  share_cpu_incumbent(c);
  return true;
}

template <typename i_t, typename f_t>
void collect_row_repair_moves(fj_cpu_climber_t<i_t, f_t>& c,
                              i_t row_begin,
                              i_t row_end,
                              f_t direction,
                              f_t tolerance,
                              std::vector<row_repair_move_t<i_t, f_t>>& out);

template <typename i_t, typename f_t>
void apply_greedy_covering_start(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_exact_k_start(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_ordinal_midpoint_start(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void repair_difficult_anchor(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_precedence_completion_start(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
void apply_affine_equality_start(fj_cpu_climber_t<i_t, f_t>& c, double budget);

template <typename i_t, typename f_t>
void apply_unit_commitment_start(fj_cpu_climber_t<i_t, f_t>& c);

template <typename i_t, typename f_t>
bool apply_fixed_charge_network_start(fj_cpu_climber_t<i_t, f_t>& c, double budget);

template <typename i_t, typename f_t>
bool apply_pmedian_start(fj_cpu_climber_t<i_t, f_t>& c, double budget);

template <typename i_t, typename f_t>
void apply_structural_completion_start(fj_cpu_climber_t<i_t, f_t>& c);

}  // namespace cuopt::mathematical_optimization::mip
