/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"
#include "starts.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void collect_row_repair_moves(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                              i_t row_begin,
                              i_t row_end,
                              f_t direction,
                              f_t tol,
                              std::vector<row_repair_move_t<i_t, f_t>>& out)
{
  out.clear();
  for (i_t i = row_begin; i < row_end; ++i) {
    const i_t var = fj_cpu.problem->variables[i];
    if (!is_integer_var<i_t, f_t>(fj_cpu, var)) continue;

    const f_t coeff   = fj_cpu.problem->coefficients[i];
    const f_t val     = fj_cpu.h_assignment[var];
    const f_t lb      = get_lower(fj_cpu.h_var_bounds[var].get());
    const f_t ub      = get_upper(fj_cpu.h_var_bounds[var].get());
    const bool is_bin = fj_cpu.h_is_binary_variable[var] != 0;

    // Raising the variable shifts the sum by `direction * coeff`; lowering it by the negation.
    const f_t raise = direction * coeff;
    if (raise > 0 && val < ub - tol) {
      const f_t new_val = is_bin ? (f_t)1 : std::floor(val) + 1;
      if (new_val > val && new_val <= ub + tol) out.push_back({raise, var, coeff, new_val});
    } else if (raise < 0 && val > lb + tol) {
      const f_t new_val = is_bin ? (f_t)0 : std::ceil(val) - 1;
      if (new_val < val && new_val >= lb - tol) out.push_back({-raise, var, coeff, new_val});
    }
  }
  std::sort(out.begin(),
            out.end(),
            [](const row_repair_move_t<i_t, f_t>& a, const row_repair_move_t<i_t, f_t>& b) {
              return a.effect > b.effect;
            });
}

template <typename i_t, typename f_t>
void apply_greedy_covering_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.problem->nnz > fj_cpu.hp.start_nnz_limit) return;

  recompute_lhs(fj_cpu);
  const i_t baseline_violated  = fj_cpu.violated_constraints.size();
  const auto anchor_assignment = fj_cpu.h_assignment;

  const i_t n_constraints = fj_cpu.problem->n_constraints;
  std::vector<i_t> row_order(n_constraints);
  for (i_t i = 0; i < n_constraints; ++i)
    row_order[i] = i;
  std::sort(row_order.begin(), row_order.end(), [&](i_t a, i_t b) {
    return (fj_cpu.problem->offsets[a + 1] - fj_cpu.problem->offsets[a]) <
           (fj_cpu.problem->offsets[b + 1] - fj_cpu.problem->offsets[b]);
  });

  const auto started         = std::chrono::steady_clock::now();
  const double time_budget_s = fj_cpu.hp.covering_budget_s;
  const f_t tol              = 1e-6;
  const i_t max_passes       = 2;
  std::vector<row_repair_move_t<i_t, f_t>> candidates;
  bool out_of_time = false;

  for (i_t pass = 0; pass < max_passes && !out_of_time; ++pass) {
    for (i_t k = 0; k < n_constraints; ++k) {
      if ((k & 0xFFF) == 0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
            time_budget_s) {
        out_of_time = true;
        break;
      }
      const i_t cstr_idx  = row_order[k];
      const i_t row_begin = fj_cpu.problem->offsets[cstr_idx];
      const i_t row_end   = fj_cpu.problem->offsets[cstr_idx + 1];
      if (row_begin == row_end) continue;

      const f_t lb      = fj_cpu.problem->cstr_lb[cstr_idx];
      const f_t ub      = fj_cpu.problem->cstr_ub[cstr_idx];
      const bool has_lb = std::isfinite(lb);
      const bool has_ub = std::isfinite(ub);
      if (!has_lb && !has_ub) continue;

      f_t sum = compensated_dot2_csr(*fj_cpu.problem, fj_cpu.h_assignment, cstr_idx);

      // Equality rows are driven to their bound; one-sided rows only to the side they violate.
      const bool is_equality = has_lb && has_ub && std::abs(lb - ub) < tol;
      f_t direction          = 0;
      f_t target             = 0;
      if (is_equality && std::abs(sum - lb) > tol) {
        direction = sum < lb ? (f_t)1 : (f_t)-1;
        target    = lb;
      } else if (has_lb && sum < lb - tol) {
        direction = 1;
        target    = lb;
      } else if (has_ub && sum > ub + tol) {
        direction = -1;
        target    = ub;
      } else {
        continue;
      }

      collect_row_repair_moves<i_t, f_t>(fj_cpu, row_begin, row_end, direction, tol, candidates);
      for (const auto& m : candidates) {
        if (direction > 0 ? sum >= target - tol : sum <= target + tol) break;
        const f_t delta = m.new_val - (f_t)fj_cpu.h_assignment[m.var];
        sum += m.coeff * delta;
        fj_cpu.h_assignment[m.var] = m.new_val;
      }
    }
  }

  recompute_lhs(fj_cpu);
  if ((i_t)fj_cpu.violated_constraints.size() >= baseline_violated) {
    fj_cpu.h_assignment = anchor_assignment;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

#if MIP_INSTANTIATE_FLOAT
template void collect_row_repair_moves<int, float>(fj_cpu_climber_t<int, float>&,
                                                   int,
                                                   int,
                                                   float,
                                                   float,
                                                   std::vector<row_repair_move_t<int, float>>&);
template void apply_greedy_covering_start<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void collect_row_repair_moves<int, double>(fj_cpu_climber_t<int, double>&,
                                                    int,
                                                    int,
                                                    double,
                                                    double,
                                                    std::vector<row_repair_move_t<int, double>>&);
template void apply_greedy_covering_start<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
