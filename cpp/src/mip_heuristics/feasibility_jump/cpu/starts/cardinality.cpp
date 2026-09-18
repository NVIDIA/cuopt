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
void apply_exact_k_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.problem->nnz > fj_cpu.hp.start_nnz_limit) return;

  const auto started = std::chrono::steady_clock::now();
  auto timed_out     = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
           fj_cpu.hp.exact_k_budget_s;
  };

  struct exact_k_row_t {
    i_t k, begin, end;
  };
  std::vector<exact_k_row_t> rows;
  for (i_t row = 0; row < fj_cpu.problem->n_constraints; ++row) {
    if ((row & 0xFFF) == 0 && timed_out()) return;

    const f_t lb = fj_cpu.problem->cstr_lb[row];
    const f_t ub = fj_cpu.problem->cstr_ub[row];
    if (!std::isfinite(lb) || !std::isfinite(ub) || std::abs(lb - ub) > fj_cpu.hp.exact_k_tol)
      continue;

    const i_t begin = fj_cpu.problem->offsets[row];
    const i_t end   = fj_cpu.problem->offsets[row + 1];
    if (end - begin < 2 || end - begin > fj_cpu.hp.exact_k_max_width) continue;

    const f_t scale = fj_cpu.problem->coefficients[begin];
    if (scale <= 0) continue;
    bool uniform_binary = true;
    for (i_t p = begin; p < end && uniform_binary; ++p) {
      const i_t var       = fj_cpu.problem->variables[p];
      const f_t coeff     = fj_cpu.problem->coefficients[p];
      const f_t agreement = fj_cpu.hp.exact_k_tol * std::max((f_t)1, std::abs(scale));
      uniform_binary =
        fj_cpu.h_is_binary_variable[var] && coeff > 0 && std::abs(coeff - scale) <= agreement;
    }
    if (!uniform_binary) continue;

    const double cardinality = (double)lb / scale;
    const i_t k              = (i_t)std::lround(cardinality);
    if (std::abs(cardinality - k) <= 1e-4 && k >= 0 && k <= end - begin)
      rows.push_back({k, begin, end});
  }
  if (rows.empty()) return;

  std::sort(rows.begin(), rows.end(), [](const exact_k_row_t& a, const exact_k_row_t& b) {
    return a.end - a.begin < b.end - b.begin;
  });

  const i_t n_variables = fj_cpu.problem->n_variables;
  std::vector<i_t> degree(n_variables, 0);
  for (const auto& row : rows)
    for (i_t p = row.begin; p < row.end; ++p)
      ++degree[fj_cpu.problem->variables[p]];

  std::vector<int8_t> state(n_variables, -1);
  std::vector<i_t> free_vars;
  for (size_t index = 0; index < rows.size(); ++index) {
    if ((index & 0xFFF) == 0 && timed_out()) break;
    const auto& row = rows[index];

    i_t selected = 0;
    free_vars.clear();
    for (i_t p = row.begin; p < row.end; ++p) {
      const i_t var = fj_cpu.problem->variables[p];
      selected += state[var] == 1;
      if (state[var] < 0) free_vars.push_back(var);
    }
    const i_t needed = row.k - selected;
    if (needed < 0 || (i_t)free_vars.size() < needed) continue;

    std::sort(free_vars.begin(), free_vars.end(), [&degree](i_t a, i_t b) {
      return degree[a] < degree[b];
    });
    for (i_t p = 0; p < (i_t)free_vars.size(); ++p)
      state[free_vars[p]] = (int8_t)(p < needed);
  }

  recompute_lhs(fj_cpu);
  const i_t baseline = fj_cpu.violated_constraints.size();
  const auto anchor  = fj_cpu.h_assignment;
  for (i_t var = 0; var < n_variables; ++var)
    if (state[var] >= 0) fj_cpu.h_assignment[var] = state[var];

  recompute_lhs(fj_cpu);
  if ((i_t)fj_cpu.violated_constraints.size() >= baseline) {
    fj_cpu.h_assignment = anchor;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

template <typename i_t, typename f_t>
void apply_ordinal_midpoint_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  const auto& offsets = fj_cpu.problem->card_row_offsets;
  const auto& members = fj_cpu.problem->card_variables;
  const auto& owner   = fj_cpu.problem->card_group_of_variable;
  if (offsets.size() <= 1 || owner.size() != fj_cpu.h_assignment.size()) return;

  std::vector<i_t> groups;
  i_t covered = 0;
  for (i_t group = 0; group + 1 < (i_t)offsets.size(); ++group) {
    const i_t begin = offsets[group], end = offsets[group + 1];
    if (end - begin < 3) continue;
    i_t selected  = 0;
    bool disjoint = true;
    for (i_t p = begin; p < end; ++p) {
      const i_t var = members[p];
      disjoint &= owner[var] == group;
      selected += fj_cpu.h_assignment[var].get() > f_t{0.5};
    }
    // The cardinality index also contains exact-k rows.  A current one-hot row is the
    // unambiguous, assignment-level certificate needed here without retaining another row table.
    if (disjoint && selected == 1) {
      groups.push_back(group);
      covered += end - begin;
    }
  }

  const i_t n_binary = fj_cpu.n_binary_vars;
  if ((i_t)groups.size() < 16 || n_binary == 0 || 4 * covered < 3 * n_binary) return;

  for (i_t group : groups) {
    const i_t begin = offsets[group], end = offsets[group + 1];
    const i_t middle = begin + (end - begin) / 2;
    for (i_t p = begin; p < end; ++p)
      fj_cpu.h_assignment[members[p]] = p == middle ? f_t{1} : f_t{0};
  }
  recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

template <typename i_t, typename f_t>
void repair_difficult_anchor(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  recompute_lhs(fj_cpu);
  const i_t baseline = fj_cpu.violated_constraints.size();
  if (baseline == 0 ||
      baseline <= fj_cpu.problem->n_constraints / fj_cpu.hp.anchor_repair_violated_share)
    return;

  const auto started = std::chrono::steady_clock::now();
  const auto anchor  = fj_cpu.h_assignment;
  const std::vector<i_t> violated(fj_cpu.violated_constraints.begin(),
                                  fj_cpu.violated_constraints.end());
  std::vector<row_repair_move_t<i_t, f_t>> candidates;

  for (i_t row : violated) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
        fj_cpu.hp.anchor_repair_budget_s)
      break;

    const f_t lb  = fj_cpu.problem->cstr_lb[row];
    const f_t ub  = fj_cpu.problem->cstr_ub[row];
    f_t sum       = fj_cpu.h_lhs[row];
    f_t target    = 0;
    f_t direction = 0;
    if (sum < lb) {
      direction = 1;
      target    = lb;
    } else if (sum > ub) {
      direction = -1;
      target    = ub;
    } else {
      continue;
    }

    collect_row_repair_moves<i_t, f_t>(fj_cpu,
                                       fj_cpu.problem->offsets[row],
                                       fj_cpu.problem->offsets[row + 1],
                                       direction,
                                       fj_cpu.hp.exact_k_tol,
                                       candidates);
    for (const auto& move : candidates) {
      if (direction > 0 ? sum >= target : sum <= target) break;
      const f_t delta = move.new_val - (f_t)fj_cpu.h_assignment[move.var];
      sum += move.coeff * delta;
      fj_cpu.h_assignment[move.var] = move.new_val;
    }
  }

  recompute_lhs(fj_cpu);
  if ((i_t)fj_cpu.violated_constraints.size() >= baseline) {
    fj_cpu.h_assignment = anchor;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

#if MIP_INSTANTIATE_FLOAT
template void apply_exact_k_start<int, float>(fj_cpu_climber_t<int, float>&);
template void apply_ordinal_midpoint_start<int, float>(fj_cpu_climber_t<int, float>&);
template void repair_difficult_anchor<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void apply_exact_k_start<int, double>(fj_cpu_climber_t<int, double>&);
template void apply_ordinal_midpoint_start<int, double>(fj_cpu_climber_t<int, double>&);
template void repair_difficult_anchor<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
