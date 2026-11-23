/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../state.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// An eliminated coordinate, recorded in original variable indices. Records are lifted in reverse.
template <typename i_t, typename f_t>
struct fj_equality_substitution_t {
  i_t variable;
  f_t constant;
  std::vector<std::pair<i_t, f_t>> terms;
};

template <typename i_t, typename f_t>
void detect_implied_integers(fj_cpu_climber_t<i_t, f_t>&, fj_cpu_problem_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void detect_free_equality_singletons(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void precompute_problem_features(fj_cpu_climber_t<i_t, f_t>&, fj_cpu_problem_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void build_cardinality_index(fj_cpu_climber_t<i_t, f_t>&, fj_cpu_problem_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void certify_epigraph_variables(fj_cpu_climber_t<i_t, f_t>&, i_t);
template <typename i_t, typename f_t>
void build_one_sided_rows(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> make_equality_reduced_climber(
  fj_cpu_climber_t<i_t, f_t>&,
  double,
  std::vector<fj_equality_substitution_t<i_t, f_t>>&,
  std::vector<i_t>&);

struct cpufj_geometry_requirements_t {
  int32_t min_groups{0};
  // Zero permits arbitrary continuous scopes.
  int32_t required_scope_size{0};
  double min_gated_row_fraction{0};
  bool require_matching_equality_count{false};
  bool require_gate_per_group{false};
  // Allow binary inequalities that do not certify an activating row.
  bool allow_other_binary_rows{false};
};

// Recognize continuous regions selected by disjoint exact-one groups. Callers
// select structural requirements and may certify an activating row with a small M.
template <typename i_t, typename f_t, typename certify_small_gate_t>
bool has_cpufj_disjunctive_geometry(const fj_cpu_climber_t<i_t, f_t>& climber,
                                    const cpufj_geometry_requirements_t& requirements,
                                    certify_small_gate_t certify_small_gate)
{
  if (climber.problem == nullptr) return false;
  const auto& problem = *climber.problem;
  const i_t groups    = static_cast<i_t>(problem.card_cardinalities.size());
  if (groups < requirements.min_groups) return false;
  for (i_t cardinality : problem.card_cardinalities)
    if (cardinality != 1) return false;
  for (i_t variable : climber.h_binary_indices) {
    const i_t group = problem.card_group_of_variable[variable];
    if (group < 0 || group >= groups) return false;
  }

  std::vector<i_t> scopes(static_cast<size_t>(groups) * requirements.required_scope_size);
  std::vector<i_t> scope_sizes(requirements.required_scope_size > 0 ? groups : 0, 0);
  std::vector<bool> gated_groups(requirements.require_gate_per_group ? groups : 0, false);
  i_t gated = 0, equalities = 0;
  for (i_t row = 0; row < problem.n_constraints; ++row) {
    const f_t lower = problem.cstr_lb[row], upper = problem.cstr_ub[row];
    if (lower == upper) {
      ++equalities;
      continue;
    }
    i_t binary           = -1;
    f_t gate_coefficient = 0, continuous_max = 0;
    bool multiple_binaries = false;
    for (i_t q = problem.offsets[row]; q < problem.offsets[row + 1]; ++q) {
      const i_t variable = problem.variables[q];
      if (climber.h_is_binary_variable[variable]) {
        if (binary >= 0) {
          if (!requirements.allow_other_binary_rows) return false;
          multiple_binaries = true;
          break;
        }
        binary           = variable;
        gate_coefficient = problem.coefficients[q];
      } else {
        continuous_max = std::max(continuous_max, std::abs(problem.coefficients[q]));
      }
    }
    if (binary < 0 || multiple_binaries) continue;
    const i_t group      = problem.card_group_of_variable[binary];
    const bool lower_row = gate_coefficient < 0 && std::isfinite(lower) && !std::isfinite(upper);
    const bool upper_row = gate_coefficient > 0 && std::isfinite(upper) && !std::isfinite(lower);
    if (group < 0 || (!lower_row && !upper_row) || !(continuous_max > 0)) {
      if (!requirements.allow_other_binary_rows) return false;
      continue;
    }
    if (!(std::abs(gate_coefficient) >= f_t{1000} * continuous_max) &&
        !certify_small_gate(row, binary, lower_row)) {
      if (!requirements.allow_other_binary_rows) return false;
      continue;
    }
    ++gated;
    if (requirements.require_gate_per_group) gated_groups[group] = true;
    if (requirements.required_scope_size == 0) continue;
    const auto scope =
      scopes.begin() + static_cast<size_t>(group) * requirements.required_scope_size;
    auto& size = scope_sizes[group];
    for (i_t q = problem.offsets[row]; q < problem.offsets[row + 1]; ++q) {
      const i_t variable = problem.variables[q];
      if (variable == binary || std::find(scope, scope + size, variable) != scope + size) continue;
      if (size == requirements.required_scope_size) return false;
      scope[size++] = variable;
    }
  }
  return (!requirements.require_matching_equality_count || equalities == groups) &&
         gated >= requirements.min_gated_row_fraction * problem.n_constraints &&
         std::all_of(scope_sizes.begin(),
                     scope_sizes.end(),
                     [&](i_t size) { return size == requirements.required_scope_size; }) &&
         std::all_of(gated_groups.begin(), gated_groups.end(), [](bool gated) { return gated; });
}

template <typename i_t, typename f_t>
bool has_cpufj_disjunctive_geometry(const fj_cpu_climber_t<i_t, f_t>& climber,
                                    const cpufj_geometry_requirements_t& requirements)
{
  return has_cpufj_disjunctive_geometry(
    climber, requirements, [](i_t, i_t, bool) { return false; });
}

}  // namespace cuopt::mathematical_optimization::mip
