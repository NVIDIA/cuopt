/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/state.hpp>
#include <mip_heuristics/utils.cuh>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {
// Reused climbers mutate cached activities during ruin/repair. Recheck the raw model
// before retaining or publishing a candidate, using the solver's configured tolerances.
template <typename i_t, typename f_t>
bool verify_cpufj_lns_feasible(const fj_cpu_problem_t<i_t, f_t>& problem,
                               const std::vector<typename type_2<f_t>::type>& bounds,
                               const std::vector<var_t>& types,
                               const std::vector<f_t>& assignment)
{
  if (assignment.size() != static_cast<size_t>(problem.n_variables)) return false;
  const f_t int_tol = problem.tolerances.integrality_tolerance;
  for (i_t v = 0; v < problem.n_variables; ++v) {
    const f_t x = assignment[v];
    if (!std::isfinite(x) || x < get_lower(bounds[v]) - int_tol ||
        x > get_upper(bounds[v]) + int_tol ||
        (types[v] == var_t::INTEGER && !problem.is_integer(x)))
      return false;
  }
  f_t objective = 0;
  for (i_t v = 0; v < problem.n_variables; ++v)
    objective += problem.h_obj_coeffs[v] * assignment[v];
  if (!std::isfinite(objective)) return false;
  for (i_t c = 0; c < problem.n_constraints; ++c) {
    f_t activity = 0, correction = 0;
    for (i_t p = problem.offsets[c]; p < problem.offsets[c + 1]; ++p) {
      const f_t term = problem.coefficients[p] * assignment[problem.variables[p]] - correction;
      const f_t next = activity + term;
      correction     = (next - activity) - term;
      activity       = next;
    }
    const f_t lb = problem.cstr_lb[c], ub = problem.cstr_ub[c];
    const f_t tol = get_cstr_tolerance<i_t, f_t>(
      lb, ub, problem.tolerances.absolute_tolerance, problem.tolerances.relative_tolerance);
    if (!std::isfinite(activity) || activity < lb - tol || activity > ub + tol) return false;
  }
  return true;
}

template <typename i_t, typename f_t>
bool verify_cpufj_lns_feasible(const fj_cpu_problem_t<i_t, f_t>& problem,
                               const std::vector<typename type_2<f_t>::type>& bounds,
                               const std::vector<f_t>& assignment)
{
  return verify_cpufj_lns_feasible(problem, bounds, problem.h_var_types, assignment);
}

// Project into the private search domain, then recheck the complete assignment.
template <typename i_t, typename f_t>
bool clamp_cpufj_lns_seed_to_domain(const fj_cpu_problem_t<i_t, f_t>& problem,
                                    const std::vector<typename type_2<f_t>::type>& bounds,
                                    const std::vector<var_t>& types,
                                    std::vector<f_t>& assignment,
                                    bool* changed = nullptr)
{
  if (changed) *changed = false;
  for (i_t v = 0; v < problem.n_variables; ++v) {
    const bool integer = types[v] == var_t::INTEGER;
    const f_t lo       = integer ? std::ceil(get_lower(bounds[v])) : get_lower(bounds[v]);
    const f_t hi       = integer ? std::floor(get_upper(bounds[v])) : get_upper(bounds[v]);
    if (lo > hi) return false;
    const f_t value = std::clamp(integer ? std::round(assignment[v]) : assignment[v], lo, hi);
    if (changed) *changed |= value != assignment[v];
    assignment[v] = value;
  }
  return verify_cpufj_lns_feasible(problem, bounds, types, assignment);
}

// Round integer values and clamp to strict domains, validating before and after adjustment.
template <typename i_t, typename f_t>
bool clamp_and_validate_cpufj_lns_seed(const fj_cpu_problem_t<i_t, f_t>& problem,
                                       const std::vector<typename type_2<f_t>::type>& bounds,
                                       const std::vector<var_t>& types,
                                       std::vector<f_t>& assignment)
{
  return verify_cpufj_lns_feasible(problem, bounds, types, assignment) &&
         clamp_cpufj_lns_seed_to_domain(problem, bounds, types, assignment);
}

template <typename i_t, typename f_t>
bool clamp_and_validate_cpufj_lns_seed(const fj_cpu_problem_t<i_t, f_t>& problem,
                                       const std::vector<typename type_2<f_t>::type>& bounds,
                                       std::vector<f_t>& assignment)
{
  return clamp_and_validate_cpufj_lns_seed(problem, bounds, problem.h_var_types, assignment);
}

// Population seeds belong to the solver model, before CPUFJ caps domains or strengthens types.
template <typename i_t, typename f_t, typename model_validator_t>
bool clamp_and_validate_cpufj_lns_seed(const fj_cpu_climber_t<i_t, f_t>& climber,
                                       std::vector<f_t>& assignment,
                                       const model_validator_t& model_feasible)
{
  if (!model_feasible(assignment)) return false;
  bool changed = false;
  return clamp_cpufj_lns_seed_to_domain(*climber.problem,
                                        climber.h_var_bounds.underlying(),
                                        climber.problem->h_var_types,
                                        assignment,
                                        &changed) &&
         (!changed || model_feasible(assignment));
}
}  // namespace cuopt::mathematical_optimization::mip
