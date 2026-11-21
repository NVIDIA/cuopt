/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>
#include <mip_heuristics/feasibility_jump/cpu/setup/structure.hpp>
#include <mip_heuristics/utils.hpp>

#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Bound strengthening can make a valid disjunction's M coefficient small. Certify
// that its inactive (binary-zero) row is already implied by the continuous bounds.
template <typename i_t, typename f_t>
bool cpufj_lns_inactive_row_redundant(const fj_cpu_climber_t<i_t, f_t>& climber,
                                      i_t row,
                                      i_t gate,
                                      bool lower_row,
                                      std::vector<f_t>& coefficients,
                                      std::vector<f_t>& endpoints)
{
  const auto& problem = *climber.problem;
  coefficients.clear();
  endpoints.clear();
  for (i_t q = problem.offsets[row]; q < problem.offsets[row + 1]; ++q) {
    const i_t variable    = problem.variables[q];
    const f_t coefficient = problem.coefficients[q];
    if (variable == gate || coefficient == f_t{0}) continue;
    const auto bounds = climber.h_var_bounds[variable];
    const f_t endpoint =
      lower_row == (coefficient > f_t{0}) ? get_lower(bounds) : get_upper(bounds);
    if (!std::isfinite(endpoint) || !std::isfinite(coefficient * endpoint)) return false;
    coefficients.push_back(coefficient);
    endpoints.push_back(endpoint);
  }
  const f_t activity = compensated_dot2(coefficients.data(), endpoints.data(), coefficients.size());
  const f_t lower = problem.cstr_lb[row], upper = problem.cstr_ub[row];
  const f_t tolerance = get_cstr_tolerance<i_t, f_t>(
    lower, upper, problem.tolerances.absolute_tolerance, problem.tolerances.relative_tolerance);
  return std::isfinite(activity) &&
         (lower_row ? activity >= lower - tolerance : activity <= upper + tolerance);
}

// LNS accepts any positive number of exact-one groups, arbitrary continuous scopes,
// and additional side constraints. Every group must select a recognized continuous
// region; inactive rows may be certified using bounds and the solver's tolerances.
// The shared model is read-only.
template <typename i_t, typename f_t>
bool has_cpufj_lns_geometry(const fj_cpu_climber_t<i_t, f_t>& climber)
{
  if (climber.problem == nullptr) return false;
  const auto& problem = *climber.problem;
  const i_t groups    = static_cast<i_t>(problem.card_cardinalities.size());
  if (climber.n_binary_vars <= 0 || climber.n_integer_vars != 0 || groups == 0 ||
      problem.h_objective_vars.empty())
    return false;
  for (i_t variable : problem.h_objective_vars)
    if (problem.h_var_types[variable] != var_t::CONTINUOUS) return false;
  std::vector<f_t> coefficients, endpoints;
  cpufj_geometry_requirements_t requirements;
  requirements.min_groups              = 1;
  requirements.require_gate_per_group  = true;
  requirements.allow_other_binary_rows = true;
  return has_cpufj_disjunctive_geometry(
    climber, requirements, [&](i_t row, i_t gate, bool lower_row) {
      return cpufj_lns_inactive_row_redundant(
        climber, row, gate, lower_row, coefficients, endpoints);
    });
}

template <typename i_t, typename f_t>
bool configure_cpufj_lns_geometry(fj_cpu_climber_t<i_t, f_t>& climber)
{
  if (!has_cpufj_lns_geometry(climber)) return false;
  // Lane seven selects existing scalar/cardinality repair without constructive
  // starts. Keep the incumbent as the start and use local continuous perturbations.
  const i_t perturb_interval = climber.perturb_interval;
  climber.low_latency        = true;
  apply_lane_diversification(climber, 7, climber.settings.seed);
  // Short LNS bursts restart the stall counter. Keep their existing repair
  // cadence instead of the long-running portfolio lane's randomized interval.
  climber.perturb_interval               = perturb_interval;
  climber.use_fixed_charge_network_start = false;
  climber.continuous_perturb_fraction    = f_t{0.2};
  climber.objective_directed_perturb     = true;
  return true;
}

}  // namespace cuopt::mathematical_optimization::mip
