/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "bounds.hpp"
#include "../audit.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void cap_integer_domains(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t n_variables)
{
  cuopt_assert(fj_cpu.h_best_assignment.size() == static_cast<size_t>(n_variables),
               "best assignment size mismatch");

  for (i_t var = 0; var < n_variables; ++var) {
    if (var_t::INTEGER != fj_cpu.problem->h_var_types[var]) continue;

    const auto bounds = fj_cpu.h_var_bounds[var].get();
    const f_t lower   = std::max(get_lower(bounds), (f_t)-fj_cpu.hp.integer_domain_limit);
    const f_t upper   = std::min(get_upper(bounds), (f_t)fj_cpu.hp.integer_domain_limit);
    if (lower > upper) continue;
    if (lower == get_lower(bounds) && upper == get_upper(bounds)) continue;

    // Both assignments, since the from-template path inherits h_lhs instead of recomputing it.
    fj_cpu.h_var_bounds[var]      = typename type_2<f_t>::type{lower, upper};
    fj_cpu.h_assignment[var]      = std::clamp((f_t)fj_cpu.h_assignment[var], lower, upper);
    fj_cpu.h_best_assignment[var] = std::clamp((f_t)fj_cpu.h_best_assignment[var], lower, upper);
  }
}

template <typename i_t, typename f_t>
void clamp_start_magnitude(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t n_variables)
{
  cuopt_assert(fj_cpu.h_assignment.size() == static_cast<size_t>(n_variables),
               "assignment size mismatch");
  cuopt_assert(fj_cpu.h_best_assignment.size() == static_cast<size_t>(n_variables),
               "best assignment size mismatch");

  for (i_t var = 0; var < n_variables; ++var) {
    const auto bounds = fj_cpu.h_var_bounds[var].get();
    const f_t lower   = std::max(get_lower(bounds), (f_t)-fj_cpu.hp.start_magnitude_limit);
    const f_t upper   = std::min(get_upper(bounds), (f_t)fj_cpu.hp.start_magnitude_limit);
    if (lower > upper) continue;

    fj_cpu.h_assignment[var]      = std::clamp((f_t)fj_cpu.h_assignment[var], lower, upper);
    fj_cpu.h_best_assignment[var] = std::clamp((f_t)fj_cpu.h_best_assignment[var], lower, upper);
  }
}

template <typename i_t, typename f_t>
bool tighten_lower_bound(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                         std::vector<f_t>& lower,
                         const std::vector<f_t>& upper,
                         i_t var,
                         f_t limit,
                         f_t commit_threshold)
{
  if (!std::isfinite(limit)) return false;
  if (is_integer_var<i_t, f_t>(fj_cpu, var))
    limit = std::ceil(limit - fj_cpu.problem->tolerances.integrality_tolerance);
  if (limit > upper[var]) return false;
  if (limit <= lower[var] + commit_threshold) return false;
  lower[var] = limit;
  return true;
}

template <typename i_t, typename f_t>
bool tighten_upper_bound(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                         const std::vector<f_t>& lower,
                         std::vector<f_t>& upper,
                         i_t var,
                         f_t limit,
                         f_t commit_threshold)
{
  if (!std::isfinite(limit)) return false;
  if (is_integer_var<i_t, f_t>(fj_cpu, var))
    limit = std::floor(limit + fj_cpu.problem->tolerances.integrality_tolerance);
  if (limit < lower[var]) return false;
  if (limit >= upper[var] - commit_threshold) return false;
  upper[var] = limit;
  return true;
}

// a light bounds propagation phase that runs much faster than the full scale presolve
// really helps on some instances.
template <typename i_t, typename f_t>
void apply_bound_propagation(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (!fj_cpu.use_bound_prop) return;
  CPUFJ_NVTX_RANGE("CPUFJ::apply_bound_propagation");
  phase_timer_t timer(fj_cpu.t_bound_prop);

  const i_t n_variables   = fj_cpu.problem->n_variables;
  const i_t n_constraints = fj_cpu.problem->n_constraints;
  const f_t commit =
    (f_t)fj_cpu.hp.bound_prop_commit_scale * fj_cpu.problem->tolerances.absolute_tolerance;

  std::vector<f_t> lower(n_variables);
  std::vector<f_t> upper(n_variables);
  for (i_t var = 0; var < n_variables; ++var) {
    auto bounds = fj_cpu.h_var_bounds[var].get();
    lower[var]  = get_lower(bounds);
    upper[var]  = get_upper(bounds);
  }

  bool changed = true;
  int32_t pass = 0;
  for (; changed && pass < fj_cpu.hp.bound_prop_rounds; ++pass) {
    changed = false;
    for (i_t row = 0; row < n_constraints; ++row) {
      const f_t row_lb  = fj_cpu.problem->cstr_lb[row];
      const f_t row_ub  = fj_cpu.problem->cstr_ub[row];
      const bool has_lb = std::isfinite(row_lb);
      const bool has_ub = std::isfinite(row_ub);
      if (!has_lb && !has_ub) continue;

      const i_t begin = fj_cpu.problem->offsets[row];
      const i_t end   = fj_cpu.problem->offsets[row + 1];

      f_t min_activity = 0;
      f_t max_activity = 0;
      bool finite_min  = true;
      bool finite_max  = true;
      for (i_t p = begin; p < end; ++p) {
        const f_t coeff = fj_cpu.problem->coefficients[p];
        if (coeff == f_t{0}) continue;
        const i_t var   = fj_cpu.problem->variables[p];
        const f_t min_x = coeff > 0 ? lower[var] : upper[var];
        const f_t max_x = coeff > 0 ? upper[var] : lower[var];
        finite_min &= std::isfinite(min_x);
        finite_max &= std::isfinite(max_x);
      }
      const auto indices = thrust::make_counting_iterator(begin);
      if (finite_min) {
        const auto min_values = thrust::make_transform_iterator(indices, [&](i_t p) {
          const f_t coeff = fj_cpu.problem->coefficients[p];
          if (coeff == f_t{0}) return f_t{0};
          const i_t var = fj_cpu.problem->variables[p];
          return coeff > 0 ? lower[var] : upper[var];
        });
        min_activity =
          compensated_dot2(fj_cpu.problem->coefficients.data() + begin, min_values, end - begin);
      }
      if (finite_max) {
        const auto max_values = thrust::make_transform_iterator(indices, [&](i_t p) {
          const f_t coeff = fj_cpu.problem->coefficients[p];
          if (coeff == f_t{0}) return f_t{0};
          const i_t var = fj_cpu.problem->variables[p];
          return coeff > 0 ? upper[var] : lower[var];
        });
        max_activity =
          compensated_dot2(fj_cpu.problem->coefficients.data() + begin, max_values, end - begin);
      }

      const bool from_row_ub = finite_min && has_ub;
      const bool from_row_lb = finite_max && has_lb;
      if (!from_row_ub && !from_row_lb) continue;

      // The activities are not refreshed as the loop below narrows the row's own variables, and a
      // stale bound is the looser one, so a deduction taken against it is the weaker one.
      for (i_t p = begin; p < end; ++p) {
        const f_t coeff = fj_cpu.problem->coefficients[p];
        if (coeff == f_t{0}) continue;
        const i_t var = fj_cpu.problem->variables[p];

        if (from_row_ub) {
          const f_t rest  = min_activity - coeff * (coeff > 0 ? lower[var] : upper[var]);
          const f_t limit = (row_ub - rest) / coeff;
          changed |= coeff > 0 ? tighten_upper_bound(fj_cpu, lower, upper, var, limit, commit)
                               : tighten_lower_bound(fj_cpu, lower, upper, var, limit, commit);
        }
        if (from_row_lb) {
          const f_t rest  = max_activity - coeff * (coeff > 0 ? upper[var] : lower[var]);
          const f_t limit = (row_lb - rest) / coeff;
          changed |= coeff > 0 ? tighten_lower_bound(fj_cpu, lower, upper, var, limit, commit)
                               : tighten_upper_bound(fj_cpu, lower, upper, var, limit, commit);
        }
      }
    }
  }

  fj_cpu.h_binary_indices.clear();
  fj_cpu.n_binary_vars           = 0;
  fj_cpu.n_integer_vars          = 0;
  [[maybe_unused]] i_t tightened = 0;
  bool clamped                   = false;
  for (i_t var = 0; var < n_variables; ++var) {
    auto bounds = fj_cpu.h_var_bounds[var].get();
    cuopt_assert(!(lower[var] < get_lower(bounds)), "propagation widened a lower bound");
    cuopt_assert(!(upper[var] > get_upper(bounds)), "propagation widened an upper bound");
    cuopt_assert(!(lower[var] > upper[var]), "propagation emptied a domain");
    const bool moved = lower[var] != get_lower(bounds) || upper[var] != get_upper(bounds);

    // Same rule as problem_t::compute_binary_var_table, fixed binaries included: a domain narrowed
    // to a point is no longer binary.
    const bool integer = is_integer_var<i_t, f_t>(fj_cpu, var);
    const bool binary  = integer && fj_cpu.problem->integer_equal(lower[var], (f_t)0) &&
                        fj_cpu.problem->integer_equal(upper[var], (f_t)1);
    fj_cpu.h_is_binary_variable[var] = binary;
    if (binary) {
      fj_cpu.h_binary_indices.push_back(var);
      ++fj_cpu.n_binary_vars;
    } else if (integer) {
      ++fj_cpu.n_integer_vars;
    }
    if (!moved) continue;

    ++tightened;
    fj_cpu.h_var_bounds[var] = typename type_2<f_t>::type{lower[var], upper[var]};

    const f_t value         = fj_cpu.h_assignment[var];
    const f_t clamped_value = std::clamp(value, lower[var], upper[var]);
    if (clamped_value != value) {
      cuopt_assert(!integer || fj_cpu.problem->is_integer(clamped_value),
                   "bound clamp broke integrality");
      fj_cpu.h_assignment[var] = clamped_value;
      clamped                  = true;
    }
    fj_cpu.h_best_assignment[var] =
      std::clamp((f_t)fj_cpu.h_best_assignment[var], lower[var], upper[var]);
  }

  if (clamped) recompute_lhs(fj_cpu);
  cuopt_func_call(audit_assignment_bounds(fj_cpu, "bound prop"));

  CUOPT_LOG_DEBUG("%sCPUFJ bound prop: %d passes, %d domains tightened, %d binary of %d integer",
                  fj_cpu.log_prefix.c_str(),
                  pass,
                  tightened,
                  fj_cpu.n_binary_vars,
                  fj_cpu.n_binary_vars + fj_cpu.n_integer_vars);
}

template <typename i_t, typename f_t>
void apply_lock_weighted_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.problem->nnz > fj_cpu.hp.start_nnz_limit) return;

  const i_t n_variables = fj_cpu.problem->n_variables;
  for (i_t var_idx = 0; var_idx < n_variables; ++var_idx) {
    const f_t lb = get_lower(fj_cpu.h_var_bounds[var_idx].get());
    const f_t ub = get_upper(fj_cpu.h_var_bounds[var_idx].get());
    if (!std::isfinite(lb) || !std::isfinite(ub) || lb >= ub) continue;

    i_t lock_up      = 0;
    i_t lock_down    = 0;
    const auto range = model_range_for_var<i_t, f_t>(fj_cpu, var_idx);
    for (i_t i = range.first; i < range.second; ++i) {
      const f_t coeff    = fj_cpu.problem->reverse_coefficients[i];
      const i_t cstr_idx = fj_cpu.problem->reverse_constraints[i];
      const bool has_lb  = std::isfinite((f_t)fj_cpu.problem->cstr_lb[cstr_idx]);
      const bool has_ub  = std::isfinite((f_t)fj_cpu.problem->cstr_ub[cstr_idx]);
      if (coeff > 0) {
        lock_up += has_ub;
        lock_down += has_lb;
      } else if (coeff < 0) {
        lock_up += has_lb;
        lock_down += has_ub;
      }
    }

    f_t new_val = lock_up <= lock_down ? ub : lb;
    if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) new_val = std::round(new_val);
    fj_cpu.h_assignment[var_idx] = new_val;
  }

  recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

template <typename i_t, typename f_t>
void apply_ambiguous_lock_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.problem->nnz > fj_cpu.hp.start_nnz_limit) return;

  std::mt19937 rng((uint32_t)fj_cpu.settings.seed ^ 0x9e3779b9u);
  std::bernoulli_distribution flip(0.5);
  for (i_t var = 0; var < fj_cpu.problem->n_variables; ++var) {
    const f_t lower = get_lower(fj_cpu.h_var_bounds[var].get());
    const f_t upper = get_upper(fj_cpu.h_var_bounds[var].get());
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper) continue;

    i_t up = 0, down = 0;
    const auto [begin, end] = model_range_for_var<i_t, f_t>(fj_cpu, var);
    for (i_t p = begin; p < end; ++p) {
      const f_t coeff = fj_cpu.problem->reverse_coefficients[p];
      const i_t row   = fj_cpu.problem->reverse_constraints[p];
      const bool lb   = std::isfinite((f_t)fj_cpu.problem->cstr_lb[row]);
      const bool ub   = std::isfinite((f_t)fj_cpu.problem->cstr_ub[row]);
      if (coeff > 0) {
        up += ub;
        down += lb;
      } else if (coeff < 0) {
        up += lb;
        down += ub;
      }
    }

    const bool ambiguous = std::abs(up - down) <= 1;
    const bool choose_up = up < down || (ambiguous && flip(rng));
    f_t value            = choose_up ? upper : lower;
    if (is_integer_var<i_t, f_t>(fj_cpu, var)) value = std::round(value);
    fj_cpu.h_assignment[var] = value;
  }
  recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

#if MIP_INSTANTIATE_FLOAT
template void cap_integer_domains<int, float>(fj_cpu_climber_t<int, float>&, int);
template void clamp_start_magnitude<int, float>(fj_cpu_climber_t<int, float>&, int);
template void apply_bound_propagation<int, float>(fj_cpu_climber_t<int, float>&);
template void apply_lock_weighted_start<int, float>(fj_cpu_climber_t<int, float>&);
template void apply_ambiguous_lock_start<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void cap_integer_domains<int, double>(fj_cpu_climber_t<int, double>&, int);
template void clamp_start_magnitude<int, double>(fj_cpu_climber_t<int, double>&, int);
template void apply_bound_propagation<int, double>(fj_cpu_climber_t<int, double>&);
template void apply_lock_weighted_start<int, double>(fj_cpu_climber_t<int, double>&);
template void apply_ambiguous_lock_start<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
