/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "audit.hpp"
#include "internal.hpp"
#include "problem.hpp"

namespace cuopt::mathematical_optimization::mip {

namespace {
constexpr double fj_audit_rel_slack           = 1e-9;
constexpr double fj_audit_abs_floor           = 1e-6;
constexpr int32_t fj_audit_row_terms_printed  = 64;
constexpr bool fj_audit_each_row_update       = false;
constexpr bool fj_audit_each_objective_update = false;
}  // namespace

template <typename i_t, typename f_t>
void audit_assignment_bounds(fj_cpu_climber_t<i_t, f_t>& fj_cpu, const char* site)
{
  for (i_t var = 0; var < fj_cpu.problem->n_variables; ++var) {
    const f_t val    = fj_cpu.h_assignment[var];
    auto bounds      = fj_cpu.h_var_bounds[var].get();
    const bool inbox = fj_cpu.check_variable_within_bounds(var, val);
    const bool integral =
      var_t::INTEGER != fj_cpu.problem->h_var_types[var] || fj_cpu.problem->is_integer(val);
    if (inbox && integral) continue;

    CUOPT_LOG_DEBUG("%sCPUFJ %s left var %d at %.17g outside [%.17g, %.17g], integer %d",
                    fj_cpu.log_prefix.c_str(),
                    site,
                    (int)var,
                    val,
                    get_lower(bounds),
                    get_upper(bounds),
                    var_t::INTEGER == fj_cpu.problem->h_var_types[var]);
    cuopt_assert(false, "assignment left the variable bounds");
    return;
  }
}

template <typename i_t, typename f_t>
f_t fresh_row_slack(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t row, const f_t* assignment)
{
  const f_t activity = compensated_dot2_csr(fj_cpu.h_offsets.data(),
                                            fj_cpu.h_variables.data(),
                                            fj_cpu.h_coefficients.data(),
                                            assignment,
                                            row);
  return (f_t)fj_cpu.h_bound[row] - activity;
}

template <typename i_t, typename f_t>
void report_row_divergence(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                           i_t cstr_idx,
                           const f_t* assignment,
                           const char* site)
{
  auto [row_begin, row_end] = fj_cpu.range_for_row(cstr_idx);
  const f_t sumcomp         = fj_cpu.h_slack_sumcomp[cstr_idx];

  CUOPT_LOG_DEBUG(
    "%sCPUFJ %s row %d state: iteration %d, width %d, slack sumcomp "
    "%.17g, bound %.17g, refresh period %d, recomputes total %lld periodic %lld bigval "
    "%lld perturb %lld restart %lld",
    fj_cpu.log_prefix.c_str(),
    site,
    (int)cstr_idx,
    (int)fj_cpu.iterations,
    (int)(row_end - row_begin),
    sumcomp,
    fj_cpu.h_bound[cstr_idx],
    (int)fj_cpu.lhs_refresh_period_used,
    (long long)fj_cpu.n_lhs_recompute_total,
    (long long)fj_cpu.n_lhs_recompute_periodic,
    (long long)fj_cpu.n_lhs_recompute_bigval,
    (long long)fj_cpu.n_lhs_recompute_perturb,
    (long long)fj_cpu.n_lhs_recompute_restart);

  i_t unreachable = 0;
  i_t mismatched  = 0;
  for (i_t p = row_begin; p < row_end; ++p) {
    const i_t var   = fj_cpu.h_variables[p];
    const f_t coeff = fj_cpu.h_coefficients[p];
    const f_t val   = assignment[var];

    // apply_move reaches this row only through the variable's slice of the transpose.
    const auto [rev_begin, rev_end] = fj_cpu.range_for_variable(var);
    bool reachable                  = false;
    f_t rev_coeff                   = 0;
    for (i_t q = rev_begin; q < rev_end; ++q) {
      if (fj_cpu.h_reverse_constraints[q] != cstr_idx) continue;
      reachable = true;
      rev_coeff = fj_cpu.h_reverse_coefficients[q];
      break;
    }

    if (!reachable) {
      ++unreachable;
    } else if (rev_coeff != coeff) {
      ++mismatched;
    }

    if (p - row_begin >= (i_t)fj_audit_row_terms_printed) continue;
    CUOPT_LOG_DEBUG(
      "%sCPUFJ %s row %d term %d: var %d integer %d degree %d, coeff %.17g x %.17g "
      "product %.17g, reachable %d transpose coeff %.17g",
      fj_cpu.log_prefix.c_str(),
      site,
      (int)cstr_idx,
      (int)(p - row_begin),
      (int)var,
      var_t::INTEGER == fj_cpu.problem->h_var_types[var],
      (int)(rev_end - rev_begin),
      coeff,
      val,
      coeff * val,
      reachable,
      rev_coeff);
  }

  CUOPT_LOG_DEBUG(
    "%sCPUFJ %s row %d structure: %d of %d variables cannot reach it through the "
    "transpose, %d carry a different transpose coefficient%s",
    fj_cpu.log_prefix.c_str(),
    site,
    (int)cstr_idx,
    (int)unreachable,
    (int)(row_end - row_begin),
    (int)mismatched,
    row_end - row_begin > (i_t)fj_audit_row_terms_printed ? " (terms truncated)" : "");
}

template <typename i_t, typename f_t>
void audit_objective_update(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx, f_t old_val, f_t delta, f_t obj_old, f_t obj_y)
{
  if (!fj_audit_each_objective_update) return;
  const f_t* const assignment = fj_cpu.h_assignment.data();

  const f_t fresh =
    compensated_dot2(fj_cpu.problem->h_obj_coeffs.data(), assignment, fj_cpu.problem->n_variables);
  const f_t gap   = std::fabs(fj_cpu.h_incumbent_objective - fresh);
  const f_t slack = (f_t)fj_audit_abs_floor + (f_t)fj_audit_rel_slack * std::fabs(fresh);
  if (!(gap > slack)) return;

  const f_t coeff   = fj_cpu.problem->h_obj_coeffs[var_idx];
  const f_t product = coeff * delta;
  // Debug messages are flushed before the abort below.
  CUOPT_LOG_DEBUG(
    "%sCPUFJ objective update: carried %.17g vs c'x %.17g, gap %.17g over slack %.17g. "
    "iteration %d, var %d moved %.17g -> %.17g by delta %.17g, objective coeff %.17g, "
    "product %.17g whose ulp is %.17g, obj_old %.17g, obj_y %.17g, sumcomp %.17g",
    fj_cpu.log_prefix.c_str(),
    fj_cpu.h_incumbent_objective,
    fresh,
    gap,
    slack,
    (int)fj_cpu.iterations,
    (int)var_idx,
    old_val,
    old_val + delta,
    delta,
    coeff,
    product,
    std::numeric_limits<f_t>::epsilon() * std::fabs(product),
    obj_old,
    obj_y,
    fj_cpu.h_objective_sumcomp);
  cuopt_assert(false, "h_incumbent_objective disagrees with c'x after a move");
}

template <typename i_t, typename f_t>
void audit_row_updates(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx, f_t old_val, f_t delta, i_t begin, i_t end)
{
  if (!fj_audit_each_row_update) return;
  const f_t* const assignment = fj_cpu.h_assignment.data();

  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.n_rows; ++cstr_idx) {
    const f_t carried = fj_cpu.row_state()[cstr_idx].slack + fj_cpu.h_slack_sumcomp[cstr_idx];
    const f_t fresh   = fresh_row_slack<i_t, f_t>(fj_cpu, cstr_idx, assignment);
    const f_t gap     = std::fabs(carried - fresh);
    // The verdict, not the gap. A row far from its bound may carry a value an ulp off the fresh one
    // with no consequence, and above |slack| ~ 4e9 one ulp already exceeds the row tolerance, so
    // any absolute threshold fires there on correct arithmetic.
    if ((carried < -fj_cpu.row_tolerance) == (fresh < -fj_cpu.row_tolerance)) continue;

    f_t incidence_coeff = 0;
    bool touched        = false;
    for (i_t i = begin; i < end; ++i) {
      if (fj_cpu.h_reverse_constraints[i] != cstr_idx) continue;
      touched         = true;
      incidence_coeff = fj_cpu.h_reverse_coefficients[i];
      break;
    }

    // Debug messages are flushed before the abort below.
    CUOPT_LOG_DEBUG(
      "%sCPUFJ row update row %d: carried slack %.17g says violated %d, fresh %.17g says "
      "%d, differ by %.17g against tol %.17g. iteration %d, var %d moved %.17g -> %.17g "
      "by delta %.17g, row in this move's support %d with coeff %.17g, row bound %.17g, "
      "slack sumcomp %.17g, width %d",
      fj_cpu.log_prefix.c_str(),
      (int)cstr_idx,
      carried,
      carried < -fj_cpu.row_tolerance,
      fresh,
      fresh < -fj_cpu.row_tolerance,
      gap,
      fj_cpu.row_tolerance,
      (int)fj_cpu.iterations,
      (int)var_idx,
      old_val,
      old_val + delta,
      delta,
      touched,
      incidence_coeff,
      fj_cpu.h_bound[cstr_idx],
      fj_cpu.h_slack_sumcomp[cstr_idx],
      (int)(fj_cpu.h_offsets[cstr_idx + 1] - fj_cpu.h_offsets[cstr_idx]));
    report_row_divergence<i_t, f_t>(fj_cpu, cstr_idx, assignment, "row update");
    cuopt_assert(false, "carried slack disagrees with a fresh sum after a move");
    return;
  }
}

template <typename i_t, typename f_t>
void audit_incremental_state(fj_cpu_climber_t<i_t, f_t>& fj_cpu, const char* site)
{
  const f_t* const assignment = fj_cpu.h_assignment.data();
  const f_t tol               = fj_cpu.row_tolerance;

  f_t fresh_total = 0;
  // The total re-derived from the carried slacks rather than from the model. Stored against this
  // isolates the total's own accounting; this against fresh_total isolates the row slacks.
  f_t carried_total = 0;

  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.n_rows; ++cstr_idx) {
    const f_t fresh   = fresh_row_slack<i_t, f_t>(fj_cpu, cstr_idx, assignment);
    const f_t cost    = fresh < f_t{0} ? fresh : f_t{0};
    const f_t carried = fj_cpu.row_state()[cstr_idx].slack + fj_cpu.h_slack_sumcomp[cstr_idx];

    const bool truly_violated = fresh < -tol;
    if (truly_violated) { fresh_total += cost; }

    const bool carried_violated = fj_cpu.violated_constraints.contains(cstr_idx);
    if (carried_violated) { carried_total += carried; }
    if (carried_violated == truly_violated) continue;

    // Debug messages are flushed before the abort below.
    CUOPT_LOG_DEBUG(
      "%sCPUFJ %s row %d: integral %d, carried violated %d actual %d, carried "
      "slack %.17g vs fresh %.17g differ by %.17g, bound %.17g, tol %.17g",
      fj_cpu.log_prefix.c_str(),
      site,
      (int)cstr_idx,
      fj_cpu.h_row_is_integral[cstr_idx],
      carried_violated,
      truly_violated,
      carried,
      fresh,
      std::fabs(carried - fresh),
      fj_cpu.h_bound[cstr_idx],
      tol);
    report_row_divergence<i_t, f_t>(fj_cpu, cstr_idx, assignment, site);
    cuopt_assert(false, "violated set disagrees with a fresh slack");
    return;
  }

  const f_t fresh_obj =
    compensated_dot2(fj_cpu.problem->h_obj_coeffs.data(), assignment, fj_cpu.problem->n_variables);
  const f_t obj_gap   = std::fabs(fj_cpu.h_incumbent_objective - fresh_obj);
  const f_t obj_slack = (f_t)fj_audit_abs_floor + (f_t)fj_audit_rel_slack * std::fabs(fresh_obj);
  if (obj_gap > obj_slack) {
    CUOPT_LOG_DEBUG(
      "%sCPUFJ %s h_incumbent_objective %.17g vs c'x %.17g, gap %.17g over slack %.17g, "
      "sumcomp %.17g",
      fj_cpu.log_prefix.c_str(),
      site,
      fj_cpu.h_incumbent_objective,
      fresh_obj,
      obj_gap,
      obj_slack,
      fj_cpu.h_objective_sumcomp);
    cuopt_assert(false, "h_incumbent_objective left c'x behind");
  }
}

template <typename i_t, typename f_t>
bool check_variable_feasibility(fj_cpu_climber_t<i_t, f_t>& fj_cpu, bool check_integer)
{
  for (i_t var_idx = 0; var_idx < fj_cpu.problem->n_variables; var_idx += 1) {
    auto val      = fj_cpu.h_assignment[var_idx];
    bool feasible = check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val);

    if (!feasible) return false;
    if (check_integer && is_integer_var<i_t, f_t>(fj_cpu, var_idx) &&
        !fj_cpu.problem->is_integer(fj_cpu.h_assignment[var_idx]))
      return false;
  }
  return true;
}

template <typename i_t, typename f_t>
void sanity_checks(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  cuopt_assert((i_t)fj_cpu.h_row_state.size() == fj_cpu.n_rows,
               "row state does not cover the search rows");
  cuopt_assert((i_t)fj_cpu.h_slack_sumcomp.size() == fj_cpu.n_rows,
               "slack compensation does not cover the search rows");

  // Check that each variable is within its bounds
  for (i_t var_idx = 0; var_idx < fj_cpu.problem->n_variables; ++var_idx) {
    f_t val = fj_cpu.h_assignment[var_idx];
    cuopt_assert(fj_cpu.check_variable_within_bounds(var_idx, val), "Variable is out of bounds");
  }

  // Check that each violated constraint is actually violated and not present in
  // satisfied_constraints
  for (const auto& cstr_idx : fj_cpu.violated_constraints) {
    cuopt_assert(!fj_cpu.satisfied_constraints.contains(cstr_idx),
                 "Violated constraint also in satisfied_constraints");
    cuopt_assert(
      fj_cpu.row_state()[cstr_idx].slack + fj_cpu.h_slack_sumcomp[cstr_idx] < -fj_cpu.row_tolerance,
      "Constraint in violated_constraints is not actually violated");
  }

  // Check that each satisfied constraint is actually satisfied and not present in
  // violated_constraints
  for (const auto& cstr_idx : fj_cpu.satisfied_constraints) {
    cuopt_assert(!fj_cpu.violated_constraints.contains(cstr_idx),
                 "Satisfied constraint also in violated_constraints");
    cuopt_assert(!(fj_cpu.row_state()[cstr_idx].slack + fj_cpu.h_slack_sumcomp[cstr_idx] <
                   -fj_cpu.row_tolerance),
                 "Constraint in satisfied_constraints is actually violated");
  }

  // Check that each constraint is in exactly one of violated_constraints or satisfied_constraints
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.n_rows; ++cstr_idx) {
    bool in_viol = fj_cpu.violated_constraints.contains(cstr_idx);
    bool in_sat  = fj_cpu.satisfied_constraints.contains(cstr_idx);
    cuopt_assert(
      in_viol != in_sat,
      "Constraint must be in exactly one of violated_constraints or satisfied_constraints");

    cuopt_assert(fj_cpu.row_state()[cstr_idx].weight >= 0, "Weights should be positive or zero");
  }
  cuopt_assert(fj_cpu.h_objective_weight >= 0, "Objective weight should be positive or zero");
  cuopt_assert(fj_cpu.objective_weight_floor >= 0,
               "Objective weight floor should be positive or zero");
}

#if MIP_INSTANTIATE_FLOAT
template void audit_assignment_bounds<int, float>(fj_cpu_climber_t<int, float>&, const char*);
template float fresh_row_slack<int, float>(fj_cpu_climber_t<int, float>&, int, const float*);
template void audit_objective_update<int, float>(
  fj_cpu_climber_t<int, float>&, int, float, float, float, float);
template void audit_row_updates<int, float>(
  fj_cpu_climber_t<int, float>&, int, float, float, int, int);
template void audit_incremental_state<int, float>(fj_cpu_climber_t<int, float>&, const char*);
template bool check_variable_feasibility<int, float>(fj_cpu_climber_t<int, float>&, bool);
template void sanity_checks<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void audit_assignment_bounds<int, double>(fj_cpu_climber_t<int, double>&, const char*);
template double fresh_row_slack<int, double>(fj_cpu_climber_t<int, double>&, int, const double*);
template void audit_objective_update<int, double>(
  fj_cpu_climber_t<int, double>&, int, double, double, double, double);
template void audit_row_updates<int, double>(
  fj_cpu_climber_t<int, double>&, int, double, double, int, int);
template void audit_incremental_state<int, double>(fj_cpu_climber_t<int, double>&, const char*);
template bool check_variable_feasibility<int, double>(fj_cpu_climber_t<int, double>&, bool);
template void sanity_checks<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
