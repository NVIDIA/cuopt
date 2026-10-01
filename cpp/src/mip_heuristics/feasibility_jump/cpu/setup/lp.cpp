/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "lp.hpp"
#include "../audit.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"
#include "../search/fp.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void eliminate_slacks(const lp_problem_t<i_t, f_t>& problem,
                      i_t n_structural,
                      csr_matrix_t<i_t, f_t>& csr_A,
                      std::vector<f_t>& row_lower,
                      std::vector<f_t>& row_upper)
{
  cuopt_assert(csr_A.m == problem.num_rows, "row count mismatch");
  cuopt_assert(csr_A.n == problem.num_cols, "column count mismatch");
  cuopt_assert(n_structural > 0, "no structural columns");
  cuopt_assert(n_structural < problem.num_cols, "no slacks to eliminate");
  cuopt_assert(problem.num_cols - n_structural <= problem.num_rows, "more slacks than rows");

  row_lower = problem.rhs;
  row_upper = problem.rhs;

  std::vector<char> row_has_slack(problem.num_rows, 0);
  for (i_t j = n_structural; j < problem.num_cols; ++j) {
    cuopt_assert(problem.A.col_length(j) == 1, "slack column is not a singleton");

    const i_t entry = problem.A.col_start[j];
    const i_t row   = problem.A.i[entry];
    const f_t alpha = problem.A.x[entry];
    cuopt_assert(std::abs(alpha) == f_t{1}, "slack coefficient is not +/-1");
    cuopt_assert(!row_has_slack[row], "row has more than one slack");
    row_has_slack[row] = 1;

    const f_t scaled_lower = alpha * problem.lower[j];
    const f_t scaled_upper = alpha * problem.upper[j];
    row_lower[row]         = problem.rhs[row] - std::max(scaled_lower, scaled_upper);
    row_upper[row]         = problem.rhs[row] - std::min(scaled_lower, scaled_upper);
    cuopt_assert(std::isfinite(row_lower[row]) || std::isfinite(row_upper[row]),
                 "eliminated row is free on both sides");
    cuopt_assert(row_lower[row] <= row_upper[row], "eliminated row has crossed bounds");
  }

  i_t out = 0;
  for (i_t row = 0; row < csr_A.m; ++row) {
    const i_t row_start  = csr_A.row_start[row];
    const i_t row_end    = csr_A.row_start[row + 1];
    csr_A.row_start[row] = out;
    for (i_t p = row_start; p < row_end; ++p) {
      if (csr_A.j[p] >= n_structural) { continue; }
      csr_A.j[out] = csr_A.j[p];
      csr_A.x[out] = csr_A.x[p];
      ++out;
    }
  }
  cuopt_assert(out == csr_A.row_start[csr_A.m] - static_cast<i_t>(problem.num_cols - n_structural),
               "slack elimination removed the wrong number of entries");

  csr_A.row_start[csr_A.m] = out;
  csr_A.j.resize(out);
  csr_A.x.resize(out);
  csr_A.nz_max = out;
  csr_A.n      = n_structural;
}

template <typename i_t, typename f_t>
void apply_lp_rounded_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu, f_t lane_time_limit)
{
  if (!fj_cpu.use_lp_start || !fj_cpu.problem->host_lp) return;
  if (fj_cpu.problem->nnz > fj_cpu.hp.lp_start_nnz_limit) return;

  // In a nonnegative all-integer equality system, flooring a feasible relaxation is a particularly
  // useful FJ start: it cannot overshoot any equality, so the residual search only has to fill
  // deficits instead of simultaneously undoing stochastic round-ups.  Give one feasibility-LP
  // persona enough time to obtain a real basic solution on this broad structural class.  The
  // ordinary LP personas retain their tiny opportunistic budget, and deep pumps retain theirs.
  bool monotone_integer_equalities =
    fj_cpu.lp_start_feasibility_objective && fj_cpu.problem->equality_fraction == 1.0 &&
    fj_cpu.n_integer_vars + fj_cpu.n_binary_vars == fj_cpu.problem->n_variables;
  if (monotone_integer_equalities) {
    for (i_t var = 0; var < fj_cpu.problem->n_variables; ++var) {
      const auto bounds = fj_cpu.h_var_bounds[var].get();
      if (get_lower(bounds) < f_t{0} || fj_cpu.problem->h_obj_coeffs[var] != f_t{0}) {
        monotone_integer_equalities = false;
        break;
      }
    }
  }
  if (monotone_integer_equalities) {
    for (i_t p = 0; p < fj_cpu.problem->nnz; ++p) {
      if (fj_cpu.problem->coefficients[p] < f_t{0}) {
        monotone_integer_equalities = false;
        break;
      }
    }
  }

  const double budget = fj_cpu.use_deep_lp_pump ? std::min(3.5, 0.7 * (double)lane_time_limit)
                        : monotone_integer_equalities
                          ? std::min(2.0, 0.45 * (double)lane_time_limit)
                          : std::min(fj_cpu.hp.lp_pump_max_budget_s,
                                     fj_cpu.hp.lp_pump_budget_share * (double)lane_time_limit);
  if (budget <= 0) return;

  CPUFJ_NVTX_RANGE("CPUFJ::apply_lp_rounded_start");
  phase_timer_t timer(fj_cpu.stats.t_lp_start);

  simplex::user_problem_t<i_t, f_t> base = *fj_cpu.problem->host_lp;
  if (fj_cpu.lp_start_feasibility_objective)
    std::fill(base.objective.begin(), base.objective.end(), f_t{0});

  // A zero-objective relaxation returns the same arbitrary basic solution in every feasibility-LP
  // lane. On a nonnegative integer equality master, all of those lanes then floor the same vertex
  // and spend most of the budget repairing the same residual. Positive random costs keep the LP
  // bounded and feasible while selecting independent vertices for the portfolio. This is gated by
  // the certificate above; ordinary objective-bearing and mixed-sign models are unchanged.
  if (monotone_integer_equalities) {
    for (i_t var = 0; var < fj_cpu.problem->n_variables; ++var)
      base.objective[var] = fj_cpu.rng.uniform(f_t{1}, f_t{2});
  }

  run_cpu_feasibility_pump(fj_cpu, base, budget, monotone_integer_equalities);
}

template <typename i_t, typename f_t>
bool apply_lp_polish(fj_cpu_climber_t<i_t, f_t>& fj_cpu, double budget_s)
{
  if (!fj_cpu.problem->host_lp || fj_cpu.problem->nnz > fj_cpu.hp.lp_polish_nnz_limit ||
      budget_s < fj_cpu.hp.lp_polish_min_budget_s)
    return false;

  const i_t n = fj_cpu.problem->n_variables;
  std::vector<f_t> incumbent(fj_cpu.h_best_assignment.begin(),
                             fj_cpu.h_best_assignment.begin() + n);
  if (fj_cpu.shared_incumbent)
    fj_cpu.shared_incumbent->adopt(fj_cpu.h_best_objective + f_t{1}, incumbent);

  std::vector<i_t> fixed_variables;
  std::vector<f_t> fixed_values;
  for (i_t var = 0; var < n; ++var) {
    if (!is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
    fixed_variables.push_back(var);
    fixed_values.push_back(std::round(incumbent[var]));
  }
  if ((i_t)fixed_variables.size() == n) return false;

  phase_timer_t timer(fj_cpu.stats.t_lp_start);
  const auto& relaxation = *fj_cpu.problem->host_lp;
  if ((i_t)relaxation.lower.size() < n || (i_t)relaxation.upper.size() < n) return false;

  std::vector<f_t> x;
  if (!solve_lp_with_fixed_variables(
        relaxation, fixed_variables, fixed_values, budget_s, x, fj_cpu.stats.t_lp_relaxation) ||
      (i_t)x.size() < n)
    return false;

  std::vector<f_t> completion(n);
  for (i_t var = 0; var < n; ++var) {
    const auto bounds = fj_cpu.h_var_bounds[var].get();
    const f_t value   = is_integer_var<i_t, f_t>(fj_cpu, var) ? std::round(incumbent[var]) : x[var];
    if (!std::isfinite(value)) return false;
    completion[var] = std::clamp(value, get_lower(bounds), get_upper(bounds));
  }

  std::copy(completion.begin(), completion.end(), fj_cpu.h_assignment.begin());
  recompute_slack(fj_cpu);
  const bool improved = fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective &&
                        fj_cpu.violated_constraints.empty() &&
                        check_variable_feasibility<i_t, f_t>(fj_cpu);
  if (!improved) {
    std::copy(fj_cpu.h_best_assignment.begin(),
              fj_cpu.h_best_assignment.begin() + n,
              fj_cpu.h_assignment.begin());
    recompute_slack(fj_cpu);
    return false;
  }

  std::copy(completion.begin(), completion.end(), fj_cpu.h_best_assignment.begin());
  fj_cpu.h_best_objective =
    fj_cpu.h_incumbent_objective - fj_cpu.settings.parameters.breakthrough_move_epsilon;
  fj_cpu.iterations_since_best = 0;
  fj_cpu.perturb_streak        = 0;
  fj_cpu.feasible_found        = true;
  report_cpu_incumbent(fj_cpu);
  share_cpu_incumbent(fj_cpu);
  return true;
}

#if MIP_INSTANTIATE_FLOAT
template void eliminate_slacks<int, float>(const lp_problem_t<int, float>&,
                                           int,
                                           csr_matrix_t<int, float>&,
                                           std::vector<float>&,
                                           std::vector<float>&);
template void apply_lp_rounded_start<int, float>(fj_cpu_climber_t<int, float>&, float);
template bool apply_lp_polish<int, float>(fj_cpu_climber_t<int, float>&, double);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void eliminate_slacks<int, double>(const lp_problem_t<int, double>&,
                                            int,
                                            csr_matrix_t<int, double>&,
                                            std::vector<double>&,
                                            std::vector<double>&);
template void apply_lp_rounded_start<int, double>(fj_cpu_climber_t<int, double>&, double);
template bool apply_lp_polish<int, double>(fj_cpu_climber_t<int, double>&, double);
#endif

}  // namespace cuopt::mathematical_optimization::mip
