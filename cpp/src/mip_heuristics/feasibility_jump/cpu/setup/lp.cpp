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
bool solve_lp_relaxation(const simplex::user_problem_t<i_t, f_t>& relaxation,
                         double time_limit,
                         std::vector<f_t>& x,
                         double& lp_seconds)
{
  simplex::lp_status_t status = simplex::lp_status_t::UNSET;
  double seconds              = 0;

  // solve_linear_program_advanced, whose status separates a limit -- which leaves a usable vertex
  // behind -- from infeasibility. Guarded on f_t because dual simplex is only built for double.
  if constexpr (std::is_same_v<f_t, double>) {
    simplex_solver_settings_t<i_t, f_t> lp_settings;
    lp_settings.relaxation = true;
    lp_settings.time_limit = time_limit;
    lp_settings.log.log    = false;
    // The portfolio already pins one CPU per lane, and the simplex default is
    // omp_get_max_threads() - 1, which would open a second portfolio inside this lane's worker.
    lp_settings.num_threads = 1;

    const f_t lp_start = tic();
    simplex::lp_solution_t<i_t, f_t> lp_solution(relaxation.num_rows, relaxation.num_cols);
    status  = simplex::solve_linear_program(relaxation, lp_settings, lp_start, lp_solution);
    x       = std::move(lp_solution.x);
    seconds = toc(lp_start);
  }
  lp_seconds += seconds;

  const bool usable =
    status == simplex::lp_status_t::OPTIMAL || status == simplex::lp_status_t::TIME_LIMIT ||
    status == simplex::lp_status_t::ITERATION_LIMIT ||
    status == simplex::lp_status_t::CONCURRENT_LIMIT || status == simplex::lp_status_t::WORK_LIMIT;
  CUOPT_LOG_DEBUG("CPUFJ LP relaxation: %s after %.3fs of %.3fs%s",
                  simplex::lp_status_to_string(status).c_str(),
                  seconds,
                  time_limit,
                  usable ? "" : ", discarded");
  return usable;
}

template <typename i_t, typename f_t>
static simplex::user_problem_t<i_t, f_t> make_lp_distance_problem(
  const simplex::user_problem_t<i_t, f_t>& base,
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  const std::vector<f_t>& rounded)
{
  std::vector<i_t> integer_vars;
  for (i_t var = 0; var < fj_cpu.problem->n_variables; ++var)
    if (is_integer_var<i_t, f_t>(fj_cpu, var)) integer_vars.push_back(var);
  const i_t n_distance = (i_t)integer_vars.size();

  simplex::user_problem_t<i_t, f_t> result(base.handle_ptr);
  result.num_rows = base.num_rows + 2 * n_distance;
  result.num_cols = base.num_cols + n_distance;

  // The model's own objective is dropped: this LP measures distance alone.
  result.objective.assign(result.num_cols, f_t{0});
  for (i_t k = 0; k < n_distance; ++k)
    result.objective[base.num_cols + k] = f_t{1};

  result.lower = base.lower;
  result.upper = base.upper;
  result.lower.resize(result.num_cols, f_t{0});
  result.upper.resize(result.num_cols, std::numeric_limits<f_t>::infinity());

  result.rhs       = base.rhs;
  result.row_sense = base.row_sense;
  result.rhs.reserve(result.num_rows);
  result.row_sense.reserve(result.num_rows);
  for (i_t k = 0; k < n_distance; ++k) {
    result.rhs.push_back(rounded[integer_vars[k]]);
    result.row_sense.push_back('L');
    result.rhs.push_back(-rounded[integer_vars[k]]);
    result.row_sense.push_back('L');
  }
  result.range_rows     = base.range_rows;
  result.range_value    = base.range_value;
  result.num_range_rows = base.num_range_rows;

  const i_t base_nnz = base.A.col_start[base.A.n];
  csc_matrix_t<i_t, f_t> matrix(result.num_rows, result.num_cols, base_nnz + 4 * n_distance);
  i_t out          = 0;
  i_t next_integer = 0;
  for (i_t j = 0; j < base.num_cols; ++j) {
    matrix.col_start[j] = out;
    for (i_t p = base.A.col_start[j]; p < base.A.col_start[j + 1]; ++p) {
      matrix.i[out]   = base.A.i[p];
      matrix.x[out++] = base.A.x[p];
    }
    if (next_integer < n_distance && integer_vars[next_integer] == j) {
      const i_t row   = base.num_rows + 2 * next_integer++;
      matrix.i[out]   = row;
      matrix.x[out++] = f_t{1};
      matrix.i[out]   = row + 1;
      matrix.x[out++] = f_t{-1};
    }
  }
  for (i_t k = 0; k < n_distance; ++k) {
    matrix.col_start[base.num_cols + k] = out;
    const i_t row                       = base.num_rows + 2 * k;
    matrix.i[out]                       = row;
    matrix.x[out++]                     = f_t{-1};
    matrix.i[out]                       = row + 1;
    matrix.x[out++]                     = f_t{-1};
  }
  matrix.col_start[result.num_cols] = out;
  cuopt_assert(out == base_nnz + 4 * n_distance, "distance problem nonzero count mismatch");
  result.A = std::move(matrix);
  return result;
}

template <typename i_t, typename f_t>
static simplex::user_problem_t<i_t, f_t> make_fixed_integer_lp(
  const simplex::user_problem_t<i_t, f_t>& base,
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  const std::vector<f_t>& rounded)
{
  simplex::user_problem_t<i_t, f_t> fixed(base.handle_ptr);
  fixed.num_rows = base.num_rows;
  fixed.num_cols = base.num_cols;
  fixed.objective.assign(base.num_cols, f_t{0});
  fixed.rhs            = base.rhs;
  fixed.row_sense      = base.row_sense;
  fixed.range_rows     = base.range_rows;
  fixed.range_value    = base.range_value;
  fixed.num_range_rows = base.num_range_rows;
  fixed.A              = base.A;
  fixed.lower          = base.lower;
  fixed.upper          = base.upper;
  for (i_t var = 0; var < (i_t)rounded.size(); ++var) {
    if (!is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
    fixed.lower[var] = fixed.upper[var] = rounded[var];
  }
  return fixed;
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
  phase_timer_t timer(fj_cpu.t_lp_start);

  simplex::user_problem_t<i_t, f_t> base = *fj_cpu.problem->host_lp;
  if (fj_cpu.lp_start_feasibility_objective)
    std::fill(base.objective.begin(), base.objective.end(), f_t{0});

  // A zero-objective relaxation returns the same arbitrary basic solution in every feasibility-LP
  // lane. On a nonnegative integer equality master, all of those lanes then floor the same vertex
  // and spend most of the budget repairing the same residual. Positive random costs keep the LP
  // bounded and feasible while selecting independent vertices for the portfolio. This is gated by
  // the certificate above; ordinary objective-bearing and mixed-sign models are unchanged.
  if (monotone_integer_equalities) {
    cuopt::pcgenerator_t objective_rng(fj_cpu.settings.seed ^ 0xd1b54a32d192ed03ULL);
    for (i_t var = 0; var < fj_cpu.problem->n_variables; ++var)
      base.objective[var] = f_t{1} + (f_t)objective_rng.next_double();
  }

  const auto started    = std::chrono::steady_clock::now();
  const i_t n_variables = fj_cpu.problem->n_variables;

  std::vector<f_t> rounded;
  std::vector<f_t> selected;
  // Keep the least-infeasible rounded LP projection as the FJ starting point.
  // total_violations sums negative excesses, so the greatest value is the least infeasible.
  f_t selected_violation = -std::numeric_limits<f_t>::infinity();

  const int32_t projections = fj_cpu.use_deep_lp_pump ? 100 : fj_cpu.hp.lp_pump_projections;
  for (int32_t projection = 0; projection < projections; ++projection) {
    const double remaining =
      budget - std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (remaining <= 0) break;

    // Projection 0 is the plain relaxation; the rest chase the previous rounding.
    const auto distance    = projection == 0 ? simplex::user_problem_t<i_t, f_t>(base.handle_ptr)
                                             : make_lp_distance_problem(base, fj_cpu, rounded);
    const auto& relaxation = projection == 0 ? base : distance;

    std::vector<f_t> x;
    if (!solve_lp_relaxation(relaxation, remaining, x, fj_cpu.t_lp_relaxation)) break;
    // convert_user_problem appends slacks, so the model's own variables are the leading columns.
    if ((i_t)x.size() < n_variables) break;

    rounded.resize(n_variables);
    cuopt::pcgenerator_t rng(fj_cpu.settings.seed + 0x9e3779b9ULL * (uint64_t)projection);
    bool valid = true;
    for (i_t var = 0; var < n_variables && valid; ++var) {
      const auto bounds = fj_cpu.h_var_bounds[var].get();
      const f_t lower   = get_lower(bounds);
      const f_t upper   = get_upper(bounds);
      f_t value         = std::clamp(x[var], lower, upper);
      if (!std::isfinite(value)) {
        valid = false;
        break;
      }
      if (is_integer_var<i_t, f_t>(fj_cpu, var)) {
        if (monotone_integer_equalities) {
          // Every coefficient is nonnegative, hence this preserves every equality's upper side.
          // Clamp once more because an LP value can sit a few ulps below an integral lower bound.
          value = std::clamp(std::floor(value), std::ceil(lower), std::floor(upper));
        } else {
          // Rounded up with probability equal to the fractional part, so successive projections of
          // the same point explore different corners.
          const f_t fraction = value - std::floor(value);
          value              = rng.next_double() < fraction ? std::ceil(value) : std::floor(value);
        }
        // A variable with no integral value inside its bounds cannot form a valid start without
        // breaking the engine's integrality invariant.
        valid = value >= lower && value <= upper;
      }
      rounded[var] = value;
    }
    if (!valid) break;

    std::vector<f_t> candidate = rounded;
    if (fj_cpu.use_deep_lp_pump) {
      const double repair_budget =
        budget - std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      if (repair_budget > 0.01) {
        auto fixed = make_fixed_integer_lp(base, fj_cpu, rounded);
        std::vector<f_t> repaired;
        if (solve_lp_relaxation(fixed, repair_budget, repaired, fj_cpu.t_lp_relaxation) &&
            (i_t)repaired.size() >= n_variables) {
          for (i_t var = 0; var < n_variables; ++var)
            if (!is_integer_var<i_t, f_t>(fj_cpu, var)) candidate[var] = repaired[var];
        }
      }
    }
    std::copy(candidate.begin(), candidate.end(), fj_cpu.h_assignment.begin());
    recompute_lhs(fj_cpu);
    cuopt_assert(fj_cpu.total_violations <= f_t{0}, "total_violations should be nonpositive");
    if (fj_cpu.total_violations > selected_violation) {
      selected_violation = fj_cpu.total_violations;
      selected           = candidate;
    }

    // The rounded point can already be integral-feasible. It never passed through apply_move, so
    // the incumbent is recorded here through the same contract that path uses.
    if (fj_cpu.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(fj_cpu)) {
      std::copy(candidate.begin(), candidate.end(), fj_cpu.h_best_assignment.begin());
      fj_cpu.h_best_objective =
        fj_cpu.h_incumbent_objective - fj_cpu.settings.parameters.breakthrough_move_epsilon;
      fj_cpu.feasible_found = true;
      report_cpu_incumbent(fj_cpu);
      return;
    }
  }

  if (selected.empty()) return;
  std::copy(selected.begin(), selected.end(), fj_cpu.h_assignment.begin());
  std::copy(selected.begin(), selected.end(), fj_cpu.h_best_assignment.begin());
  recompute_lhs(fj_cpu);
  cuopt_func_call(audit_assignment_bounds(fj_cpu, "lp pump"));
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

  bool has_continuous = false;
  for (i_t var = 0; var < n; ++var)
    has_continuous |= !is_integer_var<i_t, f_t>(fj_cpu, var);
  if (!has_continuous) return false;

  phase_timer_t timer(fj_cpu.t_lp_start);
  simplex::user_problem_t<i_t, f_t> relaxation = *fj_cpu.problem->host_lp;
  if ((i_t)relaxation.lower.size() < n || (i_t)relaxation.upper.size() < n) return false;

  for (i_t var = 0; var < n; ++var) {
    if (!is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
    relaxation.lower[var] = relaxation.upper[var] = std::round(incumbent[var]);
  }

  std::vector<f_t> x;
  if (!solve_lp_relaxation(relaxation, budget_s, x, fj_cpu.t_lp_relaxation) || (i_t)x.size() < n)
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
