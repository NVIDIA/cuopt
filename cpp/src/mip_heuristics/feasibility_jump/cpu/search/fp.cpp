/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fp.hpp"
#include "../audit.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "api.hpp"

#include <optional>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
bool solve_lp_relaxation(const simplex::user_problem_t<i_t, f_t>& relaxation,
                         double time_limit,
                         std::vector<f_t>& x,
                         double& lp_seconds)
{
  simplex::lp_status_t status = simplex::lp_status_t::UNSET;
  double seconds              = 0;

  simplex_solver_settings_t<i_t, f_t> lp_settings;
  lp_settings.relaxation = true;
  lp_settings.time_limit = time_limit;
  lp_settings.log.log    = false;

  const f_t lp_start = tic();
  simplex::lp_solution_t<i_t, f_t> lp_solution(relaxation.num_rows, relaxation.num_cols);
  status  = simplex::solve_linear_program(relaxation, lp_settings, lp_start, lp_solution);
  x       = std::move(lp_solution.x);
  seconds = toc(lp_start);
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
bool solve_lp_with_fixed_variables(const simplex::user_problem_t<i_t, f_t>& problem,
                                   const std::vector<i_t>& fixed_variables,
                                   const std::vector<f_t>& fixed_values,
                                   double time_limit,
                                   std::vector<f_t>& assignment,
                                   double& solve_seconds)
{
  cuopt_assert(fixed_variables.size() == fixed_values.size(),
               "fixed variable and value counts differ");
  auto fixed = problem;
  for (size_t k = 0; k < fixed_variables.size(); ++k) {
    const i_t var = fixed_variables[k];
    cuopt_assert(var >= 0 && var < (i_t)fixed.lower.size() && var < (i_t)fixed.upper.size(),
                 "var out of bounds");
    fixed.lower[var] = fixed.upper[var] = fixed_values[k];
  }
  return solve_lp_relaxation(fixed, time_limit, assignment, solve_seconds);
}

template <typename i_t, typename f_t>
static simplex::user_problem_t<i_t, f_t> make_lp_distance_problem(
  const simplex::user_problem_t<i_t, f_t>& base,
  const std::vector<i_t>& integer_vars,
  const std::vector<f_t>& rounded)
{
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
void run_cpu_feasibility_pump(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                              const simplex::user_problem_t<i_t, f_t>& base,
                              double budget,
                              bool monotone_integer_equalities)
{
  const double started  = tic();
  const i_t n_variables = fj_cpu.problem->n_variables;

  std::vector<i_t> integer_vars;
  for (i_t var = 0; var < n_variables; ++var)
    if (is_integer_var<i_t, f_t>(fj_cpu, var)) integer_vars.push_back(var);

  std::optional<simplex::user_problem_t<i_t, f_t>> repair_problem;
  std::vector<f_t> fixed_values;
  if (fj_cpu.use_deep_lp_pump) {
    repair_problem.emplace(base);
    std::fill(repair_problem->objective.begin(), repair_problem->objective.end(), f_t{0});
    fixed_values.resize(integer_vars.size());
  }

  std::vector<f_t> rounded;
  std::vector<f_t> selected;
  // Keep the least-infeasible rounded LP projection as the FJ starting point.
  // total_violations sums negative excesses, so the greatest value is the least infeasible.
  f_t selected_violation = -std::numeric_limits<f_t>::infinity();

  const int32_t projections = fj_cpu.use_deep_lp_pump ? 100 : fj_cpu.hp.lp_pump_projections;
  for (int32_t projection = 0; projection < projections; ++projection) {
    const double remaining = budget - toc(started);
    if (remaining <= 0) break;

    // Projection 0 is the plain relaxation; the rest chase the previous rounding.
    const auto distance    = projection == 0 ? simplex::user_problem_t<i_t, f_t>(base.handle_ptr)
                                             : make_lp_distance_problem(base, integer_vars, rounded);
    const auto& relaxation = projection == 0 ? base : distance;

    std::vector<f_t> x;
    if (!solve_lp_relaxation(relaxation, remaining, x, fj_cpu.stats.t_lp_relaxation)) break;
    // convert_user_problem appends slacks, so the model's own variables are the leading columns.
    if ((i_t)x.size() < n_variables) break;

    rounded.resize(n_variables);
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
          value = fj_cpu.rng.next_double() < fraction ? std::ceil(value) : std::floor(value);
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
      const double repair_budget = budget - toc(started);
      if (repair_budget > 0.01) {
        for (size_t k = 0; k < integer_vars.size(); ++k)
          fixed_values[k] = rounded[integer_vars[k]];
        std::vector<f_t> repaired;
        if (solve_lp_with_fixed_variables(*repair_problem,
                                          integer_vars,
                                          fixed_values,
                                          repair_budget,
                                          repaired,
                                          fj_cpu.stats.t_lp_relaxation) &&
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

#if MIP_INSTANTIATE_FLOAT
template bool solve_lp_relaxation<int, float>(const simplex::user_problem_t<int, float>&,
                                              double,
                                              std::vector<float>&,
                                              double&);
template bool solve_lp_with_fixed_variables<int, float>(const simplex::user_problem_t<int, float>&,
                                                        const std::vector<int>&,
                                                        const std::vector<float>&,
                                                        double,
                                                        std::vector<float>&,
                                                        double&);
template void run_cpu_feasibility_pump<int, float>(fj_cpu_climber_t<int, float>&,
                                                   const simplex::user_problem_t<int, float>&,
                                                   double,
                                                   bool);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool solve_lp_relaxation<int, double>(const simplex::user_problem_t<int, double>&,
                                               double,
                                               std::vector<double>&,
                                               double&);
template bool solve_lp_with_fixed_variables<int, double>(
  const simplex::user_problem_t<int, double>&,
  const std::vector<int>&,
  const std::vector<double>&,
  double,
  std::vector<double>&,
  double&);
template void run_cpu_feasibility_pump<int, double>(fj_cpu_climber_t<int, double>&,
                                                    const simplex::user_problem_t<int, double>&,
                                                    double,
                                                    bool);
#endif

}  // namespace cuopt::mathematical_optimization::mip
