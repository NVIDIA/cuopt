/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <syncstream>
#include <utility>
#include <vector>

struct row_sum_t {
  double activity;
  double positive_activity;
  double error_bound;
};

template <typename value_t, typename index_t, typename solution_t>
row_sum_t row_sum_fp64(const value_t* values,
                       const index_t* indices,
                       int64_t begin,
                       int64_t end,
                       const solution_t* solution)
{
  double activity = 0.0;
  double positive = 0.0;
  double abs_sum  = 0.0;
  for (int64_t entry = begin; entry < end; ++entry) {
    const double term = (double)values[entry] * (double)solution[indices[entry]];
    activity += term;
    abs_sum += std::fabs(term);
    if (term > 0.0) positive += term;
  }
  const double width = (double)(end - begin);
  return {activity, positive, (width + 1.0) * std::numeric_limits<double>::epsilon() * abs_sum};
}

struct row_sum_wide_t {
  _Float128 activity;
  _Float128 positive_activity;
};

template <typename value_t, typename index_t, typename solution_t>
row_sum_wide_t row_sum_exact(const value_t* values,
                             const index_t* indices,
                             int64_t begin,
                             int64_t end,
                             const solution_t* solution)
{
  _Float128 activity = 0;
  _Float128 positive = 0;
  for (int64_t entry = begin; entry < end; ++entry) {
    const _Float128 term = (_Float128)values[entry] * (_Float128)solution[indices[entry]];
    activity += term;
    if (term > 0) positive += term;
  }
  return {activity, positive};
}

inline double scaled_tolerance(double absolute_tolerance, double first, double second)
{
  return absolute_tolerance * std::max({1.0, std::fabs(first), std::fabs(second)});
}

inline std::pair<double, double> scaled_row_limits(double absolute_tolerance,
                                                   double positive_activity,
                                                   double lower_bound,
                                                   double upper_bound)
{
  return {lower_bound - scaled_tolerance(absolute_tolerance, positive_activity, lower_bound),
          upper_bound + scaled_tolerance(absolute_tolerance, positive_activity, upper_bound)};
}

inline std::pair<double, double> scaled_bound_limits(double absolute_tolerance,
                                                     double value,
                                                     double lower_bound,
                                                     double upper_bound)
{
  return {lower_bound - scaled_tolerance(absolute_tolerance, lower_bound, value),
          upper_bound + scaled_tolerance(absolute_tolerance, upper_bound, value)};
}

inline double bound_excess(double value,
                           double lower_bound,
                           double upper_bound,
                           double absolute_tolerance)
{
  const auto limits = scaled_bound_limits(absolute_tolerance, value, lower_bound, upper_bound);
  return std::max(std::max(limits.first - value, value - limits.second), 0.0);
}

struct row_verdict_t {
  double activity;
  double excess;
  double raw_excess;
  double lower_limit;
  double upper_limit;
  bool escalated;
};

template <typename value_t, typename index_t, typename solution_t, typename limits_t>
row_verdict_t check_row(const value_t* values,
                        const index_t* indices,
                        int64_t begin,
                        int64_t end,
                        const solution_t* solution,
                        double lower_bound,
                        double upper_bound,
                        limits_t limits)
{
  const row_sum_t sum = row_sum_fp64(values, indices, begin, end, solution);
  auto bound          = limits(sum.positive_activity);
  const bool satisfied =
    sum.activity - sum.error_bound >= bound.first && sum.activity + sum.error_bound <= bound.second;
  const bool undecided =
    sum.activity + sum.error_bound >= bound.first && sum.activity - sum.error_bound <= bound.second;

  if (satisfied || !undecided) {
    const double excess =
      std::max(std::max(bound.first - sum.activity, sum.activity - bound.second), 0.0);
    const double raw_excess =
      std::max(std::max(lower_bound - sum.activity, sum.activity - upper_bound), 0.0);
    return {sum.activity, excess, raw_excess, bound.first, bound.second, false};
  }

  const row_sum_wide_t wide = row_sum_exact(values, indices, begin, end, solution);
  bound                     = limits((double)wide.positive_activity);
  const _Float128 zero      = 0;
  const double excess       = (double)std::max(
    std::max((_Float128)bound.first - wide.activity, wide.activity - (_Float128)bound.second),
    zero);
  const double raw_excess = (double)std::max(
    std::max((_Float128)lower_bound - wide.activity, wide.activity - (_Float128)upper_bound), zero);
  return {(double)wide.activity, excess, raw_excess, bound.first, bound.second, true};
}

static bool verify_solution(
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& problem,
  const std::vector<double>& solution,
  double reported_objective,
  const cuopt::mathematical_optimization::mip_solver_settings_t<int, double>::tolerances_t&
    tolerances,
  size_t incumbent)
{
  const auto& values                 = problem.get_constraint_matrix_values();
  const auto& indices                = problem.get_constraint_matrix_indices();
  const auto& offsets                = problem.get_constraint_matrix_offsets();
  const auto& row_lower_bounds       = problem.get_constraint_lower_bounds();
  const auto& row_upper_bounds       = problem.get_constraint_upper_bounds();
  const auto& variable_lower_bounds  = problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds  = problem.get_variable_upper_bounds();
  const auto& variable_types         = problem.get_variable_types();
  const auto& objective_coefficients = problem.get_objective_coefficients();

  if (variable_lower_bounds.size() != variable_types.size() ||
      variable_upper_bounds.size() != variable_types.size() ||
      objective_coefficients.size() != variable_types.size()) {
    std::osyncstream(std::cerr) << "Incumbent " << incumbent
                                << " cannot be checked against malformed variable data\n";
    return false;
  }
  if (solution.size() != variable_types.size()) {
    std::osyncstream(std::cerr) << "Incumbent " << incumbent << " has " << solution.size()
                                << " variables; expected " << variable_types.size() << "\n";
    return false;
  }

  _Float128 objective = 0;
  for (size_t variable = 0; variable < solution.size(); ++variable) {
    const double value = solution[variable];
    if (!std::isfinite(value)) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                  << " variable " << variable << " is not finite: " << value
                                  << "\n";
      return false;
    }
    const auto bound_limits            = scaled_bound_limits(tolerances.absolute_tolerance,
                                                  value,
                                                  variable_lower_bounds[variable],
                                                  variable_upper_bounds[variable]);
    const double lower_bound_tolerance = variable_lower_bounds[variable] - bound_limits.first;
    const double upper_bound_tolerance = bound_limits.second - variable_upper_bounds[variable];
    if (value < bound_limits.first || value > bound_limits.second) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                  << " variable " << variable << " violates bounds: value=" << value
                                  << " lb=" << variable_lower_bounds[variable]
                                  << " ub=" << variable_upper_bounds[variable]
                                  << " lower_tolerance=" << lower_bound_tolerance
                                  << " upper_tolerance=" << upper_bound_tolerance << "\n";
      return false;
    }
    if (variable_types[variable] == 'I' && value != std::round(value)) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                  << " variable " << variable
                                  << " is not exactly integral: value=" << value
                                  << " residual=" << std::abs(value - std::round(value)) << "\n";
      return false;
    }
    objective += (_Float128)objective_coefficients[variable] * (_Float128)solution[variable];
  }

  if (row_upper_bounds.size() != row_lower_bounds.size() ||
      offsets.size() != row_lower_bounds.size() + 1) {
    std::osyncstream(std::cerr) << "Incumbent " << incumbent
                                << " cannot be checked against malformed row data\n";
    return false;
  }
  for (size_t row = 0; row < row_lower_bounds.size(); ++row) {
    const double lower_bound = row_lower_bounds[row];
    const double upper_bound = row_upper_bounds[row];

    // fp64 first. _Float128 is soft-float on x86-64; only rows whose rounding-error
    // interval meets a bound pay for it. Bound is (nnz+1)*eps*abs_sum.
    const auto verdict =
      check_row(values.data(),
                indices.data(),
                (int64_t)offsets[row],
                (int64_t)offsets[row + 1],
                solution.data(),
                lower_bound,
                upper_bound,
                [&](double positive_activity) {
                  return scaled_row_limits(
                    tolerances.absolute_tolerance, positive_activity, lower_bound, upper_bound);
                });
    if (verdict.excess > 0.0) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent << " row "
                                  << row << " violates bounds: activity=" << verdict.activity
                                  << " lb=" << lower_bound << " ub=" << upper_bound
                                  << " lower_tolerance=" << (lower_bound - verdict.lower_limit)
                                  << " upper_tolerance=" << (verdict.upper_limit - upper_bound)
                                  << "\n";
      return false;
    }
  }

  const double recomputed_objective =
    problem.get_objective_scaling_factor() * ((double)objective + problem.get_objective_offset());
  const double objective_difference = std::abs(recomputed_objective - reported_objective);
  const double objective_scale =
    std::max(std::abs(recomputed_objective), std::abs(reported_objective));
  const double relative_objective_tolerance = objective_scale * tolerances.relative_tolerance;
  if (!std::isfinite(recomputed_objective) || !std::isfinite(reported_objective) ||
      (objective_difference > tolerances.absolute_tolerance &&
       objective_difference > relative_objective_tolerance)) {
    std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                << " objective mismatch: reported=" << reported_objective
                                << " recomputed=" << recomputed_objective
                                << " difference=" << objective_difference
                                << " absolute_tolerance=" << tolerances.absolute_tolerance
                                << " relative_threshold=" << relative_objective_tolerance << "\n";
    return false;
  }
  return true;
}

template <typename f_t>
double combine_finite_abs_bounds(f_t lower, f_t upper)
{
  f_t val = f_t(0);
  if (isfinite(upper)) { val = raft::max<f_t>(val, raft::abs(upper)); }
  if (isfinite(lower)) { val = raft::max<f_t>(val, raft::abs(lower)); }
  return val;
}

template <typename f_t>
struct violation {
  violation() = default;
  violation(f_t* _scalar) {}
  __device__ __host__ f_t operator()(f_t value, f_t lower, f_t upper)
  {
    if (value < lower) {
      return lower - value;
    } else if (value > upper) {
      return value - upper;
    }
    return f_t(0);
  }
};

bool test_constraint_and_variable_sanity(
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& op_problem,
  const std::vector<double>& primal_vars,
  double abs_tol,
  double rel_tol,
  double int_tol = 1e-5)
{
  const std::vector<double>& values                  = op_problem.get_constraint_matrix_values();
  const std::vector<int>& indices                    = op_problem.get_constraint_matrix_indices();
  const std::vector<int>& offsets                    = op_problem.get_constraint_matrix_offsets();
  const std::vector<double>& constraint_lower_bounds = op_problem.get_constraint_lower_bounds();
  const std::vector<double>& constraint_upper_bounds = op_problem.get_constraint_upper_bounds();
  const std::vector<double>& variable_lower_bounds   = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds   = op_problem.get_variable_upper_bounds();
  const std::vector<char>& variable_types            = op_problem.get_variable_types();
  std::vector<double> residual(constraint_lower_bounds.size(), 0.0);
  std::vector<double> viol(constraint_lower_bounds.size(), 0.0);

  // CSR SpMV
  for (size_t i = 0; i < offsets.size() - 1; ++i) {
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      residual[i] += values[j] * primal_vars[indices[j]];
    }
  }

  auto functor = violation<double>{};

  bool feasible = true;
  // Compute violation to lower/upper bound
  for (size_t i = 0; i < residual.size(); ++i) {
    double tolerance = abs_tol + combine_finite_abs_bounds<double>(constraint_lower_bounds[i],
                                                                   constraint_upper_bounds[i]) *
                                   rel_tol;
    double viol = functor(residual[i], constraint_lower_bounds[i], constraint_upper_bounds[i]);
    if (viol > tolerance) {
      feasible = false;
      CUOPT_LOG_ERROR(
        "feasibility violation %f at cstr %d is more than total tolerance %f lb %f ub %f \n",
        viol,
        i,
        tolerance,
        constraint_lower_bounds[i],
        constraint_upper_bounds[i]);
    }
  }
  bool feasible_variables = true;
  for (size_t i = 0; i < primal_vars.size(); ++i) {
    if (variable_types[i] == 'I' && abs(primal_vars[i] - round(primal_vars[i])) > int_tol) {
      feasible_variables = false;
    }
    // Not always strictly true because we apply variable bound clamping on the scaled problem
    // After unscaling it, the variables might not respect exactly (this adding an epsilon)
    if (!(primal_vars[i] >= variable_lower_bounds[i] - int_tol &&
          primal_vars[i] <= variable_upper_bounds[i] + int_tol)) {
      CUOPT_LOG_ERROR("error at bounds var %d lb %f ub %f val %f\n",
                      i,
                      variable_lower_bounds[i],
                      variable_upper_bounds[i],
                      primal_vars[i]);
      feasible_variables = false;
    }
  }
  if (!feasible || !feasible_variables) { CUOPT_LOG_ERROR("Initial solution is infeasible"); }
  return feasible_variables;
}
