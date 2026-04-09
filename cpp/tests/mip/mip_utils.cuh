/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <algorithm>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver.cuh>
#include <mps_parser/parser.hpp>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::test {

static void test_variable_bounds(
  const cuopt::mps_parser::mps_data_model_t<int, double>& problem,
  const rmm::device_uvector<double>& solution,
  const cuopt::linear_programming::mip_solver_settings_t<int, double> settings)
{
  const double* lower_bound_ptr = problem.get_variable_lower_bounds().data();
  const double* upper_bound_ptr = problem.get_variable_upper_bounds().data();
  auto host_assignment          = cuopt::host_copy(solution, solution.stream());
  double* assignment_ptr        = host_assignment.data();
  cuopt_assert(host_assignment.size() == problem.get_variable_lower_bounds().size(), "");
  cuopt_assert(host_assignment.size() == problem.get_variable_upper_bounds().size(), "");
  std::vector<int> indices(host_assignment.size());
  std::iota(indices.begin(), indices.end(), 0);
  bool result = std::all_of(indices.begin(), indices.end(), [=](int idx) {
    bool res = true;
    if (lower_bound_ptr != nullptr) {
      res = res && (assignment_ptr[idx] >=
                    lower_bound_ptr[idx] - settings.tolerances.integrality_tolerance);
    }
    if (upper_bound_ptr != nullptr) {
      res = res && (assignment_ptr[idx] <=
                    upper_bound_ptr[idx] + settings.tolerances.integrality_tolerance);
    }
    return res;
  });
  EXPECT_TRUE(result);
}

static void test_variable_bounds(
  const cuopt::mps_parser::mps_data_model_t<int, double>& problem,
  const std::vector<double>& solution,
  const cuopt::linear_programming::mip_solver_settings_t<int, double> settings)
{
  const double* lower_bound_ptr = problem.get_variable_lower_bounds().data();
  const double* upper_bound_ptr = problem.get_variable_upper_bounds().data();
  const double* assignment_ptr  = solution.data();
  cuopt_assert(solution.size() == problem.get_variable_lower_bounds().size(), "");
  cuopt_assert(solution.size() == problem.get_variable_upper_bounds().size(), "");
  std::vector<int> indices(solution.size());
  std::iota(indices.begin(), indices.end(), 0);
  bool result = std::all_of(indices.begin(), indices.end(), [=](int idx) {
    bool res = true;
    if (lower_bound_ptr != nullptr) {
      res = res && (assignment_ptr[idx] >=
                    lower_bound_ptr[idx] - settings.tolerances.integrality_tolerance);
    }
    if (upper_bound_ptr != nullptr) {
      res = res && (assignment_ptr[idx] <=
                    upper_bound_ptr[idx] + settings.tolerances.integrality_tolerance);
    }
    return res;
  });
  EXPECT_TRUE(result);
}

template <typename f_t>
static double combine_finite_abs_bounds(f_t lower, f_t upper)
{
  f_t val = f_t(0);
  if (isfinite(upper)) { val = raft::max<f_t>(val, raft::abs(upper)); }
  if (isfinite(lower)) { val = raft::max<f_t>(val, raft::abs(lower)); }
  return val;
}

template <typename f_t>
struct violation {
  violation() {}
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

static void test_constraint_sanity_per_row(
  const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem,
  const rmm::device_uvector<double>& solution,
  double abs_tolerance,
  double rel_tolerance)
{
  const std::vector<double>& values                  = op_problem.get_constraint_matrix_values();
  const std::vector<int>& indices                    = op_problem.get_constraint_matrix_indices();
  const std::vector<int>& offsets                    = op_problem.get_constraint_matrix_offsets();
  const std::vector<double>& constraint_lower_bounds = op_problem.get_constraint_lower_bounds();
  const std::vector<double>& constraint_upper_bounds = op_problem.get_constraint_upper_bounds();
  const std::vector<double>& variable_lower_bounds   = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds   = op_problem.get_variable_upper_bounds();
  std::vector<double> residual(constraint_lower_bounds.size(), 0.0);
  auto h_solution = cuopt::host_copy(solution, solution.stream());
  // CSR SpMV
  for (size_t i = 0; i < offsets.size() - 1; ++i) {
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      residual[i] += values[j] * h_solution[indices[j]];
    }
  }

  auto functor = violation<double>{};

  // Compute violation to lower/upper bound
  for (size_t i = 0; i < residual.size(); ++i) {
    double tolerance = abs_tolerance + combine_finite_abs_bounds<double>(
                                         constraint_lower_bounds[i], constraint_upper_bounds[i]) *
                                         rel_tolerance;
    double viol = functor(residual[i], constraint_lower_bounds[i], constraint_upper_bounds[i]);
    EXPECT_LE(viol, tolerance);
  }
}

static void test_constraint_sanity_per_row(
  const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem,
  const std::vector<double>& solution,
  double abs_tolerance,
  double rel_tolerance)
{
  const std::vector<double>& values                  = op_problem.get_constraint_matrix_values();
  const std::vector<int>& indices                    = op_problem.get_constraint_matrix_indices();
  const std::vector<int>& offsets                    = op_problem.get_constraint_matrix_offsets();
  const std::vector<double>& constraint_lower_bounds = op_problem.get_constraint_lower_bounds();
  const std::vector<double>& constraint_upper_bounds = op_problem.get_constraint_upper_bounds();
  std::vector<double> residual(constraint_lower_bounds.size(), 0.0);
  // CSR SpMV
  for (size_t i = 0; i < offsets.size() - 1; ++i) {
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      residual[i] += values[j] * solution[indices[j]];
    }
  }

  auto functor = violation<double>{};

  // Compute violation to lower/upper bound
  for (size_t i = 0; i < residual.size(); ++i) {
    double tolerance = abs_tolerance + combine_finite_abs_bounds<double>(
                                         constraint_lower_bounds[i], constraint_upper_bounds[i]) *
                                         rel_tolerance;
    double viol = functor(residual[i], constraint_lower_bounds[i], constraint_upper_bounds[i]);
    EXPECT_LE(viol, tolerance);
  }
}

static std::tuple<mip_termination_status_t, double, double> test_mps_file(
  std::string test_instance,
  double time_limit     = 1,
  bool heuristics_only  = true,
  presolver_t presolver = presolver_t::Default)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  mip_solver_settings_t<int, double> settings;
  settings.time_limit                  = time_limit;
  settings.heuristics_only             = heuristics_only;
  settings.presolver                   = presolver;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  return std::make_tuple(solution.get_termination_status(),
                         solution.get_objective_value(),
                         solution.get_solution_bound());
}

struct fj_tweaks_t {
  double objective_weight = 0;
};

struct fj_state_t {
  detail::solution_t<int, double> solution;
  std::vector<double> solution_vector;
  int minimums;
  double incumbent_objective;
  double incumbent_violation;
};

static fj_state_t run_fj(detail::problem_t<int, double>& problem,
                         const detail::fj_settings_t& fj_settings,
                         fj_tweaks_t tweaks                   = {},
                         std::vector<double> initial_solution = {},
                         int determinism_mode                 = CUOPT_MODE_OPPORTUNISTIC)
{
  auto settings             = mip_solver_settings_t<int, double>{};
  settings.time_limit       = 30.;
  settings.determinism_mode = determinism_mode;
  auto timer = cuopt::termination_checker_t(30.0, cuopt::termination_checker_t::root_tag_t{});
  detail::mip_solver_t<int, double> solver(problem, settings, timer);

  detail::solution_t<int, double> solution(*solver.context.problem_ptr);
  if (initial_solution.size() > 0) {
    expand_device_copy(solution.assignment, initial_solution, solution.handle_ptr->get_stream());
  } else {
    thrust::fill(solution.handle_ptr->get_thrust_policy(),
                 solution.assignment.begin(),
                 solution.assignment.end(),
                 0.0);
  }
  solution.clamp_within_bounds();

  detail::fj_t<int, double> fj(solver.context, fj_settings);
  fj.reset_weights(solution.handle_ptr->get_stream(), 1.);
  fj.objective_weight.set_value_async(tweaks.objective_weight, solution.handle_ptr->get_stream());
  solution.handle_ptr->sync_stream();

  fj.solve(solution);
  auto solution_vector = host_copy(solution.assignment, solution.handle_ptr->get_stream());

  return {solution,
          solution_vector,
          fj.climbers[0]->local_minimums_reached.value(solution.handle_ptr->get_stream()),
          fj.climbers[0]->incumbent_objective.value(solution.handle_ptr->get_stream()),
          fj.climbers[0]->violation_score.value(solution.handle_ptr->get_stream())};
}

}  // namespace cuopt::linear_programming::test
