/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../utilities/pdlp_test_utilities.cuh"

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <linear_programming/utils.cuh>
#include <mip/presolve/third_party_presolve.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

// Helper function to compute constraint residuals for the original problem
static void compute_constraint_residuals(const std::vector<double>& coefficients,
                                         const std::vector<int>& indices,
                                         const std::vector<int>& offsets,
                                         const std::vector<double>& solution,
                                         std::vector<double>& residuals)
{
  size_t n_constraints = offsets.size() - 1;
  residuals.resize(n_constraints, 0.0);
  // CSR SpMV: A * x
  for (size_t i = 0; i < n_constraints; ++i) {
    residuals[i] = 0.0;
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      residuals[i] += coefficients[j] * solution[indices[j]];
    }
  }
}

// Helper function to compute the objective value
static double compute_objective(const std::vector<double>& objective_coeffs,
                                const std::vector<double>& solution,
                                double obj_offset)
{
  double obj = obj_offset;
  for (size_t i = 0; i < objective_coeffs.size(); ++i) {
    obj += objective_coeffs[i] * solution[i];
  }
  return obj;
}

// Helper function to check constraint satisfaction
static void check_constraint_satisfaction(const std::vector<double>& residuals,
                                          const std::vector<double>& lower_bounds,
                                          const std::vector<double>& upper_bounds,
                                          double tolerance)
{
  for (size_t i = 0; i < residuals.size(); ++i) {
    double lb = lower_bounds[i];
    double ub = upper_bounds[i];

    // Check lower bound
    if (lb != -std::numeric_limits<double>::infinity()) {
      EXPECT_GE(residuals[i], lb - tolerance) << "Constraint " << i << " violates lower bound";
    }
    // Check upper bound
    if (ub != std::numeric_limits<double>::infinity()) {
      EXPECT_LE(residuals[i], ub + tolerance) << "Constraint " << i << " violates upper bound";
    }
  }
}

// Helper function to check variable bounds
static void check_variable_bounds(const std::vector<double>& solution,
                                  const std::vector<double>& lower_bounds,
                                  const std::vector<double>& upper_bounds,
                                  double tolerance)
{
  for (size_t i = 0; i < solution.size(); ++i) {
    double lb = lower_bounds[i];
    double ub = upper_bounds[i];

    if (lb != -std::numeric_limits<double>::infinity()) {
      EXPECT_GE(solution[i], lb - tolerance) << "Variable " << i << " violates lower bound";
    }
    if (ub != std::numeric_limits<double>::infinity()) {
      EXPECT_LE(solution[i], ub + tolerance) << "Variable " << i << " violates upper bound";
    }
  }
}

// Test PSLP presolver postsolve accuracy using afiro problem
TEST(pslp_presolve, postsolve_accuracy_afiro)
{
  const raft::handle_t handle_{};
  constexpr double tolerance    = 1e-5;
  constexpr double expected_obj = -464.75314;  // Known optimal objective for afiro

  auto path           = make_path_absolute("linear_programming/afiro_original.mps");
  auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, true);

  // Store original problem data for later verification
  const auto& orig_coefficients = mps_data_model.get_constraint_matrix_values();
  const auto& orig_indices      = mps_data_model.get_constraint_matrix_indices();
  const auto& orig_offsets      = mps_data_model.get_constraint_matrix_offsets();
  const auto& orig_obj_coeffs   = mps_data_model.get_objective_coefficients();
  const auto& orig_var_lb       = mps_data_model.get_variable_lower_bounds();
  const auto& orig_var_ub       = mps_data_model.get_variable_upper_bounds();
  const auto& orig_constr_lb    = mps_data_model.get_constraint_lower_bounds();
  const auto& orig_constr_ub    = mps_data_model.get_constraint_upper_bounds();
  const double orig_obj_offset  = mps_data_model.get_objective_offset();
  const int orig_n_vars         = mps_data_model.get_n_variables();
  const int orig_n_constraints  = mps_data_model.get_n_constraints();

  // Solve with PSLP presolve enabled
  auto solver_settings                                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.tolerances.relative_primal_tolerance = 1e-6;
  solver_settings.tolerances.relative_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_primal_tolerance = 1e-6;
  solver_settings.tolerances.absolute_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_gap_tolerance    = 1e-6;
  solver_settings.tolerances.relative_gap_tolerance    = 1e-6;
  solver_settings.presolver                            = presolver_t::PSLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);

  // Get the postsolved primal solution
  auto h_primal_solution = host_copy(solution.get_primal_solution(), handle_.get_stream());

  // Verify solution size matches original problem
  EXPECT_EQ(h_primal_solution.size(), orig_n_vars)
    << "Postsolved solution size should match original problem";

  // Verify objective value
  double computed_obj = compute_objective(orig_obj_coeffs, h_primal_solution, orig_obj_offset);
  EXPECT_NEAR(computed_obj, expected_obj, 1.0)
    << "Postsolved objective should match expected optimal";

  // Verify variable bounds
  check_variable_bounds(h_primal_solution, orig_var_lb, orig_var_ub, tolerance);

  // Verify constraint satisfaction
  std::vector<double> residuals;
  compute_constraint_residuals(
    orig_coefficients, orig_indices, orig_offsets, h_primal_solution, residuals);
  EXPECT_EQ(residuals.size(), orig_n_constraints);
  check_constraint_satisfaction(residuals, orig_constr_lb, orig_constr_ub, tolerance);
}

// Test PSLP postsolve dual solution accuracy
TEST(pslp_presolve, postsolve_dual_accuracy_afiro)
{
  const raft::handle_t handle_{};
  constexpr double tolerance = 1e-5;

  auto path           = make_path_absolute("linear_programming/afiro_original.mps");
  auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, true);

  const int orig_n_vars        = mps_data_model.get_n_variables();
  const int orig_n_constraints = mps_data_model.get_n_constraints();

  // Solve with PSLP presolve and dual postsolve enabled
  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::PSLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);

  // Get postsolved solutions
  auto h_primal = host_copy(solution.get_primal_solution(), handle_.get_stream());
  auto h_dual   = host_copy(solution.get_dual_solution(), handle_.get_stream());

  // Verify sizes
  EXPECT_EQ(h_primal.size(), orig_n_vars) << "Postsolved primal size should match original";
  EXPECT_EQ(h_dual.size(), orig_n_constraints) << "Postsolved dual size should match original";

  // Verify primal and dual objectives are close (weak duality check)
  double primal_obj = solution.get_additional_termination_information().primal_objective;
  double dual_obj   = solution.get_additional_termination_information().dual_objective;
  EXPECT_NEAR(primal_obj, dual_obj, 1.0) << "Primal and dual objectives should be close at optimum";
}

// Test PSLP postsolve with a larger problem
TEST(pslp_presolve, postsolve_accuracy_larger_problem)
{
  const raft::handle_t handle_{};
  constexpr double tolerance = 1e-4;

  auto path           = make_path_absolute("linear_programming/ex10/ex10.mps");
  auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, false);

  // Store original problem dimensions
  const auto& orig_coefficients = mps_data_model.get_constraint_matrix_values();
  const auto& orig_indices      = mps_data_model.get_constraint_matrix_indices();
  const auto& orig_offsets      = mps_data_model.get_constraint_matrix_offsets();
  const auto& orig_var_lb       = mps_data_model.get_variable_lower_bounds();
  const auto& orig_var_ub       = mps_data_model.get_variable_upper_bounds();
  const auto& orig_constr_lb    = mps_data_model.get_constraint_lower_bounds();
  const auto& orig_constr_ub    = mps_data_model.get_constraint_upper_bounds();
  const int orig_n_vars         = mps_data_model.get_n_variables();
  const int orig_n_constraints  = mps_data_model.get_n_constraints();

  // Solve with PSLP presolve
  auto solver_settings                                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver                            = presolver_t::PSLP;
  solver_settings.tolerances.relative_primal_tolerance = 1e-6;
  solver_settings.tolerances.relative_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_primal_tolerance = 1e-6;
  solver_settings.tolerances.absolute_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_gap_tolerance    = 1e-6;
  solver_settings.tolerances.relative_gap_tolerance    = 1e-6;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);

  auto h_primal = host_copy(solution.get_primal_solution(), handle_.get_stream());

  // Verify solution dimension
  EXPECT_EQ(h_primal.size(), orig_n_vars);

  // Verify variable bounds
  check_variable_bounds(h_primal, orig_var_lb, orig_var_ub, tolerance);

  // Verify constraint satisfaction
  std::vector<double> residuals;
  compute_constraint_residuals(orig_coefficients, orig_indices, orig_offsets, h_primal, residuals);
  check_constraint_satisfaction(residuals, orig_constr_lb, orig_constr_ub, tolerance);
}

// Test that PSLP and no presolve give similar objective values
TEST(pslp_presolve, compare_with_no_presolve)
{
  const raft::handle_t handle_{};
  constexpr double obj_tolerance = 1e-3;

  auto path           = make_path_absolute("linear_programming/afiro_original.mps");
  auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, true);

  // Solve without presolve
  auto settings_no_presolve      = pdlp_solver_settings_t<int, double>{};
  settings_no_presolve.method    = cuopt::linear_programming::method_t::PDLP;
  settings_no_presolve.presolver = presolver_t::None;
  settings_no_presolve.tolerances.relative_primal_tolerance = 1e-6;
  settings_no_presolve.tolerances.relative_dual_tolerance   = 1e-6;
  settings_no_presolve.tolerances.absolute_primal_tolerance = 1e-6;
  settings_no_presolve.tolerances.absolute_dual_tolerance   = 1e-6;
  settings_no_presolve.tolerances.absolute_gap_tolerance    = 1e-6;
  settings_no_presolve.tolerances.relative_gap_tolerance    = 1e-6;

  optimization_problem_solution_t<int, double> solution_no_presolve =
    solve_lp(&handle_, mps_data_model, settings_no_presolve);

  // Solve with PSLP presolve
  auto settings_pslp                                 = pdlp_solver_settings_t<int, double>{};
  settings_pslp.method                               = cuopt::linear_programming::method_t::PDLP;
  settings_pslp.presolver                            = presolver_t::PSLP;
  settings_pslp.tolerances.relative_primal_tolerance = 1e-6;
  settings_pslp.tolerances.relative_dual_tolerance   = 1e-6;
  settings_pslp.tolerances.absolute_primal_tolerance = 1e-6;
  settings_pslp.tolerances.absolute_dual_tolerance   = 1e-6;
  settings_pslp.tolerances.absolute_gap_tolerance    = 1e-6;
  settings_pslp.tolerances.relative_gap_tolerance    = 1e-6;

  optimization_problem_solution_t<int, double> solution_pslp =
    solve_lp(&handle_, mps_data_model, settings_pslp);

  // Both should be optimal
  EXPECT_EQ((int)solution_no_presolve.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)solution_pslp.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);

  // Objective values should match
  double obj_no_presolve =
    solution_no_presolve.get_additional_termination_information().primal_objective;
  double obj_pslp = solution_pslp.get_additional_termination_information().primal_objective;

  EXPECT_NEAR(obj_no_presolve, obj_pslp, obj_tolerance * fabs(obj_no_presolve))
    << "PSLP presolve should give same objective as no presolve";

  // Also test the solution vector (primal) and dual solution vector for equality

  // Get the primal solution for both solves
  auto h_primal_no_presolve =
    host_copy(solution_no_presolve.get_primal_solution(), handle_.get_stream());
  auto h_primal_pslp = host_copy(solution_pslp.get_primal_solution(), handle_.get_stream());

  ASSERT_EQ(h_primal_no_presolve.size(), h_primal_pslp.size())
    << "Primal solution sizes must match";
  // Compute relative L2 error between h_primal_no_presolve and h_primal_pslp
  double num   = 0.0;
  double denom = 0.0;
  for (size_t i = 0; i < h_primal_no_presolve.size(); ++i) {
    double diff = h_primal_no_presolve[i] - h_primal_pslp[i];
    num += diff * diff;
    denom += h_primal_no_presolve[i] * h_primal_no_presolve[i];
  }
  double rel_l2_err = denom > 0 ? sqrt(num) / sqrt(denom) : sqrt(num);
  EXPECT_LT(rel_l2_err, 1e-2) << "Relative L2 error in primal solution is too large (" << rel_l2_err
                              << ")";
}

// Test PSLP postsolve with reduced costs
TEST(pslp_presolve, postsolve_reduced_costs)
{
  const raft::handle_t handle_{};

  auto path           = make_path_absolute("linear_programming/afiro_original.mps");
  auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, true);

  const int orig_n_vars = mps_data_model.get_n_variables();

  // Solve with PSLP and dual postsolve
  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::PSLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);

  // Get postsolved reduced costs
  auto h_reduced_costs = host_copy(solution.get_reduced_cost(), handle_.get_stream());

  // Verify reduced costs size matches original problem
  EXPECT_EQ(h_reduced_costs.size(), orig_n_vars)
    << "Postsolved reduced costs size should match original problem variables";
}

// Test PSLP postsolve on multiple problems to ensure consistency
TEST(pslp_presolve, postsolve_multiple_problems)
{
  const raft::handle_t handle_{};
  constexpr double tolerance = 1e-4;

  std::vector<std::pair<std::string, double>> instances{
    {"afiro_original", -464.75314},
    {"ex10/ex10", 100.0003411893773},
  };

  for (const auto& [name, expected_obj] : instances) {
    auto path           = make_path_absolute("linear_programming/" + name + ".mps");
    auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, name == "afiro_original");

    const int orig_n_vars        = mps_data_model.get_n_variables();
    const int orig_n_constraints = mps_data_model.get_n_constraints();

    auto solver_settings      = pdlp_solver_settings_t<int, double>{};
    solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
    solver_settings.presolver = presolver_t::PSLP;

    optimization_problem_solution_t<int, double> solution =
      solve_lp(&handle_, mps_data_model, solver_settings);

    EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL)
      << "Problem " << name << " should be optimal";

    auto h_primal = host_copy(solution.get_primal_solution(), handle_.get_stream());
    EXPECT_EQ(h_primal.size(), orig_n_vars)
      << "Problem " << name << " postsolved solution size mismatch";

    double primal_obj = solution.get_additional_termination_information().primal_objective;
    double rel_error  = std::abs((primal_obj - expected_obj) / expected_obj);
    EXPECT_LT(rel_error, 0.01) << "Problem " << name << " objective mismatch";
  }
}

}  // namespace cuopt::linear_programming::test

CUOPT_TEST_PROGRAM_MAIN()
