/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../utilities/pdlp_test_utilities.cuh"

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <linear_algebra/sort_csr.cuh>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <pdlp/utils.cuh>
#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::test {

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

static void compute_transpose_matvec(const std::vector<double>& values,
                                     const std::vector<int>& indices,
                                     const std::vector<int>& offsets,
                                     const std::vector<double>& y,
                                     int n_cols,
                                     std::vector<double>& result)
{
  assert(!offsets.empty());
  assert(values.size() == indices.size());
  assert(n_cols >= 0);
  size_t n_rows = offsets.size() - 1;
  assert(y.size() == n_rows);
  assert((size_t)offsets.back() == values.size());
  result.assign(n_cols, 0.0);
  std::vector<double> kahan_compensation(n_cols, 0.0);
  for (size_t i = 0; i < n_rows; ++i) {
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      int col = indices[j];
      assert(col >= 0 && col < n_cols);
      double delta            = values[j] * y[i];
      double corrected        = delta - kahan_compensation[col];
      double next             = result[col] + corrected;
      kahan_compensation[col] = (next - result[col]) - corrected;
      result[col]             = next;
    }
  }
}

// Complimentary slackness checks
static void check_reduced_cost_consistency(const std::vector<double>& reduced_cost,
                                           const std::vector<double>& primal,
                                           const std::vector<double>& var_lb,
                                           const std::vector<double>& var_ub,
                                           double bound_tol,
                                           double dual_tol)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  assert(reduced_cost.size() == primal.size());
  assert(var_lb.size() == primal.size());
  assert(var_ub.size() == primal.size());
  for (size_t j = 0; j < primal.size(); ++j) {
    const bool has_lb = var_lb[j] > -inf;
    const bool has_ub = var_ub[j] < inf;
    const double z_j  = reduced_cost[j];

    // If a side is missing, its multiplier cannot exist.
    if (!has_lb) {
      ASSERT_LE(z_j, dual_tol) << "positive reduced cost requires a finite lower bound at variable "
                               << j;
    }
    if (!has_ub) {
      ASSERT_GE(z_j, -dual_tol)
        << "negative reduced cost requires a finite upper bound at variable " << j;
    }

    // If we are strictly away from a bound, the multiplier for that side must vanish.
    if (has_lb && primal[j] - var_lb[j] > bound_tol) {
      ASSERT_LE(z_j, dual_tol)
        << "positive reduced cost requires an active lower bound at variable " << j;
    }
    if (has_ub && var_ub[j] - primal[j] > bound_tol) {
      ASSERT_GE(z_j, -dual_tol)
        << "negative reduced cost requires an active upper bound at variable " << j;
    }
  }
}

// General-form row-bound KKT for exported solver duals.
//
// Positive y_i corresponds to the lower row bound, negative y_i to the upper
// row bound. If a side is absent, the corresponding multiplier must be zero.
static void check_dual_sign_consistency(const std::vector<double>& Ax,
                                        const std::vector<double>& dual,
                                        const std::vector<double>& con_lb,
                                        const std::vector<double>& con_ub,
                                        double bound_tol,
                                        double dual_tol)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  assert(Ax.size() == dual.size());
  assert(con_lb.size() == dual.size());
  assert(con_ub.size() == dual.size());
  for (size_t i = 0; i < dual.size(); ++i) {
    const bool has_lb = con_lb[i] > -inf;
    const bool has_ub = con_ub[i] < inf;

    if (!has_lb) {
      ASSERT_LE(dual[i], dual_tol)
        << "positive row dual requires a finite lower bound at constraint " << i;
    }
    if (!has_ub) {
      ASSERT_GE(dual[i], -dual_tol)
        << "negative row dual requires a finite upper bound at constraint " << i;
    }

    if (has_lb && Ax[i] - con_lb[i] > bound_tol) {
      ASSERT_LE(dual[i], dual_tol)
        << "positive row dual requires an active lower bound at constraint " << i;
    }
    if (has_ub && con_ub[i] - Ax[i] > bound_tol) {
      ASSERT_GE(dual[i], -dual_tol)
        << "negative row dual requires an active upper bound at constraint " << i;
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
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, true);

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
  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::mathematical_optimization::method_t::PDLP;
  solver_settings.tolerances.relative_primal_tolerance = 1e-6;
  solver_settings.tolerances.relative_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_primal_tolerance = 1e-6;
  solver_settings.tolerances.absolute_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_gap_tolerance    = 1e-6;
  solver_settings.tolerances.relative_gap_tolerance    = 1e-6;
  solver_settings.presolver                            = presolver_t::PSLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

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

  auto path           = make_path_absolute("linear_programming/afiro_original.mps");
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, true);

  const int orig_n_vars        = mps_data_model.get_n_variables();
  const int orig_n_constraints = mps_data_model.get_n_constraints();

  // Solve with PSLP presolve and dual postsolve enabled
  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::mathematical_optimization::method_t::PDLP;
  solver_settings.presolver = presolver_t::PSLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

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
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);

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
  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::mathematical_optimization::method_t::PDLP;
  solver_settings.presolver = presolver_t::PSLP;
  solver_settings.tolerances.relative_primal_tolerance = 1e-6;
  solver_settings.tolerances.relative_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_primal_tolerance = 1e-6;
  solver_settings.tolerances.absolute_dual_tolerance   = 1e-6;
  solver_settings.tolerances.absolute_gap_tolerance    = 1e-6;
  solver_settings.tolerances.relative_gap_tolerance    = 1e-6;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

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
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, true);

  // Solve without presolve
  auto settings_no_presolve      = pdlp_solver_settings_t<int, double>{};
  settings_no_presolve.method    = cuopt::mathematical_optimization::method_t::PDLP;
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
  auto settings_pslp      = pdlp_solver_settings_t<int, double>{};
  settings_pslp.method    = cuopt::mathematical_optimization::method_t::PDLP;
  settings_pslp.presolver = presolver_t::PSLP;
  settings_pslp.tolerances.relative_primal_tolerance = 1e-6;
  settings_pslp.tolerances.relative_dual_tolerance   = 1e-6;
  settings_pslp.tolerances.absolute_primal_tolerance = 1e-6;
  settings_pslp.tolerances.absolute_dual_tolerance   = 1e-6;
  settings_pslp.tolerances.absolute_gap_tolerance    = 1e-6;
  settings_pslp.tolerances.relative_gap_tolerance    = 1e-6;

  optimization_problem_solution_t<int, double> solution_pslp =
    solve_lp(&handle_, mps_data_model, settings_pslp);

  // Both should be optimal
  EXPECT_EQ((int)solution_no_presolve.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)solution_pslp.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

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
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, true);

  const int orig_n_vars = mps_data_model.get_n_variables();

  // Solve with PSLP and dual postsolve
  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::mathematical_optimization::method_t::PDLP;
  solver_settings.presolver = presolver_t::PSLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, mps_data_model, solver_settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

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

  std::vector<std::pair<std::string, double>> instances{
    {"afiro_original", -464.75314},
    {"ex10/ex10", 100.0003411893773},
  };

  for (const auto& [name, expected_obj] : instances) {
    auto path = make_path_absolute("linear_programming/" + name + ".mps");
    auto mps_data_model =
      cuopt::mathematical_optimization::io::read_mps<int, double>(path, name == "afiro_original");

    const int orig_n_vars        = mps_data_model.get_n_variables();
    const int orig_n_constraints = mps_data_model.get_n_constraints();

    auto solver_settings      = pdlp_solver_settings_t<int, double>{};
    solver_settings.method    = cuopt::mathematical_optimization::method_t::PDLP;
    solver_settings.presolver = presolver_t::PSLP;

    optimization_problem_solution_t<int, double> solution =
      solve_lp(&handle_, mps_data_model, solver_settings);

    EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL)
      << "Problem " << name << " should be optimal";

    auto h_primal = host_copy(solution.get_primal_solution(), handle_.get_stream());
    EXPECT_EQ(h_primal.size(), orig_n_vars)
      << "Problem " << name << " postsolved solution size mismatch";

    double primal_obj = solution.get_additional_termination_information().primal_objective;
    double rel_error  = std::abs((primal_obj - expected_obj) / expected_obj);
    EXPECT_LT(rel_error, 0.01) << "Problem " << name << " objective mismatch";
  }
}
struct crush_test_param {
  std::string mps_path;
  bool use_pdlp;
};
class dual_crush_round_trip : public ::testing::TestWithParam<crush_test_param> {};

// Crush an optimal original-space (x, y, z) into reduced space and verify the
// user-space general-form KKT conditions on both the original and reduced LPs.
TEST_P(dual_crush_round_trip, kkt_check)
{
  const raft::handle_t handle_{};
  auto stream = handle_.get_stream();

  constexpr double bound_tol = 1e-5;
  constexpr double kkt_tol   = 9e-2;

  const auto& param = GetParam();
  auto path         = make_path_absolute(param.mps_path);
  auto mps          = cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  auto op_problem   = mps_data_model_to_optimization_problem(&handle_, mps);

  // Step 1: Presolve with a single presolver instance (same one used for crush later)
  sort_csr(op_problem);
  mip::third_party_presolve_t<int, double> presolver;
  auto result = presolver.apply(op_problem,
                                problem_category_t::LP,
                                presolver_t::Papilo,
                                /*dual_postsolve=*/true,
                                /*abs_tol=*/1e-6,
                                /*rel_tol=*/1e-9,
                                /*time_limit=*/60.0);
  ASSERT_TRUE(result.status == mip::third_party_presolve_status_t::REDUCED ||
              result.status == mip::third_party_presolve_status_t::UNCHANGED);

  // Step 2: Solve the reduced problem (no presolve, we already did it)
  auto settings           = pdlp_solver_settings_t<int, double>{};
  settings.presolver      = presolver_t::None;
  settings.dual_postsolve = true;
  settings.time_limit     = 60.0;
  if (param.use_pdlp) {
    settings.method                             = cuopt::mathematical_optimization::method_t::PDLP;
    settings.tolerances.absolute_dual_tolerance = 1e-7;
    settings.tolerances.relative_dual_tolerance = 0;
    settings.tolerances.absolute_primal_tolerance = 1e-7;
    settings.tolerances.relative_primal_tolerance = 0;
  } else {
    settings.method = cuopt::mathematical_optimization::method_t::DualSimplex;
  }

  auto reduced_solution = solve_lp(result.reduced_problem, settings);
  ASSERT_EQ(reduced_solution.get_termination_status(), pdlp_termination_status_t::Optimal);

  // Step 3: Postsolve to get original-space (x, y, z) using the same presolver.
  // For PDLP, derive z = c_red - A_red^T y_red instead of using get_reduced_cost(),
  // since PDLP's approximate solution may have inconsistent reduced costs.
  auto primal_sol = cuopt::device_copy(reduced_solution.get_primal_solution(), stream);
  auto dual_sol   = cuopt::device_copy(reduced_solution.get_dual_solution(), stream);
  rmm::device_uvector<double> rc_sol(0, stream);
  if (param.use_pdlp) {
    auto red_A_vals    = result.reduced_problem.get_constraint_matrix_values_host();
    auto red_A_indices = result.reduced_problem.get_constraint_matrix_indices_host();
    auto red_A_offsets = result.reduced_problem.get_constraint_matrix_offsets_host();
    auto red_c         = result.reduced_problem.get_objective_coefficients_host();
    auto h_y_red       = host_copy(dual_sol, stream);
    int red_n_vars     = result.reduced_problem.get_n_variables();

    std::vector<double> ATy_red;
    compute_transpose_matvec(
      red_A_vals, red_A_indices, red_A_offsets, h_y_red, red_n_vars, ATy_red);
    std::vector<double> z_red(red_n_vars);
    for (int j = 0; j < red_n_vars; ++j) {
      z_red[j] = red_c[j] - ATy_red[j];
    }
    rc_sol = cuopt::device_copy(z_red, stream);
  } else {
    rc_sol = cuopt::device_copy(reduced_solution.get_reduced_cost(), stream);
  }

  presolver.undo(primal_sol, dual_sol, rc_sol, problem_category_t::LP, false, true, stream);

  auto x_orig  = host_copy(primal_sol, stream);
  auto y_orig  = host_copy(dual_sol, stream);
  auto rc_orig = host_copy(rc_sol, stream);

  ASSERT_EQ((int)x_orig.size(), op_problem.get_n_variables());
  ASSERT_EQ((int)y_orig.size(), op_problem.get_n_constraints());

  // Step 4: Sanity-check the postsolve output in original space before crushing.
  auto orig_A_vals    = op_problem.get_constraint_matrix_values_host();
  auto orig_A_indices = op_problem.get_constraint_matrix_indices_host();
  auto orig_A_offsets = op_problem.get_constraint_matrix_offsets_host();
  auto orig_c         = op_problem.get_objective_coefficients_host();
  auto orig_var_lb    = op_problem.get_variable_lower_bounds_host();
  auto orig_var_ub    = op_problem.get_variable_upper_bounds_host();
  auto orig_con_lb    = op_problem.get_constraint_lower_bounds_host();
  auto orig_con_ub    = op_problem.get_constraint_upper_bounds_host();
  {
    int orig_n_vars = op_problem.get_n_variables();

    // Stationarity: z == c - A^T y
    std::vector<double> ATy_orig;
    compute_transpose_matvec(
      orig_A_vals, orig_A_indices, orig_A_offsets, y_orig, orig_n_vars, ATy_orig);
    for (size_t j = 0; j < rc_orig.size(); ++j) {
      double z_derived = orig_c[j] - ATy_orig[j];
      EXPECT_NEAR(rc_orig[j], z_derived, kkt_tol)
        << "postsolve stationarity violation at original variable " << j;
    }

    // Primal feasibility
    check_variable_bounds(x_orig, orig_var_lb, orig_var_ub, bound_tol);
    std::vector<double> Ax_orig;
    compute_constraint_residuals(orig_A_vals, orig_A_indices, orig_A_offsets, x_orig, Ax_orig);
    check_constraint_satisfaction(Ax_orig, orig_con_lb, orig_con_ub, bound_tol);

    // Variable- and row-bound KKT in original space
    check_reduced_cost_consistency(rc_orig, x_orig, orig_var_lb, orig_var_ub, bound_tol, kkt_tol);
    check_dual_sign_consistency(Ax_orig, y_orig, orig_con_lb, orig_con_ub, bound_tol, kkt_tol);
  }

  // Step 5: Crush using the same presolver that produced the postsolve storage
  std::vector<double> x_crushed, y_crushed, rc_crushed;
  presolver.crush_primal_dual_solution(x_orig,
                                       y_orig,
                                       x_crushed,
                                       y_crushed,
                                       rc_orig,
                                       rc_crushed,
                                       orig_A_vals,
                                       orig_A_indices,
                                       orig_A_offsets);

  int n_vars = result.reduced_problem.get_n_variables();
  int n_cons = result.reduced_problem.get_n_constraints();
  ASSERT_EQ((int)x_crushed.size(), n_vars);
  ASSERT_EQ((int)y_crushed.size(), n_cons);
  ASSERT_EQ((int)rc_crushed.size(), n_vars);

  auto A_vals    = result.reduced_problem.get_constraint_matrix_values_host();
  auto A_indices = result.reduced_problem.get_constraint_matrix_indices_host();
  auto A_offsets = result.reduced_problem.get_constraint_matrix_offsets_host();
  auto c_red     = result.reduced_problem.get_objective_coefficients_host();
  auto var_lb    = result.reduced_problem.get_variable_lower_bounds_host();
  auto var_ub    = result.reduced_problem.get_variable_upper_bounds_host();
  auto con_lb    = result.reduced_problem.get_constraint_lower_bounds_host();
  auto con_ub    = result.reduced_problem.get_constraint_upper_bounds_host();

  // Primal feasibility: variable bounds
  check_variable_bounds(x_crushed, var_lb, var_ub, bound_tol);

  // Primal feasibility: constraint satisfaction (l_c <= Ax <= u_c)
  std::vector<double> Ax;
  compute_constraint_residuals(A_vals, A_indices, A_offsets, x_crushed, Ax);
  check_constraint_satisfaction(Ax, con_lb, con_ub, bound_tol);

  // Dual feasibility: compute implied reduced costs z = c - A^T y
  std::vector<double> ATy;
  compute_transpose_matvec(A_vals, A_indices, A_offsets, y_crushed, n_vars, ATy);

  // Variable-bound KKT in reduced space
  check_reduced_cost_consistency(rc_crushed, x_crushed, var_lb, var_ub, bound_tol, kkt_tol);

  // Row-bound KKT in reduced space
  check_dual_sign_consistency(Ax, y_crushed, con_lb, con_ub, bound_tol, kkt_tol);

  // Crushed reduced costs: consistency with derived z = c - A^T y
  for (size_t j = 0; j < rc_crushed.size(); ++j) {
    double z_derived = c_red[j] - ATy[j];
    ASSERT_NEAR(rc_crushed[j], z_derived, kkt_tol)
      << "crushed reduced cost vs derived mismatch at variable " << j;
  }

  // Cross-check: crushed primal should match the solver's primal
  auto x_ref = host_copy(reduced_solution.get_primal_solution(), stream);
  ASSERT_EQ(x_crushed.size(), x_ref.size());
  for (size_t i = 0; i < x_crushed.size(); ++i) {
    EXPECT_NEAR(x_crushed[i], x_ref[i], kkt_tol) << "primal mismatch at reduced variable " << i;
  }

  // Cross-check: crushed objective should match the solver's objective
  auto obj_ref       = reduced_solution.get_additional_termination_information().primal_objective;
  double obj_crushed = 0.0;
  for (size_t i = 0; i < x_crushed.size(); ++i) {
    obj_crushed += c_red[i] * x_crushed[i];
  }
  obj_crushed += result.reduced_problem.get_objective_offset();
  EXPECT_NEAR(obj_crushed, obj_ref, kkt_tol * std::max(1.0, std::abs(obj_ref)))
    << "crushed objective mismatch";
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
  papilo_presolve,
  dual_crush_round_trip,
  ::testing::Values(
    crush_test_param{"linear_programming/graph40-40/graph40-40.mps", true},
    //crush_test_param{"linear_programming/ex10/ex10.mps", true},
    //crush_test_param{"linear_programming/datt256_lp/datt256_lp.mps", true},
    //crush_test_param{"linear_programming/woodlands09/woodlands09.mps", false},
    //crush_test_param{"linear_programming/savsched1/savsched1.mps", false},
    crush_test_param{"linear_programming/nug08-3rd/nug08-3rd.mps", true},
    crush_test_param{"linear_programming/qap15/qap15.mps", true},
    //crush_test_param{"linear_programming/scpm1/scpm1.mps", true},
    //crush_test_param{"linear_programming/neos3/neos3.mps", true},
    //crush_test_param{"linear_programming/a2864/a2864.mps", true},
    crush_test_param{"linear_programming/afiro_original.mps", false},
    crush_test_param{"mip/enlight_hard.mps", false},
    crush_test_param{"mip/enlight11.mps", false},
    crush_test_param{"mip/50v-10.mps", false},
    crush_test_param{"mip/fiball.mps", false},
    crush_test_param{"mip/gen-ip054.mps", false},
    crush_test_param{"mip/sct2.mps", false},
    crush_test_param{"mip/drayage-25-23.mps", false},
    crush_test_param{"mip/tr12-30.mps", false},
    crush_test_param{"mip/neos-3004026-krka.mps", false},
    crush_test_param{"mip/ns1208400.mps", false},
    //crush_test_param{"mip/gmu-35-50.mps", true},  // PDLP may time out in CI.
    crush_test_param{"mip/n2seq36q.mps", false},
    crush_test_param{"mip/seymour1.mps", false},
    //crush_test_param{"mip/thor50dday.mps", false},
    crush_test_param{"mip/neos8.mps", false},
    crush_test_param{"mip/app1-1.mps", false},
    crush_test_param{"mip/bnatt400.mps", false},
    crush_test_param{"mip/bnatt500.mps", false},
    //crush_test_param{"mip/brazil3.mps", false},
    crush_test_param{"mip/cbs-cta.mps", false},
    crush_test_param{"mip/CMS750_4.mps", false},
    crush_test_param{"mip/decomp2.mps", false},
    crush_test_param{"mip/dws008-01.mps", false},
    crush_test_param{"mip/germanrr.mps", false},
    crush_test_param{"mip/graph20-20-1rand.mps", false},
    crush_test_param{"mip/milo-v12-6-r2-40-1.mps", false},
    crush_test_param{"mip/neos-1445765.mps", false},
    crush_test_param{"mip/neos-1582420.mps", false},
    crush_test_param{"mip/neos-3083819-nubu.mps", false},
    crush_test_param{"mip/neos-5107597-kakapo.mps", false},
    crush_test_param{"mip/neos-5188808-nattai.mps", false},
    crush_test_param{"mip/net12.mps", false},
    crush_test_param{"mip/rocI-4-11.mps", false},
    crush_test_param{"mip/traininstance2.mps", false},
    crush_test_param{"mip/traininstance6.mps", false},
    crush_test_param{"mip/neos-787933.mps", false},
    crush_test_param{"mip/radiationm18-12-05.mps", false},
    crush_test_param{"mip/momentum1.mps", false},
    crush_test_param{"mip/rococoB10-011000.mps", false},
    crush_test_param{"mip/b1c1s1.mps", false},
    crush_test_param{"mip/nu25-pr12.mps", false},
    crush_test_param{"mip/air05.mps", false},
    crush_test_param{"mip/seymour.mps", false},
    crush_test_param{"mip/swath3.mps", false},
    crush_test_param{"mip/neos-950242.mps", false},
    crush_test_param{"mip/fastxgemm-n2r6s0t2.mps", false}
  ),
  [](const ::testing::TestParamInfo<crush_test_param>& info) {
    std::string name = info.param.mps_path;
    std::replace(name.begin(), name.end(), '/', '_');
    std::replace(name.begin(), name.end(), '.', '_');
    std::replace(name.begin(), name.end(), '-', '_');
    if (info.param.use_pdlp) name += "_pdlp";
    return name;
  }
);
// clang-format on

// Test that crushed solutions work as warmstarts in the full solving pipeline.
// Presolve → cold PDLP solve → postsolve → crush → warmstarted PDLP solve.
// The warmstarted solve should converge in fewer iterations than the cold start.
class crush_warmstart : public ::testing::TestWithParam<std::string> {};

TEST_P(crush_warmstart, round_trip)
{
  const raft::handle_t handle_{};
  auto stream = handle_.get_stream();

  auto path       = make_path_absolute(GetParam());
  auto mps        = cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps);

  // Step 1: Presolve
  sort_csr(op_problem);
  mip::third_party_presolve_t<int, double> presolver;
  auto result = presolver.apply(op_problem,
                                problem_category_t::LP,
                                presolver_t::Papilo,
                                /*dual_postsolve=*/true,
                                /*abs_tol=*/1e-6,
                                /*rel_tol=*/1e-9,
                                /*time_limit=*/60.0);
  ASSERT_TRUE(result.status == mip::third_party_presolve_status_t::REDUCED ||
              result.status == mip::third_party_presolve_status_t::UNCHANGED);

  int n_red_vars = result.reduced_problem.get_n_variables();
  int n_red_cons = result.reduced_problem.get_n_constraints();

  // Step 2: Cold PDLP solve of the reduced problem
  auto settings           = pdlp_solver_settings_t<int, double>{};
  settings.presolver      = presolver_t::None;
  settings.dual_postsolve = true;
  settings.method         = cuopt::mathematical_optimization::method_t::PDLP;
  settings.time_limit     = 60.0;

  auto cold_solution = solve_lp(result.reduced_problem, settings);
  ASSERT_EQ(cold_solution.get_termination_status(), pdlp_termination_status_t::Optimal);

  auto cold_iters = cold_solution.get_additional_termination_information().number_of_steps_taken;
  double cold_obj = cold_solution.get_additional_termination_information().primal_objective;

  // Step 3: Postsolve to original space.
  // Recompute z via kahan summation
  auto primal_sol = cuopt::device_copy(cold_solution.get_primal_solution(), stream);
  auto dual_sol   = cuopt::device_copy(cold_solution.get_dual_solution(), stream);

  auto red_A_vals    = result.reduced_problem.get_constraint_matrix_values_host();
  auto red_A_indices = result.reduced_problem.get_constraint_matrix_indices_host();
  auto red_A_offsets = result.reduced_problem.get_constraint_matrix_offsets_host();
  auto red_c         = result.reduced_problem.get_objective_coefficients_host();
  auto h_y_red       = host_copy(dual_sol, stream);

  std::vector<double> ATy_red;
  compute_transpose_matvec(red_A_vals, red_A_indices, red_A_offsets, h_y_red, n_red_vars, ATy_red);
  std::vector<double> z_red(n_red_vars);
  for (int j = 0; j < n_red_vars; ++j) {
    z_red[j] = red_c[j] - ATy_red[j];
  }
  auto rc_sol = cuopt::device_copy(z_red, stream);

  presolver.undo(primal_sol, dual_sol, rc_sol, problem_category_t::LP, false, true, stream);

  auto x_orig = host_copy(primal_sol, stream);
  auto y_orig = host_copy(dual_sol, stream);

  // Step 4: Crush back to reduced space (no rc needed for warmstart)
  auto orig_A_vals    = op_problem.get_constraint_matrix_values_host();
  auto orig_A_indices = op_problem.get_constraint_matrix_indices_host();
  auto orig_A_offsets = op_problem.get_constraint_matrix_offsets_host();

  std::vector<double> x_crushed, y_crushed, rc_unused;
  presolver.crush_primal_dual_solution(x_orig,
                                       y_orig,
                                       x_crushed,
                                       y_crushed,
                                       {},
                                       rc_unused,
                                       orig_A_vals,
                                       orig_A_indices,
                                       orig_A_offsets);

  ASSERT_EQ((int)x_crushed.size(), n_red_vars);
  ASSERT_EQ((int)y_crushed.size(), n_red_cons);

  // Step 5: Warmstarted PDLP solve of the reduced problem
  auto warm_settings = settings;
  warm_settings.set_initial_primal_solution(x_crushed.data(), n_red_vars, stream);
  warm_settings.set_initial_dual_solution(y_crushed.data(), n_red_cons, stream);

  auto warm_solution = solve_lp(result.reduced_problem, warm_settings);
  ASSERT_EQ(warm_solution.get_termination_status(), pdlp_termination_status_t::Optimal);

  double warm_obj = warm_solution.get_additional_termination_information().primal_objective;
  auto warm_iters = warm_solution.get_additional_termination_information().number_of_steps_taken;

  double obj_tol = 1e-3 * (1.0 + std::abs(cold_obj));
  EXPECT_NEAR(warm_obj, cold_obj, obj_tol) << "warmstarted objective should match cold solve";

  EXPECT_LT(warm_iters, cold_iters)
    << "warmstarted solve should not take more iterations than cold solve"
    << " (cold=" << cold_iters << ", warm=" << warm_iters << ")";
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
  papilo_presolve,
  crush_warmstart,
  ::testing::Values(
    "mip/fiball.mps",
    "mip/50v-10.mps",
    "mip/drayage-25-23.mps",
    "mip/neos-3004026-krka.mps",
    "mip/app1-1.mps",
    "mip/bnatt400.mps",
    "mip/decomp2.mps",
    "mip/graph20-20-1rand.mps",
    "mip/neos-1582420.mps",
    "mip/neos-5188808-nattai.mps",
    "mip/net12.mps",
    "mip/n2seq36q.mps",
    "mip/seymour1.mps",
    "mip/neos8.mps",
    "mip/CMS750_4.mps",
    "mip/cbs-cta.mps",
    "mip/swath3.mps",
    "mip/air05.mps",
    "mip/fastxgemm-n2r6s0t2.mps",
    "mip/dws008-01.mps",
    "mip/neos-1445765.mps",
    "mip/neos-3083819-nubu.mps",
    "mip/neos-5107597-kakapo.mps",
    "mip/rocI-4-11.mps"
  ),
  [](const ::testing::TestParamInfo<std::string>& info) {
    std::string name = info.param;
    std::replace(name.begin(), name.end(), '/', '_');
    std::replace(name.begin(), name.end(), '.', '_');
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
  }
);
// clang-format on

}  // namespace cuopt::mathematical_optimization::test

CUOPT_TEST_PROGRAM_MAIN()
