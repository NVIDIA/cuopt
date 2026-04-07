/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * Tests for semi-continuous variable support.
 *
 * A semi-continuous variable x satisfies: x = 0  OR  L <= x <= U  (0 < L < U).
 *
 * SC variables are reformulated inside solve_mip() before Papilo presolve using
 * GPU bounds propagation to derive tight upper bounds.
 */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr double kInf = std::numeric_limits<double>::infinity();

/**
 * Verify that a solution satisfies the semi-continuous constraint:
 * x == 0  OR  L <= x <= U
 */
static void check_sc_feasibility(double x, double L, double U, double tol = 1e-4)
{
  bool is_zero  = (std::abs(x) <= tol);
  bool in_range = (x >= L - tol && x <= U + tol);
  EXPECT_TRUE(is_zero || in_range)
    << "SC variable value " << x << " violates: x=0 or " << L << " <= x <= " << U;
}

// ---------------------------------------------------------------------------
// End-to-end solve tests
// Each test constructs a problem via mps_data_model_t and calls solve_mip.
// ---------------------------------------------------------------------------

/**
 * Minimize x, subject to: x = 0 or 2 <= x <= 5.
 * Optimal: x = 0.
 */
TEST(SemiContinuousSolve, MinimizeSCVar)
{
  raft::handle_t handle;
  mps_parser::mps_data_model_t<int, double> model;

  std::vector<double> c        = {1.0};
  std::vector<double> var_lb   = {2.0};
  std::vector<double> var_ub   = {5.0};
  std::vector<char> var_types  = {'S'};
  std::vector<std::string> names = {"x"};

  std::vector<int> offsets = {0};
  model.set_csr_constraint_matrix(nullptr, 0, nullptr, 0, offsets.data(), 1);
  model.set_constraint_lower_bounds(nullptr, 0);
  model.set_constraint_upper_bounds(nullptr, 0);
  model.set_objective_coefficients(c.data(), 1);
  model.set_variable_lower_bounds(var_lb.data(), 1);
  model.set_variable_upper_bounds(var_ub.data(), 1);
  model.set_variable_types(var_types);
  model.set_variable_names(names);
  model.set_maximize(false);

  mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 10;

  auto result = solve_mip(&handle, model, settings);

  EXPECT_EQ(result.get_termination_status(), mip_termination_status_t::Optimal);

  // Solution must contain exactly the 1 user variable; auxiliary binary must be stripped.
  EXPECT_EQ(result.get_solution().size(), 1u);
  auto host_sol = cuopt::host_copy(result.get_solution(), result.get_solution().stream());
  double x_val  = host_sol[0];
  check_sc_feasibility(x_val, 2.0, 5.0);
  EXPECT_NEAR(x_val, 0.0, 1e-4);
}

/**
 * Maximize x, subject to: x = 0 or 2 <= x <= 10, and x <= 8.
 * Optimal: x = 8.
 * GPU bounds propagation should tighten UB from 10 to 8.
 */
TEST(SemiContinuousSolve, MaximizeSCVar)
{
  raft::handle_t handle;
  mps_parser::mps_data_model_t<int, double> model;

  std::vector<double> c        = {1.0};
  std::vector<double> var_lb   = {2.0};
  std::vector<double> var_ub   = {10.0};
  std::vector<char> var_types  = {'S'};
  std::vector<std::string> names = {"x"};

  std::vector<double> A_val = {1.0};
  std::vector<int> A_idx    = {0};
  std::vector<int> A_off    = {0, 1};
  std::vector<double> con_lb = {-kInf};
  std::vector<double> con_ub = {8.0};

  model.set_csr_constraint_matrix(A_val.data(), 1, A_idx.data(), 1, A_off.data(), 2);
  model.set_constraint_lower_bounds(con_lb.data(), 1);
  model.set_constraint_upper_bounds(con_ub.data(), 1);
  model.set_objective_coefficients(c.data(), 1);
  model.set_variable_lower_bounds(var_lb.data(), 1);
  model.set_variable_upper_bounds(var_ub.data(), 1);
  model.set_variable_types(var_types);
  model.set_variable_names(names);
  model.set_maximize(true);

  mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 10;

  auto result = solve_mip(&handle, model, settings);

  EXPECT_EQ(result.get_termination_status(), mip_termination_status_t::Optimal);

  // Solution must contain exactly the 1 user variable; auxiliary binary must be stripped.
  EXPECT_EQ(result.get_solution().size(), 1u);

  auto host_sol = cuopt::host_copy(result.get_solution(), result.get_solution().stream());
  double x_val  = host_sol[0];
  check_sc_feasibility(x_val, 2.0, 10.0);
  EXPECT_NEAR(x_val, 8.0, 1e-4);
}

/**
 * Two SC variables: minimize x+y, x=0 or [3,5], y=0 or [2,4].
 * Constraint: x + y <= 6.
 * Optimal: x=0, y=0, obj=0.
 */
TEST(SemiContinuousSolve, TwoSCVariables)
{
  raft::handle_t handle;
  mps_parser::mps_data_model_t<int, double> model;

  std::vector<double> c        = {1.0, 1.0};
  std::vector<double> var_lb   = {3.0, 2.0};
  std::vector<double> var_ub   = {5.0, 4.0};
  std::vector<char> var_types  = {'S', 'S'};
  std::vector<std::string> names = {"x", "y"};

  std::vector<double> A_val = {1.0, 1.0};
  std::vector<int> A_idx    = {0, 1};
  std::vector<int> A_off    = {0, 2};
  std::vector<double> con_lb = {-kInf};
  std::vector<double> con_ub = {6.0};

  model.set_csr_constraint_matrix(A_val.data(), 2, A_idx.data(), 2, A_off.data(), 2);
  model.set_constraint_lower_bounds(con_lb.data(), 1);
  model.set_constraint_upper_bounds(con_ub.data(), 1);
  model.set_objective_coefficients(c.data(), 2);
  model.set_variable_lower_bounds(var_lb.data(), 2);
  model.set_variable_upper_bounds(var_ub.data(), 2);
  model.set_variable_types(var_types);
  model.set_variable_names(names);
  model.set_maximize(false);

  mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 10;

  auto result = solve_mip(&handle, model, settings);

  EXPECT_EQ(result.get_termination_status(), mip_termination_status_t::Optimal);

  // Solution must contain exactly the 2 user variables; 2 auxiliary binaries must be stripped.
  EXPECT_EQ(result.get_solution().size(), 2u);

  auto host_sol = cuopt::host_copy(result.get_solution(), result.get_solution().stream());
  double x_val  = host_sol[0];
  double y_val  = host_sol[1];
  check_sc_feasibility(x_val, 3.0, 5.0);
  check_sc_feasibility(y_val, 2.0, 4.0);
  EXPECT_NEAR(x_val + y_val, 0.0, 1e-4);
}

/**
 * SC variable forced into the non-zero range by a constraint (x >= 1).
 * x = 0 or [3, 8]; constraint x >= 1 makes x=0 infeasible.
 * Minimize x → optimal x = 3 (the lower bound L of the SC range).
 * Verifies that the lower bound L is correctly enforced when x must be non-zero.
 */
TEST(SemiContinuousSolve, ForcedNonZeroAtLowerBound)
{
  raft::handle_t handle;
  mps_parser::mps_data_model_t<int, double> model;

  const double L = 3.0;
  const double U = 8.0;

  std::vector<double> c        = {1.0};
  std::vector<double> var_lb   = {L};
  std::vector<double> var_ub   = {U};
  std::vector<char> var_types  = {'S'};
  std::vector<std::string> names = {"x"};

  // Constraint: x >= 1  (makes x=0 infeasible, forcing x into [L, U])
  std::vector<double> A_val = {1.0};
  std::vector<int> A_idx    = {0};
  std::vector<int> A_off    = {0, 1};
  std::vector<double> con_lb = {1.0};
  std::vector<double> con_ub = {kInf};

  model.set_csr_constraint_matrix(A_val.data(), 1, A_idx.data(), 1, A_off.data(), 2);
  model.set_constraint_lower_bounds(con_lb.data(), 1);
  model.set_constraint_upper_bounds(con_ub.data(), 1);
  model.set_objective_coefficients(c.data(), 1);
  model.set_variable_lower_bounds(var_lb.data(), 1);
  model.set_variable_upper_bounds(var_ub.data(), 1);
  model.set_variable_types(var_types);
  model.set_variable_names(names);
  model.set_maximize(false);

  mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 10;

  auto result = solve_mip(&handle, model, settings);

  EXPECT_EQ(result.get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_EQ(result.get_solution().size(), 1u);

  // x is always non-zero here, so test_variable_bounds (lb<=x<=ub) is valid.
  test_variable_bounds(model, result.get_solution(), settings);

  auto host_sol = cuopt::host_copy(result.get_solution(), result.get_solution().stream());
  double x_val  = host_sol[0];
  check_sc_feasibility(x_val, L, U);
  // x cannot be 0 (violates x>=1), so optimal is x=L=3
  EXPECT_NEAR(x_val, L, 1e-4);
}

/**
 * SC var with infinite UB; constraint x <= 7 provides the tight bound.
 * GPU bounds propagation should derive UB=7 and use it in the reformulation.
 * Minimize x → optimal x=0.
 */
TEST(SemiContinuousSolve, InfiniteUBDerivedFromConstraint)
{
  raft::handle_t handle;
  mps_parser::mps_data_model_t<int, double> model;

  std::vector<double> c        = {1.0};
  std::vector<double> var_lb   = {2.0};
  std::vector<double> var_ub   = {kInf};
  std::vector<char> var_types  = {'S'};
  std::vector<std::string> names = {"x"};

  std::vector<double> A_val = {1.0};
  std::vector<int> A_idx    = {0};
  std::vector<int> A_off    = {0, 1};
  std::vector<double> con_lb = {-kInf};
  std::vector<double> con_ub = {7.0};

  model.set_csr_constraint_matrix(A_val.data(), 1, A_idx.data(), 1, A_off.data(), 2);
  model.set_constraint_lower_bounds(con_lb.data(), 1);
  model.set_constraint_upper_bounds(con_ub.data(), 1);
  model.set_objective_coefficients(c.data(), 1);
  model.set_variable_lower_bounds(var_lb.data(), 1);
  model.set_variable_upper_bounds(var_ub.data(), 1);
  model.set_variable_types(var_types);
  model.set_variable_names(names);
  model.set_maximize(false);

  mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 10;

  auto result = solve_mip(&handle, model, settings);

  EXPECT_EQ(result.get_termination_status(), mip_termination_status_t::Optimal);

  // Solution must contain exactly the 1 user variable; auxiliary binary must be stripped.
  EXPECT_EQ(result.get_solution().size(), 1u);

  auto host_sol = cuopt::host_copy(result.get_solution(), result.get_solution().stream());
  double x_val  = host_sol[0];
  check_sc_feasibility(x_val, 2.0, 7.0);
  EXPECT_NEAR(x_val, 0.0, 1e-4);
}

/**
 * SC variable with negative lower bound (L=-2, U=5).
 * Since L <= 0, the value 0 is already in [L,U], so the SC constraint
 * x=0 OR L<=x<=U simplifies to plain continuous [-2,5].
 * Minimize x → optimal x=-2 (lower bound of the range).
 */
TEST(SemiContinuousSolve, NegativeLowerBound)
{
  raft::handle_t handle;
  mps_parser::mps_data_model_t<int, double> model;

  std::vector<double> c       = {1.0};
  std::vector<double> var_lb  = {-2.0};
  std::vector<double> var_ub  = {5.0};
  std::vector<char> var_types = {'S'};

  std::vector<int> offsets = {0};
  model.set_csr_constraint_matrix(nullptr, 0, nullptr, 0, offsets.data(), 1);
  model.set_constraint_lower_bounds(nullptr, 0);
  model.set_constraint_upper_bounds(nullptr, 0);
  model.set_objective_coefficients(c.data(), 1);
  model.set_variable_lower_bounds(var_lb.data(), 1);
  model.set_variable_upper_bounds(var_ub.data(), 1);
  model.set_variable_types(var_types);
  model.set_maximize(false);

  mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 10;

  // Should solve without error; no binary variable is added (L<=0 case)
  auto result = solve_mip(&handle, model, settings);

  EXPECT_EQ(result.get_termination_status(), mip_termination_status_t::Optimal);

  // L<=0: no binary added, solution size must still equal 1 (the original variable count).
  EXPECT_EQ(result.get_solution().size(), 1u);
  auto host_sol = cuopt::host_copy(result.get_solution(), result.get_solution().stream());
  // Variable is treated as continuous [-2, 5]; minimize x gives x=-2
  EXPECT_NEAR(host_sol[0], -2.0, 1e-4);
}

/**
 * SC variable with zero lower bound (L=0, U=5).
 * Since L=0, x=0 OR 0<=x<=5 is trivially x in [0,5] — plain continuous.
 * Minimize x → optimal x=0.
 */
TEST(SemiContinuousSolve, ZeroLowerBound)
{
  raft::handle_t handle;
  mps_parser::mps_data_model_t<int, double> model;

  std::vector<double> c       = {1.0};
  std::vector<double> var_lb  = {0.0};
  std::vector<double> var_ub  = {5.0};
  std::vector<char> var_types = {'S'};

  std::vector<int> offsets = {0};
  model.set_csr_constraint_matrix(nullptr, 0, nullptr, 0, offsets.data(), 1);
  model.set_constraint_lower_bounds(nullptr, 0);
  model.set_constraint_upper_bounds(nullptr, 0);
  model.set_objective_coefficients(c.data(), 1);
  model.set_variable_lower_bounds(var_lb.data(), 1);
  model.set_variable_upper_bounds(var_ub.data(), 1);
  model.set_variable_types(var_types);
  model.set_maximize(false);

  mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 10;

  auto result = solve_mip(&handle, model, settings);

  EXPECT_EQ(result.get_termination_status(), mip_termination_status_t::Optimal);

  // L=0: no binary added, solution size must still equal 1 (the original variable count).
  EXPECT_EQ(result.get_solution().size(), 1u);
  auto host_sol = cuopt::host_copy(result.get_solution(), result.get_solution().stream());
  EXPECT_NEAR(host_sol[0], 0.0, 1e-4);
}

// ---------------------------------------------------------------------------
// MPS parsing tests
// ---------------------------------------------------------------------------

/**
 * Write a minimal MPS with an SC variable and parse it.
 * Verifies the parser sets type='S' and correct bounds.
 */
TEST(SemiContinuousMPS, ParseSCVariable)
{
  const std::string mps_content = "NAME          SC_TEST\n"
                                  "ROWS\n"
                                  " N  obj\n"
                                  "COLUMNS\n"
                                  "    x         obj           1.0\n"
                                  "RHS\n"
                                  "BOUNDS\n"
                                  " LO BND       x             2.0\n"
                                  " SC BND       x             5.0\n"
                                  "ENDATA\n";

  const char* tmpdir   = std::getenv("TMPDIR");
  std::string tmp_path = std::string(tmpdir ? tmpdir : "/tmp") + "/cuopt_sc_test.mps";
  {
    std::ofstream f(tmp_path);
    ASSERT_TRUE(f.is_open()) << "Cannot create: " << tmp_path;
    f << mps_content;
  }

  auto model = cuopt::mps_parser::parse_mps<int, double>(tmp_path, false);

  EXPECT_EQ(model.get_n_variables(), 1);
  EXPECT_EQ(model.var_types_[0], 'S');
  EXPECT_DOUBLE_EQ(model.variable_lower_bounds_[0], 2.0);
  EXPECT_DOUBLE_EQ(model.variable_upper_bounds_[0], 5.0);
}

/**
 * SC variable with no explicit LO bound: parser should default L to 1.0 (CPLEX convention).
 */
TEST(SemiContinuousMPS, DefaultLowerBoundIsOne)
{
  const std::string mps_content = "NAME          SC_NOLB\n"
                                  "ROWS\n"
                                  " N  obj\n"
                                  "COLUMNS\n"
                                  "    z         obj           1.0\n"
                                  "RHS\n"
                                  "BOUNDS\n"
                                  " SC BND       z             10.0\n"
                                  "ENDATA\n";

  const char* tmpdir   = std::getenv("TMPDIR");
  std::string tmp_path = std::string(tmpdir ? tmpdir : "/tmp") + "/cuopt_sc_nolb.mps";
  {
    std::ofstream f(tmp_path);
    ASSERT_TRUE(f.is_open()) << "Cannot create: " << tmp_path;
    f << mps_content;
  }

  auto model = cuopt::mps_parser::parse_mps<int, double>(tmp_path, false);

  EXPECT_EQ(model.var_types_[0], 'S');
  EXPECT_DOUBLE_EQ(model.variable_lower_bounds_[0], 1.0);  // CPLEX default
  EXPECT_DOUBLE_EQ(model.variable_upper_bounds_[0], 10.0);
}

}  // namespace cuopt::linear_programming::test
