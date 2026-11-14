/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/bound_flipping_ratio_test.hpp>
#include <dual_simplex/farkas_certificate.hpp>
#include <dual_simplex/phase2.hpp>
#include <math_optimization/tic_toc.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::simplex::test {

class farkas_certificate_test : public ::testing::Test {
 protected:
  lp_problem_t<int, double> problem{nullptr, 1, 2, 2};
  simplex_solver_settings_t<int, double> settings;
  std::vector<double> certificate_y, certificate_lower, certificate_upper;
  double constant = 0.0;
  double work     = 0.0;

  void SetUp() override
  {
    problem.A.col_start           = {0, 1, 2};
    problem.A.i                   = {0, 0};
    problem.A.x                   = {1.0, 1.0};
    problem.rhs                   = {-1.0};
    problem.lower                 = {0.0, 0.0};
    problem.upper                 = {1.0, 1.0};
    problem.objective             = {0.0, 0.0};
    problem.obj_scale             = 1.0;
    settings.initial_perturbation = 0;
    settings.primal_tol           = 1e-6;
    settings.iteration_limit      = 20;
    settings.time_limit           = 30;
    settings.log.log              = false;
  }

  bool check(const std::vector<double>& delta_y, const std::vector<double>& delta_z)
  {
    return compute_farkas_certificate(problem,
                                      settings,
                                      delta_y,
                                      delta_z,
                                      certificate_y,
                                      certificate_lower,
                                      certificate_upper,
                                      constant,
                                      work);
  }

  int ratio_test(double direction_value, const std::vector<double>& delta_z)
  {
    sparse_vector_t<int, double> delta_y(1, 1);
    delta_y.i[0] = 0;
    delta_y.x[0] = direction_value;
    std::vector<variable_status_t> status{variable_status_t::BASIC,
                                          variable_status_t::NONBASIC_LOWER};
    std::vector<uint8_t> bounded{1, static_cast<uint8_t>(std::isfinite(problem.upper[1]))};
    std::vector<int> nonbasic{1}, indices{1}, mark{-1, 0}, flips;
    std::vector<double> reduced_cost{0.0, 0.0};
    bound_flipping_ratio_test_t<int, double> ratio(settings,
                                                   tic(),
                                                   1,
                                                   2,
                                                   1.0,
                                                   problem.lower,
                                                   problem.upper,
                                                   bounded,
                                                   status,
                                                   nonbasic,
                                                   reduced_cost,
                                                   delta_z,
                                                   indices,
                                                   mark,
                                                   problem,
                                                   delta_y);
    double step           = 0.0;
    int nonbasic_entering = -1;
    return ratio.compute_step_length(step, nonbasic_entering, flips);
  }

  dual_status_t solve(lp_solution_t<int, double>& solution)
  {
    std::vector<variable_status_t> status{variable_status_t::BASIC,
                                          variable_status_t::NONBASIC_LOWER};
    basis_update_mpf_t<int, double> factorization(1, settings.refactor_frequency);
    std::vector<int> basic(1), nonbasic;
    std::vector<double> norms;
    int iterations = 0;
    return dual_phase2_with_advanced_basis(2,
                                           0,
                                           true,
                                           tic(),
                                           problem,
                                           settings,
                                           status,
                                           factorization,
                                           basic,
                                           nonbasic,
                                           solution,
                                           iterations,
                                           norms,
                                           work);
  }
};

TEST_F(farkas_certificate_test, lower_bound_certificate)
{
  ASSERT_TRUE(check({-1.0}, {1.0, 1.0}));
  EXPECT_EQ(certificate_y, std::vector<double>({-1.0}));
  EXPECT_EQ(certificate_lower, std::vector<double>({1.0, 1.0}));
  EXPECT_EQ(certificate_upper, std::vector<double>({0.0, 0.0}));
  EXPECT_DOUBLE_EQ(constant, 1.0);
}

TEST_F(farkas_certificate_test, upper_bound_certificate)
{
  problem.rhs[0] = 3.0;
  EXPECT_TRUE(check({1.0}, {-1.0, -1.0}));
  EXPECT_DOUBLE_EQ(constant, 1.0);
}

TEST_F(farkas_certificate_test, objective_does_not_affect_certificate)
{
  problem.objective = {123.0, -456.0};
  EXPECT_TRUE(check({-1.0}, {1.0, 1.0}));
}

TEST_F(farkas_certificate_test, zero_margin_is_not_a_certificate)
{
  problem.rhs[0] = 0.0;
  EXPECT_FALSE(check({-1.0}, {1.0, 1.0}));
}

TEST_F(farkas_certificate_test, zero_direction_is_not_a_certificate)
{
  EXPECT_FALSE(check({0.0}, {0.0, 0.0}));
}

TEST_F(farkas_certificate_test, small_residual_with_large_margin)
{
  EXPECT_TRUE(check({-1.0}, {1.0 + 1e-10, 1.0}));
}

TEST_F(farkas_certificate_test, scaled_certificate_below_primal_tol)
{
  EXPECT_FALSE(check({-1e-20}, {1e-20, 1e-20}));
}

TEST_F(farkas_certificate_test, margin_below_primal_tol)
{
  problem.rhs[0] = -std::nextafter(settings.primal_tol, 0.0);
  EXPECT_FALSE(check({-1.0}, {1.0, 1.0}));
}

TEST_F(farkas_certificate_test, margin_at_primal_tol)
{
  problem.rhs[0] = -settings.primal_tol;
  EXPECT_TRUE(check({-1.0}, {1.0, 1.0}));
}

TEST_F(farkas_certificate_test, margin_above_primal_tol)
{
  problem.rhs[0] = -std::nextafter(settings.primal_tol, 1.0);
  EXPECT_TRUE(check({-1.0}, {1.0, 1.0}));
}

TEST_F(farkas_certificate_test, margin_accounts_for_residual_bound)
{
  problem.rhs[0] = -(0.25 + 2.0 * settings.primal_tol);
  EXPECT_TRUE(check({-1.0}, {1.25, 1.0}));
  problem.rhs[0] = -(0.25 + 0.5 * settings.primal_tol);
  EXPECT_FALSE(check({-1.0}, {1.25, 1.0}));
}

TEST_F(farkas_certificate_test, no_multiplier_on_infinite_bound)
{
  problem.rhs[0] = 1.0;
  problem.A.x[1] = 1e-9;
  problem.upper  = {0.0, std::numeric_limits<double>::infinity()};
  EXPECT_FALSE(check({1.0}, {-1.0, -1e-9}));
}

TEST_F(farkas_certificate_test, residual_scaled_by_large_upper_bound)
{
  problem.rhs[0] = 1.0;
  problem.A.x[1] = 1e-9;
  problem.upper  = {0.0, 1e10};
  EXPECT_FALSE(check({1.0}, {-1.0, 0.0}));
}

TEST_F(farkas_certificate_test, residual_scaled_by_large_negative_lower_bound)
{
  problem.rhs[0] = 1.0;
  problem.A.x[1] = -1e-9;
  problem.lower  = {0.0, -1e10};
  problem.upper  = {0.0, 0.0};
  EXPECT_FALSE(check({1.0}, {-1.0, 0.0}));
}

TEST_F(farkas_certificate_test, unbounded_residual_contribution)
{
  problem.rhs[0] = 1.0;
  problem.A.x[1] = 1e-9;
  problem.upper  = {0.0, std::numeric_limits<double>::infinity()};
  EXPECT_FALSE(check({1.0}, {-1.0, 0.0}));
}

TEST_F(farkas_certificate_test, valid_certificate_with_infinite_upper_bounds)
{
  problem.upper.assign(2, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(check({-1.0}, {1.0, 1.0}));
}

TEST_F(farkas_certificate_test, zero_coefficient_on_free_variable)
{
  problem.A.x[1]   = 0.0;
  problem.lower[1] = -std::numeric_limits<double>::infinity();
  problem.upper[1] = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check({-1.0}, {1.0, 0.0}));
}

TEST_F(farkas_certificate_test, ratio_test_certifies_before_relaxing)
{
  EXPECT_EQ(ratio_test(-1.0, {1.0, 1.0}), RATIO_TEST_PRIMAL_INFEASIBLE);
}

TEST_F(farkas_certificate_test, ratio_test_retains_genuine_small_pivot)
{
  problem.rhs[0] = 1.0;
  problem.A.x[1] = 1e-9;
  problem.upper  = {0.0, std::numeric_limits<double>::infinity()};
  EXPECT_EQ(ratio_test(1.0, {-1.0, -1e-9}), 1);
}

TEST_F(farkas_certificate_test, infeasible_solve_uses_certificate)
{
  problem.upper.assign(2, std::numeric_limits<double>::infinity());
  lp_solution_t<int, double> solution(1, 2);
  EXPECT_EQ(solve(solution), dual_status_t::DUAL_UNBOUNDED);
}

TEST_F(farkas_certificate_test, feasible_solve_with_small_pivot)
{
  problem.rhs[0] = 1.0;
  problem.A.x[1] = 1e-9;
  problem.upper  = {0.0, std::numeric_limits<double>::infinity()};
  lp_solution_t<int, double> solution(1, 2);
  EXPECT_EQ(solve(solution), dual_status_t::OPTIMAL);
  EXPECT_NEAR(solution.x[1], 1e9, 1e-5);
}

}  // namespace cuopt::mathematical_optimization::simplex::test
