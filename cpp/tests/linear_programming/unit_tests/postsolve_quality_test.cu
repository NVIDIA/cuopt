/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <fstream>
#include <limits>
#include <pdlp/postsolve_quality.cuh>
#include <raft/core/handle.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cuopt::mathematical_optimization::test {
TEST(PostsolveQuality, OriginalBoxedModelAndProjectedDuals)
{
  raft::handle_t handle;
  const auto stream = handle.get_stream();
  const double inf  = std::numeric_limits<double>::infinity();
  for (bool quadratic : {false, true}) {
    optimization_problem_t<int, double> model(&handle);
    const double a[]{1, 2, 2}, c[]{1000, 0}, lower[]{0, 0}, upper[]{2, inf};
    const double row_lower[]{2, -inf}, row_upper[]{4, 5};
    const int indices[]{0, 1, 0}, offsets[]{0, 2, 3};
    model.set_csr_constraint_matrix(a, 3, indices, 3, offsets, 3);
    model.set_objective_coefficients(c, 2);
    model.set_variable_lower_bounds(lower, 2);
    model.set_variable_upper_bounds(upper, 2);
    model.set_constraint_lower_bounds(row_lower, 2);
    model.set_constraint_upper_bounds(row_upper, 2);
    if (quadratic) {
      const double q[]{1, .5, .5, 1};
      const int qi[]{0, 1, 0, 1}, qo[]{0, 2, 4};
      model.set_quadratic_objective_matrix(q, 4, qi, 4, qo, 3);
    }
    auto x             = cuopt::device_copy(std::vector<double>{3, -1}, stream);
    auto y             = cuopt::device_copy(std::vector<double>{2, 3}, stream);
    auto z             = cuopt::device_copy(std::vector<double>{-2, -4}, stream);
    const auto metrics = detail::evaluate_postsolve_quality(model, x, y, z, true);
    // Ax=(1,6): both rows violate by 1. x=(3,-1): both variable bounds violate by 1.
    // b=(4,5), ||c||=1000; sign projection gives y_hat=(2,0), z_hat=(-2,0).
    // LP dual error=(1000,-4); QP Hx=(5,1) adds to give (1005,-3).
    EXPECT_DOUBLE_EQ(metrics.row_l2, std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(metrics.bound_l2, std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(metrics.primal_relative, std::sqrt(2.0) / (1 + std::sqrt(41.0)));
    EXPECT_TRUE(metrics.dual_available);
    EXPECT_DOUBLE_EQ(metrics.dual_projection_l2, 5.0);
    const double expected_dual = quadratic ? std::hypot(1005.0, 3.0) : std::hypot(1000.0, 4.0);
    EXPECT_DOUBLE_EQ(metrics.dual_l2, expected_dual);
    EXPECT_DOUBLE_EQ(metrics.dual_relative, expected_dual / 1001.0);
    EXPECT_EQ(cuopt::host_copy(y, stream), (std::vector<double>{2, 3}));
    EXPECT_EQ(cuopt::host_copy(z, stream), (std::vector<double>{-2, -4}));
    const auto missing = detail::evaluate_postsolve_quality(model, x, y, z, false);
    EXPECT_FALSE(missing.dual_available);
    EXPECT_DOUBLE_EQ(missing.row_l2, metrics.row_l2);
    EXPECT_TRUE(std::isnan(missing.dual_l2));
  }
}

TEST(PostsolveQuality, NonFiniteViolationCannotLookFeasible)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isinf(detail::postsolve_interval_violation(nan, 0.0, 1.0)));
  EXPECT_TRUE(std::isinf(detail::postsolve_interval_violation(0.0, nan, 1.0)));
}
TEST(PostsolveQuality, LogsOriginalRowsAfterBothPresolvers)
{
  raft::handle_t handle;
  for (auto presolver : {presolver_t::Papilo, presolver_t::PSLP}) {
    for (bool dual_postsolve : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "presolver=" << static_cast<int>(presolver)
                                        << " dual_postsolve=" << dual_postsolve);
      optimization_problem_t<int, double> model(&handle);
      std::vector<double> a{4, 1, 1, 2, 1, 4, 2, 1}, rhs{3.5, 3.25};
      std::vector<int> indices{0, 1, 2, 3, 0, 1, 2, 3}, offsets{0, 4, 8};
      for (int i = 0; i < 4; ++i) {
        a.insert(a.end(), {4, 1, 1, 2});
        indices.insert(indices.end(), {0, 1, 2, 3});
        offsets.push_back(a.size());
        rhs.push_back(3.5);
      }
      const double c[]{1.5, 2.25, 2.25, 0}, lower[]{0, 0, 0, 0}, upper[]{1, 1, 1, 1};
      model.set_csr_constraint_matrix(
        a.data(), a.size(), indices.data(), indices.size(), offsets.data(), offsets.size());
      model.set_objective_coefficients(c, 4);
      model.set_variable_lower_bounds(lower, 4);
      model.set_variable_upper_bounds(upper, 4);
      model.set_constraint_lower_bounds(rhs.data(), rhs.size());
      model.set_constraint_upper_bounds(rhs.data(), rhs.size());
      char log_path[] = "/tmp/cuopt-postsolve-log-XXXXXX";
      const int fd    = mkstemp(log_path);
      ASSERT_GE(fd, 0);
      close(fd);
      pdlp_solver_settings_t<int, double> settings;
      settings.method         = method_t::PDLP;
      settings.presolver      = presolver;
      settings.dual_postsolve = dual_postsolve;
      settings.set_optimality_tolerance(1e-8);
      settings.log_to_console = false;
      settings.log_file       = log_path;
      const cuopt::timer_t wall_timer(100);
      auto solution          = solve_lp(model, settings);
      const double wall_time = wall_timer.elapsed_time();
      std::ifstream input(log_path);
      std::ostringstream content;
      content << input.rdbuf();
      input.close();
      std::remove(log_path);
      ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
      const std::string log = content.str();
      const auto unscale    = log.find("Unscale:");
      const auto postsolve  = log.find("Postsolve:");
      const auto final      = log.find("Final solution quality (original model; L2 norms)");
      ASSERT_NE(unscale, std::string::npos);
      ASSERT_NE(postsolve, std::string::npos);
      ASSERT_NE(final, std::string::npos);
      EXPECT_LT(unscale, postsolve);
      EXPECT_LT(postsolve, final);
      const auto summary = log.find("Final Status: Optimal");
      ASSERT_NE(summary, std::string::npos);
      EXPECT_LT(final, summary);
      EXPECT_EQ(log.find("Final Status:", summary + 1), std::string::npos);
      const auto x = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
      const auto objective_pos = log.find("Objective:", summary);
      ASSERT_NE(objective_pos, std::string::npos);
      const double objective =
        std::stod(log.substr(objective_pos + std::string("Objective:").size()));
      EXPECT_NEAR(objective, 1.5 * x[0] + 2.25 * x[1] + 2.25 * x[2], 1e-8);
      const auto iterations_pos = log.find("Iterations:", summary);
      ASSERT_NE(iterations_pos, std::string::npos);
      EXPECT_EQ(std::stoi(log.substr(iterations_pos + std::string("Iterations:").size())),
                solution.get_additional_termination_information().number_of_steps_taken);
      const auto time_pos = log.find("Time:", summary);
      ASSERT_NE(time_pos, std::string::npos);
      const double logged_time = std::stod(log.substr(time_pos + std::string("Time:").size()));
      // Time is rounded to milliseconds. It must cover the internal solve and
      // fit within the independently measured complete API call.
      EXPECT_GE(logged_time + 0.00051,
                solution.get_additional_termination_information().solve_time);
      EXPECT_LE(logged_time, wall_time + 0.00051);
      const double e0         = 4 * x[0] + x[1] + x[2] + 2 * x[3] - 3.5;
      const double e1         = x[0] + 4 * x[1] + 2 * x[2] + x[3] - 3.25;
      const double expected   = std::sqrt(5 * e0 * e0 + e1 * e1);
      const std::string label = "Row violation (L2):";
      const auto row          = log.find(label, final);
      ASSERT_NE(row, std::string::npos);
      const double actual = std::stod(log.substr(row + label.size()));
      EXPECT_NEAR(actual, expected, 1e-14);
      if (!dual_postsolve) {
        EXPECT_NE(log.find("Dual residual unavailable", final), std::string::npos);
      } else {
        EXPECT_NE(log.find("Dual residual (projected y/z", final), std::string::npos);
      }
    }
  }
}

TEST(PostsolveQuality, FinalStatusUsesOriginalObjectiveAndWallTime)
{
  raft::handle_t handle;
  using solution_t = optimization_problem_solution_t<int, double>;
  for (bool quadratic : {false, true}) {
    for (bool maximize : {false, true}) {
      optimization_problem_t<int, double> model(&handle);
      const double sign = maximize ? -1 : 1;
      const double a[]{1, 1}, c[]{-4 * sign, -sign}, lower[]{0, 0}, upper[]{10, 10}, rhs[]{2};
      const int indices[]{0, 1}, offsets[]{0, 2};
      model.set_csr_constraint_matrix(a, 2, indices, 2, offsets, 2);
      model.set_objective_coefficients(c, 2);
      model.set_variable_lower_bounds(lower, 2);
      model.set_variable_upper_bounds(upper, 2);
      model.set_constraint_lower_bounds(rhs, 1);
      model.set_constraint_upper_bounds(rhs, 1);
      model.set_objective_offset(7);
      model.set_objective_scaling_factor(2);
      model.set_maximize(maximize);
      if (quadratic) {
        const double q[]{sign, 0.5 * sign, 0.5 * sign, sign};
        const int qi[]{0, 1, 0, 1}, qo[]{0, 2, 4};
        model.set_quadratic_objective_matrix(q, 4, qi, 4, qo, 3);
      }
      auto x = cuopt::device_copy(std::vector<double>{1, 1}, handle.get_stream());
      auto y = cuopt::device_copy(std::vector<double>{0}, handle.get_stream());
      auto z = cuopt::device_copy(std::vector<double>{0, 0}, handle.get_stream());
      for (auto status : {pdlp_termination_status_t::Optimal,
                          pdlp_termination_status_t::TimeLimit,
                          pdlp_termination_status_t::IterationLimit,
                          pdlp_termination_status_t::PrimalFeasible,
                          pdlp_termination_status_t::PrimalInfeasible}) {
        solution_t::additional_termination_information_t info;
        info.primal_objective      = 999;
        info.solve_time            = 123;
        info.number_of_steps_taken = 54;
        solution_t solution(x, y, z, "", {}, {}, info, status, &handle, true);
        char log_path[] = "/tmp/cuopt-final-status-XXXXXX";
        const int fd    = mkstemp(log_path);
        ASSERT_GE(fd, 0);
        close(fd);
        const cuopt::timer_t timer(100);
        {
          init_logger_t logger(log_path, false);
          // Pending stream work must be included even when status is non-Optimal
          // and the summary does not copy/evaluate x. Never use the stale 123s field.
          ASSERT_EQ(cudaLaunchHostFunc(
                      handle.get_stream().get(),
                      [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); },
                      nullptr),
                    cudaSuccess);
          detail::print_final_status(model, solution, timer);
        }
        const double wall_time = timer.elapsed_time();
        std::ifstream input(log_path);
        std::string line;
        std::getline(input, line);
        input.close();
        std::remove(log_path);
        ASSERT_NE(line.find("Final Status: " + solution.get_termination_status_string()),
                  std::string::npos);
        const auto objective_pos = line.find("Objective:");
        ASSERT_NE(objective_pos, std::string::npos);
        const double objective =
          std::stod(line.substr(objective_pos + std::string("Objective:").size()));
        if (status == pdlp_termination_status_t::Optimal) {
          // x=(1,1): c^T x=-5*sign and x^T Qx=3*sign.
          EXPECT_DOUBLE_EQ(objective, 2 * (sign * (quadratic ? -2 : -5) + 7));
        } else {
          EXPECT_TRUE(std::isnan(objective));
        }
        EXPECT_NE(line.find("Iterations: 54"), std::string::npos);
        const auto time_pos = line.find("Time:");
        ASSERT_NE(time_pos, std::string::npos);
        const double logged_time = std::stod(line.substr(time_pos + std::string("Time:").size()));
        EXPECT_GE(logged_time, 0.019);
        EXPECT_LE(logged_time, wall_time + 0.00051);
        EXPECT_DOUBLE_EQ(solution.get_additional_termination_information().primal_objective, 999);
        EXPECT_DOUBLE_EQ(solution.get_additional_termination_information().solve_time, 123);
      }
    }
  }
}

TEST(PostsolveQuality, FinalStatusOnPresolveTerminalResults)
{
  raft::handle_t handle;
  for (auto presolver : {presolver_t::Papilo, presolver_t::PSLP}) {
    for (bool infeasible : {false, true}) {
      optimization_problem_t<int, double> model(&handle);
      const double a[]{1}, c[]{3}, bound[]{2};
      const double lower[]{-std::numeric_limits<double>::infinity()},
        upper[]{infeasible ? 1.0 : 5.0};
      const int indices[]{0}, offsets[]{0, 1};
      model.set_csr_constraint_matrix(a, 1, indices, 1, offsets, 2);
      model.set_objective_coefficients(c, 1);
      model.set_variable_lower_bounds(bound, 1);
      model.set_variable_upper_bounds(bound, 1);
      model.set_constraint_lower_bounds(lower, 1);
      model.set_constraint_upper_bounds(upper, 1);
      model.set_objective_offset(7);
      model.set_objective_scaling_factor(2);
      char log_path[] = "/tmp/cuopt-presolve-final-XXXXXX";
      const int fd    = mkstemp(log_path);
      ASSERT_GE(fd, 0);
      close(fd);
      pdlp_solver_settings_t<int, double> settings;
      settings.method         = method_t::PDLP;
      settings.presolver      = presolver;
      settings.log_file       = log_path;
      settings.log_to_console = false;
      auto solution           = solve_lp(model, settings);
      std::ifstream input(log_path);
      std::ostringstream content;
      content << input.rdbuf();
      input.close();
      std::remove(log_path);
      const auto log     = content.str();
      const auto summary = log.find("Final Status:");
      ASSERT_NE(summary, std::string::npos);
      EXPECT_EQ(log.find("Final Status:", summary + 1), std::string::npos);
      if (infeasible) {
        EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::PrimalInfeasible);
        EXPECT_NE(log.find("Objective: nan", summary), std::string::npos);
      } else {
        EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
        EXPECT_LT(log.find("Final solution quality"), summary);
        EXPECT_NE(log.find("Objective: 2.60000000e+01", summary), std::string::npos);
        EXPECT_NE(log.find("Iterations: 0", summary), std::string::npos);
      }
    }
  }
}

}  // namespace cuopt::mathematical_optimization::test
