/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "cuopt/mathematical_optimization/mip/solver_settings.hpp"

#include "../utilities/inline_lp_test_utils.hpp"
#include "../utilities/inline_mps_test_utils.hpp"

#include <cuopt/mathematical_optimization/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::test {

struct sc_result_t {
  std::string name;
  std::string mps;
  double objective;
  double sc_value;
};

optimization_problem_t<int, double> make_sc_problem(raft::handle_t const* handle,
                                                    double sc_lb,
                                                    double sc_ub,
                                                    double row_rhs = 1.0,
                                                    double aux_lb  = 0.0,
                                                    double aux_ub  = 1.0)
{
  auto lp = std::format(
    "Minimize\n"
    "  obj: x\n"
    "Subject To\n"
    "  c1: x + y = {}\n"
    "Bounds\n"
    "  {} <= x <= {}\n"
    "  {} <= y <= {}\n"
    "Semi-Continuous\n"
    "  x\n"
    "End\n",
    row_rhs,
    sc_lb,
    sc_ub,
    aux_lb,
    aux_ub);

  auto data = cuopt::test::parse_inline_lp(lp);
  return mps_data_model_to_optimization_problem(handle, data);
}

TEST(mip_solve, semi_continuous_regressions)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 10.;

  const std::vector<sc_result_t> valid_test_instances = {
    {"sc_standard", cuopt::test::inline_mps::sc_standard_mps, 8., 0.},
    {"sc_no_ub", cuopt::test::inline_mps::sc_no_ub_mps, 8., 0.},
    {"sc_lb_zero", cuopt::test::inline_mps::sc_lb_zero_mps, 8., 0.},
    {"sc_inferred_ub", cuopt::test::inline_mps::sc_inferred_ub_mps, -4., 4.},
  };

  for (const auto& test_instance : valid_test_instances) {
    auto problem  = cuopt::test::inline_mps::parse_inline_mps(test_instance.mps);
    auto solution = solve_mip(&handle_, problem, settings);

    EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal)
      << test_instance.name;
    ASSERT_EQ(solution.get_solution().size(), static_cast<size_t>(problem.get_n_variables()))
      << test_instance.name;

    auto host_solution =
      cuopt::host_copy(solution.get_solution(), solution.get_solution().stream());
    EXPECT_NEAR(solution.get_objective_value(), test_instance.objective, 1e-6)
      << test_instance.name;
    EXPECT_NEAR(host_solution[0], test_instance.sc_value, 1e-6) << test_instance.name;
  }
}

TEST(mip_solve, semi_continuous_invalid_bounds_rejected)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 10.;

  const std::vector<std::pair<double, double>> invalid_bounds = {
    {-3.0, 5.0},
    {-5.0, -1.0},
    {-4.0, 0.0},
    {6.0, 5.0},
  };

  for (const auto& [lb, ub] : invalid_bounds) {
    SCOPED_TRACE(::testing::Message() << "bounds=[" << lb << ", " << ub << "]");
    auto problem = make_sc_problem(&handle_, lb, ub);

    auto solution     = solve_mip(problem, settings);
    const auto& error = solution.get_error_status();
    EXPECT_EQ(error.get_error_type(), cuopt::error_type_t::ValidationError);
    EXPECT_NE(std::string(error.what()).find("Semi-continuous variable"), std::string::npos);
  }
}

TEST(mip_solve, semi_continuous_equal_bounds_supported)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 10.;

  {
    auto problem  = make_sc_problem(&handle_, 5.0, 5.0);
    auto solution = solve_mip(problem, settings);

    EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);
    auto host_solution =
      cuopt::host_copy(solution.get_solution(), solution.get_solution().stream());
    EXPECT_NEAR(solution.get_objective_value(), 0.0, 1e-6);
    EXPECT_NEAR(host_solution[0], 0.0, 1e-6);
  }

  {
    auto problem  = make_sc_problem(&handle_, 5.0, 5.0, 5.0, 0.0, 0.0);
    auto solution = solve_mip(problem, settings);

    EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);
    auto host_solution =
      cuopt::host_copy(solution.get_solution(), solution.get_solution().stream());
    EXPECT_NEAR(solution.get_objective_value(), 5.0, 1e-6);
    EXPECT_NEAR(host_solution[0], 5.0, 1e-6);
  }
}

}  // namespace cuopt::mathematical_optimization::test
