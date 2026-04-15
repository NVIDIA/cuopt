/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "cuopt/linear_programming/mip/solver_settings.hpp"

#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::test {

struct sc_result_t {
  std::string file;
  double objective;
  double sc_value;
};

optimization_problem_t<int, double> make_sc_problem(raft::handle_t const* handle,
                                                    double sc_lb,
                                                    double sc_ub)
{
  optimization_problem_t<int, double> problem(handle);

  const std::vector<double> coefficients = {1.0, 1.0};
  const std::vector<int> indices         = {0, 1};
  const std::vector<int> offsets         = {0, 2};
  const std::vector<double> row_lower    = {1.0};
  const std::vector<double> row_upper    = {1.0};
  const std::vector<double> obj          = {1.0, 0.0};
  const std::vector<double> var_lower    = {sc_lb, 0.0};
  const std::vector<double> var_upper    = {sc_ub, 1.0};
  const std::vector<var_t> var_types     = {var_t::SEMI_CONTINUOUS, var_t::CONTINUOUS};

  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());
  problem.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  problem.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  problem.set_objective_coefficients(obj.data(), obj.size());
  problem.set_variable_lower_bounds(var_lower.data(), var_lower.size());
  problem.set_variable_upper_bounds(var_upper.data(), var_upper.size());
  problem.set_variable_types(var_types.data(), var_types.size());

  return problem;
}

TEST(mip_solve, semi_continuous_regressions)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 10.;

  const std::vector<sc_result_t> valid_test_instances = {
    {"mip/sc_standard.mps", 8., 0.}, {"mip/sc_no_ub.mps", 8., 0.}, {"mip/sc_lb_zero.mps", 8., 0.}};

  for (const auto& test_instance : valid_test_instances) {
    auto path     = make_path_absolute(test_instance.file);
    auto problem  = cuopt::mps_parser::parse_mps<int, double>(path, false);
    auto solution = solve_mip(&handle_, problem, settings);

    EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal)
      << test_instance.file;
    ASSERT_EQ(solution.get_solution().size(), static_cast<size_t>(problem.get_n_variables()))
      << test_instance.file;

    auto host_solution =
      cuopt::host_copy(solution.get_solution(), solution.get_solution().stream());
    EXPECT_NEAR(solution.get_objective_value(), test_instance.objective, 1e-6)
      << test_instance.file;
    EXPECT_NEAR(host_solution[0], test_instance.sc_value, 1e-6) << test_instance.file;
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
    {5.0, 5.0},
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

  {
    SCOPED_TRACE("mip/sc_both_neg.mps");
    auto path     = make_path_absolute("mip/sc_both_neg.mps");
    auto problem  = cuopt::mps_parser::parse_mps<int, double>(path, false);
    auto solution = solve_mip(&handle_, problem, settings);

    const auto& error = solution.get_error_status();
    EXPECT_EQ(error.get_error_type(), cuopt::error_type_t::ValidationError);
    EXPECT_NE(std::string(error.what()).find("Semi-continuous variable"), std::string::npos);
  }
}

}  // namespace cuopt::linear_programming::test
