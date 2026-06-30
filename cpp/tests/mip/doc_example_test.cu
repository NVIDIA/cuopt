/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <filesystem>

namespace cuopt::mathematical_optimization::test {

io::mps_data_model_t<int, double> create_doc_example_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Maximize
  obj: 5 x + 3 y
Subject To
  c1: 2 x + 4 y >= 230
  c2: 3 x + 2 y <= 190
Generals
  x
End
)LP");
}

struct result_map_t {
  std::string file;
  double cost;
};

void test_mps_file()
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem from documentation example
  auto problem = create_doc_example_problem();

  settings.time_limit                  = test_time_limit;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately 303.5
  EXPECT_NEAR(303.5, obj_val, 1.0);

  // Test solution bounds
  test_variable_bounds(problem, solution.get_solution(), settings);

  // Get solution values
  const auto& sol_values = solution.get_solution();
  // x should be approximately 37 and integer
  EXPECT_NEAR(37.0, sol_values.element(0, handle_.get_stream()), 0.1);
  EXPECT_NEAR(std::round(sol_values.element(0, handle_.get_stream())),
              sol_values.element(0, handle_.get_stream()),
              settings.tolerances.integrality_tolerance);  // Check x is integer
  // y should be approximately 39.5
  EXPECT_NEAR(39.5, sol_values.element(1, handle_.get_stream()), 0.1);
}

TEST(docs, mixed_integer_linear_programming) { test_mps_file(); }

TEST(docs, user_problem_file)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem from documentation example
  auto problem = create_doc_example_problem();

  const auto user_problem_path = std::filesystem::temp_directory_path() / "user_problem.mps";
  EXPECT_FALSE(std::filesystem::exists(user_problem_path));

  settings.time_limit        = test_time_limit;
  settings.user_problem_file = user_problem_path;
  settings.presolver         = cuopt::mathematical_optimization::presolver_t::None;
  EXPECT_EQ(solve_mip(&handle_, problem, settings).get_termination_status(),
            mip_termination_status_t::Optimal);

  EXPECT_TRUE(std::filesystem::exists(user_problem_path));

  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> problem2 =
    cuopt::mathematical_optimization::io::read_mps<int, double>(user_problem_path, false);

  EXPECT_EQ(problem2.get_n_variables(), problem.get_n_variables());
  EXPECT_EQ(problem2.get_n_constraints(), problem.get_n_constraints());
  EXPECT_EQ(problem2.get_nnz(), problem.get_nnz());

  const auto user_problem_path2 = std::filesystem::temp_directory_path() / "user_problem2.mps";
  settings.user_problem_file    = user_problem_path2;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem2, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately 303.5
  EXPECT_NEAR(303.5, obj_val, 1.0);

  // Get solution values
  const auto& sol_values = solution.get_solution();
  // x should be approximately 37 and integer
  for (int i = 0; i < problem2.get_n_variables(); i++) {
    if (problem2.get_variable_names()[i] == "x") {
      EXPECT_NEAR(37.0, sol_values.element(i, handle_.get_stream()), 0.1);
      EXPECT_NEAR(std::round(sol_values.element(i, handle_.get_stream())),
                  sol_values.element(i, handle_.get_stream()),
                  settings.tolerances.integrality_tolerance);  // Check x is integer
    } else {                                                   // y should be approximately 39.5
      EXPECT_NEAR(39.5, sol_values.element(i, handle_.get_stream()), 0.1);
    }
  }
}

}  // namespace cuopt::mathematical_optimization::test
