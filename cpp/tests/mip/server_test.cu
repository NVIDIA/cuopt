/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

namespace cuopt::linear_programming::test {

io::mps_data_model_t<int, double> create_std_lp_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 1.2 x1 + 1.7 x2
Subject To
  c1_ub: x1 + x2 <= 5000
  c1_lb: x1 + x2 >= 0
Bounds
  0 <= x1 <= 3000
  0 <= x2 <= 5000
End
)LP");
}

io::mps_data_model_t<int, double> create_std_milp_problem(bool maximize)
{
  auto problem = create_std_lp_problem();
  problem.set_maximize(maximize);
  std::vector<char> var_types = {'I', 'C'};
  problem.set_variable_types(var_types);
  return problem;
}

TEST(ServerTest, TestSampleLP)
{
  raft::handle_t handle;
  auto problem = create_std_lp_problem();

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-4);
  settings.set_time_limit(5);

  auto result = cuopt::linear_programming::solve_lp(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::linear_programming::pdlp_termination_status_t::Optimal);
}

class MILPTestParams
  : public testing::TestWithParam<
      std::tuple<bool, int, bool, cuopt::linear_programming::mip_termination_status_t>> {};

TEST_P(MILPTestParams, TestSampleMILP)
{
  bool maximize                    = std::get<0>(GetParam());
  int scaling                      = std::get<1>(GetParam());
  bool heuristics_only             = std::get<2>(GetParam());
  auto expected_termination_status = std::get<3>(GetParam());

  raft::handle_t handle;
  auto problem = create_std_milp_problem(maximize);

  cuopt::linear_programming::mip_solver_settings_t<int, double> settings{};
  settings.time_limit      = 5;
  settings.mip_scaling     = scaling;
  settings.heuristics_only = heuristics_only;

  auto result = cuopt::linear_programming::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(), expected_termination_status);
}

INSTANTIATE_TEST_SUITE_P(
  MILPTests,
  MILPTestParams,
  testing::Values(
    std::make_tuple(true,
                    CUOPT_MIP_SCALING_ON,
                    true,
                    cuopt::linear_programming::mip_termination_status_t::FeasibleFound),
    std::make_tuple(false,
                    CUOPT_MIP_SCALING_ON,
                    false,
                    cuopt::linear_programming::mip_termination_status_t::Optimal),
    std::make_tuple(true,
                    CUOPT_MIP_SCALING_OFF,
                    true,
                    cuopt::linear_programming::mip_termination_status_t::FeasibleFound),
    std::make_tuple(false,
                    CUOPT_MIP_SCALING_OFF,
                    false,
                    cuopt::linear_programming::mip_termination_status_t::Optimal)));

}  // namespace cuopt::linear_programming::test
