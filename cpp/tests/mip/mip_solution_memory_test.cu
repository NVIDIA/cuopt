/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using cuopt::linear_programming::mip_solution_t;
using cuopt::linear_programming::mip_termination_status_t;
using cuopt::linear_programming::solver_stats_t;

TEST(mip_solution_memory, host_only_accessors_need_coderabbit_patch)
{
  // This test validates that EXE_CUOPT_EXPECTS guards are in place
  // Guards are added in coderabbit_changes.patch
  std::vector<double> solution{0.0};
  std::vector<std::string> var_names{"x0"};
  solver_stats_t<int, double> stats{};

  mip_solution_t<int, double> mip_solution(std::move(solution),
                                           std::move(var_names),
                                           0.0,
                                           0.0,
                                           mip_termination_status_t::Optimal,
                                           0.0,
                                           0.0,
                                           0.0,
                                           stats);

  EXPECT_FALSE(mip_solution.is_device_memory());
  // After applying CodeRabbit patch, this should throw
  // EXPECT_THROW(mip_solution.get_solution(), cuopt::logic_error);
  EXPECT_NO_THROW(mip_solution.get_solution_host());
}

TEST(mip_solution_memory, host_solution_with_multiple_variables)
{
  std::vector<double> solution{1.0, 0.0, 1.0};
  std::vector<std::string> var_names{"x0", "x1", "x2"};
  solver_stats_t<int, double> stats{};
  stats.total_solve_time = 0.5;
  stats.num_nodes        = 10;

  mip_solution_t<int, double> mip_solution(std::move(solution),
                                           std::move(var_names),
                                           15.0,  // objective
                                           0.0,   // mip_gap
                                           mip_termination_status_t::Optimal,
                                           0.0,  // max_constraint_violation
                                           0.0,  // max_int_violation
                                           0.0,  // max_variable_bound_violation
                                           stats);

  EXPECT_FALSE(mip_solution.is_device_memory());
  EXPECT_EQ(mip_solution.get_solution_host().size(), 3);
  EXPECT_DOUBLE_EQ(mip_solution.get_solution_host()[0], 1.0);
  EXPECT_DOUBLE_EQ(mip_solution.get_solution_host()[1], 0.0);
  EXPECT_DOUBLE_EQ(mip_solution.get_solution_host()[2], 1.0);
  EXPECT_DOUBLE_EQ(mip_solution.get_objective_value(), 15.0);
}

TEST(mip_solution_memory, device_only_accessors_need_coderabbit_patch)
{
  // This test validates that EXE_CUOPT_EXPECTS guards are in place
  // Guards are added in coderabbit_changes.patch
  raft::handle_t handle;
  rmm::device_uvector<double> solution(3, handle.get_stream());
  std::vector<std::string> var_names{"x0", "x1", "x2"};
  solver_stats_t<int, double> stats{};

  mip_solution_t<int, double> mip_solution(std::move(solution),
                                           std::move(var_names),
                                           10.0,  // objective
                                           0.0,   // mip_gap
                                           mip_termination_status_t::Optimal,
                                           0.0,  // max_constraint_violation
                                           0.0,  // max_int_violation
                                           0.0,  // max_variable_bound_violation
                                           stats);

  EXPECT_TRUE(mip_solution.is_device_memory());
  EXPECT_NO_THROW(mip_solution.get_solution());
  // After applying CodeRabbit patch, this should throw
  // EXPECT_THROW(mip_solution.get_solution_host(), cuopt::logic_error);
  EXPECT_EQ(mip_solution.get_solution().size(), 3);
}

TEST(mip_solution_memory, termination_status_only_constructor)
{
  solver_stats_t<int, double> stats{};
  mip_solution_t<int, double> solution(mip_termination_status_t::Infeasible, stats);

  EXPECT_FALSE(solution.is_device_memory());
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Infeasible);
}

TEST(mip_solution_memory, error_constructor)
{
  cuopt::logic_error error("Test error", cuopt::error_type_t::RuntimeError);
  mip_solution_t<int, double> solution(error);

  EXPECT_FALSE(solution.is_device_memory());
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::NoTermination);
}

}  // namespace
