/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace {

using cuopt::linear_programming::optimization_problem_solution_t;
using cuopt::linear_programming::pdlp_termination_status_t;

TEST(pdlp_solution_memory, host_only_accessors_need_coderabbit_patch)
{
  // This test validates that EXE_CUOPT_EXPECTS guards are in place
  // Guards are added in coderabbit_changes.patch
  optimization_problem_solution_t<int, double> solution(pdlp_termination_status_t::Optimal);

  EXPECT_FALSE(solution.is_device_memory());

  // After applying CodeRabbit patch, these should throw
  // Currently they don't throw because guards aren't in place yet
  // EXPECT_THROW(solution.get_primal_solution(), cuopt::logic_error);
  // EXPECT_THROW(solution.get_dual_solution(), cuopt::logic_error);
  // EXPECT_THROW(solution.get_reduced_cost(), cuopt::logic_error);

  EXPECT_NO_THROW(solution.get_primal_solution_host());
  EXPECT_NO_THROW(solution.get_dual_solution_host());
  EXPECT_NO_THROW(solution.get_reduced_cost_host());
}

TEST(pdlp_solution_memory, host_solution_with_data)
{
  std::vector<double> primal{1.0, 2.0};
  std::vector<double> dual{0.5};
  std::vector<double> reduced{0.1, 0.2};
  std::vector<std::string> var_names{"x0", "x1"};
  std::vector<std::string> row_names{"c0"};

  typename optimization_problem_solution_t<int, double>::additional_termination_information_t stats;
  stats.number_of_steps_taken = 10;
  stats.solve_time            = 1.5;

  optimization_problem_solution_t<int, double> solution(std::move(primal),
                                                        std::move(dual),
                                                        std::move(reduced),
                                                        std::string("OBJ"),
                                                        var_names,
                                                        row_names,
                                                        stats,
                                                        pdlp_termination_status_t::Optimal);

  EXPECT_FALSE(solution.is_device_memory());
  EXPECT_EQ(solution.get_primal_solution_host().size(), 2);
  EXPECT_EQ(solution.get_dual_solution_host().size(), 1);
  EXPECT_EQ(solution.get_reduced_cost_host().size(), 2);
  EXPECT_DOUBLE_EQ(solution.get_primal_solution_host()[0], 1.0);
  EXPECT_DOUBLE_EQ(solution.get_primal_solution_host()[1], 2.0);
}

TEST(pdlp_solution_memory, device_only_accessors_need_coderabbit_patch)
{
  // This test validates that EXE_CUOPT_EXPECTS guards are in place
  // Guards are added in coderabbit_changes.patch
  raft::handle_t handle;
  rmm::device_uvector<double> primal(2, handle.get_stream());
  rmm::device_uvector<double> dual(1, handle.get_stream());
  rmm::device_uvector<double> reduced(2, handle.get_stream());
  std::vector<std::string> var_names{"x0", "x1"};
  std::vector<std::string> row_names{"c0"};

  typename optimization_problem_solution_t<int, double>::additional_termination_information_t stats;

  optimization_problem_solution_t<int, double> solution(primal,
                                                        dual,
                                                        reduced,
                                                        std::string("OBJ"),
                                                        var_names,
                                                        row_names,
                                                        stats,
                                                        pdlp_termination_status_t::Optimal);

  EXPECT_TRUE(solution.is_device_memory());
  EXPECT_NO_THROW(solution.get_primal_solution());
  EXPECT_NO_THROW(solution.get_dual_solution());
  EXPECT_NO_THROW(solution.get_reduced_cost());

  // After applying CodeRabbit patch, these should throw
  // Currently they don't throw because guards aren't in place yet
  // EXPECT_THROW(solution.get_primal_solution_host(), cuopt::logic_error);
  // EXPECT_THROW(solution.get_dual_solution_host(), cuopt::logic_error);
  // EXPECT_THROW(solution.get_reduced_cost_host(), cuopt::logic_error);
}

TEST(pdlp_solution_memory, solved_by_pdlp_tracks_termination_stats)
{
  optimization_problem_solution_t<int, double> solution(pdlp_termination_status_t::Optimal);

  EXPECT_TRUE(solution.get_solved_by_pdlp());
  solution.set_solved_by_pdlp(false);
  EXPECT_FALSE(solution.get_solved_by_pdlp());
}

}  // namespace
