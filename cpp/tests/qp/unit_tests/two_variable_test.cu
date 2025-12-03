/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/common_utils.hpp>

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip/problem/problem.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

TEST(two_variable_test, simple_test)
{
  raft::handle_t handle;

  // optimize: -8x1 - 16x2 + x1^2 + 4x1x2 + x2^2

  // Constraints set through row types
  auto op_problem    = optimization_problem_t<int, double>(&handle);
  double A_host[]    = {1.0, 1.0};
  int indices_host[] = {0, 1};
  int offset_host[]  = {0, 2};

  op_problem.set_csr_constraint_matrix(A_host, 2, indices_host, 2, offset_host, 2);

  double cnstr_lb_host[] = {5.0};
  // double cnstr_ub_host[] = {1000.0};
  double cnstr_ub_host[] = {std::numeric_limits<double>::infinity()};
  op_problem.set_constraint_lower_bounds(cnstr_lb_host, 1);
  op_problem.set_constraint_upper_bounds(cnstr_ub_host, 1);

  double lb_host[] = {0.0, 0.0};
  double ub_host[] = {10.0, 10.0};
  op_problem.set_variable_lower_bounds(lb_host, 2);
  op_problem.set_variable_upper_bounds(ub_host, 2);

  double c_host[] = {-8.0, -16.0};
  op_problem.set_objective_coefficients(c_host, 2);

  double Q_values_host[] = {1.0, 4.0};
  int Q_indices_host[]   = {0, 1};
  int Q_offsets_host[]   = {0, 1, 2};
  op_problem.set_quadratic_objective_matrix(Q_values_host, 2, Q_indices_host, 2, Q_offsets_host, 3);

  auto settings = cuopt::linear_programming::pdlp_solver_settings_t<int, double>();
  auto solution = cuopt::linear_programming::solve_lp(op_problem, settings);

  EXPECT_EQ(solution.get_termination_status(),
            cuopt::linear_programming::pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), -32.0, 1e-6);

  auto sol_vec = cuopt::host_copy(solution.get_primal_solution());
  EXPECT_NEAR(sol_vec[0], 4.0, 1e-6);
  EXPECT_NEAR(sol_vec[1], 2.0, 1e-6);
}
}  // namespace cuopt::linear_programming
