/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver.cuh>
#include <pdlp/utils.cuh>
#include <presolve/bounds_presolve.cuh>
#include <presolve/third_party_presolve.hpp>
#include <presolve/trivial_presolve.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>
#include <utilities/inline_lp_test_utils.hpp>
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::test {

TEST(problem, find_implied_integers)
{
  const raft::handle_t handle_{};

  auto path           = make_path_absolute("mip/fiball.mps");
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  auto op_problem     = mps_data_model_to_optimization_problem(&handle_, mps_data_model);
  auto presolver      = std::make_unique<mip::third_party_presolve_t<int, double>>();
  auto result         = presolver->apply_presolve_from_op_problem(
    op_problem,
    cuopt::mathematical_optimization::problem_category_t::MIP,
    cuopt::mathematical_optimization::presolver_t::Papilo,
    false,
    1e-6,
    1e-12,
    20,
    1);
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::UNBNDORINFEAS);

  auto problem = mip::problem_t<int, double>(result.reduced_problem);
  problem.set_implied_integers(result.implied_integer_indices);
  ASSERT_TRUE(result.implied_integer_indices.size() > 0);
  auto var_types = host_copy(problem.variable_types, handle_.get_stream());
  // Find the index of the one continuous variable
  auto it = std::find_if(var_types.begin(), var_types.end(), [](var_t var_type) {
    return var_type == var_t::CONTINUOUS;
  });
  ASSERT_NE(it, var_types.end());
  ASSERT_EQ(problem.presolve_data.var_flags.size(), var_types.size());
  // Ensure it is an implied integer
  EXPECT_EQ(problem.presolve_data.var_flags.element(it - var_types.begin(), handle_.get_stream()),
            ((int)mip::problem_t<int, double>::var_flags_t::VAR_IMPLIED_INTEGER));
}

TEST(presolve, substitution_cancellation_preserves_binary_feasibility)
{
  const raft::handle_t handle{};
  auto model      = cuopt::test::parse_inline_lp(R"LP(
Minimize
 obj: x
Subject To
 link: x - 1.4419880250174197 z = -6.2758387909538564
 choice: z + w = 1
Bounds
 -7 <= x <= -4
Binaries
 z w
End
)LP");
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
  mip::problem_t<int, double> problem(op_problem);
  ASSERT_EQ(problem.n_variables, 3);
  problem.preprocess_problem();

  // Probing fixes x at these endpoints for z=0 and z=1. Substitution into link
  // leaves -1.11e-15*z = -8.88e-16; dividing that roundoff would fix z to 0.8.
  problem.substitute_variables({0}, {1}, {-6.2758387909538556}, {1.4419880250174186});
  mip::trivial_presolve(problem, true);

  mip_solver_settings_t<int, double> settings;
  mip::mip_solver_t<int, double> solver(problem, settings, timer_t(10));
  mip::bound_presolve_t<int, double> bounds(solver.context);
  bounds.solve(problem);
  EXPECT_EQ(bounds.infeas_constraints_count, 0);
  const auto lower = host_copy(bounds.upd.lb, handle.get_stream());
  const auto upper = host_copy(bounds.upd.ub, handle.get_stream());
  ASSERT_EQ(lower.size(), 2);
  for (int j = 0; j < 2; ++j) {
    EXPECT_DOUBLE_EQ(lower[j], 0.0);
    EXPECT_DOUBLE_EQ(upper[j], 1.0);
  }
}

TEST(presolve, substitution_preserves_significant_residual)
{
  const raft::handle_t handle{};
  auto model      = cuopt::test::parse_inline_lp(R"LP(
Minimize
 obj: x
Subject To
 link: x - 1.4 z = -6.2
 choice: z + w = 1
Bounds
 -7 <= x <= -4
Binaries
 z w
End
)LP");
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  // The residual 0.1*z = 0 is meaningful and must still force z=0.
  problem.substitute_variables({0}, {1}, {-6.2}, {1.5});
  mip::trivial_presolve(problem, true);

  mip_solver_settings_t<int, double> settings;
  mip::mip_solver_t<int, double> solver(problem, settings, timer_t(10));
  mip::bound_presolve_t<int, double> bounds(solver.context);
  bounds.solve(problem);
  EXPECT_EQ(bounds.infeas_constraints_count, 0);
  const auto lower = host_copy(bounds.upd.lb, handle.get_stream());
  const auto upper = host_copy(bounds.upd.ub, handle.get_stream());
  ASSERT_EQ(lower.size(), 2);
  EXPECT_DOUBLE_EQ(lower[0], 0.0);
  EXPECT_DOUBLE_EQ(upper[0], 0.0);
  EXPECT_DOUBLE_EQ(lower[1], 1.0);
  EXPECT_DOUBLE_EQ(upper[1], 1.0);
}

TEST(presolve, substitution_preserves_small_coefficient_with_large_domain)
{
  const raft::handle_t handle{};
  for (const std::string z_bound : {"0 <= z <= 1000000000000", "z >= 0"}) {
    SCOPED_TRACE(z_bound);
    auto model = cuopt::test::parse_inline_lp(
      "Minimize\n obj: x\nSubject To\n link: x - z = 0\nBounds\n 0 <= x <= 2000000000000\n " +
      z_bound + "\nGenerals\n z\nEnd\n");
    auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
    mip::problem_t<int, double> problem(op_problem);
    problem.preprocess_problem();
    // This small residual can have a large activity, so the row must be retained.
    problem.substitute_variables({0}, {1}, {0.0}, {1.000000000001});
    mip::trivial_presolve(problem, true);
    EXPECT_EQ(problem.n_constraints, 1);
    ASSERT_EQ(problem.coefficients.size(), 1);
    EXPECT_GT(problem.coefficients.element(0, handle.get_stream()), 0.0);
  }
}

TEST(presolve, substitution_preserves_other_columns)
{
  const raft::handle_t handle{};
  auto model      = cuopt::test::parse_inline_lp(R"LP(
Minimize
 obj: x
Subject To
 link: x - 1.4419880250174197 z + y = -6.2758387909538564
Bounds
 -7 <= x <= -4
 -1 <= y <= 1
Binaries
 z
End
)LP");
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  // The z terms nearly cancel, but the remaining y term still restricts the solution.
  problem.substitute_variables({0}, {1}, {-6.2758387909538556}, {1.4419880250174186});
  mip::trivial_presolve(problem, true);
  EXPECT_EQ(problem.n_constraints, 1);
  const auto coefficients = host_copy(problem.coefficients, handle.get_stream());
  ASSERT_EQ(coefficients.size(), 2);
  EXPECT_NE(std::find(coefficients.begin(), coefficients.end(), 1.0), coefficients.end());
}

TEST(presolve, duplicate_substitution)
{
  const raft::handle_t handle{};
  auto problem = cuopt::test::parse_inline_lp(R"LP(
Minimize
 obj: -11 x0 + x1 + 7 x2 + y
Subject To
 c1: x0 - x1 = 0
 c2: x0 - x2 = 0
 c3: y - x0 >= 0
Bounds
 0 <= y <= 1
Binaries
 x0 x1 x2
End
)LP");

  auto op_problem = mps_data_model_to_optimization_problem(&handle, problem);
  mip::problem_t<int, double> original_problem(op_problem);
  mip::problem_t<int, double> presolved_problem(original_problem);
  presolved_problem.preprocess_problem();
  mip::trivial_presolve(presolved_problem, true);

  mip_solver_settings_t<int, double> settings;
  settings.time_limit      = 10;
  settings.presolver       = presolver_t::Papilo;
  settings.num_cpu_threads = 2;

  timer_t timer(settings.time_limit);
  mip::mip_solver_t<int, double> solver(presolved_problem, settings, timer);
  mip::solution_t<int, double> solver_solution(presolved_problem);

#pragma omp parallel num_threads(2)
  {
#pragma omp masked
    {
      solver_solution = solver.run_solver();
    }
  }
  const bool feasible_on_presolved = solver_solution.get_feasible();
  solver_solution.problem_ptr      = &original_problem;
  const bool feasible_on_original  = solver_solution.compute_feasibility();
  auto solution                    = solver_solution.get_solution(
    feasible_on_presolved || feasible_on_original, solver.get_solver_stats(), false);

  ASSERT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), -2.0, 1e-6);
}

}  // namespace cuopt::mathematical_optimization::test
