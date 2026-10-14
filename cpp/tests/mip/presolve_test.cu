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
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver.cuh>
#include <pdlp/utils.cuh>
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
