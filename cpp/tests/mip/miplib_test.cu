/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "branch_and_bound/branch_and_bound.hpp"
#include "cuopt/mathematical_optimization/mip/solver_settings.hpp"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::test {

struct result_map_t {
  std::string file;
  double cost;
};

void test_miplib_file(result_map_t test_instance, mip_solver_settings_t<int, double> settings)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute(test_instance.file);
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> problem =
    cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  handle_.sync_stream();
  // set the time limit depending on we are in assert mode or not
#ifdef ASSERT_MODE
  constexpr double test_time_limit = 60.;
#else
  constexpr double test_time_limit = 30.;
#endif

  settings.time_limit                  = test_time_limit;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  bool is_feasible = solution.get_termination_status() == mip_termination_status_t::FeasibleFound ||
                     solution.get_termination_status() == mip_termination_status_t::Optimal;
  EXPECT_TRUE(is_feasible);
  double obj_val = solution.get_objective_value();
  // for now keep a 100% error rate
  EXPECT_NEAR(test_instance.cost, obj_val, test_instance.cost);
  test_variable_bounds(problem, solution.get_solution(), settings);
  // TODO test integrality as well
}

TEST(mip_solve, run_small_tests)
{
  mip_solver_settings_t<int, double> settings;
  std::vector<result_map_t> test_instances = {
    {"mip/50v-10.mps", 11311031.}, {"mip/neos5.mps", 15.}, {"mip/swath1.mps", 1300.}};
  for (const auto& test_instance : test_instances) {
    test_miplib_file(test_instance, settings);
  }
}

// See https://github.com/NVIDIA/cuopt/pull/1111
TEST(mip_solve, low_thread_count_test)
{
  mip_solver_settings_t<int, double> settings;
  settings.num_cpu_threads = 2;
  settings.time_limit      = 30;

  const raft::handle_t handle_{};

  auto path = make_path_absolute("mip/dominating_set.mps");
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> problem =
    cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), 3.0, 1e-14);
  test_variable_bounds(problem, solution.get_solution(), settings);
}

// Verify --node-limit is respected: swath1 normally requires several thousand B&B
// nodes to prove optimality, so capping at 1000 forces the solver to stop early with
// a feasible (but not necessarily optimal) solution.
TEST(mip_solve, node_limit_test)
{
  mip_solver_settings_t<int, double> settings;
  settings.node_limit      = 1000;
  settings.time_limit      = 60;
  settings.num_cpu_threads = 8;
  double expect_obj        = 3.8151140644999992e+02;

  const raft::handle_t handle_{};

  auto path = make_path_absolute("mip/swath1.mps");
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> problem =
    cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  const auto status                    = solution.get_termination_status();
  EXPECT_TRUE(status == mip_termination_status_t::FeasibleFound);
  // for now keep a 100% error rate
  EXPECT_NEAR(expect_obj, solution.get_objective_value(), expect_obj);
  EXPECT_EQ(solution.get_num_nodes(), settings.node_limit);
  test_variable_bounds(problem, solution.get_solution(), settings);
}

}  // namespace cuopt::mathematical_optimization::test
