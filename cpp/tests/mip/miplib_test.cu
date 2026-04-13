/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "branch_and_bound/branch_and_bound.hpp"
#include "cuopt/linear_programming/mip/solver_settings.hpp"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

struct result_map_t {
  std::string file;
  double cost;
};

struct sc_result_t {
  std::string file;
  double objective;
  double sc_value;
};

void test_miplib_file(result_map_t test_instance, mip_solver_settings_t<int, double> settings)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute(test_instance.file);
  cuopt::mps_parser::mps_data_model_t<int, double> problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
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

TEST(mip_solve, semi_continuous_regressions)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 10.;

  const std::vector<sc_result_t> test_instances = {{"mip/sc_standard.mps", 8., 0.},
                                                   {"mip/sc_no_ub.mps", 8., 0.},
                                                   {"mip/sc_lb_zero.mps", 8., 0.},
                                                   {"mip/sc_neg_lb_pos_ub.mps", -1., -3.},
                                                   {"mip/sc_both_neg.mps", -11., -5.},
                                                   {"mip/sc_ub_zero.mps", -10., -4.}};

  for (const auto& test_instance : test_instances) {
    auto path = make_path_absolute(test_instance.file);
    auto problem = cuopt::mps_parser::parse_mps<int, double>(path, false);
    auto solution = solve_mip(&handle_, problem, settings);

    EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal)
      << test_instance.file;
    ASSERT_EQ(solution.get_solution().size(),
              static_cast<size_t>(problem.get_n_variables())) << test_instance.file;

    auto host_solution = cuopt::host_copy(solution.get_solution(), solution.get_solution().stream());
    EXPECT_NEAR(solution.get_objective_value(), test_instance.objective, 1e-6) << test_instance.file;
    EXPECT_NEAR(host_solution[0], test_instance.sc_value, 1e-6) << test_instance.file;
  }
}

}  // namespace cuopt::linear_programming::test
