/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

TEST(SolverSettingsTest, TestSetGet)
{
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> solver_settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  const double tolerance_value = 1e-5;

  // Setting tolerances
  solver_settings.tolerances.absolute_dual_tolerance     = tolerance_value;
  solver_settings.tolerances.relative_dual_tolerance     = tolerance_value;
  solver_settings.tolerances.absolute_primal_tolerance   = tolerance_value;
  solver_settings.tolerances.relative_primal_tolerance   = tolerance_value;
  solver_settings.tolerances.absolute_gap_tolerance      = tolerance_value;
  solver_settings.tolerances.relative_gap_tolerance      = tolerance_value;
  solver_settings.tolerances.primal_infeasible_tolerance = tolerance_value;
  solver_settings.tolerances.dual_infeasible_tolerance   = tolerance_value;

  EXPECT_FALSE(solver_settings.per_constraint_residual);
  solver_settings.per_constraint_residual = true;

  EXPECT_NEAR(solver_settings.tolerances.absolute_dual_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.relative_dual_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.absolute_primal_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.relative_primal_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.absolute_gap_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.relative_gap_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.primal_infeasible_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.dual_infeasible_tolerance, 1e-5, 1e-10);

  solver_settings.detect_infeasibility = true;
  EXPECT_TRUE(solver_settings.detect_infeasibility);

  // To avoid the "," inside the macros which are interpreted as extra parameters
  auto Stable3 = cuopt::linear_programming::pdlp_solver_mode_t::Stable3;
  auto Fast1   = cuopt::linear_programming::pdlp_solver_mode_t::Fast1;
  EXPECT_EQ(solver_settings.pdlp_solver_mode, Stable3);
  solver_settings.pdlp_solver_mode = Fast1;
  EXPECT_EQ(solver_settings.pdlp_solver_mode, Fast1);

  EXPECT_TRUE(solver_settings.per_constraint_residual);

  EXPECT_FALSE(solver_settings.save_best_primal_so_far);
  solver_settings.save_best_primal_so_far = true;
  EXPECT_TRUE(solver_settings.save_best_primal_so_far);

  EXPECT_FALSE(solver_settings.first_primal_feasible);
  solver_settings.first_primal_feasible = true;
  EXPECT_TRUE(solver_settings.first_primal_feasible);
}

TEST(SolverSettingsTest, warm_start_smaller_vector)
{
  const raft::handle_t handle_{};

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> solver_settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  std::vector<double> primal      = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> dual        = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> primal_mapping = {1, 0};     // Only two variables and 0 - 1 swapped
  std::vector<int> dual_mapping   = {0, 2, 1};  // Only three constraints and  1 - 2 swapped

  std::vector<double> primal_expected = {1.0, 0.0};
  std::vector<double> dual_expected   = {0.0, 2.0, 1.0};

  rmm::device_uvector<double> current_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> initial_primal_average =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> current_ATY = cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> sum_primal_solutions =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());

  rmm::device_uvector<double> current_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> initial_dual_average = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> sum_dual_solutions   = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());

  rmm::device_uvector<int> d_primal_mapping =
    cuopt::device_copy(primal_mapping, handle_.get_stream());
  rmm::device_uvector<int> d_dual_mapping = cuopt::device_copy(dual_mapping, handle_.get_stream());

  pdlp_warm_start_data_t<int, double> warm_start_data =
    pdlp_warm_start_data_t<int, double>(current_primal_solution,
                                        current_dual_solution,
                                        initial_primal_average,
                                        initial_dual_average,
                                        current_ATY,
                                        sum_primal_solutions,
                                        sum_dual_solutions,
                                        last_restart_duality_gap_primal_solution,
                                        last_restart_duality_gap_dual_solution,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1);
  solver_settings.set_pdlp_warm_start_data(warm_start_data, d_primal_mapping, d_dual_mapping);

  std::vector<double> h_current_primal_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_primal_solution_);
  std::vector<double> h_initial_primal_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_primal_average_);
  std::vector<double> h_current_ATY =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_ATY_);
  std::vector<double> h_sum_primal_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_primal_solutions_);
  std::vector<double> h_last_restart_duality_gap_primal_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_);

  EXPECT_EQ(h_current_primal_solution.size(), primal_expected.size());
  EXPECT_EQ(h_initial_primal_average.size(), primal_expected.size());
  EXPECT_EQ(h_current_ATY.size(), primal_expected.size());
  EXPECT_EQ(h_sum_primal_solutions.size(), primal_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution.size(), primal_expected.size());

  EXPECT_EQ(h_current_primal_solution, primal_expected);
  EXPECT_EQ(h_initial_primal_average, primal_expected);
  EXPECT_EQ(h_current_ATY, primal_expected);
  EXPECT_EQ(h_sum_primal_solutions, primal_expected);
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution, primal_expected);

  std::vector<double> h_current_dual_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_dual_solution_);
  std::vector<double> h_initial_dual_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_dual_average_);
  std::vector<double> h_sum_dual_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_dual_solutions_);
  std::vector<double> h_last_restart_duality_gap_dual_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_);

  EXPECT_EQ(h_current_dual_solution.size(), dual_expected.size());
  EXPECT_EQ(h_initial_dual_average.size(), dual_expected.size());
  EXPECT_EQ(h_sum_dual_solutions.size(), dual_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution.size(), dual_expected.size());

  EXPECT_EQ(h_current_dual_solution, dual_expected);
  EXPECT_EQ(h_initial_dual_average, dual_expected);
  EXPECT_EQ(h_sum_dual_solutions, dual_expected);
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution, dual_expected);
}

TEST(SolverSettingsTest, warm_start_bigger_vector)
{
  const raft::handle_t handle_{};

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> solver_settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  std::vector<double> primal      = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> dual        = {0.0, 1.0, 2.0};
  std::vector<int> primal_mapping = {0, 1, 2, 3, 4, 5};  // Only two variables and 0 - 1 swapped
  std::vector<int> dual_mapping   = {
    0, 1, 2, 3, 4, 5, 6};  // Only three constraints and  1 - 2 swapped

  std::vector<double> primal_expected = {0.0, 1.0, 2.0, 3.0, 0.0, 0.0};
  std::vector<double> dual_expected   = {0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0};

  rmm::device_uvector<double> current_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> initial_primal_average =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> current_ATY = cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> sum_primal_solutions =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());

  rmm::device_uvector<double> current_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> initial_dual_average = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> sum_dual_solutions   = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());

  rmm::device_uvector<int> d_primal_mapping =
    cuopt::device_copy(primal_mapping, handle_.get_stream());
  rmm::device_uvector<int> d_dual_mapping = cuopt::device_copy(dual_mapping, handle_.get_stream());

  pdlp_warm_start_data_t<int, double> warm_start_data =
    pdlp_warm_start_data_t<int, double>(current_primal_solution,
                                        current_dual_solution,
                                        initial_primal_average,
                                        initial_dual_average,
                                        current_ATY,
                                        sum_primal_solutions,
                                        sum_dual_solutions,
                                        last_restart_duality_gap_primal_solution,
                                        last_restart_duality_gap_dual_solution,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1);
  solver_settings.set_pdlp_warm_start_data(warm_start_data, d_primal_mapping, d_dual_mapping);

  std::vector<double> h_current_primal_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_primal_solution_);
  std::vector<double> h_initial_primal_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_primal_average_);
  std::vector<double> h_current_ATY =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_ATY_);
  std::vector<double> h_sum_primal_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_primal_solutions_);
  std::vector<double> h_last_restart_duality_gap_primal_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_);

  EXPECT_EQ(h_current_primal_solution.size(), primal_expected.size());
  EXPECT_EQ(h_initial_primal_average.size(), primal_expected.size());
  EXPECT_EQ(h_current_ATY.size(), primal_expected.size());
  EXPECT_EQ(h_sum_primal_solutions.size(), primal_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution.size(), primal_expected.size());

  EXPECT_EQ(h_current_primal_solution, primal_expected);
  EXPECT_EQ(h_initial_primal_average, primal_expected);
  EXPECT_EQ(h_current_ATY, primal_expected);
  EXPECT_EQ(h_sum_primal_solutions, primal_expected);
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution, primal_expected);

  std::vector<double> h_current_dual_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_dual_solution_);
  std::vector<double> h_initial_dual_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_dual_average_);
  std::vector<double> h_sum_dual_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_dual_solutions_);
  std::vector<double> h_last_restart_duality_gap_dual_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_);

  EXPECT_EQ(h_current_dual_solution.size(), dual_expected.size());
  EXPECT_EQ(h_initial_dual_average.size(), dual_expected.size());
  EXPECT_EQ(h_sum_dual_solutions.size(), dual_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution.size(), dual_expected.size());

  EXPECT_EQ(h_current_dual_solution, dual_expected);
  EXPECT_EQ(h_initial_dual_average, dual_expected);
  EXPECT_EQ(h_sum_dual_solutions, dual_expected);
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution, dual_expected);
}

}  // namespace cuopt::linear_programming
