/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "c_api_tests.h"

#include <cuopt/linear_programming/cuopt_c.h>

#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <gtest/gtest.h>

TEST(c_api, int_size) { EXPECT_EQ(test_int_size(), sizeof(int32_t)); }

TEST(c_api, float_size) { EXPECT_EQ(test_float_size(), sizeof(double)); }

TEST(c_api, afiro)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/" + "afiro_original.mps";
  int termination_status;
  EXPECT_EQ(solve_mps_file(filename.c_str(), 60, CUOPT_INFINITY, &termination_status),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_OPTIMAL);
}

// Test both LP and MIP codepaths
class TimeLimitTestFixture : public ::testing::TestWithParam<std::tuple<std::string, double, int>> {
};
TEST_P(TimeLimitTestFixture, time_limit)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename                    = rapidsDatasetRootDir + std::get<0>(GetParam());
  double target_solve_time                = std::get<1>(GetParam());
  int method                              = std::get<2>(GetParam());
  int termination_status;
  double solve_time = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(solve_mps_file(filename.c_str(),
                           target_solve_time,
                           CUOPT_INFINITY,
                           &termination_status,
                           &solve_time,
                           method),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_TIME_LIMIT);

  // Dual simplex is spending some time for factorizing the basis, and this computation does not
  // check for time limit
  double excess_allowed_time = 2.0;
  EXPECT_NEAR(solve_time, target_solve_time, excess_allowed_time);
}
INSTANTIATE_TEST_SUITE_P(
  c_api,
  TimeLimitTestFixture,
  ::testing::Values(
    std::make_tuple("/linear_programming/square41/square41.mps",
                    5,
                    CUOPT_METHOD_DUAL_SIMPLEX),  // LP, Dual Simplex
    std::make_tuple("/linear_programming/square41/square41.mps", 5, CUOPT_METHOD_PDLP),  // LP, PDLP
    std::make_tuple("/mip/enlight_hard.mps", 5, CUOPT_METHOD_DUAL_SIMPLEX)               // MIP
    ));

TEST(c_api, iteration_limit)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/" + "afiro_original.mps";
  int termination_status;
  EXPECT_EQ(solve_mps_file(filename.c_str(), 60, 1, &termination_status), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT);
}

TEST(c_api, solve_time_bb_preemption)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename                    = rapidsDatasetRootDir + "/mip/" + "bb_optimality.mps";
  int termination_status;
  double solve_time = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(solve_mps_file(filename.c_str(), 5, CUOPT_INFINITY, &termination_status, &solve_time),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_GT(solve_time, 0);  // solve time should not be equal to 0, even on very simple instances
                             // solved by B&B before the diversity solver has time to run
}

TEST(c_api, bad_parameter_name) { EXPECT_EQ(test_bad_parameter_name(), CUOPT_INVALID_ARGUMENT); }

TEST(c_api, burglar) { EXPECT_EQ(burglar_problem(), CUOPT_SUCCESS); }

TEST(c_api, test_missing_file) { EXPECT_EQ(test_missing_file(), CUOPT_MPS_FILE_ERROR); }

TEST(c_api, test_infeasible_problem) { EXPECT_EQ(test_infeasible_problem(), CUOPT_SUCCESS); }

TEST(c_api, test_ranged_problem)
{
  cuopt_int_t termination_status;
  cuopt_float_t objective;
  EXPECT_EQ(test_ranged_problem(&termination_status, &objective), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, 32.0, 1e-3);
}
