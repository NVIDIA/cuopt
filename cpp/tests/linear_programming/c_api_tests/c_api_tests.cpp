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
#include <cuopt/semantic_version.hpp>

#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <gtest/gtest.h>

TEST(c_api, int_size) { EXPECT_EQ(test_int_size(), sizeof(int32_t)); }

TEST(c_api, float_size) { EXPECT_EQ(test_float_size(), sizeof(double)); }

TEST(c_api, version)
{
  int major, minor, patch;
  test_version(&major, &minor, &patch);
  EXPECT_EQ(major, CUOPT_SEMANTIC_VERSION_MAJOR);
  EXPECT_EQ(minor, CUOPT_SEMANTIC_VERSION_MINOR);
  EXPECT_EQ(patch, CUOPT_SEMANTIC_VERSION_PATCH);
}

TEST(c_api, afiro)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/" + "afiro_original.mps";
  int termination_status;
  EXPECT_EQ(solve_mps_file(filename.c_str(), 60, CUOPT_INFINITY, &termination_status),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_OPTIMAL);
}

TEST(c_api, time_limit)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/" + "afiro_original.mps";
  int termination_status;
  EXPECT_EQ(solve_mps_file(filename.c_str(), 1e-6, CUOPT_INFINITY, &termination_status),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_TIME_LIMIT);
}

TEST(c_api, iteration_limit)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/" + "afiro_original.mps";
  int termination_status;
  EXPECT_EQ(solve_mps_file(filename.c_str(), 60, 1, &termination_status), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT);
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
