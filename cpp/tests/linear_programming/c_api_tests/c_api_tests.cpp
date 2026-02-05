/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "c_api_tests.h"

#include <filesystem>
#include <iostream>

#include <cuopt/linear_programming/cuopt_c.h>
#include <linear_programming/cuopt_c_internal.hpp>

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
  double excess_allowed_time = 3.0;
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
    std::make_tuple("/mip/supportcase22.mps", 15, CUOPT_METHOD_DUAL_SIMPLEX)             // MIP
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

TEST(c_api, mip_get_callbacks_only) { EXPECT_EQ(test_mip_get_callbacks_only(), CUOPT_SUCCESS); }

TEST(c_api, mip_get_set_callbacks) { EXPECT_EQ(test_mip_get_set_callbacks(), CUOPT_SUCCESS); }

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

TEST(c_api, test_invalid_bounds)
{
  // Test LP codepath
  EXPECT_EQ(test_invalid_bounds(false), CUOPT_SUCCESS);
  // Test MIP codepath
  EXPECT_EQ(test_invalid_bounds(true), CUOPT_SUCCESS);
}

TEST(c_api, test_quadratic_problem)
{
  cuopt_int_t termination_status;
  cuopt_float_t objective;
  EXPECT_EQ(test_quadratic_problem(&termination_status, &objective), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, -32.0, 1e-3);
}

TEST(c_api, test_quadratic_ranged_problem)
{
  cuopt_int_t termination_status;
  cuopt_float_t objective;
  EXPECT_EQ(test_quadratic_ranged_problem(&termination_status, &objective), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, (int)CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, -32.0, 1e-3);
}

TEST(c_api, test_write_problem)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string input_file = rapidsDatasetRootDir + "/linear_programming/afiro_original.mps";
  std::string temp_file = std::filesystem::temp_directory_path().string() + "/c_api_test_write.mps";
  EXPECT_EQ(test_write_problem(input_file.c_str(), temp_file.c_str()), CUOPT_SUCCESS);
  std::filesystem::remove(temp_file);
}

static bool test_mps_roundtrip(const std::string& mps_file_path)
{
  using cuopt::linear_programming::problem_and_stream_view_t;

  cuOptOptimizationProblem original_handle = nullptr;
  cuOptOptimizationProblem reread_handle   = nullptr;
  bool result                              = false;

  std::string model_basename = std::filesystem::path(mps_file_path).filename().string();
  std::string temp_file =
    std::filesystem::temp_directory_path().string() + "/roundtrip_temp_" + model_basename;

  if (cuOptReadProblem(mps_file_path.c_str(), &original_handle) != CUOPT_SUCCESS) {
    std::cerr << "Failed to read original MPS file: " << mps_file_path << std::endl;
    goto cleanup;
  }

  if (cuOptWriteProblem(original_handle, temp_file.c_str(), CUOPT_FILE_FORMAT_MPS) !=
      CUOPT_SUCCESS) {
    std::cerr << "Failed to write MPS file: " << temp_file << std::endl;
    goto cleanup;
  }

  if (cuOptReadProblem(temp_file.c_str(), &reread_handle) != CUOPT_SUCCESS) {
    std::cerr << "Failed to re-read MPS file: " << temp_file << std::endl;
    goto cleanup;
  }

  {
    auto* original_problem_wrapper = static_cast<problem_and_stream_view_t*>(original_handle);
    auto* reread_problem_wrapper   = static_cast<problem_and_stream_view_t*>(reread_handle);

    result =
      original_problem_wrapper->op_problem->is_equivalent(*reread_problem_wrapper->op_problem);
  }

cleanup:
  std::filesystem::remove(temp_file);
  cuOptDestroyProblem(&original_handle);
  cuOptDestroyProblem(&reread_handle);

  return result;
}

class WriteRoundtripTestFixture : public ::testing::TestWithParam<std::string> {};
TEST_P(WriteRoundtripTestFixture, roundtrip)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  EXPECT_TRUE(test_mps_roundtrip(rapidsDatasetRootDir + GetParam()));
}
INSTANTIATE_TEST_SUITE_P(c_api,
                         WriteRoundtripTestFixture,
                         ::testing::Values("/linear_programming/afiro_original.mps",
                                           "/mip/50v-10.mps",
                                           "/mip/fiball.mps",
                                           "/mip/gen-ip054.mps",
                                           "/mip/sct2.mps",
                                           "/mip/uccase9.mps",
                                           "/mip/drayage-25-23.mps",
                                           "/mip/tr12-30.mps",
                                           "/mip/neos-3004026-krka.mps",
                                           "/mip/ns1208400.mps",
                                           "/mip/gmu-35-50.mps",
                                           "/mip/n2seq36q.mps",
                                           "/mip/seymour1.mps",
                                           "/mip/rmatr200-p5.mps",
                                           "/mip/cvs16r128-89.mps",
                                           "/mip/thor50dday.mps",
                                           "/mip/stein9inf.mps",
                                           "/mip/neos5.mps",
                                           "/mip/neos5-free-bound.mps",
                                           "/mip/crossing_var_bounds.mps",
                                           "/mip/cod105_max.mps",
                                           "/mip/sudoku.mps",
                                           "/mip/presolve-infeasible.mps",
                                           "/mip/swath1.mps",
                                           "/mip/enlight_hard.mps",
                                           "/mip/enlight11.mps",
                                           "/mip/supportcase22.mps"));
