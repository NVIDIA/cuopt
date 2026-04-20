/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Intentionally flaky test to validate the CI flaky detection mechanism.
 * Fails on first run, passes on retry. Uses a temp file as a run counter.
 * Remove this test once the flaky detection pipeline is validated.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string get_marker_path(const std::string& test_name)
{
  const char* tmpdir = std::getenv("TMPDIR");
  if (!tmpdir) tmpdir = "/tmp";
  return std::string(tmpdir) + "/cuopt_flaky_validation_" + test_name;
}

}  // namespace

TEST(FlakyValidation, FailsFirstPassesOnRetry)
{
  auto marker = get_marker_path("gtest_flaky");

  if (std::filesystem::exists(marker)) {
    // Second run — pass and clean up
    std::filesystem::remove(marker);
    SUCCEED() << "Passed on retry (flaky validation working)";
  } else {
    // First run — create marker and fail
    std::ofstream(marker) << "first_attempt";
    FAIL() << "Intentional first-run failure for flaky detection validation";
  }
}
