/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

class cli_test_t : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Create a temporary directory for test files
    test_dir = std::filesystem::temp_directory_path() / "cuopt_cli_test";
    std::filesystem::create_directories(test_dir);

    // Create a sample MPS file
    mps_file = test_dir / "test.mps";
    std::ofstream mps(mps_file);
    mps << "NAME          TEST\n"
        << "ROWS\n"
        << " N  OBJ\n"
        << " L  R1\n"
        << " L  R2\n"
        << "COLUMNS\n"
        << "    X1        OBJ        1\n"
        << "    X1        R1         1\n"
        << "    X2        OBJ        2\n"
        << "    X2        R2         1\n"
        << "    X3        OBJ        3\n"
        << "    X3        R1         1\n"
        << "    X4        OBJ        4\n"
        << "    X4        R2         1\n"
        << "    X5        OBJ        5\n"
        << "    X5        R1         1\n"
        << "RHS\n"
        << "    RHS1      R1         5\n"
        << "    RHS1      R2         3\n"
        << "ENDATA\n";
    mps.close();

    // Create a sample solution file
    sol_file = test_dir / "test.sol";
    std::ofstream sol(sol_file);
    sol << "# Status: Optimal\n"
        << "# Objective value: 1.0\n"
        << "X1 1.0\n"
        << "X2 2.0\n"
        << "X3 3.0\n"
        << "X4 4.0\n"
        << "X5 5.0\n";
    sol.close();
  }

  void TearDown() override
  {
    // Clean up temporary files
    std::filesystem::remove_all(test_dir);
  }

  std::filesystem::path test_dir;
  std::filesystem::path mps_file;
  std::filesystem::path sol_file;

  // Helper function to run the CLI and capture output
  std::string run_cli(const std::vector<std::string>& args)
  {
    std::stringstream cmd;
    cmd << "cuopt_cli ";
    for (const auto& arg : args) {
      cmd << arg << " ";
    }

    // Redirect stderr to stdout
    cmd << "2>&1";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) { throw std::runtime_error("popen() failed!"); }

    std::string result;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }

    pclose(pipe);
    return result;
  }
};

TEST_F(cli_test_t, basic_usage)
{
  auto output = run_cli({mps_file.string()});

  // Check if solution file was created
  auto expected_sol_file = test_dir / "test.sol";
  EXPECT_TRUE(std::filesystem::exists(expected_sol_file));

  // Check if solution file contains expected content
  std::ifstream sol(expected_sol_file);
  std::string content((std::istreambuf_iterator<char>(sol)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.find("Status:") != std::string::npos);
  EXPECT_TRUE(content.find("Objective value:") != std::string::npos);
}

TEST_F(cli_test_t, with_initial_solution)
{
  auto output = run_cli({mps_file.string(), "--initial-solution", sol_file.string()});

  // Check if solution file was created
  auto expected_sol_file = test_dir / "test.sol";
  EXPECT_TRUE(std::filesystem::exists(expected_sol_file));

  // Check if solution file contains expected content
  std::ifstream sol(expected_sol_file);
  std::string content((std::istreambuf_iterator<char>(sol)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.find("Status:") != std::string::npos);
  EXPECT_TRUE(content.find("Objective value:") != std::string::npos);
}

TEST_F(cli_test_t, invalid_mps_file)
{
  auto invalid_file = test_dir / "invalid.mps";
  std::ofstream invalid(invalid_file);
  invalid << "INVALID CONTENT";
  invalid.close();

  auto output = run_cli({invalid_file.string()});
  EXPECT_TRUE(output.find("error") != std::string::npos ||
              output.find("Error") != std::string::npos);
}

TEST_F(cli_test_t, missing_required_argument)
{
  auto output = run_cli({});
  EXPECT_TRUE(output.find("0 provided") != std::string::npos ||
              output.find("Usage") != std::string::npos);
}

TEST_F(cli_test_t, unrecognized_argument)
{
  auto output = run_cli({mps_file.string(), "--dummy-argument"});
  EXPECT_TRUE(output.find("Unknown argument: --dummy-argument") != std::string::npos);
}

TEST_F(cli_test_t, wrong_parameter_type)
{
  // Test with string value for numeric parameter
  auto output = run_cli({mps_file.string(), "--time-limit", "invalid"});
  EXPECT_TRUE(output.find("error") != std::string::npos ||
              output.find("Error") != std::string::npos);

  // Test with non-numeric value for iteration limit
  output = run_cli({mps_file.string(), "--iteration-limit", "abc"});
  EXPECT_TRUE(output.find("error") != std::string::npos ||
              output.find("Error") != std::string::npos);
}

TEST_F(cli_test_t, partial_solution_file)
{
  // Create a partial solution file
  auto partial_sol_file = test_dir / "partial.sol";
  std::ofstream partial(partial_sol_file);
  partial << "X1 1.0\nX3 2.0\n";
  partial.close();

  // Run CLI with partial solution file
  auto output = run_cli({mps_file.string(), "--initial-solution", partial_sol_file.string()});
  EXPECT_TRUE(output.find("Variable not found in solution:") != std::string::npos);

  // Check if solution file was created
  auto expected_sol_file = test_dir / "test.sol";
  EXPECT_TRUE(std::filesystem::exists(expected_sol_file));

  // Check if solution file contains expected content
  std::ifstream sol(expected_sol_file);
  std::string content((std::istreambuf_iterator<char>(sol)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.find("Status:") != std::string::npos);
  EXPECT_TRUE(content.find("Objective value:") != std::string::npos);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
