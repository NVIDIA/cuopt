/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <cuopt/error.hpp>
#include <utilities/macros.cuh>

#include <fstream>
#include <string>
#include <vector>

namespace cuopt {
namespace test {

// Define RAPIDS_DATASET_ROOT_DIR using a preprocessor variable to
// allow for a build to override the default. This is useful for
// having different builds for specific default dataset locations.
#ifndef RAPIDS_DATASET_ROOT_DIR
#define RAPIDS_DATASET_ROOT_DIR "./datasets"
#endif

inline const std::string get_rapids_dataset_root_dir()
{
  const char* envVar = std::getenv("RAPIDS_DATASET_ROOT_DIR");
  std::string rdrd   = (envVar != NULL) ? envVar : RAPIDS_DATASET_ROOT_DIR;
  return rdrd;
}

inline const std::string get_cuopt_home()
{
  std::string cuopt_home("");
  const char* env_var = std::getenv("CUOPT_HOME");
  cuopt_home          = (env_var != NULL) ? env_var : "";
  return cuopt_home;
}

/**
 * @brief Returns the lines that are in the ref file
 *
 * @param ref_file Ref file that contains file names and other test params
 * @return std::vector<std::string>
 */
inline std::vector<std::string> read_tests(const std::string& ref_file)
{
  const std::string& cuopt_home = cuopt::test::get_cuopt_home();
  std::string test_file         = cuopt_home.empty() ? ref_file : cuopt_home + "/" + ref_file;
  std::ifstream infile(test_file.c_str());
  cuopt_assert(infile.is_open(), "Ref file cannot be opened");
  std::vector<std::string> param_tests;
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  for (std::string line; getline(infile, line);) {
    std::string file{};
    if ((line != "") && (line[0] != '/')) {
      file = rapidsDatasetRootDir + "/" + line;
    } else {
      file = line;
    }
    param_tests.emplace_back(std::move(file));
  }
  return param_tests;
}

inline std::vector<std::string> split(std::string const& line, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(line);
  while (std::getline(token_stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

inline std::vector<std::string> read_target_file(const std::string& ref_file)
{
  const std::string& cuopt_home = cuopt::test::get_cuopt_home();
  std::string test_file         = cuopt_home.empty() ? ref_file : cuopt_home + "/" + ref_file;
  std::ifstream infile(test_file.c_str());
  cuopt_assert(infile.is_open(), "Ref file cannot be opened");

  std::string line;
  getline(infile, line);

  auto waypoint_matrix_info = split(line, ',');

  return waypoint_matrix_info;
}

inline std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>
read_waypoint_matrix_file(const std::string& ref_file)
{
  const std::string& cuopt_home = cuopt::test::get_cuopt_home();
  std::string test_file         = cuopt_home.empty() ? ref_file : cuopt_home + "/" + ref_file;
  std::ifstream infile(test_file.c_str());
  cuopt_assert(infile.is_open(), "Ref file cannot be opened");

  std::string line;

  getline(infile, line);
  auto offsets = split(line, ',');

  getline(infile, line);
  auto indices = split(line, ',');

  getline(infile, line);
  auto weights = split(line, ',');

  return {offsets, indices, weights};
}

inline std::vector<std::string> read_matrix_file(const std::string& ref_file)
{
  const std::string& cuopt_home = cuopt::test::get_cuopt_home();
  std::string test_file         = cuopt_home.empty() ? ref_file : cuopt_home + "/" + ref_file;
  std::ifstream infile(test_file.c_str());
  cuopt_assert(infile.is_open(), "Ref file cannot be opened");

  std::vector<std::string> matrix_info;
  std::string line;
  // Skip header line
  getline(infile, line);

  for (std::string line; getline(infile, line);) {
    auto matrix_line = split(line, ';');
    // Insert line at the end of vector
    // Skip 1st token : label
    matrix_info.insert(matrix_info.end(), matrix_line.cbegin() + 1, matrix_line.cend());
  }

  return matrix_info;
}

}  // namespace test
}  // namespace cuopt
