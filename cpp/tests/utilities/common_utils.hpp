/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <utilities/macros.cuh>

#include <fstream>
#include <string>
#include <vector>

namespace cuopt {
namespace test {

inline const std::string get_datasets_dir() { return "./datasets"; }

/**
 * @brief Returns the lines that are in the ref file
 *
 * @param ref_file Ref file that contains file names and other test params
 * @return std::vector<std::string>
 */
inline std::vector<std::string> read_tests(const std::string& ref_file)
{
  std::ifstream infile(ref_file.c_str());
  cuopt_assert(infile.is_open(), "Ref file cannot be opened");
  std::vector<std::string> param_tests;
  const std::string& datasets_dir = cuopt::test::get_datasets_dir();
  for (std::string line; getline(infile, line);) {
    std::string file{};
    if ((line != "") && (line[0] != '/')) {
      file = datasets_dir + "/" + line;
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
  std::ifstream infile(ref_file.c_str());
  cuopt_assert(infile.is_open(), "Ref file cannot be opened");

  std::string line;
  getline(infile, line);

  auto waypoint_matrix_info = split(line, ',');

  return waypoint_matrix_info;
}

inline std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>
read_waypoint_matrix_file(const std::string& ref_file)
{
  std::ifstream infile(ref_file.c_str());
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
  std::ifstream infile(ref_file.c_str());
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
