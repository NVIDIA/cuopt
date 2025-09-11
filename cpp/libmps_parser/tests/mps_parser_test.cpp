/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utilities/common_utils.hpp>

#include <mps_parser.hpp>
#include <mps_parser/parser.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::mps_parser {

constexpr double tolerance = 1e-6;

mps_parser_t<int, double> read_from_mps(const std::string& file, bool fixed_format = true)
{
  std::string rel_file{};
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  // Empty problem not used in the test
  mps_data_model_t<int, double> problem;
  mps_parser_t<int, double> mps{problem, rel_file, fixed_format};
  return mps;
}

bool file_exists(const std::string& file)
{
  std::string rel_file{};
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  return std::filesystem::exists(rel_file);
}

TEST(mps_parser, bad_mps_files)
{
  std::stringstream ss;
  static constexpr int NumMpsFiles = 15;
  for (int i = 1; i <= NumMpsFiles; ++i) {
    ss << "linear_programming/bad-mps-" << i << ".mps";
    // Check if file exists
    if (file_exists(ss.str())) ASSERT_THROW(read_from_mps(ss.str()), std::logic_error);
    ss.str(std::string{});
    ss.clear();
  }
}

TEST(mps_parser, good_mps_file_1)
{
  auto mps = read_from_mps("linear_programming/good-mps-1.mps");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser, good_mps_file_clrf)
{
  auto mps = read_from_mps("linear_programming/good-mps-1-clrf.mps");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser, good_mps_free_file_clrf)
{
  auto mps = read_from_mps("linear_programming/good-mps-1-clrf.mps", false);
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser, good_mps_file_comments)
{
  auto mps = read_from_mps("linear_programming/good-mps-1-comments.mps", false);
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(1), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(1), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser, good_mps_file_no_name)
{
  // Should not throw an error
  read_from_mps("linear_programming/good-mps-fixed-no-name.mps");
}

TEST(mps_parser, good_mps_file_empty_name)
{
  // Should not throw an error
  read_from_mps("linear_programming/good-mps-fixed-empty-name.mps");
}

TEST(mps_parser, good_mps_file_2)
{
  auto mps = read_from_mps("linear_programming/good-fixed-mps-2.mps");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("RO W1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VA R1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser_free_format, free_format_mps_file_1)
{  // tests for arbitrary spacing in rows, column, rhs
  auto mps = read_from_mps("linear_programming/free-format-mps-1.mps", false);
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
  EXPECT_EQ(false, mps.maximize);
}

TEST(mps_parser_free_format, bad_free_format_mps_with_spaces_in_names)
{
  ASSERT_THROW(read_from_mps("linear_programming/good-fixed-mps-2.mps", false), std::logic_error);
}

TEST(mps_parser_free_format, bad_mps_files_free_format)
{
  std::stringstream ss;
  static constexpr int NumMpsFiles = 13;
  for (int i = 1; i <= NumMpsFiles; ++i) {
    ss << "linear_programming/bad-mps-" << i << ".mps";
    if (file_exists(ss.str())) ASSERT_THROW(read_from_mps(ss.str(), false), std::logic_error);
    ss.str(std::string{});
    ss.clear();
  }
}

TEST(mps_bounds, up_low_bounds)
{
  auto mps = read_from_mps("linear_programming/lp_model_with_var_bounds.mps", false);
  EXPECT_EQ("lp_model_with_var_bounds", mps.problem_name);

  ASSERT_EQ(int(1), mps.row_names.size());
  EXPECT_EQ("con", mps.row_names[0]);
  ASSERT_EQ(int(1), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ("OBJ", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("x", mps.var_names[0]);
  EXPECT_EQ("y", mps.var_names[1]);
  ASSERT_EQ(int(1), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(1), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(1., mps.A_values[0][0]);
  EXPECT_EQ(1., mps.A_values[0][1]);
  ASSERT_EQ(int(1), mps.b_values.size());
  EXPECT_EQ(3., mps.b_values[0]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(2., mps.c_values[0]);
  EXPECT_EQ(-1., mps.c_values[1]);
  EXPECT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(1., mps.variable_lower_bounds[1]);
  EXPECT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(1., mps.variable_upper_bounds[0]);
  EXPECT_EQ(2., mps.variable_upper_bounds[1]);
}

TEST(mps_bounds, standard_var_bounds_0_inf)
{
  auto mps = read_from_mps("linear_programming/free-format-mps-1.mps", false);

  // standard bounds are 0,inf when no var bounds are specified
  EXPECT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  EXPECT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[0]);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[1]);
}

TEST(mps_bounds, only_some_UP_LO_var_bounds)
{
  auto mps = read_from_mps("linear_programming/good-mps-some-var-bounds.mps");

  // standard bounds are 0,inf when no var bounds are specified
  EXPECT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(-1., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  EXPECT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[0]);
  EXPECT_EQ(2., mps.variable_upper_bounds[1]);
}

TEST(mps_bounds, fixed_var_bound)
{
  auto mps = read_from_mps("linear_programming/good-mps-fixed-var.mps");

  // standard bounds are 0,inf when no var bounds are specified
  EXPECT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(2., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  EXPECT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(2., mps.variable_upper_bounds[0]);
  EXPECT_EQ(std ::numeric_limits<double>::infinity(), mps.variable_upper_bounds[1]);
}

TEST(mps_bounds, free_var_bound)
{
  auto mps = read_from_mps("linear_programming/good-mps-free-var.mps");

  // standard bounds are 0,inf when no var bounds are specified
  EXPECT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  EXPECT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[0]);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[1]);
}

TEST(mps_bounds, lower_inf_var_bound)
{
  auto mps = read_from_mps("linear_programming/good-mps-lower-bound-inf-var.mps");

  // standard bounds are 0,inf when no var bounds are specified
  EXPECT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  EXPECT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[0]);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[1]);
}

TEST(mps_bounds, rhs_cost)
{
  auto mps = read_from_mps("linear_programming/good-mps-rhs-cost.mps");

  // objective value offset should be set to -5
  EXPECT_EQ(int(-5), mps.objective_offset_value);
}

TEST(mps_bounds, upper_inf_var_bound)
{
  auto mps = read_from_mps("linear_programming/good-mps-upper-bound-inf-var.mps");

  // standard bounds are 0,inf when no var bounds are specified
  EXPECT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  EXPECT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[0]);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[1]);
}

TEST(mps_ranges, fixed_ranges)
{
  std::string file = "linear_programming/good-mps-fixed-ranges.mps";
  auto mps         = read_from_mps(file);

  EXPECT_NEAR(4.2, mps.ranges_values[0], tolerance);   //  ROW1 range value
  EXPECT_NEAR(3.4, mps.ranges_values[1], tolerance);   //  ROW2 range value
  EXPECT_NEAR(-1.6, mps.ranges_values[2], tolerance);  // ROW3 range value
  EXPECT_NEAR(3.4, mps.ranges_values[3], tolerance);   //  ROW3 range value

  std::string rel_file{};
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  auto data_model                         = parse_mps<int, double>(rel_file, true);

  EXPECT_NEAR(1.2, data_model.get_constraint_lower_bounds()[0], tolerance);  // ROW1 lower bound
  EXPECT_NEAR(5.4, data_model.get_constraint_upper_bounds()[0], tolerance);  // ROW1 upper bound
  EXPECT_NEAR(1.5, data_model.get_constraint_lower_bounds()[1], tolerance);  // ROW2 lower bound
  EXPECT_NEAR(4.9, data_model.get_constraint_upper_bounds()[1], tolerance);  // ROW2 upper bound
  EXPECT_NEAR(
    7.9, data_model.get_constraint_lower_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    9.5, data_model.get_constraint_upper_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    3.5, data_model.get_constraint_lower_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(
    6.9, data_model.get_constraint_upper_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_lower_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_upper_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_lower_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_upper_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
}

TEST(mps_ranges, free_ranges)
{
  std::string file = "linear_programming/good-mps-free-ranges.mps";
  auto mps         = read_from_mps(file, false);

  EXPECT_NEAR(4.2, mps.ranges_values[0], tolerance);   //  ROW1 range value
  EXPECT_NEAR(3.4, mps.ranges_values[1], tolerance);   //  ROW2 range value
  EXPECT_NEAR(-1.6, mps.ranges_values[2], tolerance);  // ROW3 range value
  EXPECT_NEAR(3.4, mps.ranges_values[3], tolerance);   //  ROW3 range value

  std::string rel_file{};
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  auto data_model                         = parse_mps<int, double>(rel_file, false);

  EXPECT_NEAR(1.2, data_model.get_constraint_lower_bounds()[0], tolerance);  // ROW1 lower bound
  EXPECT_NEAR(5.4, data_model.get_constraint_upper_bounds()[0], tolerance);  // ROW1 upper bound
  EXPECT_NEAR(1.5, data_model.get_constraint_lower_bounds()[1], tolerance);  // ROW2 lower bound
  EXPECT_NEAR(4.9, data_model.get_constraint_upper_bounds()[1], tolerance);  // ROW2 upper bound
  EXPECT_NEAR(
    7.9, data_model.get_constraint_lower_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    9.5, data_model.get_constraint_upper_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    3.5, data_model.get_constraint_lower_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(
    6.9, data_model.get_constraint_upper_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_lower_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_upper_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_lower_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_upper_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
}

TEST(mps_name, two_objectives)
{
  std::string file = "linear_programming/good-mps-fixed-two-objectives.mps";
  auto mps         = read_from_mps(file, false);

  // Objective name should be first one found and not trigger an error
  EXPECT_EQ(mps.objective_name, "COST");
}

TEST(mps_objname, two_objectives)
{
  std::string file = "linear_programming/good-mps-fixed-two-objectives-objname.mps";
  auto mps         = read_from_mps(file, false);

  // Objective name is the second one found since it's specified as objname
  EXPECT_EQ(mps.objective_name, "COST6679327");
}

TEST(mps_objname, two_objectives_next_line)
{
  std::string file = "linear_programming/good-mps-fixed-two-objectives-objname-next-line.mps";
  auto mps         = read_from_mps(file, false);

  // Objective name is the second one found since it's specified as objname
  EXPECT_EQ(mps.objective_name, "COST6679327");
}

TEST(mps_objname, bad_after)
{
  std::string file = "linear_programming/bad-mps-fixed-objname-after-rows.mps";
  ASSERT_THROW(read_from_mps(file, false), std::logic_error);
}

TEST(mps_objname, bad_no_fixed)
{
  std::string file = "linear_programming/bad-mps-fixed-objname-after-rows.mps";
  ASSERT_THROW(read_from_mps(file, true), std::logic_error);
}

TEST(mps_ranges, bad_name)
{
  ASSERT_THROW(read_from_mps("linear_programming/bad-mps-fixed-ranges-name.mps", false),
               std::logic_error);
}

TEST(mps_ranges, bad_value)
{
  ASSERT_THROW(read_from_mps("linear_programming/bad-mps-fixed-ranges-value.mps", false),
               std::logic_error);
}

TEST(mps_bounds, unsupported_or_invalid_mps_types)
{
  std::stringstream ss;
  static constexpr int NumMpsFiles = 2;
  for (int i = 1; i <= NumMpsFiles; ++i) {
    ss << "linear_programming/bad-mps-bound-" << i << ".mps";
    ASSERT_THROW(read_from_mps(ss.str(), false), std::logic_error);
    ss.str(std::string{});
    ss.clear();
  };
}

TEST(mps_parser, good_mps_file_mip_1)
{
  auto mps = read_from_mps("mixed_integer_programming/good-mip-mps-1.mps", false);

  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(8000., mps.A_values[0][0]);
  EXPECT_EQ(4000., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(15., mps.A_values[1][0]);
  EXPECT_EQ(30., mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(40000., mps.b_values[0]);
  EXPECT_EQ(200., mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(100., mps.c_values[0]);
  EXPECT_EQ(150., mps.c_values[1]);
  ASSERT_EQ(int(2), mps.var_types.size());
  EXPECT_EQ('I', mps.var_types[0]);
  EXPECT_EQ('I', mps.var_types[1]);
  ASSERT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  ASSERT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(10., mps.variable_upper_bounds[0]);
  EXPECT_EQ(10., mps.variable_upper_bounds[1]);
}

TEST(mps_parser, good_mps_file_mip_no_marker)
{
  auto mps = read_from_mps("mixed_integer_programming/good-mip-mps-1-no-mark.mps", false);

  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(8000., mps.A_values[0][0]);
  EXPECT_EQ(4000., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(15., mps.A_values[1][0]);
  EXPECT_EQ(30., mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(40000., mps.b_values[0]);
  EXPECT_EQ(200., mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(100., mps.c_values[0]);
  EXPECT_EQ(150., mps.c_values[1]);
  ASSERT_EQ(int(2), mps.var_types.size());
  EXPECT_EQ('I', mps.var_types[0]);
  EXPECT_EQ('I', mps.var_types[1]);
  ASSERT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  ASSERT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(10., mps.variable_upper_bounds[0]);
  EXPECT_EQ(10., mps.variable_upper_bounds[1]);
}

TEST(mps_parser, good_mps_file_no_bounds)
{
  auto mps = read_from_mps("mixed_integer_programming/good-mip-mps-no-bounds.mps", false);

  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(8000., mps.A_values[0][0]);
  EXPECT_EQ(4000., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(15., mps.A_values[1][0]);
  EXPECT_EQ(30., mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(40000., mps.b_values[0]);
  EXPECT_EQ(200., mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(100., mps.c_values[0]);
  EXPECT_EQ(150., mps.c_values[1]);
  ASSERT_EQ(int(2), mps.var_types.size());
  EXPECT_EQ('I', mps.var_types[0]);
  EXPECT_EQ('C', mps.var_types[1]);

  ASSERT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  ASSERT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(1.0, mps.variable_upper_bounds[0]);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), mps.variable_upper_bounds[1]);
}

TEST(mps_parser, good_mps_file_partial_bounds)
{
  auto mps = read_from_mps("mixed_integer_programming/good-mip-mps-partial-bounds.mps", false);

  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(8000., mps.A_values[0][0]);
  EXPECT_EQ(4000., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(15., mps.A_values[1][0]);
  EXPECT_EQ(30., mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(40000., mps.b_values[0]);
  EXPECT_EQ(200., mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(100., mps.c_values[0]);
  EXPECT_EQ(150., mps.c_values[1]);
  ASSERT_EQ(int(2), mps.var_types.size());
  EXPECT_EQ('I', mps.var_types[0]);
  EXPECT_EQ('C', mps.var_types[1]);

  ASSERT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  ASSERT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(1.0, mps.variable_upper_bounds[0]);
  EXPECT_EQ(10.0, mps.variable_upper_bounds[1]);
}

#ifdef MPS_PARSER_WITH_BZIP2
TEST(mps_parser, good_mps_file_bzip2_compressed)
{
  auto mps = read_from_mps("linear_programming/good-mps-1.mps.bz2");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}
#endif  // MPS_PARSER_WITH_BZIP2

#ifdef MPS_PARSER_WITH_ZLIB
TEST(mps_parser, good_mps_file_zlib_compressed)
{
  auto mps = read_from_mps("linear_programming/good-mps-1.mps.gz");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}
#endif  // MPS_PARSER_WITH_ZLIB

// ================================================================================================
// QPS (Quadratic Programming) Support Tests
// ================================================================================================

// QPS-specific tests for quadratic programming support
TEST(qps_parser, quadratic_objective_basic)
{
  // Create a simple QPS test to verify quadratic objective parsing
  // This would require actual QPS test files - for now, test the API
  mps_data_model_t<int, double> model;

  // Test setting quadratic objective matrix
  std::vector<double> Q_values = {2.0, 1.0, 1.0, 2.0};  // 2x2 matrix
  std::vector<int> Q_indices   = {0, 1, 0, 1};
  std::vector<int> Q_offsets   = {0, 2, 4};  // CSR offsets

  model.set_quadratic_objective_matrix(Q_values.data(),
                                       Q_values.size(),
                                       Q_indices.data(),
                                       Q_indices.size(),
                                       Q_offsets.data(),
                                       Q_offsets.size());

  // Verify the data was stored correctly
  EXPECT_TRUE(model.has_quadratic_objective());
  EXPECT_EQ(4, model.get_quadratic_objective_values().size());
  EXPECT_EQ(2.0, model.get_quadratic_objective_values()[0]);
  EXPECT_EQ(1.0, model.get_quadratic_objective_values()[1]);
}

// Test actual QPS files from the dataset
TEST(qps_parser, test_qps_files)
{
  // Test QP_Test_1.qps if it exists
  if (file_exists("quadratic_programming/QP_Test_1.qps")) {
    auto parsed_data = parse_mps<int, double>(
      cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_1.qps", false);

    EXPECT_EQ("QP_Test_1", parsed_data.get_problem_name());
    EXPECT_EQ(2, parsed_data.get_n_variables());    // C------1 and C------2
    EXPECT_EQ(1, parsed_data.get_n_constraints());  // R------1
    EXPECT_TRUE(parsed_data.has_quadratic_objective());

    // Check variable bounds
    const auto& lower_bounds = parsed_data.get_variable_lower_bounds();
    const auto& upper_bounds = parsed_data.get_variable_upper_bounds();

    EXPECT_NEAR(2.0, lower_bounds[0], tolerance);    // C------1 lower bound
    EXPECT_NEAR(50.0, upper_bounds[0], tolerance);   // C------1 upper bound
    EXPECT_NEAR(-50.0, lower_bounds[1], tolerance);  // C------2 lower bound
    EXPECT_NEAR(50.0, upper_bounds[1], tolerance);   // C------2 upper bound
  }

  // Test QP_Test_2.qps if it exists
  if (file_exists("quadratic_programming/QP_Test_2.qps")) {
    auto parsed_data = parse_mps<int, double>(
      cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_2.qps", false);

    EXPECT_EQ("QP_Test_2", parsed_data.get_problem_name());
    EXPECT_EQ(3, parsed_data.get_n_variables());    // C------1, C------2, C------3
    EXPECT_EQ(1, parsed_data.get_n_constraints());  // R------1
    EXPECT_TRUE(parsed_data.has_quadratic_objective());

    // Check that quadratic objective matrix has values
    const auto& Q_values = parsed_data.get_quadratic_objective_values();
    EXPECT_GT(Q_values.size(), 0) << "Quadratic objective should have non-zero elements";
  }
}

}  // namespace cuopt::mps_parser
