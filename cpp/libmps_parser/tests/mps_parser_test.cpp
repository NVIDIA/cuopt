/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/common_utils.hpp>

#include <mps_parser.hpp>
#include <mps_parser/mps_writer.hpp>
#include <mps_parser/parser.hpp>

#include <gtest/gtest.h>

#include <cmath>
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

// ================================================================================================
// MPS Round-Trip Tests (Read -> Write -> Read -> Compare)
// ================================================================================================

// Helper function to compare two data models
template <typename i_t, typename f_t>
void compare_data_models(const mps_data_model_t<i_t, f_t>& original,
                         const mps_data_model_t<i_t, f_t>& reloaded,
                         f_t tol = 1e-9)
{
  // Compare basic dimensions
  EXPECT_EQ(original.get_n_variables(), reloaded.get_n_variables());
  EXPECT_EQ(original.get_n_constraints(), reloaded.get_n_constraints());

  // Compare objective coefficients
  auto orig_c   = original.get_objective_coefficients();
  auto reload_c = reloaded.get_objective_coefficients();
  ASSERT_EQ(orig_c.size(), reload_c.size());
  for (size_t i = 0; i < orig_c.size(); ++i) {
    EXPECT_NEAR(orig_c[i], reload_c[i], tol) << "Objective coefficient mismatch at index " << i;
  }

  // Compare constraint matrix values
  auto orig_A   = original.get_constraint_matrix_values();
  auto reload_A = reloaded.get_constraint_matrix_values();
  ASSERT_EQ(orig_A.size(), reload_A.size());
  for (size_t i = 0; i < orig_A.size(); ++i) {
    EXPECT_NEAR(orig_A[i], reload_A[i], tol) << "Constraint matrix value mismatch at index " << i;
  }

  // Compare constraint matrix indices
  auto orig_A_idx   = original.get_constraint_matrix_indices();
  auto reload_A_idx = reloaded.get_constraint_matrix_indices();
  ASSERT_EQ(orig_A_idx.size(), reload_A_idx.size());
  for (size_t i = 0; i < orig_A_idx.size(); ++i) {
    EXPECT_EQ(orig_A_idx[i], reload_A_idx[i]) << "Constraint matrix index mismatch at index " << i;
  }

  // Compare constraint matrix offsets
  auto orig_A_off   = original.get_constraint_matrix_offsets();
  auto reload_A_off = reloaded.get_constraint_matrix_offsets();
  ASSERT_EQ(orig_A_off.size(), reload_A_off.size());
  for (size_t i = 0; i < orig_A_off.size(); ++i) {
    EXPECT_EQ(orig_A_off[i], reload_A_off[i]) << "Constraint matrix offset mismatch at index " << i;
  }

  // Compare variable bounds
  auto orig_lb   = original.get_variable_lower_bounds();
  auto reload_lb = reloaded.get_variable_lower_bounds();
  ASSERT_EQ(orig_lb.size(), reload_lb.size());
  for (size_t i = 0; i < orig_lb.size(); ++i) {
    if (std::isinf(orig_lb[i]) && std::isinf(reload_lb[i])) {
      EXPECT_EQ(std::signbit(orig_lb[i]), std::signbit(reload_lb[i]))
        << "Variable lower bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_lb[i], reload_lb[i], tol) << "Variable lower bound mismatch at index " << i;
    }
  }

  auto orig_ub   = original.get_variable_upper_bounds();
  auto reload_ub = reloaded.get_variable_upper_bounds();
  ASSERT_EQ(orig_ub.size(), reload_ub.size());
  for (size_t i = 0; i < orig_ub.size(); ++i) {
    if (std::isinf(orig_ub[i]) && std::isinf(reload_ub[i])) {
      EXPECT_EQ(std::signbit(orig_ub[i]), std::signbit(reload_ub[i]))
        << "Variable upper bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_ub[i], reload_ub[i], tol) << "Variable upper bound mismatch at index " << i;
    }
  }

  // Compare constraint bounds
  auto orig_cl   = original.get_constraint_lower_bounds();
  auto reload_cl = reloaded.get_constraint_lower_bounds();
  ASSERT_EQ(orig_cl.size(), reload_cl.size());
  for (size_t i = 0; i < orig_cl.size(); ++i) {
    if (std::isinf(orig_cl[i]) && std::isinf(reload_cl[i])) {
      EXPECT_EQ(std::signbit(orig_cl[i]), std::signbit(reload_cl[i]))
        << "Constraint lower bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_cl[i], reload_cl[i], tol)
        << "Constraint lower bound mismatch at index " << i;
    }
  }

  auto orig_cu   = original.get_constraint_upper_bounds();
  auto reload_cu = reloaded.get_constraint_upper_bounds();
  ASSERT_EQ(orig_cu.size(), reload_cu.size());
  for (size_t i = 0; i < orig_cu.size(); ++i) {
    if (std::isinf(orig_cu[i]) && std::isinf(reload_cu[i])) {
      EXPECT_EQ(std::signbit(orig_cu[i]), std::signbit(reload_cu[i]))
        << "Constraint upper bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_cu[i], reload_cu[i], tol)
        << "Constraint upper bound mismatch at index " << i;
    }
  }

  // Compare quadratic objective if present
  EXPECT_EQ(original.has_quadratic_objective(), reloaded.has_quadratic_objective());
  if (original.has_quadratic_objective() && reloaded.has_quadratic_objective()) {
    auto orig_Q       = original.get_quadratic_objective_values();
    auto orig_Q_idx   = original.get_quadratic_objective_indices();
    auto orig_Q_off   = original.get_quadratic_objective_offsets();
    auto reload_Q     = reloaded.get_quadratic_objective_values();
    auto reload_Q_idx = reloaded.get_quadratic_objective_indices();
    auto reload_Q_off = reloaded.get_quadratic_objective_offsets();

    // Compare Q matrix structure and values
    ASSERT_EQ(orig_Q.size(), reload_Q.size()) << "Q values size mismatch";
    ASSERT_EQ(orig_Q_idx.size(), reload_Q_idx.size()) << "Q indices size mismatch";
    ASSERT_EQ(orig_Q_off.size(), reload_Q_off.size()) << "Q offsets size mismatch";

    for (size_t i = 0; i < orig_Q.size(); ++i) {
      EXPECT_NEAR(orig_Q[i], reload_Q[i], tol) << "Q value mismatch at index " << i;
    }
    for (size_t i = 0; i < orig_Q_idx.size(); ++i) {
      EXPECT_EQ(orig_Q_idx[i], reload_Q_idx[i]) << "Q index mismatch at index " << i;
    }
    for (size_t i = 0; i < orig_Q_off.size(); ++i) {
      EXPECT_EQ(orig_Q_off[i], reload_Q_off[i]) << "Q offset mismatch at index " << i;
    }
  }
}

TEST(mps_roundtrip, linear_programming_basic)
{
  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/linear_programming/good-mps-1.mps";
  std::string temp_file = "/tmp/mps_roundtrip_lp_test.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, true);

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

TEST(mps_roundtrip, linear_programming_with_bounds)
{
  if (!file_exists("linear_programming/lp_model_with_var_bounds.mps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/linear_programming/lp_model_with_var_bounds.mps";
  std::string temp_file = "/tmp/mps_roundtrip_lp_bounds_test.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, false);

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

TEST(mps_roundtrip, quadratic_programming_qp_test_1)
{
  if (!file_exists("quadratic_programming/QP_Test_1.qps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_1.qps";
  std::string temp_file = "/tmp/mps_roundtrip_qp_test_1.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, false);
  ASSERT_TRUE(original.has_quadratic_objective()) << "Original should have quadratic objective";

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);
  ASSERT_TRUE(reloaded.has_quadratic_objective()) << "Reloaded should have quadratic objective";

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

TEST(mps_roundtrip, quadratic_programming_qp_test_2)
{
  if (!file_exists("quadratic_programming/QP_Test_2.qps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_2.qps";
  std::string temp_file = "/tmp/mps_roundtrip_qp_test_2.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, false);
  ASSERT_TRUE(original.has_quadratic_objective()) << "Original should have quadratic objective";

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);
  ASSERT_TRUE(reloaded.has_quadratic_objective()) << "Reloaded should have quadratic objective";

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

}  // namespace cuopt::mps_parser
