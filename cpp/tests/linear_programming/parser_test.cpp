/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/common_utils.hpp>
#include <utilities/inline_mps_test_utils.hpp>

#include <cuopt/mathematical_optimization/io/mps_writer.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <mps_parser_internal.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace cuopt::mathematical_optimization::io {

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

namespace {

// Non-template forwarding wrapper around read_lp_from_string<int, double>.
// Exists only so EXPECT_THROW(read_lp_string(R"LP(...)LP"), exc) is parsed
// correctly — gtest's macro splits its args on top-level commas, and the
// comma inside <int, double> would otherwise be treated as a macro-arg
// separator.
mps_data_model_t<int, double> read_lp_string(std::string_view content)
{
  return read_lp_from_string<int, double>(content);
}

// Returns the index of `name` in the variable list, or -1 if absent.
int find_var(const mps_data_model_t<int, double>& m, const std::string& name)
{
  const auto& names = m.get_variable_names();
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return static_cast<int>(i);
  }
  return -1;
}

int find_row(const mps_data_model_t<int, double>& m, const std::string& name)
{
  const auto& names = m.get_row_names();
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return static_cast<int>(i);
  }
  return -1;
}

// Returns A[row, col] by scanning the CSR row. Zero if the entry is missing.
double a_entry(const mps_data_model_t<int, double>& m, int row, int col)
{
  const auto& offsets = m.get_constraint_matrix_offsets();
  const auto& indices = m.get_constraint_matrix_indices();
  const auto& values  = m.get_constraint_matrix_values();
  for (int k = offsets[row]; k < offsets[row + 1]; ++k) {
    if (indices[k] == col) return values[k];
  }
  return 0.0;
}

// Returns Q[row, col] by scanning the CSR row of the quadratic matrix.
double q_entry(const mps_data_model_t<int, double>& m, int row, int col)
{
  const auto& offsets = m.get_quadratic_objective_offsets();
  const auto& indices = m.get_quadratic_objective_indices();
  const auto& values  = m.get_quadratic_objective_values();
  if (offsets.empty()) return 0.0;
  for (int k = offsets[row]; k < offsets[row + 1]; ++k) {
    if (indices[k] == col) return values[k];
  }
  return 0.0;
}

}  // namespace

// ===========================================================================
// Per-fixture test classes. Each class describes one named problem fixture
// and owns the checker for that problem's expected parsed data model. The
// MPS and LP TEST_F cases within a fixture share the same `check_model`
// method, so the expected values live in exactly one place per fixture.
//
// All fixtures inherit a common base that supplies read_mps_file and
// read_lp_file helpers.
// ===========================================================================

class parser_fixture_base : public ::testing::Test {
 protected:
  static mps_data_model_t<int, double> read_mps_file(const std::string& file,
                                                     bool fixed_format = true)
  {
    const std::string& root = cuopt::test::get_rapids_dataset_root_dir();
    return read_mps<int, double>(root + "/" + file, fixed_format);
  }

  static mps_data_model_t<int, double> read_lp_file(const std::string& file)
  {
    const std::string& root = cuopt::test::get_rapids_dataset_root_dir();
    return read_lp<int, double>(root + "/" + file);
  }
};

// 2 vars (continuous, default [0,inf) bounds), 2 <= constraints.
//   min 0.2*VAR1 + 0.1*VAR2
//   ROW1: 3*VAR1 + 4*VAR2 <= 5.4
//   ROW2: 2.7*VAR1 + 10.1*VAR2 <= 4.9
class good_mps_1_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_FALSE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    ASSERT_EQ(2, m.get_n_constraints());
    EXPECT_EQ("VAR1", m.get_variable_names()[0]);
    EXPECT_EQ("VAR2", m.get_variable_names()[1]);
    EXPECT_EQ("ROW1", m.get_row_names()[0]);
    EXPECT_EQ("ROW2", m.get_row_names()[1]);
    EXPECT_EQ('C', m.get_variable_types()[0]);
    EXPECT_EQ('C', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
    EXPECT_NEAR(0.2, m.get_objective_coefficients()[0], tolerance);
    EXPECT_NEAR(0.1, m.get_objective_coefficients()[1], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[0]);
    EXPECT_NEAR(5.4, m.get_constraint_upper_bounds()[0], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[1]);
    EXPECT_NEAR(4.9, m.get_constraint_upper_bounds()[1], tolerance);
    const auto& off = m.get_constraint_matrix_offsets();
    const auto& idx = m.get_constraint_matrix_indices();
    const auto& val = m.get_constraint_matrix_values();
    ASSERT_EQ(3u, off.size());
    EXPECT_EQ(0, off[0]);
    EXPECT_EQ(2, off[1]);
    EXPECT_EQ(4, off[2]);
    EXPECT_EQ(0, idx[0]);
    EXPECT_NEAR(3.0, val[0], tolerance);
    EXPECT_EQ(1, idx[1]);
    EXPECT_NEAR(4.0, val[1], tolerance);
    EXPECT_EQ(0, idx[2]);
    EXPECT_NEAR(2.7, val[2], tolerance);
    EXPECT_EQ(1, idx[3]);
    EXPECT_NEAR(10.1, val[3], tolerance);
    EXPECT_FALSE(m.has_quadratic_objective());
  }
};

// min 2x - y; x+y <= 3; 0<=x<=1, 1<=y<=2.
class up_low_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_FALSE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    ASSERT_EQ(1, m.get_n_constraints());
    EXPECT_EQ("x", m.get_variable_names()[0]);
    EXPECT_EQ("y", m.get_variable_names()[1]);
    EXPECT_EQ("con", m.get_row_names()[0]);
    EXPECT_NEAR(2.0, m.get_objective_coefficients()[0], tolerance);
    EXPECT_NEAR(-1.0, m.get_objective_coefficients()[1], tolerance);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(2.0, m.get_variable_upper_bounds()[1]);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[0]);
    EXPECT_NEAR(3.0, m.get_constraint_upper_bounds()[0], tolerance);
    const auto& val = m.get_constraint_matrix_values();
    ASSERT_EQ(2u, val.size());
    EXPECT_NEAR(1.0, val[0], tolerance);
    EXPECT_NEAR(1.0, val[1], tolerance);
  }
};

// good-mps-1 objective/matrix/rows; -1 <= VAR1 <= inf, 0 <= VAR2 <= 2.
class some_var_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(-1.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(2.0, m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 fixed at 2; VAR2 default [0, inf).
class fixed_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(2.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(2.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 free (-inf, +inf); VAR2 default [0, +inf).
class free_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 lower=-inf (MI in MPS / -inf in LP), upper default +inf; VAR2 default.
// Effective bounds match free_var_bound_test — the two fixtures differ only in
// how the lower -inf is spelled (free vs explicit -inf bound).
class lower_inf_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 upper=+inf (PL in MPS / inf in LP); both default lower 0. Effective
// bounds match two default [0, +inf) variables.
class upper_inf_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// 2 integer vars bounded [0, 10]; max 100 VAR1 + 150 VAR2;
// 8000 VAR1 + 4000 VAR2 <= 40000 ; 15 VAR1 + 30 VAR2 <= 200.
class mip_with_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_TRUE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    ASSERT_EQ(2, m.get_n_constraints());
    EXPECT_EQ("VAR1", m.get_variable_names()[0]);
    EXPECT_EQ("VAR2", m.get_variable_names()[1]);
    EXPECT_EQ('I', m.get_variable_types()[0]);
    EXPECT_EQ('I', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(10.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(10.0, m.get_variable_upper_bounds()[1]);
    EXPECT_NEAR(100.0, m.get_objective_coefficients()[0], tolerance);
    EXPECT_NEAR(150.0, m.get_objective_coefficients()[1], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[0]);
    EXPECT_NEAR(40000.0, m.get_constraint_upper_bounds()[0], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[1]);
    EXPECT_NEAR(200.0, m.get_constraint_upper_bounds()[1], tolerance);
    const auto& val = m.get_constraint_matrix_values();
    ASSERT_EQ(4u, val.size());
    EXPECT_NEAR(8000.0, val[0], tolerance);
    EXPECT_NEAR(4000.0, val[1], tolerance);
    EXPECT_NEAR(15.0, val[2], tolerance);
    EXPECT_NEAR(30.0, val[3], tolerance);
  }
};

// Like mip_with_bounds but VAR1 is binary ([0,1]) and VAR2 is continuous,
// default upper +inf. (MPS: no explicit bounds on integer => [0,1]. LP: VAR1
// listed under Binaries.)
class mip_no_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_TRUE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ('I', m.get_variable_types()[0]);
    EXPECT_EQ('C', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 binary ([0,1]); VAR2 continuous with explicit upper 10.
class mip_partial_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_TRUE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ('I', m.get_variable_types()[0]);
    EXPECT_EQ('C', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(10.0, m.get_variable_upper_bounds()[1]);
  }
};

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

TEST_F(good_mps_1_test, mps)
{
  check_model(read_mps_file("linear_programming/good-mps-1.mps"));
  // Parser-struct fields that are MPS-only (not exposed via the data model).
  auto mps = read_from_mps("linear_programming/good-mps-1.mps");
  EXPECT_EQ("good-1", mps.problem_name);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
}

TEST_F(good_mps_1_test, lp) { check_model(read_lp_file("linear_programming/good-mps-1.lp")); }

// Compressed-LP coverage: read_lp() shares file_to_string() with read_mps(),
// so the same dlopen-based decompression path that handles .mps.gz / .mps.bz2
// must also work for .lp.gz / .lp.bz2.
TEST_F(good_mps_1_test, lp_zlib_compressed)
{
  check_model(read_lp_file("linear_programming/good-mps-1.lp.gz"));
}

TEST_F(good_mps_1_test, lp_bzip2_compressed)
{
  check_model(read_lp_file("linear_programming/good-mps-1.lp.bz2"));
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

TEST_F(up_low_bounds_test, mps)
{
  check_model(read_mps_file("linear_programming/lp_model_with_var_bounds.mps", false));
  auto mps = read_from_mps("linear_programming/lp_model_with_var_bounds.mps", false);
  EXPECT_EQ("lp_model_with_var_bounds", mps.problem_name);
  EXPECT_EQ("OBJ", mps.objective_name);
  ASSERT_EQ(int(1), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
}

TEST_F(up_low_bounds_test, lp)
{
  check_model(read_lp_file("linear_programming/lp_model_with_var_bounds.lp"));
}

TEST_F(good_mps_1_test, mps_free_format)
{
  // free-format-mps-1.mps encodes the same problem as good-mps-1 with default
  // [0, +inf) bounds (no BOUNDS section), so it satisfies the same checker.
  check_model(read_mps_file("linear_programming/free-format-mps-1.mps", false));
}

TEST_F(some_var_bounds_test, mps)
{
  check_model(read_mps_file("linear_programming/good-mps-some-var-bounds.mps"));
}

TEST_F(some_var_bounds_test, lp)
{
  check_model(read_lp_file("linear_programming/good-mps-some-var-bounds.lp"));
}

TEST_F(fixed_var_bound_test, mps)
{
  check_model(read_mps_file("linear_programming/good-mps-fixed-var.mps"));
}

TEST_F(fixed_var_bound_test, lp)
{
  check_model(read_lp_file("linear_programming/good-mps-fixed-var.lp"));
}

TEST_F(free_var_bound_test, mps)
{
  check_model(read_mps_file("linear_programming/good-mps-free-var.mps"));
}

TEST_F(free_var_bound_test, lp)
{
  check_model(read_lp_file("linear_programming/good-mps-free-var.lp"));
}

TEST_F(lower_inf_var_bound_test, mps)
{
  check_model(read_mps_file("linear_programming/good-mps-lower-bound-inf-var.mps"));
}

TEST_F(lower_inf_var_bound_test, lp)
{
  check_model(read_lp_file("linear_programming/good-mps-lower-bound-inf-var.lp"));
}

TEST(mps_bounds, rhs_cost)
{
  auto mps = read_from_mps("linear_programming/good-mps-rhs-cost.mps");

  // objective value offset should be set to -5
  EXPECT_EQ(int(-5), mps.objective_offset_value);
}

TEST_F(upper_inf_var_bound_test, mps)
{
  check_model(read_mps_file("linear_programming/good-mps-upper-bound-inf-var.mps"));
}

TEST_F(upper_inf_var_bound_test, lp)
{
  check_model(read_lp_file("linear_programming/good-mps-upper-bound-inf-var.lp"));
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
  auto data_model                         = read_mps<int, double>(rel_file, true);

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
  auto data_model                         = read_mps<int, double>(rel_file, false);

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

TEST_F(mip_with_bounds_test, mps)
{
  check_model(read_mps_file("mixed_integer_programming/good-mip-mps-1.mps", false));
  auto mps = read_from_mps("mixed_integer_programming/good-mip-mps-1.mps", false);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
}

TEST_F(mip_with_bounds_test, lp)
{
  check_model(read_lp_file("mixed_integer_programming/good-mip-mps-1.lp"));
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

TEST_F(mip_no_bounds_test, mps)
{
  check_model(read_mps_file("mixed_integer_programming/good-mip-mps-no-bounds.mps", false));
}

TEST_F(mip_no_bounds_test, lp)
{
  check_model(read_lp_file("mixed_integer_programming/good-mip-mps-no-bounds.lp"));
}

TEST_F(mip_partial_bounds_test, mps)
{
  check_model(read_mps_file("mixed_integer_programming/good-mip-mps-partial-bounds.mps", false));
}

TEST_F(mip_partial_bounds_test, lp)
{
  check_model(read_lp_file("mixed_integer_programming/good-mip-mps-partial-bounds.lp"));
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

  model.set_quadratic_objective_matrix(Q_values, Q_indices, Q_offsets);

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
    auto parsed_data = read_mps<int, double>(
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
    auto parsed_data = read_mps<int, double>(
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

// RAII temp file path: builds a unique path under temp_directory_path() and
// removes it on scope exit, so write/parse/compare can throw without leaking
// the file and parallel test runs don't collide on a shared name. The file
// is not created at construction; it appears when the writer writes to
// `path()`.
struct temp_file_t {
  std::filesystem::path p;
  explicit temp_file_t(const std::string& suffix)
  {
    static std::atomic<unsigned long> counter{0};
    const auto pid = static_cast<unsigned long>(::getpid());
    const auto n   = counter.fetch_add(1, std::memory_order_relaxed);
    p              = std::filesystem::temp_directory_path() /
        ("cuopt_test_" + std::to_string(pid) + "_" + std::to_string(n) + suffix);
  }
  ~temp_file_t()
  {
    std::error_code ec;
    std::filesystem::remove(p, ec);
  }
  temp_file_t(const temp_file_t&)            = delete;
  temp_file_t& operator=(const temp_file_t&) = delete;
  std::string string() const { return p.string(); }
};

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
  temp_file_t temp_file(".mps");

  // Read original
  auto original = read_mps<int, double>(input_file, true);

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  // Read back
  auto reloaded = read_mps<int, double>(temp_file.string(), false);

  // Compare
  compare_data_models(original, reloaded);
}

TEST(mps_roundtrip, linear_programming_with_bounds)
{
  if (!file_exists("linear_programming/lp_model_with_var_bounds.mps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/linear_programming/lp_model_with_var_bounds.mps";
  temp_file_t temp_file(".mps");

  // Read original
  auto original = read_mps<int, double>(input_file, false);

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  // Read back
  auto reloaded = read_mps<int, double>(temp_file.string(), false);

  // Compare
  compare_data_models(original, reloaded);
}

TEST(mps_roundtrip, quadratic_programming_qp_test_1)
{
  if (!file_exists("quadratic_programming/QP_Test_1.qps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_1.qps";
  temp_file_t temp_file(".mps");

  // Read original
  auto original = read_mps<int, double>(input_file, false);
  ASSERT_TRUE(original.has_quadratic_objective()) << "Original should have quadratic objective";

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  // Read back
  auto reloaded = read_mps<int, double>(temp_file.string(), false);
  ASSERT_TRUE(reloaded.has_quadratic_objective()) << "Reloaded should have quadratic objective";

  // Compare
  compare_data_models(original, reloaded);
}

TEST(mps_roundtrip, quadratic_programming_qp_test_2)
{
  if (!file_exists("quadratic_programming/QP_Test_2.qps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_2.qps";
  temp_file_t temp_file(".mps");

  // Read original
  auto original = read_mps<int, double>(input_file, false);
  ASSERT_TRUE(original.has_quadratic_objective()) << "Original should have quadratic objective";

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  // Read back
  auto reloaded = read_mps<int, double>(temp_file.string(), false);
  ASSERT_TRUE(reloaded.has_quadratic_objective()) << "Reloaded should have quadratic objective";

  // Compare
  compare_data_models(original, reloaded);
}

// ================================================================================================
// LP -> MPS Round-Trip Tests (Read LP -> Write MPS -> Read MPS -> Compare)
// ================================================================================================
// Parses an LP file, writes the resulting data model out as MPS, reads it
// back, and checks that the reloaded data model matches the one produced by
// the LP parser. Exercises the LP reader + the writer + the MPS reader end
// to end, without trusting any direct LP<->MPS comparison.

TEST_F(good_mps_1_test, lp_roundtrip)
{
  temp_file_t temp_file(".mps");

  auto original = read_lp_file("linear_programming/good-mps-1.lp");

  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  auto reloaded = read_mps<int, double>(temp_file.string(), false);

  compare_data_models(original, reloaded);
}

TEST_F(up_low_bounds_test, lp_roundtrip)
{
  temp_file_t temp_file(".mps");

  auto original = read_lp_file("linear_programming/lp_model_with_var_bounds.lp");

  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  auto reloaded = read_mps<int, double>(temp_file.string(), false);

  compare_data_models(original, reloaded);
}

TEST_F(mip_with_bounds_test, lp_roundtrip)
{
  temp_file_t temp_file(".mps");

  auto original = read_lp_file("mixed_integer_programming/good-mip-mps-1.lp");

  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  auto reloaded = read_mps<int, double>(temp_file.string(), false);

  compare_data_models(original, reloaded);
}

// ================================================================================================
// LP syntax / feature / error-path tests (read_lp on inline LP content)
// ================================================================================================

TEST(lp_parser, trivial)
{
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 lb_constr: x >= 2.5
Bounds
 x <= 10
End
)LP");

  EXPECT_FALSE(m.get_sense());  // minimize
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  int x = find_var(m, "x");
  ASSERT_GE(x, 0);
  EXPECT_EQ(m.get_variable_types()[x], 'C');
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], 0.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[x], 10.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[x], 1.0, tolerance);

  ASSERT_EQ(m.get_row_names().size(), 1u);
  int r = find_row(m, "lb_constr");
  ASSERT_GE(r, 0);
  // 'G' relation ⇒ finite lower bound, +inf upper bound.
  EXPECT_NEAR(m.get_constraint_lower_bounds()[r], 2.5, tolerance);
  EXPECT_TRUE(std::isinf(m.get_constraint_upper_bounds()[r]));
  EXPECT_NEAR(a_entry(m, r, x), 1.0, tolerance);
}

TEST(lp_parser, basic_lp_with_float_coefficients)
{
  auto m = read_lp_string(R"LP(
Minimize
  x1 + x2
Subject To
 c1: 2.5 x1 + x2 <= 10
 c2: x1 + 1.5 x2 <= 8
 c3: x1 + x2 <= 6
End
)LP");

  EXPECT_EQ(m.get_variable_names().size(), 2u);
  int x1 = find_var(m, "x1");
  int x2 = find_var(m, "x2");
  ASSERT_GE(x1, 0);
  ASSERT_GE(x2, 0);
  // Default bounds for continuous variables.
  EXPECT_NEAR(m.get_variable_lower_bounds()[x1], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x1]));

  ASSERT_EQ(m.get_row_names().size(), 3u);
  int c1 = find_row(m, "c1");
  int c2 = find_row(m, "c2");
  ASSERT_GE(c1, 0);
  ASSERT_GE(c2, 0);
  EXPECT_NEAR(a_entry(m, c1, x1), 2.5, tolerance);
  EXPECT_NEAR(a_entry(m, c1, x2), 1.0, tolerance);
  EXPECT_NEAR(a_entry(m, c2, x2), 1.5, tolerance);
  EXPECT_NEAR(m.get_constraint_upper_bounds()[c1], 10.0, tolerance);
}

TEST(lp_parser, maximize_flips_sense)
{
  auto m = read_lp_string(R"LP(
Maximize
  3 x + 2 y
Subject To
 c1: x + y <= 6
 c2: 2 x + y <= 8
End
)LP");

  EXPECT_TRUE(m.get_sense());
  int x = find_var(m, "x");
  int y = find_var(m, "y");
  EXPECT_NEAR(m.get_objective_coefficients()[x], 3.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 2.0, tolerance);
}

TEST(lp_parser, equality_constraints)
{
  auto m = read_lp_string(R"LP(
Minimize
  c1 + 2 c2 + 3 c3 + 4 c4
Subject To
 s1: c1 + c2 = 10
 s2: c3 + c4 = 12
 d1: c1 + c3 = 9
 d2: c2 + c4 = 13
End
)LP");

  ASSERT_EQ(m.get_row_names().size(), 4u);
  // All four are equality constraints ⇒ lb == ub for every row.
  const auto& clb = m.get_constraint_lower_bounds();
  const auto& cub = m.get_constraint_upper_bounds();
  for (size_t i = 0; i < clb.size(); ++i) {
    EXPECT_NEAR(clb[i], cub[i], tolerance);
  }
  int s1 = find_row(m, "s1");
  EXPECT_NEAR(m.get_constraint_lower_bounds()[s1], 10.0, tolerance);
  EXPECT_NEAR(m.get_constraint_upper_bounds()[s1], 10.0, tolerance);
}

TEST(lp_parser, mixed_constraint_relations)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + 2 y + 3 z
Subject To
 eq1: x + y + z = 10
 geq1: x + 2 y >= 6
 leq1: y + z <= 8
End
)LP");

  int eq  = find_row(m, "eq1");
  int geq = find_row(m, "geq1");
  int leq = find_row(m, "leq1");
  // Relation is recovered from the constraint lower/upper bounds:
  //   'E' ⇒ lb == ub
  //   'G' ⇒ ub = +inf
  //   'L' ⇒ lb = -inf
  EXPECT_NEAR(m.get_constraint_lower_bounds()[eq], m.get_constraint_upper_bounds()[eq], tolerance);
  EXPECT_NEAR(m.get_constraint_lower_bounds()[geq], 6.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_constraint_upper_bounds()[geq]));
  EXPECT_NEAR(m.get_constraint_upper_bounds()[leq], 8.0, tolerance);
  EXPECT_TRUE(std::isinf(-m.get_constraint_lower_bounds()[leq]));
}

TEST(lp_parser, free_and_negative_lower_bound_variables)
{
  auto m = read_lp_string(R"LP(
Minimize
  xfree + xneg_lb + xstd
Subject To
 sum_lb: xfree + xneg_lb + xstd >= 1
 diff_ub: xfree - xneg_lb <= 3
 xst_cap: xstd <= 5
Bounds
 xfree free
 -3 <= xneg_lb <= 10
End
)LP");

  int xf = find_var(m, "xfree");
  int xn = find_var(m, "xneg_lb");
  int xs = find_var(m, "xstd");
  EXPECT_TRUE(std::isinf(-m.get_variable_lower_bounds()[xf]));
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xf]));
  EXPECT_NEAR(m.get_variable_lower_bounds()[xn], -3.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xn], 10.0, tolerance);
  EXPECT_NEAR(m.get_variable_lower_bounds()[xs], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xs]));

  // - xneg_lb → coefficient -1 in the diff_ub row
  int dr = find_row(m, "diff_ub");
  EXPECT_NEAR(a_entry(m, dr, xn), -1.0, tolerance);
}

TEST(lp_parser, bounds_variety)
{
  auto m = read_lp_string(R"LP(
Minimize
  xfixed + xub_only + xlb_pos
Subject To
 c1: xfixed + xub_only + xlb_pos >= 1
Bounds
 xfixed = 3
 xub_only <= 7.5
 xlb_pos >= 2
End
)LP");

  int xfixed = find_var(m, "xfixed");
  int xub    = find_var(m, "xub_only");
  int xlb    = find_var(m, "xlb_pos");
  EXPECT_NEAR(m.get_variable_lower_bounds()[xfixed], 3.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xfixed], 3.0, tolerance);
  EXPECT_NEAR(m.get_variable_lower_bounds()[xub], 0.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xub], 7.5, tolerance);
  EXPECT_NEAR(m.get_variable_lower_bounds()[xlb], 2.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xlb]));
}

TEST(lp_parser, general_integers)
{
  auto m = read_lp_string(R"LP(
Maximize
  3 x + 5 y
Subject To
 c1: x + 2 y <= 12
 c2: 2 x + y <= 10
Generals
 x y
End
)LP");

  int x = find_var(m, "x");
  int y = find_var(m, "y");
  EXPECT_EQ(m.get_variable_types()[x], 'I');
  EXPECT_EQ(m.get_variable_types()[y], 'I');
  // Generals alone does NOT force [0,1]; default bounds remain [0, +inf).
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x]));
}

TEST(lp_parser, binaries_set_zero_one_bounds)
{
  auto m = read_lp_string(R"LP(
Maximize
  3 x1 + 5 x2 + 4 x3 + 2 x4
Subject To
 knapsack: 2 x1 + 3 x2 + x3 + x4 <= 5
Binaries
 x1 x2 x3 x4
End
)LP");

  for (const std::string& n : {"x1", "x2", "x3", "x4"}) {
    int v = find_var(m, n);
    EXPECT_EQ(m.get_variable_types()[v], 'I');
    EXPECT_NEAR(m.get_variable_lower_bounds()[v], 0.0, tolerance);
    EXPECT_NEAR(m.get_variable_upper_bounds()[v], 1.0, tolerance);
  }
}

TEST(lp_parser, mixed_continuous_integer_binary)
{
  auto m = read_lp_string(R"LP(
Maximize
  3 xc + 4 xi + 7 xb
Subject To
 c1: xc + xi + xb <= 10
Generals
 xi
Binaries
 xb
End
)LP");

  int xc = find_var(m, "xc");
  int xi = find_var(m, "xi");
  int xb = find_var(m, "xb");
  EXPECT_EQ(m.get_variable_types()[xc], 'C');
  EXPECT_EQ(m.get_variable_types()[xi], 'I');
  EXPECT_EQ(m.get_variable_types()[xb], 'I');
  EXPECT_NEAR(m.get_variable_upper_bounds()[xb], 1.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xi]));
}

TEST(lp_parser, quadratic_diagonal_only)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + y + [ 2 x ^2 + 4 y ^2 ] / 2
Subject To
 c1: x + y >= 1
Bounds
 x free
 y free
End
)LP");

  ASSERT_TRUE(m.has_quadratic_objective());
  int x = find_var(m, "x");
  int y = find_var(m, "y");
  // LP [2 x^2]/2 = x^2  ⇒  Q[x,x] should be 1 in cuOpt's x^T Q x form.
  EXPECT_NEAR(q_entry(m, x, x), 1.0, tolerance);
  // LP [4 y^2]/2 = 2 y^2  ⇒  Q[y,y] = 2.
  EXPECT_NEAR(q_entry(m, y, y), 2.0, tolerance);
  EXPECT_NEAR(q_entry(m, x, y), 0.0, tolerance);
  // Linear part is preserved.
  EXPECT_NEAR(m.get_objective_coefficients()[x], 1.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 1.0, tolerance);
}

TEST(lp_parser, quadratic_with_cross_terms)
{
  auto m = read_lp_string(R"LP(
Minimize
  - 3 x - 4 y - 2 z + [ 2 x ^2 + 2 x * y + 2 y ^2 + 2 y * z + 2 z ^2 ] / 2
Subject To
 c1: x + y + z <= 10
 c2: x + y >= 1
End
)LP");

  ASSERT_TRUE(m.has_quadratic_objective());
  int x = find_var(m, "x");
  int y = find_var(m, "y");
  int z = find_var(m, "z");
  // The LP parser stores Q in upper-triangular form (i <= j). cuOpt's
  // set_quadratic_objective_matrix symmetrizes via H = Q + Q^T, and the
  // solver minimizes (1/2) x^T H x.
  // Diagonal 2 x^2 / 2 → Q[x,x] = 1.
  EXPECT_NEAR(q_entry(m, x, x), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, y, y), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, z, z), 1.0, tolerance);
  // Cross 2 x*y / 2 → stored as Q[x,y] = 1 only (no Q[y,x]).
  EXPECT_NEAR(q_entry(m, x, y), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, y, x), 0.0, tolerance);
  EXPECT_NEAR(q_entry(m, y, z), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, z, y), 0.0, tolerance);
  // x and z have no cross term.
  EXPECT_NEAR(q_entry(m, x, z), 0.0, tolerance);

  EXPECT_NEAR(m.get_objective_coefficients()[x], -3.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], -4.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[z], -2.0, tolerance);
}

TEST(lp_parser, miqp_integer_with_quadratic_objective)
{
  auto m = read_lp_string(R"LP(
Minimize
  - 4 xi - 2 xc + [ 2 xi ^2 + 2 xc ^2 ] / 2
Subject To
 c1: xi + xc <= 5
Bounds
 xi <= 4
Generals
 xi
End
)LP");

  int xi = find_var(m, "xi");
  int xc = find_var(m, "xc");
  EXPECT_EQ(m.get_variable_types()[xi], 'I');
  EXPECT_EQ(m.get_variable_types()[xc], 'C');
  EXPECT_NEAR(m.get_variable_upper_bounds()[xi], 4.0, tolerance);
  EXPECT_NEAR(q_entry(m, xi, xi), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, xc, xc), 1.0, tolerance);
}

TEST(lp_parser, infeasible_model_parses_faithfully)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 high: x + y >= 15
 low: x + y <= 8
Bounds
 x <= 5
 y <= 5
End
)LP");

  EXPECT_EQ(m.get_row_names().size(), 2u);
  int high = find_row(m, "high");
  int low  = find_row(m, "low");
  EXPECT_NEAR(m.get_constraint_lower_bounds()[high], 15.0, tolerance);
  EXPECT_NEAR(m.get_constraint_upper_bounds()[low], 8.0, tolerance);
}

TEST(lp_parser, unbounded_model_parses)
{
  auto m = read_lp_string(R"LP(
Maximize
  x + y
Subject To
 c1: x - y <= 5
End
)LP");

  int x = find_var(m, "x");
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x]));
  EXPECT_TRUE(m.get_sense());
}

TEST(lp_parser, missing_objective_throws)
{
  EXPECT_THROW(read_lp_string(R"LP(
Subject To
 c1: x + y <= 5
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unsupported_sos_section_throws)
{
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
SOS
 s1: S1 :: x : 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, semi_continuous_basic)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 c1: x + y >= 1
Bounds
 2 <= x <= 10
 y <= 5
Semi-Continuous
 x
End
)LP");
  ASSERT_EQ(m.get_variable_names().size(), 2u);
  int xi = find_var(m, "x");
  int yi = find_var(m, "y");
  ASSERT_GE(xi, 0);
  ASSERT_GE(yi, 0);
  EXPECT_EQ(m.get_variable_types()[xi], 'S');
  EXPECT_EQ(m.get_variable_types()[yi], 'C');
  EXPECT_NEAR(m.get_variable_lower_bounds()[xi], 2.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xi], 10.0, tolerance);
}

TEST(lp_parser, semi_continuous_bare_semi_keyword)
{
  // The LP-format convention accepts the bare "Semi" keyword as a synonym
  // for the "Semi-Continuous" section header.
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 0
Bounds
 2 <= x <= 10
Semi
 x
End
)LP");
  int xi = find_var(m, "x");
  ASSERT_GE(xi, 0);
  EXPECT_EQ(m.get_variable_types()[xi], 'S');
  EXPECT_NEAR(m.get_variable_lower_bounds()[xi], 2.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xi], 10.0, tolerance);
}

TEST(lp_parser, semi_continuous_bare_semis_keyword)
{
  // The LP-format convention accepts the bare "Semis" keyword as a synonym
  // for the "Semi-Continuous" section header.
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 0
Bounds
 2 <= x <= 10
Semis
 x
End
)LP");
  int xi = find_var(m, "x");
  ASSERT_GE(xi, 0);
  EXPECT_EQ(m.get_variable_types()[xi], 'S');
  EXPECT_NEAR(m.get_variable_lower_bounds()[xi], 2.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xi], 10.0, tolerance);
}

TEST(lp_parser, semi_continuous_default_lower_is_zero)
{
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 0
Bounds
 x <= 3
Semi-Continuous
 x
End
)LP");
  int xi = find_var(m, "x");
  ASSERT_GE(xi, 0);
  EXPECT_EQ(m.get_variable_types()[xi], 'S');
  // No explicit lower in Bounds ⇒ default 0.
  EXPECT_NEAR(m.get_variable_lower_bounds()[xi], 0.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xi], 3.0, tolerance);
}

TEST(lp_parser, semi_continuous_missing_upper_throws)
{
  // No upper bound specified ⇒ infinity ⇒ semantics degenerate, reject.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 0
Semi-Continuous
 x
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, semi_continuous_and_generals_conflict_throws)
{
  // Variable appearing in both Semi-Continuous and Generals is ambiguous
  // (integer vs. continuous-or-zero) ⇒ reject.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 0
Bounds
 x <= 5
Generals
 x
Semi-Continuous
 x
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, semi_continuous_and_binaries_conflict_throws)
{
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 0
Bounds
 x <= 5
Binaries
 x
Semi-Continuous
 x
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, semi_continuous_before_generals_conflict_throws)
{
  // Conflict must also be detected when Semi-Continuous is declared first.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 0
Bounds
 x <= 5
Semi-Continuous
 x
Generals
 x
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unsupported_pwlobj_section_throws)
{
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
PWLObj
 x: 0 0 1 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unsupported_lazy_constraints_section_throws)
{
  // Lazy constraints and user cuts are scope-limited out: LP/MIP/QP only.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
Lazy Constraints
 lc: x <= 10
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unsupported_user_cuts_section_throws)
{
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
User Cuts
 uc: x <= 10
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unknown_file_throws)
{
  auto call = [] { return read_lp<int, double>("/definitely/does/not/exist.lp"); };
  EXPECT_THROW(call(), std::logic_error);
}

TEST(lp_parser, case_insensitive_section_keywords)
{
  auto m = read_lp_string(R"LP(
MINIMIZE
  x
SUBJECT TO
 c1: x >= 1
BOUNDS
 x <= 5
END
)LP");
  int x  = find_var(m, "x");
  EXPECT_NEAR(m.get_variable_upper_bounds()[x], 5.0, tolerance);
}

TEST(lp_parser, backslash_comments_are_ignored)
{
  auto m = read_lp_string(R"LP(
\ This is a comment
Minimize
  x \ trailing comment
Subject To \ another comment
 c1: x >= 1
End
)LP");
  int x  = find_var(m, "x");
  EXPECT_NEAR(m.get_objective_coefficients()[x], 1.0, tolerance);
}

TEST(lp_parser, missing_end_warns_but_succeeds)
{
  // No End — should still parse. (A warning is printed; see parse_all().)
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
)LP");
  EXPECT_EQ(m.get_variable_names().size(), 1u);
}

TEST(lp_parser, auto_generates_names_for_unlabeled_constraints)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 x + y <= 10
 x - y >= 0
End
)LP");
  ASSERT_EQ(m.get_row_names().size(), 2u);
  // Default auto-generated names are R0, R1.
  EXPECT_EQ(m.get_row_names()[0], "R0");
  EXPECT_EQ(m.get_row_names()[1], "R1");
}

TEST(lp_parser, infinity_keyword_in_bounds)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 c1: x + y >= 0
Bounds
 -inf <= x <= inf
 -infinity <= y
End
)LP");
  int x  = find_var(m, "x");
  int y  = find_var(m, "y");
  EXPECT_TRUE(std::isinf(-m.get_variable_lower_bounds()[x]));
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x]));
  EXPECT_TRUE(std::isinf(-m.get_variable_lower_bounds()[y]));
}

TEST(lp_parser, coefficient_one_implicit_with_leading_minus)
{
  auto m = read_lp_string(R"LP(
Minimize
  - x + y
Subject To
 c1: - x + y <= 0
End
)LP");
  int x  = find_var(m, "x");
  int y  = find_var(m, "y");
  EXPECT_NEAR(m.get_objective_coefficients()[x], -1.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 1.0, tolerance);
  int r = find_row(m, "c1");
  EXPECT_NEAR(a_entry(m, r, x), -1.0, tolerance);
}

TEST(lp_parser, quadratic_without_slash_two_is_rejected)
{
  // The quadratic bracket in the objective must be followed by '/ 2'.
  // Without it there's no unambiguous way to tell whether the user meant
  // '/ 2' and forgot or intended the bare coefficients, so cuopt rejects.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  [ 1 x ^2 ]
Subject To
 c1: x >= 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, leading_coefficient_before_objective_bracket_rejected)
{
  // '2 [ x^2 ] / 2' is ambiguous between "constant 2 plus 0.5 x^2" and
  // "scalar 2 times 0.5 x^2"; the LP convention is to place coefficients
  // inside the brackets, so reject.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  2 [ x ^ 2 ] / 2
Subject To
 c1: x >= 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, leading_coefficient_before_constraint_bracket_rejected)
{
  // Same ambiguity as the objective case, in a quadratic constraint.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 q1: 2 [ x ^ 2 ] <= 5
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, constant_then_signed_bracket_in_objective_is_accepted)
{
  // The positive form: a literal constant in the objective followed by a
  // signed quadratic bracket still parses (constant becomes objective offset).
  auto m = read_lp_string(R"LP(
Minimize
  5 + [ x ^ 2 ] / 2
Subject To
 c1: x >= 1
End
)LP");
  EXPECT_NEAR(m.get_objective_offset(), 5.0, tolerance);
  EXPECT_TRUE(m.has_quadratic_objective());
}

TEST(lp_parser, stray_star_after_number_without_variable_rejected)
{
  // '3 *' followed by a relation, section header, or EOL must error rather
  // than silently drop the '*' and treat the '3' as a bare constant.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  3 *
Subject To
 c1: x >= 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, explicit_star_between_coefficient_and_variable_is_accepted)
{
  // The positive form: '3 * x' is the same as '3 x'.
  auto m = read_lp_string(R"LP(
Minimize
  3 * x
Subject To
 c1: x >= 1
End
)LP");
  int x  = find_var(m, "x");
  ASSERT_GE(x, 0);
  EXPECT_NEAR(m.get_objective_coefficients()[x], 3.0, tolerance);
}

// ===========================================================================
// Quadratic constraints (LHS contains [ ... ] without the /2 divisor).
// ===========================================================================

// Returns the kth quadratic constraint of `m` (`k` is the 0-indexed position
// in the order quadratic constraints were declared in the LP file).
const auto& nth_qc(const mps_data_model_t<int, double>& m, size_t k)
{
  const auto& qcs = m.get_quadratic_constraints();
  return qcs.at(k);
}

TEST(lp_parser, qc_basic_diagonal_only)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 q1: [ x ^ 2 + y ^ 2 ] <= 10
Bounds
 x free
 y free
End
)LP");
  ASSERT_EQ(m.get_quadratic_constraints().size(), 1u);
  const auto& qc = nth_qc(m, 0);
  EXPECT_EQ(qc.constraint_row_name, "q1");
  EXPECT_EQ(qc.constraint_row_type, static_cast<char>(LesserThanOrEqual));
  EXPECT_NEAR(qc.rhs_value, 10.0, tolerance);
  EXPECT_TRUE(qc.linear_indices.empty());
  // Q = diag(1, 1) stored as COO triplets (row, col, value).
  EXPECT_EQ(qc.rows, (std::vector<int>{0, 1}));
  EXPECT_EQ(qc.cols, (std::vector<int>{0, 1}));
  ASSERT_EQ(qc.vals.size(), 2u);
  EXPECT_NEAR(qc.vals[0], 1.0, tolerance);
  EXPECT_NEAR(qc.vals[1], 1.0, tolerance);
}

TEST(lp_parser, qc_cross_term_splits_symmetrically)
{
  // `4 x*y` in the LP source means coefficient on x_i * x_j = 4 in the
  // symmetric x^T Q x. Split into Q[x,y] = Q[y,x] = 2 in the CSR.
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 q1: [ x ^ 2 + 4 x * y + y ^ 2 ] <= 5
End
)LP");
  ASSERT_EQ(m.get_quadratic_constraints().size(), 1u);
  const auto& qc = nth_qc(m, 0);
  // Q has 4 entries (all of [[1,2],[2,1]]) stored as COO triplets.
  EXPECT_EQ(qc.rows, (std::vector<int>{0, 0, 1, 1}));
  EXPECT_EQ(qc.cols, (std::vector<int>{0, 1, 0, 1}));
  ASSERT_EQ(qc.vals.size(), 4u);
  EXPECT_NEAR(qc.vals[0], 1.0, tolerance);  // (0, 0)
  EXPECT_NEAR(qc.vals[1], 2.0, tolerance);  // (0, 1)
  EXPECT_NEAR(qc.vals[2], 2.0, tolerance);  // (1, 0)
  EXPECT_NEAR(qc.vals[3], 1.0, tolerance);  // (1, 1)
}

TEST(lp_parser, qc_linear_and_quadratic_mixed)
{
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 q1: 3 x + 2 y + [ x ^ 2 + y ^ 2 ] <= 7
End
)LP");
  ASSERT_EQ(m.get_quadratic_constraints().size(), 1u);
  const auto& qc = nth_qc(m, 0);
  EXPECT_NEAR(qc.rhs_value, 7.0, tolerance);
  // Linear part: 3 x + 2 y.
  ASSERT_EQ(qc.linear_indices.size(), 2u);
  ASSERT_EQ(qc.linear_values.size(), 2u);
  // Indices may be in any order; check coefficients via lookup.
  std::vector<int> xi_yi = {find_var(m, "x"), find_var(m, "y")};
  std::vector<double> expected_coefs;
  for (size_t i = 0; i < qc.linear_indices.size(); ++i) {
    if (qc.linear_indices[i] == xi_yi[0]) EXPECT_NEAR(qc.linear_values[i], 3.0, tolerance);
    if (qc.linear_indices[i] == xi_yi[1]) EXPECT_NEAR(qc.linear_values[i], 2.0, tolerance);
  }
}

TEST(lp_parser, qc_multiple_constraints_indexing)
{
  // 2 linear constraints, then 2 quadratic constraints. Per the data-model
  // convention, quadratic rows are indexed after all linear rows.
  auto m = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 c1: x + y <= 100
 c2: x - y >= -50
 q1: [ x ^ 2 ] <= 1
 q2: [ y ^ 2 ] <= 4
End
)LP");
  EXPECT_EQ(m.get_row_names().size(), 2u);  // linear rows only
  ASSERT_EQ(m.get_quadratic_constraints().size(), 2u);
  EXPECT_EQ(nth_qc(m, 0).constraint_row_index, 2);
  EXPECT_EQ(nth_qc(m, 0).constraint_row_name, "q1");
  EXPECT_EQ(nth_qc(m, 1).constraint_row_index, 3);
  EXPECT_EQ(nth_qc(m, 1).constraint_row_name, "q2");
}

TEST(lp_parser, qc_outer_minus_sign_flips_quadratic)
{
  // `- 2 x + 5 - [ x^2 ]` on the LHS contributes -x^2 - 2 x + 5 to the LHS.
  // After moving the constant to the RHS: -x^2 - 2 x <= rhs - 5.
  // Here the RHS is 10, so the row becomes: -x^2 - 2 x <= 5  (in x^T Q x form
  // Q[x,x] = -1).
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 q1: - 2 x + 5 - [ x ^ 2 ] <= 10
Bounds
 x free
End
)LP");
  ASSERT_EQ(m.get_quadratic_constraints().size(), 1u);
  const auto& qc = nth_qc(m, 0);
  EXPECT_NEAR(qc.rhs_value, 5.0, tolerance);
  ASSERT_EQ(qc.vals.size(), 1u);
  EXPECT_NEAR(qc.vals[0], -1.0, tolerance);
  ASSERT_EQ(qc.linear_indices.size(), 1u);
  EXPECT_NEAR(qc.linear_values[0], -2.0, tolerance);
}

TEST(lp_parser, bare_linear_inside_objective_bracket_rejected)
{
  // The LP-format convention reserves `[ ... ]` for quadratic terms only
  // (squared and product). A bare linear term like `2 x` inside the
  // bracket is malformed; the user should write it outside.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  obj: [ x ^ 2 + 2 x ] / 2
Subject To
  c1: x >= 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, bare_linear_inside_constraint_bracket_rejected)
{
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
  q1: [ x ^ 2 + 2 x ] <= 5
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, qc_named_constraint)
{
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 my_quad: [ x ^ 2 ] <= 1
End
)LP");
  ASSERT_EQ(m.get_quadratic_constraints().size(), 1u);
  EXPECT_EQ(nth_qc(m, 0).constraint_row_name, "my_quad");
}

TEST(lp_parser, qc_ge_relation_throws)
{
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 q1: [ x ^ 2 ] >= 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, qc_eq_relation_throws)
{
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 q1: [ x ^ 2 ] = 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, qc_with_slash_two_is_rejected)
{
  // '/ 2' is reserved for the objective bracket; using it in a constraint
  // bracket is rejected so the convention is unambiguous.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 q1: [ x ^ 2 ] / 2 <= 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, qc_linear_only_bracket_is_rejected)
{
  // A bracket with no quadratic terms inside is meaningless in a constraint
  // (the user could just write the linear terms directly).
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 q1: [ 2 x ] <= 5
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, qc_objective_quadratic_still_requires_slash_two)
{
  // Regression: the existing '/ 2' requirement on the objective bracket
  // must not change after adding constraint-bracket support.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  [ x ^ 2 ]
Subject To
 c1: x >= 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, duplicate_coefficient_accumulates)
{
  // Repeated variable in the objective should sum coefficients.
  auto m = read_lp_string(R"LP(
Minimize
  2 x + 3 x + y
Subject To
 c1: x + y >= 1
End
)LP");
  int x  = find_var(m, "x");
  int y  = find_var(m, "y");
  EXPECT_NEAR(m.get_objective_coefficients()[x], 5.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 1.0, tolerance);
}

TEST(lp_parser, subject_to_variant_st_dot)
{
  // 'st.' with a trailing period is a Subject-To synonym in the LP-format
  // convention.
  auto m = read_lp_string(R"LP(
Minimize
  x
st.
 c: x >= 1
End
)LP");
  EXPECT_EQ(m.get_row_names().size(), 1u);
  EXPECT_EQ(m.get_row_names()[0], "c");
}

TEST(lp_parser, swapped_relational_operators_eq_lt_and_eq_gt)
{
  // '=<' is an alias for '<=' and '=>' for '>=', in both constraints and
  // bounds. Tokenizer must produce LessEq / GreaterEq tokens regardless of
  // spelling.
  auto m   = read_lp_string(R"LP(
Minimize
  x + y
Subject To
 c_le: x + y =< 10
 c_ge: x + y => 1
Bounds
 y =< 5
 x => 0
End
)LP");
  int c_le = find_row(m, "c_le");
  int c_ge = find_row(m, "c_ge");
  EXPECT_TRUE(std::isinf(-m.get_constraint_lower_bounds()[c_le]));
  EXPECT_NEAR(m.get_constraint_upper_bounds()[c_le], 10.0, tolerance);
  EXPECT_NEAR(m.get_constraint_lower_bounds()[c_ge], 1.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_constraint_upper_bounds()[c_ge]));
  int x = find_var(m, "x");
  int y = find_var(m, "y");
  EXPECT_NEAR(m.get_variable_upper_bounds()[y], 5.0, tolerance);
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], 0.0, tolerance);
}

TEST(lp_parser, variable_names_with_special_characters)
{
  // Per the LP-format convention, variable names may contain assorted
  // punctuation beyond letters + underscore. The names are treated as
  // opaque identifiers; cuopt just has to keep them distinct.
  auto m = read_lp_string(R"LP(
Minimize
  x!a + x#b + x$c + x@d + x'e + x~f + x.g + x_h + x|i + x{j} + x(k) + a/b
Subject To
 c1: x!a + x#b + x$c + x@d + x'e + x~f + x.g + x_h + x|i + x{j} + x(k) + a/b >= 1
End
)LP");
  ASSERT_EQ(m.get_variable_names().size(), 12u);
  for (const std::string& n :
       {"x!a", "x#b", "x$c", "x@d", "x'e", "x~f", "x.g", "x_h", "x|i", "x{j}", "x(k)", "a/b"}) {
    EXPECT_GE(find_var(m, n), 0) << "missing variable '" << n << "'";
  }
}

TEST(lp_parser, negative_upper_without_explicit_lower_throws)
{
  // 'x <= -1' with no explicit lower makes the default lb=0 collide with the
  // upper. cuopt rejects rather than accept a silently infeasible problem.
  EXPECT_THROW(read_lp_string(R"LP(
Minimize
  x
Subject To
 c: x <= 10
Bounds
 x <= -1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, negative_upper_with_explicit_lower_ok)
{
  // Same test as above, but now the lower bound is explicit: no error.
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 c: x <= 10
Bounds
 x >= -5
 x <= -1
End
)LP");
  int x  = find_var(m, "x");
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], -5.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[x], -1.0, tolerance);
}

TEST(lp_parser, negative_upper_with_range_bound_ok)
{
  // -5 <= x <= -1 declares both bounds in a single line: no error.
  auto m = read_lp_string(R"LP(
Minimize
  x
Subject To
 c: x <= 10
Bounds
 -5 <= x <= -1
End
)LP");
  int x  = find_var(m, "x");
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], -5.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[x], -1.0, tolerance);
}

// ================================================================================================
// read dispatch tests
//
// Verifies the extension-based dispatch used by cuopt_cli and the C API.
// ================================================================================================

namespace {

// Writes `content` to a temp file with the given suffix, parses it via
// read, and returns the resulting model. temp_file_t removes the
// file on every scope exit (including when read throws).
mps_data_model_t<int, double> dispatch_parse(const std::string& content, const std::string& suffix)
{
  temp_file_t tmp(suffix);
  {
    std::ofstream out(tmp.string());
    out << content;
  }
  return read<int, double>(tmp.string());
}

constexpr const char* kTrivialLp = R"LP(
Minimize
  x
Subject To
 c1: x >= 2.5
Bounds
 x <= 10
End
)LP";

constexpr const char* kTrivialMps = R"MPS(NAME trivial
ROWS
 N OBJ
 G c1
COLUMNS
 x OBJ 1
 x c1 1
RHS
 RHS1 c1 2.5
BOUNDS
 UP BND1 x 10
ENDATA
)MPS";

}  // namespace

TEST(read, lp_extension_dispatches_to_lp_parser)
{
  auto m = dispatch_parse(kTrivialLp, ".lp");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
  EXPECT_NEAR(m.get_variable_upper_bounds()[0], 10.0, tolerance);
}

TEST(read, lp_gz_extension_dispatches_to_lp_parser)
{
  // Real compressed LP fixture; successful parse proves dispatch picked the
  // LP path. (Routing a .lp.gz to read_mps would either fail at
  // decompression or fail to parse the LP content as MPS.)
  auto m = read<int, double>(cuopt::test::get_rapids_dataset_root_dir() +
                             "/linear_programming/good-mps-1.lp.gz");
  ASSERT_EQ(m.get_variable_names().size(), 2u);
  EXPECT_EQ(m.get_variable_names()[0], "VAR1");
}

TEST(read, lp_bz2_extension_dispatches_to_lp_parser)
{
  auto m = read<int, double>(cuopt::test::get_rapids_dataset_root_dir() +
                             "/linear_programming/good-mps-1.lp.bz2");
  ASSERT_EQ(m.get_variable_names().size(), 2u);
  EXPECT_EQ(m.get_variable_names()[0], "VAR1");
}

TEST(read, mps_extension_dispatches_to_mps_parser)
{
  auto m = dispatch_parse(kTrivialMps, ".mps");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
  EXPECT_NEAR(m.get_variable_upper_bounds()[0], 10.0, tolerance);
}

TEST(read, qps_extension_dispatches_to_mps_parser)
{
  // QPS is a superset of MPS; the MPS parser handles both. We just need
  // read to route ".qps" to it.
  auto m = dispatch_parse(kTrivialMps, ".qps");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
}

TEST(read, mps_gz_extension_dispatches_to_mps_parser)
{
  auto m = read<int, double>(cuopt::test::get_rapids_dataset_root_dir() +
                             "/linear_programming/good-mps-1.mps.gz");
  EXPECT_EQ("good-1", m.get_problem_name());
}

TEST(read, mps_bz2_extension_dispatches_to_mps_parser)
{
  auto m = read<int, double>(cuopt::test::get_rapids_dataset_root_dir() +
                             "/linear_programming/good-mps-1.mps.bz2");
  EXPECT_EQ("good-1", m.get_problem_name());
}

TEST(read, uppercase_lp_extension_dispatches_to_lp_parser)
{
  // Matching is case-insensitive: .LP must still route to read_lp.
  auto m = dispatch_parse(kTrivialLp, ".LP");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
}

TEST(read, mixed_case_mps_extension_dispatches_to_mps_parser)
{
  auto m = dispatch_parse(kTrivialMps, ".MpS");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
}

TEST(read, unrecognized_extension_throws)
{
  // Extensionless and unrelated suffixes are rejected; case doesn't matter
  // (matching is case-insensitive, so ".lpgz" stays rejected too).
  for (const char* suffix : {".txt", ".lpgz", ""}) {
    SCOPED_TRACE(suffix);
    EXPECT_THROW(dispatch_parse(kTrivialLp, suffix), std::logic_error);
  }
}

// ===========================================================================
// MPS-syntax-specific tests: bound codes (UP/LO/MI/PL/BV/SC) and QCMATRIX
// blocks. LP-equivalent semantic coverage lives above.
// ===========================================================================

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

TEST(mps_bounds, semi_continuous_var_bounds_from_dataset)
{
  struct Case {
    const char* name;
    const char* mps;
    int n_vars;
    double lower;
    double upper;
  };
  const std::vector<Case> cases = {
    {"sc_standard", cuopt::test::inline_mps::sc_standard_mps, 2, 2.0, 10.0},
    {"sc_lb_zero", cuopt::test::inline_mps::sc_lb_zero_mps, 2, 0.0, 10.0},
    {"sc_no_ub", cuopt::test::inline_mps::sc_no_ub_mps, 2, 2.0, 1e30},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    auto mps              = cuopt::test::inline_mps::parse_inline_mps(c.mps);
    const auto& var_types = mps.get_variable_types();
    const auto& lower     = mps.get_variable_lower_bounds();
    const auto& upper     = mps.get_variable_upper_bounds();

    ASSERT_EQ(c.n_vars, static_cast<int>(var_types.size()));
    EXPECT_EQ('S', var_types[0]);
    ASSERT_EQ(c.n_vars, static_cast<int>(lower.size()));
    ASSERT_EQ(c.n_vars, static_cast<int>(upper.size()));
    EXPECT_DOUBLE_EQ(c.lower, lower[0]);
    EXPECT_DOUBLE_EQ(c.upper, upper[0]);
  }
}

TEST(mps_bounds, semi_continuous_missing_lower_defaults_to_zero)
{
  auto mps = cuopt::test::inline_mps::parse_inline_mps(cuopt::test::inline_mps::sc_lb_zero_mps);
  const auto& var_types = mps.get_variable_types();
  const auto& lower     = mps.get_variable_lower_bounds();
  const auto& upper     = mps.get_variable_upper_bounds();

  ASSERT_EQ(2, static_cast<int>(var_types.size()));
  EXPECT_EQ('S', var_types[0]);
  ASSERT_EQ(2, static_cast<int>(lower.size()));
  ASSERT_EQ(2, static_cast<int>(upper.size()));
  EXPECT_DOUBLE_EQ(0.0, lower[0]);
  EXPECT_DOUBLE_EQ(10.0, upper[0]);
}

TEST(mps_bounds, semi_continuous_missing_upper_rejected)
{
  EXPECT_THROW(
    cuopt::test::inline_mps::parse_inline_mps(cuopt::test::inline_mps::sc_missing_upper_mps),
    std::logic_error);
}

TEST(mps_bounds, semi_continuous_bound_type)
{
  auto mps = read_from_mps("linear_programming/good-mps-semi-continuous-bound.mps", false);

  ASSERT_EQ(int(2), mps.var_names.size());
  ASSERT_EQ(int(2), mps.var_types.size());
  EXPECT_EQ('S', mps.var_types[0]);
  ASSERT_EQ(int(2), mps.variable_lower_bounds.size());
  ASSERT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_DOUBLE_EQ(0.0, mps.variable_lower_bounds[0]);
  EXPECT_DOUBLE_EQ(2.0, mps.variable_upper_bounds[0]);
}

TEST(mps_bounds, invalid_bound_type)
{
  ASSERT_THROW(read_from_mps("linear_programming/bad-mps-bound-1.mps", false), std::logic_error);
}

TEST(qps_parser, qcmatrix_append_api)
{
  using model_t = mps_data_model_t<int, double>;
  model_t model;

  // Validate default-constructed struct shape.
  model_t::quadratic_constraint_t default_qcm;
  EXPECT_EQ(0, default_qcm.constraint_row_index);
  EXPECT_TRUE(default_qcm.vals.empty());
  EXPECT_TRUE(default_qcm.rows.empty());
  EXPECT_TRUE(default_qcm.cols.empty());
  EXPECT_TRUE(default_qcm.linear_values.empty());
  EXPECT_TRUE(default_qcm.linear_indices.empty());
  EXPECT_EQ(0.0, default_qcm.rhs_value);

  // QC0: [[10, 2], [2, 2]]
  const std::vector<double> qc0_values        = {10.0, 2.0, 2.0, 2.0};
  const std::vector<int> qc0_row_indices      = {0, 0, 1, 1};
  const std::vector<int> qc0_col_indices      = {0, 1, 0, 1};
  const std::vector<double> qc0_linear_values = {1.0, 1.0};
  const std::vector<int> qc0_linear_indices   = {0, 1};
  model.append_quadratic_constraint(0,
                                    "QC0",
                                    'L',
                                    qc0_linear_values,
                                    qc0_linear_indices,
                                    5.0,
                                    qc0_values,
                                    qc0_row_indices,
                                    qc0_col_indices);

  // QC1: [[4, 1], [1, 6]]
  const std::vector<double> qc1_values        = {4.0, 1.0, 1.0, 6.0};
  const std::vector<int> qc1_row_indices      = {0, 0, 1, 1};
  const std::vector<int> qc1_col_indices      = {0, 1, 0, 1};
  const std::vector<double> qc1_linear_values = {3.0, 1.0};
  const std::vector<int> qc1_linear_indices   = {0, 1};
  model.append_quadratic_constraint(1,
                                    "QC1",
                                    'L',
                                    qc1_linear_values,
                                    qc1_linear_indices,
                                    10.0,
                                    qc1_values,
                                    qc1_row_indices,
                                    qc1_col_indices);

  ASSERT_TRUE(model.has_quadratic_constraints());
  const auto& qcs = model.get_quadratic_constraints();
  ASSERT_EQ(2u, qcs.size());

  EXPECT_EQ(0, qcs[0].constraint_row_index);
  EXPECT_EQ("QC0", qcs[0].constraint_row_name);
  EXPECT_EQ('L', qcs[0].constraint_row_type);
  EXPECT_EQ(qc0_linear_values, qcs[0].linear_values);
  EXPECT_EQ(qc0_linear_indices, qcs[0].linear_indices);
  EXPECT_EQ(5.0, qcs[0].rhs_value);
  EXPECT_EQ(qc0_values, qcs[0].vals);
  EXPECT_EQ(qc0_row_indices, qcs[0].rows);
  EXPECT_EQ(qc0_col_indices, qcs[0].cols);

  EXPECT_EQ(1, qcs[1].constraint_row_index);
  EXPECT_EQ("QC1", qcs[1].constraint_row_name);
  EXPECT_EQ('L', qcs[1].constraint_row_type);
  EXPECT_EQ(qc1_linear_values, qcs[1].linear_values);
  EXPECT_EQ(qc1_linear_indices, qcs[1].linear_indices);
  EXPECT_EQ(10.0, qcs[1].rhs_value);
  EXPECT_EQ(qc1_values, qcs[1].vals);
  EXPECT_EQ(qc1_row_indices, qcs[1].rows);
  EXPECT_EQ(qc1_col_indices, qcs[1].cols);
}

// QCQP MPS: each quadratic constraint bundles row + linear + rhs + quadratic.
TEST(qps_parser, qcmatrix_mps_linear_rhs_and_bounds)
{
  if (!file_exists("qcqp/QC_Test_1.mps")) {
    GTEST_SKIP() << "qcqp/QC_Test_1.mps not in dataset root";
  }
  const auto model = read_mps<int, double>(
    cuopt::test::get_rapids_dataset_root_dir() + "/qcqp/QC_Test_1.mps", false);

  ASSERT_TRUE(model.has_quadratic_constraints());
  const auto& qcs = model.get_quadratic_constraints();
  ASSERT_EQ(2u, qcs.size());

  ASSERT_EQ(1, model.get_n_constraints());
  ASSERT_EQ(1u, model.get_row_names().size());
  EXPECT_EQ("LIN0", model.get_row_names()[0]);
  EXPECT_EQ('L', model.get_row_types()[0]);

  // LIN0: 2*x1 + x2 ≤ 15 (linear row only; not duplicated in quadratic_constraints)
  EXPECT_DOUBLE_EQ(-std::numeric_limits<double>::infinity(),
                   model.get_constraint_lower_bounds()[0]);
  EXPECT_DOUBLE_EQ(15.0, model.get_constraint_upper_bounds()[0]);
  const auto& A_off = model.get_constraint_matrix_offsets();
  const auto& A_val = model.get_constraint_matrix_values();
  const auto& A_idx = model.get_constraint_matrix_indices();
  ASSERT_EQ(2, A_off[1] - A_off[0]);
  EXPECT_EQ(2.0, A_val[A_off[0] + 0]);
  EXPECT_EQ(1.0, A_val[A_off[0] + 1]);
  EXPECT_EQ(0, A_idx[A_off[0] + 0]);
  EXPECT_EQ(1, A_idx[A_off[0] + 1]);

  // QC0: x1 + x2 + xᵀQ₀x ≤ 5 (MPS ROWS declaration index 1; OBJ 'N' rows are not counted)
  EXPECT_EQ(1, qcs[0].constraint_row_index);
  EXPECT_EQ("QC0", qcs[0].constraint_row_name);
  EXPECT_EQ('L', qcs[0].constraint_row_type);
  ASSERT_EQ(2u, qcs[0].linear_values.size());
  EXPECT_EQ(1.0, qcs[0].linear_values[0]);
  EXPECT_EQ(1.0, qcs[0].linear_values[1]);
  EXPECT_EQ(0, qcs[0].linear_indices[0]);
  EXPECT_EQ(1, qcs[0].linear_indices[1]);
  EXPECT_DOUBLE_EQ(5.0, qcs[0].rhs_value);
  EXPECT_FALSE(qcs[0].vals.empty());

  // QC1: 3*x1 + x2 + xᵀQ₁x ≤ 10
  EXPECT_EQ(2, qcs[1].constraint_row_index);
  EXPECT_EQ("QC1", qcs[1].constraint_row_name);
  EXPECT_EQ('L', qcs[1].constraint_row_type);
  ASSERT_EQ(2u, qcs[1].linear_values.size());
  EXPECT_EQ(3.0, qcs[1].linear_values[0]);
  EXPECT_EQ(1.0, qcs[1].linear_values[1]);
  EXPECT_DOUBLE_EQ(10.0, qcs[1].rhs_value);
}

TEST(qps_parser, qcqp_p0033_mps_sections)
{
  if (!file_exists("qcqp/p0033_qc1.mps")) {
    GTEST_SKIP() << "qcqp/p0033_qc1.mps not in dataset root";
  }
  const auto model = read_mps<int, double>(
    cuopt::test::get_rapids_dataset_root_dir() + "/qcqp/p0033_qc1.mps", false);

  EXPECT_EQ(12, model.get_n_constraints());
  EXPECT_EQ(33, model.get_n_variables());
  ASSERT_EQ(12u, model.get_row_types().size());
  ASSERT_EQ(12u, model.get_row_names().size());

  const auto& qcs = model.get_quadratic_constraints();
  ASSERT_EQ(4u, qcs.size());
  EXPECT_EQ(12, qcs[0].constraint_row_index);
  ASSERT_EQ(1u, qcs[0].linear_values.size());
  EXPECT_DOUBLE_EQ(1.0, qcs[0].linear_values[0]);

  const auto& vnames = model.get_variable_names();
  auto c159_it       = std::find(vnames.begin(), vnames.end(), std::string("C159"));
  ASSERT_NE(c159_it, vnames.end());
  EXPECT_EQ(static_cast<int>(c159_it - vnames.begin()), qcs[0].linear_indices[0]);

  EXPECT_DOUBLE_EQ(1.0, qcs[0].rhs_value);
  EXPECT_FALSE(qcs[0].vals.empty());
}

TEST(mps_roundtrip, qcqp_p0033_qc1)
{
  if (!file_exists("qcqp/p0033_qc1.mps")) { GTEST_SKIP() << "Test file not found"; }

  std::string input_file = cuopt::test::get_rapids_dataset_root_dir() + "/qcqp/p0033_qc1.mps";
  temp_file_t temp_file(".mps");
  temp_file_t temp_file_2(".mps");

  auto original = read_mps<int, double>(input_file, false);
  ASSERT_TRUE(original.has_quadratic_objective());
  ASSERT_TRUE(original.has_quadratic_constraints());

  mps_writer_t<int, double> writer(original);
  writer.write(temp_file.string());

  auto reloaded = read_mps<int, double>(temp_file.string(), false);
  mps_writer_t<int, double> writer_r2(reloaded);
  writer_r2.write(temp_file_2.string());
  auto reloaded_2 = read_mps<int, double>(temp_file_2.string(), false);
  compare_data_models(reloaded, reloaded_2);
}
}  // namespace cuopt::mathematical_optimization::io
