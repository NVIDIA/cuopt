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
 #include <mps_parser/data_model_view.hpp>

 #include <gtest/gtest.h>

 #include <cstdint>
 #include <filesystem>
 #include <sstream>
 #include <string>
 #include <vector>
 #include <fstream>
 #include <algorithm>

 namespace cuopt::mps_parser {

 constexpr double tolerance = 1e-6;

 // Enumeration for file format types
 enum class ProblemFileFormat {
   MPS,    // Linear Programming (MPS format)
   QPS,    // Quadratic Programming (QPS format)
   UNKNOWN // Cannot determine format
 };

 // Structure to hold file format analysis results
 struct FileFormatInfo {
   ProblemFileFormat format;
   bool has_quadratic_objective;
   std::string detected_extension;
   std::string problem_name;
 };

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

 /**
  * @brief Detect file format by analyzing file contents and extension
  *
  * This function determines whether a file is MPS or QPS format by:
  * 1. Checking the file extension (.mps vs .QPS/.qps)
  * 2. Scanning file contents for QPS-specific sections QUADOBJ
  * 3. Extracting problem name and quadratic programming features
  *
  * @param file Relative path to the file to analyze
  * @return FileFormatInfo structure with detected format and features
  */
 FileFormatInfo detect_file_format(const std::string& file)
 {
   FileFormatInfo info;
   info.format = ProblemFileFormat::UNKNOWN;
   info.has_quadratic_objective = false;

   // Get full file path
   const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
   std::string full_path = rapidsDatasetRootDir + "/" + file;

   // Extract file extension
   std::filesystem::path filepath(full_path);
   info.detected_extension = filepath.extension().string();

   if (!std::filesystem::exists(full_path)) {
     return info; // Return UNKNOWN for non-existent files
   }

   // Read and analyze file contents
   std::ifstream infile(full_path);
   if (!infile.is_open()) {
     return info; // Return UNKNOWN if cannot open file
   }

   std::string line;
   bool found_quadobj = false;

   while (std::getline(infile, line)) {
     // Trim whitespace
     line.erase(0, line.find_first_not_of(" \t\r\n"));
     line.erase(line.find_last_not_of(" \t\r\n") + 1);

     // Skip empty lines and comments
     if (line.empty() || line[0] == '*') continue;

     // Convert to uppercase for case-insensitive comparison
     std::string upper_line = line;
     std::transform(upper_line.begin(), upper_line.end(), upper_line.begin(), ::toupper);

     // Extract problem name
     if (upper_line.find("NAME") == 0 && info.problem_name.empty()) {
       std::istringstream iss(line);
       std::string keyword, name;
       iss >> keyword >> name;
       info.problem_name = name;
     }

     // Check for QUADOBJ section
     if (upper_line.find("QUADOBJ") == 0) {
       found_quadobj = true;
       info.has_quadratic_objective = true;
     }



     // Stop at ENDATA
     if (upper_line.find("ENDATA") == 0) break;
   }

   infile.close();

   // Determine format based on findings
   if (found_quadobj) {
     info.format = ProblemFileFormat::QPS;
   } else {
     // Also check extension as secondary indicator
     std::string lower_ext = info.detected_extension;
     std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);

     if (lower_ext == ".qps") {
       info.format = ProblemFileFormat::QPS; // QPS extension but no quadratic terms found
     } else if (lower_ext == ".mps") {
       info.format = ProblemFileFormat::MPS;
     } else {
       // Assume MPS if no quadratic sections found
       info.format = ProblemFileFormat::MPS;
     }
   }

   return info;
 }

 /**
  * @brief Helper function to print format information
  */
 std::string format_to_string(ProblemFileFormat format) {
   switch (format) {
     case ProblemFileFormat::MPS: return "MPS (Linear Programming)";
     case ProblemFileFormat::QPS: return "QPS (Quadratic Programming)";
     case ProblemFileFormat::UNKNOWN: return "UNKNOWN";
     default: return "INVALID";
   }
 }

 /**
  * @brief Enhanced file reading function that also validates format
  */
 mps_parser_t<int, double> read_from_mps_with_format_validation(const std::string& file,
                                                               bool fixed_format = true,
                                                               ProblemFileFormat expected_format = ProblemFileFormat::UNKNOWN)
 {
   // First detect the format
   auto format_info = detect_file_format(file);

   // If expected format is specified, validate it
   if (expected_format != ProblemFileFormat::UNKNOWN) {
     if (format_info.format != expected_format) {
       throw std::logic_error("Format mismatch: expected " + format_to_string(expected_format) +
                             " but detected " + format_to_string(format_info.format) +
                             " for file: " + file);
     }
   }

   // Proceed with normal parsing
   std::string rel_file{};
   const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
   rel_file = rapidsDatasetRootDir + "/" + file;
   mps_data_model_t<int, double> problem;
   mps_parser_t<int, double> mps{problem, rel_file, fixed_format};

   return mps;
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

 // QPS-specific tests for quadratic programming support
 TEST(qps_parser, quadratic_objective_basic)
 {
   // Create a simple QPS test to verify quadratic objective parsing
   // This would require actual QPS test files - for now, test the API
   mps_data_model_t<int, double> model;

   // Test setting quadratic objective matrix
   std::vector<double> Q_values = {2.0, 1.0, 1.0, 2.0};  // 2x2 matrix
   std::vector<int> Q_indices = {0, 1, 0, 1};
   std::vector<int> Q_offsets = {0, 2, 4};  // CSR offsets

   model.set_quadratic_objective_matrix(Q_values.data(), Q_values.size(),
                                       Q_indices.data(), Q_indices.size(),
                                       Q_offsets.data(), Q_offsets.size());

   // Verify the data was stored correctly
   EXPECT_TRUE(model.has_quadratic_objective());
   EXPECT_EQ(4, model.get_quadratic_objective_values().size());
   EXPECT_EQ(2.0, model.get_quadratic_objective_values()[0]);
   EXPECT_EQ(1.0, model.get_quadratic_objective_values()[1]);
 }



 TEST(qps_parser, data_model_view_quadratic_support)
 {
   // Test data_model_view_t with quadratic data
   data_model_view_t<int, double> view;

   std::vector<double> Q_values = {2.0, 1.0};
   std::vector<int> Q_indices = {0, 1};
   std::vector<int> Q_offsets = {0, 1, 2};

   view.set_quadratic_objective_matrix(Q_values.data(), Q_values.size(),
                                      Q_indices.data(), Q_indices.size(),
                                      Q_offsets.data(), Q_offsets.size());

   EXPECT_TRUE(view.has_quadratic_objective());
   EXPECT_EQ(2, view.get_quadratic_objective_values().size());
   EXPECT_EQ(2.0, view.get_quadratic_objective_values().data()[0]);
   EXPECT_EQ(1.0, view.get_quadratic_objective_values().data()[1]);
 }

 // ================================================================================================
 // FORMAT DETECTION TESTS
 // ================================================================================================

 // Removed test for afiro.mps as it doesn't exist in the dataset

 // Removed test for various MPS files as they don't exist in the dataset (afiro.mps, adlittle.mps, maros.mps, testprob.mps)

 // Removed test for test_quadratic.qps as it doesn't exist in the dataset

 TEST(format_detection, detect_qps_format_by_extension)
 {
   // Test detection based on QPS extension even without quadratic content
   // This tests the fallback logic when extension suggests QPS but no quadratic sections found
   std::string test_file = "some_file.qps";

   // Create a temporary minimal file for this test
   std::string full_path = cuopt::test::get_rapids_dataset_root_dir() + "/temp_test.qps";
   std::ofstream temp_file(full_path);
   if (temp_file.is_open()) {
     temp_file << "NAME TEST\n";
     temp_file << "ROWS\n";
     temp_file << " N OBJ\n";
     temp_file << "COLUMNS\n";
     temp_file << "RHS\n";
     temp_file << "ENDATA\n";
     temp_file.close();

     auto info = detect_file_format("temp_test.qps");

     EXPECT_EQ(ProblemFileFormat::QPS, info.format);
     EXPECT_FALSE(info.has_quadratic_objective);
     EXPECT_EQ("TEST", info.problem_name);

     // Clean up
     std::filesystem::remove(full_path);
   }
 }

 TEST(format_detection, detect_unknown_format)
 {
   // Test detection of unknown format for non-existent files
   auto info = detect_file_format("non_existent_file.xyz");

   EXPECT_EQ(ProblemFileFormat::UNKNOWN, info.format);
   EXPECT_FALSE(info.has_quadratic_objective);
   EXPECT_TRUE(info.problem_name.empty());
 }

 TEST(format_detection, comprehensive_format_analysis)
 {
   // Test comprehensive analysis including quadratic features
   std::string full_path = cuopt::test::get_rapids_dataset_root_dir() + "/test_comprehensive.qps";
   std::ofstream temp_file(full_path);
   if (temp_file.is_open()) {
     temp_file << "NAME COMPREHENSIVE_TEST\n";
     temp_file << "ROWS\n";
     temp_file << " N OBJ\n";
     temp_file << " L CON1\n";
     temp_file << " L CON2\n";
     temp_file << "COLUMNS\n";
     temp_file << " X1 OBJ 1.0\n";
     temp_file << " X1 CON1 1.0\n";
     temp_file << " X2 OBJ 2.0\n";
     temp_file << "RHS\n";
     temp_file << " RHS1 CON1 5.0\n";
     temp_file << "QUADOBJ\n";
     temp_file << " X1 X1 2.0\n";
     temp_file << " X1 X2 1.0\n";
     temp_file << "ENDATA\n";
     temp_file.close();

     auto info = detect_file_format("test_comprehensive.qps");

     EXPECT_EQ(ProblemFileFormat::QPS, info.format);
     EXPECT_TRUE(info.has_quadratic_objective);
     EXPECT_EQ("COMPREHENSIVE_TEST", info.problem_name);
     EXPECT_EQ(".qps", info.detected_extension);

     // Clean up
     std::filesystem::remove(full_path);
   }
 }

 TEST(format_detection, edge_cases)
 {
   // Test edge cases in format detection

   // Case 1: Empty file
   std::string empty_file_path = cuopt::test::get_rapids_dataset_root_dir() + "/empty_test.mps";
   std::ofstream empty_file(empty_file_path);
   empty_file.close();

   auto info = detect_file_format("empty_test.mps");
   EXPECT_EQ(ProblemFileFormat::MPS, info.format); // Should default to MPS based on extension
   EXPECT_TRUE(info.problem_name.empty());

   std::filesystem::remove(empty_file_path);

   // Case 2: File with only comments
   std::string comment_file_path = cuopt::test::get_rapids_dataset_root_dir() + "/comment_test.mps";
   std::ofstream comment_file(comment_file_path);
   comment_file << "* This is a comment\n";
   comment_file << "* Another comment\n";
   comment_file << "NAME TEST_COMMENTS\n";
   comment_file << "ENDATA\n";
   comment_file.close();

   info = detect_file_format("comment_test.mps");
   EXPECT_EQ(ProblemFileFormat::MPS, info.format);
   EXPECT_EQ("TEST_COMMENTS", info.problem_name);

   std::filesystem::remove(comment_file_path);
 }

 // Removed test for enhanced MPS parsing validation as it references non-existent afiro.mps

 TEST(format_detection, real_world_file_analysis)
 {
   // Test format detection on existing test files

   // Test the internal QPS test file if available
   std::string test_qps_path = cuopt::test::get_rapids_dataset_root_dir() + "/test_real_qps.qps";
   std::ofstream test_file(test_qps_path);
   if (test_file.is_open()) {
     // Create a realistic QPS file
     test_file << "NAME          REAL_WORLD_QPS\n";
     test_file << "ROWS\n";
     test_file << " N  OBJ\n";
     test_file << " L  BUDGET\n";
     test_file << " L  CAPACITY\n";
     test_file << "COLUMNS\n";
     test_file << "    PRODUCT1  OBJ       -10.0\n";
     test_file << "    PRODUCT1  BUDGET    2.5\n";
     test_file << "    PRODUCT1  CAPACITY  1.2\n";
     test_file << "    PRODUCT2  OBJ       -15.0\n";
     test_file << "    PRODUCT2  BUDGET    3.0\n";
     test_file << "    PRODUCT2  CAPACITY  1.8\n";
     test_file << "RHS\n";
     test_file << "    LIMITS    BUDGET    100.0\n";
     test_file << "    LIMITS    CAPACITY  50.0\n";
     test_file << "BOUNDS\n";
     test_file << " UP BOUNDS    PRODUCT1  20.0\n";
     test_file << " UP BOUNDS    PRODUCT2  15.0\n";
     test_file << "QUADOBJ\n";
     test_file << "    PRODUCT1  PRODUCT1  0.1\n";
     test_file << "    PRODUCT1  PRODUCT2  0.05\n";
     test_file << "    PRODUCT2  PRODUCT2  0.2\n";
     test_file << "ENDATA\n";
     test_file.close();

     auto info = detect_file_format("test_real_qps.qps");

     EXPECT_EQ(ProblemFileFormat::QPS, info.format);
     EXPECT_TRUE(info.has_quadratic_objective);
     EXPECT_EQ("REAL_WORLD_QPS", info.problem_name);
     EXPECT_EQ(".qps", info.detected_extension);

     // Test that parser can actually parse this file
     EXPECT_NO_THROW({
       auto mps = read_from_mps_with_format_validation("test_real_qps.qps",
                                                      false, ProblemFileFormat::QPS);
       EXPECT_EQ("REAL_WORLD_QPS", mps.problem_name);
     });

     // Clean up
     std::filesystem::remove(test_qps_path);
   }
 }

 // Removed format consistency check test as it references non-existent files (afiro.mps, adlittle.mps)

 // ================================================================================================
 // QPS FILES SAMPLING TESTS - Tests on actual QPS files from attached directory
 // ================================================================================================

 /**
  * @brief Helper function to get QPS file path relative to the test data root
  */
 std::string get_qps_file_path(const std::string& filename) {
     const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
     return rapidsDatasetRootDir + "/quadratic_programming/" + filename;
 }

 /**
  * @brief Structure to hold expected QPS file properties for validation
  */
 struct QpsFileExpectation {
     std::string filename;
     std::string expected_problem_name;
     bool should_have_quadratic_objective;
     int min_variables;      // Minimum expected variables (-1 for no check)
     int min_constraints;    // Minimum expected constraints (-1 for no check)
     std::string description; // Description for test output
 };

 TEST(qps_parser, test_qps_files)
{
    // Test the actual QPS test files that exist in the dataset
    std::vector<QpsFileExpectation> test_qps_files = {
        {"QP_Test_1.qps", "QP_Test_1", true, 2, 1, "QP_Test_1: Test quadratic programming problem 1"},
        {"QP_Test_2.qps", "QP_Test_2", true, 3, 1, "QP_Test_2: Test quadratic programming problem 2"}
    };

    for (const auto& expectation : test_qps_files) {
        std::string qps_path = get_qps_file_path(expectation.filename);

        // Skip if file doesn't exist (graceful handling)
        if (!std::filesystem::exists(qps_path)) {
            GTEST_SKIP() << "QPS file not found: " << qps_path;
        }

        SCOPED_TRACE("Testing " + expectation.description);

        // Test our format detection
        auto info = detect_file_format("quadratic_programming/" + expectation.filename);

        EXPECT_EQ(ProblemFileFormat::QPS, info.format)
            << "Should detect QPS format for: " << expectation.filename;
        EXPECT_EQ(expectation.expected_problem_name, info.problem_name)
            << "Problem name mismatch for: " << expectation.filename;
        EXPECT_EQ(".qps", info.detected_extension)
            << "Extension should be .qps for: " << expectation.filename;

        // Test actual parsing
        decltype(auto) parsed_data = cuopt::mps_parser::parse_mps<int, double>(qps_path, false);

        // Verify problem properties
        EXPECT_EQ(expectation.expected_problem_name, parsed_data.get_problem_name())
            << "Parsed problem name mismatch for: " << expectation.filename;

        // Check variable and constraint counts
        if (expectation.min_variables > 0) {
            EXPECT_GE(parsed_data.get_n_variables(), expectation.min_variables)
                << "Variable count too low for: " << expectation.filename;
        }
        if (expectation.min_constraints > 0) {
            EXPECT_GE(parsed_data.get_n_constraints(), expectation.min_constraints)
                << "Constraint count too low for: " << expectation.filename;
        }

        // Check quadratic features
        EXPECT_EQ(expectation.should_have_quadratic_objective, parsed_data.has_quadratic_objective())
            << "Quadratic objective expectation failed for: " << expectation.filename;

        // Verify format detection matches parser results
        EXPECT_EQ(info.has_quadratic_objective, parsed_data.has_quadratic_objective())
            << "Format detection vs parser mismatch (objective) for: " << expectation.filename;
    }
}

TEST(qps_parser, qps_test_1_detailed)
{
    // Detailed test for QP_Test_1.qps
    std::string qps_path = get_qps_file_path("QP_Test_1.qps");

    if (!std::filesystem::exists(qps_path)) {
        GTEST_SKIP() << "QP_Test_1.qps not found";
    }

    // Test format detection
    auto info = detect_file_format("quadratic_programming/QP_Test_1.qps");
    EXPECT_EQ(ProblemFileFormat::QPS, info.format);
    EXPECT_EQ("QP_Test_1", info.problem_name);
    EXPECT_TRUE(info.has_quadratic_objective);

    // Test parsing
    auto parsed_data = cuopt::mps_parser::parse_mps<int, double>(qps_path, false);

    // Verify problem structure based on the file content
    EXPECT_EQ("QP_Test_1", parsed_data.get_problem_name());
    EXPECT_EQ(2, parsed_data.get_n_variables());  // C------1 and C------2
    EXPECT_EQ(1, parsed_data.get_n_constraints()); // R------1

    // Verify quadratic objective exists
    EXPECT_TRUE(parsed_data.has_quadratic_objective());

    // Check variable bounds (from BOUNDS section)
    const auto& lower_bounds = parsed_data.get_variable_lower_bounds();
    const auto& upper_bounds = parsed_data.get_variable_upper_bounds();

    EXPECT_NEAR(2.0, lower_bounds[0], tolerance);   // C------1 lower bound
    EXPECT_NEAR(50.0, upper_bounds[0], tolerance);  // C------1 upper bound
    EXPECT_NEAR(-50.0, lower_bounds[1], tolerance); // C------2 lower bound
    EXPECT_NEAR(50.0, upper_bounds[1], tolerance);  // C------2 upper bound
}

TEST(qps_parser, qps_test_2_detailed)
{
    // Detailed test for QP_Test_2.qps
    std::string qps_path = get_qps_file_path("QP_Test_2.qps");

    if (!std::filesystem::exists(qps_path)) {
        GTEST_SKIP() << "QP_Test_2.qps not found";
    }

    // Test format detection
    auto info = detect_file_format("quadratic_programming/QP_Test_2.qps");
    EXPECT_EQ(ProblemFileFormat::QPS, info.format);
    EXPECT_EQ("QP_Test_2", info.problem_name);
    EXPECT_TRUE(info.has_quadratic_objective);

    // Test parsing
    auto parsed_data = cuopt::mps_parser::parse_mps<int, double>(qps_path, false);

    // Verify problem structure based on the file content
    EXPECT_EQ("QP_Test_2", parsed_data.get_problem_name());
    EXPECT_EQ(3, parsed_data.get_n_variables());  // C------1, C------2, C------3
    EXPECT_EQ(1, parsed_data.get_n_constraints()); // R------1

    // Verify quadratic objective exists
    EXPECT_TRUE(parsed_data.has_quadratic_objective());

    // Check that quadratic objective matrix has values
    const auto& Q_values = parsed_data.get_quadratic_objective_values();
    EXPECT_GT(Q_values.size(), 0) << "Quadratic objective should have non-zero elements";
}

 }  // namespace cuopt::mps_parser
