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

#include <mps_parser.hpp>

#include <utilities/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace cuopt::mps_parser {

template <typename i_t>
std::string_view get_next_string(std::string_view line, i_t& pos, i_t& end)
{
  pos = line.find_first_not_of(" \t", end);
  if (pos == std::string_view::npos) return "";
  end = line.find_first_of(" \t\n\r", pos);
  if (end == std::string_view::npos) return line.substr(pos);
  return line.substr(pos, end - pos);
}

std::string_view trim(std::string_view str)
{
  auto start = str.find_first_not_of(" \r\t");
  if (start == std::string::npos) { return ""; }
  auto end = str.find_last_not_of(" \r\t");
  return str.substr(start, end - start + 1);
}

BoundType convert(std::string_view str)
{
  if (str == "LO") {
    return LowerBound;
  } else if (str == "UP") {
    return UpperBound;
  } else if (str == "FX") {
    return Fixed;
  } else if (str == "FR") {
    return Free;
  } else if (str == "MI") {
    return LowerBoundNegInf;
  } else if (str == "PL") {
    return UpperBoundInf;
  } else if (str == "BV") {
    return BinaryVariable;
  } else if (str == "LI") {
    return LowerBoundIntegerVariable;
  } else if (str == "UI") {
    return UpperBoundIntegerVariable;
  } else if (str == "LC") {
    return SemiContiniousVariable;
  } else {
    mps_parser_expects(false,
                       error_type_t::ValidationError,
                       "Invalid variable bound type found in BOUNDS section! Bound type=%s",
                       std::string(str).c_str());
    return SemiContiniousVariable;
  }
}

ObjSenseType convert_to_obj_sense(const std::string& str)
{
  if (str == "MIN" || str == "MINIMIZE") {
    return Minimize;
  } else if (str == "MAX" || str == "MAXIMIZE") {
    return Maximize;
  } else {
    mps_parser_expects(false,
                       error_type_t::ValidationError,
                       "Invalid variable bound type found in OBJSENSE section! Objsense type=%s",
                       str.c_str());
    return Minimize;
  }
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::fill_problem(mps_data_model_t<i_t, f_t>& problem)
{
  {
    std::vector<i_t> h_offsets{}, h_indices{};
    std::vector<f_t> h_values{};

    h_offsets.push_back(0);
    for (i_t i = 0; i < (i_t)A_indices.size(); ++i) {
      i_t off = h_offsets.size() > 0 ? h_offsets[h_offsets.size() - 1] : 0;
      for (const auto& idx_itr : A_indices[i]) {
        h_indices.push_back(idx_itr);
      }
      for (const auto& val_itr : A_values[i]) {
        h_values.push_back(val_itr);
      }
      off += A_indices[i].size();
      h_offsets.push_back(off);
    }

    problem.set_csr_constraint_matrix(h_values.data(),
                                      h_values.size(),
                                      h_indices.data(),
                                      h_indices.size(),
                                      h_offsets.data(),
                                      h_offsets.size());

    mps_parser_expects(A_indices.size() + 1 == h_offsets.size(),
                       error_type_t::ValidationError,
                       "The row indexing vector for the constraint matrix was not constructed "
                       "successfully. Should be size %zu, but was size %zu",
                       A_indices.size() + 1,
                       h_offsets.size());
    mps_parser_expects(
      h_indices.size() == h_values.size(),
      error_type_t::ValidationError,
      "The nonzero value vector or the column indexing vector for the constraint "
      "matrix was not constructed "
      "successfully. Should be the same size but nonzeroes were of size %zu and column "
      "indexing vector of size %zu ",
      h_indices.size(),
      h_values.size());
    mps_parser_expects(
      h_offsets[h_offsets.size() - 1] == (i_t)h_values.size(),
      error_type_t::ValidationError,
      "The last row offset for the constraint matrix is not equal to the size of the "
      "nonzero vector. Nonzero has size %zu but the last offset is %d.",
      h_values.size(),
      h_offsets[h_offsets.size() - 1]);
  }

  // Set b & c
  problem.set_constraint_bounds(b_values.data(), b_values.size());
  problem.set_objective_coefficients(c_values.data(), c_values.size());

  // Set offset and scaling factor of objective function
  problem.set_objective_scaling_factor(objective_scaling_factor_value);
  problem.set_objective_offset(objective_offset_value);

  // Set lower and upper bounds
  problem.set_variable_lower_bounds(variable_lower_bounds.data(), variable_lower_bounds.size());
  problem.set_variable_upper_bounds(variable_upper_bounds.data(), variable_upper_bounds.size());

  mps_parser_expects(
    (problem.get_variable_lower_bounds().size() == problem.get_variable_upper_bounds().size()) &&
      (problem.get_variable_upper_bounds().size() == problem.get_objective_coefficients().size()),
    error_type_t::ValidationError,
    "Sizes for vectors related to the variables are not the same. The objective "
    "vector has size %zu, the variable lower bounds vector has size %zu and the "
    "variable upper bounds vector has size %zu.",
    problem.get_objective_coefficients().size(),
    problem.get_variable_lower_bounds().size(),
    problem.get_variable_upper_bounds().size());

  // Determine the constraint bounds based on row types
  {
    std::vector<f_t> h_constraint_lower_bounds{};
    std::vector<f_t> h_constraint_upper_bounds{};
    for (i_t i = 0; i < (i_t)row_types.size(); ++i) {
      if (row_types[i] == Equality) {
        h_constraint_lower_bounds.push_back(b_values[i]);
        h_constraint_upper_bounds.push_back(b_values[i]);
        if (ranges_values.size() > 0 &&
            ranges_values[i] != unset_range_value)  // Add range value if specified
        {
          mps_parser_expects(!std::isnan(h_constraint_lower_bounds[i]),
                             error_type_t::ValidationError,
                             "Constraints lower bound %d shouldn't be nan",
                             i);
          mps_parser_expects(!std::isnan(h_constraint_upper_bounds[i]),
                             error_type_t::ValidationError,
                             "Constraints upper bound %d shouldn't be nan",
                             i);
          mps_parser_expects(!std::isnan(ranges_values[i]),
                             error_type_t::ValidationError,
                             "Equality range value %d shouldn't be nan",
                             i);
          if (ranges_values[i] < f_t(0))
            h_constraint_lower_bounds[i] = h_constraint_lower_bounds[i] + ranges_values[i];
          else  // Positive
            h_constraint_upper_bounds[i] = h_constraint_upper_bounds[i] + ranges_values[i];
        }
      } else if (row_types[i] == GreaterThanOrEqual) {
        h_constraint_lower_bounds.push_back(b_values[i]);
        h_constraint_upper_bounds.push_back(std::numeric_limits<f_t>::infinity());
        if (ranges_values.size() > 0 &&
            ranges_values[i] != unset_range_value)  // Add range value if specified
        {
          mps_parser_expects(!std::isnan(h_constraint_lower_bounds[i]),
                             error_type_t::ValidationError,
                             "Constraints lower bound %d shouldn't be nan",
                             i);
          mps_parser_expects(!std::isnan(ranges_values[i]),
                             error_type_t::ValidationError,
                             "Greater range value %d shouldn't be nan",
                             i);
          h_constraint_upper_bounds[i] = h_constraint_lower_bounds[i] + std::abs(ranges_values[i]);
        }
      } else if (row_types[i] == LesserThanOrEqual) {
        h_constraint_lower_bounds.push_back(-std::numeric_limits<f_t>::infinity());
        h_constraint_upper_bounds.push_back(b_values[i]);
        if (ranges_values.size() > 0 &&
            ranges_values[i] != unset_range_value)  // Add range value if specified
        {
          mps_parser_expects(!std::isnan(h_constraint_upper_bounds[i]),
                             error_type_t::ValidationError,
                             "Constraints upper bound %d shouldn't be nan",
                             i);
          mps_parser_expects(!std::isnan(ranges_values[i]),
                             error_type_t::ValidationError,
                             "Lesser range value %d shouldn't be nan",
                             i);
          h_constraint_lower_bounds[i] = h_constraint_upper_bounds[i] - std::abs(ranges_values[i]);
        }
      } else {
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "Unsupported row type was passed to the Optimization Problem");
      }
      mps_parser_expects(
        !std::isnan(h_constraint_lower_bounds[i]), error_type_t::ValidationError, "Cannot be nan");
      mps_parser_expects(
        !std::isnan(h_constraint_upper_bounds[i]), error_type_t::ValidationError, "Cannot be nan");
    }

    problem.set_constraint_lower_bounds(h_constraint_lower_bounds.data(),
                                        h_constraint_lower_bounds.size());
    problem.set_constraint_upper_bounds(h_constraint_upper_bounds.data(),
                                        h_constraint_upper_bounds.size());

    mps_parser_expects(
      (problem.get_constraint_lower_bounds().size() ==
       problem.get_constraint_upper_bounds().size()) &&
        (problem.get_constraint_upper_bounds().size() == problem.get_constraint_bounds().size()),
      error_type_t::ValidationError,
      "Sizes for vectors related to the constraints are not the same. The right hand side "
      "vector has size %zu, the constraint lower bounds vector has size %zu and the "
      "constraint upper bounds vector has size %zu.",
      problem.get_constraint_bounds().size(),
      problem.get_constraint_lower_bounds().size(),
      problem.get_constraint_upper_bounds().size());
  }

  problem.set_problem_name(problem_name);
  problem.set_objective_name(objective_name);
  problem.set_variable_names(std::move(var_names));
  problem.set_variable_types(std::move(var_types));
  problem.set_row_names(std::move(row_names));
  problem.set_maximize(maximize);
}

template <typename i_t, typename f_t>
std::vector<char> mps_parser_t<i_t, f_t>::file_to_string(const std::string& file)
{
  // raft::common::nvtx::range fun_scope("file to string");

  // Faster than using C++ I/O
  FILE* fp = fopen(file.c_str(), "r");
  mps_parser_expects(fp != nullptr,
                     error_type_t::ValidationError,
                     "Error opening MPS file! Given path: %s",
                     mps_file.c_str());

  mps_parser_expects(fseek(fp, 0L, SEEK_END) == 0,
                     error_type_t::ValidationError,
                     "File browsing MPS file! Given path: %s",
                     mps_file.c_str());
  const long bufsize = ftell(fp);
  mps_parser_expects(bufsize != -1L,
                     error_type_t::ValidationError,
                     "File browsing MPS file! Given path: %s",
                     mps_file.c_str());
  std::vector<char> buf(bufsize + 1);
  rewind(fp);

  mps_parser_expects(fread(buf.data(), sizeof(char), bufsize, fp) == bufsize,
                     error_type_t::ValidationError,
                     "Error reading MPS file! Given path: %s",
                     mps_file.c_str());
  buf[bufsize] = '\0';

  mps_parser_expects(fclose(fp) == 0,
                     error_type_t::ValidationError,
                     "Error closing MPS file! Given path: %s",
                     mps_file.c_str());

  return buf;
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_string(char* buf)
{
  // raft::common::nvtx::range fun_scope("parse string");

  // Faster than C++ std::get_line
  char* c_line   = strtok(buf, "\n");
  bool skip_line = false;

  mps_parser_expects(c_line != nullptr,
                     error_type_t::ValidationError,
                     "Error parsing MPS file! No line return found (\"\\n\")");

  do {
    std::string_view line(c_line);
    // ignore empty lines and comments
    if (line.empty() || line[0] == '*' || line[0] == '$' || line[0] == '\n' || line[0] == '\r') {
      continue;
    }
    // these lines mark the start of a particular "section"
    if (line[0] != ' ') {
      skip_line = false;
      if (line.find("NAME", 0, 4) == 0) {
        encountered_sections.insert("NAME");
        auto name_start = line.find_first_not_of(" \t", 4);
        if (name_start != std::string::npos) {
          // max of 8 chars allowed
          if (fixed_mps_format) {
            problem_name = std::string(trim(line.substr(name_start, 8)));
          } else {
            std::stringstream ss{std::string(line)};
            ss.seekg(name_start);

            ss >> problem_name;
          }
        }
      } else if (line.find("ROWS", 0, 4) == 0) {
        encountered_sections.insert("ROWS");
        inside_rows_     = true;
        inside_columns_  = false;
        inside_rhs_      = false;
        inside_bounds_   = false;
        inside_objsense_ = false;
        inside_ranges_   = false;
        inside_objname_  = false;
      } else if (line.find("COLUMNS", 0, 7) == 0) {
        encountered_sections.insert("COLUMNS");
        inside_rows_     = false;
        inside_columns_  = true;
        inside_rhs_      = false;
        inside_bounds_   = false;
        inside_objsense_ = false;
        inside_ranges_   = false;
        inside_objname_  = false;
        A_indices.resize(row_names.size());
        A_values.resize(row_names.size());
        b_values.resize(row_names.size());
        // Needed if not all rows are mentioned in RHS
        std::fill(b_values.begin(), b_values.end(), f_t(0));
      } else if (line.find("RHS", 0, 3) == 0) {
        encountered_sections.insert("RHS");
        inside_rows_     = false;
        inside_columns_  = false;
        inside_rhs_      = true;
        inside_bounds_   = false;
        inside_objsense_ = false;
        inside_ranges_   = false;
        inside_objname_  = false;
      } else if (line.find("BOUNDS", 0, 6) == 0) {
        encountered_sections.insert("BOUNDS");
        inside_rows_     = false;
        inside_columns_  = false;
        inside_rhs_      = false;
        inside_bounds_   = true;
        inside_objsense_ = false;
        inside_ranges_   = false;
        inside_objname_  = false;
        variable_lower_bounds.resize(var_names.size());
        variable_upper_bounds.resize(var_names.size());
        std::fill(variable_lower_bounds.begin(), variable_lower_bounds.end(), f_t(0));
        std::fill(variable_upper_bounds.begin(),
                  variable_upper_bounds.end(),
                  +std::numeric_limits<f_t>::infinity());
      } else if (line.find("RANGES", 0, 6) == 0) {
        encountered_sections.insert("RANGES");
        inside_rows_     = false;
        inside_columns_  = false;
        inside_rhs_      = false;
        inside_bounds_   = false;
        inside_objsense_ = false;
        inside_ranges_   = true;
        inside_objname_  = false;
        ranges_values.resize(row_types.size());
        std::fill(ranges_values.begin(), ranges_values.end(), unset_range_value);
      } else if (line.find("OBJSENSE", 0, 8) == 0) {
        // Optimization direction is on same line
        if (!std::none_of(line.begin() + 8, line.end(), ::isalpha)) {
          parse_objsense(line);
          continue;
        }
        encountered_sections.insert("OBJSENSE");
        inside_rows_     = false;
        inside_columns_  = false;
        inside_rhs_      = false;
        inside_bounds_   = false;
        inside_ranges_   = false;
        inside_objname_  = false;
        inside_objsense_ = true;
      } else if (line.find("OBJNAME", 0, 7) == 0) {
        encountered_sections.insert("OBJNAME");
        // Objective name is on same line
        if (!std::none_of(line.begin() + 7, line.end(), ::isalpha)) {
          parse_objname(line);
          continue;
        }
        inside_rows_     = false;
        inside_columns_  = false;
        inside_rhs_      = false;
        inside_bounds_   = false;
        inside_ranges_   = false;
        inside_objname_  = true;
        inside_objsense_ = false;
      } else if (line.find("ENDATA", 0, 6) == 0) {
        encountered_sections.insert("ENDATA");
        break;
      }
      // treating lazy constraints as normal constraints
      else if (line.find("LAZYCONS", 0, 8) == 0) {
        encountered_sections.insert("LAZYCONS");
        inside_rows_     = true;
        inside_columns_  = false;
        inside_rhs_      = false;
        inside_bounds_   = false;
        inside_objsense_ = false;
        inside_ranges_   = false;
        inside_objname_  = false;
      } else {
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "Invalid named block found! Line=%s",
                           std::string(line).c_str());
      }
    } else if (skip_line) {
      continue;
    } else if (inside_rows_) {
      parse_rows(line);
    } else if (inside_columns_) {
      parse_columns(line);
    } else if (inside_rhs_) {
      parse_rhs(line);
    } else if (inside_bounds_) {
      parse_bounds(line);
    } else if (inside_ranges_) {
      parse_ranges(line);
    } else if (inside_objsense_) {
      parse_objsense(line);
    } else if (inside_objname_) {
      parse_objname(line);
    } else {
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "Ended up at a bad parser state! Line=%s",
                         std::string(line).c_str());
    }
  } while ((c_line = strtok(nullptr, "\n")) != nullptr);
  mps_parser_expects(!objective_name.empty(), error_type_t::ValidationError, "No objective found!");

  mps_parser_expects(
    encountered_sections.count("ROWS"), error_type_t::ValidationError, "ROWS section is missing");
  mps_parser_expects(encountered_sections.count("COLUMNS"),
                     error_type_t::ValidationError,
                     "COLUMNS section is missing");
  mps_parser_expects(
    encountered_sections.count("RHS"), error_type_t::ValidationError, "RHS section is missing");

  // Those sections are mandatory according to the MPS format specification, however some test cases
  // rely on their absence Emit a warning in this case
  if (!encountered_sections.count("NAME")) { printf("NAME section is missing"); }
  if (!encountered_sections.count("ENDATA")) { printf("ENDATA section is missing"); }

  if (variable_upper_bounds.size() == 0)  // No variables bounds given, add the default values
  {
    variable_lower_bounds.resize(var_names.size());
    variable_upper_bounds.resize(var_names.size());
    std::fill(variable_lower_bounds.begin(), variable_lower_bounds.end(), f_t(0));
    std::fill(variable_upper_bounds.begin(),
              variable_upper_bounds.end(),
              +std::numeric_limits<f_t>::infinity());
  }
  mps_parser_expects(variable_lower_bounds.size() == variable_upper_bounds.size() &&
                       variable_upper_bounds.size() == var_names.size(),
                     error_type_t::ValidationError,
                     "MPS Parser Internal Error - Please contact cuOpt team");

  // Set all integer variables with bounds unspecified to [0, 1]
  // Also bounds sanity check
  for (i_t i = 0; i < var_names.size(); ++i) {
    if (!bounds_defined_for_var_id.count(i) && var_types[i] == 'I') {
      variable_lower_bounds[i] = 0;
      variable_upper_bounds[i] = 1;
    }
    mps_parser_expects(variable_lower_bounds[i] <= variable_upper_bounds[i],
                       error_type_t::ValidationError,
                       "MPS Parser Internal Error - Please contact cuOpt team");
  }
}

template <typename i_t, typename f_t>
mps_parser_t<i_t, f_t>::mps_parser_t(mps_data_model_t<i_t, f_t>& problem,
                                     const std::string& file,
                                     bool _fixed_mps_format)
  : mps_file{file}, fixed_mps_format(_fixed_mps_format)
{
  // raft::common::nvtx::range fun_scope("mps parser");

  std::vector<char> buf = file_to_string(file);

  parse_string(buf.data());

  fill_problem(problem);
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_rows(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse rows");

  RowType type;
  std::string name;

  if (fixed_mps_format) {
    type = static_cast<RowType>(line[1]);
    name = trim(line.substr(4, 8));  // max of 8 chars allowed
  } else {
    std::stringstream ss{std::string(line)};
    char read_word;
    ss >> read_word;
    type = static_cast<RowType>(read_word);

    ss >> name;
  }
  if (type == Objective) {
    // Keep only the first name or OBJNAME since it was set before
    if (objective_name.empty())
      objective_name = name;
    else
      ignored_objective_names.emplace(name);
    // If we wanted to stricly follow MPS definition: a new objective row ('N') should be treated
    // as an unbounded constraints, aka an extra contraints row with lower bound -infinity and upper
    // bound +infinity. Most solver ignore it to simplify the constraint matrix. We keep
    // it in record as ignored to not consider finding it in COLUMNS section as an error.
    return;
  }
  mps_parser_expects(row_names_map.find(name) == row_names_map.end(),
                     error_type_t::ValidationError,
                     "Duplicate row named '%s' found! line=%s",
                     name.c_str(),
                     std::string(line).c_str());
  auto n_rows = row_names.size();
  row_names.push_back(name);
  row_names_map.insert(std::make_pair(name, n_rows));
  row_types.push_back(type);
}

template <typename i_t, typename f_t>
i_t mps_parser_t<i_t, f_t>::parse_column_var_name(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse columns var name");

  std::string_view var_name;
  i_t pos;
  if (fixed_mps_format) {
    mps_parser_expects(line.size() >= 25,
                       error_type_t::ValidationError,
                       "COLUMNS should have atleast 3 entities! line=%s",
                       std::string(line).c_str());
    var_name = trim(line.substr(4, 8));  // max of 8 chars allowed

    pos = 14;
  } else {
    i_t end_var = 0;
    var_name    = get_next_string(line, pos, end_var);
    pos         = end_var;
  }
  if (line.find("\'MARKER\'") != std::string::npos) {
    if (line.find("INTORG") != std::string::npos) {
      mps_parser_expects(!inside_intcapture_,
                         error_type_t::ValidationError,
                         "Cannot capture an int section while already capturing an int section");
      inside_intcapture_ = true;
    }
    if (line.find("INTEND") != std::string::npos) {
      mps_parser_expects(inside_intcapture_,
                         error_type_t::ValidationError,
                         "Cannot stop int capture when a previous capture is not started");
      inside_intcapture_ = false;
    }
    return -1;
  }
  char var_type = inside_intcapture_ ? 'I' : 'C';
  if (!var_names.empty()) {
    const auto& last = var_names.back();
    if (last != var_name) {
      mps_parser_expects(var_names_map.find(std::string(var_name)) == var_names_map.end(),
                         error_type_t::ValidationError,
                         "All rows for the column (%s) should occur contiguously! line=%s",
                         std::string(var_name).c_str(),
                         std::string(line).c_str());
      var_names.emplace_back(var_name);
      var_types.emplace_back(var_type);
      var_names_map.insert(std::make_pair(std::string(var_name), var_names.size() - 1));
      c_values.emplace_back(f_t(0));
    }
  } else {
    var_names.emplace_back(var_name);
    var_types.emplace_back(var_type);
    var_names_map.insert(std::make_pair(var_name, var_names.size() - 1));
    c_values.emplace_back(f_t(0));
  }
  return pos;
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_column_row_and_value(std::string_view line, i_t pos)
{
  // raft::common::nvtx::range fun_scope("parse column row and value");

  auto var_id = var_names.size() - 1;

  if (fixed_mps_format) {
    pos = read_row_and_value(line, pos, var_id);
    if (pos == -1) return;
    pos = 39;
  } else {
    pos = read_row_and_value(line, pos, var_id);
    if (pos == -1) return;
  }

  if (line.find_last_not_of(" \r\t\n") > pos) { read_row_and_value(line, pos, var_id); }
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_columns(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse columns");

  i_t pos;
  if ((pos = parse_column_var_name(line)) == -1) return;

  parse_column_row_and_value(line, pos);
}

template <typename i_t, typename f_t>
std::tuple<std::string_view, std::string_view, i_t> mps_parser_t<i_t, f_t>::parse_row_name_and_num(
  std::string_view line, i_t start)
{
  // raft::common::nvtx::range fun_scope("parse_row_name_and_num");

  std::string_view row_name;
  std::string_view num;

  if (fixed_mps_format) {
    row_name = trim(line.substr(start, 8));  // max of 8 chars allowed
    num      = line.substr(start + 10, 12);  // max of 12 chars for numerical values
    if (row_name[0] == '$') return std::tuple("", "", -1);
  } else {
    i_t pos;
    row_name = get_next_string(line, pos, start);
    if (row_name[0] == '$') return std::tuple("", "", -1);
    num = get_next_string(line, pos, start);
  }

  return std::tuple(row_name, num, start);
}
template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::insert_row_name_and_value(std::string_view line,
                                                       std::string_view row_name,
                                                       std::string_view num,
                                                       i_t var_id)
{
  // raft::common::nvtx::range fun_scope("insert_row_name_and_value");
  static_assert(std::is_same_v<f_t, float> || std::is_same_v<f_t, double>,
                "f_t must be float or double");

  // Value for an ignored objective, can just skip it
  if (ignored_objective_names.find(std::string(row_name)) != ignored_objective_names.end()) return;

  f_t val;
  mps_parser_no_except(
    if constexpr (std::is_same_v<f_t, float>) {
      val = std::stof(std::string(num));
    } else if constexpr (std::is_same_v<f_t, double>) { val = std::stod(std::string(num)); },
    error_type_t::ValidationError,
    "Bad value found for row=%s in COLUMNS! line=%s. Num is %s",
    std::string(row_name).c_str(),
    std::string(line).c_str(),
    std::string(num).c_str());
  if (row_name == objective_name) {
    c_values[var_id] = val;
    return;
  }
  auto itr = row_names_map.find(std::string(row_name));
  mps_parser_expects(itr != row_names_map.end(),
                     error_type_t::ValidationError,
                     "Bad row name found '%s' in COLUMNS! line=%s",
                     std::string(row_name).c_str(),
                     std::string(line).c_str());
  auto row_id = itr->second;
  A_indices[row_id].emplace_back(var_id);
  A_values[row_id].emplace_back(val);
}

template <typename i_t, typename f_t>
i_t mps_parser_t<i_t, f_t>::read_row_and_value(std::string_view line, i_t start, i_t var_id)
{
  // raft::common::nvtx::range fun_scope("read_row_and_value");

  auto [row_name, num, end] = parse_row_name_and_num(line, start);
  if (row_name.empty()) return -1;

  insert_row_name_and_value(line, row_name, num, var_id);

  return end;
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_rhs(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse rhs");

  i_t pos = 0;
  if (fixed_mps_format) {
    mps_parser_expects(line.size() >= 25,
                       error_type_t::ValidationError,
                       "RHS should have atleast 3 entities! line=%s",
                       std::string(line).c_str());
    pos = 14;
    pos = read_rhs_row_and_value(line, pos);
    if (pos == -1) return;
    pos = 39;
  } else {
    // get the first field (which may or may not be the RHS name)
    i_t first_field_start = 0;
    auto first_field      = get_next_string(line, first_field_start, pos);
    if (first_field == objective_name || row_names_map.count(std::string(first_field))) {
      // first field corresponds to a row name, therefore we can assume that there is no RHS name
      // field. Reset pos.
      pos = 0;
    }
    pos = read_rhs_row_and_value(line, pos);
    if (pos == -1) return;
  }

  if (line.find_last_not_of(" \r\t\n") > pos) { read_rhs_row_and_value(line, pos); }
}

template <typename i_t, typename f_t>
i_t mps_parser_t<i_t, f_t>::read_rhs_row_and_value(std::string_view line, i_t start)
{
  static_assert(std::is_same_v<f_t, float> || std::is_same_v<f_t, double>,
                "f_t must be float or double");

  std::string_view row_name;
  std::string_view num;

  if (fixed_mps_format) {
    row_name = trim(line.substr(start, 8));  // max of 8 chars allowed
    if (row_name[0] == '$') return -1;
    num = line.substr(start + 10, 12);  // max of 12 chars for numerical values
  } else {
    i_t pos;
    row_name = get_next_string(line, pos, start);
    if (row_name[0] == '$') return -1;
    num = get_next_string(line, pos, start);
  }

  f_t val;
  mps_parser_no_except(
    if constexpr (std::is_same_v<f_t, float>) {
      val = std::stof(std::string(num));
    } else if constexpr (std::is_same_v<f_t, double>) { val = std::stod(std::string(num)); },
    error_type_t::ValidationError,
    "Bad value found for row=%s in RHS! line=%s",
    std::string(row_name).c_str(),
    std::string(line).c_str());
  if (row_name == objective_name) {
    // We treat minus the right hand side of OBJ as the objective offset, in
    // line with what the MPS writer does
    objective_offset_value = -val;
  } else {
    auto itr = row_names_map.find(std::string(row_name));
    mps_parser_expects(itr != row_names_map.end(),
                       error_type_t::ValidationError,
                       "Bad row name found '%s' in RHS! line=%s",
                       std::string(row_name).c_str(),
                       std::string(line).c_str());
    auto row_id      = itr->second;
    b_values[row_id] = val;
  }

  // Start is now pointing to the end of the val string
  return start;
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_bounds(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse bounds");

  BoundType type;
  std::string_view bound_name;
  std::string_view var_name;
  i_t pos;
  i_t end = 0;

  if (fixed_mps_format) {
    mps_parser_expects(line.size() >= 14,
                       error_type_t::ValidationError,
                       "BOUNDS should have atleast 2 entities! line=%s",
                       std::string(line).c_str());
    type       = static_cast<BoundType>(convert(line.substr(1, 2)));
    bound_name = trim(line.substr(4, 8));   // max of 8 chars allowed
    var_name   = trim(line.substr(14, 8));  // max of 8 chars allowed
    if (var_name[0] == '$') return;
    end = 24;
  } else {
    type = static_cast<BoundType>(convert(get_next_string(line, pos, end)));

    bound_name                = get_next_string(line, pos, end);
    i_t pos_after_first_field = pos;
    i_t end_after_first_field = end;
    var_name                  = get_next_string(line, pos, end);

    // If bound_name refers to an existing variable name, assume this row doesn't contain
    // a bound name. This is the case for some older MPS files following the SIF format.
    // c.f.
    // https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4dd23bcc5afe4c19a5d21c5be86e2aea2b426beb
    if (var_names_map.count(std::string(bound_name))) {
      var_name = bound_name;
      // go back to before the second field is read
      pos = pos_after_first_field;
      end = end_after_first_field;
    }

    if (var_name[0] == '$') return;
  }

  auto itr = var_names_map.find(std::string(var_name));
  // Define a var in bounds
  // Has no impact on objective function but is not an error in itself
  if (itr == var_names_map.end()) {
    var_names.emplace_back(var_name);
    var_names_map.insert(std::make_pair(std::string(var_name), var_names.size() - 1));
    c_values.emplace_back(f_t(0));
    variable_lower_bounds.emplace_back(0);
    variable_upper_bounds.emplace_back(+std::numeric_limits<f_t>::infinity());
    var_types.resize(var_types.size() + 1);
    itr = var_names_map.find(std::string(var_name));
  }
  i_t var_id = itr->second;

  read_bound_and_value(line, type, var_id, end);
  bounds_defined_for_var_id.insert(var_id);
}

template <typename i_t, typename f_t>
i_t mps_parser_t<i_t, f_t>::insert_range_value(std::string_view line, bool skip_range)
{
  std::string_view row_name;
  f_t value;
  i_t pos;
  i_t end;

  if (fixed_mps_format) {
    // RANGES name is ignored
    // range_name = trim(line.substr(4, 8)); // max of 8 chars allowed
    row_name = trim(line.substr(14, 8));  // max of 8 chars allowed
    if (row_name[0] == '$') return -1;
    end   = 24;  // Value position
    value = get_numerical_bound<true>(line, end);
    end   = 39;  // Start of potential next section
  } else {
    pos = line.find_first_not_of(" \t");                   // Skip initial space
    if (skip_range) pos = line.find_first_of(" \t", pos);  // Skip inital RANGES name
    end      = pos;
    row_name = get_next_string(line, pos, end);
    if (row_name[0] == '$') return -1;
    value = get_numerical_bound<true>(line, end);
  }

  auto itr = row_names_map.find(std::string(row_name));
  mps_parser_expects(itr != row_names_map.end(),
                     error_type_t::ValidationError,
                     "Bad row name found '%s' in RANGES! line=%s",
                     std::string(row_name).c_str(),
                     std::string(line).c_str());
  auto row_id           = itr->second;
  ranges_values[row_id] = value;

  return end;
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_ranges(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse ranges");

  constexpr i_t length_first_section = 25;

  if (fixed_mps_format) {
    mps_parser_expects(line.size() >= length_first_section,
                       error_type_t::ValidationError,
                       "RANGES should have atleast 2 entities! line=%s",
                       std::string(line).c_str());
  }

  i_t end = insert_range_value(line);
  if (end == -1) return;

  if (line.find_last_not_of(" \t") > end) {
    if (fixed_mps_format)
      insert_range_value(
        std::string_view(line.data() + length_first_section, line.size() - length_first_section));
    else  // false to not skip RANGE as it's only appears in first section
      insert_range_value(std::string_view(line.data() + end, line.size() - end), false);
  }
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_objsense(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse objsense");

  if (fixed_mps_format) {
    mps_parser_expects(
      false, error_type_t::ValidationError, "OBJSENSE only exist in Free MPS format");
  } else {
    std::stringstream ss{std::string(line)};
    std::string read_type;
    ObjSenseType type;
    // One-liner OBJSENSE
    if (line.find("OBJSENSE", 0, 8) == 0) ss >> read_type;
    ss >> read_type;
    type = static_cast<ObjSenseType>(convert_to_obj_sense(read_type));
    if (type == ObjSenseType::Minimize)
      maximize = false;
    else if (type == ObjSenseType::Maximize)
      maximize = true;
  }
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::parse_objname(std::string_view line)
{
  // raft::common::nvtx::range fun_scope("parse objname");

  if (fixed_mps_format) {
    mps_parser_expects(
      false, error_type_t::ValidationError, "OBJNAME only exist in Free MPS format");
  } else {
    std::stringstream ss{std::string(line)};
    std::string name;
    // One-liner OBJNAME, skip OBJNAME
    if (line.find("OBJNAME", 0, 7) == 0) ss >> name;
    ss >> name;
    if (!objective_name.empty()) {
      mps_parser_expects(
        false, error_type_t::ValidationError, "OBJNAME section should appear before ROWS section");
    }
    // Since OBJNAME always is before ROWS, this objective name will keep priority
    objective_name = name;
  }
}

template <typename i_t, typename f_t>
template <bool bounds_or_ranges, int fixed_length>
f_t mps_parser_t<i_t, f_t>::get_numerical_bound(std::string_view line, i_t& start)
{
  f_t val;
  std::string_view num;

  if (fixed_mps_format) {
    num = line.substr(start, fixed_length);
  } else {
    // Go to beginning of value
    i_t pos;
    num = get_next_string(line, pos, start);
  }
  if constexpr (bounds_or_ranges) {
    mps_parser_no_except(
      if constexpr (std::is_same_v<f_t, float>) {
        val = std::stof(std::string(num));
      } else if constexpr (std::is_same_v<f_t, double>) { val = std::stod(std::string(num)); },
      error_type_t::ValidationError,
      "Bad value found in RANGES! line=%s",
      std::string(line).c_str());
  } else {
    mps_parser_no_except(
      if constexpr (std::is_same_v<f_t, float>) {
        val = std::stof(std::string(num));
      } else if constexpr (std::is_same_v<f_t, double>) { val = std::stod(std::string(num)); },
      error_type_t::ValidationError,
      "Bad value found in BOUNDS! line=%s",
      std::string(line).c_str());
  }
  return val;
}

template <typename i_t, typename f_t>
void mps_parser_t<i_t, f_t>::read_bound_and_value(std::string_view line,
                                                  BoundType bound_type,
                                                  i_t var_id,
                                                  i_t start)
{
  switch (bound_type) {
    case LowerBound: {
      variable_lower_bounds[var_id] = get_numerical_bound(line, start);
      break;
    }
    case UpperBound: {
      variable_upper_bounds[var_id] = get_numerical_bound(line, start);
      // From CPLEX MPS reference:
      // > If an upper bound of less than 0 is specified and no
      // > other bound is specified, the lower bound is automatically set to -∞
      if (!bounds_defined_for_var_id.count(var_id) && variable_upper_bounds[var_id] < f_t(0)) {
        variable_lower_bounds[var_id] = -std::numeric_limits<f_t>::infinity();
      }
      break;
    }
    case Fixed: {
      const f_t val                 = get_numerical_bound(line, start);
      variable_lower_bounds[var_id] = val;
      variable_upper_bounds[var_id] = val;
      break;
    }
    case Free: {
      variable_lower_bounds[var_id] = -std::numeric_limits<f_t>::infinity();
      variable_upper_bounds[var_id] = +std::numeric_limits<f_t>::infinity();
      break;
    }
    case LowerBoundNegInf:
      variable_lower_bounds[var_id] = -std::numeric_limits<f_t>::infinity();
      break;
    case UpperBoundInf:
      variable_upper_bounds[var_id] = +std::numeric_limits<f_t>::infinity();
      return;
    case BinaryVariable:
      variable_lower_bounds[var_id] = 0;
      variable_upper_bounds[var_id] = 1;
      var_types[var_id]             = 'I';
      break;
    case LowerBoundIntegerVariable:
      // CPLEX MPS file references seems to imply that integer variables default to an upper bound
      // of +inf when only the lower bound is defined
      if (!bounds_defined_for_var_id.count(var_id)) {
        variable_upper_bounds[var_id] = +std::numeric_limits<f_t>::infinity();
      }
      variable_lower_bounds[var_id] = get_numerical_bound(line, start);
      var_types[var_id]             = 'I';
      break;
    case UpperBoundIntegerVariable:
      variable_upper_bounds[var_id] = get_numerical_bound(line, start);
      // From CPLEX MPS reference:
      // > If an upper bound of less than 0 is specified and no
      // > other bound is specified, the lower bound is automatically set to -∞
      if (!bounds_defined_for_var_id.count(var_id) && variable_upper_bounds[var_id] < f_t(0)) {
        variable_lower_bounds[var_id] = -std::numeric_limits<f_t>::infinity();
      }
      var_types[var_id] = 'I';
      break;
    case SemiContiniousVariable:
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "Unsupported semi continous bound type found! Line=%s",
                         std::string(line).c_str());
      break;
    default:
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "Invalid bound type found! Line=%s",
                         std::string(line).c_str());
      break;
  }
}

// NOTE: Explicitly instantiate all types here in order to avoid linker error
template class mps_parser_t<int, float>;

template class mps_parser_t<int, double>;

}  // namespace cuopt::mps_parser
