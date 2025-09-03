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

#pragma once

#include <mps_parser/mps_data_model.hpp>

#include <stdarg.h>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuopt::mps_parser {

/**
 * @brief Different possible types of 'ROWS'
 */
enum RowType {
  Objective          = 'N',
  LesserThanOrEqual  = 'L',
  GreaterThanOrEqual = 'G',
  Equality           = 'E',
};  // enum RowType

/**
 * @brief Different possible types of 'BOUNDS'
 */
enum BoundType {
  LowerBound,
  UpperBound,
  Fixed,
  Free,
  LowerBoundNegInf,
  UpperBoundInf,
  BinaryVariable,
  LowerBoundIntegerVariable,
  UpperBoundIntegerVariable,
  SemiContiniousVariable,
};  // enum BoundType

/**
 * @brief Different possible types of 'OBJSENSE'
 */
enum ObjSenseType {
  Minimize,
  Maximize,
};  // enum ObjSenseType

/**
 * @brief Main parser class for MPS files
 *
 * @tparam f_t  data type of the weights and variables
 *
 * @note this parser assumes that the sections occur in the following order:
 *       `NAME -> ROWS -> COLUMNS -> RHS`
 */
template <typename i_t, typename f_t>
class mps_parser_t {
 public:
  /**
   * @brief Ctor. Parses the MPS file and generates the internal representation
   *        of the equations, which can then be used to convert to `standard_form`
   *
   * @param[out] problem Problem representation that will be filled after parsing the MPS file
   * @param[in] file Path to the MPS file to be parsed
   * @param[in] fixed_mps_format Bool which describes whether the MPS file is in fixed format or
   * not. Default is true.
   */
  mps_parser_t(mps_data_model_t<i_t, f_t>& problem,
               const std::string& file,
               bool fixed_mps_format = true);

  /** path to the mps file being parsed */
  std::string mps_file{};
  /** whether the MPS file is in fixed format or not */
  bool fixed_mps_format;
  /** name of the problem as found in the MPS file */
  std::string problem_name{};
  /** names of each of the rows (aka constraints or objective) in the LP */
  std::vector<std::string> row_names{};
  /** types of each rows in the LP (excluding the objective) */
  std::vector<RowType> row_types{};
  /** name of the objective */
  std::string objective_name;
  /** names of each of the variables in the LP */
  std::vector<std::string> var_names{};
  /** types of variables 'I' or 'C' */
  std::vector<char> var_types{};
  /** every variable that is part of each row */
  std::vector<std::vector<i_t>> A_indices{};
  /** values of the constraint matrix A */
  std::vector<std::vector<f_t>> A_values{};
  /** values of the RHS of the constraints */
  std::vector<f_t> b_values{};
  /** weights used in the objective */
  std::vector<f_t> c_values{};
  /** scaling factor used in the objective */
  f_t objective_scaling_factor_value{1};
  /** offset factor used in the objective */
  f_t objective_offset_value{0};
  /** weights for the upper bound of variable value */
  std::vector<f_t> variable_upper_bounds{};
  /** weights for the lower bound of variable value */
  std::vector<f_t> variable_lower_bounds{};
  /** ranges values for each constraint */
  std::vector<f_t> ranges_values{};
  /** Objection function sense (maximize of minimize) */
  bool maximize{false};

  // QPS-specific data for quadratic programming
  /** Quadratic objective matrix entries */
  std::vector<std::tuple<i_t, i_t, f_t>> quadobj_entries{};

 private:
  bool inside_rows_{false};
  bool inside_columns_{false};
  bool inside_rhs_{false};
  bool inside_bounds_{false};
  bool inside_ranges_{false};
  bool inside_objsense_{false};
  bool inside_intcapture_{false};
  bool inside_objname_{false};
  // QPS-specific parsing states
  bool inside_quadobj_{false};
  std::unordered_set<std::string> encountered_sections{};
  std::unordered_map<std::string, i_t> row_names_map{};
  std::unordered_map<std::string, i_t> var_names_map{};
  std::unordered_set<std::string> ignored_objective_names{};
  std::unordered_set<i_t> bounds_defined_for_var_id{};
  static constexpr f_t unset_range_value = std::numeric_limits<f_t>::infinity();

  /* Reads the equation from the input text file which is MPS formatted
   *
   * Read this link http://lpsolve.sourceforge.net/5.5/mps-format.htm for more
   * details on this format.
   */
  std::vector<char> file_to_string(const std::string& file);
  void fill_problem(mps_data_model_t<i_t, f_t>& problem);
  void parse_string(char* buf);
  void parse_rows(std::string_view line);
  void parse_columns(std::string_view line);
  i_t parse_column_var_name(std::string_view line);
  std::tuple<std::string_view, std::string_view, i_t> parse_row_name_and_num(std::string_view line,
                                                                             i_t start);
  void insert_row_name_and_value(std::string_view line,
                                 std::string_view row_name,
                                 std::string_view num,
                                 i_t var_id);
  void parse_column_row_and_value(std::string_view line, i_t pos);
  i_t read_row_and_value(std::string_view line, i_t start, i_t var_id);
  void parse_rhs(std::string_view line);
  template <bool bounds_or_ranges = false, int fixed_length = 12>
  f_t get_numerical_bound(std::string_view line, i_t& start);
  i_t read_rhs_row_and_value(std::string_view line, i_t start);
  void parse_bounds(std::string_view line);
  void parse_objsense(std::string_view line);
  void parse_objname(std::string_view line);
  void read_bound_and_value(std::string_view line, BoundType type, i_t var_id, i_t start);
  void parse_ranges(std::string_view line);
  i_t insert_range_value(std::string_view line, bool skip_range = true);

  // QPS-specific parsing methods
  void parse_quadobj(std::string_view line);

};  // class mps_parser_t

}  // namespace cuopt::mps_parser
