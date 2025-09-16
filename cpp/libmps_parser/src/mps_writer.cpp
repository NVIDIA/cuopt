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

#include <mps_parser/mps_writer.hpp>

#include <mps_parser/data_model_view.hpp>
#include <utilities/error.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

namespace cuopt::mps_parser {

template <typename i_t, typename f_t>
mps_writer_t<i_t, f_t>::mps_writer_t(const data_model_view_t<i_t, f_t>& problem) : problem_(problem)
{
}

template <typename i_t, typename f_t>
void mps_writer_t<i_t, f_t>::write(const std::string& mps_file_path)
{
  std::ofstream mps_file(mps_file_path);

  mps_parser_expects(mps_file.is_open(),
                     error_type_t::ValidationError,
                     "Error creating output MPS file! Given path: %s",
                     mps_file_path.c_str());

  i_t n_variables = problem_.get_variable_lower_bounds().size();
  i_t n_constraints;
  if (problem_.get_constraint_bounds().size() > 0)
    n_constraints = problem_.get_constraint_bounds().size();
  else
    n_constraints = problem_.get_constraint_lower_bounds().size();

  std::vector<f_t> objective_coefficients(problem_.get_objective_coefficients().size());
  std::vector<f_t> constraint_lower_bounds(n_constraints);
  std::vector<f_t> constraint_upper_bounds(n_constraints);
  std::vector<f_t> constraint_bounds(problem_.get_constraint_bounds().size());
  std::vector<f_t> variable_lower_bounds(problem_.get_variable_lower_bounds().size());
  std::vector<f_t> variable_upper_bounds(problem_.get_variable_upper_bounds().size());
  std::vector<char> variable_types(problem_.get_variable_types().size());
  std::vector<char> row_types(problem_.get_row_types().size());
  std::vector<i_t> constraint_matrix_offsets(problem_.get_constraint_matrix_offsets().size());
  std::vector<i_t> constraint_matrix_indices(problem_.get_constraint_matrix_indices().size());
  std::vector<f_t> constraint_matrix_values(problem_.get_constraint_matrix_values().size());

  std::copy(
    problem_.get_objective_coefficients().data(),
    problem_.get_objective_coefficients().data() + problem_.get_objective_coefficients().size(),
    objective_coefficients.data());
  std::copy(problem_.get_constraint_bounds().data(),
            problem_.get_constraint_bounds().data() + problem_.get_constraint_bounds().size(),
            constraint_bounds.data());
  std::copy(
    problem_.get_variable_lower_bounds().data(),
    problem_.get_variable_lower_bounds().data() + problem_.get_variable_lower_bounds().size(),
    variable_lower_bounds.data());
  std::copy(
    problem_.get_variable_upper_bounds().data(),
    problem_.get_variable_upper_bounds().data() + problem_.get_variable_upper_bounds().size(),
    variable_upper_bounds.data());
  std::copy(problem_.get_variable_types().data(),
            problem_.get_variable_types().data() + problem_.get_variable_types().size(),
            variable_types.data());
  std::copy(problem_.get_row_types().data(),
            problem_.get_row_types().data() + problem_.get_row_types().size(),
            row_types.data());
  std::copy(problem_.get_constraint_matrix_offsets().data(),
            problem_.get_constraint_matrix_offsets().data() +
              problem_.get_constraint_matrix_offsets().size(),
            constraint_matrix_offsets.data());
  std::copy(problem_.get_constraint_matrix_indices().data(),
            problem_.get_constraint_matrix_indices().data() +
              problem_.get_constraint_matrix_indices().size(),
            constraint_matrix_indices.data());
  std::copy(
    problem_.get_constraint_matrix_values().data(),
    problem_.get_constraint_matrix_values().data() + problem_.get_constraint_matrix_values().size(),
    constraint_matrix_values.data());

  if (problem_.get_constraint_lower_bounds().size() == 0 ||
      problem_.get_constraint_upper_bounds().size() == 0) {
    for (size_t i = 0; i < (size_t)n_constraints; i++) {
      constraint_lower_bounds[i] = constraint_bounds[i];
      constraint_upper_bounds[i] = constraint_bounds[i];
      if (row_types[i] == 'L') {
        constraint_lower_bounds[i] = -std::numeric_limits<f_t>::infinity();
      } else if (row_types[i] == 'G') {
        constraint_upper_bounds[i] = std::numeric_limits<f_t>::infinity();
      }
    }
  } else {
    std::copy(
      problem_.get_constraint_lower_bounds().data(),
      problem_.get_constraint_lower_bounds().data() + problem_.get_constraint_lower_bounds().size(),
      constraint_lower_bounds.data());
    std::copy(
      problem_.get_constraint_upper_bounds().data(),
      problem_.get_constraint_upper_bounds().data() + problem_.get_constraint_upper_bounds().size(),
      constraint_upper_bounds.data());
  }

  // save coefficients with full precision
  mps_file << std::setprecision(std::numeric_limits<f_t>::max_digits10);

  // NAME section
  mps_file << "NAME          " << problem_.get_problem_name() << "\n";

  if (problem_.get_sense()) { mps_file << "OBJSENSE\n MAXIMIZE\n"; }

  // ROWS section
  mps_file << "ROWS\n";
  mps_file << " N  "
           << (problem_.get_objective_name().empty() ? "OBJ" : problem_.get_objective_name())
           << "\n";
  for (size_t i = 0; i < (size_t)n_constraints; i++) {
    std::string row_name =
      i < problem_.get_row_names().size() ? problem_.get_row_names()[i] : "R" + std::to_string(i);
    char type = 'L';
    if (constraint_lower_bounds[i] == constraint_upper_bounds[i])
      type = 'E';
    else if (std::isinf(constraint_upper_bounds[i]))
      type = 'G';
    mps_file << " " << type << "  " << row_name << "\n";
  }

  // COLUMNS section
  mps_file << "COLUMNS\n";

  // Keep a single integer section marker by going over constraints twice and writing out
  // integral/nonintegral nonzeros ordered map
  std::map<i_t, std::vector<std::pair<i_t, f_t>>> integral_col_nnzs;
  std::map<i_t, std::vector<std::pair<i_t, f_t>>> continuous_col_nnzs;
  for (size_t row_id = 0; row_id < (size_t)n_constraints; row_id++) {
    for (size_t k = (size_t)constraint_matrix_offsets[row_id];
         k < (size_t)constraint_matrix_offsets[row_id + 1];
         k++) {
      size_t var = (size_t)constraint_matrix_indices[k];
      if (variable_types[var] == 'I') {
        integral_col_nnzs[var].emplace_back(row_id, constraint_matrix_values[k]);
      } else {
        continuous_col_nnzs[var].emplace_back(row_id, constraint_matrix_values[k]);
      }
    }
  }

  for (size_t is_integral = 0; is_integral < 2; is_integral++) {
    auto& col_map = is_integral ? integral_col_nnzs : continuous_col_nnzs;
    if (is_integral) mps_file << "    MARK0001  'MARKER'                 'INTORG'\n";
    for (auto& [var_id, nnzs] : col_map) {
      std::string col_name = var_id < problem_.get_variable_names().size()
                               ? problem_.get_variable_names()[var_id]
                               : "C" + std::to_string(var_id);
      for (auto& nnz : nnzs) {
        std::string row_name = nnz.first < problem_.get_row_names().size()
                                 ? problem_.get_row_names()[nnz.first]
                                 : "R" + std::to_string(nnz.first);
        mps_file << "    " << col_name << " " << row_name << " " << nnz.second << "\n";
      }
      // Write objective coefficients
      if (objective_coefficients[var_id] != 0.0) {
        mps_file << "    " << col_name << " "
                 << (problem_.get_objective_name().empty() ? "OBJ" : problem_.get_objective_name())
                 << " " << objective_coefficients[var_id] << "\n";
      }
    }
    if (is_integral) mps_file << "    MARK0001  'MARKER'                 'INTEND'\n";
  }

  // RHS section
  mps_file << "RHS\n";
  for (size_t i = 0; i < (size_t)n_constraints; i++) {
    std::string row_name =
      i < problem_.get_row_names().size() ? problem_.get_row_names()[i] : "R" + std::to_string(i);

    f_t rhs;
    if (constraint_bounds.size() > 0)
      rhs = constraint_bounds[i];
    else if (std::isinf(constraint_lower_bounds[i])) {
      rhs = constraint_upper_bounds[i];
    } else if (std::isinf(constraint_upper_bounds[i])) {
      rhs = constraint_lower_bounds[i];
    } else {  // RANGES, encode the lower bound
      rhs = constraint_lower_bounds[i];
    }

    if (std::isfinite(rhs) && rhs != 0.0) {
      mps_file << "    RHS1      " << row_name << " " << rhs << "\n";
    }
  }
  if (std::isfinite(problem_.get_objective_offset()) && problem_.get_objective_offset() != 0.0) {
    mps_file << "    RHS1      "
             << (problem_.get_objective_name().empty() ? "OBJ" : problem_.get_objective_name())
             << " " << -problem_.get_objective_offset() << "\n";
  }

  // RANGES section if needed
  bool has_ranges = false;
  for (size_t i = 0; i < (size_t)n_constraints; i++) {
    if (constraint_lower_bounds[i] != -std::numeric_limits<f_t>::infinity() &&
        constraint_upper_bounds[i] != std::numeric_limits<f_t>::infinity() &&
        constraint_lower_bounds[i] != constraint_upper_bounds[i]) {
      if (!has_ranges) {
        mps_file << "RANGES\n";
        has_ranges = true;
      }
      std::string row_name = "R" + std::to_string(i);
      mps_file << "    RNG1      " << row_name << " "
               << (constraint_upper_bounds[i] - constraint_lower_bounds[i]) << "\n";
    }
  }

  // BOUNDS section
  mps_file << "BOUNDS\n";
  for (size_t j = 0; j < (size_t)n_variables; j++) {
    std::string col_name = j < problem_.get_variable_names().size()
                             ? problem_.get_variable_names()[j]
                             : "C" + std::to_string(j);

    if (variable_lower_bounds[j] == -std::numeric_limits<f_t>::infinity() &&
        variable_upper_bounds[j] == std::numeric_limits<f_t>::infinity()) {
      mps_file << " FR BOUND1    " << col_name << "\n";
    } else {
      if (variable_lower_bounds[j] != 0.0 || objective_coefficients[j] == 0.0 ||
          variable_types[j] != 'C') {
        if (variable_lower_bounds[j] == -std::numeric_limits<f_t>::infinity()) {
          mps_file << " MI BOUND1    " << col_name << "\n";
        } else {
          mps_file << " LO BOUND1    " << col_name << " " << variable_lower_bounds[j] << "\n";
        }
      }
      if (variable_upper_bounds[j] != std::numeric_limits<f_t>::infinity()) {
        mps_file << " UP BOUND1    " << col_name << " " << variable_upper_bounds[j] << "\n";
      }
    }
  }

  mps_file << "ENDATA\n";
  mps_file.close();
}

template class mps_writer_t<int, float>;
template class mps_writer_t<int, double>;

}  // namespace cuopt::mps_parser
