/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "problem.cuh"

#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

#include <mip/mip_constants.hpp>

#include "utilities/copy_helpers.hpp"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::write_as_mps(const std::string& path)
{
  // Get host copies of device data
  auto h_reverse_coefficients = cuopt::host_copy(reverse_coefficients, handle_ptr->get_stream());
  auto h_reverse_constraints  = cuopt::host_copy(reverse_constraints, handle_ptr->get_stream());
  auto h_reverse_offsets      = cuopt::host_copy(reverse_offsets, handle_ptr->get_stream());
  auto h_obj_coeffs           = cuopt::host_copy(objective_coefficients, handle_ptr->get_stream());
  auto h_var_lb               = cuopt::host_copy(variable_lower_bounds, handle_ptr->get_stream());
  auto h_var_ub               = cuopt::host_copy(variable_upper_bounds, handle_ptr->get_stream());
  auto h_cstr_lb              = cuopt::host_copy(constraint_lower_bounds, handle_ptr->get_stream());
  auto h_cstr_ub              = cuopt::host_copy(constraint_upper_bounds, handle_ptr->get_stream());
  auto h_var_types            = cuopt::host_copy(variable_types, handle_ptr->get_stream());

  std::ofstream mps_file(path);
  if (!mps_file.is_open()) {
    CUOPT_LOG_ERROR("Could not open file %s for writing", path.c_str());
    return;
  }

  // save coefficients with full precision
  mps_file << std::setprecision(std::numeric_limits<f_t>::max_digits10);

  // NAME section
  mps_file << "NAME          " << original_problem_ptr->get_problem_name() << "\n";

  // ROWS section
  mps_file << "ROWS\n";
  mps_file << " N  " << (objective_name.empty() ? "OBJ" : objective_name) << "\n";
  for (size_t i = 0; i < (size_t)n_constraints; i++) {
    std::string row_name = i < row_names.size() ? row_names[i] : "R" + std::to_string(i);

    char type = 'L';
    if (h_cstr_lb[i] == h_cstr_ub[i])
      type = 'E';
    else if (std::isinf(h_cstr_ub[i]))
      type = 'G';
    mps_file << " " << type << "  " << row_name << "\n";
  }

  // COLUMNS section
  mps_file << "COLUMNS\n";

  bool in_integer_section = false;
  for (size_t j = 0; j < (size_t)n_variables; j++) {
    std::string col_name = j < var_names.size() ? var_names[j] : "C" + std::to_string(j);

    // Write integer marker if needed
    if (h_var_types[j] != var_t::CONTINUOUS) {
      if (!in_integer_section) {
        mps_file << "    MARK0001  'MARKER'                 'INTORG'\n";
        in_integer_section = true;
      }
    }

    // Write objective coefficient if non-zero
    if (h_obj_coeffs[j] != 0.0) {
      mps_file << "    " << col_name << " " << (objective_name.empty() ? "OBJ" : objective_name)
               << " " << h_obj_coeffs[j] << "\n";
    }

    // Write constraint coefficients
    for (size_t k = (size_t)h_reverse_offsets[j]; k < (size_t)h_reverse_offsets[j + 1]; k++) {
      size_t row           = (size_t)h_reverse_constraints[k];
      std::string row_name = row < row_names.size() ? row_names[row] : "R" + std::to_string(row);
      mps_file << "    " << col_name << " " << row_name << " " << h_reverse_coefficients[k] << "\n";
    }

    if (h_var_types[j] != var_t::CONTINUOUS && in_integer_section) {
      if (j == (size_t)n_variables - 1 || h_var_types[j + 1] == var_t::CONTINUOUS) {
        mps_file << "    MARK0001  'MARKER'                 'INTEND'\n";
        in_integer_section = false;
      }
    }
  }

  // RHS section
  mps_file << "RHS\n";
  for (size_t i = 0; i < (size_t)n_constraints; i++) {
    std::string row_name = i < row_names.size() ? row_names[i] : "R" + std::to_string(i);

    f_t rhs;
    if (std::isinf(h_cstr_lb[i])) {
      rhs = h_cstr_ub[i];
    } else if (std::isinf(h_cstr_ub[i])) {
      rhs = h_cstr_lb[i];
    } else {  // RANGES, encode the lower bound
      rhs = h_cstr_lb[i];
    }

    if (isfinite(rhs) && rhs != 0.0) {
      mps_file << "    RHS1      " << row_name << " " << rhs << "\n";
    }
  }

  // RANGES section if needed
  bool has_ranges = false;
  for (size_t i = 0; i < (size_t)n_constraints; i++) {
    if (h_cstr_lb[i] != -std::numeric_limits<f_t>::infinity() &&
        h_cstr_ub[i] != std::numeric_limits<f_t>::infinity() && h_cstr_lb[i] != h_cstr_ub[i]) {
      if (!has_ranges) {
        mps_file << "RANGES\n";
        has_ranges = true;
      }
      std::string row_name = i < row_names.size() ? row_names[i] : "R" + std::to_string(i);
      mps_file << "    RNG1      " << row_name << " " << (h_cstr_ub[i] - h_cstr_lb[i]) << "\n";
    }
  }

  // BOUNDS section
  mps_file << "BOUNDS\n";
  for (size_t j = 0; j < (size_t)n_variables; j++) {
    std::string col_name = j < var_names.size() ? var_names[j] : "C" + std::to_string(j);

    if (h_var_lb[j] == -std::numeric_limits<f_t>::infinity() &&
        h_var_ub[j] == std::numeric_limits<f_t>::infinity()) {
      mps_file << " FR BOUND1    " << col_name << "\n";
    } else {
      if (h_var_lb[j] != 0.0) {
        if (h_var_lb[j] == -std::numeric_limits<f_t>::infinity()) {
          mps_file << " MI BOUND1    " << col_name << "\n";
        } else {
          mps_file << " LO BOUND1    " << col_name << " " << h_var_lb[j] << "\n";
        }
      }
      if (h_var_ub[j] != std::numeric_limits<f_t>::infinity()) {
        mps_file << " UP BOUND1    " << col_name << " " << h_var_ub[j] << "\n";
      }
    }
  }

  mps_file << "ENDATA\n";
  mps_file.close();
}

#if MIP_INSTANTIATE_FLOAT
template void problem_t<int, float>::write_as_mps(const std::string& path);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void problem_t<int, double>::write_as_mps(const std::string& path);
#endif

}  // namespace cuopt::linear_programming::detail
