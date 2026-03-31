/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mps_parser/mps_writer.hpp>

#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <utilities/error.hpp>
#include <utilities/sparse_matrix_helpers.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>

namespace cuopt::mps_parser {

template <typename i_t, typename f_t>
mps_writer_t<i_t, f_t>::mps_writer_t(const data_model_view_t<i_t, f_t>& problem) : problem_(problem)
{
}

template <typename i_t, typename f_t>
data_model_view_t<i_t, f_t> mps_writer_t<i_t, f_t>::create_view(
  const mps_data_model_t<i_t, f_t>& model)
{
  data_model_view_t<i_t, f_t> view;

  // Set basic data
  view.set_maximize(model.get_sense());

  // Constraint matrix
  const auto& A_values  = model.get_constraint_matrix_values();
  const auto& A_indices = model.get_constraint_matrix_indices();
  const auto& A_offsets = model.get_constraint_matrix_offsets();
  if (!A_values.empty()) {
    view.set_csr_constraint_matrix(A_values.data(),
                                   static_cast<i_t>(A_values.size()),
                                   A_indices.data(),
                                   static_cast<i_t>(A_indices.size()),
                                   A_offsets.data(),
                                   static_cast<i_t>(A_offsets.size()));
  }

  // Constraint bounds
  const auto& b = model.get_constraint_bounds();
  if (!b.empty()) { view.set_constraint_bounds(b.data(), static_cast<i_t>(b.size())); }

  // Objective coefficients
  const auto& c = model.get_objective_coefficients();
  if (!c.empty()) { view.set_objective_coefficients(c.data(), static_cast<i_t>(c.size())); }

  view.set_objective_scaling_factor(model.get_objective_scaling_factor());
  view.set_objective_offset(model.get_objective_offset());

  // Variable bounds
  const auto& lb = model.get_variable_lower_bounds();
  const auto& ub = model.get_variable_upper_bounds();
  if (!lb.empty()) { view.set_variable_lower_bounds(lb.data(), static_cast<i_t>(lb.size())); }
  if (!ub.empty()) { view.set_variable_upper_bounds(ub.data(), static_cast<i_t>(ub.size())); }

  // Variable types
  const auto& var_types = model.get_variable_types();
  if (!var_types.empty()) {
    view.set_variable_types(var_types.data(), static_cast<i_t>(var_types.size()));
  }

  // Row types
  const auto& row_types = model.get_row_types();
  if (!row_types.empty()) {
    view.set_row_types(row_types.data(), static_cast<i_t>(row_types.size()));
  }

  // Constraint bounds (lower/upper)
  const auto& cl = model.get_constraint_lower_bounds();
  const auto& cu = model.get_constraint_upper_bounds();
  if (!cl.empty()) { view.set_constraint_lower_bounds(cl.data(), static_cast<i_t>(cl.size())); }
  if (!cu.empty()) { view.set_constraint_upper_bounds(cu.data(), static_cast<i_t>(cu.size())); }

  // Names
  view.set_problem_name(model.get_problem_name());
  view.set_objective_name(model.get_objective_name());
  view.set_variable_names(model.get_variable_names());
  view.set_row_names(model.get_row_names());

  // Quadratic objective
  const auto& Q_values  = model.get_quadratic_objective_values();
  const auto& Q_indices = model.get_quadratic_objective_indices();
  const auto& Q_offsets = model.get_quadratic_objective_offsets();
  if (!Q_values.empty()) {
    view.set_quadratic_objective_matrix(Q_values.data(),
                                        static_cast<i_t>(Q_values.size()),
                                        Q_indices.data(),
                                        static_cast<i_t>(Q_indices.size()),
                                        Q_offsets.data(),
                                        static_cast<i_t>(Q_offsets.size()));
  }

  return view;
}

template <typename i_t, typename f_t>
mps_writer_t<i_t, f_t>::mps_writer_t(const mps_data_model_t<i_t, f_t>& problem)
  : owned_view_(std::make_unique<data_model_view_t<i_t, f_t>>(create_view(problem))),
    problem_(*owned_view_)
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
  std::vector<bool> var_in_constraint(n_variables, false);
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
      var_in_constraint[var] = true;
    }
  }

  // Record and explicitely declared variables not contained in any constraint.
  std::vector<i_t> orphan_continuous_vars;
  std::vector<i_t> orphan_integer_vars;
  for (i_t var = 0; var < n_variables; ++var) {
    if (!var_in_constraint[var]) {
      if (variable_types[var] == 'I') {
        orphan_integer_vars.push_back(var);
      } else {
        orphan_continuous_vars.push_back(var);
      }
    }
  }

  for (size_t is_integral = 0; is_integral < 2; is_integral++) {
    auto& col_map     = is_integral ? integral_col_nnzs : continuous_col_nnzs;
    auto& orphan_vars = is_integral ? orphan_integer_vars : orphan_continuous_vars;
    if (is_integral) mps_file << "    MARK0001  'MARKER'                 'INTORG'\n";
    for (auto& var_id : orphan_vars) {
      std::string col_name = var_id < problem_.get_variable_names().size()
                               ? problem_.get_variable_names()[var_id]
                               : "C" + std::to_string(var_id);
      // Write that column even if it is orphan as has a zero objective coefficient.
      // Some tools require variables to be declared in "COLUMNS" before any "BOUNDS" statements.
      mps_file << "    " << col_name << " "
               << (problem_.get_objective_name().empty() ? "OBJ" : problem_.get_objective_name())
               << " " << objective_coefficients[var_id] << "\n";
    }
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
    std::string col_name        = j < problem_.get_variable_names().size()
                                    ? problem_.get_variable_names()[j]
                                    : "C" + std::to_string(j);
    std::string lower_bound_str = variable_types[j] == 'I' ? "LI" : "LO";
    std::string upper_bound_str = variable_types[j] == 'I' ? "UI" : "UP";

    if (variable_lower_bounds[j] == -std::numeric_limits<f_t>::infinity() &&
        variable_upper_bounds[j] == std::numeric_limits<f_t>::infinity()) {
      mps_file << " FR BOUND1    " << col_name << "\n";
    }
    // Ambiguity exists in the spec about the case where upper_bound == 0 and lower_bound == 0, and
    // only UP is specified. Handle fixed variables explicitely to avoid this pitfall.
    else if (variable_lower_bounds[j] == variable_upper_bounds[j]) {
      mps_file << " FX BOUND1    " << col_name << " " << variable_lower_bounds[j] << "\n";
    } else {
      if (variable_lower_bounds[j] != 0.0) {
        if (variable_lower_bounds[j] == -std::numeric_limits<f_t>::infinity()) {
          mps_file << " MI BOUND1    " << col_name << "\n";
        } else {
          mps_file << " " << lower_bound_str << " BOUND1    " << col_name << " "
                   << variable_lower_bounds[j] << "\n";
        }
      }
      // Integer variables get different default bounds compared to continuous variables
      if (variable_upper_bounds[j] != std::numeric_limits<f_t>::infinity() ||
          variable_types[j] == 'I') {
        mps_file << " " << upper_bound_str << " BOUND1    " << col_name << " "
                 << variable_upper_bounds[j] << "\n";
      }
    }
  }

  // QUADOBJ section for quadratic objective terms (if present)
  // MPS format: QUADOBJ stores upper triangular elements (row <= col)
  // MPS uses (1/2) x^T H x, cuOpt uses x^T Q x
  // For equivalence: H[i,j] = Q[i,j] + Q[j,i] (works for both diagonal and off-diagonal)
  // We symmetrize Q first (H = Q + Q^T), then extract upper triangular
  if (problem_.has_quadratic_objective()) {
    auto Q_values_span  = problem_.get_quadratic_objective_values();
    auto Q_indices_span = problem_.get_quadratic_objective_indices();
    auto Q_offsets_span = problem_.get_quadratic_objective_offsets();

    // Copy span data to local vectors for indexed access
    std::vector<f_t> Q_values(Q_values_span.data(), Q_values_span.data() + Q_values_span.size());
    std::vector<i_t> Q_indices(Q_indices_span.data(),
                               Q_indices_span.data() + Q_indices_span.size());
    std::vector<i_t> Q_offsets(Q_offsets_span.data(),
                               Q_offsets_span.data() + Q_offsets_span.size());

    if (Q_values.size() > 0) {
      // Symmetrize Q: compute H = Q + Q^T
      std::vector<f_t> H_values;
      std::vector<i_t> H_indices;
      std::vector<i_t> H_offsets;

      if (problem_.is_Q_symmetrized()) {
        H_values  = std::move(Q_values);
        H_indices = std::move(Q_indices);
        H_offsets = std::move(Q_offsets);
      } else {
        cuopt::symmetrize_csr<i_t, f_t>(
          Q_values, Q_indices, Q_offsets, H_values, H_indices, H_offsets);
      }

      i_t n_rows = static_cast<i_t>(H_offsets.size()) - 1;

      mps_file << "QUADOBJ\n";

      // Write upper triangular entries from symmetric H
      for (i_t i = 0; i < n_rows; ++i) {
        std::string row_name = static_cast<size_t>(i) < problem_.get_variable_names().size()
                                 ? problem_.get_variable_names()[i]
                                 : "C" + std::to_string(i);

        for (i_t p = H_offsets[i]; p < H_offsets[i + 1]; ++p) {
          i_t j = H_indices[p];
          f_t v = H_values[p];

          // Only write upper triangular (i <= j)
          if (i <= j && v != f_t(0)) {
            std::string col_name = static_cast<size_t>(j) < problem_.get_variable_names().size()
                                     ? problem_.get_variable_names()[j]
                                     : "C" + std::to_string(j);
            mps_file << "    " << row_name << " " << col_name << " " << v << "\n";
          }
        }
      }
    }
  }

  mps_file << "ENDATA\n";
  mps_file.close();
}

template class mps_writer_t<int, float>;
template class mps_writer_t<int, double>;

}  // namespace cuopt::mps_parser
