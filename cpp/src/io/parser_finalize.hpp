/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/io/mps_data_model.hpp>
#include <mps_parser_internal.hpp>
#include <utilities/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::io::detail {

// Consumes the LP parser's intermediate parsed data and populates `problem`.
//
// CSR flatten, row-type → constraint-bound conversion, quadratic objective
// matrix construction, metadata setters. MPS uses its own fill_problem
// because it handles QCMATRIX quadratic constraints; the LP format does not
// support quadratic constraints, so the two finalization paths intentionally
// diverge.
//
// Required fields on `parser`:
//   problem_name, objective_name, row_names, row_types, var_names,
//   var_types, A_indices, A_values, b_values, c_values,
//   variable_lower_bounds, variable_upper_bounds, objective_offset_value,
//   maximize, quadobj_entries.
//
// The requires-expression branches for objective_scaling_factor_value,
// ranges_values, and qmatrix_entries are dormant for LP but kept so the
// template would remain reusable if a non-LP caller ever wants them.
template <typename Parser, typename i_t, typename f_t>
void finalize_problem(mps_data_model_t<i_t, f_t>& problem, Parser& parser)
{
  const i_t n_vars = static_cast<i_t>(parser.var_names.size());
  const i_t n_rows = static_cast<i_t>(parser.row_names.size());

  // Pad per-variable vectors that may have grown after their initial size
  // (e.g., a variable first appeared after c_values was already initialized).
  if (static_cast<i_t>(parser.c_values.size()) < n_vars) parser.c_values.resize(n_vars, f_t(0));
  if (static_cast<i_t>(parser.variable_lower_bounds.size()) < n_vars) {
    parser.variable_lower_bounds.resize(n_vars, f_t(0));
  }
  if (static_cast<i_t>(parser.variable_upper_bounds.size()) < n_vars) {
    parser.variable_upper_bounds.resize(n_vars, std::numeric_limits<f_t>::infinity());
  }
  if (static_cast<i_t>(parser.var_types.size()) < n_vars) parser.var_types.resize(n_vars, 'C');

  // Flatten the ragged A_indices / A_values into a single CSR.
  std::vector<i_t> offsets;
  std::vector<i_t> indices;
  std::vector<f_t> values;
  offsets.reserve(n_rows + 1);
  offsets.push_back(0);
  for (i_t i = 0; i < n_rows; ++i) {
    for (i_t idx : parser.A_indices[i])
      indices.push_back(idx);
    for (f_t v : parser.A_values[i])
      values.push_back(v);
    offsets.push_back(static_cast<i_t>(values.size()));
  }
  problem.set_csr_constraint_matrix(values, indices, offsets);

  mps_parser_expects(indices.size() == values.size(),
                     error_type_t::ValidationError,
                     "Constraint matrix nonzero vector (%zu) and column-index vector (%zu) "
                     "must have the same size.",
                     indices.size(),
                     values.size());
  mps_parser_expects(!offsets.empty() && offsets.back() == static_cast<i_t>(values.size()),
                     error_type_t::ValidationError,
                     "CSR offset tail (%d) must equal the nonzero count (%zu).",
                     offsets.empty() ? 0 : offsets.back(),
                     values.size());

  problem.set_constraint_bounds(parser.b_values);
  problem.set_objective_coefficients(parser.c_values);

  f_t scaling = f_t(1);
  if constexpr (requires { parser.objective_scaling_factor_value; }) {
    scaling = parser.objective_scaling_factor_value;
  }
  problem.set_objective_scaling_factor(scaling);
  problem.set_objective_offset(parser.objective_offset_value);

  problem.set_variable_lower_bounds(parser.variable_lower_bounds);
  problem.set_variable_upper_bounds(parser.variable_upper_bounds);

  mps_parser_expects(
    (problem.get_variable_lower_bounds().size() == problem.get_variable_upper_bounds().size()) &&
      (problem.get_variable_upper_bounds().size() == problem.get_objective_coefficients().size()),
    error_type_t::ValidationError,
    "Per-variable vectors are inconsistently sized. objective=%zu, lb=%zu, ub=%zu.",
    problem.get_objective_coefficients().size(),
    problem.get_variable_lower_bounds().size(),
    problem.get_variable_upper_bounds().size());

  // Semi-continuous variables must have a finite upper bound; otherwise the
  // "x = 0 or lb <= x <= ub" semantics collapse to a regular continuous
  // variable. Matches the MPS parser's rule.
  for (i_t i = 0; i < n_vars; ++i) {
    if (parser.var_types[i] == 'S') {
      mps_parser_expects(!std::isinf(parser.variable_upper_bounds[i]),
                         error_type_t::ValidationError,
                         "Semi-continuous variable '%s' must have a finite upper bound",
                         parser.var_names[i].c_str());
    }
  }

  // Row types + RHS (+ MPS ranges) → explicit constraint lower/upper bounds.
  const f_t inf = std::numeric_limits<f_t>::infinity();
  std::vector<f_t> clb;
  std::vector<f_t> cub;
  clb.reserve(n_rows);
  cub.reserve(n_rows);
  constexpr bool has_ranges = requires { parser.ranges_values; };
  for (i_t i = 0; i < n_rows; ++i) {
    switch (parser.row_types[i]) {
      case Equality:
        clb.push_back(parser.b_values[i]);
        cub.push_back(parser.b_values[i]);
        if constexpr (has_ranges) {
          if (!parser.ranges_values.empty() && parser.ranges_values[i] != inf) {
            mps_parser_expects(!std::isnan(parser.ranges_values[i]),
                               error_type_t::ValidationError,
                               "Equality range value %d is NaN",
                               i);
            if (parser.ranges_values[i] < f_t(0)) {
              clb.back() += parser.ranges_values[i];
            } else {
              cub.back() += parser.ranges_values[i];
            }
          }
        }
        break;
      case GreaterThanOrEqual:
        clb.push_back(parser.b_values[i]);
        cub.push_back(inf);
        if constexpr (has_ranges) {
          if (!parser.ranges_values.empty() && parser.ranges_values[i] != inf) {
            mps_parser_expects(!std::isnan(parser.ranges_values[i]),
                               error_type_t::ValidationError,
                               "Greater range value %d is NaN",
                               i);
            cub.back() = clb.back() + std::abs(parser.ranges_values[i]);
          }
        }
        break;
      case LesserThanOrEqual:
        clb.push_back(-inf);
        cub.push_back(parser.b_values[i]);
        if constexpr (has_ranges) {
          if (!parser.ranges_values.empty() && parser.ranges_values[i] != inf) {
            mps_parser_expects(!std::isnan(parser.ranges_values[i]),
                               error_type_t::ValidationError,
                               "Lesser range value %d is NaN",
                               i);
            clb.back() = cub.back() - std::abs(parser.ranges_values[i]);
          }
        }
        break;
      default:
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "Unsupported row type for row '%s'",
                           parser.row_names[i].c_str());
    }
    mps_parser_expects(!std::isnan(clb.back()) && !std::isnan(cub.back()),
                       error_type_t::ValidationError,
                       "Constraint bound for row '%s' is NaN",
                       parser.row_names[i].c_str());
  }
  problem.set_constraint_lower_bounds(clb);
  problem.set_constraint_upper_bounds(cub);

  mps_parser_expects(
    (problem.get_constraint_lower_bounds().size() ==
     problem.get_constraint_upper_bounds().size()) &&
      (problem.get_constraint_upper_bounds().size() == problem.get_constraint_bounds().size()),
    error_type_t::ValidationError,
    "Per-constraint vectors are inconsistently sized. rhs=%zu, lb=%zu, ub=%zu.",
    problem.get_constraint_bounds().size(),
    problem.get_constraint_lower_bounds().size(),
    problem.get_constraint_upper_bounds().size());

  problem.set_problem_name(parser.problem_name);
  problem.set_objective_name(parser.objective_name);
  // Setters take const refs — pass the fields directly to avoid an extra
  // temporary copy.
  problem.set_variable_names(parser.var_names);
  problem.set_variable_types(parser.var_types);
  problem.set_row_names(parser.row_names);
  std::vector<char> row_types_chars(parser.row_types.size());
  for (size_t i = 0; i < parser.row_types.size(); ++i) {
    row_types_chars[i] = static_cast<char>(parser.row_types[i]);
  }
  problem.set_row_types(row_types_chars);
  problem.set_maximize(parser.maximize);

  // Quadratic objective: build a full symmetric Q via double-transpose.
  //   - QUADOBJ entries are upper-triangular; each off-diagonal entry is
  //     mirrored to its transpose when assembling.
  //   - QMATRIX entries are already the full symmetric matrix.
  // Every stored value is multiplied by 0.5 to convert from the file's
  // '0.5 x^T Q x' convention to cuOpt's 'x^T Q x'. See mps_parser.cpp for
  // the original derivation.
  auto build_q_csr = [&](const std::vector<std::tuple<i_t, i_t, f_t>>& entries,
                         bool mirror_off_diagonal) {
    std::vector<std::vector<std::pair<i_t, f_t>>> csc(n_vars);
    for (const auto& [row, col, val] : entries) {
      csc[col].emplace_back(row, val);
      if (mirror_off_diagonal && row != col) { csc[row].emplace_back(col, val); }
    }
    std::vector<std::vector<std::pair<i_t, f_t>>> csr(n_vars);
    for (i_t col = 0; col < n_vars; ++col) {
      for (const auto& [row, val] : csc[col]) {
        csr[row].emplace_back(col, val);
      }
    }
    // Within each row the entries are naturally ordered by column because
    // the outer loop above walks columns in ascending order — no sort needed.
    std::vector<f_t> q_values;
    std::vector<i_t> q_indices;
    std::vector<i_t> q_offsets;
    q_offsets.reserve(n_vars + 1);
    q_offsets.push_back(0);
    for (i_t row = 0; row < n_vars; ++row) {
      for (const auto& [col, val] : csr[row]) {
        q_values.push_back(val * f_t(0.5));
        q_indices.push_back(col);
      }
      q_offsets.push_back(static_cast<i_t>(q_values.size()));
    }
    problem.set_quadratic_objective_matrix(q_values, q_indices, q_offsets);
  };

  if (!parser.quadobj_entries.empty()) {
    build_q_csr(parser.quadobj_entries, /*mirror_off_diagonal=*/true);
  } else if constexpr (requires { parser.qmatrix_entries; }) {
    if (!parser.qmatrix_entries.empty()) {
      build_q_csr(parser.qmatrix_entries, /*mirror_off_diagonal=*/false);
    }
  }
}

}  // namespace cuopt::linear_programming::io::detail
