/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Local LPfeas experiment: permute native PSLP's owned host result, never PSLP state.
#pragma once

#include "experimental_rcm_algorithm.hpp"
#include "experimental_rcm_state.hpp"

#include <span>
#include <utility>

namespace lpfeas_ordering {

template <class T>
std::vector<T> permute_optional(const std::vector<T>& values, const std::vector<int>& new_to_old)
{
  if (values.empty()) { return {}; }
  if (values.size() != new_to_old.size()) {
    throw std::runtime_error("Native RCM metadata has the wrong row/column dimension");
  }
  std::vector<T> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[i] = values[new_to_old[i]];
  }
  return result;
}

// row_lower/upper are materialized by cuOpt's existing expand_rhs(), without
// applying its objective sign normalization. The model retains its native sense.
template <class Model>
native_rcm_state reorder_native_mps(Model& model,
                                    std::size_t hub_threshold,
                                    const std::vector<double>& row_lower,
                                    const std::vector<double>& row_upper)
{
  const auto m = static_cast<std::size_t>(model.get_n_constraints());
  const auto n = static_cast<std::size_t>(model.get_n_variables());
  if (m == 0 || n == 0) { return {}; }
  if (hub_threshold == 0) { throw std::runtime_error("Native RCM hub threshold must be positive"); }
  if (!model.get_initial_primal_solution().empty() || !model.get_initial_dual_solution().empty()) {
    throw std::runtime_error("Native RCM currently supports cold LP solves only");
  }
  if (row_lower.size() != m || row_upper.size() != m) {
    throw std::runtime_error("Native RCM requires materialized ranged row bounds");
  }
  auto lower = model.get_variable_lower_bounds();
  auto upper = model.get_variable_upper_bounds();
  if (lower.empty()) { lower.assign(n, -std::numeric_limits<double>::infinity()); }
  if (upper.empty()) { upper.assign(n, std::numeric_limits<double>::infinity()); }
  const auto& c       = model.get_objective_coefficients();
  const auto& values  = model.get_constraint_matrix_values();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& offsets = model.get_constraint_matrix_offsets();
  if (lower.size() != n || upper.size() != n || c.size() != n || offsets.size() != m + 1 ||
      indices.size() != values.size()) {
    throw std::runtime_error("Native RCM received inconsistent reduced-model dimensions");
  }
  problem_view input{m,
                     n,
                     values.size(),
                     offsets.data(),
                     indices.data(),
                     values.data(),
                     c.data(),
                     row_lower.data(),
                     row_upper.data(),
                     lower.data(),
                     upper.data()};
  auto ordered = reorder(input, hub_threshold);

  // Validate and gather all optional indexed metadata before modifying the model.
  const auto rhs       = permute_optional(model.get_constraint_bounds(), ordered.row_new_to_old);
  const auto row_types = permute_optional(model.get_row_types(), ordered.row_new_to_old);
  const auto row_names = permute_optional(model.get_row_names(), ordered.row_new_to_old);
  const auto variable_types =
    permute_optional(model.get_variable_types(), ordered.column_new_to_old);
  const auto variable_names =
    permute_optional(model.get_variable_names(), ordered.column_new_to_old);

  model.set_csr_constraint_matrix(std::span<const double>(ordered.Ax),
                                  std::span<const int>(ordered.Ai),
                                  std::span<const int>(ordered.Ap));
  model.set_objective_coefficients(std::span<const double>(ordered.c));
  model.set_variable_lower_bounds(std::span<const double>(ordered.lbs));
  model.set_variable_upper_bounds(std::span<const double>(ordered.ubs));
  model.set_constraint_lower_bounds(std::span<const double>(ordered.lhs));
  model.set_constraint_upper_bounds(std::span<const double>(ordered.rhs));
  if (!rhs.empty()) { model.set_constraint_bounds(std::span<const double>(rhs)); }
  if (!row_types.empty()) { model.set_row_types(std::span<const char>(row_types)); }
  if (!row_names.empty()) { model.set_row_names(row_names); }
  if (!variable_types.empty()) { model.set_variable_types(variable_types); }
  if (!variable_names.empty()) { model.set_variable_names(variable_names); }
  // Sense, objective offset/scaling, and scalar names are deliberately untouched.
  return {true,
          std::move(ordered.row_new_to_old),
          std::move(ordered.column_new_to_old),
          ordered.excluded_rows,
          ordered.excluded_nnz,
          ordered.graph_components};
}

}  // namespace lpfeas_ordering
