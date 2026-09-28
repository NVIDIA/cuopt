/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "indicator_strengthening.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename f_t>
void strengthen_indicators(papilo::Problem<f_t>& problem)
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;
  const auto& upper_bounds      = domains.upper_bounds;

  const int num_rows = constraint_matrix.getNRows();
  const int num_cols = problem.getNCols();

  std::vector<bool> is_binary(num_cols);
  for (int col = 0; col < num_cols; ++col) {
    is_binary[col] = col_flags[col].test(papilo::ColFlag::kIntegral) &&
                     !col_flags[col].test(papilo::ColFlag::kLbInf, papilo::ColFlag::kUbInf) &&
                     lower_bounds[col] == 0.0 && upper_bounds[col] == 1.0;
  }

  // +1 when the stored row reads a.x <= rhs, -1 when it reads a.x >= lhs, 0 for equations, ranges
  // and free rows.
  std::vector<int> orientation(num_rows, 0);
  for (int row = 0; row < num_rows; ++row) {
    const bool lhs_infinite = row_flags[row].test(papilo::RowFlag::kLhsInf);
    const bool rhs_infinite = row_flags[row].test(papilo::RowFlag::kRhsInf);
    if (lhs_infinite != rhs_infinite) { orientation[row] = lhs_infinite ? 1 : -1; }
  }

  // The variable upper bounds x <= z over binaries, in CSR form over the members: the indicators
  // bounding column j are vub_indicators[vub_offsets[j] .. vub_offsets[j+1]).
  std::vector<int> vub_offsets(num_cols + 1, 0);
  std::vector<int> vub_indicators;
  for (int col = 0; col < num_cols; ++col) {
    if (is_binary[col]) {
      auto col_coefficients = constraint_matrix.getColumnCoefficients(col);
      const int* rows       = col_coefficients.getIndices();
      const f_t* col_values = col_coefficients.getValues();
      for (int p = 0; p < col_coefficients.getLength(); ++p) {
        const int row       = rows[p];
        const int direction = orientation[row];
        if (direction == 0 || direction * col_values[p] != 1.0) { continue; }
        auto row_coefficients = constraint_matrix.getRowCoefficients(row);
        if (row_coefficients.getLength() != 2) { continue; }
        const f_t side = direction == 1 ? rhs_values[row] : lhs_values[row];
        if (side != 0.0) { continue; }

        const int* indices = row_coefficients.getIndices();
        const f_t* values  = row_coefficients.getValues();
        const int other    = indices[0] == col ? 1 : 0;
        if (is_binary[indices[other]] && direction * values[other] == -1.0) {
          vub_indicators.push_back(indices[other]);
        }
      }
    }
    vub_offsets[col + 1] = vub_indicators.size();
  }
  if (vub_indicators.empty()) { return; }

  papilo::Vec<papilo::Triplet<f_t>> entries;
  entries.reserve(constraint_matrix.getNnz() + num_rows);
  papilo::Vec<f_t> lhs(lhs_values.begin(), lhs_values.end());
  papilo::Vec<f_t> rhs(rhs_values.begin(), rhs_values.end());
  papilo::Vec<papilo::RowFlags> flags(row_flags.begin(), row_flags.end());

  papilo::Vec<papilo::Triplet<f_t>> implied_entries;
  papilo::RowFlags implied_flags;
  implied_flags.set(papilo::RowFlag::kLhsInf);
  int num_implied = 0;
  int num_lifted  = 0;

  std::vector<int> indicators;
  std::vector<int> members;

  for (int row = 0; row < num_rows; ++row) {
    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const int len         = row_coefficients.getLength();
    const int* indices    = row_coefficients.getIndices();
    const f_t* values     = row_coefficients.getValues();
    for (int p = 0; p < len; ++p) {
      entries.emplace_back(row, indices[p], values[p]);
    }

    const int direction = orientation[row];
    if (direction == 0) { continue; }
    const f_t capacity = direction * (direction == 1 ? rhs_values[row] : lhs_values[row]);

    // Implication row y - sum_{j in S} x_j <= 0: one +1 head against a tail of -1 members.
    if (capacity == 0.0) {
      if (len < 3) { continue; }
      int head    = -1;
      bool usable = true;
      indicators.clear();
      for (int p = 0; p < len && usable; ++p) {
        const int col = indices[p];
        const f_t v   = direction * values[p];
        if (!is_binary[col]) {
          usable = false;
        } else if (v == 1.0) {
          usable = head < 0;
          head   = col;
        } else if (v == -1.0) {
          if (vub_offsets[col] == vub_offsets[col + 1]) {
            indicators.push_back(col);
          } else {
            indicators.insert(indicators.end(),
                              vub_indicators.begin() + vub_offsets[col],
                              vub_indicators.begin() + vub_offsets[col + 1]);
          }
        } else {
          usable = false;
        }
      }
      if (!usable || head < 0) { continue; }

      std::sort(indicators.begin(), indicators.end());
      indicators.erase(std::unique(indicators.begin(), indicators.end()), indicators.end());
      const size_t num_members = len - 1;
      if (indicators.size() >= num_members) { continue; }
      if (std::binary_search(indicators.begin(), indicators.end(), head)) { continue; }

      const int implied_row = num_rows + num_implied;
      implied_entries.emplace_back(implied_row, head, f_t{1});
      for (int z : indicators) {
        implied_entries.emplace_back(implied_row, z, f_t{-1});
      }
      lhs.push_back(0.0);
      rhs.push_back(0.0);
      flags.push_back(implied_flags);
      ++num_implied;
      continue;
    }
    if (capacity < 0.0) { continue; }

    // Capacity row sum_{i in S} x_i - s <= K with every x_i bounded by a common indicator z.
    members.clear();
    bool usable = true;
    for (int p = 0; p < len && usable; ++p) {
      const int col = indices[p];
      const f_t v   = direction * values[p];
      if (col_flags[col].test(papilo::ColFlag::kIntegral)) {
        usable = is_binary[col] && v == 1.0;
        if (usable) { members.push_back(col); }
      } else {
        usable =
          v < 0.0 && !col_flags[col].test(papilo::ColFlag::kLbInf) && lower_bounds[col] >= 0.0;
      }
    }
    if (!usable || members.size() < 2) { continue; }
    const f_t n_members = members.size();
    if (capacity >= n_members) { continue; }

    int indicator = -1;
    for (int p = vub_offsets[members[0]]; p < vub_offsets[members[0] + 1] && indicator < 0; ++p) {
      const int z = vub_indicators[p];
      bool shared = true;
      for (size_t k = 1; k < members.size() && shared; ++k) {
        const auto span_begin = vub_indicators.begin() + vub_offsets[members[k]];
        const auto span_end   = vub_indicators.begin() + vub_offsets[members[k] + 1];
        shared                = std::find(span_begin, span_end, z) != span_end;
      }
      if (shared) { indicator = z; }
    }
    if (indicator < 0) { continue; }

    entries.emplace_back(row, indicator, direction * -capacity);
    if (direction == 1) {
      rhs[row] = 0.0;
    } else {
      lhs[row] = 0.0;
    }
    ++num_lifted;
  }

  if (num_implied == 0 && num_lifted == 0) { return; }
  entries.insert(entries.end(), implied_entries.begin(), implied_entries.end());

  if (!problem.getConstraintNames().empty()) {
    papilo::Vec<papilo::String> names = problem.getConstraintNames();
    for (int k = 0; k < num_implied; ++k) {
      names.push_back("implied_indicator_" + std::to_string(k));
    }
    problem.setConstraintNames(std::move(names));
  }

  papilo::SparseStorage<f_t> storage(
    std::move(entries), num_rows + num_implied, num_cols, false, 4.0, 30);
  problem.setConstraintMatrix(std::move(storage), std::move(lhs), std::move(rhs), std::move(flags));

  CUOPT_LOG_DEBUG(
    "Indicator strengthening: %d implied indicator rows added, %d capacity rows lifted over %zu "
    "variable upper bounds",
    num_implied,
    num_lifted,
    vub_indicators.size());
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template void strengthen_indicators<float>(papilo::Problem<float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void strengthen_indicators<double>(papilo::Problem<double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
