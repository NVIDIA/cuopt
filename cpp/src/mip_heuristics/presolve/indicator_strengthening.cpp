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
#include <iterator>
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
  if (num_rows <= 0 || num_cols <= 0) { return; }

  auto is_free_binary = [&](int col) {
    const auto& flags = col_flags[col];
    return flags.test(papilo::ColFlag::kIntegral) && !flags.test(papilo::ColFlag::kLbInf) &&
           !flags.test(papilo::ColFlag::kUbInf) && !flags.test(papilo::ColFlag::kFixed) &&
           lower_bounds[col] == 0.0 && upper_bounds[col] == 1.0;
  };

  // +1 when the stored row reads a.x <= rhs, -1 when it reads a.x >= lhs, 0 for equations, ranges
  // and free rows.
  auto orientation = [&](int row) {
    const bool lhs_infinite = row_flags[row].test(papilo::RowFlag::kLhsInf);
    const bool rhs_infinite = row_flags[row].test(papilo::RowFlag::kRhsInf);
    if (lhs_infinite == rhs_infinite) { return 0; }
    return lhs_infinite ? 1 : -1;
  };

  // The variable upper bounds x <= z over binaries, in CSR form over the members: the indicators
  // bounding column j are vub_indicators[vub_offsets[j] .. vub_offsets[j+1]), sorted and unique.
  std::vector<std::pair<int, int>> vubs;
  for (int row = 0; row < num_rows; ++row) {
    const int direction = orientation(row);
    if (direction == 0) { continue; }
    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    if (row_coefficients.getLength() != 2) { continue; }
    const f_t side = direction == 1 ? rhs_values[row] : lhs_values[row];
    if (side != 0.0) { continue; }

    const int* indices = row_coefficients.getIndices();
    const f_t* values  = row_coefficients.getValues();
    int member = -1, indicator = -1;
    for (int p = 0; p < 2; ++p) {
      if (!is_free_binary(indices[p])) {
        member = indicator = -1;
        break;
      }
      const f_t v = direction * values[p];
      if (v == 1.0) {
        member = indices[p];
      } else if (v == -1.0) {
        indicator = indices[p];
      }
    }
    if (member >= 0 && indicator >= 0) { vubs.emplace_back(member, indicator); }
  }
  if (vubs.empty()) { return; }

  std::sort(vubs.begin(), vubs.end());
  vubs.erase(std::unique(vubs.begin(), vubs.end()), vubs.end());
  std::vector<int> vub_offsets(num_cols + 1, 0);
  std::vector<int> vub_indicators(vubs.size());
  for (size_t k = 0; k < vubs.size(); ++k) {
    ++vub_offsets[vubs[k].first + 1];
    vub_indicators[k] = vubs[k].second;
  }
  for (int col = 0; col < num_cols; ++col) {
    vub_offsets[col + 1] += vub_offsets[col];
  }

  papilo::Vec<papilo::Triplet<f_t>> entries;
  entries.reserve(constraint_matrix.getNnz() + num_rows);
  papilo::Vec<f_t> lhs(lhs_values.begin(), lhs_values.end());
  papilo::Vec<f_t> rhs(rhs_values.begin(), rhs_values.end());
  papilo::Vec<papilo::RowFlags> flags(row_flags.begin(), row_flags.end());

  std::vector<int> implied_heads;
  std::vector<int> implied_offsets{0};
  std::vector<int> implied_indicators;
  int num_lifted = 0;

  std::vector<int> indicators;
  std::vector<int> members;
  std::vector<int> support;
  std::vector<int> common;
  std::vector<int> intersection;

  for (int row = 0; row < num_rows; ++row) {
    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const int len         = row_coefficients.getLength();
    const int* indices    = row_coefficients.getIndices();
    const f_t* values     = row_coefficients.getValues();
    for (int p = 0; p < len; ++p) {
      entries.emplace_back(row, indices[p], values[p]);
    }

    const int direction = orientation(row);
    if (direction == 0 || len < 2) { continue; }
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
        if (!is_free_binary(col)) {
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
      if (!usable || head < 0 || indicators.empty()) { continue; }

      std::sort(indicators.begin(), indicators.end());
      indicators.erase(std::unique(indicators.begin(), indicators.end()), indicators.end());
      const size_t num_members = len - 1;
      if (indicators.size() >= num_members) { continue; }
      if (std::binary_search(indicators.begin(), indicators.end(), head)) { continue; }

      implied_heads.push_back(head);
      implied_indicators.insert(implied_indicators.end(), indicators.begin(), indicators.end());
      implied_offsets.push_back(implied_indicators.size());
      continue;
    }
    if (capacity < 0.0) { continue; }

    // Capacity row sum_{i in S} x_i - s <= K with every x_i bounded by a common indicator z.
    members.clear();
    support.assign(indices, indices + len);
    bool usable = true;
    for (int p = 0; p < len && usable; ++p) {
      const int col = indices[p];
      const f_t v   = direction * values[p];
      if (col_flags[col].test(papilo::ColFlag::kIntegral)) {
        usable = is_free_binary(col) && v == 1.0;
        if (usable) { members.push_back(col); }
      } else {
        usable =
          v < 0.0 && !col_flags[col].test(papilo::ColFlag::kLbInf) && lower_bounds[col] >= 0.0;
      }
    }
    if (!usable || members.size() < 2) { continue; }
    const f_t n_members = members.size();
    if (capacity >= n_members) { continue; }

    common.assign(vub_indicators.begin() + vub_offsets[members[0]],
                  vub_indicators.begin() + vub_offsets[members[0] + 1]);
    for (size_t k = 1; k < members.size() && !common.empty(); ++k) {
      const int col = members[k];
      intersection.clear();
      std::set_intersection(common.begin(),
                            common.end(),
                            vub_indicators.begin() + vub_offsets[col],
                            vub_indicators.begin() + vub_offsets[col + 1],
                            std::back_inserter(intersection));
      common.swap(intersection);
    }
    if (common.empty()) { continue; }

    std::sort(support.begin(), support.end());
    int indicator = -1;
    for (int z : common) {
      if (std::binary_search(support.begin(), support.end(), z)) { continue; }
      indicator = z;
      break;
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

  const int num_implied = implied_heads.size();
  if (num_implied == 0 && num_lifted == 0) { return; }

  for (int k = 0; k < num_implied; ++k) {
    const int row = num_rows + k;
    entries.emplace_back(row, implied_heads[k], f_t{1});
    for (int p = implied_offsets[k]; p < implied_offsets[k + 1]; ++p) {
      entries.emplace_back(row, implied_indicators[p], f_t{-1});
    }
    lhs.push_back(0.0);
    rhs.push_back(0.0);
    papilo::RowFlags row_flag;
    row_flag.set(papilo::RowFlag::kLhsInf);
    flags.push_back(row_flag);
  }

  if (!problem.getConstraintNames().empty()) {
    papilo::Vec<papilo::String> names = problem.getConstraintNames();
    for (int k = 0; k < num_implied; ++k) {
      names.push_back("implied_indicator_" + std::to_string(k));
    }
    problem.setConstraintNames(std::move(names));
  }

  const int num_vubs = vub_indicators.size();
  papilo::SparseStorage<f_t> storage(
    std::move(entries), num_rows + num_implied, num_cols, false, 4.0, 30);
  problem.setConstraintMatrix(std::move(storage), std::move(lhs), std::move(rhs), std::move(flags));

  CUOPT_LOG_INFO(
    "Indicator strengthening: %d implied indicator rows added, %d capacity rows lifted over %d "
    "variable upper bounds",
    num_implied,
    num_lifted,
    num_vubs);
}

#define INSTANTIATE(F_TYPE) template void strengthen_indicators<F_TYPE>(papilo::Problem<F_TYPE>&);

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
