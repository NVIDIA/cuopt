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
#include <tuple>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

template <typename i_t, typename f_t>
class indicator_strengthening_t {
 public:
  indicator_strengthening_t(const papilo::Problem<f_t>& problem);

  // Implication row y - sum_{j in S} x_j <= 0: one +1 head against a tail of -1 members.
  i_t add_implied_indicator_rows(papilo::Vec<papilo::Triplet<f_t>>& implied_entries) const;

  // Capacity row sum_{i in S} x_i - s <= K with every x_i bounded by a common indicator z.
  i_t lift_capacity_rows(papilo::Vec<papilo::Triplet<f_t>>& lifted_entries) const;

  i_t num_vubs() const { return vub_indicators.size(); }
  i_t row_orientation(i_t row) const { return orientation[row]; }

 private:
  const papilo::Problem<f_t>& problem;

  std::vector<char> is_binary;

  // +1 when the stored row reads a.x <= rhs, -1 when it reads a.x >= lhs, 0 for equations, ranges
  // and free rows.
  std::vector<i_t> orientation;
  // The variable upper bounds x <= z over binaries, in CSR form over the members: the indicators
  // bounding column j are vub_indicators[vub_offsets[j] .. vub_offsets[j+1]).
  std::vector<i_t> vub_offsets;
  std::vector<i_t> vub_indicators;

  i_t max_row_size = 0;
};

template <typename i_t, typename f_t>
indicator_strengthening_t<i_t, f_t>::indicator_strengthening_t(const papilo::Problem<f_t>& problem)
  : problem(problem)
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& row_sizes         = constraint_matrix.getRowSizes();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;
  const auto& upper_bounds      = domains.upper_bounds;

  const i_t num_rows = constraint_matrix.getNRows();
  const i_t num_cols = problem.getNCols();

  is_binary.resize(num_cols);
  for (i_t col = 0; col < num_cols; ++col) {
    is_binary[col] = col_flags[col].test(papilo::ColFlag::kIntegral) &&
                     !col_flags[col].test(papilo::ColFlag::kLbInf, papilo::ColFlag::kUbInf) &&
                     lower_bounds[col] == 0.0 && upper_bounds[col] == 1.0;
  }

  orientation.assign(num_rows, 0);
  std::vector<std::pair<i_t, i_t>> vubs;
  vubs.reserve(std::count(row_sizes.begin(), row_sizes.end(), 2));
  for (i_t row = 0; row < num_rows; ++row) {
    max_row_size            = std::max(max_row_size, row_sizes[row]);
    const bool lhs_infinite = row_flags[row].test(papilo::RowFlag::kLhsInf);
    const bool rhs_infinite = row_flags[row].test(papilo::RowFlag::kRhsInf);
    if (lhs_infinite == rhs_infinite) { continue; }
    const i_t direction = lhs_infinite ? 1 : -1;
    orientation[row]    = direction;
    if (row_sizes[row] != 2) { continue; }
    const f_t side = direction == 1 ? rhs_values[row] : lhs_values[row];
    if (side != 0.0) { continue; }

    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const i_t* indices    = row_coefficients.getIndices();
    const f_t* values     = row_coefficients.getValues();
    if (!is_binary[indices[0]] || !is_binary[indices[1]]) { continue; }
    const f_t v0 = direction * values[0];
    const f_t v1 = direction * values[1];
    if (v0 == 1.0 && v1 == -1.0) {
      vubs.emplace_back(indices[0], indices[1]);
    } else if (v0 == -1.0 && v1 == 1.0) {
      vubs.emplace_back(indices[1], indices[0]);
    }
  }

  vub_offsets.assign(num_cols + 1, 0);
  for (const auto& vub : vubs) {
    ++vub_offsets[vub.first + 1];
  }
  for (i_t col = 0; col < num_cols; ++col) {
    vub_offsets[col + 1] += vub_offsets[col];
  }
  vub_indicators.resize(vubs.size());
  std::vector<i_t> next(vub_offsets.begin(), vub_offsets.end() - 1);
  for (const auto& [member, indicator] : vubs) {
    vub_indicators[next[member]++] = indicator;
  }
}

template <typename i_t, typename f_t>
i_t indicator_strengthening_t<i_t, f_t>::add_implied_indicator_rows(
  papilo::Vec<papilo::Triplet<f_t>>& implied_entries) const
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_sizes         = constraint_matrix.getRowSizes();

  const i_t num_rows = constraint_matrix.getNRows();
  const i_t num_cols = problem.getNCols();

  i_t num_implied = 0;

  std::vector<i_t> indicators;
  indicators.reserve(max_row_size);
  std::vector<i_t> mark(num_cols, -1);

  for (i_t row = 0; row < num_rows; ++row) {
    const i_t direction = orientation[row];
    if (direction == 0 || row_sizes[row] < 3) { continue; }
    const f_t side = direction == 1 ? rhs_values[row] : lhs_values[row];
    if (side != 0.0) { continue; }

    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const i_t len         = row_coefficients.getLength();
    const i_t* indices    = row_coefficients.getIndices();
    const f_t* values     = row_coefficients.getValues();

    const size_t num_members = len - 1;
    i_t head                 = -1;
    bool usable              = true;
    indicators.clear();
    for (i_t p = 0; p < len; ++p) {
      const i_t col = indices[p];
      const f_t v   = direction * values[p];
      if (!is_binary[col]) {
        usable = false;
        break;
      }

      if (v == 1.0) {
        usable = head < 0;
        head   = col;
      } else if (v == -1.0) {
        const bool bounded = vub_offsets[col] < vub_offsets[col + 1];
        const i_t* first   = bounded ? vub_indicators.data() + vub_offsets[col] : &col;
        const i_t* last    = bounded ? vub_indicators.data() + vub_offsets[col + 1] : &col + 1;
        for (const i_t* z = first; z != last && usable; ++z) {
          if (mark[*z] == row) { continue; }
          mark[*z] = row;
          indicators.push_back(*z);
          usable = indicators.size() < num_members;
        }
      } else {
        usable = false;
        break;
      }
    }
    if (!usable || head < 0 || mark[head] == row) { continue; }

    std::sort(indicators.begin(), indicators.end());
    const i_t implied_row = num_rows + num_implied;
    const auto split      = std::lower_bound(indicators.begin(), indicators.end(), head);
    for (auto it = indicators.begin(); it != split; ++it) {
      implied_entries.emplace_back(implied_row, *it, f_t{-1});
    }
    implied_entries.emplace_back(implied_row, head, f_t{1});
    for (auto it = split; it != indicators.end(); ++it) {
      implied_entries.emplace_back(implied_row, *it, f_t{-1});
    }
    ++num_implied;
  }

  return num_implied;
}

template <typename i_t, typename f_t>
i_t indicator_strengthening_t<i_t, f_t>::lift_capacity_rows(
  papilo::Vec<papilo::Triplet<f_t>>& lifted_entries) const
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;

  const i_t num_rows = constraint_matrix.getNRows();

  i_t num_lifted = 0;

  std::vector<i_t> members;
  members.reserve(max_row_size);

  for (i_t row = 0; row < num_rows; ++row) {
    const i_t direction = orientation[row];
    if (direction == 0) { continue; }
    const f_t capacity = direction * (direction == 1 ? rhs_values[row] : lhs_values[row]);
    if (capacity <= 0.0) { continue; }

    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const i_t len         = row_coefficients.getLength();
    const i_t* indices    = row_coefficients.getIndices();
    const f_t* values     = row_coefficients.getValues();

    members.clear();
    i_t pivot   = -1;
    bool usable = true;
    for (i_t p = 0; p < len && usable; ++p) {
      const i_t col = indices[p];
      const f_t v   = direction * values[p];
      if (col_flags[col].test(papilo::ColFlag::kIntegral)) {
        const i_t num_vubs = vub_offsets[col + 1] - vub_offsets[col];
        usable             = is_binary[col] && v == 1.0 && num_vubs > 0;
        if (!usable) { continue; }
        members.push_back(col);
        if (pivot < 0 || num_vubs < vub_offsets[pivot + 1] - vub_offsets[pivot]) { pivot = col; }
      } else {
        usable =
          v < 0.0 && !col_flags[col].test(papilo::ColFlag::kLbInf) && lower_bounds[col] >= 0.0;
      }
    }
    if (!usable || members.size() < 2) { continue; }
    const f_t n_members = members.size();
    if (capacity >= n_members) { continue; }

    i_t indicator = -1;
    for (i_t p = vub_offsets[pivot]; p < vub_offsets[pivot + 1] && indicator < 0; ++p) {
      const i_t z = vub_indicators[p];
      bool shared = true;
      for (size_t k = 0; k < members.size() && shared; ++k) {
        if (members[k] == pivot) { continue; }
        const auto span_begin = vub_indicators.begin() + vub_offsets[members[k]];
        const auto span_end   = vub_indicators.begin() + vub_offsets[members[k] + 1];
        shared                = std::find(span_begin, span_end, z) != span_end;
      }
      if (shared) { indicator = z; }
    }
    if (indicator < 0) { continue; }

    lifted_entries.emplace_back(row, indicator, direction * -capacity);
    ++num_lifted;
  }

  return num_lifted;
}

}  // namespace

template <typename i_t, typename f_t>
void strengthen_indicators(papilo::Problem<f_t>& problem)
{
  const indicator_strengthening_t<i_t, f_t> strengthening(problem);
  if (strengthening.num_vubs() == 0) { return; }

  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& row_sizes         = constraint_matrix.getRowSizes();

  const i_t num_rows = constraint_matrix.getNRows();
  const i_t num_cols = problem.getNCols();

  i_t max_implied_entries = 0;
  i_t max_lifted_rows     = 0;
  for (i_t row = 0; row < num_rows; ++row) {
    const i_t direction = strengthening.row_orientation(row);
    if (direction == 0) { continue; }
    const f_t capacity = direction * (direction == 1 ? rhs_values[row] : lhs_values[row]);
    if (capacity == 0.0 && row_sizes[row] >= 3) {
      max_implied_entries += row_sizes[row] - 1;
    } else if (capacity > 0.0) {
      ++max_lifted_rows;
    }
  }

  papilo::Vec<papilo::Triplet<f_t>> implied_entries;
  papilo::Vec<papilo::Triplet<f_t>> lifted_entries;
  implied_entries.reserve(max_implied_entries);
  lifted_entries.reserve(max_lifted_rows);
  const i_t num_implied = strengthening.add_implied_indicator_rows(implied_entries);
  const i_t num_lifted  = strengthening.lift_capacity_rows(lifted_entries);
  if (num_implied == 0 && num_lifted == 0) { return; }

  papilo::Vec<papilo::Triplet<f_t>> entries;
  entries.reserve(constraint_matrix.getNnz() + lifted_entries.size() + implied_entries.size());
  auto lifted = lifted_entries.begin();
  for (i_t row = 0; row < num_rows; ++row) {
    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const i_t len         = row_coefficients.getLength();
    const i_t* indices    = row_coefficients.getIndices();
    const f_t* values     = row_coefficients.getValues();
    i_t p                 = 0;
    if (lifted != lifted_entries.end() && std::get<0>(*lifted) == row) {
      for (; p < len && indices[p] < std::get<1>(*lifted); ++p) {
        entries.emplace_back(row, indices[p], values[p]);
      }
      entries.push_back(*lifted);
      ++lifted;
    }
    for (; p < len; ++p) {
      entries.emplace_back(row, indices[p], values[p]);
    }
  }
  entries.insert(entries.end(), implied_entries.begin(), implied_entries.end());

  papilo::Vec<f_t> lhs;
  papilo::Vec<f_t> rhs;
  papilo::Vec<papilo::RowFlags> flags;
  lhs.reserve(num_rows + num_implied);
  rhs.reserve(num_rows + num_implied);
  flags.reserve(num_rows + num_implied);
  lhs.assign(lhs_values.begin(), lhs_values.end());
  rhs.assign(rhs_values.begin(), rhs_values.end());
  flags.assign(row_flags.begin(), row_flags.end());
  for (const auto& entry : lifted_entries) {
    const i_t row = std::get<0>(entry);
    if (strengthening.row_orientation(row) == 1) {
      rhs[row] = 0.0;
    } else {
      lhs[row] = 0.0;
    }
  }
  papilo::RowFlags implied_flags;
  implied_flags.set(papilo::RowFlag::kLhsInf);
  lhs.resize(num_rows + num_implied, 0.0);
  rhs.resize(num_rows + num_implied, 0.0);
  flags.resize(num_rows + num_implied, implied_flags);

  const auto& constraint_names = problem.getConstraintNames();
  if (!constraint_names.empty()) {
    papilo::Vec<papilo::String> names;
    names.reserve(constraint_names.size() + num_implied);
    names.assign(constraint_names.begin(), constraint_names.end());
    for (i_t k = 0; k < num_implied; ++k) {
      names.push_back("implied_indicator_" + std::to_string(k));
    }
    problem.setConstraintNames(std::move(names));
  }

  papilo::SparseStorage<f_t> storage(
    std::move(entries), num_rows + num_implied, num_cols, true, 4.0, 30);
  problem.setConstraintMatrix(std::move(storage), std::move(lhs), std::move(rhs), std::move(flags));

  CUOPT_LOG_DEBUG(
    "Indicator strengthening: %d implied indicator rows added, %d capacity rows lifted over %d "
    "variable upper bounds",
    num_implied,
    num_lifted,
    strengthening.num_vubs());
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template void strengthen_indicators<int, float>(papilo::Problem<float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void strengthen_indicators<int, double>(papilo::Problem<double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
