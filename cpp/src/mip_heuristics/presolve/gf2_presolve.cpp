/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "gf2_presolve.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/macros.cuh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>

#if GF2_PRESOLVE_DEBUG
#define NOT_GF2(reason, ...)                                                  \
  do {                                                                        \
    printf("NO : Cons %d is not gf2: " reason "\n", cstr_idx, ##__VA_ARGS__); \
    goto not_valid;                                                           \
  } while (0)
#else
#define NOT_GF2(reason, ...) \
  do {                       \
    goto not_valid;          \
  } while (0)
#endif

namespace cuopt::mathematical_optimization::mip {

template <typename i_t>
static inline i_t positive_modulo(i_t i, i_t n)
{
  return (i % n + n) % n;
}

// Value the row is pinned to, if any. Papilo drops a side once the row activity makes it
// redundant, so an equality can reach a presolver as a one-sided row; conversely a one-sided row
// whose activity reaches its side exactly holds with equality at every feasible point.
template <typename f_t>
static std::optional<f_t> pinned_row_value(const papilo::RowFlags& flags,
                                           const papilo::RowActivity<f_t>& activity,
                                           f_t lhs,
                                           f_t rhs,
                                           const papilo::Num<f_t>& num)
{
  if (flags.test(papilo::RowFlag::kEquation)) { return lhs; }
  if (!flags.test(papilo::RowFlag::kRhsInf) && activity.ninfmin == 0 &&
      num.isEq(activity.min, rhs)) {
    return rhs;
  }
  if (!flags.test(papilo::RowFlag::kLhsInf) && activity.ninfmax == 0 &&
      num.isEq(activity.max, lhs)) {
    return lhs;
  }
  return std::nullopt;
}

static constexpr int GF2_WORD_BITS = 64;

// up to the mantissa bits of float, to err on the safe side
static constexpr int GF2_MAX_ROW_VALUE = 1 << std::numeric_limits<float>::digits;

static inline int gf2_nwords(int N) { return (N + GF2_WORD_BITS - 1) / GF2_WORD_BITS; }

static inline bool gf2_test_bit(const std::vector<uint64_t>& row, int col)
{
  return (row[col / GF2_WORD_BITS] >> (col % GF2_WORD_BITS)) & uint64_t{1};
}

static inline void gf2_set_bit(std::vector<uint64_t>& row, int col)
{
  row[col / GF2_WORD_BITS] |= (uint64_t{1} << (col % GF2_WORD_BITS));
}

// this is kind-of a stopgap implementation (as in practice MIPLIB2017 only contains a couple of GF2
// problems and they're small) but cuDSS could be used for this since A is likely to be sparse and
// low-bandwidth (i think?) unlikely to occur in real-world problems however. doubt it'd be worth
// the effort
gf2_status_t gf2_solve(std::vector<std::vector<uint64_t>>& A,
                       int n_cols,
                       std::vector<int>& b,
                       std::vector<int>& x,
                       std::vector<uint8_t>& determined)
{
  const int m      = (int)A.size();
  const int n      = n_cols;
  const int nwords = gf2_nwords(n);
  cuopt_assert(m > 0, "");
  cuopt_assert(n >= 0, "");
  cuopt_assert((int)b.size() == m, "");
  cuopt_assert((int)A[0].size() == nwords, "");

  // pivot_row_of_col[c] = row holding the pivot for column c, or -1 if free
  std::vector<int> pivot_row_of_col(n, -1);
  int next_pivot_row = 0;

  for (int col = 0; col < n; col++) {
    int pivot = -1;
    for (int r = next_pivot_row; r < m; r++) {
      if (gf2_test_bit(A[r], col)) {
        pivot = r;
        break;
      }
    }
    if (pivot == -1) continue;  // free column

    if (pivot != next_pivot_row) {
      std::swap(A[next_pivot_row], A[pivot]);
      std::swap(b[next_pivot_row], b[pivot]);
    }

    // Eliminate column from all other rows (RREF)
    for (int r = 0; r < m; r++) {
      if (r != next_pivot_row && gf2_test_bit(A[r], col)) {
        for (int w = 0; w < nwords; w++)
          A[r][w] ^= A[next_pivot_row][w];
        b[r] ^= b[next_pivot_row];
      }
    }

    pivot_row_of_col[col] = next_pivot_row;
    next_pivot_row++;
  }

  const int rank = next_pivot_row;
  for (int r = rank; r < m; r++) {
    for (int w = 0; w < nwords; w++) {
      cuopt_assert(A[r][w] == 0, "RREF unused row must be zero");
    }
    if (b[r]) return gf2_status_t::Infeasible;
  }

  std::vector<uint64_t> free_mask(nwords, 0);
  for (int c = 0; c < n; c++) {
    if (pivot_row_of_col[c] == -1) gf2_set_bit(free_mask, c);
  }

  determined.assign(n, 0);
  x.assign(n, 0);

  for (int col = 0; col < n; col++) {
    int row = pivot_row_of_col[col];
    if (row == -1) continue;  // free: x=0, determined=false

    bool has_free_support = false;
    for (int w = 0; w < nwords; w++) {
      if (A[row][w] & free_mask[w]) {
        has_free_support = true;
        break;
      }
    }
    // Particular solution with free vars = 0: x[pivot] = b[row]
    x[col]          = b[row];
    determined[col] = !has_free_support;
  }

  return gf2_status_t::Feasible;
}

template <typename f_t>
papilo::PresolveStatus GF2Presolve<f_t>::execute(const papilo::Problem<f_t>& problem,
                                                 const papilo::ProblemUpdate<f_t>& problemUpdate,
                                                 const papilo::Num<f_t>& num,
                                                 papilo::Reductions<f_t>& reductions,
                                                 const papilo::Timer& timer,
                                                 int& reason_of_infeasibility)
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;
  const auto& upper_bounds      = domains.upper_bounds;

  const auto& row_activities = problem.getRowActivities();

  const int num_rows = constraint_matrix.getNRows();
  cuopt_assert(row_activities.size() == num_rows, "row activities not initialized");

  std::unordered_map<size_t, size_t> gf2_bin_vars;
  std::unordered_map<size_t, size_t> gf2_key_vars;
  std::vector<gf2_constraint_t> gf2_constraints;

  const f_t integrality_tolerance = num.getFeasTol();

  for (int cstr_idx = 0; cstr_idx < num_rows; ++cstr_idx) {
    int key_var_idx   = -1;
    f_t key_var_coeff = 0.0;

    std::vector<std::pair<size_t, f_t>> constraint_bin_vars;

    // Check constraint coefficients
    auto row_coeff         = constraint_matrix.getRowCoefficients(cstr_idx);
    const int* row_indices = row_coeff.getIndices();
    const f_t* row_values  = row_coeff.getValues();
    const int row_length   = row_coeff.getLength();
    int rhs                = 0;

    const std::optional<f_t> row_value = pinned_row_value(row_flags[cstr_idx],
                                                          row_activities[cstr_idx],
                                                          lhs_values[cstr_idx],
                                                          rhs_values[cstr_idx],
                                                          num);

    if (row_flags[cstr_idx].test(papilo::RowFlag::kRedundant)) NOT_GF2("redundant");

    if (!row_value.has_value()) NOT_GF2("not eq");
    if (!std::isfinite(*row_value)) NOT_GF2("not finite", *row_value);
    if (!is_integer(*row_value, integrality_tolerance)) NOT_GF2("not integer", *row_value);
    if (std::abs(*row_value) > GF2_MAX_ROW_VALUE) NOT_GF2("side too large", *row_value);
    rhs = (int)std::round(*row_value);

    for (int j = 0; j < row_length; ++j) {
      if (!is_integer(row_values[j], integrality_tolerance)) {
        NOT_GF2("coeff not integer", row_values[j]);
      }

      int var_idx = row_indices[j];
      f_t coeff   = std::round(row_values[j]);

      // Check if variable is integer
      if (!col_flags[var_idx].test(papilo::ColFlag::kIntegral)) {
        NOT_GF2("not integral", var_idx);
      }

      bool is_binary = col_flags[var_idx].test(papilo::ColFlag::kLbInf) ? false
                       : col_flags[var_idx].test(papilo::ColFlag::kUbInf)
                         ? false
                         : (lower_bounds[var_idx] == 0.0 && upper_bounds[var_idx] == 1.0);

      // Check coefficient constraints
      if (is_binary && (std::abs(coeff) != 1.0 && std::abs(coeff) != 2.0)) {
        NOT_GF2("not binary", var_idx);
      }
      if (!is_binary && (std::abs(coeff) != 2.0)) { NOT_GF2("not binary", var_idx); }

      // Key variable (coefficient of 2)
      if (std::abs(coeff) == 2.0) {
        if (key_var_idx != -1) { NOT_GF2("multiple key variables", var_idx); }
        key_var_idx   = var_idx;
        key_var_coeff = coeff;
      } else {
        // Binary variable
        constraint_bin_vars.push_back({var_idx, coeff});
      }
    }

    if (key_var_idx == -1) NOT_GF2("missing key variable");

    // Commit to global maps only after the row is fully accepted
    gf2_key_vars.insert({(size_t)key_var_idx, gf2_key_vars.size()});
    for (auto [bin_var, _] : constraint_bin_vars) {
      gf2_bin_vars.insert({bin_var, gf2_bin_vars.size()});
    }

    gf2_constraints.emplace_back((size_t)cstr_idx,
                                 std::move(constraint_bin_vars),
                                 std::pair<size_t, f_t>{key_var_idx, key_var_coeff},
                                 rhs);
    continue;
  not_valid:
    continue;
  }

  // If no GF2 constraints found, return unchanged
  if (gf2_constraints.empty()) { return papilo::PresolveStatus::kUnchanged; }

  // one unique key per GF2 row. #bins may differ from #rows.
  if (gf2_key_vars.size() != gf2_constraints.size()) { return papilo::PresolveStatus::kUnchanged; }

  // Skip if that would cause computational explosion (dense GE ~ O(m * n * min(m,n)))
  if (gf2_constraints.size() > 1000 || gf2_bin_vars.size() > 1000) {
    return papilo::PresolveStatus::kUnchanged;
  }

  // Create inverse mappings
  std::unordered_map<size_t, size_t> gf2_bin_vars_invmap;
  for (const auto& [var_idx, gf2_idx] : gf2_bin_vars) {
    gf2_bin_vars_invmap.insert({gf2_idx, var_idx});
  }

  // Build binary matrix as packed uint64_t words
  const int m      = (int)gf2_constraints.size();
  const int n      = (int)gf2_bin_vars.size();
  const int nwords = gf2_nwords(n);
  std::vector<std::vector<uint64_t>> A(m, std::vector<uint64_t>(nwords, 0));
  std::vector<int> b(m);
  for (int gf2_cstr_idx = 0; gf2_cstr_idx < m; ++gf2_cstr_idx) {
    const auto& cons = gf2_constraints[gf2_cstr_idx];
    for (auto [bin_var, _] : cons.bin_vars) {
      gf2_set_bit(A[gf2_cstr_idx], (int)gf2_bin_vars[bin_var]);
    }
    b[gf2_cstr_idx] = positive_modulo(cons.rhs, 2);
  }

  std::vector<int> solution(n);
  std::vector<uint8_t> determined(n);
  gf2_status_t gf2_status = gf2_solve(A, n, b, solution, determined);
  if (gf2_status == gf2_status_t::Infeasible) { return papilo::PresolveStatus::kInfeasible; }

  std::unordered_map<size_t, f_t> fixings;

  // Fix only uniquely determined binaries
  for (int sol_idx = 0; sol_idx < n; ++sol_idx) {
    if (determined[sol_idx]) { fixings[gf2_bin_vars_invmap[sol_idx]] = solution[sol_idx]; }
  }

  // Fix key only when every binary in that constraint is uniquely determined
  for (const auto& cons : gf2_constraints) {
    bool all_bins_determined = true;
    for (auto [bin_var, _] : cons.bin_vars) {
      cuopt_assert(gf2_bin_vars.count(bin_var), "");
      if (!determined[gf2_bin_vars[bin_var]]) {
        all_bins_determined = false;
        break;
      }
    }
    if (!all_bins_determined) continue;

    auto [key_var_idx, key_var_coeff] = cons.key_var;
    f_t lhs                           = -cons.rhs;
    for (auto [bin_var, coeff] : cons.bin_vars) {
      cuopt_assert(fixings.count(bin_var), "");
      lhs += fixings[bin_var] * coeff;
    }
    const f_t key_val = std::round(-lhs / key_var_coeff);

    // Residual must be exactly 0 after rounding (rejects half-integer / inconsistent carry)
    if (!num.isEq(lhs + key_val * key_var_coeff, f_t{0})) {
      return papilo::PresolveStatus::kInfeasible;
    }
    // Dual-role: same var already fixed as a GF(2) binary
    if (fixings.count(key_var_idx) && !num.isEq(fixings[key_var_idx], key_val)) {
      return papilo::PresolveStatus::kInfeasible;
    }
    if (!col_flags[key_var_idx].test(papilo::ColFlag::kLbInf) &&
        key_val < lower_bounds[key_var_idx] - integrality_tolerance) {
      return papilo::PresolveStatus::kInfeasible;
    }
    if (!col_flags[key_var_idx].test(papilo::ColFlag::kUbInf) &&
        key_val > upper_bounds[key_var_idx] + integrality_tolerance) {
      return papilo::PresolveStatus::kInfeasible;
    }

    fixings[key_var_idx] = key_val;
  }

  // necessary because Papilo asserts on empty TransactionGuard
  if (fixings.empty()) { return papilo::PresolveStatus::kUnchanged; }

  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;
  papilo::TransactionGuard rg{reductions};
  for (const auto& [var_idx, fixing] : fixings) {
    if (num.isZero(fixing)) {
      reductions.fixCol(var_idx, 0);
    } else {
      reductions.fixCol(var_idx, fixing);
    }
    status = papilo::PresolveStatus::kReduced;
  }

  return status;
}

#define INSTANTIATE(F_TYPE) template class GF2Presolve<F_TYPE>;

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
