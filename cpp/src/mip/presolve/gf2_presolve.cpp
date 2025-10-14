/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "gf2_presolve.hpp"

#include <mip/mip_constants.hpp>

#include <cmath>
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

namespace cuopt::linear_programming::detail {

template <typename i_t>
static inline i_t positive_modulo(i_t i, i_t n)
{
  return (i % n + n) % n;
}

// this is kind-of a stopgap implementation (as in practice MIPLIB2017 only contains a couple of GF2
// problems and they're small) but cuDSS could be used for this since A is likely to be sparse and
// low-bandwidth (i think?) unlikely to occur in real-world problems however. doubt it'd be worth
// the effort trashes A and b, return true if solved
static bool gf2_solve(std::vector<std::vector<int>>& A, std::vector<int>& b, std::vector<int>& x)
{
  int i, j, k;
  const int N = A.size();
  for (i = 0; i < N; i++) {
    // Find pivot
    int pivot = -1;
    for (j = i; j < N; j++) {
      if (A[j][i]) {
        pivot = j;
        break;
      }
    }
    if (pivot == -1) return false;  // No solution

    // Swap current row with pivot row if needed
    if (pivot != i) {
      for (k = 0; k < N; k++) {
        int temp    = A[i][k];
        A[i][k]     = A[pivot][k];
        A[pivot][k] = temp;
      }
      int temp = b[i];
      b[i]     = b[pivot];
      b[pivot] = temp;
    }

    // Eliminate downwards
    for (j = i + 1; j < N; j++) {
      if (A[j][i]) {
        for (k = i; k < N; k++)
          A[j][k] ^= A[i][k];
        b[j] ^= b[i];
      }
    }
  }

  // Back-substitution
  for (i = N - 1; i >= 0; i--) {
    x[i] = b[i];
    for (j = i + 1; j < N; j++)
      x[i] ^= (A[i][j] & x[j]);
    if (!A[i][i] && x[i]) return false;  // No solution
  }
  return true;  // Success
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

  const int num_rows = constraint_matrix.getNRows();

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
    f_t rhs                = std::round(lhs_values[cstr_idx]);

    // Check if this is an equality constraint
    if (!num.isEq(lhs_values[cstr_idx], rhs_values[cstr_idx]))
      NOT_GF2("not eq", lhs_values[cstr_idx], rhs_values[cstr_idx]);
    if (!std::isfinite(lhs_values[cstr_idx])) NOT_GF2("not finite", lhs_values[cstr_idx]);
    if (!is_integer(lhs_values[cstr_idx], integrality_tolerance))
      NOT_GF2("not integer", lhs_values[cstr_idx]);

    // Only accept 0, 1, -1 as rhs
    if (rhs != 0.0 && rhs != 1.0 && rhs != -1.0) NOT_GF2("not 0, 1, -1", rhs);

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
        gf2_key_vars.insert({var_idx, gf2_key_vars.size()});
      } else {
        // Binary variable
        constraint_bin_vars.push_back({var_idx, coeff});
        gf2_bin_vars.insert({var_idx, gf2_bin_vars.size()});
      }
    }

    if (key_var_idx == -1) NOT_GF2("missing key variable");

    gf2_constraints.emplace_back((size_t)cstr_idx,
                                 std::move(constraint_bin_vars),
                                 std::pair<size_t, f_t>{key_var_idx, key_var_coeff},
                                 positive_modulo((int)rhs, 2));
    continue;
  not_valid:
    continue;
  }

  // If no GF2 constraints found, return unchanged
  if (gf2_constraints.empty()) { return papilo::PresolveStatus::kUnchanged; }

  // Skip if that would cause computational explosion (O(n^3) with simple gaussian elimination)
  if (gf2_constraints.size() > 1000) { return papilo::PresolveStatus::kUnchanged; }

  // Validate structure
  if (gf2_key_vars.size() != gf2_constraints.size() ||
      gf2_bin_vars.size() != gf2_constraints.size()) {
    return papilo::PresolveStatus::kUnchanged;
  }

  // Create inverse mappings
  std::unordered_map<size_t, size_t> gf2_bin_vars_invmap;
  for (const auto& [var_idx, gf2_idx] : gf2_bin_vars) {
    gf2_bin_vars_invmap.insert({gf2_idx, var_idx});
  }

  // Build binary matrix
  // Could be a flat vector but. oh well. in practice N is small
  std::vector<std::vector<int>> A(gf2_constraints.size(),
                                  std::vector<int>(gf2_constraints.size(), 0));
  std::vector<int> b(gf2_constraints.size());
  for (const auto& cons : gf2_constraints) {
    for (auto [bin_var, _] : cons.bin_vars) {
      A[cons.cstr_idx][gf2_bin_vars[bin_var]] = 1;
    }
    b[cons.cstr_idx] = cons.rhs;
  }

  std::vector<int> solution(gf2_constraints.size());
  bool feasible = gf2_solve(A, b, solution);
  if (!feasible) { return papilo::PresolveStatus::kInfeasible; }

  std::unordered_map<size_t, f_t> fixings;
  // Fix binary variables
  for (size_t sol_idx = 0; sol_idx < gf2_constraints.size(); ++sol_idx) {
    fixings[gf2_bin_vars_invmap[sol_idx]] = solution[sol_idx];
  }

  // Compute fixings for key variables by solving for the constraint
  for (const auto& cons : gf2_constraints) {
    auto [key_var_idx, key_var_coeff] = cons.key_var;
    f_t constraint_rhs                = lhs_values[cons.cstr_idx];  // equality constraint
    f_t lhs                           = -constraint_rhs;
    for (auto [bin_var, coeff] : cons.bin_vars) {
      lhs += fixings[bin_var] * coeff;
    }
    fixings[key_var_idx] = std::round(-lhs / key_var_coeff);
  }

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

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
