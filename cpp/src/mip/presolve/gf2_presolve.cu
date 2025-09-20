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

#include <mip/presolve/gf2_presolve.cuh>

#include <mip/mip_constants.hpp>
#include <mip/presolve/bounds_update_helpers.cuh>

#include <unordered_set>

namespace cuopt::linear_programming::detail {

// Inspired from
// https://github.com/snowberryfield/printemps/blob/0dc1f9d2b78a667451c11d8ab156861ce54e9fb7/printemps/model_component/constraint.h
// (MIT License)

#define NOT_GF2(reason, ...)                                                  \
  {                                                                           \
    printf("NO : Cons %d is not gf2: " reason "\n", cstr_idx, ##__VA_ARGS__); \
    goto not_valid;                                                           \
  }

class BinaryMatrix {
 private:
  std::vector<std::vector<int>> m_rows;

 public:
  /*************************************************************************/
  BinaryMatrix(void) { this->initialize(); }

  /*************************************************************************/
  BinaryMatrix(const int a_NUMBER_OF_ROWS, const int a_NUMBER_OF_COLUMNS)
  {
    this->setup(a_NUMBER_OF_ROWS, a_NUMBER_OF_COLUMNS);
  }

  /*************************************************************************/
  inline void initialize(void) { m_rows.clear(); }

  /*************************************************************************/
  inline void setup(const int a_NUMBER_OF_ROWS, const int a_NUMBER_OF_COLUMNS)
  {
    m_rows.resize(a_NUMBER_OF_ROWS, std::vector<int>(a_NUMBER_OF_COLUMNS, 0));
  }

  /*************************************************************************/
  inline int number_of_rows(void) const { return m_rows.size(); }

  /*************************************************************************/
  inline int number_of_columns(void) const { return m_rows.front().size(); }

  /*************************************************************************/
  inline std::vector<int>& operator[](const int a_ROW) { return m_rows[a_ROW]; }

  /*************************************************************************/
  inline std::vector<int> const& operator[](const int a_ROW) const { return m_rows[a_ROW]; }

  /*************************************************************************/
  inline void print(void) const
  {
    const int NUMBER_OF_ROWS    = this->number_of_rows();
    const int NUMBER_OF_COLUMNS = this->number_of_columns();
    for (auto i = 0; i < NUMBER_OF_ROWS; i++) {
      for (auto j = 0; j < NUMBER_OF_COLUMNS; j++) {
        std::cout << m_rows[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  /*************************************************************************/
  inline std::pair<BinaryMatrix, int> inverse_and_rank(void) const
  {
    const int SIZE = this->number_of_rows();
    BinaryMatrix A = *this;
    BinaryMatrix B(SIZE, SIZE);
    int rank = 0;

    for (auto i = 0; i < SIZE; i++) {
      B[i][i] = 1;
    }

    for (auto j = 0; j < SIZE; j++) {
      int row = -1;
      for (auto i = rank; i < SIZE; i++) {
        if (A[i][j] == 1) {
          row = i;
          break;
        }
      }
      if (row == -1) { continue; }

      if (row != rank) {
        swap(A[row], A[rank]);
        swap(B[row], B[rank]);
      }

      for (auto i = rank + 1; i < SIZE; i++) {
        if (A[i][j] == 1) {
          for (auto k = 0; k < SIZE; k++) {
            A[i][k] = (A[i][k] + A[rank][k]) & 1;
            B[i][k] = (B[i][k] + B[rank][k]) & 1;
          }
        }
      }
      rank++;
    }

    if (rank == SIZE) {
      for (auto j = SIZE - 1; j > 0; j--) {
        for (auto i = j - 1; i >= 0; i--) {
          if (A[i][j] == 1) {
            for (auto k = 0; k < SIZE; k++) {
              A[i][k] = (A[j][k] + A[i][k]) & 1;
              B[i][k] = (B[j][k] + B[i][k]) & 1;
            }
          }
        }
      }
    }

    return {B, rank};
  }

  /*************************************************************************/
  inline std::vector<int> dot(const std::vector<int>& a_VECTOR) const
  {
    std::vector<int> result(m_rows.size(), 0);
    const int NUMBER_OF_ROWS    = this->number_of_rows();
    const int NUMBER_OF_COLUMNS = this->number_of_columns();
    for (auto i = 0; i < NUMBER_OF_ROWS; i++) {
      for (auto j = 0; j < NUMBER_OF_COLUMNS; j++) {
        result[i] += m_rows[i][j] * a_VECTOR[j];
      }
      result[i] &= 1;
    }
    return result;
  }

  /*************************************************************************/
  inline BinaryMatrix dot(const BinaryMatrix& a_MATRIX) const
  {
    const int NUMBER_OF_ROWS           = this->number_of_rows();
    const int NUMBER_OF_COLUMNS        = this->number_of_columns();
    const int RESULT_NUMBER_OF_COLUMNS = a_MATRIX.number_of_columns();

    BinaryMatrix result(NUMBER_OF_ROWS, RESULT_NUMBER_OF_COLUMNS);

    for (auto i = 0; i < NUMBER_OF_ROWS; i++) {
      for (auto j = 0; j < RESULT_NUMBER_OF_COLUMNS; j++) {
        for (auto k = 0; k < NUMBER_OF_COLUMNS; k++) {
          result[i][j] += m_rows[i][k] * a_MATRIX[k][j];
        }
        result[i][j] &= 1;
      }
    }
    return result;
  }

  /*************************************************************************/
  inline BinaryMatrix reachability(void) const
  {
    auto reachability = *this;
    const int SIZE    = m_rows.size();

    std::vector<std::unordered_set<int>> nonzeros(SIZE);
    for (auto i = 0; i < SIZE; i++) {
      for (auto j = 0; j < SIZE; j++) {
        if (reachability[i][j] > 0) { nonzeros[i].insert(j); }
      }
    }

    for (auto l = 0; l < SIZE; l++) {
      bool is_updated = false;
      for (auto i = 0; i < SIZE; i++) {
        for (auto j = 0; j < SIZE; j++) {
          if (reachability[i][j] > 0) { continue; }

          for (auto&& k : nonzeros[i]) {
            if (reachability[k][j]) {
              reachability[i][j] = 1;
              is_updated         = true;
              break;
            }
          }
        }
      }
      if (!is_updated) { break; }
    }

    return reachability;
  }

  /*************************************************************************/
  inline static BinaryMatrix identity(const int a_SIZE)
  {
    auto identity = BinaryMatrix(a_SIZE, a_SIZE);
    for (auto i = 0; i < a_SIZE; i++) {
      identity[i][i] = 1;
    }
    return identity;
  }
};

struct gf2_constraint_t {
  size_t cstr_idx;
  std::vector<std::pair<size_t, double>> bin_vars;
  std::pair<size_t, double> key_var;
  size_t rhs;  // 0 or 1

  gf2_constraint_t() = default;
  gf2_constraint_t(size_t cstr_idx,
                   std::vector<std::pair<size_t, double>> bin_vars,
                   std::pair<size_t, double> key_var,
                   size_t rhs)
    : cstr_idx(cstr_idx), bin_vars(std::move(bin_vars)), key_var(key_var), rhs(rhs)
  {
  }
  gf2_constraint_t(const gf2_constraint_t& other)                = default;
  gf2_constraint_t(gf2_constraint_t&& other) noexcept            = default;
  gf2_constraint_t& operator=(const gf2_constraint_t& other)     = default;
  gf2_constraint_t& operator=(gf2_constraint_t&& other) noexcept = default;
};

// TODO: implement fully on GPU + cuDSS?
template <typename i_t, typename f_t>
void gf2_presolve(problem_t<i_t, f_t>& problem)
{
  auto h_constraint_lower_bounds = cuopt::host_copy(problem.constraint_lower_bounds);
  auto h_constraint_upper_bounds = cuopt::host_copy(problem.constraint_upper_bounds);
  auto h_coefficients            = cuopt::host_copy(problem.coefficients);
  auto h_variables               = cuopt::host_copy(problem.variables);
  auto h_is_binary_variable      = cuopt::host_copy(problem.is_binary_variable);
  auto h_variable_bounds         = cuopt::host_copy(problem.variable_bounds);
  auto h_offsets                 = cuopt::host_copy(problem.offsets);
  auto h_variable_types          = cuopt::host_copy(problem.variable_types);

  // maps problem:var_idx -> gf2matrix:bin/key_idx
  std::unordered_map<size_t, size_t> gf2_bin_vars;
  std::unordered_map<size_t, size_t> gf2_key_vars;

  std::vector<gf2_constraint_t> gf2_constraints;

  for (i_t cstr_idx = 0; cstr_idx < problem.n_constraints; ++cstr_idx) {
    i_t key_var_idx              = -1;
    f_t key_var_coeff            = 0.0;
    f_t key_var_lb               = 0.0;
    f_t key_var_ub               = 0.0;
    f_t min_activity_without_key = 0.0;
    f_t max_activity_without_key = 0.0;

    std::vector<std::pair<size_t, double>> constraint_bin_vars;

    //__asm__ __volatile__("int3");

    f_t rhs = round(h_constraint_lower_bounds[cstr_idx]);

    // needs to be an equality constraint
    if (h_constraint_lower_bounds[cstr_idx] != h_constraint_upper_bounds[cstr_idx])
      NOT_GF2("not an equality constraint");
    if (!isfinite(h_constraint_lower_bounds[cstr_idx])) NOT_GF2("not finite");
    if (!is_integer(h_constraint_lower_bounds[cstr_idx], problem.tolerances.integrality_tolerance))
      NOT_GF2("rhs not integer %f", h_constraint_lower_bounds[cstr_idx]);

    // only accept 0, 1, -1 as rhs
    if (rhs != 0.0 && rhs != 1.0 && rhs != -1.0) NOT_GF2("invalid rhs %f", rhs);

    // check if nnzs match the pattern of a gf2 constraint
    for (i_t j = h_offsets[cstr_idx]; j < h_offsets[cstr_idx + 1]; ++j) {
      if (!is_integer(h_coefficients[j], problem.tolerances.integrality_tolerance))
        NOT_GF2("coeff not integer (%d,%f)", j, h_coefficients[j]);
      i_t var_idx = h_variables[j];
      f_t coeff   = round(h_coefficients[j]);
      if (h_variable_types[var_idx] != var_t::INTEGER)
        NOT_GF2("var not integer (%d,%f)", var_idx, h_coefficients[j]);
      bool is_binary = h_is_binary_variable[var_idx];

      // only the key variable can have a coefficient of 2
      if (is_binary && (abs(coeff) != 1.0 && abs(coeff) != 2.0))
        NOT_GF2("invalid coef bin (%d,%f)", var_idx, coeff);
      if (!is_binary && (abs(coeff) != 2.0))
        NOT_GF2("invalid coef non-bin (%d,%f)", var_idx, coeff);

      // key var
      if (abs(coeff) == 2.0) {
        // can only be one
        if (key_var_idx != -1) NOT_GF2("multiple key variables");
        key_var_idx   = var_idx;
        key_var_coeff = coeff;

        gf2_key_vars.insert({var_idx, gf2_key_vars.size()});
      } else {
        min_activity_without_key += min_act_of_var(
          coeff, get_lower(h_variable_bounds[var_idx]), get_upper(h_variable_bounds[var_idx]));
        max_activity_without_key += max_act_of_var(
          coeff, get_lower(h_variable_bounds[var_idx]), get_upper(h_variable_bounds[var_idx]));

        constraint_bin_vars.push_back({var_idx, (double)coeff});
        gf2_bin_vars.insert({var_idx, gf2_bin_vars.size()});
      }
    }

    // no key var found
    if (key_var_idx == -1) NOT_GF2("no key variable");

    if (key_var_coeff > 0) {
      std::swap(min_activity_without_key, max_activity_without_key);
      min_activity_without_key *= -1;
      max_activity_without_key *= -1;
    }
    cuopt_assert(min_activity_without_key <= max_activity_without_key, "invalid activities?");

    key_var_lb = get_lower(h_variable_bounds[key_var_idx]);
    key_var_ub = get_upper(h_variable_bounds[key_var_idx]);
    if (isfinite(key_var_lb) && key_var_lb > ceil(min_activity_without_key / 2.0))
      NOT_GF2("invalid key variable bounds lower ((%f,%f), %f)",
              key_var_lb,
              key_var_ub,
              min_activity_without_key);
    if (isfinite(key_var_ub) && key_var_ub < floor(max_activity_without_key / 2.0))
      NOT_GF2("invalid key variable bounds upper ((%f,%f), %f)",
              key_var_lb,
              key_var_ub,
              max_activity_without_key);

    printf("YES: Cons %d is gf2\n", cstr_idx);
    gf2_constraints.emplace_back((size_t)cstr_idx,
                                 std::move(constraint_bin_vars),
                                 std::pair<size_t, double>{key_var_idx, (double)key_var_coeff},
                                 ((size_t)rhs) % 2);
    continue;

  not_valid:
    continue;
  }

  if (gf2_key_vars.size() != gf2_constraints.size()) {
    printf(
      "invalid key variable count: %d/%d", (int)gf2_key_vars.size(), (int)gf2_constraints.size());
    exit(0);
  }
  if (gf2_bin_vars.size() != gf2_constraints.size()) {
    printf("invalid binary variable count: %d/%d",
           (int)gf2_bin_vars.size(),
           (int)gf2_constraints.size());
    exit(0);
  }

  printf("Valid GF2 structure!\n");

  // maps gf2:bin/key_idx -> problem:var_idx
  std::unordered_map<size_t, size_t> gf2_bin_vars_invmap;
  std::unordered_map<size_t, size_t> gf2_key_vars_invmap;
  for (const auto& [var_idx, gf2_idx] : gf2_bin_vars) {
    cuopt_assert(gf2_bin_vars_invmap.count(gf2_idx) == 0, "not a bijection??");
    gf2_bin_vars_invmap.insert({gf2_idx, var_idx});
  }
  for (const auto& [var_idx, gf2_idx] : gf2_key_vars) {
    cuopt_assert(gf2_key_vars_invmap.count(gf2_idx) == 0, "not a bijection??");
    gf2_key_vars_invmap.insert({gf2_idx, var_idx});
  }

  BinaryMatrix A(gf2_constraints.size(), gf2_constraints.size());
  std::vector<i_t> b(gf2_constraints.size());
  for (const auto& cons : gf2_constraints) {
    for (auto [bin_var, _] : cons.bin_vars) {
      cuopt_assert(gf2_bin_vars.count(bin_var) == 1, "");
      A[cons.cstr_idx][gf2_bin_vars[bin_var]] = 1;
    }
    b[cons.cstr_idx] = cons.rhs;
  }

  auto [inverse, rank] = A.inverse_and_rank();
  if (rank != (int)gf2_constraints.size()) {
    printf("non invertible\n");
    exit(0);
  }

  auto solution = inverse.dot(b);

  std::unordered_map<size_t, f_t> fixings;

  for (size_t sol_idx = 0; sol_idx < gf2_constraints.size(); ++sol_idx) {
    fixings[gf2_bin_vars_invmap[sol_idx]] = solution[sol_idx];
  }

  // compute fixings for the key variables by solving the corresponding constraint
  for (const auto& cons : gf2_constraints) {
    auto [key_var_idx, key_var_coeff] = cons.key_var;
    f_t rhs                           = h_constraint_lower_bounds[cons.cstr_idx];  // eq constraint
    f_t lhs                           = 0.0;
    for (auto [bin_var, coeff] : cons.bin_vars) {
      lhs += fixings[bin_var] * coeff;
    }
    lhs -= rhs;

    cuopt_assert(fixings.count(key_var_idx) == 0, "key var unexpectedly already fixed");
    cuopt_assert(key_var_coeff != 0, "key var coeff is 0");
    fixings[key_var_idx] = round(-lhs / key_var_coeff);
  }

  printf("Fixings:\n");
  for (const auto& [var_idx, fixing] : fixings) {
    if (fixing != 0) printf("%s %d\n", problem.var_names[var_idx].c_str(), (int)round(fixing));
  }

  exit(0);
}

#define INSTANTIATE(F_TYPE) \
  template void gf2_presolve<int, F_TYPE>(problem_t<int, F_TYPE> & problem);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
