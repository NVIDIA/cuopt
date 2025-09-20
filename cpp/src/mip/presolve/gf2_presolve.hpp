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

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#include <papilo/Config.hpp>
#include <papilo/core/PresolveMethod.hpp>
#include <papilo/core/Problem.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#pragma GCC diagnostic pop

namespace cuopt::linear_programming::detail {

template <typename f_t>
class GF2Presolve : public papilo::PresolveMethod<f_t> {
 public:
  GF2Presolve() : papilo::PresolveMethod<f_t>()
  {
    this->setName("gf2presolve");
    this->setType(papilo::PresolverType::kIntegralCols);
    this->setTiming(papilo::PresolverTiming::kMedium);
  }

  papilo::PresolveStatus execute(const papilo::Problem<f_t>& problem,
                                 const papilo::ProblemUpdate<f_t>& problemUpdate,
                                 const papilo::Num<f_t>& num,
                                 papilo::Reductions<f_t>& reductions,
                                 const papilo::Timer& timer,
                                 int& reason_of_infeasibility) override;

 private:
  struct gf2_constraint_t {
    size_t cstr_idx;
    std::vector<std::pair<size_t, f_t>> bin_vars;
    std::pair<size_t, f_t> key_var;
    size_t rhs;  // 0 or 1

    gf2_constraint_t() = default;
    gf2_constraint_t(size_t cstr_idx,
                     std::vector<std::pair<size_t, f_t>> bin_vars,
                     std::pair<size_t, f_t> key_var,
                     size_t rhs)
      : cstr_idx(cstr_idx), bin_vars(std::move(bin_vars)), key_var(key_var), rhs(rhs)
    {
    }
    gf2_constraint_t(const gf2_constraint_t& other)                = default;
    gf2_constraint_t(gf2_constraint_t&& other) noexcept            = default;
    gf2_constraint_t& operator=(const gf2_constraint_t& other)     = default;
    gf2_constraint_t& operator=(gf2_constraint_t&& other) noexcept = default;
  };

  class BinaryMatrix {
   private:
    std::vector<std::vector<int>> m_rows;

   public:
    BinaryMatrix(void) { this->initialize(); }
    BinaryMatrix(const int a_NUMBER_OF_ROWS, const int a_NUMBER_OF_COLUMNS)
    {
      this->setup(a_NUMBER_OF_ROWS, a_NUMBER_OF_COLUMNS);
    }

    inline void initialize(void) { m_rows.clear(); }
    inline void setup(const int a_NUMBER_OF_ROWS, const int a_NUMBER_OF_COLUMNS)
    {
      m_rows.resize(a_NUMBER_OF_ROWS, std::vector<int>(a_NUMBER_OF_COLUMNS, 0));
    }

    inline int number_of_rows(void) const { return m_rows.size(); }
    inline int number_of_columns(void) const { return m_rows.front().size(); }
    inline std::vector<int>& operator[](const int a_ROW) { return m_rows[a_ROW]; }
    inline std::vector<int> const& operator[](const int a_ROW) const { return m_rows[a_ROW]; }

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
  };

  bool is_integer(f_t value, f_t tolerance) const
  {
    return std::abs(value - std::round(value)) <= tolerance;
  }

  f_t min_act_of_var(f_t coeff, f_t lb, f_t ub) const
  {
    if (coeff >= 0) return coeff * lb;
    return coeff * ub;
  }

  f_t max_act_of_var(f_t coeff, f_t lb, f_t ub) const
  {
    if (coeff >= 0) return coeff * ub;
    return coeff * lb;
  }
};

}  // namespace cuopt::linear_programming::detail
