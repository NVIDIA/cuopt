/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "lp.hpp"
#include "../audit.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void eliminate_slacks(const lp_problem_t<i_t, f_t>& problem,
                      i_t n_structural,
                      csr_matrix_t<i_t, f_t>& csr_A,
                      std::vector<f_t>& row_lower,
                      std::vector<f_t>& row_upper)
{
  cuopt_assert(csr_A.m == problem.num_rows, "row count mismatch");
  cuopt_assert(csr_A.n == problem.num_cols, "column count mismatch");
  cuopt_assert(n_structural > 0, "no structural columns");
  cuopt_assert(n_structural < problem.num_cols, "no slacks to eliminate");
  cuopt_assert(problem.num_cols - n_structural <= problem.num_rows, "more slacks than rows");

  row_lower = problem.rhs;
  row_upper = problem.rhs;

  std::vector<char> row_has_slack(problem.num_rows, 0);
  for (i_t j = n_structural; j < problem.num_cols; ++j) {
    cuopt_assert(problem.A.col_length(j) == 1, "slack column is not a singleton");

    const i_t entry = problem.A.col_start[j];
    const i_t row   = problem.A.i[entry];
    const f_t alpha = problem.A.x[entry];
    cuopt_assert(std::abs(alpha) == f_t{1}, "slack coefficient is not +/-1");
    cuopt_assert(!row_has_slack[row], "row has more than one slack");
    row_has_slack[row] = 1;

    const f_t scaled_lower = alpha * problem.lower[j];
    const f_t scaled_upper = alpha * problem.upper[j];
    row_lower[row]         = problem.rhs[row] - std::max(scaled_lower, scaled_upper);
    row_upper[row]         = problem.rhs[row] - std::min(scaled_lower, scaled_upper);
    cuopt_assert(std::isfinite(row_lower[row]) || std::isfinite(row_upper[row]),
                 "eliminated row is free on both sides");
    cuopt_assert(row_lower[row] <= row_upper[row], "eliminated row has crossed bounds");
  }

  i_t out = 0;
  for (i_t row = 0; row < csr_A.m; ++row) {
    const i_t row_start  = csr_A.row_start[row];
    const i_t row_end    = csr_A.row_start[row + 1];
    csr_A.row_start[row] = out;
    for (i_t p = row_start; p < row_end; ++p) {
      if (csr_A.j[p] >= n_structural) { continue; }
      csr_A.j[out] = csr_A.j[p];
      csr_A.x[out] = csr_A.x[p];
      ++out;
    }
  }
  cuopt_assert(out == csr_A.row_start[csr_A.m] - static_cast<i_t>(problem.num_cols - n_structural),
               "slack elimination removed the wrong number of entries");

  csr_A.row_start[csr_A.m] = out;
  csr_A.j.resize(out);
  csr_A.x.resize(out);
  csr_A.nz_max = out;
  csr_A.n      = n_structural;
}

#if MIP_INSTANTIATE_FLOAT
template void eliminate_slacks<int, float>(const lp_problem_t<int, float>&,
                                           int,
                                           csr_matrix_t<int, float>&,
                                           std::vector<float>&,
                                           std::vector<float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void eliminate_slacks<int, double>(const lp_problem_t<int, double>&,
                                            int,
                                            csr_matrix_t<int, double>&,
                                            std::vector<double>&,
                                            std::vector<double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
