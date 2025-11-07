/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/phase1.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/sparse_matrix.hpp>

#include <cassert>
#include <cstdio>
#include <limits>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t create_phase1_problem(const lp_problem_t<i_t, f_t>& lp, lp_problem_t<i_t, f_t>& out)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  std::vector<i_t> positive_infeasible;
  std::vector<i_t> negative_infeasible;
  std::vector<i_t> free;
  std::vector<i_t> col_to_clear(n);

  std::vector<f_t> lower(n);
  std::vector<f_t> upper(n);
  std::vector<f_t> objective(n);
  i_t num_to_clear = 0;
  for (i_t j = 0; j < n; ++j) {
    if (lp.lower[j] == -inf && lp.upper[j] < inf) {
      positive_infeasible.push_back(j);  // infeasible if z_j > 0
      lower[j]        = -1.0;
      upper[j]        = 0.0;
      objective[j]    = lp.objective[j];
      col_to_clear[j] = 0;
    } else if (lp.lower[j] > -inf && lp.upper[j] == inf) {
      // infeasible if z_j < 0
      negative_infeasible.push_back(j);
      lower[j]        = 0.0;
      upper[j]        = 1.0;
      objective[j]    = lp.objective[j];
      col_to_clear[j] = 0;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
      // infeasible if z_j != 0
      free.push_back(j);
      lower[j]        = -1e4;
      upper[j]        = 1e4;
      objective[j]    = lp.objective[j];
      col_to_clear[j] = 0;
    } else {
      // z_j can have any sign
      lower[j]        = 0.0;
      upper[j]        = 0.0;
      objective[j]    = lp.objective[j];
      col_to_clear[j] = 1;
      num_to_clear++;
    }
  }

  out.A                  = lp.A;
  out.num_cols           = out.A.n;
  out.num_rows           = out.A.m;
  constexpr bool verbose = false;
  if (verbose) {
    printf("Pos inf %lu Neg inf %lu Free %lu\n",
           positive_infeasible.size(),
           negative_infeasible.size(),
           free.size());
    printf("Created a phase 1 problem with %d rows and %d columns and %d nonzeros\n",
           out.num_rows,
           out.num_cols,
           out.A.col_start[out.num_cols]);
  }
  out.objective    = objective;
  out.obj_constant = 0;
  out.obj_scale    = 1.0;
  out.lower        = lower;
  out.upper        = upper;
  out.rhs.resize(m);
  std::fill(out.rhs.begin(), out.rhs.end(), 0.0);
  return 0;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int create_phase1_problem<int, double>(const lp_problem_t<int, double>& lp,
                                                lp_problem_t<int, double>& out);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
