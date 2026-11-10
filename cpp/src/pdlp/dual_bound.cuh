/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip_heuristics/problem/problem.cuh>
#include <utilities/type_2.hpp>

#include <cuda/stream_ref>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

// A valid lower bound on the objective of an LP from any dual vector y, without a basis and without
// waiting for convergence (Lagrangian / Neumaier-Shcherbina bound):
//   clip y to its sign set (a row keeps a positive dual only with a finite lower side, a negative
//   one only with a finite upper side), d = c - A^T y,
//   LB = sum_i (y_i > 0 ? y_i * row_lb_i : y_i * row_ub_i)
//      + sum_j (d_j > 0 ? d_j * l_j : d_j * u_j).
// A column whose d_j has the wrong sign for an infinite bound makes LB = -inf. Up to
// max_repair_columns of them are fixed by shifting the dual of one row in their column so that
// d_j becomes 0 (the row keeps a sign its finite sides allow), and the whole bound is recomputed.
// A relative margin covers the rounding of the double-precision sums.
// The quality of the bound follows the dual residual of y, not the duality gap: a PDLP iterate far
// from its termination tolerances can already give a bound close to the LP optimum.
template <typename i_t, typename f_t>
class lagrangian_dual_bound_t {
 public:
  static constexpr i_t max_repair_columns = 64;
  static constexpr i_t max_repair_rounds  = 3;
  static constexpr f_t relative_margin    = 1e-9;

  lagrangian_dual_bound_t(i_t n_rows, i_t n_cols, cuda::stream_ref stream)
    : y_(n_rows, stream), d_(n_cols, stream), term_(n_cols, stream), bad_(n_cols, stream)
  {
  }

  // Lower bound on c^T x over the feasible set of `problem` (solver space: without the objective
  // offset and scaling factor), or -inf when it cannot be made finite. `dual` holds the dual
  // iterate of the unscaled problem in the minimization sign convention.
  f_t compute(const mip::problem_t<i_t, f_t>& problem, const f_t* dual, cuda::stream_ref stream)
  {
    constexpr f_t inf = std::numeric_limits<f_t>::infinity();
    const i_t m       = problem.n_constraints;
    const i_t n       = problem.n_variables;
    if (problem.maximize || static_cast<i_t>(y_.size()) != m || static_cast<i_t>(d_.size()) != n) {
      return -inf;
    }
    const auto& policy = problem.handle_ptr->get_thrust_policy();
    const f_t* row_lb  = problem.constraint_lower_bounds.data();
    const f_t* row_ub  = problem.constraint_upper_bounds.data();
    thrust::transform(policy,
                      thrust::make_counting_iterator<i_t>(0),
                      thrust::make_counting_iterator<i_t>(m),
                      y_.begin(),
                      [dual, row_lb, row_ub] __device__(i_t i) -> f_t {
                        const f_t v = dual[i];
                        if (v > f_t(0) && !(row_lb[i] > -inf)) { return f_t(0); }
                        if (v < f_t(0) && !(row_ub[i] < inf)) { return f_t(0); }
                        return v;
                      });

    for (i_t round = 0;; ++round) {
      f_t row_sum{}, row_abs{}, col_sum{}, col_abs{};
      const i_t n_bad = evaluate(problem, policy, row_sum, row_abs, col_sum, col_abs);
      if (n_bad == 0) { return row_sum + col_sum - relative_margin * (row_abs + col_abs); }
      if (n_bad > max_repair_columns || round == max_repair_rounds) { return -inf; }
      if (!repair(problem, n_bad, policy, stream)) { return -inf; }
    }
  }

  // Public only because CUDA extended device lambdas cannot live in private member functions.
  // Computes d = c - A^T y and the row and column terms; returns the number of columns whose d_j
  // has the wrong sign for an infinite bound (those contribute nothing to the sums).
  i_t evaluate(const mip::problem_t<i_t, f_t>& problem,
               const rmm::exec_policy_nosync& policy,
               f_t& row_sum,
               f_t& row_abs,
               f_t& col_sum,
               f_t& col_abs)
  {
    constexpr f_t inf = std::numeric_limits<f_t>::infinity();
    const i_t m       = problem.n_constraints;
    const i_t n       = problem.n_variables;
    const f_t* row_lb = problem.constraint_lower_bounds.data();
    const f_t* row_ub = problem.constraint_upper_bounds.data();
    const f_t* y      = y_.data();

    auto row_term = [y, row_lb, row_ub] __device__(i_t i) -> f_t {
      const f_t v = y[i];
      return v > f_t(0) ? v * row_lb[i] : (v < f_t(0) ? v * row_ub[i] : f_t(0));
    };
    row_sum = thrust::transform_reduce(policy,
                                       thrust::make_counting_iterator<i_t>(0),
                                       thrust::make_counting_iterator<i_t>(m),
                                       row_term,
                                       f_t(0),
                                       thrust::plus<f_t>{});
    row_abs = thrust::transform_reduce(
      policy,
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(m),
      [row_term] __device__(i_t i) -> f_t {
        const f_t t = row_term(i);
        return t < f_t(0) ? -t : t;
      },
      f_t(0),
      thrust::plus<f_t>{});

    const f_t* c       = problem.objective_coefficients.data();
    const i_t* offsets = problem.reverse_offsets.data();
    const i_t* rows    = problem.reverse_constraints.data();
    const f_t* coeffs  = problem.reverse_coefficients.data();
    const auto* bounds = problem.variable_bounds.data();
    f_t* d             = d_.data();
    f_t* term          = term_.data();
    uint8_t* bad       = bad_.data();
    thrust::for_each(policy,
                     thrust::make_counting_iterator<i_t>(0),
                     thrust::make_counting_iterator<i_t>(n),
                     [c, offsets, rows, coeffs, bounds, y, d, term, bad] __device__(i_t j) {
                       f_t dj = c[j];
                       for (i_t k = offsets[j]; k < offsets[j + 1]; ++k) {
                         dj -= coeffs[k] * y[rows[k]];
                       }
                       d[j]         = dj;
                       auto bound   = bounds[j];
                       const f_t lb = get_lower(bound);
                       const f_t ub = get_upper(bound);
                       f_t t        = 0;
                       uint8_t b    = 0;
                       if (dj > f_t(0)) {
                         if (lb > -inf) {
                           t = dj * lb;
                         } else {
                           b = 1;
                         }
                       } else if (dj < f_t(0)) {
                         if (ub < inf) {
                           t = dj * ub;
                         } else {
                           b = 1;
                         }
                       }
                       term[j] = t;
                       bad[j]  = b;
                     });
    col_sum = thrust::reduce(policy, term_.begin(), term_.end(), f_t(0), thrust::plus<f_t>{});
    col_abs = thrust::transform_reduce(
      policy,
      term_.begin(),
      term_.end(),
      [] __device__(f_t t) -> f_t { return t < f_t(0) ? -t : t; },
      f_t(0),
      thrust::plus<f_t>{});
    return static_cast<i_t>(thrust::count(policy, bad_.begin(), bad_.end(), uint8_t(1)));
  }

  // Shifts one row dual per offending column so that its reduced cost becomes zero. Host side: the
  // offending columns are few (at most max_repair_columns).
  bool repair(const mip::problem_t<i_t, f_t>& problem,
              i_t n_bad,
              const rmm::exec_policy_nosync& policy,
              cuda::stream_ref stream)
  {
    constexpr f_t inf = std::numeric_limits<f_t>::infinity();
    rmm::device_uvector<i_t> bad_columns(n_bad, stream);
    thrust::copy_if(policy,
                    thrust::make_counting_iterator<i_t>(0),
                    thrust::make_counting_iterator<i_t>(problem.n_variables),
                    bad_.begin(),
                    bad_columns.begin(),
                    [] __device__(uint8_t flag) { return flag != 0; });
    std::vector<i_t> columns(n_bad);
    raft::copy(columns.data(), bad_columns.data(), n_bad, stream);
    stream.sync();

    std::unordered_set<i_t> shifted_rows;
    bool any = false;
    for (const i_t j : columns) {
      f_t dj{};
      i_t range[2];
      raft::copy(&dj, d_.data() + j, 1, stream);
      raft::copy(range, problem.reverse_offsets.data() + j, 2, stream);
      stream.sync();
      const i_t len = range[1] - range[0];
      if (len <= 0) { continue; }
      std::vector<i_t> rows(len);
      std::vector<f_t> coeffs(len), lbs(len), ubs(len), ys(len);
      raft::copy(rows.data(), problem.reverse_constraints.data() + range[0], len, stream);
      raft::copy(coeffs.data(), problem.reverse_coefficients.data() + range[0], len, stream);
      stream.sync();
      for (i_t k = 0; k < len; ++k) {
        raft::copy(&lbs[k], problem.constraint_lower_bounds.data() + rows[k], 1, stream);
        raft::copy(&ubs[k], problem.constraint_upper_bounds.data() + rows[k], 1, stream);
        raft::copy(&ys[k], y_.data() + rows[k], 1, stream);
      }
      stream.sync();
      i_t best      = -1;
      f_t best_step = inf;
      for (i_t k = 0; k < len; ++k) {
        if (coeffs[k] == f_t(0) || shifted_rows.count(rows[k]) > 0) { continue; }
        const f_t step     = dj / coeffs[k];  // makes d_j exactly zero
        const f_t new_y    = ys[k] + step;
        const bool lb_ok   = lbs[k] > -inf;
        const bool ub_ok   = ubs[k] < inf;
        const bool allowed = (lb_ok && ub_ok) || (new_y >= f_t(0) && lb_ok) ||
                             (new_y <= f_t(0) && ub_ok) || new_y == f_t(0);
        if (allowed && std::abs(step) < best_step) {
          best_step = std::abs(step);
          best      = k;
        }
      }
      if (best < 0) { continue; }
      const f_t new_y = ys[best] + dj / coeffs[best];
      raft::copy(y_.data() + rows[best], &new_y, 1, stream);
      shifted_rows.insert(rows[best]);
      any = true;
    }
    stream.sync();
    return any;
  }

 private:
  rmm::device_uvector<f_t> y_;
  rmm::device_uvector<f_t> d_;
  rmm::device_uvector<f_t> term_;
  rmm::device_uvector<uint8_t> bad_;
};

}  // namespace cuopt::mathematical_optimization::pdlp
