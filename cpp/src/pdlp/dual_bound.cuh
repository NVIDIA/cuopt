/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip_heuristics/problem/problem.cuh>
#include <utilities/type_2.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/stream_ref>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

// The sums of one bound evaluation: row and column terms with their absolute values (for the
// rounding margin), and the number of columns whose reduced cost has the wrong sign for an infinite
// bound. Reduced in a single pass so that one evaluation costs one host round trip.
template <typename i_t, typename f_t>
struct bound_terms_t {
  f_t row_sum;
  f_t row_abs;
  f_t col_sum;
  f_t col_abs;
  i_t n_dual_infeasible_columns;
};

template <typename i_t, typename f_t>
struct bound_terms_plus_t {
  __host__ __device__ bound_terms_t<i_t, f_t> operator()(const bound_terms_t<i_t, f_t>& a,
                                                         const bound_terms_t<i_t, f_t>& b) const
  {
    return {a.row_sum + b.row_sum,
            a.row_abs + b.row_abs,
            a.col_sum + b.col_sum,
            a.col_abs + b.col_abs,
            a.n_dual_infeasible_columns + b.n_dual_infeasible_columns};
  }
};

// Index k < m is row k, index k >= m is column k - m (its term and flag computed beforehand).
template <typename i_t, typename f_t>
struct bound_term_op_t {
  const f_t* y;
  const f_t* row_lb;
  const f_t* row_ub;
  const f_t* term;
  const uint8_t* dual_infeasible;
  i_t m;

  __host__ __device__ bound_terms_t<i_t, f_t> operator()(i_t k) const
  {
    if (k < m) {
      const f_t v = y[k];
      const f_t t = v > f_t(0) ? v * row_lb[k] : (v < f_t(0) ? v * row_ub[k] : f_t(0));
      return {t, t < f_t(0) ? -t : t, f_t(0), f_t(0), i_t(0)};
    }
    const f_t t = term[k - m];
    return {f_t(0), f_t(0), t, t < f_t(0) ? -t : t, static_cast<i_t>(dual_infeasible[k - m])};
  }
};

// One nonzero of a dual infeasible column, with the data of its row, gathered for the host repair.
template <typename i_t, typename f_t>
struct repair_entry_t {
  i_t row;
  f_t coeff;
  f_t row_lb;
  f_t row_ub;
  f_t y;
};

// A valid lower bound on the objective of an LP from any dual vector y, without a basis and without
// waiting for convergence (Lagrangian / Neumaier-Shcherbina bound):
//   clip y to its sign set (a row keeps a positive dual only with a finite lower side, a negative
//   one only with a finite upper side), d = c - A^T y,
//   LB = sum_i (y_i > 0 ? y_i * row_lb_i : y_i * row_ub_i)
//      + sum_j (d_j > 0 ? d_j * l_j : d_j * u_j).
// A dual infeasible column (d_j has the wrong sign for an infinite bound) makes LB = -inf. Up to
// max_dual_infeasible_columns of them are fixed by shifting the dual of one row in their column so
// that d_j becomes 0 (the row keeps a sign its finite sides allow), and the whole bound is
// recomputed, for up to max_repair_rounds rounds.
// A relative margin covers the rounding of the double-precision sums.
// The quality of the bound follows the dual residual of y, not the duality gap: a PDLP iterate far
// from its termination tolerances can already give a bound close to the LP optimum.
template <typename i_t, typename f_t>
class lagrangian_dual_bound_t {
 public:
  static constexpr i_t max_dual_infeasible_columns = 64;
  static constexpr i_t max_repair_rounds           = 3;
  static constexpr f_t relative_margin             = 1e-9;

  lagrangian_dual_bound_t(i_t n_rows, i_t n_cols, cuda::stream_ref stream)
    : y_(n_rows, stream),
      d_(n_cols, stream),
      term_(n_cols, stream),
      dual_infeasible_(n_cols, stream),
      dual_infeasible_columns_(max_dual_infeasible_columns, stream),
      terms_(stream)
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
      const i_t n_dual_infeasible_columns =
        evaluate(problem, policy, stream, row_sum, row_abs, col_sum, col_abs);
      if (n_dual_infeasible_columns == 0) {
        return row_sum + col_sum - relative_margin * (row_abs + col_abs);
      }
      if (n_dual_infeasible_columns > max_dual_infeasible_columns || round == max_repair_rounds) {
        return -inf;
      }
      // A shift can make another column of its row dual infeasible; the next round handles it.
      if (!repair(problem, n_dual_infeasible_columns, policy, stream)) { return -inf; }
    }
  }

  // Public only because CUDA extended device lambdas cannot live in private member functions.
  // Computes d = c - A^T y and the row and column terms; returns the number of dual infeasible
  // columns, whose d_j has the wrong sign for an infinite bound (they add nothing to the sums).
  i_t evaluate(const mip::problem_t<i_t, f_t>& problem,
               const rmm::exec_policy_nosync& policy,
               cuda::stream_ref stream,
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

    const f_t* c        = problem.objective_coefficients.data();
    const i_t* offsets  = problem.reverse_offsets.data();
    const i_t* rows     = problem.reverse_constraints.data();
    const f_t* coeffs   = problem.reverse_coefficients.data();
    const auto* bounds  = problem.variable_bounds.data();
    f_t* d              = d_.data();
    f_t* term           = term_.data();
    uint8_t* infeasible = dual_infeasible_.data();
    thrust::for_each(policy,
                     thrust::make_counting_iterator<i_t>(0),
                     thrust::make_counting_iterator<i_t>(n),
                     [c, offsets, rows, coeffs, bounds, y, d, term, infeasible] __device__(i_t j) {
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
                       term[j]       = t;
                       infeasible[j] = b;
                     });

    // All four sums and the count in one reduction over the m rows and n columns.
    const bound_term_op_t<i_t, f_t> term_op{y, row_lb, row_ub, term, infeasible, m};
    const bound_terms_t<i_t, f_t> zero{f_t(0), f_t(0), f_t(0), f_t(0), i_t(0)};
    size_t bytes = 0;
    RAFT_CUDA_TRY(cub::DeviceReduce::TransformReduce(nullptr,
                                                     bytes,
                                                     thrust::make_counting_iterator<i_t>(0),
                                                     terms_.data(),
                                                     m + n,
                                                     bound_terms_plus_t<i_t, f_t>{},
                                                     term_op,
                                                     zero,
                                                     stream.get()));
    temp_storage_.resize(bytes, stream);  // keeps its capacity across evaluations
    RAFT_CUDA_TRY(cub::DeviceReduce::TransformReduce(temp_storage_.data(),
                                                     bytes,
                                                     thrust::make_counting_iterator<i_t>(0),
                                                     terms_.data(),
                                                     m + n,
                                                     bound_terms_plus_t<i_t, f_t>{},
                                                     term_op,
                                                     zero,
                                                     stream.get()));
    const bound_terms_t<i_t, f_t> sums = terms_.value(stream);
    row_sum                            = sums.row_sum;
    row_abs                            = sums.row_abs;
    col_sum                            = sums.col_sum;
    col_abs                            = sums.col_abs;
    return sums.n_dual_infeasible_columns;
  }

  // One repair round. For each dual infeasible column in order, shift the dual of one row so that
  // the column's reduced cost becomes zero: among the rows whose dual may take the shift (by the
  // row's finite sides) and that no earlier column of this round has shifted, the one with the
  // smallest shift. The columns' data is gathered on the GPU and copied in bulk, the choice is made
  // on the host, and the shifts are written back in one copy. Returns whether any shift was
  // applied; whether the round fixed every dual infeasible column is seen by the next evaluation.
  bool repair(const mip::problem_t<i_t, f_t>& problem,
              i_t n_columns,
              const rmm::exec_policy_nosync& policy,
              cuda::stream_ref stream)
  {
    constexpr f_t inf = std::numeric_limits<f_t>::infinity();
    thrust::copy_if(policy,
                    thrust::make_counting_iterator<i_t>(0),
                    thrust::make_counting_iterator<i_t>(problem.n_variables),
                    dual_infeasible_.begin(),
                    dual_infeasible_columns_.begin(),
                    [] __device__(uint8_t flag) { return flag != 0; });

    // Where each column's nonzeros go in the packed arrays.
    const i_t* columns = dual_infeasible_columns_.data();
    const i_t* offsets = problem.reverse_offsets.data();
    rmm::device_uvector<i_t> lengths(n_columns, stream);
    thrust::transform(
      policy, columns, columns + n_columns, lengths.begin(), [offsets] __device__(i_t j) {
        return offsets[j + 1] - offsets[j];
      });
    std::vector<i_t> start(n_columns + 1, 0);
    raft::copy(start.data() + 1, lengths.data(), n_columns, stream);
    stream.sync();
    for (i_t b = 0; b < n_columns; ++b) {
      start[b + 1] += start[b];
    }
    const i_t n_entries = start[n_columns];

    // Gather every nonzero of those columns with its row's bounds and dual, and each column's
    // reduced cost.
    rmm::device_uvector<i_t> d_start(n_columns + 1, stream);
    raft::copy(d_start.data(), start.data(), n_columns + 1, stream);
    rmm::device_uvector<repair_entry_t<i_t, f_t>> entries(n_entries, stream);
    rmm::device_uvector<f_t> column_d(n_columns, stream);
    const i_t* rows                  = problem.reverse_constraints.data();
    const f_t* coeffs                = problem.reverse_coefficients.data();
    const f_t* row_lb                = problem.constraint_lower_bounds.data();
    const f_t* row_ub                = problem.constraint_upper_bounds.data();
    const f_t* d                     = d_.data();
    const f_t* y                     = y_.data();
    const i_t* begin                 = d_start.data();
    repair_entry_t<i_t, f_t>* packed = entries.data();
    f_t* dj_out                      = column_d.data();
    thrust::for_each(
      policy,
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(n_columns),
      [columns, offsets, rows, coeffs, row_lb, row_ub, d, y, begin, packed, dj_out] __device__(
        i_t b) {
        const i_t j = columns[b];
        dj_out[b]   = d[j];
        for (i_t k = offsets[j], p = begin[b]; k < offsets[j + 1]; ++k, ++p) {
          const i_t r = rows[k];
          packed[p]   = {r, coeffs[k], row_lb[r], row_ub[r], y[r]};
        }
      });
    std::vector<repair_entry_t<i_t, f_t>> entry(n_entries);
    std::vector<f_t> dj(n_columns);
    raft::copy(entry.data(), entries.data(), n_entries, stream);
    raft::copy(dj.data(), column_d.data(), n_columns, stream);
    stream.sync();

    // The choice, on the host. Within a round only the duals of rows already shifted change, and
    // those rows are skipped, so the gathered duals are the current ones.
    std::unordered_set<i_t> shifted_rows;
    std::vector<i_t> shift_rows;
    std::vector<f_t> shift_values;
    for (i_t b = 0; b < n_columns; ++b) {
      i_t best      = -1;
      f_t best_step = inf;
      for (i_t p = start[b]; p < start[b + 1]; ++p) {
        const auto& e = entry[p];
        if (e.coeff == f_t(0) || shifted_rows.count(e.row) > 0) { continue; }
        const f_t step     = dj[b] / e.coeff;  // makes d_j exactly zero
        const f_t new_y    = e.y + step;
        const bool lb_ok   = e.row_lb > -inf;
        const bool ub_ok   = e.row_ub < inf;
        const bool allowed = (lb_ok && ub_ok) || (new_y >= f_t(0) && lb_ok) ||
                             (new_y <= f_t(0) && ub_ok) || new_y == f_t(0);
        if (allowed && std::abs(step) < best_step) {
          best_step = std::abs(step);
          best      = p;
        }
      }
      if (best < 0) { continue; }
      shift_rows.push_back(entry[best].row);
      shift_values.push_back(entry[best].y + dj[b] / entry[best].coeff);
      shifted_rows.insert(entry[best].row);
    }
    if (shift_rows.empty()) { return false; }

    // Write the new duals back in one copy.
    const i_t n_shifts = static_cast<i_t>(shift_rows.size());
    rmm::device_uvector<i_t> d_rows(n_shifts, stream);
    rmm::device_uvector<f_t> d_values(n_shifts, stream);
    raft::copy(d_rows.data(), shift_rows.data(), n_shifts, stream);
    raft::copy(d_values.data(), shift_values.data(), n_shifts, stream);
    thrust::scatter(policy, d_values.begin(), d_values.end(), d_rows.begin(), y_.begin());
    stream.sync();  // the host vectors above are the copy sources
    return true;
  }

 private:
  rmm::device_uvector<f_t> y_;
  rmm::device_uvector<f_t> d_;
  rmm::device_uvector<f_t> term_;
  rmm::device_uvector<uint8_t> dual_infeasible_;
  rmm::device_uvector<i_t> dual_infeasible_columns_;  // the columns of the current repair round
  rmm::device_scalar<bound_terms_t<i_t, f_t>> terms_;
  rmm::device_buffer temp_storage_;
};

}  // namespace cuopt::mathematical_optimization::pdlp
