/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>

#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/user_problem.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <math_optimization/tic_toc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::barrier {

/** Convert MPS >= ('G') quadratic row to <= ('L') form on a working copy for SOC conversion. */
template <typename qc_t, typename f_t>
void normalize_quadratic_constraint_greater_to_less(qc_t& qc)
{
  if (qc.constraint_row_type != 'G') { return; }
  for (f_t& v : qc.linear_values) {
    v = -v;
  }
  for (f_t& v : qc.vals) {
    v = -v;
  }
  qc.rhs_value           = -qc.rhs_value;
  qc.constraint_row_type = 'L';
}

/**
 * @brief Expand QCMATRIX second-order cone (and rotated / affine variants) into the
 *        canonical slack form expected by the simplex/PDLP path: extra variables, equality
 *        rows, optional cone aliases, column permutation, and `user_problem` cone metadata.
 *
 * Preconditions: `csr_A` and `user_problem` already reflect the linear model for `n` variables
 * and original rows; this routine augments dimensions and CSR row storage in place.
 */
template <typename i_t, typename f_t>
void convert_quadratic_constraints_to_second_order_cones(
  i_t n,
  const std::vector<typename optimization_problem_interface_t<i_t, f_t>::quadratic_constraint_t>&
    qcs,
  csr_matrix_t<i_t, f_t>& csr_A,
  simplex::user_problem_t<i_t, f_t>& user_problem)
{
  cuopt_expects(!qcs.empty(),
                error_type_t::ValidationError,
                "Quadratic-constraint flag is set, but no constraints were provided");

  // Use a practical tolerance for text-parsed MPS numeric values.
  const f_t tol = std::numeric_limits<f_t>::epsilon() * 2;

  // Derive implied lower bounds from singleton inequality rows.
  // Used to check if SOC head variables have implied non-negativity from the constraint system
  // without actually modifying the variable bounds (which would add barrier terms).
  std::vector<f_t> implied_lower(n, -std::numeric_limits<f_t>::infinity());
  for (i_t i = 0; i < csr_A.m; i++) {
    const i_t row_start = csr_A.row_start[i];
    const i_t row_end   = csr_A.row_start[i + 1];
    if (row_end - row_start != 1) { continue; }
    const i_t j      = csr_A.j[row_start];
    const f_t a      = csr_A.x[row_start];
    const f_t b      = user_problem.rhs[i];
    const char sense = user_problem.row_sense[i];
    if (std::abs(a) < tol) { continue; }
    const f_t bound = b / a;
    if (sense == 'G' && a > 0) {
      implied_lower[j] = std::max(implied_lower[j], bound);
    } else if (sense == 'L' && a < 0) {
      implied_lower[j] = std::max(implied_lower[j], bound);
    }
  }

  // SOC conversion routes each quadratic constraint as follows:
  //
  // Fast path (pattern-matched, rhs = 0, no linear part in COLUMNS):
  //   1) standard Lorentz SOC — diagonal QCMATRIX with one head -s and tail diagonals +s
  //        (common s > 0; dividing the row by s normalizes without changing feasibility)
  //   2) rotated SOC — one off-diagonal cross -2*d on two heads plus tail diagonals +s
  //        (d > 0, s > 0; stored as Q cross (head0, head1, -2*d); lift uses sqrt(d/s))
  //
  // General path (LDLT check on H = Q + Q^T, must be PSD):
  //   - any QC with a nonzero linear part (COLUMNS): x^T Q x + a^T x <= alpha
  //   - any other convex QC that fails the fast-path shape checks (nonzero rhs, non-uniform
  //     diagonals, off-diagonal structure, etc.)
  //   Adds linking equalities and a standard SOC of dimension rank(Q)+2; the linear term
  //   enters the s_0 / s_{r+1} rows directly (no separate auxiliary variable).
  //
  // Post-conversion: rotated fast-path cones are lifted to standard SOC coordinates via slacks;
  // cone variables are permuted into a trailing block [linear vars | cone vars] for the barrier.
  struct rotated_soc_t {
    i_t head0{};
    i_t head1{};
    std::vector<i_t> tails{};
    bool head1_is_constant_half{false};
    /// For two-head rotated SOC: sqrt(d/s) where Q cross = -2*d and tail diagonals +s (canonical
    /// 1).
    f_t head_lift_sqrt_ratio{1};
  };

  std::vector<std::vector<i_t>> cone_vars;
  std::vector<i_t> cone_dims;
  std::vector<char> cone_is_rotated;
  std::vector<rotated_soc_t> rotated_cones;
  std::vector<char> is_cone_var(n, 0);
  cone_vars.reserve(qcs.size());
  cone_dims.reserve(qcs.size());
  cone_is_rotated.reserve(qcs.size());
  rotated_cones.reserve(qcs.size());
  std::vector<f_t> qc_soc_uniform_scale(qcs.size(), 1);

  for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
    auto qc = qcs[qc_i];
    cuopt_expects(qc.constraint_row_type == 'L' || qc.constraint_row_type == 'G',
                  error_type_t::ValidationError,
                  "Quadratic constraint '%s' ROWS type must be 'L' (<=) or 'G' (>=)",
                  qc.constraint_row_name.c_str());
    normalize_quadratic_constraint_greater_to_less<decltype(qc), f_t>(qc);
    cuopt_expects(qc.linear_values.size() == qc.linear_indices.size(),
                  error_type_t::ValidationError,
                  "Quadratic constraint '%s' linear_values and linear_indices length mismatch",
                  qc.constraint_row_name.c_str());

    const i_t q_nnz = static_cast<i_t>(qc.vals.size());
    cuopt_expects(
      qc.rows.size() == static_cast<size_t>(q_nnz) && qc.cols.size() == static_cast<size_t>(q_nnz),
      error_type_t::ValidationError,
      "Quadratic constraint '%s' Q COO row/col/value length mismatch",
      qc.constraint_row_name.c_str());
    cuopt_expects(q_nnz >= 1,
                  error_type_t::ValidationError,
                  "Quadratic constraint '%s': second-order cone must have at least 1 entry in Q "
                  "(nnz %d)",
                  qc.constraint_row_name.c_str(),
                  static_cast<int>(q_nnz));

    // Detect nonzero linear part; any such QC uses the general LDLT check.
    bool has_linear_part = false;
    if (!qc.linear_values.empty()) {
      size_t nonzero_terms = 0;
      for (size_t p = 0; p < qc.linear_values.size(); ++p) {
        const i_t idx = qc.linear_indices[p];
        const f_t v   = qc.linear_values[p];
        cuopt_expects(idx >= 0 && idx < n,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' linear index %d is outside [0, %d)",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(idx),
                      static_cast<int>(n));
        if (v > -tol && v < tol) { continue; }
        ++nonzero_terms;
      }
      cuopt_expects(nonzero_terms > 0,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' has linear section but all linear coefficients are "
                    "zero",
                    qc.constraint_row_name.c_str());
      has_linear_part = true;
    }

    // Verify Q as either:
    // - standard SOC: one diagonal -s (head), tail diagonals +s for a common s > 0,
    // - rotated SOC: one off-diagonal cross term (-2*d) on the two heads, tails +s.

    auto approx_eq_scaled = [&](f_t a, f_t b) {
      const f_t scale = std::max({f_t(1), std::abs(a), std::abs(b)});
      return std::abs(a - b) <= tol * scale;
    };

    // Sort COO by (row, col); O(nnz log nnz).
    std::vector<size_t> perm(q_nnz);
    std::iota(perm.begin(), perm.end(), size_t{0});
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
      const i_t ra = qc.rows[a];
      const i_t rb = qc.rows[b];
      if (ra != rb) { return ra < rb; }
      return qc.cols[a] < qc.cols[b];
    });

    std::vector<std::tuple<i_t, i_t, f_t>> q_entries;
    q_entries.reserve(q_nnz);
    bool has_duplicate_rows = false;
    for (size_t t = 0; t < static_cast<size_t>(q_nnz); ++t) {
      const size_t ix = perm[t];
      const i_t r     = qc.rows[ix];
      const i_t c     = qc.cols[ix];
      const f_t v     = qc.vals[ix];
      cuopt_expects(r >= 0 && r < n && c >= 0 && c < n,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' Q entry (%d,%d) outside [0,%d)",
                    qc.constraint_row_name.c_str(),
                    static_cast<int>(r),
                    static_cast<int>(c),
                    static_cast<int>(n));
      if (!q_entries.empty()) {
        const i_t prev_r = std::get<0>(q_entries.back());
        if (r == prev_r) { has_duplicate_rows = true; }
      }
      q_entries.emplace_back(r, c, v);
    }

    std::vector<std::pair<i_t, f_t>> pos_diag_rows;
    std::vector<std::pair<i_t, f_t>> neg_diag_rows;
    std::vector<std::tuple<i_t, i_t, f_t>> offdiag_entries;
    pos_diag_rows.reserve(q_entries.size());
    neg_diag_rows.reserve(1);
    offdiag_entries.reserve(4);

    bool has_near_zero_diag = false;
    for (const auto& [r, c, v] : q_entries) {
      if (r == c) {
        if (v > tol) {
          pos_diag_rows.emplace_back(r, v);
        } else if (v < -tol) {
          neg_diag_rows.emplace_back(r, v);
        } else {
          has_near_zero_diag = true;
        }
      } else {
        offdiag_entries.emplace_back(r, c, v);
      }
    }

    std::vector<i_t> tail_vars;
    tail_vars.reserve(pos_diag_rows.size());
    for (const std::pair<i_t, f_t>& pr : pos_diag_rows) {
      tail_vars.push_back(pr.first);
    }

    // Route to a fast SOC path when the quadratic constraint matches a known pattern (rhs = 0, no
    // linear part); otherwise use the general LDLT check. Eligibility must be exact: valid SOCs
    // have an indefinite raw Hessian H = Q + Q^T and would be rejected by the general PSD check.
    const bool has_nonzero_rhs = !(qc.rhs_value < tol && qc.rhs_value > -tol);
    bool has_nonuniform_diag   = false;
    if (pos_diag_rows.size() > 1) {
      const f_t first_val = pos_diag_rows[0].second;
      for (size_t k = 1; k < pos_diag_rows.size(); k++) {
        const f_t scale =
          std::max({f_t(1), std::abs(first_val), std::abs(pos_diag_rows[k].second)});
        if (std::abs(pos_diag_rows[k].second - first_val) > tol * scale) {
          has_nonuniform_diag = true;
          break;
        }
      }
    }
    const bool rotated_soc_cross_eligible = [&]() {
      if (offdiag_entries.size() != 1 || has_linear_part || !neg_diag_rows.empty()) {
        return false;
      }
      const i_t a = std::get<0>(offdiag_entries[0]);
      const i_t b = std::get<1>(offdiag_entries[0]);
      const f_t v = std::get<2>(offdiag_entries[0]);
      // Match cross_d = -v/2 > tol validated on the rotated SOC fast path below.
      return a != b && (-v / f_t(2)) > tol;
    }();
    const bool fast_soc_shape_ok = !has_linear_part && !has_duplicate_rows && !has_near_zero_diag &&
                                   !has_nonzero_rhs && !has_nonuniform_diag;
    const bool standard_soc_eligible =
      fast_soc_shape_ok && offdiag_entries.empty() && neg_diag_rows.size() == 1;
    const bool rotated_soc_eligible = fast_soc_shape_ok && rotated_soc_cross_eligible;
    const bool use_general_path     = !(standard_soc_eligible || rotated_soc_eligible);

    std::vector<i_t> cone;
    i_t cone_dim    = 0;
    char is_rotated = 0;
    i_t head        = -1;

    if (!use_general_path) {
      // Special-case rhs == 0 requirement for SOC patterns
      cuopt_expects(
        (qc.rhs_value < tol) && (qc.rhs_value > -tol),
        error_type_t::ValidationError,
        "second-order cone conversion currently requires rhs = 0 for quadratic constraints "
        "(constraint '%s' has rhs %.17g)",
        qc.constraint_row_name.c_str(),
        static_cast<double>(qc.rhs_value));

      // Infer the uniform scale s.
      f_t uniform_s        = 0;
      bool have_uniform_s  = false;
      auto note_positive_s = [&](f_t v) {
        cuopt_expects(v > tol,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s': expected strictly positive diagonal tail in Q"
                      "coefficient, got %.17g",
                      qc.constraint_row_name.c_str(),
                      static_cast<double>(v));
        if (!have_uniform_s) {
          uniform_s      = v;
          have_uniform_s = true;
        } else {
          cuopt_expects(
            approx_eq_scaled(v, uniform_s),
            error_type_t::ValidationError,
            "Quadratic constraint '%s': all positive diagonal coefficients in Q must match; got "
            "%.17g vs %.17g",
            qc.constraint_row_name.c_str(),
            static_cast<double>(v),
            static_cast<double>(uniform_s));
        }
      };

      // Collect all positive diagonal entries (tails) and validate their scale s.
      for (const std::pair<i_t, f_t>& pr : pos_diag_rows) {
        note_positive_s(pr.second);
      }

      // No +s tail diagonals means a head-only standard cone, where s comes from the lone -s head
      // entry.
      if (!have_uniform_s) {
        cuopt_expects(offdiag_entries.empty(),
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s': rotated second-order cone Q must have at least "
                      "one positive tail "
                      "diagonal +s to define the scale s",
                      qc.constraint_row_name.c_str());
        uniform_s = -neg_diag_rows[0].second;
      }

      // s must be positive.
      cuopt_expects(uniform_s > tol,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s': uniform scale s in Q must be "
                    "positive (got %.17g)",
                    qc.constraint_row_name.c_str(),
                    static_cast<double>(uniform_s));
      qc_soc_uniform_scale[qc_i] = uniform_s;

      if (offdiag_entries.empty()) {
        // Standard Lorentz SOC
        const f_t neg_v = neg_diag_rows[0].second;
        cuopt_expects(
          approx_eq_scaled(neg_v, -uniform_s),
          error_type_t::ValidationError,
          "Quadratic constraint '%s': second-order cone head diagonal in Q must be -s with the "
          "same s as "
          "positive tail diagonals; head %.17g vs -s = %.17g",
          qc.constraint_row_name.c_str(),
          static_cast<double>(neg_v),
          static_cast<double>(-uniform_s));

        // One head -s plus its tail diagonals +s.
        head = neg_diag_rows[0].first;
        // The SOC ||tail|| <= head requires head >= 0 (for the 1-element cone this is just
        // head >= 0). Check explicit bound or implied bound from singleton inequality constraints
        // so the conversion cannot tighten the original constraint.
        cuopt_expects(std::max(user_problem.lower[head], implied_lower[head]) >= 0,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s': second-order cone head variable (index %d) must "
                      "have a "
                      "non-negative lower bound for the constraint to be convex",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(head));
        cone.reserve(q_nnz);
        cone.push_back(head);
        cone.insert(cone.end(), tail_vars.begin(), tail_vars.end());
        cone_dim   = static_cast<i_t>(cone.size());
        is_rotated = 0;
      } else {
        // Rotated SOC
        const i_t a                    = std::get<0>(offdiag_entries[0]);
        const i_t b                    = std::get<1>(offdiag_entries[0]);
        const f_t cross_d              = -std::get<2>(offdiag_entries[0]) / f_t(2);
        const f_t head_lift_sqrt_ratio = std::sqrt(cross_d / uniform_s);
        cuopt_expects(std::isfinite(static_cast<double>(head_lift_sqrt_ratio)),
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s': rotated second-order cone Q head lift ratio "
                      "sqrt(d/s) is not "
                      "finite (d=%.17g, s=%.17g)",
                      qc.constraint_row_name.c_str(),
                      static_cast<double>(cross_d),
                      static_cast<double>(uniform_s));
        cuopt_expects(
          q_nnz >= 2,
          error_type_t::ValidationError,
          "Quadratic constraint '%s': rotated second-order cone in Q must have at least 1 "
          "tail entry",
          qc.constraint_row_name.c_str());

        cone.reserve(q_nnz);
        cone.push_back(a);
        cone.push_back(b);
        cone.insert(cone.end(), tail_vars.begin(), tail_vars.end());
        cone_dim   = static_cast<i_t>(cone.size());
        is_rotated = 1;
        // Rotated SOC ||tail||^2 <= 2*a*b requires a >= 0 and b >= 0.
        // Check explicit bound or implied bound from singleton inequality constraints.
        cuopt_expects(std::max(user_problem.lower[a], implied_lower[a]) >= 0,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s': rotated second-order cone head variable (index "
                      "%d) must have a "
                      "non-negative lower bound for the constraint to be convex",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(a));
        cuopt_expects(std::max(user_problem.lower[b], implied_lower[b]) >= 0,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s': rotated second-order cone head variable (index "
                      "%d) must have a "
                      "non-negative lower bound for the constraint to be convex",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(b));
        rotated_cones.push_back(rotated_soc_t{a, b, tail_vars, false, head_lift_sqrt_ratio});
      }

      cuopt_expects(
        static_cast<i_t>(tail_vars.size()) == q_nnz - 1,
        error_type_t::ValidationError,
        "Quadratic constraint '%s': second-order cone expected %d diagonal +s entries (tails) in "
        "Q, found %zu",
        qc.constraint_row_name.c_str(),
        static_cast<int>(q_nnz - 1),
        tail_vars.size());

      cone_dims.push_back(cone_dim);
      cone_vars.push_back(std::move(cone));
      cone_is_rotated.push_back(is_rotated);

    } else {
      // =========================================================================
      // General convex quadratic constraint path:
      //   x^T Q x + c^T x <= alpha
      // where Q is (possibly unsymmetric) and H = Q + Q^T must be PSD.
      // =========================================================================
      const f_t alpha = qc.rhs_value;

      // Step 1: Build H such that (1/2) x^T H x equals the quadratic form sum_k
      // v_k*x_{r_k}*x_{c_k}. For diagonal entry (r,r,v): H(r,r) += 2*v  (since (1/2)*H(r,r)*x_r^2 =
      // v*x_r^2) For off-diagonal entry (r,c,v): H(max,min) += v  (since
      // (1/2)*(H(r,c)+H(c,r))*x_r*x_c = v*x_r*x_c) Store lower triangle only in CSC.
      //
      // Use a dense accumulator indexed by the variables appearing in Q.

      // Collect distinct variable indices and build local-to-global mapping
      std::vector<i_t> var_set;
      var_set.reserve(2 * q_nnz);
      std::vector<i_t> global_to_local(n, -1);
      for (size_t t = 0; t < static_cast<size_t>(q_nnz); ++t) {
        const i_t r = qc.rows[t];
        const i_t c = qc.cols[t];
        if (global_to_local[r] == -1) {
          global_to_local[r] = static_cast<i_t>(var_set.size());
          var_set.push_back(r);
        }
        if (global_to_local[c] == -1) {
          global_to_local[c] = static_cast<i_t>(var_set.size());
          var_set.push_back(c);
        }
      }
      const i_t n_local = static_cast<i_t>(var_set.size());

      // Dense lower-triangle accumulator (column-major: H_dense[col * n_local + row] for row >=
      // col)
      std::vector<f_t> H_dense(n_local * n_local, f_t(0));
      for (size_t t = 0; t < static_cast<size_t>(q_nnz); ++t) {
        const i_t r = global_to_local[qc.rows[t]];
        const i_t c = global_to_local[qc.cols[t]];
        const f_t v = qc.vals[t];
        if (r == c) {
          H_dense[c * n_local + r] += f_t(2) * v;
        } else {
          const i_t hi = std::max(r, c);
          const i_t hj = std::min(r, c);
          H_dense[hj * n_local + hi] += v;
        }
      }

      // Gather nonzeros from dense accumulator into CSC (lower triangle, local indices)
      i_t h_nnz = 0;
      for (i_t j = 0; j < n_local; j++) {
        for (i_t i = j; i < n_local; i++) {
          if (H_dense[j * n_local + i] != f_t(0)) { h_nnz++; }
        }
      }

      csc_matrix_t<i_t, f_t> H_csc(n_local, n_local, h_nnz);
      {
        i_t p = 0;
        for (i_t j = 0; j < n_local; j++) {
          H_csc.col_start[j] = p;
          for (i_t i = j; i < n_local; i++) {
            const f_t val = H_dense[j * n_local + i];
            if (val != f_t(0)) {
              H_csc.i[p] = i;
              H_csc.x[p] = val;
              p++;
            }
          }
        }
        H_csc.col_start[n_local] = p;
      }

      // Step 2: Factorize H = P * L * D * L^T * P^T
      simplex::simplex_solver_settings_t<i_t, f_t> ldlt_settings;
      std::vector<i_t> ldlt_perm;
      csc_matrix_t<i_t, f_t> L_factor(n, n, 1);
      std::vector<f_t> D_factor;
      f_t ldlt_work  = 0;
      f_t ldlt_start = tic();

      i_t rank = simplex::right_looking_ldlt(
        H_csc, ldlt_settings, f_t(1e-12), ldlt_start, ldlt_perm, L_factor, D_factor, ldlt_work);

      // ldlt_settings uses default time_limit=inf and concurrent_halt=nullptr,
      // so only INDEFINITE_MATRIX_RETURN is possible as a negative return code.
      cuopt_expects(rank != INDEFINITE_MATRIX_RETURN,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' is non-convex (Q matrix is indefinite)",
                    qc.constraint_row_name.c_str());

      // q_nnz >= 1 implies H is nonzero, but diagonal LDLT may still return rank 0 (e.g. cross-only
      // indefinite H with zero diagonals). Reject before building a degenerate r=0 SOC lift.
      cuopt_expects(rank >= 1,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' is non-convex or could not be converted to a "
                    "second-order cone (LDLT rank %d; Q may be indefinite or have zero diagonal "
                    "with cross terms)",
                    qc.constraint_row_name.c_str(),
                    static_cast<int>(rank));

      // Step 4: Build standard SOC of dimension rank + 2.
      // New variables: y_0,...,y_{r-1}, s_0 (head), s_{r+1} (tail)
      // Linking rows:
      //   y_k - sqrt(D[k]) * (L^T P x)_k = 0   for k = 0,...,r-1
      //   s_0 + c^T x = alpha + 1/2
      //   s_{r+1} + c^T x = alpha - 1/2

      const i_t r          = rank;
      const i_t n_new_vars = r + 2;  // y_0..y_{r-1}, s_0, s_{r+1}
      const i_t n_new_rows = r + 2;
      const i_t var_base   = csr_A.n;  // first new variable index
      const i_t y_base     = var_base;
      const i_t s0_idx     = var_base + r;
      const i_t sr1_idx    = var_base + r + 1;

      // Extend problem dimensions
      const f_t pos_inf = std::numeric_limits<f_t>::infinity();
      const f_t neg_inf = -pos_inf;
      user_problem.objective.resize(var_base + n_new_vars, 0);
      user_problem.lower.resize(var_base + n_new_vars, neg_inf);
      user_problem.upper.resize(var_base + n_new_vars, pos_inf);
      user_problem.var_types.resize(var_base + n_new_vars, simplex::variable_type_t::CONTINUOUS);
      if (!user_problem.col_names.empty()) {
        user_problem.col_names.resize(var_base + n_new_vars);
        for (i_t k = 0; k < r; k++) {
          user_problem.col_names[y_base + k] =
            "_CUOPT_qc_y_" + std::to_string(qc_i) + "_" + std::to_string(k);
        }
        user_problem.col_names[s0_idx]  = "_CUOPT_qc_s0_" + std::to_string(qc_i);
        user_problem.col_names[sr1_idx] = "_CUOPT_qc_sr1_" + std::to_string(qc_i);
      }
      // s_0 (cone head) — do NOT set lower=0 here; cone membership implies s_0 >= 0
      // and the barrier solver's bound-split logic handles this automatically.

      csr_A.n = var_base + n_new_vars;
      is_cone_var.resize(var_base + n_new_vars, 0);

      // Extend row storage
      const i_t m_before = csr_A.m;
      user_problem.rhs.resize(m_before + n_new_rows);
      user_problem.row_sense.resize(m_before + n_new_rows);
      if (!user_problem.row_names.empty()) { user_problem.row_names.resize(m_before + n_new_rows); }

      sparse_vector_t<i_t, f_t> eq_row;
      eq_row.n = csr_A.n;

      // y-linking rows: y_k - sqrt(D[k]) * [row k of L^T P] * x = 0
      // L is unit lower triangular in permuted local indices.
      // Column k of L has: L(k,k)=1 at local perm[k], L(j,k) at local perm[j] for j>k.
      // (L^T P x)_k = sum_j L(j,k) * x_{var_set[perm[j]]}
      for (i_t k = 0; k < r; k++) {
        const f_t sqrt_dk = std::sqrt(D_factor[k]);
        eq_row.i.clear();
        eq_row.x.clear();
        // y_k coefficient
        eq_row.i.push_back(y_base + k);
        eq_row.x.push_back(f_t(1));
        // -sqrt(D[k]) * L(:,k) entries applied to x_{var_set[perm[j]]}
        for (i_t p = L_factor.col_start[k]; p < L_factor.col_start[k + 1]; p++) {
          const i_t j          = L_factor.i[p];  // permuted local row index
          const f_t l_val      = L_factor.x[p];
          const i_t global_var = var_set[ldlt_perm[j]];
          eq_row.i.push_back(global_var);
          eq_row.x.push_back(-sqrt_dk * l_val);
        }
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[m_before + k] = 'E';
        user_problem.rhs[m_before + k]       = 0;
        if (!user_problem.row_names.empty()) {
          user_problem.row_names[m_before + k] =
            "_CUOPT_qc_y_link_" + std::to_string(qc_i) + "_" + std::to_string(k);
        }
      }

      // s_0 linking row: s_0 + c^T x = alpha + 1/2
      {
        eq_row.i.clear();
        eq_row.x.clear();
        eq_row.i.push_back(s0_idx);
        eq_row.x.push_back(f_t(1));
        for (size_t p = 0; p < qc.linear_values.size(); ++p) {
          const f_t v = qc.linear_values[p];
          if (std::abs(v) < tol) continue;
          eq_row.i.push_back(qc.linear_indices[p]);
          eq_row.x.push_back(v);
        }
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[m_before + r] = 'E';
        user_problem.rhs[m_before + r]       = alpha + f_t(0.5);
        if (!user_problem.row_names.empty()) {
          user_problem.row_names[m_before + r] = "_CUOPT_qc_s0_link_" + std::to_string(qc_i);
        }
      }

      // s_{r+1} linking row: s_{r+1} + c^T x = alpha - 1/2
      {
        eq_row.i.clear();
        eq_row.x.clear();
        eq_row.i.push_back(sr1_idx);
        eq_row.x.push_back(f_t(1));
        for (size_t p = 0; p < qc.linear_values.size(); ++p) {
          const f_t v = qc.linear_values[p];
          if (std::abs(v) < tol) continue;
          eq_row.i.push_back(qc.linear_indices[p]);
          eq_row.x.push_back(v);
        }
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[m_before + r + 1] = 'E';
        user_problem.rhs[m_before + r + 1]       = alpha - f_t(0.5);
        if (!user_problem.row_names.empty()) {
          user_problem.row_names[m_before + r + 1] = "_CUOPT_qc_sr1_link_" + std::to_string(qc_i);
        }
      }

      // Register the cone: standard SOC, dim = r+2, head = s_0, tails = (y_0,...,y_{r-1}, s_{r+1})
      cone.clear();
      cone.reserve(r + 2);
      cone.push_back(s0_idx);
      for (i_t k = 0; k < r; k++) {
        cone.push_back(y_base + k);
      }
      cone.push_back(sr1_idx);
      cone_dim   = r + 2;
      is_rotated = 0;

      for (const i_t var : cone) {
        is_cone_var[var] = 1;
      }
      cone_dims.push_back(cone_dim);
      cone_vars.push_back(std::move(cone));
      cone_is_rotated.push_back(is_rotated);
    }
  }

  i_t n_prob = csr_A.n;

  // Convert rotated SOC cones to standard SOC cones.
  if (!rotated_cones.empty()) {
    const f_t inf        = std::numeric_limits<f_t>::infinity();
    const f_t inv_sqrt_2 = f_t(1) / std::sqrt(f_t(2));
    const f_t half       = f_t(0.5);

    for (const rotated_soc_t& rc : rotated_cones) {
      cuopt_expects(user_problem.var_types[rc.head0] ==
                      cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS,
                    error_type_t::ValidationError,
                    "Rotated second-order cone head variables must be continuous");
      if (!rc.head1_is_constant_half) {
        cuopt_expects(user_problem.var_types[rc.head1] ==
                        cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS,
                      error_type_t::ValidationError,
                      "Rotated second-order cone head variables must be continuous");
      }
      for (const i_t t : rc.tails) {
        cuopt_expects(user_problem.var_types[t] ==
                        cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS,
                      error_type_t::ValidationError,
                      "Rotated second-order cone tail variables must be continuous");
      }
    }

    // Lift each rotated cone into standard SOC coordinates with two slacks:
    //   With x_i' = sqrt(d/s)*x_hi, canonical s0 = (x_0'+x_1')/sqrt(2), s1 = (x_0'-x_1')/sqrt(2)
    // so 2*d*x_h0*x_h1 >= s*sum tail^2  <=>  2*x_0'*x_1' >= sum (x_tail)^2  =>  s0^2 >= s1^2 +
    // ... Only the rotated heads are replaced by slacks; tails stay as original variables.
    i_t n_slack_total = 0;
    for (size_t ci = 0; ci < cone_is_rotated.size(); ++ci) {
      if (cone_is_rotated[ci]) { n_slack_total += 2; }
    }

    const i_t n_old = n_prob;
    n_prob          = static_cast<i_t>(n_old + n_slack_total);

    user_problem.objective.resize(n_prob, 0);
    user_problem.lower.resize(n_prob, -inf);
    user_problem.upper.resize(n_prob, inf);
    user_problem.var_types.resize(
      n_prob, cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS);
    if (!user_problem.col_names.empty()) {
      user_problem.col_names.resize(n_prob);
      for (i_t j = n_old; j < n_prob; ++j) {
        user_problem.col_names[j] = "_CUOPT_rsoc_slack_" + std::to_string(j - n_old);
      }
    }

    is_cone_var.resize(n_prob, 0);

    const i_t m_old = csr_A.m;
    user_problem.rhs.resize(m_old + n_slack_total);
    user_problem.row_sense.resize(m_old + n_slack_total);
    if (!user_problem.row_names.empty()) {
      user_problem.row_names.resize(m_old + n_slack_total);
      for (i_t r = m_old; r < m_old + n_slack_total; ++r) {
        user_problem.row_names[r] = "_CUOPT_rsoc_lift_" + std::to_string(r - m_old);
      }
    }

    csr_A.n = n_prob;

    sparse_vector_t<i_t, f_t> eq_row;
    size_t ri      = 0;
    i_t slack_base = n_old;
    i_t row_idx    = m_old;

    for (size_t ci = 0; ci < cone_vars.size(); ++ci) {
      if (!cone_is_rotated[ci]) { continue; }
      const rotated_soc_t& rc = rotated_cones[ri++];
      const i_t dim           = cone_dims[ci];
      std::vector<i_t> new_cone;
      new_cone.reserve(dim);
      new_cone.push_back(slack_base);
      new_cone.push_back(slack_base + 1);
      new_cone.insert(new_cone.end(), rc.tails.begin(), rc.tails.end());
      cone_vars[ci] = std::move(new_cone);

      is_cone_var[slack_base]     = 1;
      is_cone_var[slack_base + 1] = 1;

      eq_row.n = n_prob;
      // If the second head is not constant half, we need to lift it.
      if (!rc.head1_is_constant_half) {
        const f_t h = inv_sqrt_2 * rc.head_lift_sqrt_ratio;
        // s_0 - h * x_h0 - h * x_h1 = 0  (h = inv_sqrt_2 * sqrt(d/s))
        eq_row.i = {rc.head0, rc.head1, slack_base};
        eq_row.x = {-h, -h, f_t(1)};
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[row_idx] = 'E';
        user_problem.rhs[row_idx]       = 0;
        ++row_idx;

        // s_1 - h * x_h0 + h * x_h1 = 0
        eq_row.i = {rc.head0, rc.head1, slack_base + 1};
        eq_row.x = {-h, h, f_t(1)};
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[row_idx] = 'E';
        user_problem.rhs[row_idx]       = 0;
        ++row_idx;

        is_cone_var[rc.head0] = 0;
        is_cone_var[rc.head1] = 0;
      } else {
        // One head is constant half, so we can lift it directly.
        // s_0 - inv_sqrt_2 * x_h0 = inv_sqrt_2 * (1/2)
        eq_row.i = {rc.head0, slack_base};
        eq_row.x = {-inv_sqrt_2, f_t(1)};
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[row_idx] = 'E';
        user_problem.rhs[row_idx]       = inv_sqrt_2 * half;
        ++row_idx;

        // s_1 - inv_sqrt_2 * x_h0 = -inv_sqrt_2 * (1/2)
        eq_row.i = {rc.head0, slack_base + 1};
        eq_row.x = {-inv_sqrt_2, f_t(1)};
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[row_idx] = 'E';
        user_problem.rhs[row_idx]       = -inv_sqrt_2 * half;
        ++row_idx;

        is_cone_var[rc.head0] = 0;
      }

      slack_base += 2;
    }

    cuopt_expects(ri == rotated_cones.size(),
                  error_type_t::RuntimeError,
                  "Internal error: rotated second-order cone metadata mismatch");
    cuopt_expects(slack_base == n_prob,
                  error_type_t::RuntimeError,
                  "Internal error: slack variable count mismatch");
    cuopt_expects(row_idx == m_old + n_slack_total,
                  error_type_t::RuntimeError,
                  "Internal error: rotated second-order cone equality row count mismatch");
    cuopt_expects(csr_A.m == m_old + n_slack_total,
                  error_type_t::RuntimeError,
                  "Internal error: CSR row count after rotated second-order cone lift");
  }

  // If a variable appears in multiple cones, create per-cone aliases and add linking rows
  // alias - original = 0 so cone variable blocks are disjoint.
  {
    std::vector<i_t> first_owner(n_prob, -1);
    std::vector<std::pair<i_t, i_t>> cone_alias_pairs;  // (alias, original)

    for (size_t ci = 0; ci < cone_vars.size(); ++ci) {
      std::vector<i_t>& cone = cone_vars[ci];
      for (i_t& var : cone) {
        cuopt_expects(var >= 0 && var < n_prob,
                      error_type_t::ValidationError,
                      "second-order cone variable index %d is outside [0, %d)",
                      static_cast<int>(var),
                      static_cast<int>(n_prob));
        if (first_owner[var] == -1) {
          first_owner[var] = static_cast<i_t>(ci);
          continue;
        }
        if (first_owner[var] != static_cast<i_t>(ci)) {
          const i_t alias = static_cast<i_t>(n_prob + cone_alias_pairs.size());
          cone_alias_pairs.emplace_back(alias, var);
          var = alias;
        }
      }
    }

    if (!cone_alias_pairs.empty()) {
      const i_t n_old = n_prob;
      const i_t n_new = static_cast<i_t>(n_old + cone_alias_pairs.size());
      const i_t m_old = csr_A.m;
      const i_t m_new = static_cast<i_t>(m_old + cone_alias_pairs.size());

      user_problem.objective.resize(n_new, 0);
      user_problem.lower.resize(n_new, -std::numeric_limits<f_t>::infinity());
      user_problem.upper.resize(n_new, std::numeric_limits<f_t>::infinity());
      user_problem.var_types.resize(
        n_new, cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS);
      if (!user_problem.col_names.empty()) { user_problem.col_names.resize(n_new); }

      for (const auto& [alias, original] : cone_alias_pairs) {
        // Cone copies are not box-constrained; linking rows tie them to the linear original.
        user_problem.lower[alias]     = -std::numeric_limits<f_t>::infinity();
        user_problem.upper[alias]     = std::numeric_limits<f_t>::infinity();
        user_problem.var_types[alias] = user_problem.var_types[original];
        // Keep objective unchanged: alias coefficient stays zero and alias==original links
        // values.
        if (!user_problem.col_names.empty()) {
          user_problem.col_names[alias] = "_CUOPT_cone_alias_" + std::to_string(alias - n_old);
        }
      }

      user_problem.rhs.resize(m_new);
      user_problem.row_sense.resize(m_new);
      if (!user_problem.row_names.empty()) { user_problem.row_names.resize(m_new); }

      csr_A.n = n_new;
      sparse_vector_t<i_t, f_t> eq_row;
      eq_row.n    = n_new;
      i_t row_idx = m_old;
      for (const auto& [alias, original] : cone_alias_pairs) {
        eq_row.i = {alias, original};
        eq_row.x = {f_t(1), f_t(-1)};
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[row_idx] = 'E';
        user_problem.rhs[row_idx]       = 0;
        if (!user_problem.row_names.empty()) {
          user_problem.row_names[row_idx] =
            "_CUOPT_cone_alias_link_" + std::to_string(row_idx - m_old);
        }
        ++row_idx;
      }

      cuopt_expects(row_idx == m_new,
                    error_type_t::RuntimeError,
                    "Internal error: cone alias linking row count mismatch");
      cuopt_expects(csr_A.m == m_new,
                    error_type_t::RuntimeError,
                    "Internal error: CSR row count after cone alias linking");

      n_prob = n_new;
    }
  }

  // Bounded cone participants cannot sit in the cone block:
  // introduce a free cone copy and alias - original = 0 so the original keeps its bounds
  // in the linear block while the barrier sees an unconstrained cone variable.
  // Exception: cone heads with lower = 0 need no split because cone membership
  // already implies x_0 >= ||x_tail|| >= 0.
  {
    const f_t neg_inf = -std::numeric_limits<f_t>::infinity();
    const f_t pos_inf = std::numeric_limits<f_t>::infinity();
    std::vector<std::pair<i_t, i_t>> bound_split_pairs;  // (cone_alias, linear_original)

    for (std::vector<i_t>& cone : cone_vars) {
      for (size_t idx = 0; idx < cone.size(); idx++) {
        i_t& var = cone[idx];
        cuopt_expects(var >= 0 && var < n_prob,
                      error_type_t::ValidationError,
                      "second-order cone variable index %d is outside [0, %d)",
                      static_cast<int>(var),
                      static_cast<int>(n_prob));
        if (user_problem.lower[var] == neg_inf && user_problem.upper[var] == pos_inf) { continue; }
        // Cone heads with lower = 0 need no split: cone membership implies x_0 >= ||x_tail|| >= 0.
        if (idx == 0 && user_problem.lower[var] == 0 && user_problem.upper[var] == pos_inf) {
          continue;
        }
        const i_t alias = static_cast<i_t>(n_prob + bound_split_pairs.size());
        bound_split_pairs.emplace_back(alias, var);
        var = alias;
      }
    }

    if (!bound_split_pairs.empty()) {
      const i_t n_old = n_prob;
      const i_t n_new = static_cast<i_t>(n_old + bound_split_pairs.size());
      const i_t m_old = csr_A.m;
      const i_t m_new = static_cast<i_t>(m_old + bound_split_pairs.size());

      user_problem.objective.resize(n_new, 0);
      user_problem.lower.resize(n_new, neg_inf);
      user_problem.upper.resize(n_new, pos_inf);
      user_problem.var_types.resize(
        n_new, cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS);
      if (!user_problem.col_names.empty()) { user_problem.col_names.resize(n_new); }

      for (const auto& [alias, original] : bound_split_pairs) {
        user_problem.var_types[alias] = user_problem.var_types[original];
        if (!user_problem.col_names.empty()) {
          user_problem.col_names[alias] =
            "_CUOPT_cone_bound_split_" + std::to_string(alias - n_old);
        }
      }

      user_problem.rhs.resize(m_new);
      user_problem.row_sense.resize(m_new);
      if (!user_problem.row_names.empty()) { user_problem.row_names.resize(m_new); }

      csr_A.n = n_new;
      sparse_vector_t<i_t, f_t> eq_row;
      eq_row.n    = n_new;
      i_t row_idx = m_old;
      for (const auto& [alias, original] : bound_split_pairs) {
        eq_row.i = {alias, original};
        eq_row.x = {f_t(1), f_t(-1)};
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[row_idx] = 'E';
        user_problem.rhs[row_idx]       = 0;
        if (!user_problem.row_names.empty()) {
          user_problem.row_names[row_idx] =
            "_CUOPT_cone_bound_split_link_" + std::to_string(row_idx - m_old);
        }
        ++row_idx;
      }

      cuopt_expects(row_idx == m_new,
                    error_type_t::RuntimeError,
                    "Internal error: cone bound-split linking row count mismatch");
      cuopt_expects(csr_A.m == m_new,
                    error_type_t::RuntimeError,
                    "Internal error: CSR row count after cone bound-split linking");

      n_prob = n_new;
    }
  }

  is_cone_var.assign(n_prob, 0);
  for (const std::vector<i_t>& cone : cone_vars) {
    for (const i_t var : cone) {
      cuopt_expects(var >= 0 && var < n_prob,
                    error_type_t::ValidationError,
                    "second-order cone variable index %d is outside [0, %d) after cone aliasing",
                    static_cast<int>(var),
                    static_cast<int>(n_prob));
      is_cone_var[var] = 1;
    }
  }

  std::vector<i_t> old_to_new(n_prob, i_t{-1});
  std::vector<i_t> new_to_old;
  new_to_old.reserve(n_prob);
  for (i_t j = 0; j < n_prob; ++j) {
    if (is_cone_var[j]) { continue; }
    old_to_new[j] = static_cast<i_t>(new_to_old.size());
    new_to_old.push_back(j);
  }
  const i_t cone_var_start = static_cast<i_t>(new_to_old.size());
  for (const std::vector<i_t>& cone : cone_vars) {
    for (const i_t old_j : cone) {
      old_to_new[old_j] = static_cast<i_t>(new_to_old.size());
      new_to_old.push_back(old_j);
    }
  }
  cuopt_expects(static_cast<i_t>(new_to_old.size()) == n_prob,
                error_type_t::RuntimeError,
                "Internal error while building second-order cone variable permutation");

  for (i_t row = 0; row < csr_A.m; ++row) {
    for (i_t p = csr_A.row_start[row]; p < csr_A.row_start[row + 1]; ++p) {
      const i_t old_j = csr_A.j[p];
      cuopt_expects(old_j >= 0 && old_j < n_prob,
                    error_type_t::ValidationError,
                    "Linear constraint matrix column index %d is outside [0, %d)",
                    static_cast<int>(old_j),
                    static_cast<int>(n_prob));
      csr_A.j[p] = old_to_new[old_j];
    }
  }

  auto permute_dense_by_old_to_new = [&](auto& values, const char* name) {
    if (values.empty()) { return; }
    using value_t = typename std::decay_t<decltype(values)>::value_type;
    cuopt_expects(values.size() == static_cast<size_t>(n_prob),
                  error_type_t::ValidationError,
                  "%s length %zu does not match number of variables %d",
                  name,
                  values.size(),
                  static_cast<int>(n_prob));
    std::vector<value_t> permuted(values.size());
    for (i_t old_j = 0; old_j < n_prob; ++old_j) {
      permuted[old_to_new[old_j]] = std::move(values[old_j]);
    }
    values = std::move(permuted);
  };

  permute_dense_by_old_to_new(user_problem.objective, "objective");
  permute_dense_by_old_to_new(user_problem.lower, "lower bounds");
  permute_dense_by_old_to_new(user_problem.upper, "upper bounds");
  permute_dense_by_old_to_new(user_problem.var_types, "variable types");
  permute_dense_by_old_to_new(user_problem.col_names, "column names");

  if (!user_problem.Q_values.empty()) {
    const i_t n_model = static_cast<i_t>(n);
    cuopt_expects(user_problem.Q_indices.size() == user_problem.Q_values.size(),
                  error_type_t::ValidationError,
                  "Quadratic objective indices and values length mismatch");
    cuopt_expects(user_problem.Q_offsets.size() == static_cast<size_t>(n_model) + 1,
                  error_type_t::ValidationError,
                  "Quadratic objective CSR offsets length must be n+1 when second-order cone "
                  "QCMATRIX "
                  "conversion permutes variables");
    cuopt_expects(user_problem.Q_offsets[0] == 0,
                  error_type_t::ValidationError,
                  "Quadratic objective CSR offsets[0] must be 0");
    cuopt_expects(user_problem.Q_offsets[n_model] == static_cast<i_t>(user_problem.Q_values.size()),
                  error_type_t::ValidationError,
                  "Quadratic objective CSR last offset must equal number of nonzeros");

    std::vector<i_t> q_offsets(n_prob + 1, 0);
    for (i_t old_row = 0; old_row < n_model; ++old_row) {
      const i_t p_beg = user_problem.Q_offsets[old_row];
      const i_t p_end = user_problem.Q_offsets[old_row + 1];
      cuopt_expects(
        p_beg >= 0 && p_beg <= p_end && p_end <= static_cast<i_t>(user_problem.Q_values.size()),
        error_type_t::ValidationError,
        "Quadratic objective CSR offsets are invalid at row %d",
        static_cast<int>(old_row));
      const i_t new_row      = old_to_new[old_row];
      q_offsets[new_row + 1] = p_end - p_beg;
    }
    for (i_t row = 0; row < n_prob; ++row) {
      q_offsets[row + 1] += q_offsets[row];
    }

    std::vector<i_t> q_indices(user_problem.Q_values.size());
    std::vector<f_t> q_values(user_problem.Q_values.size());
    std::vector<i_t> q_write = q_offsets;
    for (i_t old_row = 0; old_row < n_model; ++old_row) {
      const i_t new_row = old_to_new[old_row];
      for (i_t p = user_problem.Q_offsets[old_row]; p < user_problem.Q_offsets[old_row + 1]; ++p) {
        const i_t old_col = user_problem.Q_indices[p];
        cuopt_expects(old_col >= 0 && old_col < n_model,
                      error_type_t::ValidationError,
                      "Quadratic objective column index %d is outside [0, %d)",
                      static_cast<int>(old_col),
                      static_cast<int>(n_model));
        const i_t dst  = q_write[new_row]++;
        q_indices[dst] = old_to_new[old_col];
        q_values[dst]  = user_problem.Q_values[p];
      }
    }

    user_problem.Q_offsets = std::move(q_offsets);
    user_problem.Q_indices = std::move(q_indices);
    user_problem.Q_values  = std::move(q_values);
  }

  user_problem.cone_var_start         = cone_var_start;
  user_problem.second_order_cone_dims = std::move(cone_dims);
  user_problem.num_rows               = csr_A.m;
  user_problem.num_cols               = n_prob;

  user_problem.original_num_cols = static_cast<i_t>(n);
  user_problem.original_col_to_expanded_col.resize(n);
  for (i_t old_j = 0; old_j < static_cast<i_t>(n); ++old_j) {
    user_problem.original_col_to_expanded_col[old_j] = old_to_new[old_j];
  }
}

/** Map barrier primal/reduced-cost vectors from expanded SOC layout back to original model columns.
 */
template <typename i_t, typename f_t>
void project_barrier_solution_to_model_variables(
  const simplex::user_problem_t<i_t, f_t>& user_problem, simplex::lp_solution_t<i_t, f_t>& solution)
{
  const i_t n_original = user_problem.original_num_cols;
  if (n_original <= 0) { return; }
  if (static_cast<i_t>(user_problem.original_col_to_expanded_col.size()) != n_original) { return; }

  std::vector<f_t> model_x(n_original);
  std::vector<f_t> model_z(n_original);
  for (i_t j = 0; j < n_original; ++j) {
    const i_t expanded_j = user_problem.original_col_to_expanded_col[j];
    model_x[j]           = solution.x[expanded_j];
    model_z[j]           = solution.z[expanded_j];
  }
  const i_t m = static_cast<i_t>(solution.y.size());
  solution.resize(m, n_original);
  solution.x = std::move(model_x);
  solution.z = std::move(model_z);
}

}  // namespace cuopt::mathematical_optimization::barrier
