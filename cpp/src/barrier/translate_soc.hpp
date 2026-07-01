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

  // SOC conversion accepts:
  //   1) diagonal Lorentz-form QCMATRIX rows:
  //        -s*x_head^2 + sum_i s*x_tail_i^2 <= 0   (any common s > 0; divide by s to normalize)
  //   2) rotated SOC rows:
  //        -2*d*x_head0*x_head1 + sum_i s*x_tail_i^2 <= 0   (d>0, s>0; canonical d=s)
  //      symmetric Q off-diagonals (-d,-d) give x^T Q x cross term -2*d*x0*x1, i.e. a*x0*x1
  //      in the inequality 2*d*x0*x1 >= s*||tail||^2 with a = 2*d. Lift uses sqrt(d/s) on heads.
  //   3) quadratic rows with linear part:
  //        sum_i s*x_tail_i^2 + a^T x <= 0
  //      represented as diagonal +s QCMATRIX entries plus linear terms in COLUMNS.
  //      We introduce an auxiliary t = -(1/s)*a^T x so the row becomes:
  //        sum_i x_tail_i^2 - t <= 0
  //      then lift it as rotated SOC with implicit second head fixed at 1/2.
  // The barrier consumes SOCs as trailing variable blocks [head, tails...], so we validate all
  // QCMATRIX blocks first, convert rotated cones via slack variables in standard SOC coordinates,
  // then apply a single column permutation to the linear model.
  struct rotated_soc_t {
    i_t head0{};
    i_t head1{};
    std::vector<i_t> tails{};
    bool head1_is_constant_half{false};
    /// For two-head rotated SOC: sqrt(d/s) where Q_off = -d and tail diagonals +s (canonical 1).
    f_t head_lift_sqrt_ratio{1};
  };
  // This is the index of the auxiliary variable for the linear part of the quadratic constraint.
  std::vector<i_t> qc_affine_heads(qcs.size(), -1);
  i_t n_affine_linear_aux = 0;
  for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
    if (!qcs[qc_i].linear_values.empty()) {
      qc_affine_heads[qc_i] = static_cast<i_t>(n + n_affine_linear_aux);
      ++n_affine_linear_aux;
    }
  }

  const i_t n_with_affine_aux = static_cast<i_t>(n + n_affine_linear_aux);

  std::vector<std::vector<i_t>> cone_vars;
  std::vector<i_t> cone_dims;
  std::vector<char> cone_is_rotated;
  std::vector<rotated_soc_t> rotated_cones;
  std::vector<char> is_cone_var(n_with_affine_aux, 0);
  cone_vars.reserve(qcs.size());
  cone_dims.reserve(qcs.size());
  cone_is_rotated.reserve(qcs.size());
  rotated_cones.reserve(qcs.size());
  std::vector<f_t> qc_soc_uniform_scale(qcs.size(), 1);

  for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
    auto qc = qcs[qc_i];
    cuopt_expects(qc.constraint_row_type != 'E',
                  error_type_t::ValidationError,
                  "Equality quadratic constraints are not supported for SOC conversion");
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
                  "Quadratic constraint '%s' SOC must have at least 1 entry in Q (nnz %d)",
                  qc.constraint_row_name.c_str(),
                  static_cast<int>(q_nnz));

    // This is the index of the auxiliary variable for the linear part of the quadratic
    // constraint.
    const i_t affine_head      = qc_affine_heads[qc_i];
    const bool has_linear_part = affine_head >= 0;
    if (has_linear_part) {
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
    }

    // Verify Q as either:
    // - standard SOC: one diagonal -s (head), tail diagonals +s for a common s > 0,
    // - rotated SOC: symmetric (-s,-s) off-diagonal pair on the two heads, tails +s,
    // - affine SOC: tail diagonals +s and linear terms (no Q off-diagonals).
    // Feasibility is unchanged after dividing the quadratic row by s; affine rows also scale
    // linear coefficients when forming the auxiliary t = -(1/s) a^T x.

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

    // Determine whether to use the general convex quadratic path.
    // The general path is needed when Q does not fit any special SOC pattern,
    // or when the RHS is nonzero (special cases require rhs = 0).
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
    const bool use_general_path = has_duplicate_rows || has_near_zero_diag || has_nonzero_rhs ||
                                  has_nonuniform_diag || offdiag_entries.size() > 2 ||
                                  (offdiag_entries.size() == 1) || (neg_diag_rows.size() > 1) ||
                                  (!neg_diag_rows.empty() && has_linear_part) ||
                                  (!neg_diag_rows.empty() && !offdiag_entries.empty());

    f_t uniform_s        = 0;
    bool have_uniform_s  = false;
    auto note_positive_s = [&](f_t v) {
      cuopt_expects(v > tol,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' SOC Q: expected strictly positive diagonal tail "
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
          "Quadratic constraint '%s' SOC Q: all positive diagonal coefficients must match; got "
          "%.17g vs %.17g",
          qc.constraint_row_name.c_str(),
          static_cast<double>(v),
          static_cast<double>(uniform_s));
      }
    };

    std::vector<i_t> cone;
    i_t cone_dim    = 0;
    char is_rotated = 0;
    i_t head        = -1;

    if (!use_general_path) {
      // Special-case rhs == 0 requirement for SOC patterns
      cuopt_expects((qc.rhs_value < tol) && (qc.rhs_value > -tol),
                    error_type_t::ValidationError,
                    "SOC conversion currently requires rhs = 0 for quadratic constraints "
                    "(constraint '%s' has rhs %.17g)",
                    qc.constraint_row_name.c_str(),
                    static_cast<double>(qc.rhs_value));

      if (offdiag_entries.empty()) {
        if (!has_linear_part) {
          if (pos_diag_rows.empty()) {
            cuopt_expects(
              neg_diag_rows.size() == 1 && q_nnz == 1,
              error_type_t::ValidationError,
              "Quadratic constraint '%s' SOC Q: expected tail diagonals +s with head -s, "
              "or a single head row with q_nnz=1",
              qc.constraint_row_name.c_str());
            const f_t neg_v = neg_diag_rows[0].second;
            cuopt_expects(neg_v < -tol,
                          error_type_t::ValidationError,
                          "Quadratic constraint '%s' SOC Q: cone head diagonal must be negative "
                          "(%.17g)",
                          qc.constraint_row_name.c_str(),
                          static_cast<double>(neg_v));
            uniform_s      = -neg_v;
            have_uniform_s = true;
            head           = neg_diag_rows[0].first;
            cuopt_expects(
              static_cast<i_t>(tail_vars.size()) == q_nnz - 1,
              error_type_t::ValidationError,
              "Quadratic constraint '%s' SOC Q: expected %d diagonal +s entries (tails), found %zu",
              qc.constraint_row_name.c_str(),
              static_cast<int>(q_nnz - 1),
              tail_vars.size());
            cone.reserve(1);
            cone.push_back(head);
            cone_dim   = static_cast<i_t>(cone.size());
            is_rotated = 0;
          } else {
            for (const std::pair<i_t, f_t>& pr : pos_diag_rows) {
              note_positive_s(pr.second);
            }
            cuopt_expects(
              have_uniform_s,
              error_type_t::ValidationError,
              "Quadratic constraint '%s' SOC Q: could not infer uniform positive scale s",
              qc.constraint_row_name.c_str());
            cuopt_expects(
              neg_diag_rows.size() == 1,
              error_type_t::ValidationError,
              "Quadratic constraint '%s' SOC Q: expected exactly one diagonal -s (cone head) for "
              "%zu tail entries, found %zu negative diagonals",
              qc.constraint_row_name.c_str(),
              tail_vars.size(),
              neg_diag_rows.size());
            cuopt_expects(
              static_cast<i_t>(tail_vars.size()) == q_nnz - 1,
              error_type_t::ValidationError,
              "Quadratic constraint '%s' SOC Q: expected %d diagonal +s entries (tails), found %zu",
              qc.constraint_row_name.c_str(),
              static_cast<int>(q_nnz - 1),
              tail_vars.size());
            const f_t neg_v = neg_diag_rows[0].second;
            cuopt_expects(
              approx_eq_scaled(neg_v, -uniform_s),
              error_type_t::ValidationError,
              "Quadratic constraint '%s' SOC Q: cone head diagonal must be -s with the same s as "
              "positive tail diagonals; head %.17g vs -s = %.17g",
              qc.constraint_row_name.c_str(),
              static_cast<double>(neg_v),
              static_cast<double>(-uniform_s));
            head = neg_diag_rows[0].first;
            // The SOC ||tail|| <= head requires head >= 0. Check explicit bound
            // or implied bound from singleton inequality constraints.
            cuopt_expects(std::max(user_problem.lower[head], implied_lower[head]) >= 0,
                          error_type_t::ValidationError,
                          "Quadratic constraint '%s' SOC head variable (index %d) must have a "
                          "non-negative lower bound for the constraint to be convex",
                          qc.constraint_row_name.c_str(),
                          static_cast<int>(head));
            cone.reserve(q_nnz);
            cone.push_back(head);
            cone.insert(cone.end(), tail_vars.begin(), tail_vars.end());
            cone_dim   = static_cast<i_t>(cone.size());
            is_rotated = 0;
          }
        } else {
          cuopt_expects(
            neg_diag_rows.empty(),
            error_type_t::ValidationError,
            "Quadratic constraint '%s' with linear terms cannot contain negative diagonal "
            "Q entries",
            qc.constraint_row_name.c_str());
          cuopt_expects(affine_head >= 0,
                        error_type_t::ValidationError,
                        "Quadratic constraint '%s' internal error: affine SOC head index invalid",
                        qc.constraint_row_name.c_str());
          for (const std::pair<i_t, f_t>& pr : pos_diag_rows) {
            note_positive_s(pr.second);
          }
          cuopt_expects(have_uniform_s,
                        error_type_t::ValidationError,
                        "Quadratic constraint '%s' with linear terms must have at least one "
                        "diagonal +s term in Q",
                        qc.constraint_row_name.c_str());
          cuopt_expects(!tail_vars.empty(),
                        error_type_t::ValidationError,
                        "Quadratic constraint '%s' with linear terms must have at least one "
                        "diagonal +s term in Q",
                        qc.constraint_row_name.c_str());
          for (const i_t tail : tail_vars) {
            cuopt_expects(
              tail != affine_head,
              error_type_t::ValidationError,
              "Quadratic constraint '%s' with linear terms requires the linear head variable to be "
              "distinct from quadratic diagonal variables",
              qc.constraint_row_name.c_str());
          }

          cone.reserve(tail_vars.size() + 1);
          cone.push_back(affine_head);
          cone.insert(cone.end(), tail_vars.begin(), tail_vars.end());
          cone_dim   = static_cast<i_t>(tail_vars.size() + 2);
          is_rotated = 1;
          rotated_cones.push_back(rotated_soc_t{affine_head, -1, tail_vars, true, 1});
        }
      } else {
        cuopt_expects(!has_linear_part,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' with linear terms cannot include rotated-SOC "
                      "off-diagonal entries",
                      qc.constraint_row_name.c_str());
        cuopt_expects(neg_diag_rows.empty(),
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC Q cannot contain diagonal head "
                      "entries; found %zu negative diagonals",
                      qc.constraint_row_name.c_str(),
                      neg_diag_rows.size());
        for (const std::pair<i_t, f_t>& pr : pos_diag_rows) {
          note_positive_s(pr.second);
        }
        cuopt_expects(have_uniform_s,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC Q: could not infer uniform scale s",
                      qc.constraint_row_name.c_str());
        cuopt_expects(
          offdiag_entries.size() == 2,
          error_type_t::ValidationError,
          "Quadratic constraint '%s' rotated SOC Q must contain exactly one symmetric off-diagonal "
          "pair (-d,-d); found %zu off-diagonal entries",
          qc.constraint_row_name.c_str(),
          offdiag_entries.size());

        const i_t a  = std::get<0>(offdiag_entries[0]);
        const i_t b  = std::get<1>(offdiag_entries[0]);
        const f_t v0 = std::get<2>(offdiag_entries[0]);
        cuopt_expects(
          v0 < -tol,
          error_type_t::ValidationError,
          "Quadratic constraint '%s' rotated SOC Q off-diagonal must be negative; got %.17g",
          qc.constraint_row_name.c_str(),
          static_cast<double>(v0));
        cuopt_expects(a != b,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC Q off-diagonal pair must use distinct "
                      "variables",
                      qc.constraint_row_name.c_str());
        cuopt_expects(std::get<0>(offdiag_entries[1]) == b && std::get<1>(offdiag_entries[1]) == a,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC Q must have symmetric entries (a,b) "
                      "and (b,a) with the same value",
                      qc.constraint_row_name.c_str());
        const f_t v1 = std::get<2>(offdiag_entries[1]);
        cuopt_expects(
          v1 < -tol,
          error_type_t::ValidationError,
          "Quadratic constraint '%s' rotated SOC Q off-diagonal must be negative; got %.17g",
          qc.constraint_row_name.c_str(),
          static_cast<double>(v1));
        cuopt_expects(
          approx_eq_scaled(v0, v1),
          error_type_t::ValidationError,
          "Quadratic constraint '%s' rotated SOC Q symmetric off-diagonals must match; got %.17g "
          "and %.17g",
          qc.constraint_row_name.c_str(),
          static_cast<double>(v0),
          static_cast<double>(v1));
        const f_t cross_d = -v0;
        cuopt_expects(
          cross_d > tol,
          error_type_t::ValidationError,
          "Quadratic constraint '%s' rotated SOC Q cross coefficient d = -Q_off must be positive",
          qc.constraint_row_name.c_str());
        const f_t head_lift_sqrt_ratio = std::sqrt(cross_d / uniform_s);
        cuopt_expects(std::isfinite(static_cast<double>(head_lift_sqrt_ratio)),
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC Q head lift ratio sqrt(d/s) is not "
                      "finite (d=%.17g, s=%.17g)",
                      qc.constraint_row_name.c_str(),
                      static_cast<double>(cross_d),
                      static_cast<double>(uniform_s));
        cuopt_expects(static_cast<i_t>(tail_vars.size()) == q_nnz - 2,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC Q: expected %d diagonal +s entries "
                      "(tails), found %zu",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(q_nnz - 2),
                      tail_vars.size());
        cuopt_expects(q_nnz >= 3,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC Q must have at least 1 tail entry",
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
                      "Quadratic constraint '%s' rotated SOC head variable (index %d) must have a "
                      "non-negative lower bound for the constraint to be convex",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(a));
        cuopt_expects(std::max(user_problem.lower[b], implied_lower[b]) >= 0,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' rotated SOC head variable (index %d) must have a "
                      "non-negative lower bound for the constraint to be convex",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(b));
        rotated_cones.push_back(rotated_soc_t{a, b, tail_vars, false, head_lift_sqrt_ratio});
      }

      cuopt_expects(have_uniform_s && uniform_s > tol,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' SOC Q: uniform scale s must be positive (got %.17g)",
                    qc.constraint_row_name.c_str(),
                    static_cast<double>(uniform_s));
      qc_soc_uniform_scale[qc_i] = uniform_s;

      for (const i_t var : cone) {
        cuopt_expects(var >= 0 && var < static_cast<i_t>(is_cone_var.size()),
                      error_type_t::ValidationError,
                      "SOC variable index %d is outside [0, %zu)",
                      static_cast<int>(var),
                      is_cone_var.size());
      }
      cone_dims.push_back(cone_dim);
      cone_vars.push_back(std::move(cone));
      cone_is_rotated.push_back(is_rotated);

    } else {
      // =========================================================================
      // General convex quadratic constraint path:
      //   x^T Q x + c^T x <= alpha

      // Invalidate affine head for this QC — the general path handles the linear part directly
      qc_affine_heads[qc_i] = -1;
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

  // Recount affine linear aux variables (some may have been invalidated by the general path)
  n_affine_linear_aux = 0;
  for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
    if (qc_affine_heads[qc_i] >= 0) { ++n_affine_linear_aux; }
  }

  // Add affine linear auxiliary variables and linking rows.
  if (n_affine_linear_aux > 0) {
    const f_t inf        = std::numeric_limits<f_t>::infinity();
    const i_t n_old      = static_cast<i_t>(n);
    const i_t n_aug      = n_with_affine_aux;
    const i_t m_old      = csr_A.m;
    const i_t m_aug      = static_cast<i_t>(m_old + n_affine_linear_aux);
    i_t row_write_cursor = m_old;

    user_problem.objective.resize(n_aug, 0);
    user_problem.lower.resize(n_aug, -inf);
    user_problem.upper.resize(n_aug, inf);
    user_problem.var_types.resize(
      n_aug, cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS);
    if (!user_problem.col_names.empty()) { user_problem.col_names.resize(n_aug); }

    for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
      const i_t aux_j = qc_affine_heads[qc_i];
      if (aux_j < 0) { continue; }
      user_problem.lower[aux_j] = 0;
      user_problem.upper[aux_j] = inf;
      if (!user_problem.col_names.empty()) {
        user_problem.col_names[aux_j] = "_CUOPT_qc_linear_aux_" + std::to_string(aux_j - n_old);
      }
    }

    user_problem.rhs.resize(m_aug);
    user_problem.row_sense.resize(m_aug);
    if (!user_problem.row_names.empty()) { user_problem.row_names.resize(m_aug); }

    csr_A.n = std::max(csr_A.n, n_aug);
    sparse_vector_t<i_t, f_t> eq_row;
    eq_row.n = csr_A.n;

    for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
      const i_t aux_j = qc_affine_heads[qc_i];
      if (aux_j < 0) { continue; }
      const auto& qc = qcs[qc_i];
      eq_row.i.clear();
      eq_row.x.clear();
      // Define auxiliary as t = -(1/s) a^T x so QC linear part matches normalized cone row.
      const f_t inv_s = 1 / qc_soc_uniform_scale[qc_i];
      eq_row.i.push_back(aux_j);
      eq_row.x.push_back(1);
      for (size_t p = 0; p < qc.linear_values.size(); ++p) {
        const f_t v = qc.linear_values[p];
        if (v > -tol && v < tol) { continue; }
        eq_row.i.push_back(qc.linear_indices[p]);
        eq_row.x.push_back(v * inv_s);
      }
      eq_row.sort();
      csr_A.append_row(eq_row);
      user_problem.row_sense[row_write_cursor] = 'E';
      user_problem.rhs[row_write_cursor]       = 0;
      if (!user_problem.row_names.empty()) {
        user_problem.row_names[row_write_cursor] =
          "_CUOPT_qc_linear_link_" + qc.constraint_row_name;
      }
      ++row_write_cursor;
    }

    cuopt_expects(row_write_cursor == m_aug,
                  error_type_t::RuntimeError,
                  "Internal error: affine QC linking row count mismatch");
    cuopt_expects(csr_A.m == m_aug,
                  error_type_t::RuntimeError,
                  "Internal error: CSR row count after affine QC linking");
  }

  i_t n_prob = csr_A.n;

  // Convert rotated SOC cones to standard SOC cones.
  if (!rotated_cones.empty()) {
    cuopt_expects(user_problem.Q_values.empty(),
                  error_type_t::ValidationError,
                  "Rotated SOC conversion is currently not supported when the objective has "
                  "quadratic terms");

    const f_t inf        = std::numeric_limits<f_t>::infinity();
    const f_t inv_sqrt_2 = f_t(1) / std::sqrt(f_t(2));
    const f_t half       = f_t(0.5);

    for (const rotated_soc_t& rc : rotated_cones) {
      cuopt_expects(user_problem.var_types[rc.head0] ==
                      cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS,
                    error_type_t::ValidationError,
                    "Rotated SOC head variables must be continuous");
      if (!rc.head1_is_constant_half) {
        cuopt_expects(user_problem.var_types[rc.head1] ==
                        cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS,
                      error_type_t::ValidationError,
                      "Rotated SOC head variables must be continuous");
      }
      for (const i_t t : rc.tails) {
        cuopt_expects(user_problem.var_types[t] ==
                        cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS,
                      error_type_t::ValidationError,
                      "Rotated SOC tail variables must be continuous");
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
                  "Internal error: rotated SOC cone metadata mismatch");
    cuopt_expects(slack_base == n_prob,
                  error_type_t::RuntimeError,
                  "Internal error: slack variable count mismatch");
    cuopt_expects(row_idx == m_old + n_slack_total,
                  error_type_t::RuntimeError,
                  "Internal error: rotated SOC equality row count mismatch");
    cuopt_expects(csr_A.m == m_old + n_slack_total,
                  error_type_t::RuntimeError,
                  "Internal error: CSR row count after rotated SOC lift");
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
                      "SOC variable index %d is outside [0, %d)",
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
                      "SOC variable index %d is outside [0, %d)",
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
                    "SOC variable index %d is outside [0, %d) after cone aliasing",
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
                "Internal error while building SOC variable permutation");

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
                  "Quadratic objective CSR offsets length must be n+1 when SOC QCMATRIX "
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
