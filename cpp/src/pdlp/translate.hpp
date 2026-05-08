/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/sparse_vector.hpp>

#include <utilities/copy_helpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
static dual_simplex::user_problem_t<i_t, f_t> cuopt_problem_to_simplex_problem(
  raft::handle_t const* handle_ptr, detail::problem_t<i_t, f_t>& model)
{
  dual_simplex::user_problem_t<i_t, f_t> user_problem(handle_ptr);

  int m                  = model.n_constraints;
  int n                  = model.n_variables;
  int nz                 = model.nnz;
  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = cuopt::host_copy(model.objective_coefficients, handle_ptr->get_stream());

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.x = std::vector<f_t>(cuopt::host_copy(model.coefficients, handle_ptr->get_stream()));
  csr_A.j = std::vector<i_t>(cuopt::host_copy(model.variables, handle_ptr->get_stream()));
  csr_A.row_start = std::vector<i_t>(cuopt::host_copy(model.offsets, handle_ptr->get_stream()));

  user_problem.rhs.resize(m);
  user_problem.row_sense.resize(m);
  user_problem.range_rows.clear();
  user_problem.range_value.clear();

  auto model_constraint_lower_bounds =
    cuopt::host_copy(model.constraint_lower_bounds, handle_ptr->get_stream());
  auto model_constraint_upper_bounds =
    cuopt::host_copy(model.constraint_upper_bounds, handle_ptr->get_stream());

  // All constraints have lower and upper bounds
  // lr <= a_i^T x <= ur
  for (int i = 0; i < m; ++i) {
    const double constraint_lower_bound = model_constraint_lower_bounds[i];
    const double constraint_upper_bound = model_constraint_upper_bounds[i];
    if (constraint_lower_bound == constraint_upper_bound) {
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_upper_bound == std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'G';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_lower_bound == -std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'L';
      user_problem.rhs[i]       = constraint_upper_bound;
    } else {
      // This is range row
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
      user_problem.range_rows.push_back(i);
      const double bound_difference = constraint_upper_bound - constraint_lower_bound;
      user_problem.range_value.push_back(bound_difference);
    }
  }
  user_problem.num_range_rows = user_problem.range_rows.size();
  std::tie(user_problem.lower, user_problem.upper) =
    extract_host_bounds<f_t>(model.variable_bounds, handle_ptr);
  user_problem.problem_name = model.original_problem_ptr->get_problem_name();
  if (model.row_names.size() > 0) {
    user_problem.row_names.resize(m);
    for (int i = 0; i < m; ++i) {
      user_problem.row_names[i] = model.row_names[i];
    }
  }
  if (model.var_names.size() > 0) {
    user_problem.col_names.resize(n);
    for (int j = 0; j < n; ++j) {
      if (j < (int)model.var_names.size()) {
        user_problem.col_names[j] = model.var_names[j];
      } else {
        user_problem.col_names[j] = "_CUOPT_x" + std::to_string(j);
      }
    }
  }
  user_problem.obj_constant = model.presolve_data.objective_offset;
  user_problem.obj_scale    = model.presolve_data.objective_scaling_factor;
  user_problem.var_types.resize(n);

  auto model_variable_types = cuopt::host_copy(model.variable_types, handle_ptr->get_stream());
  for (int j = 0; j < n; ++j) {
    user_problem.var_types[j] =
      model_variable_types[j] == var_t::CONTINUOUS
        ? cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS
        : cuopt::linear_programming::dual_simplex::variable_type_t::INTEGER;
  }

  user_problem.Q_offsets = model.Q_offsets;
  user_problem.Q_indices = model.Q_indices;
  user_problem.Q_values  = model.Q_values;

  if (model.original_problem_ptr->has_quadratic_constraints()) {
    const auto& qcs = model.original_problem_ptr->get_quadratic_constraints();
    cuopt_expects(!qcs.empty(),
                  error_type_t::ValidationError,
                  "Quadratic-constraint flag is set, but no constraints were provided");

    // Use a practical tolerance for text-parsed MPS numeric values.
    const f_t tol = std::numeric_limits<f_t>::epsilon() * 2;

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
      f_t head_lift_sqrt_ratio{f_t(1)};
    };
    // This is the index of the auxiliary variable for the linear part of the quadratic constraint.
    std::vector<i_t> qc_affine_heads(qcs.size(), static_cast<i_t>(-1));
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
    std::vector<char> is_cone_var(static_cast<size_t>(n_with_affine_aux), 0);
    cone_vars.reserve(qcs.size());
    cone_dims.reserve(qcs.size());
    cone_is_rotated.reserve(qcs.size());
    rotated_cones.reserve(qcs.size());
    std::vector<f_t> qc_soc_uniform_scale(qcs.size(), f_t(1));

    for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
      const auto& qc = qcs[qc_i];
      cuopt_expects(qc.constraint_row_type == 'L',
                    error_type_t::ValidationError,
                    "Only <= quadratic constraints are supported for SOC conversion");
      cuopt_expects(qc.rhs_value < tol && qc.rhs_value > -tol,
                    error_type_t::ValidationError,
                    "SOC conversion currently requires rhs = 0 for quadratic constraints");
      cuopt_expects(qc.linear_values.size() == qc.linear_indices.size(),
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' linear_values and linear_indices length mismatch",
                    qc.constraint_row_name.c_str());

      cuopt_expects(qc.quadratic_offsets.size() >= 2,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' has invalid CSR offsets (need at least 2 entries)",
                    qc.constraint_row_name.c_str());
      cuopt_expects(qc.quadratic_values.size() == qc.quadratic_indices.size(),
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' quadratic_values and quadratic_indices length "
                    "mismatch for CSR Q",
                    qc.constraint_row_name.c_str());

      const i_t q_nnz = static_cast<i_t>(qc.quadratic_values.size());
      cuopt_expects(q_nnz >= 1,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' SOC must have at least 1 entry in Q (nnz %d)",
                    qc.constraint_row_name.c_str(),
                    static_cast<int>(q_nnz));

      cuopt_expects(
        qc.quadratic_offsets.size() == static_cast<size_t>(n) + 1,
        error_type_t::ValidationError,
        "Quadratic constraint '%s' Q must be n by n in CSR: expected %zu CSR row pointers (offsets "
        "length n+1), got %zu (n = %d)",
        qc.constraint_row_name.c_str(),
        static_cast<size_t>(n) + 1,
        qc.quadratic_offsets.size(),
        static_cast<int>(n));
      cuopt_expects(
        qc.quadratic_offsets[static_cast<size_t>(n)] == q_nnz,
        error_type_t::ValidationError,
        "Quadratic constraint '%s' Q last CSR offset %d must equal number of nonzeros (nnz) %d",
        qc.constraint_row_name.c_str(),
        static_cast<int>(qc.quadratic_offsets[static_cast<size_t>(n)]),
        static_cast<int>(q_nnz));
      cuopt_expects(qc.quadratic_offsets[0] == 0,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' Q CSR offsets[0] must be 0",
                    qc.constraint_row_name.c_str());
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
        cuopt_expects(
          nonzero_terms > 0,
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

      std::vector<std::tuple<i_t, i_t, f_t>> q_entries;
      q_entries.reserve(static_cast<size_t>(q_nnz));
      for (i_t r = 0; r < n; ++r) {
        const i_t p_beg = qc.quadratic_offsets[static_cast<size_t>(r)];
        const i_t p_end = qc.quadratic_offsets[static_cast<size_t>(r + 1)];
        cuopt_expects(p_beg >= 0 && p_beg <= p_end && p_end <= q_nnz,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' Q row %d has invalid CSR offsets [%d, %d)",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(r),
                      static_cast<int>(p_beg),
                      static_cast<int>(p_end));

        if (p_beg == p_end) { continue; }

        cuopt_expects(p_beg + 1 == p_end,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' Q row %d: expected at most one stored entry per "
                      "row (got end - beg = %d)",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(r),
                      static_cast<int>(p_end - p_beg));

        const i_t col = qc.quadratic_indices[static_cast<size_t>(p_beg)];
        const f_t v   = qc.quadratic_values[static_cast<size_t>(p_beg)];
        q_entries.emplace_back(r, col, v);
      }
      cuopt_expects(static_cast<i_t>(q_entries.size()) == q_nnz,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' Q row nnz mismatch (expected %d stored entries, got "
                    "%zu)",
                    qc.constraint_row_name.c_str(),
                    static_cast<int>(q_nnz),
                    q_entries.size());

      std::vector<std::pair<i_t, f_t>> pos_diag_rows;
      std::vector<std::pair<i_t, f_t>> neg_diag_rows;
      std::vector<std::tuple<i_t, i_t, f_t>> offdiag_entries;
      pos_diag_rows.reserve(q_entries.size());
      neg_diag_rows.reserve(1);
      offdiag_entries.reserve(4);

      for (const auto& [r, c, v] : q_entries) {
        if (r == c) {
          if (v > tol) {
            pos_diag_rows.emplace_back(r, v);
          } else if (v < -tol) {
            neg_diag_rows.emplace_back(r, v);
          } else {
            cuopt_expects(false,
                          error_type_t::ValidationError,
                          "Quadratic constraint '%s' Q row %d: diagonal SOC entry is near zero "
                          "(%.17g)",
                          qc.constraint_row_name.c_str(),
                          static_cast<int>(r),
                          static_cast<double>(v));
          }
        } else {
          offdiag_entries.emplace_back(r, c, v);
        }
      }

      std::vector<i_t> tail_vars;
      tail_vars.reserve(pos_diag_rows.size());
      for (const auto& pr : pos_diag_rows) {
        tail_vars.push_back(pr.first);
      }

      f_t uniform_s        = f_t(0);
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
      i_t head        = static_cast<i_t>(-1);

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
            is_rotated = static_cast<char>(0);
          } else {
            for (const auto& pr : pos_diag_rows) {
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
            cone.reserve(static_cast<size_t>(q_nnz));
            cone.push_back(head);
            cone.insert(cone.end(), tail_vars.begin(), tail_vars.end());
            cone_dim   = static_cast<i_t>(cone.size());
            is_rotated = static_cast<char>(0);
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
          for (const auto& pr : pos_diag_rows) {
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
          is_rotated = static_cast<char>(1);
          rotated_cones.push_back(
            rotated_soc_t{affine_head, static_cast<i_t>(-1), tail_vars, true, f_t(1)});
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
        for (const auto& pr : pos_diag_rows) {
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

        cone.reserve(static_cast<size_t>(q_nnz));
        cone.push_back(a);
        cone.push_back(b);
        cone.insert(cone.end(), tail_vars.begin(), tail_vars.end());
        cone_dim   = static_cast<i_t>(cone.size());
        is_rotated = static_cast<char>(1);
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
                      "SOC variable index %d is outside [0, %d)",
                      static_cast<int>(var),
                      static_cast<int>(is_cone_var.size()));
      }
      cone_dims.push_back(cone_dim);
      cone_vars.push_back(std::move(cone));
      cone_is_rotated.push_back(is_rotated);
    }
    // Add affine linear auxiliary variables and linking rows.
    if (n_affine_linear_aux > 0) {
      const f_t inf        = std::numeric_limits<f_t>::infinity();
      const i_t n_old      = static_cast<i_t>(n);
      const i_t n_aug      = n_with_affine_aux;
      const i_t m_old      = csr_A.m;
      const i_t m_aug      = static_cast<i_t>(m_old + n_affine_linear_aux);
      i_t row_write_cursor = m_old;

      user_problem.objective.resize(static_cast<size_t>(n_aug), f_t(0));
      user_problem.lower.resize(static_cast<size_t>(n_aug), -inf);
      user_problem.upper.resize(static_cast<size_t>(n_aug), inf);
      user_problem.var_types.resize(
        static_cast<size_t>(n_aug),
        cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS);
      if (!user_problem.col_names.empty()) {
        user_problem.col_names.resize(static_cast<size_t>(n_aug));
      }

      for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
        const i_t aux_j = qc_affine_heads[qc_i];
        if (aux_j < 0) { continue; }
        user_problem.lower[static_cast<size_t>(aux_j)] = f_t(0);
        user_problem.upper[static_cast<size_t>(aux_j)] = inf;
        if (!user_problem.col_names.empty()) {
          user_problem.col_names[static_cast<size_t>(aux_j)] =
            "_CUOPT_qc_linear_aux_" + std::to_string(static_cast<int>(aux_j - n_old));
        }
      }

      user_problem.rhs.resize(static_cast<size_t>(m_aug));
      user_problem.row_sense.resize(static_cast<size_t>(m_aug));
      if (!user_problem.row_names.empty()) {
        user_problem.row_names.resize(static_cast<size_t>(m_aug));
      }

      csr_A.n = n_aug;
      dual_simplex::sparse_vector_t<i_t, f_t> eq_row;
      eq_row.n = n_aug;

      for (size_t qc_i = 0; qc_i < qcs.size(); ++qc_i) {
        const i_t aux_j = qc_affine_heads[qc_i];
        if (aux_j < 0) { continue; }
        const auto& qc = qcs[qc_i];
        eq_row.i.clear();
        eq_row.x.clear();
        // Define auxiliary as t = -(1/s) a^T x so QC linear part matches normalized cone row.
        const f_t inv_s = f_t(1) / qc_soc_uniform_scale[qc_i];
        eq_row.i.push_back(aux_j);
        eq_row.x.push_back(f_t(1));
        for (size_t p = 0; p < qc.linear_values.size(); ++p) {
          const f_t v = qc.linear_values[p];
          if (v > -tol && v < tol) { continue; }
          eq_row.i.push_back(qc.linear_indices[p]);
          eq_row.x.push_back(v * inv_s);
        }
        eq_row.sort();
        csr_A.append_row(eq_row);
        user_problem.row_sense[static_cast<size_t>(row_write_cursor)] = 'E';
        user_problem.rhs[static_cast<size_t>(row_write_cursor)]       = f_t(0);
        if (!user_problem.row_names.empty()) {
          user_problem.row_names[static_cast<size_t>(row_write_cursor)] =
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

    i_t n_prob = n_with_affine_aux;

    // Convert rotated SOC cones to standard SOC cones.
    if (!rotated_cones.empty()) {
      cuopt_expects(user_problem.Q_values.empty(),
                    error_type_t::ValidationError,
                    "Rotated SOC conversion is currently not supported when the objective has "
                    "quadratic terms");

      const f_t inf        = std::numeric_limits<f_t>::infinity();
      const f_t inv_sqrt_2 = f_t(1) / std::sqrt(f_t(2));
      const f_t half       = f_t(0.5);

      for (const auto& rc : rotated_cones) {
        cuopt_expects(user_problem.var_types[static_cast<size_t>(rc.head0)] ==
                        cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS,
                      error_type_t::ValidationError,
                      "Rotated SOC head variables must be continuous");
        if (!rc.head1_is_constant_half) {
          cuopt_expects(user_problem.var_types[static_cast<size_t>(rc.head1)] ==
                          cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS,
                        error_type_t::ValidationError,
                        "Rotated SOC head variables must be continuous");
        }
        for (const i_t t : rc.tails) {
          cuopt_expects(user_problem.var_types[static_cast<size_t>(t)] ==
                          cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS,
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

      user_problem.objective.resize(static_cast<size_t>(n_prob), f_t(0));
      user_problem.lower.resize(static_cast<size_t>(n_prob), -inf);
      user_problem.upper.resize(static_cast<size_t>(n_prob), inf);
      user_problem.var_types.resize(
        static_cast<size_t>(n_prob),
        cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS);
      if (!user_problem.col_names.empty()) {
        user_problem.col_names.resize(static_cast<size_t>(n_prob));
        for (i_t j = n_old; j < n_prob; ++j) {
          user_problem.col_names[static_cast<size_t>(j)] =
            "_CUOPT_rsoc_slack_" + std::to_string(static_cast<int>(j - n_old));
        }
      }

      is_cone_var.resize(static_cast<size_t>(n_prob), 0);

      const i_t m_old = csr_A.m;
      user_problem.rhs.resize(static_cast<size_t>(m_old + n_slack_total));
      user_problem.row_sense.resize(static_cast<size_t>(m_old + n_slack_total));
      if (!user_problem.row_names.empty()) {
        user_problem.row_names.resize(static_cast<size_t>(m_old + n_slack_total));
        for (i_t r = m_old; r < m_old + n_slack_total; ++r) {
          user_problem.row_names[static_cast<size_t>(r)] =
            "_CUOPT_rsoc_lift_" + std::to_string(static_cast<int>(r - m_old));
        }
      }

      csr_A.n = n_prob;

      dual_simplex::sparse_vector_t<i_t, f_t> eq_row;
      size_t ri      = 0;
      i_t slack_base = n_old;
      i_t row_idx    = m_old;

      for (size_t ci = 0; ci < cone_vars.size(); ++ci) {
        if (!cone_is_rotated[ci]) { continue; }
        const auto& rc = rotated_cones[ri++];
        const i_t dim  = cone_dims[ci];
        std::vector<i_t> new_cone;
        new_cone.reserve(static_cast<size_t>(dim));
        new_cone.push_back(slack_base);
        new_cone.push_back(slack_base + 1);
        new_cone.insert(new_cone.end(), rc.tails.begin(), rc.tails.end());
        cone_vars[ci] = std::move(new_cone);

        is_cone_var[static_cast<size_t>(slack_base)]     = 1;
        is_cone_var[static_cast<size_t>(slack_base + 1)] = 1;

        eq_row.n = n_prob;
        // If the second head is not constant half, we need to lift it.
        if (!rc.head1_is_constant_half) {
          const f_t h = inv_sqrt_2 * rc.head_lift_sqrt_ratio;
          // s_0 - h * x_h0 - h * x_h1 = 0  (h = inv_sqrt_2 * sqrt(d/s))
          eq_row.i = {rc.head0, rc.head1, slack_base};
          eq_row.x = {-h, -h, f_t(1)};
          eq_row.sort();
          csr_A.append_row(eq_row);
          user_problem.row_sense[static_cast<size_t>(row_idx)] = 'E';
          user_problem.rhs[static_cast<size_t>(row_idx)]       = f_t(0);
          ++row_idx;

          // s_1 - h * x_h0 + h * x_h1 = 0
          eq_row.i = {rc.head0, rc.head1, slack_base + 1};
          eq_row.x = {-h, h, f_t(1)};
          eq_row.sort();
          csr_A.append_row(eq_row);
          user_problem.row_sense[static_cast<size_t>(row_idx)] = 'E';
          user_problem.rhs[static_cast<size_t>(row_idx)]       = f_t(0);
          ++row_idx;

          is_cone_var[static_cast<size_t>(rc.head0)] = 0;
          is_cone_var[static_cast<size_t>(rc.head1)] = 0;
        } else {
          // One head is constant half, so we can lift it directly.
          // s_0 - inv_sqrt_2 * x_h0 = inv_sqrt_2 * (1/2)
          eq_row.i = {rc.head0, slack_base};
          eq_row.x = {-inv_sqrt_2, f_t(1)};
          eq_row.sort();
          csr_A.append_row(eq_row);
          user_problem.row_sense[static_cast<size_t>(row_idx)] = 'E';
          user_problem.rhs[static_cast<size_t>(row_idx)]       = inv_sqrt_2 * half;
          ++row_idx;

          // s_1 - inv_sqrt_2 * x_h0 = -inv_sqrt_2 * (1/2)
          eq_row.i = {rc.head0, slack_base + 1};
          eq_row.x = {-inv_sqrt_2, f_t(1)};
          eq_row.sort();
          csr_A.append_row(eq_row);
          user_problem.row_sense[static_cast<size_t>(row_idx)] = 'E';
          user_problem.rhs[static_cast<size_t>(row_idx)]       = -inv_sqrt_2 * half;
          ++row_idx;

          is_cone_var[static_cast<size_t>(rc.head0)] = 0;
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
      std::vector<i_t> first_owner(static_cast<size_t>(n_prob), static_cast<i_t>(-1));
      std::vector<std::pair<i_t, i_t>> cone_alias_pairs;  // (alias, original)

      for (size_t ci = 0; ci < cone_vars.size(); ++ci) {
        auto& cone = cone_vars[ci];
        for (auto& var : cone) {
          cuopt_expects(var >= 0 && var < n_prob,
                        error_type_t::ValidationError,
                        "SOC variable index %d is outside [0, %d)",
                        static_cast<int>(var),
                        static_cast<int>(n_prob));
          const auto idx = static_cast<size_t>(var);
          if (first_owner[idx] == static_cast<i_t>(-1)) {
            first_owner[idx] = static_cast<i_t>(ci);
            continue;
          }
          if (first_owner[idx] != static_cast<i_t>(ci)) {
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

        user_problem.objective.resize(static_cast<size_t>(n_new), f_t(0));
        user_problem.lower.resize(static_cast<size_t>(n_new),
                                  -std::numeric_limits<f_t>::infinity());
        user_problem.upper.resize(static_cast<size_t>(n_new), std::numeric_limits<f_t>::infinity());
        user_problem.var_types.resize(
          static_cast<size_t>(n_new),
          cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS);
        if (!user_problem.col_names.empty()) {
          user_problem.col_names.resize(static_cast<size_t>(n_new));
        }

        for (const auto& [alias, original] : cone_alias_pairs) {
          user_problem.lower[static_cast<size_t>(alias)] =
            user_problem.lower[static_cast<size_t>(original)];
          user_problem.upper[static_cast<size_t>(alias)] =
            user_problem.upper[static_cast<size_t>(original)];
          user_problem.var_types[static_cast<size_t>(alias)] =
            user_problem.var_types[static_cast<size_t>(original)];
          // Keep objective unchanged: alias coefficient stays zero and alias==original links
          // values.
          if (!user_problem.col_names.empty()) {
            user_problem.col_names[static_cast<size_t>(alias)] =
              "_CUOPT_cone_alias_" + std::to_string(static_cast<int>(alias - n_old));
          }
        }

        user_problem.rhs.resize(static_cast<size_t>(m_new));
        user_problem.row_sense.resize(static_cast<size_t>(m_new));
        if (!user_problem.row_names.empty()) {
          user_problem.row_names.resize(static_cast<size_t>(m_new));
        }

        csr_A.n = n_new;
        dual_simplex::sparse_vector_t<i_t, f_t> eq_row;
        eq_row.n    = n_new;
        i_t row_idx = m_old;
        for (const auto& [alias, original] : cone_alias_pairs) {
          eq_row.i = {alias, original};
          eq_row.x = {f_t(1), f_t(-1)};
          eq_row.sort();
          csr_A.append_row(eq_row);
          user_problem.row_sense[static_cast<size_t>(row_idx)] = 'E';
          user_problem.rhs[static_cast<size_t>(row_idx)]       = f_t(0);
          if (!user_problem.row_names.empty()) {
            user_problem.row_names[static_cast<size_t>(row_idx)] =
              "_CUOPT_cone_alias_link_" + std::to_string(static_cast<int>(row_idx - m_old));
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

    is_cone_var.assign(static_cast<size_t>(n_prob), 0);
    for (const auto& cone : cone_vars) {
      for (const i_t var : cone) {
        cuopt_expects(var >= 0 && var < n_prob,
                      error_type_t::ValidationError,
                      "SOC variable index %d is outside [0, %d) after cone aliasing",
                      static_cast<int>(var),
                      static_cast<int>(n_prob));
        is_cone_var[static_cast<size_t>(var)] = 1;
      }
    }

    std::vector<i_t> old_to_new(static_cast<size_t>(n_prob), i_t{-1});
    std::vector<i_t> new_to_old;
    new_to_old.reserve(static_cast<size_t>(n_prob));
    for (i_t j = 0; j < n_prob; ++j) {
      if (is_cone_var[static_cast<size_t>(j)]) { continue; }
      old_to_new[static_cast<size_t>(j)] = static_cast<i_t>(new_to_old.size());
      new_to_old.push_back(j);
    }
    const i_t cone_var_start = static_cast<i_t>(new_to_old.size());
    for (const auto& cone : cone_vars) {
      for (const i_t old_j : cone) {
        old_to_new[static_cast<size_t>(old_j)] = static_cast<i_t>(new_to_old.size());
        new_to_old.push_back(old_j);
      }
    }
    cuopt_expects(static_cast<i_t>(new_to_old.size()) == n_prob,
                  error_type_t::RuntimeError,
                  "Internal error while building SOC variable permutation");

    for (i_t row = 0; row < csr_A.m; ++row) {
      for (i_t p = csr_A.row_start[static_cast<size_t>(row)];
           p < csr_A.row_start[static_cast<size_t>(row + 1)];
           ++p) {
        const i_t old_j = csr_A.j[static_cast<size_t>(p)];
        cuopt_expects(old_j >= 0 && old_j < n_prob,
                      error_type_t::ValidationError,
                      "Linear constraint matrix column index %d is outside [0, %d)",
                      static_cast<int>(old_j),
                      static_cast<int>(n_prob));
        csr_A.j[static_cast<size_t>(p)] = old_to_new[static_cast<size_t>(old_j)];
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
        permuted[static_cast<size_t>(old_to_new[static_cast<size_t>(old_j)])] =
          std::move(values[static_cast<size_t>(old_j)]);
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
      cuopt_expects(user_problem.Q_offsets[static_cast<size_t>(n_model)] ==
                      static_cast<i_t>(user_problem.Q_values.size()),
                    error_type_t::ValidationError,
                    "Quadratic objective CSR last offset must equal number of nonzeros");

      std::vector<i_t> q_offsets(static_cast<size_t>(n_prob) + 1, 0);
      for (i_t old_row = 0; old_row < n_model; ++old_row) {
        const i_t p_beg = user_problem.Q_offsets[static_cast<size_t>(old_row)];
        const i_t p_end = user_problem.Q_offsets[static_cast<size_t>(old_row + 1)];
        cuopt_expects(
          p_beg >= 0 && p_beg <= p_end && p_end <= static_cast<i_t>(user_problem.Q_values.size()),
          error_type_t::ValidationError,
          "Quadratic objective CSR offsets are invalid at row %d",
          static_cast<int>(old_row));
        const i_t new_row                           = old_to_new[static_cast<size_t>(old_row)];
        q_offsets[static_cast<size_t>(new_row + 1)] = p_end - p_beg;
      }
      for (i_t row = 0; row < n_prob; ++row) {
        q_offsets[static_cast<size_t>(row + 1)] += q_offsets[static_cast<size_t>(row)];
      }

      std::vector<i_t> q_indices(user_problem.Q_values.size());
      std::vector<f_t> q_values(user_problem.Q_values.size());
      auto q_write = q_offsets;
      for (i_t old_row = 0; old_row < n_model; ++old_row) {
        const i_t new_row = old_to_new[static_cast<size_t>(old_row)];
        for (i_t p = user_problem.Q_offsets[static_cast<size_t>(old_row)];
             p < user_problem.Q_offsets[static_cast<size_t>(old_row + 1)];
             ++p) {
          const i_t old_col = user_problem.Q_indices[static_cast<size_t>(p)];
          cuopt_expects(old_col >= 0 && old_col < n_model,
                        error_type_t::ValidationError,
                        "Quadratic objective column index %d is outside [0, %d)",
                        static_cast<int>(old_col),
                        static_cast<int>(n_model));
          const i_t dst                       = q_write[static_cast<size_t>(new_row)]++;
          q_indices[static_cast<size_t>(dst)] = old_to_new[static_cast<size_t>(old_col)];
          q_values[static_cast<size_t>(dst)]  = user_problem.Q_values[static_cast<size_t>(p)];
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
  }

  csr_A.to_compressed_col(user_problem.A);

  return user_problem;
}

template <typename i_t, typename f_t>
void translate_to_crossover_problem(const detail::problem_t<i_t, f_t>& problem,
                                    optimization_problem_solution_t<i_t, f_t>& sol,
                                    dual_simplex::lp_problem_t<i_t, f_t>& lp,
                                    dual_simplex::lp_solution_t<i_t, f_t>& initial_solution)
{
  CUOPT_LOG_DEBUG("Starting translation");

  auto stream                     = problem.handle_ptr->get_stream();
  std::vector<f_t> pdlp_objective = cuopt::host_copy(problem.objective_coefficients, stream);

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(
    problem.n_constraints, problem.n_variables, problem.nnz);
  csr_A.x         = std::vector<f_t>(cuopt::host_copy(problem.coefficients, stream));
  csr_A.j         = std::vector<i_t>(cuopt::host_copy(problem.variables, stream));
  csr_A.row_start = std::vector<i_t>(cuopt::host_copy(problem.offsets, stream));

  stream.synchronize();
  CUOPT_LOG_DEBUG("Converting to compressed column");
  csr_A.to_compressed_col(lp.A);
  CUOPT_LOG_DEBUG("Converted to compressed column");

  std::vector<f_t> slack(problem.n_constraints);
  std::vector<f_t> tmp_x = cuopt::host_copy(sol.get_primal_solution(), stream);
  stream.synchronize();
  dual_simplex::matrix_vector_multiply(lp.A, f_t(1.0), tmp_x, f_t(0.0), slack);
  CUOPT_LOG_DEBUG("Multiplied A and x");

  lp.A.col_start.resize(problem.n_variables + problem.n_constraints + 1);
  lp.A.x.resize(problem.nnz + problem.n_constraints);
  lp.A.i.resize(problem.nnz + problem.n_constraints);
  i_t nz = problem.nnz;
  for (i_t j = problem.n_variables; j < problem.n_variables + problem.n_constraints; ++j) {
    lp.A.col_start[j] = nz;
    lp.A.i[nz]        = j - problem.n_variables;
    lp.A.x[nz]        = -1.0;
    ++nz;
  }
  lp.A.col_start[problem.n_variables + problem.n_constraints] = nz;
  CUOPT_LOG_DEBUG("Finished with A");

  const i_t n = problem.n_variables + problem.n_constraints;
  const i_t m = problem.n_constraints;
  lp.num_cols = n;
  lp.num_rows = m;
  lp.A.n      = n;
  lp.rhs.resize(m, 0.0);
  lp.lower.resize(n);
  lp.upper.resize(n);
  lp.obj_constant = problem.presolve_data.objective_offset;
  lp.obj_scale    = problem.presolve_data.objective_scaling_factor;

  auto [lower, upper] = extract_host_bounds<f_t>(problem.variable_bounds, problem.handle_ptr);

  std::vector<f_t> constraint_lower = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  std::vector<f_t> constraint_upper = cuopt::host_copy(problem.constraint_upper_bounds, stream);

  lp.objective.resize(n, 0.0);
  std::copy(
    pdlp_objective.begin(), pdlp_objective.begin() + problem.n_variables, lp.objective.begin());
  std::copy(lower.begin(), lower.begin() + problem.n_variables, lp.lower.begin());
  std::copy(upper.begin(), upper.begin() + problem.n_variables, lp.upper.begin());

  problem.handle_ptr->get_stream().synchronize();
  for (i_t i = 0; i < m; ++i) {
    lp.lower[problem.n_variables + i] = constraint_lower[i];
    lp.upper[problem.n_variables + i] = constraint_upper[i];
  }
  CUOPT_LOG_DEBUG("Finished with lp");

  initial_solution.resize(m, n);

  std::copy(tmp_x.begin(), tmp_x.begin() + problem.n_variables, initial_solution.x.begin());
  for (i_t j = problem.n_variables; j < n; ++j) {
    initial_solution.x[j] = slack[j - problem.n_variables];
    // Project slack variables inside their bounds
    if (initial_solution.x[j] < lp.lower[j]) { initial_solution.x[j] = lp.lower[j]; }
    if (initial_solution.x[j] > lp.upper[j]) { initial_solution.x[j] = lp.upper[j]; }
  }
  CUOPT_LOG_DEBUG("Finished with x");
  initial_solution.y = cuopt::host_copy(sol.get_dual_solution(), stream);

  std::vector<f_t> tmp_z = cuopt::host_copy(sol.get_reduced_cost(), stream);
  stream.synchronize();
  std::copy(tmp_z.begin(), tmp_z.begin() + problem.n_variables, initial_solution.z.begin());
  for (i_t j = problem.n_variables; j < n; ++j) {
    initial_solution.z[j] = initial_solution.y[j - problem.n_variables];
  }
  CUOPT_LOG_DEBUG("Finished with z");

  CUOPT_LOG_DEBUG("Finished translating");
}

}  // namespace cuopt::linear_programming
