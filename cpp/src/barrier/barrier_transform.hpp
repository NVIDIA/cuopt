/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization {

/**
 * User space to presolve space transform retained on barrier_cache_t after Optimal:
 * convert / presolve / scaling, plus the scaled LP.
 * Enough to crush new linear objective or RHS data from the original problem into
 * presolved space and to uncrush a solution without rerunning those algorithms.
 */
struct barrier_transform_t {
  int user_num_cols{0};
  int user_num_rows{0};
  int original_num_cols{0};
  int original_num_rows{0};
  double obj_scale{1.0};
  double obj_constant{0.0};
  bool maximize{false};  // first-solve sense; cached Q/c were negated if true

  // Enough of the user problem for reuse uncrush without rebuilding A.
  std::vector<char> row_sense;
  int cone_var_start{0};
  std::vector<int> second_order_cone_dims;
  int expanded_original_num_cols{0};
  std::vector<int> original_col_to_expanded_col;

  cuopt::mathematical_optimization::simplex::presolve_info_t<int, double> presolve_info;
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  // Barrier linear objective minus crush(user c) from the first solve (Q*ell shift, etc.).
  std::vector<double> linear_obj_shift;
  // Barrier RHS minus crush(user b) from the first solve (fixed/lower-bound shifts).
  std::vector<double> rhs_shift;
  // False when range rows or folding put the user RHS somewhere other than barrier_lp->rhs.
  bool rhs_update_supported{false};
  std::unique_ptr<cuopt::mathematical_optimization::simplex::lp_problem_t<int, double>> barrier_lp;
  // CSC Q with slack columns, as consumed by iteration_data_t. Not the same object as
  // barrier_lp->Q.
  std::unique_ptr<csc_matrix_t<int, double>> barrier_Q;
};

// Shared by the outer QCQP entry (which may swap in a slim fabricated user_problem) and the
// inner barrier path (which skips convert/presolve/scaling). Both must agree, or the slim
// problem is fed to a full convert/presolve.
inline bool can_reuse_barrier_cache(barrier_transform_t const* xf,
                                    int bound_free_variables,
                                    int num_cols,
                                    int num_rows,
                                    bool has_quadratic_objective,
                                    bool user_has_soc)
{
  return xf != nullptr && xf->barrier_lp != nullptr && has_quadratic_objective && !user_has_soc &&
         xf->second_order_cone_dims.empty() && xf->barrier_lp->second_order_cone_dims.empty() &&
         bound_free_variables == 0 && xf->presolve_info.bounded_free_variables.empty() &&
         static_cast<int>(xf->row_sense.size()) == xf->user_num_rows &&
         num_cols == xf->user_num_cols && num_rows == xf->user_num_rows;
}

inline std::vector<double> crush_user_linear_objective(barrier_transform_t const& xf,
                                                       double const* c,
                                                       int n)
{
  if (c == nullptr || n != xf.user_num_cols) {
    throw std::invalid_argument(
      "update_linear_objective: linear objective length must match the cached user column count.");
  }
  if (xf.original_num_cols < xf.user_num_cols) {
    throw std::invalid_argument(
      "update_linear_objective: cached original column count is smaller than user n.");
  }
  if (xf.barrier_lp == nullptr) {
    throw std::invalid_argument("update_linear_objective: cached barrier LP is missing.");
  }

  std::vector<double> orig(static_cast<std::size_t>(xf.original_num_cols), 0.0);
  for (int j = 0; j < n; ++j) {
    orig[static_cast<std::size_t>(j)] = c[j];
  }
  for (int j : xf.presolve_info.negated_variables) {
    orig[static_cast<std::size_t>(j)] *= -1.0;
  }

  std::vector<double> presolved;
  if (!xf.presolve_info.remaining_variables.empty()) {
    presolved.resize(xf.presolve_info.remaining_variables.size());
    for (std::size_t k = 0; k < xf.presolve_info.remaining_variables.size(); ++k) {
      presolved[k] = orig[static_cast<std::size_t>(xf.presolve_info.remaining_variables[k])];
    }
  } else {
    presolved = std::move(orig);
  }

  auto const& pairs = xf.presolve_info.free_variable_pairs;
  if (!pairs.empty()) {
    if (pairs.size() % 2 != 0) {
      throw std::invalid_argument("update_linear_objective: free_variable_pairs size is not even.");
    }
    std::size_t extra = pairs.size() / 2;
    presolved.resize(presolved.size() + extra);
    for (std::size_t k = 0; k < extra; ++k) {
      int u                                  = pairs[2 * k];
      int v                                  = pairs[2 * k + 1];
      presolved[static_cast<std::size_t>(v)] = -presolved[static_cast<std::size_t>(u)];
    }
  }

  if (static_cast<int>(presolved.size()) != xf.barrier_lp->num_cols ||
      xf.column_scales.size() != presolved.size()) {
    throw std::invalid_argument(
      "update_linear_objective: crushed objective size does not match barrier columns / "
      "column_scales.");
  }
  for (std::size_t j = 0; j < presolved.size(); ++j) {
    presolved[j] /= xf.column_scales[j];
  }
  return presolved;
}

// success: crushed is written. invalid: error is set and the caller reports a validation
// failure. infeasible: a presolve-dropped empty row cannot hold the new RHS; the caller records
// that for the next Solve instead of treating it as a validation failure.
enum class crush_rhs_status_t { success = 0, invalid = -1, infeasible = -2 };

template <typename i_t, typename f_t>
inline crush_rhs_status_t crush_user_rhs(
  barrier_transform_t const& xf, f_t const* b, i_t m, std::vector<f_t>& crushed, std::string& error)
{
  auto invalid = [&](char const* message) {
    error = message;
    return crush_rhs_status_t::invalid;
  };
  if (b == nullptr || m != xf.user_num_rows) {
    return invalid("update_rhs: RHS length must match the cached user row count.");
  }
  if (!xf.rhs_update_supported) {
    return invalid("update_rhs: cached convert used range rows or folding; run a full Solve.");
  }
  if (xf.original_num_rows != xf.user_num_rows) {
    return invalid("update_rhs: cached original row count does not match the user row count.");
  }
  if (static_cast<i_t>(xf.row_sense.size()) != xf.user_num_rows) {
    return invalid("update_rhs: cached row-sense count does not match the user row count.");
  }
  if (xf.barrier_lp == nullptr) { return invalid("update_rhs: cached barrier LP is missing."); }

  // convert turns 'G' rows into 'L' rows by negating the row and its RHS.
  std::vector<f_t> original(static_cast<std::size_t>(xf.original_num_rows));
  for (i_t i = 0; i < m; ++i) {
    original[static_cast<std::size_t>(i)] =
      xf.row_sense[static_cast<std::size_t>(i)] == 'G' ? -b[i] : b[i];
  }

  // Presolve drops only empty equalities, and only when the RHS is exactly 0.
  for (i_t i : xf.presolve_info.removed_constraints) {
    if (i < 0 || i >= m) {
      return invalid("update_rhs: removed constraint index is out of range.");
    }
    if (original[i] != f_t(0)) { return crush_rhs_status_t::infeasible; }
  }

  // Empty remaining_constraints means either no empty-row pass ran, or every row was dropped
  // and accepted above.
  std::vector<f_t> presolved;
  if (!xf.presolve_info.remaining_constraints.empty()) {
    presolved.resize(xf.presolve_info.remaining_constraints.size());
    for (std::size_t k = 0; k < xf.presolve_info.remaining_constraints.size(); ++k) {
      presolved[k] = original[static_cast<std::size_t>(xf.presolve_info.remaining_constraints[k])];
    }
  } else if (xf.presolve_info.removed_constraints.empty()) {
    presolved = std::move(original);
  }

  if (static_cast<i_t>(presolved.size()) != xf.barrier_lp->num_rows ||
      xf.row_scales.size() != presolved.size()) {
    return invalid("update_rhs: crushed RHS size does not match barrier rows / row_scales.");
  }
  for (std::size_t i = 0; i < presolved.size(); ++i) {
    presolved[i] /= xf.row_scales[i];
  }
  crushed = std::move(presolved);
  return crush_rhs_status_t::success;
}

}  // namespace cuopt::mathematical_optimization
