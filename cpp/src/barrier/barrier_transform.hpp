/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization {

/**
 * User space to presolved space transform retained on barrier_cache_t after Optimal:
 * convert / presolve / scaling, plus the scaled LP.
 * Enough to crush new linear objective or RHS data from the original problem into
 * presolved space and to uncrush a solution without rerunning those algorithms.
 */
template <typename i_t, typename f_t>
struct barrier_transform_t {
  i_t user_num_cols{0};
  i_t user_num_rows{0};
  i_t original_num_cols{0};
  i_t original_num_rows{0};
  f_t obj_scale{1.0};
  f_t obj_constant{0.0};
  bool maximize{false};  // first-solve sense; cached Q/c were negated if true

  // Enough of the user problem for reuse uncrush without rebuilding A.
  std::vector<char> row_sense;
  i_t cone_var_start{0};
  // cone_var_start after convert, which inserts inequality slacks ahead of the cone block.
  i_t converted_cone_var_start{0};
  std::vector<i_t> second_order_cone_dims;
  // Quadratic constraint count the cached expansion was built from.
  i_t num_quadratic_constraints{0};
  // Dimensions before the QCMATRIX->SOC expansion, which permutes columns and appends rows.
  // Updates arrive in these coordinates, not the expanded ones. Zero when no expansion ran.
  i_t pre_expansion_num_cols{0};
  i_t pre_expansion_num_rows{0};
  std::vector<i_t> original_col_to_expanded_col;
  // RHS of the rows the expansion appended. Fixed by the quadratic constraints, so an RHS
  // update keeps them and only overwrites the problem's own rows.
  std::vector<f_t> cone_row_rhs;
  std::vector<simplex::cone_head_bound_t<i_t, f_t>> cone_head_bounds;

  cuopt::mathematical_optimization::simplex::presolve_info_t<i_t, f_t> presolve_info;
  std::vector<f_t> column_scales;
  std::vector<f_t> row_scales;
  // Barrier linear objective minus crush(user c) from the first solve (Q*ell shift, etc.).
  std::vector<f_t> linear_obj_shift;
  // Barrier RHS minus crush(user b) from the first solve (fixed/lower-bound shifts).
  std::vector<f_t> rhs_shift;
  // False when range rows or folding put the user RHS somewhere other than barrier_lp->rhs.
  bool rhs_update_supported{false};
  std::unique_ptr<cuopt::mathematical_optimization::simplex::lp_problem_t<i_t, f_t>> barrier_lp;
  // CSC Q with slack columns, as consumed by iteration_data_t. Not the same object as
  // barrier_lp->Q.
  std::unique_ptr<csc_matrix_t<i_t, f_t>> barrier_Q;
};

// Shared reuse gate from the update-API work. A cone problem skips the size
// check: the caller compares either expanded user counts or pre-expansion
// problem counts, which are not the same number.
template <typename i_t, typename f_t>
inline bool can_reuse_barrier_cache(barrier_transform_t<i_t, f_t> const* xf,
                                    i_t bound_free_variables,
                                    i_t num_cols,
                                    i_t num_rows,
                                    bool has_quadratic_objective,
                                    bool user_has_soc)
{
  if (xf == nullptr || xf->barrier_lp == nullptr) { return false; }
  if (bound_free_variables != 0 || !xf->presolve_info.bounded_free_variables.empty()) {
    return false;
  }
  if (user_has_soc) { return true; }
  // LP and QP reuse when the sizes match. A cached quadratic objective does not match an LP.
  return xf->second_order_cone_dims.empty() && xf->barrier_lp->second_order_cone_dims.empty() &&
         static_cast<i_t>(xf->row_sense.size()) == xf->user_num_rows &&
         num_cols == xf->user_num_cols && num_rows == xf->user_num_rows &&
         (has_quadratic_objective || xf->barrier_lp->Q.n == 0);
}

// Column count an update is sized in: the pre-expansion count when the SOC expansion grew the
// problem, otherwise the cached user count.
template <typename i_t, typename f_t>
inline i_t problem_num_cols(barrier_transform_t<i_t, f_t> const& xf)
{
  return xf.pre_expansion_num_cols > 0 ? xf.pre_expansion_num_cols : xf.user_num_cols;
}

// Row count an update is sized in.
template <typename i_t, typename f_t>
inline i_t problem_num_rows(barrier_transform_t<i_t, f_t> const& xf)
{
  return xf.original_col_to_expanded_col.empty() ? xf.user_num_rows : xf.pre_expansion_num_rows;
}

// The expansion permutes problem columns into a [linear | cone] layout. An empty map means no
// expansion ran, so the layouts coincide.
template <typename i_t, typename f_t>
inline i_t problem_col_to_expanded_col(barrier_transform_t<i_t, f_t> const& xf, i_t problem_col)
{
  return xf.original_col_to_expanded_col.empty() ? problem_col
                                                 : xf.original_col_to_expanded_col[problem_col];
}

// convert inserts the inequality slacks ahead of the cone block, pushing every cone column
// right. Mirrors user_col_to_problem_col in presolve.cpp.
template <typename i_t, typename f_t>
inline i_t expanded_col_to_converted_col(barrier_transform_t<i_t, f_t> const& xf, i_t expanded_col)
{
  if (xf.second_order_cone_dims.empty() || xf.converted_cone_var_start <= xf.cone_var_start ||
      expanded_col < xf.cone_var_start) {
    return expanded_col;
  }
  return xf.converted_cone_var_start + (expanded_col - xf.cone_var_start);
}

// Move the problem's own coefficients into the expanded layout the cached problem is sized for.
template <typename i_t, typename f_t, typename value_t>
std::vector<value_t> scatter_problem_objective(barrier_transform_t<i_t, f_t> const& xf,
                                               std::vector<value_t> const& problem_objective)
{
  std::vector<value_t> expanded(static_cast<std::size_t>(xf.user_num_cols), value_t(0));
  for (i_t j = 0; j < static_cast<i_t>(problem_objective.size()); ++j) {
    expanded[problem_col_to_expanded_col(xf, j)] = problem_objective[j];
  }
  return expanded;
}

// Inverse of scatter_problem_objective, giving crush_user_linear_objective the problem-sized input
// it expects. The expansion adds no objective coefficients, so nothing is lost.
template <typename i_t, typename f_t>
std::vector<f_t> gather_problem_objective(barrier_transform_t<i_t, f_t> const& xf,
                                          std::vector<f_t> const& expanded_objective)
{
  std::vector<f_t> problem_objective(static_cast<std::size_t>(problem_num_cols(xf)));
  for (i_t j = 0; j < static_cast<i_t>(problem_objective.size()); ++j) {
    problem_objective[j] = expanded_objective[problem_col_to_expanded_col(xf, j)];
  }
  return problem_objective;
}

// Reuse never re-runs the expansion, so the cone block must be laid out exactly as the cached
// one left it.
template <typename i_t, typename f_t>
bool cone_layout_matches(barrier_transform_t<i_t, f_t> const& xf,
                         simplex::user_problem_t<i_t, f_t> const& user_problem)
{
  return user_problem.cone_var_start == xf.cone_var_start &&
         user_problem.second_order_cone_dims.size() == xf.second_order_cone_dims.size() &&
         std::equal(user_problem.second_order_cone_dims.begin(),
                    user_problem.second_order_cone_dims.end(),
                    xf.second_order_cone_dims.begin(),
                    [](i_t dim, i_t cached) { return dim == cached; });
}

// Equality substitution is the last presolve step and is not represented by remaining_variables
// or free_variable_pairs. It drops zero-cost free columns (and their pivot rows). Indices in the
// elimination records are into the vector produced by the earlier presolve maps.
template <typename i_t, typename f_t>
inline void drop_substituted_free_variables(barrier_transform_t<i_t, f_t> const& xf,
                                            std::vector<f_t>& values,
                                            char const* what)
{
  auto const& eliminations = xf.presolve_info.free_variable_eliminations;
  if (eliminations.empty()) { return; }
  auto const& keep = xf.presolve_info.free_elimination_remaining_variables;
  for (auto const& elimination : eliminations) {
    if (elimination.variable < 0 ||
        static_cast<std::size_t>(elimination.variable) >= values.size()) {
      throw std::invalid_argument(std::string(what) +
                                  ": eliminated free variable index is out of range.");
    }
    // Substitution is valid only while this coefficient stays zero. A nonzero cost would have
    // kept the column, so the cached factorization cannot be reused.
    if (values[elimination.variable] != 0.0) {
      throw std::invalid_argument(
        std::string(what) +
        ": a free variable removed by equality substitution has a nonzero objective "
        "coefficient; run a full Solve.");
    }
  }
  std::vector<f_t> reduced(keep.size());
  for (std::size_t k = 0; k < keep.size(); ++k) {
    i_t const column = keep[k];
    if (column < 0 || static_cast<std::size_t>(column) >= values.size()) {
      throw std::invalid_argument(std::string(what) +
                                  ": free-elimination column index is out of range.");
    }
    reduced[k] = values[column];
  }
  values = std::move(reduced);
}

// Replay rhs[row] -= factor * rhs[pivot] in elimination order, then drop the pivot rows.
// Factors come from the matrix and do not depend on the right-hand side.
template <typename i_t, typename f_t>
inline void substitute_free_variable_rows(barrier_transform_t<i_t, f_t> const& xf,
                                          std::vector<f_t>& values,
                                          char const* what)
{
  auto const& eliminations = xf.presolve_info.free_variable_eliminations;
  if (eliminations.empty()) { return; }
  for (auto const& elimination : eliminations) {
    if (elimination.pivot_row < 0 ||
        static_cast<std::size_t>(elimination.pivot_row) >= values.size()) {
      throw std::invalid_argument(std::string(what) +
                                  ": free-elimination pivot row is out of range.");
    }
    if (elimination.affected_rows.size() != elimination.factors.size()) {
      throw std::invalid_argument(std::string(what) +
                                  ": free-elimination factors do not match affected rows.");
    }
    f_t const pivot_rhs = values[elimination.pivot_row];
    for (std::size_t k = 0; k < elimination.affected_rows.size(); ++k) {
      i_t const row = elimination.affected_rows[k];
      if (row < 0 || static_cast<std::size_t>(row) >= values.size()) {
        throw std::invalid_argument(std::string(what) +
                                    ": free-elimination affected row is out of range.");
      }
      values[row] -= elimination.factors[k] * pivot_rhs;
    }
  }
  auto const& keep = xf.presolve_info.free_elimination_remaining_constraints;
  std::vector<f_t> reduced(keep.size());
  for (std::size_t k = 0; k < keep.size(); ++k) {
    i_t const row = keep[k];
    if (row < 0 || static_cast<std::size_t>(row) >= values.size()) {
      throw std::invalid_argument(std::string(what) +
                                  ": free-elimination row index is out of range.");
    }
    reduced[k] = values[row];
  }
  values = std::move(reduced);
}

template <typename i_t, typename f_t>
inline std::vector<f_t> crush_user_linear_objective(barrier_transform_t<i_t, f_t> const& xf,
                                                    f_t const* c,
                                                    i_t n)
{
  if (c == nullptr || n != problem_num_cols(xf)) {
    throw std::invalid_argument(
      "update_linear_objective: linear objective length must match the cached problem column "
      "count.");
  }
  if (xf.original_num_cols < xf.user_num_cols) {
    throw std::invalid_argument(
      "update_linear_objective: cached original column count is smaller than user n.");
  }
  if (xf.barrier_lp == nullptr) {
    throw std::invalid_argument("update_linear_objective: cached barrier LP is missing.");
  }
  if (!xf.original_col_to_expanded_col.empty() &&
      static_cast<i_t>(xf.original_col_to_expanded_col.size()) != n) {
    throw std::invalid_argument(
      "update_linear_objective: cached column map does not cover the problem columns.");
  }

  // The expansion leaves the variables it adds out of the objective, so only the positions of
  // the problem's own coefficients move.
  std::vector<f_t> orig(static_cast<std::size_t>(xf.original_num_cols), 0.0);
  for (i_t j = 0; j < n; ++j) {
    i_t const converted_col = expanded_col_to_converted_col(xf, problem_col_to_expanded_col(xf, j));
    if (converted_col < 0 || converted_col >= xf.original_num_cols) {
      throw std::invalid_argument(
        "update_linear_objective: cached column map points outside the converted problem.");
    }
    orig[converted_col] = c[j];
  }
  for (i_t j : xf.presolve_info.negated_variables) {
    orig[j] *= -1.0;
  }

  std::vector<f_t> presolved;
  if (!xf.presolve_info.remaining_variables.empty()) {
    presolved.resize(xf.presolve_info.remaining_variables.size());
    for (std::size_t k = 0; k < xf.presolve_info.remaining_variables.size(); ++k) {
      presolved[k] = orig[xf.presolve_info.remaining_variables[k]];
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
      i_t u        = pairs[2 * k];
      i_t v        = pairs[2 * k + 1];
      presolved[v] = -presolved[u];
    }
  }

  drop_substituted_free_variables<i_t>(xf, presolved, "update_linear_objective");

  if (static_cast<i_t>(presolved.size()) != xf.barrier_lp->num_cols ||
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
// that for the next solve instead of treating it as a validation failure.
enum class crush_rhs_status_t { success = 0, invalid = -1, infeasible = -2 };

template <typename i_t, typename f_t>
inline crush_rhs_status_t crush_user_rhs(barrier_transform_t<i_t, f_t> const& xf,
                                         f_t const* b,
                                         i_t m,
                                         std::vector<f_t>& crushed,
                                         std::string& error)
{
  auto invalid = [&](char const* message) {
    error = message;
    return crush_rhs_status_t::invalid;
  };
  if (m != problem_num_rows(xf) || (b == nullptr && m != 0)) {
    return invalid("update_rhs: RHS length must match the cached problem row count.");
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

  // The quadratic constraints fix the RHS of the appended rows, so an update overwrites the
  // problem's own rows and keeps the cached tail. The tail is empty without an expansion.
  std::vector<f_t> expanded(m);
  for (i_t i = 0; i < m; ++i) {
    expanded[i] = b[i];
  }
  for (f_t rhs : xf.cone_row_rhs) {
    expanded.push_back(rhs);
  }
  if (static_cast<i_t>(expanded.size()) != xf.user_num_rows) {
    return invalid("update_rhs: cached cone-row RHS does not span the expanded rows.");
  }

  // The expansion proved these heads nonnegative from the old RHS. A full solve rejects the
  // problem once that no longer holds, so re-prove it here rather than trust the cached verdict.
  for (simplex::cone_head_bound_t<i_t, f_t> const& bound : xf.cone_head_bounds) {
    f_t implied = -std::numeric_limits<f_t>::infinity();
    for (auto const& [row, coefficient] : bound.rows) {
      implied = std::max(implied, expanded[row] / coefficient);
    }
    if (!(implied >= 0.0)) {
      error = "update_rhs: new RHS no longer implies second-order cone head variable " +
              std::to_string(bound.head_col) + " is nonnegative.";
      return crush_rhs_status_t::invalid;
    }
  }

  // convert turns 'G' rows into 'L' rows by negating the row and its RHS.
  std::vector<f_t> original(static_cast<std::size_t>(xf.original_num_rows));
  const i_t user_num_rows = xf.user_num_rows;
  for (i_t i = 0; i < user_num_rows; ++i) {
    original[i] = xf.row_sense[i] == 'G' ? -expanded[i] : expanded[i];
  }

  // Presolve drops only empty equalities, and only when the RHS is exactly 0.
  for (i_t i : xf.presolve_info.removed_constraints) {
    if (i < 0 || i >= xf.user_num_rows) {
      return invalid("update_rhs: removed constraint index is out of range.");
    }
    if (original[i] != 0.0) { return crush_rhs_status_t::infeasible; }
  }

  // Empty remaining_constraints means either no empty-row pass ran, or every row was dropped
  // and accepted above.
  std::vector<f_t> presolved;
  if (!xf.presolve_info.remaining_constraints.empty()) {
    presolved.resize(xf.presolve_info.remaining_constraints.size());
    for (std::size_t k = 0; k < xf.presolve_info.remaining_constraints.size(); ++k) {
      presolved[k] = original[xf.presolve_info.remaining_constraints[k]];
    }
  } else if (xf.presolve_info.removed_constraints.empty()) {
    presolved = std::move(original);
  }

  substitute_free_variable_rows<i_t>(xf, presolved, "update_rhs");

  if (static_cast<i_t>(presolved.size()) != xf.barrier_lp->num_rows ||
      xf.row_scales.size() != presolved.size()) {
    return invalid("update_rhs: crushed RHS size does not match barrier rows / row_scales.");
  }
  for (std::size_t i = 0; i < presolved.size(); ++i) {
    presolved[i] /= xf.row_scales[i];
  }
  crushed.assign(presolved.begin(), presolved.end());
  return crush_rhs_status_t::success;
}

}  // namespace cuopt::mathematical_optimization
