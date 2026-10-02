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
 * Singleton rows that force a cone head nonnegative. The SOC expansion requires every head to be
 * provably >= 0, and proves it from these rows when the head has no explicit bound, so a new RHS
 * can invalidate a cached expansion. Only heads that need the proof are recorded.
 */
struct cone_head_bound_t {
  int head_col{0};
  // (row, coefficient) pairs, each implying head >= rhs[row] / coefficient.
  std::vector<std::pair<int, double>> rows;
};

/**
 * User-to-barrier transform retained on barrier_cache_t after Optimal:
 * convert / presolve / scaling, plus the scaled LP.
 * Enough to crush new linear objective or RHS data from the original problem into
 * barrier coordinates and to uncrush a solution without rerunning those algorithms.
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
  // cone_var_start after convert, which inserts inequality slacks ahead of the cone block.
  int converted_cone_var_start{0};
  std::vector<int> second_order_cone_dims;
  // Quadratic constraint count the cached expansion was built from.
  int num_quadratic_constraints{0};
  // Dimensions before the QCMATRIX->SOC expansion, which permutes columns and appends rows.
  // Updates arrive in these coordinates, not the expanded ones. Zero when no expansion ran.
  int pre_expansion_num_cols{0};
  int pre_expansion_num_rows{0};
  std::vector<int> original_col_to_expanded_col;
  // RHS of the rows the expansion appended. Fixed by the quadratic constraints, so an RHS
  // update keeps them and only overwrites the model's own rows.
  std::vector<double> cone_row_rhs;
  std::vector<cone_head_bound_t> cone_head_bounds;

  cuopt::mathematical_optimization::simplex::presolve_info_t<int, double> presolve_info;
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  // Barrier linear objective minus crush(user c) from the first solve (Q*ell shift, etc.).
  std::vector<double> linear_obj_shift;
  // Barrier RHS minus crush(user b) from the first solve (fixed/lower-bound shifts).
  std::vector<double> rhs_shift;
  // False when range rows or folding put the user RHS somewhere other than barrier_lp->rhs.
  bool rhs_update_supported{false};
  // Absolute primal tolerance of the first solve, used to test rows presolve dropped as empty.
  double primal_tol{1e-6};
  std::unique_ptr<cuopt::mathematical_optimization::simplex::lp_problem_t<int, double>> barrier_lp;
  // CSC Q with slack columns, as consumed by iteration_data_t. Not the same object as
  // barrier_lp->Q.
  std::unique_ptr<csc_matrix_t<int, double>> barrier_Q;
};

// Dimensions an update is sized in: the cached user counts, or the smaller pre-expansion counts
// when the QCMATRIX->SOC expansion grew the problem.
inline int model_num_cols(barrier_transform_t const& xf)
{
  return xf.pre_expansion_num_cols > 0 ? xf.pre_expansion_num_cols : xf.user_num_cols;
}

inline int model_num_rows(barrier_transform_t const& xf)
{
  return xf.pre_expansion_num_rows > 0 ? xf.pre_expansion_num_rows : xf.user_num_rows;
}

// The expansion permutes model columns into a [linear | cone] layout. An empty map means no
// expansion ran, so the layouts coincide.
inline int model_col_to_expanded_col(barrier_transform_t const& xf, int model_col)
{
  return xf.original_col_to_expanded_col.empty()
           ? model_col
           : xf.original_col_to_expanded_col[static_cast<std::size_t>(model_col)];
}

// convert inserts the inequality slacks ahead of the cone block, pushing every cone column
// right. Mirrors user_col_to_problem_col in presolve.cpp.
inline int expanded_col_to_converted_col(barrier_transform_t const& xf, int expanded_col)
{
  if (xf.second_order_cone_dims.empty() || xf.converted_cone_var_start <= xf.cone_var_start ||
      expanded_col < xf.cone_var_start) {
    return expanded_col;
  }
  return xf.converted_cone_var_start + (expanded_col - xf.cone_var_start);
}

// Move the model's own coefficients into the expanded layout the cached problem is sized for.
template <typename f_t>
std::vector<f_t> scatter_model_objective(barrier_transform_t const& xf,
                                         std::vector<f_t> const& model_objective)
{
  std::vector<f_t> expanded(static_cast<std::size_t>(xf.user_num_cols), f_t(0));
  for (int j = 0; j < static_cast<int>(model_objective.size()); ++j) {
    expanded[static_cast<std::size_t>(model_col_to_expanded_col(xf, j))] =
      model_objective[static_cast<std::size_t>(j)];
  }
  return expanded;
}

// Inverse of scatter_model_objective, giving crush_user_linear_objective the model-sized input
// it expects. The expansion adds no objective coefficients, so nothing is lost.
template <typename f_t>
std::vector<double> gather_model_objective(barrier_transform_t const& xf,
                                           std::vector<f_t> const& expanded_objective)
{
  std::vector<double> model_objective(static_cast<std::size_t>(model_num_cols(xf)));
  for (int j = 0; j < static_cast<int>(model_objective.size()); ++j) {
    model_objective[static_cast<std::size_t>(j)] = static_cast<double>(
      expanded_objective[static_cast<std::size_t>(model_col_to_expanded_col(xf, j))]);
  }
  return model_objective;
}

// Reuse never re-runs the expansion, so the cone block must be laid out exactly as the cached
// one left it.
template <typename i_t, typename f_t>
bool cone_layout_matches(barrier_transform_t const& xf,
                         simplex::user_problem_t<i_t, f_t> const& user_problem)
{
  return user_problem.cone_var_start == static_cast<i_t>(xf.cone_var_start) &&
         user_problem.second_order_cone_dims.size() == xf.second_order_cone_dims.size() &&
         std::equal(user_problem.second_order_cone_dims.begin(),
                    user_problem.second_order_cone_dims.end(),
                    xf.second_order_cone_dims.begin(),
                    [](i_t dim, int cached) { return dim == static_cast<i_t>(cached); });
}

// A cone head with no explicit nonnegative bound is only admissible because some singleton row
// forces it nonnegative. The expansion checks that once; collect the rows it relied on so an RHS
// update can re-check them against the new RHS.
template <typename i_t, typename f_t>
std::vector<cone_head_bound_t> record_cone_head_bounds(
  simplex::user_problem_t<i_t, f_t> const& user_problem)
{
  std::vector<cone_head_bound_t> bounds;
  if (user_problem.second_order_cone_dims.empty()) { return bounds; }

  // Cones only reach here via the expansion, which always sets original_num_rows.
  const auto& A        = user_problem.A;
  const i_t model_rows = user_problem.original_num_rows;
  std::vector<i_t> row_nz(model_rows, 0);
  for (i_t j = 0; j < user_problem.num_cols; ++j) {
    for (i_t p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
      if (A.i[p] < model_rows) { ++row_nz[A.i[p]]; }
    }
  }

  // Only heads that were already model variables carry the precondition. A head the expansion
  // created is nonnegative by cone membership, so no row has to prove it.
  std::vector<char> is_model_col(user_problem.num_cols, 0);
  for (i_t expanded : user_problem.original_col_to_expanded_col) {
    if (expanded >= 0 && expanded < user_problem.num_cols) { is_model_col[expanded] = 1; }
  }

  i_t head = user_problem.cone_var_start;
  for (i_t q_k : user_problem.second_order_cone_dims) {
    if (head < 0 || head >= user_problem.num_cols) { break; }
    if (is_model_col[head] && !(user_problem.lower[head] >= f_t(0))) {
      cone_head_bound_t bound;
      bound.head_col = static_cast<int>(head);
      // A is CSC, so the head's own column already lists every row it appears in.
      for (i_t p = A.col_start[head]; p < A.col_start[head + 1]; ++p) {
        const i_t i = A.i[p];
        if (i >= model_rows || row_nz[i] != 1) { continue; }
        const f_t a      = A.x[p];
        const char sense = user_problem.row_sense[i];
        if ((sense == 'G' && a > f_t(0)) || (sense == 'L' && a < f_t(0))) {
          bound.rows.emplace_back(static_cast<int>(i), static_cast<double>(a));
        }
      }
      bounds.push_back(std::move(bound));
    }
    head += q_k;
  }
  return bounds;
}

inline std::vector<double> crush_user_linear_objective(barrier_transform_t const& xf,
                                                       double const* c,
                                                       int n)
{
  if (c == nullptr || n != model_num_cols(xf)) {
    throw std::invalid_argument(
      "update_linear_objective: linear objective length must match the cached model column count.");
  }
  if (xf.original_num_cols < xf.user_num_cols) {
    throw std::invalid_argument(
      "update_linear_objective: cached original column count is smaller than user n.");
  }
  if (xf.barrier_lp == nullptr) {
    throw std::invalid_argument("update_linear_objective: cached barrier LP is missing.");
  }
  if (!xf.original_col_to_expanded_col.empty() &&
      static_cast<int>(xf.original_col_to_expanded_col.size()) != n) {
    throw std::invalid_argument(
      "update_linear_objective: cached column map does not cover the model columns.");
  }

  // The expansion leaves the variables it adds out of the objective, so only the positions of
  // the model's own coefficients move.
  std::vector<double> orig(static_cast<std::size_t>(xf.original_num_cols), 0.0);
  for (int j = 0; j < n; ++j) {
    int const converted_col = expanded_col_to_converted_col(xf, model_col_to_expanded_col(xf, j));
    if (converted_col < 0 || converted_col >= xf.original_num_cols) {
      throw std::invalid_argument(
        "update_linear_objective: cached column map points outside the converted problem.");
    }
    orig[static_cast<std::size_t>(converted_col)] = c[j];
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

// Distinct from the invalid_argument cases so the caller can report INFEASIBLE rather than a
// validation failure.
struct update_rhs_infeasible_error : std::runtime_error {
  explicit update_rhs_infeasible_error(std::string const& message) : std::runtime_error(message) {}
};

inline std::vector<double> crush_user_rhs(barrier_transform_t const& xf, double const* b, int m)
{
  if (b == nullptr || m != model_num_rows(xf)) {
    throw std::invalid_argument("update_rhs: RHS length must match the cached model row count.");
  }
  if (!xf.rhs_update_supported) {
    throw std::invalid_argument(
      "update_rhs: cached convert used range rows or folding; run a full Solve.");
  }
  if (xf.original_num_rows != xf.user_num_rows) {
    throw std::invalid_argument(
      "update_rhs: cached original row count does not match the user row count.");
  }
  if (static_cast<int>(xf.row_sense.size()) != xf.user_num_rows) {
    throw std::invalid_argument(
      "update_rhs: cached row-sense count does not match the user row count.");
  }
  if (xf.barrier_lp == nullptr) {
    throw std::invalid_argument("update_rhs: cached barrier LP is missing.");
  }

  // The quadratic constraints fix the RHS of the appended rows, so an update overwrites the
  // model's own rows and keeps the cached tail. The tail is empty without an expansion.
  std::vector<double> expanded(b, b + m);
  expanded.insert(expanded.end(), xf.cone_row_rhs.begin(), xf.cone_row_rhs.end());
  if (static_cast<int>(expanded.size()) != xf.user_num_rows) {
    throw std::invalid_argument("update_rhs: cached cone-row RHS does not span the expanded rows.");
  }

  // The expansion proved these heads nonnegative from the old RHS. A full solve rejects the
  // model once that no longer holds, so re-prove it here rather than trust the cached verdict.
  for (cone_head_bound_t const& bound : xf.cone_head_bounds) {
    double implied = -std::numeric_limits<double>::infinity();
    for (auto const& [row, coefficient] : bound.rows) {
      implied = std::max(implied, expanded[static_cast<std::size_t>(row)] / coefficient);
    }
    if (!(implied >= 0.0)) {
      throw std::invalid_argument(
        "update_rhs: new RHS no longer implies second-order cone head variable " +
        std::to_string(bound.head_col) + " is nonnegative.");
    }
  }

  // convert turns 'G' rows into 'L' rows by negating the row and its RHS.
  std::vector<double> original(static_cast<std::size_t>(xf.original_num_rows));
  for (int i = 0; i < xf.user_num_rows; ++i) {
    original[static_cast<std::size_t>(i)] =
      xf.row_sense[static_cast<std::size_t>(i)] == 'G' ? -expanded[i] : expanded[i];
  }

  // Dropped rows were empty, so the new RHS never reaches the barrier: 'E' needs 0 == b_i and
  // the rest need 0 <= b_i.
  for (int i : xf.presolve_info.removed_constraints) {
    if (i < 0 || i >= xf.user_num_rows) {
      throw std::invalid_argument("update_rhs: removed constraint index is out of range.");
    }
    double const converted_rhs = original[static_cast<std::size_t>(i)];
    bool const infeasible      = xf.row_sense[static_cast<std::size_t>(i)] == 'E'
                                   ? std::abs(converted_rhs) > xf.primal_tol
                                   : converted_rhs < -xf.primal_tol;
    if (infeasible) {
      throw update_rhs_infeasible_error("update_rhs: empty constraint row " + std::to_string(i) +
                                        " is infeasible with the new RHS.");
    }
  }

  // Empty remaining_constraints means either no empty-row pass ran, or every row was dropped
  // and accepted above.
  std::vector<double> presolved;
  if (!xf.presolve_info.remaining_constraints.empty()) {
    presolved.resize(xf.presolve_info.remaining_constraints.size());
    for (std::size_t k = 0; k < xf.presolve_info.remaining_constraints.size(); ++k) {
      presolved[k] = original[static_cast<std::size_t>(xf.presolve_info.remaining_constraints[k])];
    }
  } else if (xf.presolve_info.removed_constraints.empty()) {
    presolved = std::move(original);
  }

  if (static_cast<int>(presolved.size()) != xf.barrier_lp->num_rows ||
      xf.row_scales.size() != presolved.size()) {
    throw std::invalid_argument(
      "update_rhs: crushed RHS size does not match barrier rows / row_scales.");
  }
  for (std::size_t i = 0; i < presolved.size(); ++i) {
    presolved[i] /= xf.row_scales[i];
  }
  return presolved;
}

}  // namespace cuopt::mathematical_optimization
