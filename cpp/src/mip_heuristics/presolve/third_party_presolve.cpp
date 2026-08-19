/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Papilo's ProbingView::reset() guards bounds restoration with #ifndef NDEBUG.
// This causes invalid (-1) column indices due to bugs in the Probing presolver.
// Force-include ProbingView.hpp with NDEBUG undefined so the restoration is compiled in.
#ifdef NDEBUG
#undef NDEBUG
#include <papilo/core/ProbingView.hpp>
#define NDEBUG
#endif

#include <PSLP/PSLP_sol.h>
#include <PSLP/PSLP_stats.h>
#include <PSLP/PSLP_status.h>
#include <cuopt/error.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#pragma clang diagnostic ignored "-Wimplicit-const-int-float-conversion"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
#include <papilo/core/Presolve.hpp>
#include <papilo/core/ProblemBuilder.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/gf2_presolve.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>
#include <utilities/timer.hpp>

#include <raft/core/nvtx.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <span>
#include <tuple>
#include <unordered_map>

namespace cuopt::mathematical_optimization::mip {

// Backend-agnostic normalisation of the mutable presolve fields:
//   * sign-flip `obj_coeffs` / `objective_offset` when maximise,
//   * materialise ranged `constr_lb` / `constr_ub` from `row_types` +
//     `constraint_bounds` when the ranged pair is absent from the mps.
//
// Precondition: the caller has already copy-initialised each vector from with mps data
template <typename i_t, typename f_t>
void normalize_for_presolve(io::mps_data_model_t<i_t, f_t> const& mps,
                            bool maximize,
                            std::vector<f_t>& obj_coeffs,
                            f_t& objective_offset,
                            std::vector<f_t>& var_lb,
                            std::vector<f_t>& var_ub,
                            std::vector<f_t>& constr_lb,
                            std::vector<f_t>& constr_ub)
{
  if (maximize) {
    for (auto& c : obj_coeffs) {
      c = -c;
    }
    objective_offset = -objective_offset;
  }

  if (constr_lb.empty() && constr_ub.empty()) {
    const auto& row_types         = mps.get_row_types();
    const auto& constraint_bounds = mps.get_constraint_bounds();
    for (size_t i = 0; i < row_types.size(); ++i) {
      if (row_types[i] == 'L') {
        constr_lb.push_back(-std::numeric_limits<f_t>::infinity());
        constr_ub.push_back(constraint_bounds[i]);
      } else if (row_types[i] == 'G') {
        constr_lb.push_back(constraint_bounds[i]);
        constr_ub.push_back(std::numeric_limits<f_t>::infinity());
      } else if (row_types[i] == 'E') {
        constr_lb.push_back(constraint_bounds[i]);
        constr_ub.push_back(constraint_bounds[i]);
      }
    }
  }
}

// Build a papilo::Problem
template <typename i_t, typename f_t>
papilo::Problem<f_t> build_papilo_problem(io::mps_data_model_t<i_t, f_t> const& mps,
                                          bool maximize,
                                          problem_category_t category)
{
  raft::common::nvtx::range fun_scope("Build papilo::Problem from mps_data_model");

  const i_t n_cols = mps.get_n_variables();
  const i_t n_rows = mps.get_n_constraints();
  const i_t nnz    = mps.get_nnz();

  std::vector<f_t> obj_coeffs(mps.get_objective_coefficients());
  std::vector<f_t> var_lb(mps.get_variable_lower_bounds());
  std::vector<f_t> var_ub(mps.get_variable_upper_bounds());
  std::vector<f_t> constr_lb(mps.get_constraint_lower_bounds());
  std::vector<f_t> constr_ub(mps.get_constraint_upper_bounds());
  f_t objective_offset = mps.get_objective_offset();
  normalize_for_presolve<i_t, f_t>(
    mps, maximize, obj_coeffs, objective_offset, var_lb, var_ub, constr_lb, constr_ub);

  const auto& coefficients   = mps.get_constraint_matrix_values();
  const auto& indices        = mps.get_constraint_matrix_indices();
  const auto& offsets        = mps.get_constraint_matrix_offsets();
  const auto& variable_names = mps.get_variable_names();
  const auto& mps_var_types  = mps.get_variable_types();

  papilo::ProblemBuilder<f_t> builder;
  builder.reserve(nnz, n_rows, n_cols);
  builder.setNumCols(n_cols);
  builder.setNumRows(n_rows);
  builder.setObjAll(obj_coeffs);
  builder.setObjOffset(objective_offset);

  if (!var_lb.empty() && !var_ub.empty()) {
    builder.setColLbAll(var_lb);
    builder.setColUbAll(var_ub);
    if (variable_names.size() == static_cast<size_t>(n_cols)) {
      builder.setColNameAll(variable_names);
    }
  }

  if (category == problem_category_t::MIP) {
    for (size_t i = 0; i < mps_var_types.size(); ++i) {
      builder.setColIntegral(i, char_to_var_type(mps_var_types[i]) == var_t::INTEGER);
    }
  }

  if (!constr_lb.empty() && !constr_ub.empty()) {
    builder.setRowLhsAll(constr_lb);
    builder.setRowRhsAll(constr_ub);
  }

  std::vector<papilo::RowFlags> h_row_flags(constr_lb.size());
  std::vector<std::tuple<i_t, i_t, f_t>> h_entries;
  for (size_t i = 0; i < constr_lb.size(); ++i) {
    const i_t row_start   = offsets[i];
    const i_t row_end     = offsets[i + 1];
    const i_t num_entries = row_end - row_start;
    for (size_t j = 0; j < num_entries; ++j) {
      h_entries.push_back(std::make_tuple(i, indices[row_start + j], coefficients[row_start + j]));
    }

    if (constr_lb[i] == -std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kLhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kLhsInf);
    }
    if (constr_ub[i] == std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kRhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kRhsInf);
    }

    // Papilo stores finite dummies in place of ±inf; the flags above are the
    // source of truth for infinity. Zero out the dummies before setConstraintMatrix.
    if (constr_lb[i] == -std::numeric_limits<f_t>::infinity()) { constr_lb[i] = 0; }
    if (constr_ub[i] == std::numeric_limits<f_t>::infinity()) { constr_ub[i] = 0; }
  }

  for (size_t i = 0; i < var_lb.size(); ++i) {
    builder.setColLbInf(i, var_lb[i] == -std::numeric_limits<f_t>::infinity());
    builder.setColUbInf(i, var_ub[i] == std::numeric_limits<f_t>::infinity());
    if (var_lb[i] == -std::numeric_limits<f_t>::infinity()) { builder.setColLb(i, 0); }
    if (var_ub[i] == std::numeric_limits<f_t>::infinity()) { builder.setColUb(i, 0); }
  }

  auto problem = builder.build();

  if (h_entries.size()) {
    auto constexpr const sorted_entries = true;
    // MIP reductions like clique merging and substituition require more fillin
    const double spare_ratio      = category == problem_category_t::MIP ? 4.0 : 2.0;
    const int min_inter_row_space = category == problem_category_t::MIP ? 30 : 4;
    auto csr_storage              = papilo::SparseStorage<f_t>(
      h_entries, n_rows, n_cols, sorted_entries, spare_ratio, min_inter_row_space);
    problem.setConstraintMatrix(csr_storage, constr_lb, constr_ub, h_row_flags);

    papilo::ConstraintMatrix<f_t>& matrix = problem.getConstraintMatrix();
    for (int i = 0; i < problem.getNRows(); ++i) {
      papilo::RowFlags rowFlag = matrix.getRowFlags()[i];
      if (!rowFlag.test(papilo::RowFlag::kRhsInf) && !rowFlag.test(papilo::RowFlag::kLhsInf) &&
          matrix.getLeftHandSides()[i] == matrix.getRightHandSides()[i])
        matrix.getRowFlags()[i].set(papilo::RowFlag::kEquation);
    }
  }

  return problem;
}

// NOTE: Parallel conversion helpers for mps_data_model_t and
// simplex::user_problem_t are temporary duplication until we settle on a
// unified problem representation. Keep both working for now.
template <typename i_t, typename f_t>
papilo::Problem<f_t> build_papilo_problem(const simplex::user_problem_t<i_t, f_t>& problem)
{
  raft::common::nvtx::range fun_scope("Build papilo problem");
  // Build a papilo problem from a (host-side) dual-simplex user_problem_t. Unlike the
  // optimization_problem_t overload, all data already lives on the host and the constraint
  // matrix is stored column-major (CSC), so there are no device copies and no COO step: the
  // CSC matrix is converted once to CSR and handed straight to papilo's SparseStorage.
  papilo::ProblemBuilder<f_t> builder;

  const i_t num_cols = problem.num_cols;
  const i_t num_rows = problem.num_rows;
  const i_t nnz      = problem.A.nnz();

  builder.reserve(nnz, num_rows, num_cols);

  const std::vector<f_t>& obj_coeffs                     = problem.objective;
  const std::vector<f_t>& var_lb                         = problem.lower;
  const std::vector<f_t>& var_ub                         = problem.upper;
  const std::vector<simplex::variable_type_t>& var_types = problem.var_types;
  const std::vector<char>& row_sense                     = problem.row_sense;
  const std::vector<f_t>& rhs                            = problem.rhs;

  // Range rows carry an extra width and are listed separately; mark them so the row-bound
  // derivation below matches convert_user_problem in dual_simplex/presolve.cpp. papilo
  // represents ranged rows natively as two-sided lhs <= a^T x <= rhs, so we do not add slack
  // columns.
  std::vector<f_t> range_of_row(num_rows, 0);
  std::vector<bool> is_range_row(num_rows, false);
  for (i_t k = 0; k < problem.num_range_rows; ++k) {
    const i_t row     = problem.range_rows[k];
    is_range_row[row] = true;
    range_of_row[row] = problem.range_value[k];
  }

  // Derive two-sided row bounds [lhs, rhs] from the row sense.
  std::vector<f_t> h_constr_lb(num_rows);
  std::vector<f_t> h_constr_ub(num_rows);
  for (i_t i = 0; i < num_rows; ++i) {
    const f_t b = rhs[i];
    if (is_range_row[i]) {
      auto [lower, upper] = simplex::get_range_bounds_from_sense(row_sense[i], b, range_of_row[i]);
      h_constr_lb[i]      = lower;
      h_constr_ub[i]      = upper;
    } else if (row_sense[i] == 'L') {
      h_constr_lb[i] = -std::numeric_limits<f_t>::infinity();
      h_constr_ub[i] = b;
    } else if (row_sense[i] == 'G') {
      h_constr_lb[i] = b;
      h_constr_ub[i] = std::numeric_limits<f_t>::infinity();
    } else {  // 'E'
      h_constr_lb[i] = b;
      h_constr_ub[i] = b;
    }
  }

  builder.setNumCols(num_cols);
  builder.setNumRows(num_rows);

  // user_problem_t stores the objective already in minimization sense (obj_scale carries the
  // original min/max direction for reporting only), so no sign flip is needed here.
  builder.setObjAll(obj_coeffs);
  builder.setObjOffset(problem.obj_constant);

  if (!var_lb.empty() && !var_ub.empty()) {
    builder.setColLbAll(var_lb);
    builder.setColUbAll(var_ub);
    if (static_cast<i_t>(problem.col_names.size()) == num_cols) {
      builder.setColNameAll(problem.col_names);
    }
  }

  for (i_t j = 0; j < num_cols; ++j) {
    builder.setColIntegral(j, var_types[j] != simplex::variable_type_t::CONTINUOUS);
  }

  // Row bounds + infinity flags, set on the builder so build() materializes the constraint
  // matrix directly. build() also sets RowFlag::kEquation where a finite lhs == rhs.
  if (num_rows > 0) {
    builder.setRowLhsAll(h_constr_lb);
    builder.setRowRhsAll(h_constr_ub);
  }
  // Per-row inf flags (mirrors the optimization_problem_t overload). The zeroed lhs/rhs and the
  // flags are handed to setConstraintMatrix below.
  std::vector<papilo::RowFlags> h_row_flags(num_rows);
  for (i_t i = 0; i < num_rows; ++i) {
    const bool lhs_inf = h_constr_lb[i] == -std::numeric_limits<f_t>::infinity();
    const bool rhs_inf = h_constr_ub[i] == std::numeric_limits<f_t>::infinity();
    if (lhs_inf) {
      h_row_flags[i].set(papilo::RowFlag::kLhsInf);
      h_constr_lb[i] = 0;
    }
    if (rhs_inf) {
      h_row_flags[i].set(papilo::RowFlag::kRhsInf);
      h_constr_ub[i] = 0;
    }
  }

  for (i_t j = 0; j < num_cols; ++j) {
    builder.setColLbInf(j, var_lb[j] == -std::numeric_limits<f_t>::infinity());
    builder.setColUbInf(j, var_ub[j] == std::numeric_limits<f_t>::infinity());
    if (var_lb[j] == -std::numeric_limits<f_t>::infinity()) { builder.setColLb(j, 0); }
    if (var_ub[j] == std::numeric_limits<f_t>::infinity()) { builder.setColUb(j, 0); }
  }

  // Assemble COO entries (row, col, value) from the CSC storage and hand the matrix to papilo via
  // SparseStorage with the MIP fill-in headroom, exactly like the optimization_problem_t overload.
  // The default ProblemBuilder path (addColEntries) omits that headroom, which leaves papilo's
  // in-place presolve in a state where DualInfer can assert on a row it reduced.
  std::vector<std::tuple<i_t, i_t, f_t>> h_entries;
  h_entries.reserve(nnz);
  const std::vector<i_t>& col_start = problem.A.col_start;
  const std::vector<i_t>& row_index = problem.A.i;
  const std::vector<f_t>& values    = problem.A.x;
  for (i_t j = 0; j < num_cols; ++j) {
    for (i_t p = col_start[j]; p < col_start[j + 1]; ++p) {
      h_entries.push_back(std::make_tuple(row_index[p], j, values[p]));
    }
  }

  auto papilo_problem = builder.build();
  if (!h_entries.empty()) {
    // CSC iteration is column-major, so entries are not row-sorted; let papilo sort them.
    constexpr bool sorted_entries = false;
    // MIP reductions like clique merging and substitution require more fillin.
    const double spare_ratio      = 10.0;
    const int min_inter_row_space = 30;
    auto csr_storage              = papilo::SparseStorage<f_t>(
      h_entries, num_rows, num_cols, sorted_entries, spare_ratio, min_inter_row_space);
    papilo_problem.setConstraintMatrix(csr_storage, h_constr_lb, h_constr_ub, h_row_flags);

    papilo::ConstraintMatrix<f_t>& matrix = papilo_problem.getConstraintMatrix();
    for (i_t i = 0; i < papilo_problem.getNRows(); ++i) {
      papilo::RowFlags rowFlag = matrix.getRowFlags()[i];
      if (!rowFlag.test(papilo::RowFlag::kRhsInf) && !rowFlag.test(papilo::RowFlag::kLhsInf) &&
          matrix.getLeftHandSides()[i] == matrix.getRightHandSides()[i])
        matrix.getRowFlags()[i].set(papilo::RowFlag::kEquation);
    }
  }

  return papilo_problem;
}

// Read a reduced (presolved) papilo problem back into a host-side dual-simplex user_problem_t,
// overwriting `problem` in place.
template <typename i_t, typename f_t>
void build_user_problem(papilo::Problem<f_t> const& papilo_problem,
                        simplex::user_problem_t<i_t, f_t>& problem)
{
  raft::common::nvtx::range fun_scope("Build user problem");

  const i_t reduced_rows        = papilo_problem.getNRows();
  const i_t reduced_cols        = papilo_problem.getNCols();
  auto const& constraint_matrix = papilo_problem.getConstraintMatrix();

  // Objective (already minimization sense).
  auto const& obj = papilo_problem.getObjective();
  problem.objective.assign(obj.coefficients.begin(), obj.coefficients.end());
  problem.obj_constant = obj.offset;

  // Column bounds and integrality.
  auto const& col_lower = papilo_problem.getLowerBounds();
  auto const& col_upper = papilo_problem.getUpperBounds();
  auto const& col_flags = papilo_problem.getColFlags();
  problem.lower.resize(reduced_cols);
  problem.upper.resize(reduced_cols);
  problem.var_types.resize(reduced_cols);
  for (i_t j = 0; j < reduced_cols; ++j) {
    problem.lower[j]     = col_flags[j].test(papilo::ColFlag::kLbInf)
                             ? -std::numeric_limits<f_t>::infinity()
                             : col_lower[j];
    problem.upper[j]     = col_flags[j].test(papilo::ColFlag::kUbInf)
                             ? std::numeric_limits<f_t>::infinity()
                             : col_upper[j];
    problem.var_types[j] = col_flags[j].test(papilo::ColFlag::kIntegral)
                             ? simplex::variable_type_t::INTEGER
                             : simplex::variable_type_t::CONTINUOUS;
  }

  // Row sense / rhs / ranges -- inverse of the derivation in build_papilo_problem_mip.
  auto const& lhs       = constraint_matrix.getLeftHandSides();
  auto const& rhs_v     = constraint_matrix.getRightHandSides();
  auto const& row_flags = constraint_matrix.getRowFlags();

  problem.row_sense.clear();
  problem.rhs.clear();
  problem.row_sense.reserve(reduced_rows);
  problem.rhs.reserve(reduced_rows);
  problem.range_rows.clear();
  problem.range_value.clear();
  for (i_t r = 0; r < reduced_rows; ++r) {
    const bool lhs_inf = row_flags[r].test(papilo::RowFlag::kLhsInf);
    const bool rhs_inf = row_flags[r].test(papilo::RowFlag::kRhsInf);
    const bool eq      = row_flags[r].test(papilo::RowFlag::kEquation);
    if (eq || (!lhs_inf && !rhs_inf && lhs[r] == rhs_v[r])) {
      problem.row_sense.push_back('E');
      problem.rhs.push_back(rhs_v[r]);
    } else if (lhs_inf && !rhs_inf) {
      problem.row_sense.push_back('L');
      problem.rhs.push_back(rhs_v[r]);
    } else if (!lhs_inf && rhs_inf) {
      problem.row_sense.push_back('G');
      problem.rhs.push_back(lhs[r]);
    } else if (!lhs_inf && !rhs_inf) {
      problem.row_sense.push_back('E');
      problem.rhs.push_back(lhs[r]);
      problem.range_rows.push_back(r);
      problem.range_value.push_back(rhs_v[r] - lhs[r]);
    } else {
      assert(false && "Papilo should remove all the free rows");
    }
  }

  problem.num_range_rows = problem.range_rows.size();

  // Constraint matrix: read papilo's column-major (CSC) transpose straight into A, packing out
  // the spare gaps SparseStorage leaves between columns.
  const i_t reduced_nnz = constraint_matrix.getNnz();
  problem.A.resize(reduced_rows, reduced_cols, reduced_nnz);
  auto const& csc        = constraint_matrix.getMatrixTranspose();
  const auto* col_ranges = csc.getRowRanges();  // per-column [start, end)
  const int* row_indices = csc.getColumns();    // transpose columns == original rows
  const f_t* values      = csc.getValues();
  i_t pos                = 0;
  for (i_t j = 0; j < reduced_cols; ++j) {
    problem.A.col_start[j] = pos;
    for (i_t p = col_ranges[j].start; p < col_ranges[j].end; ++p) {
      problem.A.i[pos] = row_indices[p];
      problem.A.x[pos] = values[p];
      ++pos;
    }
  }
  problem.A.col_start[reduced_cols] = pos;
  cuopt_assert(pos == reduced_nnz, "papilo CSC nonzero count mismatch");

  problem.num_cols = reduced_cols;
  problem.num_rows = reduced_rows;
}

template <typename i_t, typename f_t>
void papilo_round_trip(simplex::user_problem_t<i_t, f_t>& problem)
{
  papilo::Problem<f_t> papilo_problem = build_papilo_problem(problem);
  build_user_problem(papilo_problem, problem);
}

// Presolved mps_data_model builder from PSLP
template <typename i_t, typename f_t>
io::mps_data_model_t<i_t, f_t> build_reduced_mps_from_pslp(Presolver* pslp_presolver,
                                                           bool maximize,
                                                           f_t original_obj_offset)
{
  raft::common::nvtx::range fun_scope("Build mps_data_model from PSLP");
  io::mps_data_model_t<i_t, f_t> mps;

  if constexpr (std::is_same_v<f_t, double>) {
    auto* reduced    = pslp_presolver->reduced_prob;
    const i_t n_rows = static_cast<i_t>(reduced->m);
    const i_t n_cols = static_cast<i_t>(reduced->n);
    const i_t nnz    = static_cast<i_t>(reduced->nnz);
    // PSLP folds the sign flip into obj_offset for maximise problems, and does
    // not track the original mps's objective offset — put both back.
    const f_t obj_offset =
      (maximize ? -reduced->obj_offset : reduced->obj_offset) + original_obj_offset;
    mps.set_maximize(maximize);
    mps.set_objective_offset(obj_offset);

    if (n_cols == 0 && n_rows == 0) {
      std::vector<i_t> empty_offsets{0};
      mps.set_csr_constraint_matrix(
        {}, {}, std::span<const i_t>(empty_offsets.data(), empty_offsets.size()));
      return mps;
    }

    mps.set_csr_constraint_matrix(
      std::span<const f_t>(reduced->Ax, static_cast<size_t>(nnz)),
      std::span<const i_t>(reduced->Ai, static_cast<size_t>(nnz)),
      std::span<const i_t>(reduced->Ap, static_cast<size_t>(n_rows + 1)));

    if (maximize) {
      std::vector<f_t> h_obj_coeffs(reduced->c, reduced->c + n_cols);
      for (auto& c : h_obj_coeffs) {
        c = -c;
      }
      mps.set_objective_coefficients(
        std::span<const f_t>(h_obj_coeffs.data(), h_obj_coeffs.size()));
    } else {
      mps.set_objective_coefficients(std::span<const f_t>(reduced->c, static_cast<size_t>(n_cols)));
    }
    mps.set_constraint_lower_bounds(
      std::span<const f_t>(reduced->lhs, static_cast<size_t>(n_rows)));
    mps.set_constraint_upper_bounds(
      std::span<const f_t>(reduced->rhs, static_cast<size_t>(n_rows)));
    mps.set_variable_lower_bounds(std::span<const f_t>(reduced->lbs, static_cast<size_t>(n_cols)));
    mps.set_variable_upper_bounds(std::span<const f_t>(reduced->ubs, static_cast<size_t>(n_cols)));
  } else {
    cuopt_expects(false, error_type_t::ValidationError, "PSLP only supports double precision");
  }

  return mps;
}

template <typename i_t, typename f_t>
io::mps_data_model_t<i_t, f_t> build_reduced_mps_from_papilo(
  papilo::Problem<f_t> const& papilo_problem, bool maximize)
{
  raft::common::nvtx::range fun_scope("Reduced mps <- Papilo");
  io::mps_data_model_t<i_t, f_t> mps;

  auto obj = papilo_problem.getObjective();
  mps.set_maximize(maximize);
  mps.set_objective_offset(maximize ? -obj.offset : obj.offset);

  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) {
    std::vector<i_t> empty_offsets{0};
    mps.set_csr_constraint_matrix(
      {}, {}, std::span<const i_t>(empty_offsets.data(), empty_offsets.size()));
    return mps;
  }

  if (maximize) {
    for (auto& c : obj.coefficients) {
      c = -c;
    }
  }
  mps.set_objective_coefficients(
    std::span<const f_t>(obj.coefficients.data(), obj.coefficients.size()));

  auto& constraint_matrix = papilo_problem.getConstraintMatrix();

  // Row bounds: copy out (papilo returns by value) then substitute ±inf per flag.
  auto row_lower = constraint_matrix.getLeftHandSides();
  auto row_upper = constraint_matrix.getRightHandSides();
  auto row_flags = constraint_matrix.getRowFlags();
  for (size_t i = 0; i < row_flags.size(); i++) {
    if (row_flags[i].test(papilo::RowFlag::kLhsInf)) {
      row_lower[i] = -std::numeric_limits<f_t>::infinity();
    }
    if (row_flags[i].test(papilo::RowFlag::kRhsInf)) {
      row_upper[i] = std::numeric_limits<f_t>::infinity();
    }
  }
  mps.set_constraint_lower_bounds(std::span<const f_t>(row_lower.data(), row_lower.size()));
  mps.set_constraint_upper_bounds(std::span<const f_t>(row_upper.data(), row_upper.size()));

  // CSR offsets have to be synthesised from papilo's RangeInfo (non-contiguous
  // in general); values and column indices are contiguous, so span in-place.
  auto [index_range, nrows] = constraint_matrix.getRangeInfo();
  std::vector<i_t> offsets(nrows + 1);
  const size_t start = index_range[0].start;
  for (i_t i = 0; i < nrows; i++) {
    offsets[i] = static_cast<i_t>(index_range[i].start - start);
  }
  offsets[nrows] = static_cast<i_t>(index_range[nrows - 1].end - start);
  const i_t nnz  = static_cast<i_t>(constraint_matrix.getNnz());
  assert(offsets[nrows] == nnz);
  const int* cols   = constraint_matrix.getConstraintMatrix().getColumns();
  const f_t* coeffs = constraint_matrix.getConstraintMatrix().getValues();
  mps.set_csr_constraint_matrix(std::span<const f_t>(&coeffs[start], static_cast<size_t>(nnz)),
                                std::span<const i_t>(&cols[start], static_cast<size_t>(nnz)),
                                std::span<const i_t>(offsets.data(), offsets.size()));

  // Col bounds + var_types: same copy-then-fixup pattern.
  auto col_lower = papilo_problem.getLowerBounds();
  auto col_upper = papilo_problem.getUpperBounds();
  auto col_flags = papilo_problem.getColFlags();
  std::vector<char> var_types(col_flags.size());
  for (size_t i = 0; i < col_flags.size(); i++) {
    var_types[i] = col_flags[i].test(papilo::ColFlag::kIntegral) ? 'I' : 'C';
    if (col_flags[i].test(papilo::ColFlag::kLbInf)) {
      col_lower[i] = -std::numeric_limits<f_t>::infinity();
    }
    if (col_flags[i].test(papilo::ColFlag::kUbInf)) {
      col_upper[i] = std::numeric_limits<f_t>::infinity();
    }
  }
  mps.set_variable_lower_bounds(std::span<const f_t>(col_lower.data(), col_lower.size()));
  mps.set_variable_upper_bounds(std::span<const f_t>(col_upper.data(), col_upper.size()));
  mps.set_variable_types(var_types);

  return mps;
}

void check_presolve_status(const papilo::PresolveStatus& status)
{
  switch (status) {
    case papilo::PresolveStatus::kUnchanged:
      CUOPT_LOG_INFO("Presolve status: did not result in any changes");
      break;
    case papilo::PresolveStatus::kReduced:
      CUOPT_LOG_INFO("Presolve status: reduced the problem");
      break;
    case papilo::PresolveStatus::kUnbndOrInfeas:
      CUOPT_LOG_INFO("Presolve status: found an unbounded or infeasible problem");
      break;
    case papilo::PresolveStatus::kInfeasible:
      CUOPT_LOG_INFO("Presolve status: found an infeasible problem");
      break;
    case papilo::PresolveStatus::kUnbounded:
      CUOPT_LOG_INFO("Presolve status: found an unbounded problem");
      break;
  }
}

third_party_presolve_status_t convert_papilo_presolve_status_to_third_party_presolve_status(
  const papilo::PresolveStatus& status)
{
  switch (status) {
    case papilo::PresolveStatus::kUnchanged: return third_party_presolve_status_t::UNCHANGED;
    case papilo::PresolveStatus::kReduced: return third_party_presolve_status_t::REDUCED;
    case papilo::PresolveStatus::kUnbndOrInfeas:
      return third_party_presolve_status_t::UNBNDORINFEAS;
    case papilo::PresolveStatus::kInfeasible: return third_party_presolve_status_t::INFEASIBLE;
    case papilo::PresolveStatus::kUnbounded:
      return third_party_presolve_status_t::UNBOUNDED;
      // Do not implement default case to trigger compile time error if new enum is added
  }
  return third_party_presolve_status_t::UNCHANGED;
}

third_party_presolve_status_t convert_pslp_presolve_status_to_third_party_presolve_status(
  const PresolveStatus& status)
{
  switch (status) {
    case PresolveStatus_::UNCHANGED: return third_party_presolve_status_t::UNCHANGED;
    case PresolveStatus_::REDUCED: return third_party_presolve_status_t::REDUCED;
    case PresolveStatus_::INFEASIBLE: return third_party_presolve_status_t::INFEASIBLE;
    case PresolveStatus_::UNBNDORINFEAS:
      return third_party_presolve_status_t::UNBNDORINFEAS;
      // Do not implement default case to trigger compile time error if new enum is added
  }
  return third_party_presolve_status_t::UNCHANGED;
}

void check_postsolve_status(const papilo::PostsolveStatus& status)
{
  switch (status) {
    case papilo::PostsolveStatus::kOk: CUOPT_LOG_DEBUG("Post-solve status: succeeded"); break;
    case papilo::PostsolveStatus::kFailed:
      CUOPT_LOG_INFO(
        "Post-solve status: Post solved solution violates constraints. This is most likely due to "
        "different tolerances.");
      break;
  }
}

template <typename f_t>
void set_presolve_methods(
  papilo::Presolve<f_t>& presolver,
  problem_category_t category,
  bool dual_postsolve,
  std::optional<std::unordered_set<std::string>> const& method_allowlist = std::nullopt)
{
  using uptr = std::unique_ptr<papilo::PresolveMethod<f_t>>;

  auto maybe_add = [&](uptr method) {
    if (method_allowlist.has_value()) {
      const std::string& name = method->getName();
      if (!method_allowlist->count(name)) { return; }
    }
    presolver.addPresolveMethod(std::move(method));
  };

  if (category == problem_category_t::MIP) {
    // cuOpt custom GF2 presolver
    maybe_add(uptr(new cuopt::mathematical_optimization::mip::GF2Presolve<f_t>()));
  }
  // fast presolvers
  maybe_add(uptr(new papilo::SingletonCols<f_t>()));
  maybe_add(uptr(new papilo::CoefficientStrengthening<f_t>()));
  maybe_add(uptr(new papilo::ConstraintPropagation<f_t>()));

  // medium presolvers
  maybe_add(uptr(new papilo::FixContinuous<f_t>()));
  maybe_add(uptr(new papilo::SimpleProbing<f_t>()));
  maybe_add(uptr(new papilo::ParallelRowDetection<f_t>()));
  maybe_add(uptr(new papilo::ParallelColDetection<f_t>()));
  maybe_add(uptr(new papilo::DualFix<f_t>()));
  maybe_add(uptr(new papilo::SimplifyInequalities<f_t>()));
  maybe_add(uptr(new papilo::CliqueMerging<f_t>()));

  // exhaustive presolvers
  maybe_add(uptr(new papilo::ImplIntDetection<f_t>()));
  maybe_add(uptr(new papilo::DominatedCols<f_t>()));
  maybe_add(uptr(new papilo::Probing<f_t>()));

  if (!dual_postsolve) {
    // SingletonStuffing causes dual crushing failures on:
    //   tr12-30, ns1208400, gmu-35-50, dws008-01, neos-1445765,
    //   neos-5107597-kakapo, rocI-4-11, traininstance2, traininstance6,
    //   radiationm18-12-05, rococoB10-011000, b1c1s1
    maybe_add(uptr(new papilo::SingletonStuffing<f_t>()));
    maybe_add(uptr(new papilo::DualInfer<f_t>()));
    maybe_add(uptr(new papilo::SimpleSubstitution<f_t>()));
    maybe_add(uptr(new papilo::Sparsify<f_t>()));
    maybe_add(uptr(new papilo::Substitution<f_t>()));
  } else {
    CUOPT_LOG_INFO("Disabling the presolver methods that do not support dual postsolve");
  }
}

template <typename i_t, typename f_t>
void set_presolve_options(papilo::Presolve<f_t>& presolver,
                          problem_category_t category,
                          f_t absolute_tolerance,
                          f_t relative_tolerance,
                          f_t time_limit,
                          bool dual_postsolve,
                          i_t num_cpu_threads,
                          i_t max_rounds)
{
  presolver.getPresolveOptions().tlim    = time_limit;
  presolver.getPresolveOptions().threads = num_cpu_threads;  //  user setting or  0 (automatic)
  presolver.getPresolveOptions().feastol = absolute_tolerance;
  if (max_rounds > 0) { presolver.getPresolveOptions().maxrounds = max_rounds; }
  if (dual_postsolve) {
    presolver.getPresolveOptions().componentsmaxint = -1;
    presolver.getPresolveOptions().detectlindep     = 0;
  }
}

template <typename f_t>
void set_presolve_parameters(
  papilo::Presolve<f_t>& presolver,
  problem_category_t category,
  int nrows,
  int ncols,
  int max_badgesize,
  std::optional<std::unordered_set<std::string>> const& method_allowlist = std::nullopt)
{
  // It looks like a copy. But this copy has the pointers to relevant variables in papilo
  auto params = presolver.getParameters();
  if (category == problem_category_t::MIP) {
    auto reduction_allowed = [&](char const* name) {
      return !method_allowlist.has_value() || method_allowlist->count(name) > 0;
    };
    // Papilo has work unit measurements for probing. Because of this when the first batch fails to
    // produce any reductions, the algorithm stops. To avoid stopping the algorithm, we set a
    // minimum badge size to a huge value. The time limit makes sure that we exit if it takes too
    // long.
    // An uncapped ncols/2 forces one probing pass to span the whole problem, so probing never
    // reaches its work-based stop and runs unbounded on large MIPs whenever the clock is infinite.
    // Capping the badge keeps it large enough to still find reductions while Papilo's per-badge
    // working limit (~2*nnz) bounds a single pass. <=0 restores the uncapped behaviour.
    if (reduction_allowed("probing")) {
      int min_badgesize = std::max(ncols / 2, 32);
      if (max_badgesize > 0) { min_badgesize = std::min(min_badgesize, max_badgesize); }
      params.setParameter("probing.minbadgesize", min_badgesize);
    }
    if (reduction_allowed("cliquemerging")) {
      params.setParameter("cliquemerging.enabled", true);
      params.setParameter("cliquemerging.maxcalls", 50);
    }
  }
}

template <typename i_t, typename f_t>
third_party_presolve_status_t third_party_presolve_t<i_t, f_t>::apply_pslp(
  io::mps_data_model_t<i_t, f_t> const& mps, double time_limit)
{
  raft::common::nvtx::range fun_scope("Apply PSLP presolver on host");

  if constexpr (std::is_same_v<f_t, double>) {
    const i_t n_cols = mps.get_n_variables();
    const i_t n_rows = mps.get_n_constraints();
    const i_t nnz    = mps.get_nnz();

    // Local owned copies of the fields that need mutation (sign flip on
    // maximise / ±inf fill for empty bounds / row_types materialisation);
    // matrix arrays are read straight from mps below (const&).
    std::vector<f_t> obj_coeffs(mps.get_objective_coefficients());
    std::vector<f_t> var_lb(mps.get_variable_lower_bounds());
    std::vector<f_t> var_ub(mps.get_variable_upper_bounds());
    std::vector<f_t> constr_lb(mps.get_constraint_lower_bounds());
    std::vector<f_t> constr_ub(mps.get_constraint_upper_bounds());
    f_t objective_offset = mps.get_objective_offset();
    normalize_for_presolve<i_t, f_t>(
      mps, maximize_, obj_coeffs, objective_offset, var_lb, var_ub, constr_lb, constr_ub);
    if (var_lb.empty()) { var_lb.assign(n_cols, -std::numeric_limits<f_t>::infinity()); }
    if (var_ub.empty()) { var_ub.assign(n_cols, std::numeric_limits<f_t>::infinity()); }
    const auto& coefficients = mps.get_constraint_matrix_values();
    const auto& indices      = mps.get_constraint_matrix_indices();
    const auto& offsets      = mps.get_constraint_matrix_offsets();

    Settings* settings = default_settings();
    settings->verbose  = false;
    settings->max_time = time_limit;

    auto start_time      = std::chrono::high_resolution_clock::now();
    Presolver* presolver = new_presolver(coefficients.data(),
                                         indices.data(),
                                         offsets.data(),
                                         n_rows,
                                         n_cols,
                                         nnz,
                                         constr_lb.data(),
                                         constr_ub.data(),
                                         var_lb.data(),
                                         var_ub.data(),
                                         obj_coeffs.data(),
                                         settings);
    assert(presolver != nullptr && "Presolver initialization failed");
    const PresolveStatus pslp_status = run_presolver(presolver);
    auto end_time                    = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    CUOPT_LOG_DEBUG("PSLP presolver time: %d milliseconds", duration.count());
    CUOPT_LOG_INFO("PSLP Presolved problem: %d constraints, %d variables, %d non-zeros",
                   presolver->stats->n_rows_reduced,
                   presolver->stats->n_cols_reduced,
                   presolver->stats->nnz_reduced);

    // Free previously allocated presolver and settings (if any) and stash the
    // new ones so undo_pslp / build_reduced_mps_from_pslp can find them later.
    if (pslp_presolver_ != nullptr) { free_presolver(pslp_presolver_); }
    if (pslp_stgs_ != nullptr) { free_settings(pslp_stgs_); }
    pslp_presolver_ = presolver;
    pslp_stgs_      = settings;

    return convert_pslp_presolve_status_to_third_party_presolve_status(pslp_status);
  } else {
    cuopt_expects(
      false, error_type_t::ValidationError, "PSLP presolver only supports double precision");
    return third_party_presolve_status_t::UNCHANGED;  // unreachable
  }
}

template <typename i_t, typename f_t>
third_party_presolve_status_t third_party_presolve_t<i_t, f_t>::apply_papilo(
  papilo::Problem<f_t>& papilo_problem,
  problem_category_t category,
  bool dual_postsolve,
  f_t absolute_tolerance,
  f_t relative_tolerance,
  double time_limit,
  i_t num_cpu_threads,
  i_t max_rounds,
  i_t max_badgesize)
{
  raft::common::nvtx::range fun_scope("Apply Papilo presolve on host");

  // Capture original dimensions before papilo.apply() mutates papilo_problem
  // in place into its reduced form.
  const i_t original_n_vars = static_cast<i_t>(papilo_problem.getNCols());
  const i_t original_n_cons = static_cast<i_t>(papilo_problem.getNRows());
  const i_t original_nnz    = static_cast<i_t>(papilo_problem.getConstraintMatrix().getNnz());

  CUOPT_LOG_DEBUG("Original problem: %d constraints, %d variables, %d nonzeros",
                  original_n_cons,
                  original_n_vars,
                  original_nnz);
  CUOPT_LOG_INFO("\nRunning Papilo presolve (git hash %s)", PAPILO_GITHASH);
  if (category == problem_category_t::MIP) { dual_postsolve = false; }
  papilo::Presolve<f_t> papilo_presolver;
  set_presolve_methods(papilo_presolver, category, dual_postsolve, reduction_allowlist_);
  set_presolve_options<i_t, f_t>(papilo_presolver,
                                 category,
                                 absolute_tolerance,
                                 relative_tolerance,
                                 time_limit,
                                 dual_postsolve,
                                 num_cpu_threads,
                                 max_rounds);
  set_presolve_parameters(papilo_presolver,
                          category,
                          original_n_cons,
                          original_n_vars,
                          max_badgesize,
                          reduction_allowlist_);
  papilo_presolver.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);
  CUOPT_LOG_DEBUG(
    "PRESOLVE_PAPILO_BUDGET rounds=%d badge_cap=%d tlim=%g", max_rounds, max_badgesize, time_limit);

  const auto papilo_t0 = std::chrono::steady_clock::now();
  auto result          = papilo_presolver.apply(papilo_problem);
  const double papilo_wall =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - papilo_t0).count();
  // The effective badge is what set_presolve_parameters actually installed; the cap alone is
  // misleading because it only binds once ncols/2 exceeds it.
  int effective_badge = std::max(original_n_vars / 2, 32);
  if (max_badgesize > 0) { effective_badge = std::min(effective_badge, max_badgesize); }
  // hit_tlim distinguishes "presolve converged" from "presolve was cut off mid-round", which
  // changes how the reduced problem below should be read.
  CUOPT_LOG_DEBUG(
    "PRESOLVE_PAPILO wall=%.3f tlim=%g hit_tlim=%d rounds_cap=%d badge_cap=%d badge_effective=%d",
    papilo_wall,
    time_limit,
    (int)(papilo_wall >= 0.99 * time_limit),
    max_rounds,
    max_badgesize,
    effective_badge);
  check_presolve_status(result.status);
  auto status = convert_papilo_presolve_status_to_third_party_presolve_status(result.status);
  if (result.status == papilo::PresolveStatus::kInfeasible ||
      result.status == papilo::PresolveStatus::kUnbndOrInfeas ||
      result.status == papilo::PresolveStatus::kUnbounded) {
    return status;
  }
  papilo_post_solve_storage_.reset(new papilo::PostsolveStorage<f_t>(result.postsolve));
  CUOPT_LOG_INFO("Presolve removed: %d constraints, %d variables, %d nonzeros",
                 original_n_cons - papilo_problem.getNRows(),
                 original_n_vars - papilo_problem.getNCols(),
                 original_nnz - papilo_problem.getConstraintMatrix().getNnz());

  i_t n_integer = 0;
  {
    auto col_flags = papilo_problem.getColFlags();
    for (size_t i = 0; i < col_flags.size(); ++i) {
      if (col_flags[i].test(papilo::ColFlag::kIntegral)) n_integer++;
    }
  }
  CUOPT_LOG_INFO("Presolved problem: %d constraints, %d variables (%d integer), %d nonzeros",
                 papilo_problem.getNRows(),
                 papilo_problem.getNCols(),
                 n_integer,
                 papilo_problem.getConstraintMatrix().getNnz());

  // Check if presolve found the optimal solution (problem fully reduced)
  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) {
    status = third_party_presolve_status_t::OPTIMAL;
  }

  auto const& col_map = result.postsolve.origcol_mapping;
  reduced_to_original_map_.assign(col_map.begin(), col_map.end());
  original_to_reduced_map_.assign(original_n_vars, -1);
  for (size_t i = 0; i < reduced_to_original_map_.size(); ++i) {
    auto original_idx = reduced_to_original_map_[i];
    if (original_idx >= 0 && static_cast<size_t>(original_idx) < original_to_reduced_map_.size()) {
      original_to_reduced_map_[original_idx] = static_cast<i_t>(i);
    }
  }
  return status;
}

// Wrapper around apply_presolve_from_mps_data
// Generates an mps_data, presolve it and turn it back into a reduced op_problem
template <typename i_t, typename f_t>
third_party_presolve_device_result_t<i_t, f_t>
third_party_presolve_t<i_t, f_t>::apply_presolve_from_op_problem(
  optimization_problem_t<i_t, f_t> const& op_problem,
  problem_category_t category,
  cuopt::mathematical_optimization::presolver_t presolver,
  bool dual_postsolve,
  f_t absolute_tolerance,
  f_t relative_tolerance,
  double time_limit,
  i_t num_cpu_threads,
  i_t max_rounds,
  i_t max_badgesize)
{
  auto* handle = op_problem.get_handle_ptr();

  cuopt_expects(!op_problem.has_quadratic_objective(),
                error_type_t::ValidationError,
                "Presolve does not support optimization_problem with a quadratic objective");
  cuopt_expects(!op_problem.has_quadratic_constraints(),
                error_type_t::ValidationError,
                "Presolve does not support optimization_problem with quadratic constraints");

  auto mps = ::cuopt::mathematical_optimization::op_problem_to_mps_data_model<i_t, f_t>(op_problem);

  auto host_res = apply_presolve_from_mps_data(mps,
                                               category,
                                               presolver,
                                               dual_postsolve,
                                               absolute_tolerance,
                                               relative_tolerance,
                                               time_limit,
                                               num_cpu_threads,
                                               max_rounds,
                                               max_badgesize);

  // On terminal statuses the mps entry returns an empty reduced problem;
  // mirror that shape on the device side without going through H->D.
  if (host_res.status == third_party_presolve_status_t::INFEASIBLE ||
      host_res.status == third_party_presolve_status_t::UNBOUNDED ||
      host_res.status == third_party_presolve_status_t::UNBNDORINFEAS) {
    return third_party_presolve_device_result_t<i_t, f_t>{
      host_res.status,
      optimization_problem_t<i_t, f_t>(handle),
      std::move(host_res.implied_integer_indices),
      std::move(host_res.reduced_to_original_map),
      std::move(host_res.original_to_reduced_map)};
  }

  auto reduced_opt =
    ::cuopt::mathematical_optimization::mps_data_model_to_optimization_problem<i_t, f_t>(
      handle, host_res.reduced_problem);
  // mps_data_model doesn't carry problem_category; plumb it here so the
  // reduced op_problem matches the input's category
  reduced_opt.set_problem_category(category);

  return third_party_presolve_device_result_t<i_t, f_t>{
    host_res.status,
    std::move(reduced_opt),
    std::move(host_res.implied_integer_indices),
    std::move(host_res.reduced_to_original_map),
    std::move(host_res.original_to_reduced_map)};
}

template <typename i_t, typename f_t>
third_party_presolve_host_result_t<i_t, f_t>
third_party_presolve_t<i_t, f_t>::apply_presolve_from_mps_data(
  io::mps_data_model_t<i_t, f_t> const& mps,
  problem_category_t category,
  cuopt::mathematical_optimization::presolver_t presolver,
  bool dual_postsolve,
  f_t absolute_tolerance,
  f_t relative_tolerance,
  double time_limit,
  i_t num_cpu_threads,
  i_t max_rounds,
  i_t max_badgesize)
{
  presolver_ = presolver;
  maximize_  = mps.get_sense();

  cuopt_expects(!(category == problem_category_t::MIP &&
                  presolver == cuopt::mathematical_optimization::presolver_t::PSLP),
                error_type_t::RuntimeError,
                "PSLP presolver is not supported for MIP problems");

  // Neither PSLP nor Papilo handle quadratic objective / constraints.
  cuopt_expects(!mps.has_quadratic_objective(),
                error_type_t::ValidationError,
                "Presolve does not support mps_data_models with a quadratic objective");
  cuopt_expects(!mps.has_quadratic_constraints(),
                error_type_t::ValidationError,
                "Presolve does not support mps_data_models with quadratic constraints");

  // PSLP branch:  apply_pslp -> reduced mps.
  if (presolver == cuopt::mathematical_optimization::presolver_t::PSLP) {
    const f_t original_obj_offset = mps.get_objective_offset();
    auto status                   = apply_pslp(mps, time_limit);

    if (status == third_party_presolve_status_t::INFEASIBLE ||
        status == third_party_presolve_status_t::UNBNDORINFEAS) {
      return third_party_presolve_host_result_t<i_t, f_t>{
        status, io::mps_data_model_t<i_t, f_t>{}, {}, {}, {}};
    }

    auto reduced_mps =
      build_reduced_mps_from_pslp<i_t, f_t>(pslp_presolver_, maximize_, original_obj_offset);
    // mps_data_model_t deep-copies every PSLP array. Keep only PSLP's compact
    // postsolve state while the reduced problem is being solved.
    free_presolver_reduced_problem(pslp_presolver_);
    reduced_mps.set_problem_name(mps.get_problem_name());
    reduced_mps.set_objective_scaling_factor(mps.get_objective_scaling_factor());
    return third_party_presolve_host_result_t<i_t, f_t>{status, std::move(reduced_mps), {}, {}, {}};
  } else {
    // Papilo branch:  build papilo::Problem ->
    //                 apply_papilo -> reduced mps.
    // Stash the pre-presolve objective for post-solve diagnostics (e.g. the
    // debug crushed-vs-original objective check in diversity_manager).
    original_objective_coefficients_   = mps.get_objective_coefficients();
    original_objective_offset_         = mps.get_objective_offset();
    original_objective_scaling_factor_ = mps.get_objective_scaling_factor();

    auto papilo_problem = build_papilo_problem<i_t, f_t>(mps, maximize_, category);
    auto status         = apply_papilo(papilo_problem,
                               category,
                               dual_postsolve,
                               absolute_tolerance,
                               relative_tolerance,
                               time_limit,
                               num_cpu_threads,
                               max_rounds,
                               max_badgesize);

    if (status == third_party_presolve_status_t::INFEASIBLE ||
        status == third_party_presolve_status_t::UNBOUNDED ||
        status == third_party_presolve_status_t::UNBNDORINFEAS) {
      return third_party_presolve_host_result_t<i_t, f_t>{
        status, io::mps_data_model_t<i_t, f_t>{}, {}, {}, {}};
    }

    auto reduced_mps = build_reduced_mps_from_papilo<i_t, f_t>(papilo_problem, maximize_);
    reduced_mps.set_problem_name(mps.get_problem_name());
    reduced_mps.set_objective_scaling_factor(mps.get_objective_scaling_factor());

    auto col_flags = papilo_problem.getColFlags();
    std::vector<i_t> implied_integer_indices;
    for (size_t i = 0; i < col_flags.size(); ++i) {
      if (col_flags[i].test(papilo::ColFlag::kImplInt)) {
        implied_integer_indices.push_back(static_cast<i_t>(i));
      }
    }

    return third_party_presolve_host_result_t<i_t, f_t>{status,
                                                        std::move(reduced_mps),
                                                        std::move(implied_integer_indices),
                                                        reduced_to_original_map_,
                                                        original_to_reduced_map_};
  }
}

template <typename i_t, typename f_t>
third_party_presolve_status_t third_party_presolve_t<i_t, f_t>::apply_to_subproblem(
  simplex::user_problem_t<i_t, f_t>& problem,
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  f_t time_limit,
  i_t num_threads)
{
  const bool dual_postsolve = false;
  presolver_                = cuopt::mathematical_optimization::presolver_t::Papilo;
  // build_papilo_problem_mip keeps the objective in minimization sense (user_problem_t carries
  // the direction in obj_scale), so the read-back must not flip signs either.
  maximize_ = false;

  // Capture original dimensions before the problem is overwritten in place.
  const i_t orig_cols = problem.num_cols;
  const i_t orig_rows = problem.num_rows;
  const i_t orig_nnz  = problem.A.nnz();

  papilo::Problem<f_t> papilo_problem = build_papilo_problem(problem);

  settings.log.debug("Presolve input: %d constraints, %d variables, %d nonzeros",
                     papilo_problem.getNRows(),
                     papilo_problem.getNCols(),
                     papilo_problem.getConstraintMatrix().getNnz());

  papilo::Presolve<f_t> papilo_presolver;
  set_presolve_methods(
    papilo_presolver, problem_category_t::MIP, dual_postsolve, reduction_allowlist_);
  set_presolve_options<i_t, f_t>(papilo_presolver,
                                 problem_category_t::MIP,
                                 settings.primal_tol,
                                 settings.dual_tol,
                                 time_limit,
                                 dual_postsolve,
                                 num_threads,
                                 -1);
  // Node presolve already runs under a finite time limit, so it keeps the unbounded round count and
  // uncapped badge; the budgets apply to root presolve only.
  set_presolve_parameters(
    papilo_presolver, problem_category_t::MIP, orig_rows, orig_cols, -1, reduction_allowlist_);

  // Disable papilo logs
  papilo_presolver.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);

  auto result = papilo_presolver.apply(papilo_problem);
  auto status = convert_papilo_presolve_status_to_third_party_presolve_status(result.status);

  // Infeasible / unbounded: leave `problem` untouched; the caller branches on the status.
  if (result.status == papilo::PresolveStatus::kInfeasible ||
      result.status == papilo::PresolveStatus::kUnbndOrInfeas ||
      result.status == papilo::PresolveStatus::kUnbounded) {
    return status;
  }

  papilo_post_solve_storage_.reset(new papilo::PostsolveStorage<f_t>(result.postsolve));

  const i_t reduced_rows = papilo_problem.getNRows();
  const i_t reduced_cols = papilo_problem.getNCols();
  const i_t reduced_nnz  = papilo_problem.getConstraintMatrix().getNnz();
  settings.log.debug("Presolve removed: %d constraints, %d variables, %d nonzeros",
                     orig_rows - reduced_rows,
                     orig_cols - reduced_cols,
                     orig_nnz - reduced_nnz);

  // Presolve fully solved the problem.
  if (reduced_rows == 0 && reduced_cols == 0) { status = third_party_presolve_status_t::OPTIMAL; }

  // Rebuild `problem` in place from the reduced papilo problem.
  build_user_problem<i_t, f_t>(papilo_problem, problem);

  // Presolve changes the dimensions, so the original row/column names no longer line up with the
  // reduced problem. They are not needed for the sub-MIP solve, so clear them and let downstream
  // size checks skip them.
  problem.col_names.clear();
  problem.row_names.clear();

  // Column maps for postsolve (reduced -> original and its inverse).
  auto const& col_map = result.postsolve.origcol_mapping;
  reduced_to_original_map_.assign(col_map.begin(), col_map.end());
  original_to_reduced_map_.assign(orig_cols, -1);
  for (size_t i = 0; i < reduced_to_original_map_.size(); ++i) {
    auto original_idx = reduced_to_original_map_[i];
    if (original_idx >= 0 && original_idx < original_to_reduced_map_.size()) {
      original_to_reduced_map_[original_idx] = i;
    }
  }

  return status;
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo_from_device(rmm::device_uvector<f_t>& primal_solution,
                                                        rmm::device_uvector<f_t>& dual_solution,
                                                        rmm::device_uvector<f_t>& reduced_costs,
                                                        problem_category_t category,
                                                        bool status_to_skip,
                                                        bool dual_postsolve,
                                                        rmm::cuda_stream_view stream_view)
{
  std::vector<f_t> h_primal(primal_solution.size());
  std::vector<f_t> h_dual(dual_solution.size());
  std::vector<f_t> h_rc(reduced_costs.size());
  raft::copy(h_primal.data(), primal_solution.data(), primal_solution.size(), stream_view);
  raft::copy(h_dual.data(), dual_solution.data(), dual_solution.size(), stream_view);
  raft::copy(h_rc.data(), reduced_costs.data(), reduced_costs.size(), stream_view);
  stream_view.synchronize();

  undo(h_primal, h_dual, h_rc, category, status_to_skip, dual_postsolve);

  primal_solution.resize(h_primal.size(), stream_view);
  dual_solution.resize(h_dual.size(), stream_view);
  reduced_costs.resize(h_rc.size(), stream_view);
  raft::copy(primal_solution.data(), h_primal.data(), h_primal.size(), stream_view);
  raft::copy(dual_solution.data(), h_dual.data(), h_dual.size(), stream_view);
  raft::copy(reduced_costs.data(), h_rc.data(), h_rc.size(), stream_view);
  stream_view.synchronize();
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo_pslp(std::vector<f_t>& primal_solution,
                                                 std::vector<f_t>& dual_solution,
                                                 std::vector<f_t>& reduced_costs)
{
  if constexpr (std::is_same_v<f_t, double>) {
    // PSLP postsolve reads from the passed-in host buffers and writes the
    // uncrushed solution into pslp_presolver_->sol->{x, y, z}.
    postsolve(pslp_presolver_, primal_solution.data(), dual_solution.data(), reduced_costs.data());

    auto uncrushed_sol = pslp_presolver_->sol;
    const int n_cols   = uncrushed_sol->dim_x;
    const int n_rows   = uncrushed_sol->dim_y;

    primal_solution.assign(uncrushed_sol->x, uncrushed_sol->x + n_cols);
    dual_solution.assign(uncrushed_sol->y, uncrushed_sol->y + n_rows);
    reduced_costs.assign(uncrushed_sol->z, uncrushed_sol->z + n_cols);
  } else {
    cuopt_expects(
      false, error_type_t::ValidationError, "PSLP postsolve only supports double precision");
  }
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo_papilo(std::vector<f_t>& primal_solution,
                                                   std::vector<f_t>& dual_solution,
                                                   std::vector<f_t>& reduced_costs,
                                                   bool dual_postsolve)
{
  papilo::Solution<f_t> reduced_sol(primal_solution);
  if (dual_postsolve) {
    reduced_sol.dual         = dual_solution;
    reduced_sol.reducedCosts = reduced_costs;
    reduced_sol.type         = papilo::SolutionType::kPrimalDual;
  }
  papilo::Solution<f_t> full_sol;

  papilo::Message Msg{};
  Msg.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);
  papilo::Postsolve<f_t> post_solver{Msg, papilo_post_solve_storage_->getNum()};

  bool is_optimal = false;
  auto status = post_solver.undo(reduced_sol, full_sol, *papilo_post_solve_storage_, is_optimal);
  check_postsolve_status(status);

  primal_solution = std::move(full_sol.primal);
  dual_solution   = std::move(full_sol.dual);
  reduced_costs   = std::move(full_sol.reducedCosts);
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo(std::vector<f_t>& primal_solution,
                                            std::vector<f_t>& dual_solution,
                                            std::vector<f_t>& reduced_costs,
                                            problem_category_t /*category*/,
                                            bool status_to_skip,
                                            bool dual_postsolve)
{
  if (presolver_ == cuopt::mathematical_optimization::presolver_t::PSLP) {
    undo_pslp(primal_solution, dual_solution, reduced_costs);
    return;
  } else {  // Papilo branch
    if (status_to_skip) { return; }
    undo_papilo(primal_solution, dual_solution, reduced_costs, dual_postsolve);
  }
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::uncrush_primal_solution(
  const std::vector<f_t>& reduced_primal, std::vector<f_t>& full_primal) const
{
  if (presolver_ == cuopt::mathematical_optimization::presolver_t::PSLP) {
    cuopt_expects(false,
                  error_type_t::RuntimeError,
                  "This code path should be never called, as this is meant for callbacks and they "
                  "are not supported for LPs");
    return;
  }

  papilo::Solution<f_t> reduced_sol(reduced_primal);
  papilo::Solution<f_t> full_sol;
  papilo::Message Msg{};
  Msg.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);
  papilo::Postsolve<f_t> post_solver{Msg, papilo_post_solve_storage_->getNum()};

  bool is_optimal = false;
  auto status = post_solver.undo(reduced_sol, full_sol, *papilo_post_solve_storage_, is_optimal);
  check_postsolve_status(status);
  full_primal = std::move(full_sol.primal);
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::crush_primal_solution(
  const optimization_problem_t<i_t, f_t>& reduced_problem,
  const std::vector<f_t>& original_primal,
  std::vector<f_t>& reduced_primal) const
{
  cuopt_expects(presolver_ == cuopt::mathematical_optimization::presolver_t::Papilo,
                error_type_t::RuntimeError,
                "Primal crushing is only supported for PaPILO presolve");
  cuopt_assert(papilo_post_solve_storage_ != nullptr, "No postsolve storage available");
  std::vector<f_t> unused_y, unused_z;
  std::vector<f_t> empty_vals;
  std::vector<i_t> empty_indices, empty_offsets;
  crush_primal_dual_solution(original_primal,
                             {},
                             reduced_primal,
                             unused_y,
                             {},
                             unused_z,
                             empty_vals,
                             empty_indices,
                             empty_offsets);

  // Dual bound strengthening (e.g. DualFix kVarBoundChange which aren't emitted in primal mode)
  // can tighten a bound past a value
  // that was feasible in the original polytope. A simple clamp is often enough to crush a solution
  // inside the tightened polytope without breaking feasibility (dualfix seems to emit purely dual
  // based bounds reductions so primality is safe when clamping)
  cuopt_assert(reduced_problem.get_n_variables() == (i_t)reduced_to_original_map_.size(),
               "reduced_problem does not match this presolver's reduction");
  const std::vector<f_t> lb = reduced_problem.get_variable_lower_bounds_host();
  const std::vector<f_t> ub = reduced_problem.get_variable_upper_bounds_host();
  cuopt_assert(reduced_primal.size() == lb.size() && reduced_primal.size() == ub.size(),
               "reduced problem must match crush output dimension");
  for (size_t j = 0; j < reduced_primal.size(); ++j) {
    reduced_primal[j] = std::clamp(reduced_primal[j], lb[j], ub[j]);
  }
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::crush_primal_solution(
  const simplex::user_problem_t<i_t, f_t>& reduced_problem,
  const std::vector<f_t>& original_primal,
  std::vector<f_t>& reduced_primal) const
{
  cuopt_expects(presolver_ == cuopt::mathematical_optimization::presolver_t::Papilo,
                error_type_t::RuntimeError,
                "Primal crushing is only supported for PaPILO presolve");
  cuopt_assert(papilo_post_solve_storage_ != nullptr, "No postsolve storage available");
  std::vector<f_t> unused_y, unused_z;
  std::vector<f_t> empty_vals;
  std::vector<i_t> empty_indices, empty_offsets;
  crush_primal_dual_solution(original_primal,
                             {},
                             reduced_primal,
                             unused_y,
                             {},
                             unused_z,
                             empty_vals,
                             empty_indices,
                             empty_offsets);

  // Dual bound strengthening (e.g. DualFix kVarBoundChange which aren't emitted in primal mode)
  // can tighten a bound past a value
  // that was feasible in the original polytope. A simple clamp is often enough to crush a solution
  // inside the tightened polytope without breaking feasibility (dualfix seems to emit purely dual
  // based bounds reductions so primality is safe when clamping)
  cuopt_assert(reduced_problem.num_cols == (i_t)reduced_to_original_map_.size(),
               "reduced_problem does not match this presolver's reduction");
  cuopt_assert(reduced_primal.size() == reduced_problem.lower.size() &&
                 reduced_primal.size() == reduced_problem.upper.size(),
               "reduced problem must match crush output dimension");
  for (size_t j = 0; j < reduced_primal.size(); ++j) {
    reduced_primal[j] =
      std::clamp(reduced_primal[j], reduced_problem.lower[j], reduced_problem.upper[j]);
  }
}

/**
 * Crush an original-space primal+dual solution into the presolved (reduced) space.
 *
 * This is the forward counterpart of Papilo's Postsolve::undo(). It replays
 * each presolve reduction in forward order to transform variable/dual values,
 * then projects onto the surviving columns/rows via origcol/origrow_mapping.
 *
 * Only two reductions actually transform survivor coordinates:
 *   kParallelCol             — merges x[col1] into x[col2]; survivor rc is z[col2] if
 *                              nonzero, else z[col1] / scale (inverse of PaPILO postsolve)
 *   kRowBoundChangeForcedByRow — conditionally transfers y[deleted_row] → y[kept_row]
 */
template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::crush_primal_dual_solution(
  const std::vector<f_t>& x_original,
  const std::vector<f_t>& y_original,
  std::vector<f_t>& x_reduced,
  std::vector<f_t>& y_reduced,
  const std::vector<f_t>& z_original,
  std::vector<f_t>& z_reduced,
  const std::vector<f_t>& A_values,
  const std::vector<i_t>& A_indices,
  const std::vector<i_t>& A_offsets) const
{
  cuopt_expects(presolver_ == cuopt::mathematical_optimization::presolver_t::Papilo,
                error_type_t::RuntimeError,
                "Crushing is only supported for PaPILO presolve");
  cuopt_assert(papilo_post_solve_storage_ != nullptr, "No postsolve storage available");

  const auto& storage = *papilo_post_solve_storage_;
  const auto& types   = storage.types;
  const auto& indices = storage.indices;
  const auto& values  = storage.values;
  const auto& start   = storage.start;
  const auto& num     = storage.num;

  cuopt_assert((int)x_original.size() == (int)storage.nColsOriginal, "");

  const bool crush_dual = !y_original.empty();
  if (crush_dual) { cuopt_assert((int)y_original.size() == (int)storage.nRowsOriginal, ""); }

  const bool crush_rc = !z_original.empty() && crush_dual;
  if (crush_rc) { cuopt_assert((int)z_original.size() == (int)storage.nColsOriginal, ""); }

  std::vector<f_t> x(x_original.begin(), x_original.end());
  std::vector<f_t> y(y_original.begin(), y_original.end());
  std::vector<f_t> z(z_original.begin(), z_original.end());

  // Track current coefficient values for entries modified by kCoefficientChange,
  // so repeated changes to the same (row, col) are handled correctly.
  std::unordered_map<i_t, f_t> coeff_current;

  const i_t n_cols_original = (i_t)storage.nColsOriginal;

  auto coeff_key = [&](int row, int col) -> i_t { return (i_t)row * n_cols_original + (i_t)col; };

  auto get_coeff = [&](int row, int col) -> f_t {
    auto it = coeff_current.find(coeff_key(row, col));
    if (it != coeff_current.end()) return it->second;
    for (i_t p = A_offsets[row]; p < A_offsets[row + 1]; ++p) {
      if (A_indices[p] == col) return A_values[p];
    }
    return 0;
  };

  for (int i = 0; i < (int)types.size(); ++i) {
    int first = start[i];

    switch (types[i]) {
      case ReductionType::kParallelCol: {
        // Storage layout: [orig_col1, flags1, orig_col2, flags2, -1]
        //                 [col1lb,    col1ub, col2lb,    col2ub, col2scale]
        int col1         = indices[first];
        int col2         = indices[first + 2];
        const f_t& scale = values[first + 4];
        x[col2] += scale * x[col1];
        if (crush_rc) {
          // Inverse of Postsolve::apply_parallel_col_to_original_solution reduced-cost split.
          if (num.isZero(z[col2]) && !num.isZero(z[col1])) {
            cuopt_assert(!num.isZero(scale), "parallel column scale must be nonzero");
            z[col2] = z[col1] / scale;
          }
        }
        break;
      }

      case ReductionType::kRowBoundChangeForcedByRow: {
        if (!crush_dual) break;
        cuopt_assert(i >= 1 && types[i - 1] == ReductionType::kReasonForRowBoundChangeForcedByRow,
                     "kRowBoundChangeForcedByRow must be preceded by its reason record");

        bool is_lhs = indices[first] == 1;
        int row     = (int)values[first];

        int reason_first = start[i - 1];
        int deleted_row  = indices[reason_first + 1];
        f_t factor       = values[reason_first];
        cuopt_assert(factor != 0, "parallel row factor must be nonzero");

        // Forward rule: if the deleted row carried dual signal that the
        // reverse would have attributed to the kept row, transfer it back.
        f_t candidate = y[deleted_row] / factor;
        bool sign_ok  = is_lhs ? num.isGT(candidate, (f_t)0) : num.isLT(candidate, (f_t)0);

        if (sign_ok) {
          f_t y_old = y[row];
          y[row]    = candidate;
          // Maintain z = c - A^T y: propagate the y change into reduced costs
          if (crush_rc) {
            f_t delta_y = candidate - y_old;
            for (i_t p = A_offsets[row]; p < A_offsets[row + 1]; ++p) {
              f_t a = get_coeff(row, A_indices[p]);
              z[A_indices[p]] -= delta_y * a;
            }
          }
        }
        break;
      }

      case ReductionType::kCoefficientChange: {
        if (!crush_rc) break;
        int row                            = indices[first];
        int col                            = indices[first + 1];
        f_t a_new                          = values[first];
        f_t a_old                          = get_coeff(row, col);
        coeff_current[coeff_key(row, col)] = a_new;
        z[col] += (a_old - a_new) * y[row];
        break;
      }

      case ReductionType::kSubstitutedColWithDual: {
        // Singleton substitution: column j is expressed via equality row k as
        //   x_j = (rhs_k - Σ_{l≠j} a_kl·x_l) / a_kj
        // This changes the objective for every column l in row k:
        //   c_red[l] = c_orig[l] - (c_j / a_kj) · a_kl
        // Adjust z accordingly:  Δz[l] = -(a_kl / a_kj)·z[j] - a_kl·y[k]
        if (!crush_rc) break;
        int row_k      = indices[first];  // equality row (original space)
        int row_length = (int)values[first];
        // Row coefficients start at first+3
        int row_coef_start = first + 3;
        // Substituted column index is after the row coefficients
        int col_j = indices[row_coef_start + row_length];

        // Find a_kj (coefficient of col j in row k)
        f_t a_kj = 0;
        for (int p = 0; p < row_length; ++p) {
          if (indices[row_coef_start + p] == col_j) {
            a_kj = values[row_coef_start + p];
            break;
          }
        }
        if (a_kj == 0) break;  // shouldn't happen

        f_t z_j = z[col_j];
        f_t y_k = y[row_k];

        // Adjust z for each surviving column l in the equality row (l ≠ j)
        for (int p = 0; p < row_length; ++p) {
          int col_l = indices[row_coef_start + p];
          if (col_l == col_j) continue;
          f_t a_kl = values[row_coef_start + p];
          z[col_l] -= (a_kl / a_kj) * z_j + a_kl * y_k;
        }
        break;
      }

      case ReductionType::kFixedCol:                            // Handled via projection
      case ReductionType::kSubstitutedCol:                      // Col is dropped
      case ReductionType::kFixedInfCol:                         // Col is dropped
      case ReductionType::kVarBoundChange:                      // Noop
      case ReductionType::kRedundantRow:                        // Noop
      case ReductionType::kRowBoundChange:                      // Noop
      case ReductionType::kReasonForRowBoundChangeForcedByRow:  // Metadata for above
      case ReductionType::kSaveRow:                             // Metadata
      case ReductionType::kReducedBoundsCost:                   // Noop
      case ReductionType::kColumnDualValue:                     // Column reduced-cost only
      case ReductionType::kRowDualValue:                        // Handled via projection
        break;
        // no default: case to let the compiler yell at us if a new reduction is later introduced
    }
  }

  const auto& col_map = storage.origcol_mapping;
  const auto& row_map = storage.origrow_mapping;

  // Cancel contributions from removed rows.  The original-space z was
  // computed as z = c - A^T y over ALL rows.  The reduced-space stationarity
  // only involves surviving rows, so we must add back the terms from removed
  // rows: z[j] += y[i] * a_{i,j} for every removed row i with materially nonzero y[i].
  if (crush_rc) {
    std::vector<bool> row_survives((int)storage.nRowsOriginal, false);
    for (size_t k = 0; k < row_map.size(); ++k) {
      row_survives[row_map[k]] = true;
    }
    for (int i = 0; i < (int)storage.nRowsOriginal; ++i) {
      if (row_survives[i] || num.isZero(y[i])) continue;
      for (i_t p = A_offsets[i]; p < A_offsets[i + 1]; ++p) {
        z[A_indices[p]] += y[i] * get_coeff(i, A_indices[p]);
      }
    }
  }

  x_reduced.resize(col_map.size());
  for (size_t k = 0; k < col_map.size(); ++k) {
    x_reduced[k] = x[col_map[k]];
  }

  if (crush_dual) {
    y_reduced.resize(row_map.size());
    for (size_t k = 0; k < row_map.size(); ++k) {
      y_reduced[k] = y[row_map[k]];
    }
  }

  if (crush_rc) {
    z_reduced.resize(col_map.size());
    for (size_t k = 0; k < col_map.size(); ++k) {
      z_reduced[k] = z[col_map[k]];
    }
  }
}

template <typename i_t, typename f_t>
third_party_presolve_t<i_t, f_t>::~third_party_presolve_t()
{
  if (pslp_presolver_ != nullptr) { free_presolver(pslp_presolver_); }
  if (pslp_stgs_ != nullptr) { free_settings(pslp_stgs_); }
}

template <typename f_t>
void papilo_postsolve_deleter<f_t>::operator()(papilo::PostsolveStorage<f_t>* ptr) const
{
  delete ptr;
}

template <typename i_t, typename f_t>
presolve_features_t papilo_presolve_features(optimization_problem_t<i_t, f_t> const& op_problem)
{
  presolve_features_t f{};
  f.n_vars = op_problem.get_n_variables();
  f.n_cons = op_problem.get_n_constraints();
  f.nnz    = op_problem.get_nnz();

  const auto var_types = op_problem.get_variable_types_host();
  const auto lower     = op_problem.get_variable_lower_bounds_host();
  const auto upper     = op_problem.get_variable_upper_bounds_host();
  for (size_t j = 0; j < var_types.size(); ++j) {
    if (var_types[j] != var_t::INTEGER) { continue; }
    f.n_int += 1.0;
    if (lower[j] >= 0.0 && upper[j] <= 1.0) { f.n_bin += 1.0; }
  }

  const auto offsets = op_problem.get_constraint_matrix_offsets_host();
  for (size_t i = 0; i + 1 < offsets.size(); ++i) {
    f.max_row_len = std::max<double>(f.max_row_len, offsets[i + 1] - offsets[i]);
  }
  return f;
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template struct papilo_postsolve_deleter<float>;
template class third_party_presolve_t<int, float>;
template void papilo_round_trip(simplex::user_problem_t<int, float>&);
template presolve_features_t papilo_presolve_features(optimization_problem_t<int, float> const&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template struct papilo_postsolve_deleter<double>;
template class third_party_presolve_t<int, double>;
template void papilo_round_trip(simplex::user_problem_t<int, double>&);
template presolve_features_t papilo_presolve_features(optimization_problem_t<int, double> const&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
