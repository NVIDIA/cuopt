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
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/gf2_presolve.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>
#include <utilities/timer.hpp>

#include <raft/core/nvtx.hpp>

#include <unordered_map>

namespace cuopt::mathematical_optimization::mip {

// Host-only gather + input normalisation for PSLP from an mps_data_model.
// Used by the mps-facing entry point (apply_presolve_from_mps_data). The
// op_problem-facing entry goes through build_pslp_host_arrays_from_op_problem
// instead (direct D->H, no mps roundtrip).
template <typename i_t, typename f_t>
pslp_input_t<i_t, f_t> build_pslp_host_arrays_from_mps_data(
  const io::mps_data_model_t<i_t, f_t>& mps, bool maximize)
{
  raft::common::nvtx::range fun_scope("Build PSLP host arrays from mps_data_model");

  pslp_input_t<i_t, f_t> arrays;
  arrays.n_cols = mps.get_n_variables();
  arrays.n_rows = mps.get_n_constraints();
  arrays.nnz    = mps.get_nnz();

  // Copy (mps's getters return by const-ref; we own these vectors so we can
  // mutate them during normalisation).
  arrays.coefficients = mps.get_constraint_matrix_values();
  arrays.indices      = mps.get_constraint_matrix_indices();
  arrays.offsets      = mps.get_constraint_matrix_offsets();
  arrays.obj_coeffs   = mps.get_objective_coefficients();
  arrays.var_lb       = mps.get_variable_lower_bounds();
  arrays.var_ub       = mps.get_variable_upper_bounds();
  arrays.constr_lb    = mps.get_constraint_lower_bounds();
  arrays.constr_ub    = mps.get_constraint_upper_bounds();

  const auto& h_bounds    = mps.get_constraint_bounds();
  const auto& h_row_types = mps.get_row_types();

  if (maximize) {
    for (auto& c : arrays.obj_coeffs)
      c = -c;
  }

  if (arrays.constr_lb.empty() && arrays.constr_ub.empty()) {
    for (size_t i = 0; i < h_row_types.size(); ++i) {
      if (h_row_types[i] == 'L') {
        arrays.constr_lb.push_back(-std::numeric_limits<f_t>::infinity());
        arrays.constr_ub.push_back(h_bounds[i]);
      } else if (h_row_types[i] == 'G') {
        arrays.constr_lb.push_back(h_bounds[i]);
        arrays.constr_ub.push_back(std::numeric_limits<f_t>::infinity());
      } else if (h_row_types[i] == 'E') {
        arrays.constr_lb.push_back(h_bounds[i]);
        arrays.constr_ub.push_back(h_bounds[i]);
      }
    }
  }

  if (arrays.var_lb.empty()) {
    arrays.var_lb.assign(arrays.n_cols, -std::numeric_limits<f_t>::infinity());
  }
  if (arrays.var_ub.empty()) {
    arrays.var_ub.assign(arrays.n_cols, std::numeric_limits<f_t>::infinity());
  }

  return arrays;
}

// Host-only gather + papilo::Problem construction from an mps_data_model.
// Used by the mps-facing entry point (apply_presolve_from_mps_data). The
// op_problem-facing entry goes through build_papilo_problem_from_op_problem
// instead (direct D->H, no mps roundtrip).
template <typename i_t, typename f_t>
papilo::Problem<f_t> build_papilo_problem_from_mps_data(const io::mps_data_model_t<i_t, f_t>& mps,
                                                        problem_category_t category,
                                                        bool maximize)
{
  raft::common::nvtx::range fun_scope("Build papilo problem from mps_data_model");
  papilo::ProblemBuilder<f_t> builder;

  const i_t num_cols = mps.get_n_variables();
  const i_t num_rows = mps.get_n_constraints();
  const i_t nnz      = mps.get_nnz();

  builder.reserve(nnz, num_rows, num_cols);

  // Local mutable copies (we sign-flip / fill empties).
  auto h_coefficients = mps.get_constraint_matrix_values();
  auto h_offsets      = mps.get_constraint_matrix_offsets();
  auto h_variables    = mps.get_constraint_matrix_indices();
  auto h_obj_coeffs   = mps.get_objective_coefficients();
  auto h_var_lb       = mps.get_variable_lower_bounds();
  auto h_var_ub       = mps.get_variable_upper_bounds();
  auto h_constr_lb    = mps.get_constraint_lower_bounds();
  auto h_constr_ub    = mps.get_constraint_upper_bounds();

  const auto& h_bounds    = mps.get_constraint_bounds();
  const auto& h_row_types = mps.get_row_types();
  const auto& h_var_types = mps.get_variable_types();

  if (maximize) {
    for (auto& c : h_obj_coeffs)
      c = -c;
  }

  if (h_constr_lb.empty() && h_constr_ub.empty()) {
    for (size_t i = 0; i < h_row_types.size(); ++i) {
      if (h_row_types[i] == 'L') {
        h_constr_lb.push_back(-std::numeric_limits<f_t>::infinity());
        h_constr_ub.push_back(h_bounds[i]);
      } else if (h_row_types[i] == 'G') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(std::numeric_limits<f_t>::infinity());
      } else if (h_row_types[i] == 'E') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(h_bounds[i]);
      }
    }
  }

  builder.setNumCols(num_cols);
  builder.setNumRows(num_rows);

  builder.setObjAll(h_obj_coeffs);
  builder.setObjOffset(maximize ? -mps.get_objective_offset() : mps.get_objective_offset());

  if (!h_var_lb.empty() && !h_var_ub.empty()) {
    builder.setColLbAll(h_var_lb);
    builder.setColUbAll(h_var_ub);
    if (mps.get_variable_names().size() == static_cast<size_t>(num_cols)) {
      builder.setColNameAll(mps.get_variable_names());
    }
  }

  if (category == problem_category_t::MIP) {
    for (size_t i = 0; i < h_var_types.size(); ++i) {
      builder.setColIntegral(i, char_to_var_type(h_var_types[i]) == var_t::INTEGER);
    }
  }

  if (!h_constr_lb.empty() && !h_constr_ub.empty()) {
    builder.setRowLhsAll(h_constr_lb);
    builder.setRowRhsAll(h_constr_ub);
  }

  std::vector<papilo::RowFlags> h_row_flags(h_constr_lb.size());
  std::vector<std::tuple<i_t, i_t, f_t>> h_entries;
  for (size_t i = 0; i < h_constr_lb.size(); ++i) {
    i_t row_start   = h_offsets[i];
    i_t row_end     = h_offsets[i + 1];
    i_t num_entries = row_end - row_start;
    for (size_t j = 0; j < num_entries; ++j) {
      h_entries.push_back(
        std::make_tuple(i, h_variables[row_start + j], h_coefficients[row_start + j]));
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kLhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kLhsInf);
    }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kRhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kRhsInf);
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) { h_constr_lb[i] = 0; }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) { h_constr_ub[i] = 0; }
  }

  for (size_t i = 0; i < h_var_lb.size(); ++i) {
    builder.setColLbInf(i, h_var_lb[i] == -std::numeric_limits<f_t>::infinity());
    builder.setColUbInf(i, h_var_ub[i] == std::numeric_limits<f_t>::infinity());
    if (h_var_lb[i] == -std::numeric_limits<f_t>::infinity()) { builder.setColLb(i, 0); }
    if (h_var_ub[i] == std::numeric_limits<f_t>::infinity()) { builder.setColUb(i, 0); }
  }

  auto problem = builder.build();

  if (h_entries.size()) {
    auto constexpr const sorted_entries = true;
    const double spare_ratio            = category == problem_category_t::MIP ? 4.0 : 2.0;
    const int min_inter_row_space       = category == problem_category_t::MIP ? 30 : 4;
    auto csr_storage                    = papilo::SparseStorage<f_t>(
      h_entries, num_rows, num_cols, sorted_entries, spare_ratio, min_inter_row_space);
    problem.setConstraintMatrix(csr_storage, h_constr_lb, h_constr_ub, h_row_flags);

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

// Host-only result builder: write a (reduced) PSLP presolver state into a
// fresh mps_data_model_t.
template <typename i_t, typename f_t>
io::mps_data_model_t<i_t, f_t> build_mps_data_from_pslp(Presolver* pslp_presolver,
                                                        bool maximize,
                                                        f_t original_obj_offset)
{
  raft::common::nvtx::range fun_scope("Build mps_data_model from PSLP");
  io::mps_data_model_t<i_t, f_t> mps;

  if constexpr (std::is_same_v<f_t, double>) {
    cuopt_expects(pslp_presolver != nullptr && pslp_presolver->reduced_prob != nullptr,
                  error_type_t::RuntimeError,
                  "PSLP presolver is not initialized");
    auto reduced_prob = pslp_presolver->reduced_prob;
    const i_t n_rows  = reduced_prob->m;
    const i_t n_cols  = reduced_prob->n;
    const i_t nnz     = reduced_prob->nnz;
    f_t obj_offset    = reduced_prob->obj_offset;

    obj_offset = maximize ? -obj_offset : obj_offset;
    // PSLP does not allow setting an objective offset, so we fold the
    // original input's offset into the reduced one.
    obj_offset += original_obj_offset;
    mps.set_objective_offset(obj_offset);
    mps.set_maximize(maximize);

    if (n_cols == 0 && n_rows == 0) {
      std::vector<i_t> empty_offsets = {0};
      mps.set_csr_constraint_matrix(
        {}, {}, std::span<const i_t>(empty_offsets.data(), empty_offsets.size()));
      return mps;
    }

    mps.set_csr_constraint_matrix(
      std::span<const f_t>(reduced_prob->Ax, static_cast<size_t>(nnz)),
      std::span<const i_t>(reduced_prob->Ai, static_cast<size_t>(nnz)),
      std::span<const i_t>(reduced_prob->Ap, static_cast<size_t>(n_rows + 1)));

    std::vector<f_t> h_obj_coeffs(reduced_prob->c, reduced_prob->c + n_cols);
    if (maximize) {
      for (auto& c : h_obj_coeffs)
        c = -c;
    }
    mps.set_objective_coefficients(std::span<const f_t>(h_obj_coeffs.data(), h_obj_coeffs.size()));
    mps.set_constraint_lower_bounds(
      std::span<const f_t>(reduced_prob->lhs, static_cast<size_t>(n_rows)));
    mps.set_constraint_upper_bounds(
      std::span<const f_t>(reduced_prob->rhs, static_cast<size_t>(n_rows)));
    mps.set_variable_lower_bounds(
      std::span<const f_t>(reduced_prob->lbs, static_cast<size_t>(n_cols)));
    mps.set_variable_upper_bounds(
      std::span<const f_t>(reduced_prob->ubs, static_cast<size_t>(n_cols)));
  } else {
    cuopt_expects(false, error_type_t::ValidationError, "PSLP only supports double precision");
  }
  return mps;
}

// Host-only result builder: write the (reduced) papilo::Problem into a fresh
// mps_data_model_t.
template <typename i_t, typename f_t>
io::mps_data_model_t<i_t, f_t> build_mps_data_from_papilo(
  papilo::Problem<f_t> const& papilo_problem, bool maximize)
{
  raft::common::nvtx::range fun_scope("Build mps_data_model from papilo");
  io::mps_data_model_t<i_t, f_t> mps;

  auto obj = papilo_problem.getObjective();
  mps.set_objective_offset(maximize ? -obj.offset : obj.offset);
  mps.set_maximize(maximize);

  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) {
    std::vector<i_t> h_offsets{0};
    mps.set_csr_constraint_matrix({}, {}, std::span<const i_t>(h_offsets.data(), h_offsets.size()));
    return mps;
  }
  if (maximize) {
    for (size_t i = 0; i < obj.coefficients.size(); ++i) {
      obj.coefficients[i] = -obj.coefficients[i];
    }
  }
  mps.set_objective_coefficients(
    std::span<const f_t>(obj.coefficients.data(), obj.coefficients.size()));

  auto& constraint_matrix = papilo_problem.getConstraintMatrix();
  auto row_lower          = constraint_matrix.getLeftHandSides();
  auto row_upper          = constraint_matrix.getRightHandSides();
  auto col_lower          = papilo_problem.getLowerBounds();
  auto col_upper          = papilo_problem.getUpperBounds();

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

  auto [index_range, nrows] = constraint_matrix.getRangeInfo();
  std::vector<i_t> offsets(nrows + 1);
  size_t start = index_range[0].start;
  for (i_t i = 0; i < nrows; i++) {
    offsets[i] = index_range[i].start - start;
  }
  offsets[nrows] = index_range[nrows - 1].end - start;

  i_t nnz = constraint_matrix.getNnz();
  assert(offsets[nrows] == nnz);

  const int* cols   = constraint_matrix.getConstraintMatrix().getColumns();
  const f_t* coeffs = constraint_matrix.getConstraintMatrix().getValues();

  mps.set_csr_constraint_matrix(std::span<const f_t>(&coeffs[start], static_cast<size_t>(nnz)),
                                std::span<const i_t>(&cols[start], static_cast<size_t>(nnz)),
                                std::span<const i_t>(offsets.data(), offsets.size()));

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

// D->H direct: gather op_problem's device buffers into a pslp_input_t
template <typename i_t, typename f_t>
pslp_input_t<i_t, f_t> build_pslp_host_arrays_from_op_problem(
  const optimization_problem_t<i_t, f_t>& op, bool maximize)
{
  raft::common::nvtx::range fun_scope("Build PSLP host arrays from op_problem (D->H)");

  pslp_input_t<i_t, f_t> arrays;
  arrays.n_cols = op.get_n_variables();
  arrays.n_rows = op.get_n_constraints();
  arrays.nnz    = op.get_nnz();

  const auto& d_coefficients = op.get_constraint_matrix_values();
  const auto& d_indices      = op.get_constraint_matrix_indices();
  const auto& d_offsets      = op.get_constraint_matrix_offsets();
  const auto& d_obj_coeffs   = op.get_objective_coefficients();
  const auto& d_var_lb       = op.get_variable_lower_bounds();
  const auto& d_var_ub       = op.get_variable_upper_bounds();
  const auto& d_constr_lb    = op.get_constraint_lower_bounds();
  const auto& d_constr_ub    = op.get_constraint_upper_bounds();
  const auto& d_bounds       = op.get_constraint_bounds();
  const auto& d_row_types    = op.get_row_types();

  arrays.coefficients.resize(d_coefficients.size());
  arrays.indices.resize(d_indices.size());
  arrays.offsets.resize(d_offsets.size());
  arrays.obj_coeffs.resize(d_obj_coeffs.size());
  arrays.var_lb.resize(d_var_lb.size());
  arrays.var_ub.resize(d_var_ub.size());
  arrays.constr_lb.resize(d_constr_lb.size());
  arrays.constr_ub.resize(d_constr_ub.size());
  std::vector<f_t> h_bounds(d_bounds.size());
  std::vector<char> h_row_types(d_row_types.size());

  auto stream = op.get_handle_ptr()->get_stream();
  raft::copy(arrays.coefficients.data(), d_coefficients.data(), d_coefficients.size(), stream);
  raft::copy(arrays.indices.data(), d_indices.data(), d_indices.size(), stream);
  raft::copy(arrays.offsets.data(), d_offsets.data(), d_offsets.size(), stream);
  raft::copy(arrays.obj_coeffs.data(), d_obj_coeffs.data(), d_obj_coeffs.size(), stream);
  raft::copy(arrays.var_lb.data(), d_var_lb.data(), d_var_lb.size(), stream);
  raft::copy(arrays.var_ub.data(), d_var_ub.data(), d_var_ub.size(), stream);
  raft::copy(arrays.constr_lb.data(), d_constr_lb.data(), d_constr_lb.size(), stream);
  raft::copy(arrays.constr_ub.data(), d_constr_ub.data(), d_constr_ub.size(), stream);
  raft::copy(h_bounds.data(), d_bounds.data(), d_bounds.size(), stream);
  raft::copy(h_row_types.data(), d_row_types.data(), d_row_types.size(), stream);
  stream.synchronize();

  if (maximize) {
    for (auto& c : arrays.obj_coeffs)
      c = -c;
  }

  if (arrays.constr_lb.empty() && arrays.constr_ub.empty()) {
    for (size_t i = 0; i < h_row_types.size(); ++i) {
      if (h_row_types[i] == 'L') {
        arrays.constr_lb.push_back(-std::numeric_limits<f_t>::infinity());
        arrays.constr_ub.push_back(h_bounds[i]);
      } else if (h_row_types[i] == 'G') {
        arrays.constr_lb.push_back(h_bounds[i]);
        arrays.constr_ub.push_back(std::numeric_limits<f_t>::infinity());
      } else if (h_row_types[i] == 'E') {
        arrays.constr_lb.push_back(h_bounds[i]);
        arrays.constr_ub.push_back(h_bounds[i]);
      }
    }
  }

  if (arrays.var_lb.empty()) {
    arrays.var_lb.assign(arrays.n_cols, -std::numeric_limits<f_t>::infinity());
  }
  if (arrays.var_ub.empty()) {
    arrays.var_ub.assign(arrays.n_cols, std::numeric_limits<f_t>::infinity());
  }

  return arrays;
}

// D->H direct: gather op_problem's device buffers, then feed a
// papilo::ProblemBuilder<f_t>. Mirrors build_papilo_problem_from_mps_data
// exactly (same normalisation and flag handling), but consumes op_problem's
// device-side arrays with a single stream sync — no mps_data_model roundtrip.
// var_types come out of op_problem as `var_t` enums so we skip the char <->
// enum conversion the mps path needs.
template <typename i_t, typename f_t>
papilo::Problem<f_t> build_papilo_problem_from_op_problem(
  const optimization_problem_t<i_t, f_t>& op, problem_category_t category, bool maximize)
{
  raft::common::nvtx::range fun_scope("Build papilo problem from op_problem (D->H)");
  papilo::ProblemBuilder<f_t> builder;

  const i_t num_cols = op.get_n_variables();
  const i_t num_rows = op.get_n_constraints();
  const i_t nnz      = op.get_nnz();

  builder.reserve(nnz, num_rows, num_cols);

  const auto& d_coefficients = op.get_constraint_matrix_values();
  const auto& d_indices      = op.get_constraint_matrix_indices();
  const auto& d_offsets      = op.get_constraint_matrix_offsets();
  const auto& d_obj_coeffs   = op.get_objective_coefficients();
  const auto& d_var_lb       = op.get_variable_lower_bounds();
  const auto& d_var_ub       = op.get_variable_upper_bounds();
  const auto& d_constr_lb    = op.get_constraint_lower_bounds();
  const auto& d_constr_ub    = op.get_constraint_upper_bounds();
  const auto& d_bounds       = op.get_constraint_bounds();
  const auto& d_row_types    = op.get_row_types();
  const auto& d_var_types    = op.get_variable_types();

  std::vector<f_t> h_coefficients(d_coefficients.size());
  std::vector<i_t> h_variables(d_indices.size());
  std::vector<i_t> h_offsets(d_offsets.size());
  std::vector<f_t> h_obj_coeffs(d_obj_coeffs.size());
  std::vector<f_t> h_var_lb(d_var_lb.size());
  std::vector<f_t> h_var_ub(d_var_ub.size());
  std::vector<f_t> h_constr_lb(d_constr_lb.size());
  std::vector<f_t> h_constr_ub(d_constr_ub.size());
  std::vector<f_t> h_bounds(d_bounds.size());
  std::vector<char> h_row_types(d_row_types.size());
  std::vector<var_t> h_var_types(d_var_types.size());

  auto stream = op.get_handle_ptr()->get_stream();
  raft::copy(h_coefficients.data(), d_coefficients.data(), d_coefficients.size(), stream);
  raft::copy(h_variables.data(), d_indices.data(), d_indices.size(), stream);
  raft::copy(h_offsets.data(), d_offsets.data(), d_offsets.size(), stream);
  raft::copy(h_obj_coeffs.data(), d_obj_coeffs.data(), d_obj_coeffs.size(), stream);
  raft::copy(h_var_lb.data(), d_var_lb.data(), d_var_lb.size(), stream);
  raft::copy(h_var_ub.data(), d_var_ub.data(), d_var_ub.size(), stream);
  raft::copy(h_constr_lb.data(), d_constr_lb.data(), d_constr_lb.size(), stream);
  raft::copy(h_constr_ub.data(), d_constr_ub.data(), d_constr_ub.size(), stream);
  raft::copy(h_bounds.data(), d_bounds.data(), d_bounds.size(), stream);
  raft::copy(h_row_types.data(), d_row_types.data(), d_row_types.size(), stream);
  raft::copy(h_var_types.data(), d_var_types.data(), d_var_types.size(), stream);
  stream.synchronize();

  if (maximize) {
    for (auto& c : h_obj_coeffs)
      c = -c;
  }

  if (h_constr_lb.empty() && h_constr_ub.empty()) {
    for (size_t i = 0; i < h_row_types.size(); ++i) {
      if (h_row_types[i] == 'L') {
        h_constr_lb.push_back(-std::numeric_limits<f_t>::infinity());
        h_constr_ub.push_back(h_bounds[i]);
      } else if (h_row_types[i] == 'G') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(std::numeric_limits<f_t>::infinity());
      } else if (h_row_types[i] == 'E') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(h_bounds[i]);
      }
    }
  }

  builder.setNumCols(num_cols);
  builder.setNumRows(num_rows);

  builder.setObjAll(h_obj_coeffs);
  builder.setObjOffset(maximize ? -op.get_objective_offset() : op.get_objective_offset());

  if (!h_var_lb.empty() && !h_var_ub.empty()) {
    builder.setColLbAll(h_var_lb);
    builder.setColUbAll(h_var_ub);
    if (op.get_variable_names().size() == static_cast<size_t>(num_cols)) {
      builder.setColNameAll(op.get_variable_names());
    }
  }

  if (category == problem_category_t::MIP) {
    for (size_t i = 0; i < h_var_types.size(); ++i) {
      builder.setColIntegral(i, h_var_types[i] == var_t::INTEGER);
    }
  }

  if (!h_constr_lb.empty() && !h_constr_ub.empty()) {
    builder.setRowLhsAll(h_constr_lb);
    builder.setRowRhsAll(h_constr_ub);
  }

  std::vector<papilo::RowFlags> h_row_flags(h_constr_lb.size());
  std::vector<std::tuple<i_t, i_t, f_t>> h_entries;
  for (size_t i = 0; i < h_constr_lb.size(); ++i) {
    i_t row_start   = h_offsets[i];
    i_t row_end     = h_offsets[i + 1];
    i_t num_entries = row_end - row_start;
    for (size_t j = 0; j < num_entries; ++j) {
      h_entries.push_back(
        std::make_tuple(i, h_variables[row_start + j], h_coefficients[row_start + j]));
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kLhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kLhsInf);
    }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kRhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kRhsInf);
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) { h_constr_lb[i] = 0; }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) { h_constr_ub[i] = 0; }
  }

  for (size_t i = 0; i < h_var_lb.size(); ++i) {
    builder.setColLbInf(i, h_var_lb[i] == -std::numeric_limits<f_t>::infinity());
    builder.setColUbInf(i, h_var_ub[i] == std::numeric_limits<f_t>::infinity());
    if (h_var_lb[i] == -std::numeric_limits<f_t>::infinity()) { builder.setColLb(i, 0); }
    if (h_var_ub[i] == std::numeric_limits<f_t>::infinity()) { builder.setColUb(i, 0); }
  }

  auto problem = builder.build();

  if (h_entries.size()) {
    auto constexpr const sorted_entries = true;
    const double spare_ratio            = category == problem_category_t::MIP ? 4.0 : 2.0;
    const int min_inter_row_space       = category == problem_category_t::MIP ? 30 : 4;
    auto csr_storage                    = papilo::SparseStorage<f_t>(
      h_entries, num_rows, num_cols, sorted_entries, spare_ratio, min_inter_row_space);
    problem.setConstraintMatrix(csr_storage, h_constr_lb, h_constr_ub, h_row_flags);

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

// H->D direct: read PSLP's reduced host arrays and build a device
// optimization_problem_t on `handle`. Mirrors build_mps_data_from_pslp but
// skips the mps_data_model intermediate — arrays flow straight from PSLP into
// op_problem setters (which each perform an async raft::copy on handle's
// stream). We sync once at the end so the local sign-flipped obj_coeffs
// temp is safe to release.
template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> build_op_problem_from_pslp(raft::handle_t const* handle,
                                                            Presolver* pslp_presolver,
                                                            bool maximize,
                                                            f_t original_obj_offset)
{
  raft::common::nvtx::range fun_scope("Build op_problem from PSLP (H->D)");
  optimization_problem_t<i_t, f_t> op(handle);

  if constexpr (std::is_same_v<f_t, double>) {
    cuopt_expects(pslp_presolver != nullptr && pslp_presolver->reduced_prob != nullptr,
                  error_type_t::RuntimeError,
                  "PSLP presolver is not initialized");
    auto reduced_prob = pslp_presolver->reduced_prob;
    const i_t n_rows  = reduced_prob->m;
    const i_t n_cols  = reduced_prob->n;
    const i_t nnz     = reduced_prob->nnz;
    f_t obj_offset    = reduced_prob->obj_offset;

    obj_offset = maximize ? -obj_offset : obj_offset;
    // PSLP does not allow setting an objective offset, so we fold the
    // original input's offset into the reduced one.
    obj_offset += original_obj_offset;
    op.set_maximize(maximize);
    op.set_objective_offset(obj_offset);

    if (n_cols == 0 && n_rows == 0) {
      std::vector<i_t> empty_offsets{0};
      op.set_csr_constraint_matrix(nullptr, 0, nullptr, 0, empty_offsets.data(), 1);
      handle->sync_stream();
      return op;
    }

    op.set_csr_constraint_matrix(
      reduced_prob->Ax, nnz, reduced_prob->Ai, nnz, reduced_prob->Ap, n_rows + 1);

    std::vector<f_t> h_obj_coeffs(reduced_prob->c, reduced_prob->c + n_cols);
    if (maximize) {
      for (auto& c : h_obj_coeffs)
        c = -c;
    }
    op.set_objective_coefficients(h_obj_coeffs.data(), n_cols);

    op.set_constraint_lower_bounds(reduced_prob->lhs, n_rows);
    op.set_constraint_upper_bounds(reduced_prob->rhs, n_rows);
    op.set_variable_lower_bounds(reduced_prob->lbs, n_cols);
    op.set_variable_upper_bounds(reduced_prob->ubs, n_cols);

    // set_* enqueued async raft::copy on the handle's stream; sync so the
    // sign-flipped h_obj_coeffs temp lives long enough.
    handle->sync_stream();
  } else {
    cuopt_expects(false, error_type_t::ValidationError, "PSLP only supports double precision");
  }
  return op;
}

// H->D direct: read the reduced papilo::Problem and build a device
// optimization_problem_t on `handle`. Mirrors build_mps_data_from_papilo but
// skips the mps_data_model intermediate. var_types are produced as `var_t`
// enums (no char <-> enum round-trip). Single stream sync at the end so
// temporaries (var_types, bound arrays with inf substitution, sign-flipped
// obj_coeffs) live past the async H->D copies.
template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> build_op_problem_from_papilo(
  raft::handle_t const* handle, papilo::Problem<f_t> const& papilo_problem, bool maximize)
{
  raft::common::nvtx::range fun_scope("Build op_problem from papilo (H->D)");
  optimization_problem_t<i_t, f_t> op(handle);

  auto obj = papilo_problem.getObjective();
  op.set_maximize(maximize);
  op.set_objective_offset(maximize ? -obj.offset : obj.offset);

  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) {
    std::vector<i_t> h_offsets{0};
    op.set_csr_constraint_matrix(nullptr, 0, nullptr, 0, h_offsets.data(), 1);
    handle->sync_stream();
    return op;
  }

  if (maximize) {
    for (size_t i = 0; i < obj.coefficients.size(); ++i) {
      obj.coefficients[i] = -obj.coefficients[i];
    }
  }
  op.set_objective_coefficients(obj.coefficients.data(), static_cast<i_t>(obj.coefficients.size()));

  auto& constraint_matrix = papilo_problem.getConstraintMatrix();
  auto row_lower          = constraint_matrix.getLeftHandSides();
  auto row_upper          = constraint_matrix.getRightHandSides();
  auto col_lower          = papilo_problem.getLowerBounds();
  auto col_upper          = papilo_problem.getUpperBounds();

  auto row_flags = constraint_matrix.getRowFlags();
  for (size_t i = 0; i < row_flags.size(); i++) {
    if (row_flags[i].test(papilo::RowFlag::kLhsInf)) {
      row_lower[i] = -std::numeric_limits<f_t>::infinity();
    }
    if (row_flags[i].test(papilo::RowFlag::kRhsInf)) {
      row_upper[i] = std::numeric_limits<f_t>::infinity();
    }
  }

  op.set_constraint_lower_bounds(row_lower.data(), static_cast<i_t>(row_lower.size()));
  op.set_constraint_upper_bounds(row_upper.data(), static_cast<i_t>(row_upper.size()));

  auto [index_range, nrows] = constraint_matrix.getRangeInfo();
  std::vector<i_t> offsets(nrows + 1);
  size_t start = index_range[0].start;
  for (i_t i = 0; i < nrows; i++) {
    offsets[i] = index_range[i].start - start;
  }
  offsets[nrows] = index_range[nrows - 1].end - start;

  i_t nnz = constraint_matrix.getNnz();
  assert(offsets[nrows] == nnz);

  const int* cols   = constraint_matrix.getConstraintMatrix().getColumns();
  const f_t* coeffs = constraint_matrix.getConstraintMatrix().getValues();

  op.set_csr_constraint_matrix(
    &coeffs[start], nnz, &cols[start], nnz, offsets.data(), static_cast<i_t>(offsets.size()));

  auto col_flags = papilo_problem.getColFlags();
  std::vector<var_t> var_types(col_flags.size());
  for (size_t i = 0; i < col_flags.size(); i++) {
    var_types[i] =
      col_flags[i].test(papilo::ColFlag::kIntegral) ? var_t::INTEGER : var_t::CONTINUOUS;
    if (col_flags[i].test(papilo::ColFlag::kLbInf)) {
      col_lower[i] = -std::numeric_limits<f_t>::infinity();
    }
    if (col_flags[i].test(papilo::ColFlag::kUbInf)) {
      col_upper[i] = std::numeric_limits<f_t>::infinity();
    }
  }

  op.set_variable_lower_bounds(col_lower.data(), static_cast<i_t>(col_lower.size()));
  op.set_variable_upper_bounds(col_upper.data(), static_cast<i_t>(col_upper.size()));
  op.set_variable_types(var_types.data(), static_cast<i_t>(var_types.size()));

  // Every set_* above enqueued async raft::copy on handle's stream. Sync so
  // the temporary vectors (var_types, adjusted lb/ub, offsets, sign-flipped
  // obj coefficients) outlive the copies.
  handle->sync_stream();

  return op;
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
void set_presolve_methods(papilo::Presolve<f_t>& presolver,
                          problem_category_t category,
                          bool dual_postsolve)
{
  using uptr = std::unique_ptr<papilo::PresolveMethod<f_t>>;

  if (category == problem_category_t::MIP) {
    // cuOpt custom GF2 presolver
    presolver.addPresolveMethod(
      uptr(new cuopt::mathematical_optimization::mip::GF2Presolve<f_t>()));
  }
  // fast presolvers
  presolver.addPresolveMethod(uptr(new papilo::SingletonCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::CoefficientStrengthening<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ConstraintPropagation<f_t>()));

  // medium presolvers
  presolver.addPresolveMethod(uptr(new papilo::FixContinuous<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimpleProbing<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelRowDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelColDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DualFix<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimplifyInequalities<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::CliqueMerging<f_t>()));

  // exhaustive presolvers
  presolver.addPresolveMethod(uptr(new papilo::ImplIntDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DominatedCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::Probing<f_t>()));

  if (!dual_postsolve) {
    // SingletonStuffing causes dual crushing failures on:
    //   tr12-30, ns1208400, gmu-35-50, dws008-01, neos-1445765,
    //   neos-5107597-kakapo, rocI-4-11, traininstance2, traininstance6,
    //   radiationm18-12-05, rococoB10-011000, b1c1s1
    presolver.addPresolveMethod(uptr(new papilo::SingletonStuffing<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::DualInfer<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::SimpleSubstitution<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::Sparsify<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::Substitution<f_t>()));
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
                          i_t num_cpu_threads)
{
  presolver.getPresolveOptions().tlim    = time_limit;
  presolver.getPresolveOptions().threads = num_cpu_threads;  //  user setting or  0 (automatic)
  presolver.getPresolveOptions().feastol = 1e-5;
  if (dual_postsolve) {
    presolver.getPresolveOptions().componentsmaxint = -1;
    presolver.getPresolveOptions().detectlindep     = 0;
  }
}

template <typename f_t>
void set_presolve_parameters(papilo::Presolve<f_t>& presolver,
                             problem_category_t category,
                             int nrows,
                             int ncols)
{
  // It looks like a copy. But this copy has the pointers to relevant variables in papilo
  auto params = presolver.getParameters();
  if (category == problem_category_t::MIP) {
    // Papilo has work unit measurements for probing. Because of this when the first batch fails to
    // produce any reductions, the algorithm stops. To avoid stopping the algorithm, we set a
    // minimum badge size to a huge value. The time limit makes sure that we exit if it takes too
    // long
    int min_badgesize = std::max(ncols / 2, 32);
    params.setParameter("probing.minbadgesize", min_badgesize);
    params.setParameter("cliquemerging.enabled", true);
    params.setParameter("cliquemerging.maxcalls", 50);
  }
}

template <typename i_t, typename f_t>
third_party_presolve_status_t third_party_presolve_t<i_t, f_t>::apply_pslp(
  pslp_input_t<i_t, f_t>& arrays, double time_limit)
{
  if constexpr (std::is_same_v<f_t, double>) {
    raft::common::nvtx::range fun_scope("Apply PSLP presolver on host");

    Settings* settings = default_settings();
    settings->verbose  = false;
    settings->max_time = time_limit;

    auto start_time      = std::chrono::high_resolution_clock::now();
    Presolver* presolver = new_presolver(arrays.coefficients.data(),
                                         arrays.indices.data(),
                                         arrays.offsets.data(),
                                         arrays.n_rows,
                                         arrays.n_cols,
                                         arrays.nnz,
                                         arrays.constr_lb.data(),
                                         arrays.constr_ub.data(),
                                         arrays.var_lb.data(),
                                         arrays.var_ub.data(),
                                         arrays.obj_coeffs.data(),
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
    // new ones so undo_pslp_host / build_{mps_data,op_problem}_from_pslp can
    // find them later.
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
  i_t num_cpu_threads)
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
  set_presolve_methods(papilo_presolver, category, dual_postsolve);
  set_presolve_options<i_t, f_t>(papilo_presolver,
                                 category,
                                 absolute_tolerance,
                                 relative_tolerance,
                                 time_limit,
                                 dual_postsolve,
                                 num_cpu_threads);
  set_presolve_parameters(papilo_presolver, category, original_n_cons, original_n_vars);
  papilo_presolver.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);

  auto result = papilo_presolver.apply(papilo_problem);
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

// Direct op_problem -> presolver_input -> apply -> op_problem path
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
  i_t num_cpu_threads)
{
  auto* handle = op_problem.get_handle_ptr();
  presolver_   = presolver;
  maximize_    = op_problem.get_sense();

  cuopt_expects(!(category == problem_category_t::MIP &&
                  presolver == cuopt::mathematical_optimization::presolver_t::PSLP),
                error_type_t::RuntimeError,
                "PSLP presolver is not supported for MIP problems");

  // Neither PSLP nor Papilo handle quadratic objective / constraints.
  cuopt_expects(!op_problem.has_quadratic_objective(),
                error_type_t::ValidationError,
                "Presolve does not support optimization_problem with a quadratic objective");
  cuopt_expects(!op_problem.has_quadratic_constraints(),
                error_type_t::ValidationError,
                "Presolve does not support optimization_problem with quadratic constraints");

  // PSLP branch:  D->H gather  ->  apply_pslp (host)  ->  H->D build
  if (presolver == cuopt::mathematical_optimization::presolver_t::PSLP) {
    if constexpr (std::is_same_v<f_t, double>) {
      const f_t original_obj_offset = op_problem.get_objective_offset();
      auto arrays                   = build_pslp_host_arrays_from_op_problem(op_problem, maximize_);
      auto status                   = apply_pslp(arrays, time_limit);

      if (status == third_party_presolve_status_t::INFEASIBLE ||
          status == third_party_presolve_status_t::UNBNDORINFEAS) {
        return third_party_presolve_device_result_t<i_t, f_t>{
          status, optimization_problem_t<i_t, f_t>(handle), {}, {}, {}};
      }

      auto reduced_opt = build_op_problem_from_pslp<i_t, f_t>(
        handle, pslp_presolver_, maximize_, original_obj_offset);
      reduced_opt.set_problem_name(op_problem.get_problem_name());
      reduced_opt.set_objective_scaling_factor(op_problem.get_objective_scaling_factor());
      reduced_opt.set_problem_category(category);
      return third_party_presolve_device_result_t<i_t, f_t>{
        status, std::move(reduced_opt), {}, {}, {}};
    } else {
      cuopt_expects(
        false, error_type_t::ValidationError, "PSLP presolver only supports double precision");
      return third_party_presolve_device_result_t<i_t, f_t>{
        third_party_presolve_status_t::UNCHANGED,
        optimization_problem_t<i_t, f_t>(handle),
        {},
        {},
        {}};  // unreachable
    }
  } else {
    // Papilo branch:  D->H gather  ->  apply_papilo (host)  ->  H->D build
    auto papilo_problem = build_papilo_problem_from_op_problem(op_problem, category, maximize_);
    auto status         = apply_papilo(papilo_problem,
                               category,
                               dual_postsolve,
                               absolute_tolerance,
                               relative_tolerance,
                               time_limit,
                               num_cpu_threads);

    if (status == third_party_presolve_status_t::INFEASIBLE ||
        status == third_party_presolve_status_t::UNBOUNDED ||
        status == third_party_presolve_status_t::UNBNDORINFEAS) {
      return third_party_presolve_device_result_t<i_t, f_t>{
        status, optimization_problem_t<i_t, f_t>(handle), {}, {}, {}};
    }

    auto reduced_opt = build_op_problem_from_papilo<i_t, f_t>(handle, papilo_problem, maximize_);
    reduced_opt.set_problem_name(op_problem.get_problem_name());
    reduced_opt.set_objective_scaling_factor(op_problem.get_objective_scaling_factor());
    reduced_opt.set_problem_category(category);

    std::vector<i_t> implied_integer_indices;
    {
      auto col_flags = papilo_problem.getColFlags();
      for (size_t i = 0; i < col_flags.size(); ++i) {
        if (col_flags[i].test(papilo::ColFlag::kImplInt)) {
          implied_integer_indices.push_back(static_cast<i_t>(i));
        }
      }
    }

    return third_party_presolve_device_result_t<i_t, f_t>{status,
                                                          std::move(reduced_opt),
                                                          std::move(implied_integer_indices),
                                                          reduced_to_original_map_,
                                                          original_to_reduced_map_};
  }
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
  i_t num_cpu_threads)
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

  // PSLP branch:  host gather  ->  apply_pslp (host)  ->  host build
  if (presolver == cuopt::mathematical_optimization::presolver_t::PSLP) {
    if constexpr (std::is_same_v<f_t, double>) {
      const f_t original_obj_offset = mps.get_objective_offset();
      auto arrays                   = build_pslp_host_arrays_from_mps_data(mps, maximize_);
      auto status                   = apply_pslp(arrays, time_limit);

      if (status == third_party_presolve_status_t::INFEASIBLE ||
          status == third_party_presolve_status_t::UNBNDORINFEAS) {
        return third_party_presolve_host_result_t<i_t, f_t>{
          status, io::mps_data_model_t<i_t, f_t>{}, {}, {}, {}};
      }

      auto reduced_mps =
        build_mps_data_from_pslp<i_t, f_t>(pslp_presolver_, maximize_, original_obj_offset);
      reduced_mps.set_problem_name(mps.get_problem_name());
      reduced_mps.set_objective_scaling_factor(mps.get_objective_scaling_factor());
      return third_party_presolve_host_result_t<i_t, f_t>{
        status, std::move(reduced_mps), {}, {}, {}};
    } else {
      cuopt_expects(
        false, error_type_t::ValidationError, "PSLP presolver only supports double precision");
      return third_party_presolve_host_result_t<i_t, f_t>{third_party_presolve_status_t::UNCHANGED,
                                                          io::mps_data_model_t<i_t, f_t>{},
                                                          {},
                                                          {},
                                                          {}};  // unreachable
    }
  } else {
    // Papilo branch:  host gather  ->  apply_papilo (host)  ->  host build
    auto papilo_problem = build_papilo_problem_from_mps_data(mps, category, maximize_);
    auto status         = apply_papilo(papilo_problem,
                               category,
                               dual_postsolve,
                               absolute_tolerance,
                               relative_tolerance,
                               time_limit,
                               num_cpu_threads);

    if (status == third_party_presolve_status_t::INFEASIBLE ||
        status == third_party_presolve_status_t::UNBOUNDED ||
        status == third_party_presolve_status_t::UNBNDORINFEAS) {
      return third_party_presolve_host_result_t<i_t, f_t>{
        status, io::mps_data_model_t<i_t, f_t>{}, {}, {}, {}};
    }

    auto reduced_mps = build_mps_data_from_papilo<i_t, f_t>(papilo_problem, maximize_);
    reduced_mps.set_problem_name(mps.get_problem_name());
    reduced_mps.set_objective_scaling_factor(mps.get_objective_scaling_factor());

    std::vector<i_t> implied_integer_indices;
    {
      auto col_flags = papilo_problem.getColFlags();
      for (size_t i = 0; i < col_flags.size(); ++i) {
        if (col_flags[i].test(papilo::ColFlag::kImplInt)) {
          implied_integer_indices.push_back(static_cast<i_t>(i));
        }
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
void third_party_presolve_t<i_t, f_t>::undo_pslp_host(std::vector<f_t>& primal_solution,
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
void third_party_presolve_t<i_t, f_t>::undo_papilo_host(std::vector<f_t>& primal_solution,
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
void third_party_presolve_t<i_t, f_t>::undo_host(std::vector<f_t>& primal_solution,
                                                 std::vector<f_t>& dual_solution,
                                                 std::vector<f_t>& reduced_costs,
                                                 problem_category_t /*category*/,
                                                 bool status_to_skip,
                                                 bool dual_postsolve)
{
  // Matches apply()'s dispatch: PSLP is the only branch that's special-cased;
  // every other value of `presolver_` (Papilo / Default / None) runs the
  // Papilo postsolve, which is a no-op short-circuit on status_to_skip.
  if (presolver_ == cuopt::mathematical_optimization::presolver_t::PSLP) {
    undo_pslp_host(primal_solution, dual_solution, reduced_costs);
    return;
  }

  if (status_to_skip) { return; }
  undo_papilo_host(primal_solution, dual_solution, reduced_costs, dual_postsolve);
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo(rmm::device_uvector<f_t>& primal_solution,
                                            rmm::device_uvector<f_t>& dual_solution,
                                            rmm::device_uvector<f_t>& reduced_costs,
                                            problem_category_t category,
                                            bool status_to_skip,
                                            bool dual_postsolve,
                                            rmm::cuda_stream_view stream_view)
{
  // PSLP path always runs (it owns the lifted solution in-place); the Papilo
  // path (used for Papilo/Default/None) is allowed to short-circuit via
  // status_to_skip without touching the data. Mirror that here so we don't pay
  // for unnecessary D->H->D copies.
  if (status_to_skip && presolver_ != cuopt::mathematical_optimization::presolver_t::PSLP) {
    return;
  }

  std::vector<f_t> h_primal(primal_solution.size());
  std::vector<f_t> h_dual(dual_solution.size());
  std::vector<f_t> h_rc(reduced_costs.size());
  raft::copy(h_primal.data(), primal_solution.data(), primal_solution.size(), stream_view);
  raft::copy(h_dual.data(), dual_solution.data(), dual_solution.size(), stream_view);
  raft::copy(h_rc.data(), reduced_costs.data(), reduced_costs.size(), stream_view);
  stream_view.synchronize();

  undo_host(h_primal, h_dual, h_rc, category, status_to_skip, dual_postsolve);

  primal_solution.resize(h_primal.size(), stream_view);
  dual_solution.resize(h_dual.size(), stream_view);
  reduced_costs.resize(h_rc.size(), stream_view);
  raft::copy(primal_solution.data(), h_primal.data(), h_primal.size(), stream_view);
  raft::copy(dual_solution.data(), h_dual.data(), h_dual.size(), stream_view);
  raft::copy(reduced_costs.data(), h_rc.data(), h_rc.size(), stream_view);
  stream_view.synchronize();
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
  const std::vector<f_t>& original_primal, std::vector<f_t>& reduced_primal) const
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

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template struct papilo_postsolve_deleter<float>;
template class third_party_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template struct papilo_postsolve_deleter<double>;
template class third_party_presolve_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
