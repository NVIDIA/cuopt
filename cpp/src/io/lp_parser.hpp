/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/io/mps_data_model.hpp>
#include <mps_parser_internal.hpp>

#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace cuopt::linear_programming::io {

/**
 * @brief Parser for the LP format.
 *
 * The class is a thin holder for the parsed problem data. All parsing
 * machinery (tokenizer, expression/section parsers, token types) lives in
 * src/lp_parser.cpp and is never exposed.
 *
 * The public fields mirror mps_parser_t so tests and tools can introspect
 * the same shape of intermediate data from either parser. Finalization
 * (CSR flatten, constraint-bound derivation, quadratic objective assembly)
 * is performed by finalize_problem() inside src/io/lp_parser.cpp.
 */
template <typename i_t, typename f_t>
class lp_parser_t {
 public:
  // Parses `file` and populates `problem`.
  lp_parser_t(mps_data_model_t<i_t, f_t>& problem, const std::string& file);

  // Parses `input` (LP format text already loaded in memory) and populates
  // `problem`. Used by read_lp_from_string() — compressed inputs are only
  // supported via the file-path constructor since compression is detected
  // from the path suffix.
  lp_parser_t(mps_data_model_t<i_t, f_t>& problem, std::string_view input);

  // Intermediate parsed problem data (mirrors mps_parser_t's public fields).
  std::string problem_name{};
  std::vector<std::string> row_names{};
  std::vector<RowType> row_types{};
  std::string objective_name{"OBJ"};
  std::vector<std::string> var_names{};
  std::vector<char> var_types{};
  std::vector<std::vector<i_t>> A_indices{};
  std::vector<std::vector<f_t>> A_values{};
  std::vector<f_t> b_values{};
  std::vector<f_t> c_values{};
  f_t objective_offset_value{0};
  std::vector<f_t> variable_upper_bounds{};
  std::vector<f_t> variable_lower_bounds{};
  bool maximize{false};
  // Quadratic objective entries (row, col, value) in upper-triangular
  // QUADOBJ convention; finalize_problem() mirrors to the full symmetric
  // matrix and applies the *0.5 factor required by cuOpt's x^T Q x form.
  std::vector<std::tuple<i_t, i_t, f_t>> quadobj_entries{};

  // Per-row data for constraints whose LHS contains a quadratic bracket.
  // These rows do NOT appear in row_names/row_types/A_indices/A_values/
  // b_values — those vectors carry only the linear constraints — and they
  // are emitted to the data model after finalize_problem via
  // mps_data_model_t::append_quadratic_constraint(), with row indices
  // assigned linear_row_count..linear_row_count+nqc (mirroring MPS's
  // QCMATRIX handling).
  struct quadratic_constraint_block_t {
    std::string row_name{};
    RowType row_type{};
    std::vector<i_t> linear_indices{};
    std::vector<f_t> linear_values{};
    f_t rhs_value{};
    // Upper-triangular (i <= j) raw triples directly from the LP source
    // (face value, no /2). The post-pass mirrors and halves off-diagonals
    // to build the symmetric Q in CSR.
    std::vector<std::tuple<i_t, i_t, f_t>> quad_triples{};
  };
  std::vector<quadratic_constraint_block_t> quadratic_constraint_blocks{};
};

}  // namespace cuopt::linear_programming::io
