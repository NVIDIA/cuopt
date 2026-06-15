/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::io {

/**
 * @brief Options for quadratic-constraint Q COO canonicalization.
 *
 * Internal storage uses one triplet per variable pair: off-diagonal cross terms store the full
 * coefficient on x_i*x_j (e.g. -2*d for rotated SOC), not symmetric halves.
 *
 * Canonicalization runs at ingestion boundaries only:
 * - MPS/LP: append_quadratic_constraint()
 * - C API: optimization_problem add_quadratic_constraint()
 * - Python view -> solver: populate_from_data_model_view()
 * - gRPC decode: before set_quadratic_constraints()
 * - MPS export: mps_writer QCMATRIX emission
 */
template <typename f_t>
struct qc_coo_canonicalize_options_t {
  /** When true (MPS QCMATRIX), every off-diagonal pair must appear symmetrically in the input. */
  bool require_symmetric_offdiagonal_pairs{false};
  f_t tol{1e-12};
  std::string constraint_name{};
};

/**
 * @brief Canonicalize quadratic-constraint Q in COO form.
 *
 * - Merges duplicate (row, col) indices by summing coefficients.
 * - Collapses symmetric off-diagonal pairs (a,b,v)+(b,a,v) into a single (min,max,2v) entry.
 * - Leaves genuinely unsymmetric pairs as separate entries (general convex quadratics).
 * - Sorts output by (row, col).
 */
template <typename i_t, typename f_t>
void canonicalize_qc_coo(std::vector<i_t>& rows,
                         std::vector<i_t>& cols,
                         std::vector<f_t>& vals,
                         const qc_coo_canonicalize_options_t<f_t>& opts = {});

/** Canonicalize Q on a constraint entry with rows/cols/vals and constraint_row_name. */
template <typename QC>
void canonicalize_qc_entry(QC& qc, bool require_symmetric_offdiagonal_pairs = false)
{
  using f_t = typename std::decay_t<decltype(qc.vals)>::value_type;
  qc_coo_canonicalize_options_t<f_t> opts;
  opts.require_symmetric_offdiagonal_pairs = require_symmetric_offdiagonal_pairs;
  opts.constraint_name                     = qc.constraint_row_name;
  canonicalize_qc_coo(qc.rows, qc.cols, qc.vals, opts);
}

}  // namespace cuopt::linear_programming::io
