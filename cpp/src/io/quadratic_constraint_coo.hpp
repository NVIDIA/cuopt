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
 * @brief Options for symmetric COO matrix canonicalization.
 *
 * Canonical storage uses one triplet per variable pair. Off-diagonal cross terms store the full
 * coefficient on x_i*x_j in x^T Q x (e.g. -2*d for rotated SOC), not separate (i,j) and (j,i)
 * entries.
 */
template <typename f_t>
struct qc_coo_canonicalize_options_t {
  /**
   * Input validation only; does not change canonical output rules.
   *
   * When false (default, API/LP/C paths): a single off-diagonal (i,j,v) is accepted for x_i*x_j.
   * Matching symmetric pairs (i,j,v)+(j,i,v) still merge to one stored entry with coefficient 2*v.
   *
   * When true (MPS QCMATRIX): each off-diagonal pair must appear as both (i,j,v) and (j,i,v)
   * with equal v before merge; incomplete pairs are rejected. MPS files encode cross terms as
   * symmetric halves; this flag enforces that input shape.
   */
  bool require_symmetric_offdiagonal_pairs{false};
  f_t tol{1e-12};
  std::string constraint_name{};
};

/**
 * @brief Canonicalize a symmetric matrix in COO form to one triplet per variable pair.
 *
 * - Merges duplicate (row, col) indices by summing coefficients.
 * - Collapses matching symmetric off-diagonal pairs (a,b,v)+(b,a,v) into a single (min,max,2v).
 * - Leaves genuinely unsymmetric pairs as separate entries (general convex quadratics).
 * - Sorts output by (row, col).
 */
template <typename i_t, typename f_t>
void canonicalize_qc_coo(std::vector<i_t>& rows,
                         std::vector<i_t>& cols,
                         std::vector<f_t>& vals,
                         const qc_coo_canonicalize_options_t<f_t>& opts = {});

/** @copydoc canonicalize_qc_coo */
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
