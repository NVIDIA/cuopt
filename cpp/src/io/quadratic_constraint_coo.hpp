/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/error.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::io {

/**
 * @brief Canonicalize a symmetric matrix in COO form to lower-triangular storage.
 *
 * - Merges duplicate (row, col) indices by summing coefficients.
 * - Merges each off-diagonal variable pair into one (min,max) entry with coefficient
 *   Q[min,max]+Q[max,min] (x^T Q x semantics).
 * - Sorts output by (row, col).
 *
 * @param require_symmetric_offdiagonal_pairs Input validation only (MPS QCMATRIX). When true, each
 * off-diagonal pair must appear as both (i,j,v) and (j,i,v) with equal v before merge; incomplete
 * or mismatched pairs are rejected. When false (default), a single orientation is accepted and
 * merged with zero for the missing half.
 */
template <typename i_t, typename f_t>
void canonicalize_coo_matrix(std::vector<i_t>& rows,
                             std::vector<i_t>& cols,
                             std::vector<f_t>& vals,
                             bool require_symmetric_offdiagonal_pairs = false);

}  // namespace cuopt::linear_programming::io
