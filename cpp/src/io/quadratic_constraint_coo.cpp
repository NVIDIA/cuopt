/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#include <quadratic_constraint_coo.hpp>

#include <cmath>
#include <limits>

namespace cuopt::linear_programming::io {

namespace {

template <typename i_t>
struct pair_hash {
  size_t operator()(const std::pair<i_t, i_t>& p) const noexcept
  {
    return std::hash<i_t>{}(p.first) ^ (std::hash<i_t>{}(p.second) << 1);
  }
};

template <typename i_t, typename f_t>
f_t lookup_coeff(const std::unordered_map<std::pair<i_t, i_t>, f_t, pair_hash<i_t>>& agg,
                 i_t r,
                 i_t c)
{
  const f_t eps = std::numeric_limits<f_t>::epsilon();
  const auto it = agg.find({r, c});
  if (it == agg.end() || std::abs(it->second) <= eps) { return f_t(0); }
  return it->second;
}

}  // namespace

template <typename i_t, typename f_t>
void canonicalize_coo_matrix(std::vector<i_t>& rows,
                             std::vector<i_t>& cols,
                             std::vector<f_t>& vals,
                             bool require_symmetric_offdiagonal_pairs)
{
  const size_t n = vals.size();
  cuopt_expects(rows.size() == n && cols.size() == n,
                error_type_t::ValidationError,
                "COO rows/cols/vals length mismatch");

  if (n == 0) {
    rows.clear();
    cols.clear();
    vals.clear();
    return;
  }

  // Aggregate duplicate entries
  std::unordered_map<std::pair<i_t, i_t>, f_t, pair_hash<i_t>> agg;
  agg.reserve(n);
  for (size_t t = 0; t < n; ++t) {
    const i_t r = rows[t];
    const i_t c = cols[t];
    const f_t v = vals[t];
    if (std::abs(v) <= std::numeric_limits<f_t>::epsilon()) { continue; }
    agg[{r, c}] += v;
  }

  std::vector<std::tuple<i_t, i_t, f_t>> out;
  out.reserve(agg.size());

  // One lower-triangular entry per off-diagonal variable pair: coeff = sum of both orientations.
  std::unordered_map<std::pair<i_t, i_t>, char, pair_hash<i_t>> visited;
  for (const auto& [rc, v] : agg) {
    if (rc.first == rc.second) { continue; }
    const i_t lo = std::min(rc.first, rc.second);
    const i_t hi = std::max(rc.first, rc.second);
    if (visited[{lo, hi}]) { continue; }
    visited[{lo, hi}] = 1;

    const f_t eps        = std::numeric_limits<f_t>::epsilon();
    const f_t v_lo_hi    = lookup_coeff(agg, lo, hi);
    const f_t v_hi_lo    = lookup_coeff(agg, hi, lo);
    const bool has_lo_hi = std::abs(v_lo_hi) > eps;
    const bool has_hi_lo = std::abs(v_hi_lo) > eps;

    if (require_symmetric_offdiagonal_pairs) {
      cuopt_expects(has_lo_hi && has_hi_lo,
                    error_type_t::ValidationError,
                    "QCMATRIX off-diagonal (%d,%d) requires a matching (%d,%d) entry",
                    static_cast<int>(lo),
                    static_cast<int>(hi),
                    static_cast<int>(hi),
                    static_cast<int>(lo));
      cuopt_expects(
        std::abs(v_lo_hi - v_hi_lo) <= eps,
        error_type_t::ValidationError,
        "QCMATRIX symmetric off-diagonals (%d,%d) and (%d,%d) must match; got %.17g and %.17g",
        static_cast<int>(lo),
        static_cast<int>(hi),
        static_cast<int>(hi),
        static_cast<int>(lo),
        static_cast<double>(v_lo_hi),
        static_cast<double>(v_hi_lo));
    }

    const f_t cross = v_lo_hi + v_hi_lo;
    if (std::abs(cross) > eps) { out.emplace_back(lo, hi, cross); }
  }

  for (const auto& [rc, v] : agg) {
    if (rc.first == rc.second && std::abs(v) > std::numeric_limits<f_t>::epsilon()) {
      out.emplace_back(rc.first, rc.second, v);
    }
  }

  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    if (std::get<0>(a) != std::get<0>(b)) { return std::get<0>(a) < std::get<0>(b); }
    return std::get<1>(a) < std::get<1>(b);
  });

  rows.resize(out.size());
  cols.resize(out.size());
  vals.resize(out.size());
  for (size_t t = 0; t < out.size(); ++t) {
    rows[t] = std::get<0>(out[t]);
    cols[t] = std::get<1>(out[t]);
    vals[t] = std::get<2>(out[t]);
  }
}

template void canonicalize_coo_matrix<int, float>(std::vector<int>&,
                                                  std::vector<int>&,
                                                  std::vector<float>&,
                                                  bool);
template void canonicalize_coo_matrix<int, double>(std::vector<int>&,
                                                   std::vector<int>&,
                                                   std::vector<double>&,
                                                   bool);

}  // namespace cuopt::linear_programming::io
