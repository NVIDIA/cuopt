/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#include <quadratic_constraint_coo.hpp>

namespace cuopt::linear_programming::io {

namespace {

template <typename i_t>
struct pair_hash {
  size_t operator()(const std::pair<i_t, i_t>& p) const noexcept
  {
    return std::hash<i_t>{}(p.first) ^ (std::hash<i_t>{}(p.second) << 1);
  }
};

template <typename f_t>
bool approx_eq(f_t a, f_t b, f_t tol)
{
  const f_t scale = std::max({f_t(1), std::abs(a), std::abs(b)});
  return std::abs(a - b) <= tol * scale;
}

template <typename i_t, typename f_t>
f_t lookup_coeff(const std::unordered_map<std::pair<i_t, i_t>, f_t, pair_hash<i_t>>& agg,
                 i_t r,
                 i_t c,
                 f_t tol)
{
  const auto it = agg.find({r, c});
  if (it == agg.end()) { return f_t(0); }
  return std::abs(it->second) > tol ? it->second : f_t(0);
}

}  // namespace

template <typename i_t, typename f_t>
void canonicalize_qc_coo(std::vector<i_t>& rows,
                         std::vector<i_t>& cols,
                         std::vector<f_t>& vals,
                         const qc_coo_canonicalize_options_t<f_t>& opts)
{
  const size_t n = vals.size();
  cuopt_expects(rows.size() == n && cols.size() == n,
                error_type_t::ValidationError,
                "Q COO rows/cols/vals length mismatch");

  if (n == 0) {
    rows.clear();
    cols.clear();
    vals.clear();
    return;
  }

  std::unordered_map<std::pair<i_t, i_t>, f_t, pair_hash<i_t>> agg;
  agg.reserve(n);
  for (size_t t = 0; t < n; ++t) {
    const i_t r = rows[t];
    const i_t c = cols[t];
    const f_t v = vals[t];
    if (std::abs(v) <= opts.tol) { continue; }
    agg[{r, c}] += v;
  }

  std::vector<std::tuple<i_t, i_t, f_t>> out;
  out.reserve(agg.size());

  std::unordered_map<std::pair<i_t, i_t>, char, pair_hash<i_t>> visited;
  for (const auto& [rc, v] : agg) {
    if (rc.first == rc.second) { continue; }
    const i_t lo = std::min(rc.first, rc.second);
    const i_t hi = std::max(rc.first, rc.second);
    if (visited[{lo, hi}]) { continue; }
    visited[{lo, hi}] = 1;

    const f_t v_lo_hi    = lookup_coeff(agg, lo, hi, opts.tol);
    const f_t v_hi_lo    = lookup_coeff(agg, hi, lo, opts.tol);
    const bool has_lo_hi = std::abs(v_lo_hi) > opts.tol;
    const bool has_hi_lo = std::abs(v_hi_lo) > opts.tol;

    if (opts.require_symmetric_offdiagonal_pairs) {
      cuopt_expects(has_lo_hi && has_hi_lo,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' QCMATRIX off-diagonal (%d,%d) requires a matching "
                    "(%d,%d) entry",
                    opts.constraint_name.c_str(),
                    static_cast<int>(lo),
                    static_cast<int>(hi),
                    static_cast<int>(hi),
                    static_cast<int>(lo));
      cuopt_expects(
        approx_eq(v_lo_hi, v_hi_lo, opts.tol),
        error_type_t::ValidationError,
        "Quadratic constraint '%s' QCMATRIX symmetric off-diagonals (%d,%d) and (%d,%d) must "
        "match; got %.17g and %.17g",
        opts.constraint_name.c_str(),
        static_cast<int>(lo),
        static_cast<int>(hi),
        static_cast<int>(hi),
        static_cast<int>(lo),
        static_cast<double>(v_lo_hi),
        static_cast<double>(v_hi_lo));
    }

    if (has_lo_hi && has_hi_lo && approx_eq(v_lo_hi, v_hi_lo, opts.tol)) {
      out.emplace_back(lo, hi, v_lo_hi + v_hi_lo);
    } else {
      if (has_lo_hi) { out.emplace_back(lo, hi, v_lo_hi); }
      if (has_hi_lo) { out.emplace_back(hi, lo, v_hi_lo); }
    }
  }

  for (const auto& [rc, v] : agg) {
    if (rc.first == rc.second) { out.emplace_back(rc.first, rc.second, v); }
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

template void canonicalize_qc_coo<int, float>(std::vector<int>&,
                                              std::vector<int>&,
                                              std::vector<float>&,
                                              const qc_coo_canonicalize_options_t<float>&);
template void canonicalize_qc_coo<int, double>(std::vector<int>&,
                                               std::vector<int>&,
                                               std::vector<double>&,
                                               const qc_coo_canonicalize_options_t<double>&);

}  // namespace cuopt::linear_programming::io
