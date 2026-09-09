/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "fj_cpu_binary.cuh"

#include "feasibility_jump.cuh"
#include "fj_cpu.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/integer_scaling.hpp>

#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

constexpr int64_t fj_bin_scale_cap = std::numeric_limits<int16_t>::max();

// prefetch distance
// TODO: check if it actually matters at all for performance
constexpr int32_t fj_bin_pf_dist = 8;

template <typename i_t, typename f_t>
static bool fj_bin_fixed_binary(const fj_cpu_climber_t<i_t, f_t>& c, int32_t v)
{
  if (c.problem->h_var_types[v] != var_t::INTEGER) return false;
  const auto bounds = c.h_var_bounds[v];
  const double lb   = (double)cuopt::get_lower(bounds);
  const double ub   = (double)cuopt::get_upper(bounds);
  return lb == ub && (lb == 0.0 || lb == 1.0);
}

// Width-independent eligibility scan over the climber's host mirrors. Mutates nothing.
template <typename i_t, typename f_t>
fj_bin_scan_t fj_bin_scan(const fj_cpu_climber_t<i_t, f_t>& c, fj_bin_setup_times_t& times)
{
  phase_timer_t timer(times.scan);
  fj_bin_scan_t out;
  const int32_t n = c.problem->n_variables;
  const int32_t m = c.problem->n_constraints;
  if (n <= 0 || m <= 0) {
    out.reject = fj_binary_reject_t::empty_problem;
    return out;
  }

  const double tol               = c.problem->tolerances.integrality_tolerance;
  const auto& is_binary_variable = c.h_is_binary_variable;
  cuopt_assert((int32_t)is_binary_variable.size() == n, "is_binary_variable size mismatch");

  const uint8_t* ignore_var = c.has_bin_elimination ? c.bin_ignore_var.data() : nullptr;
  const uint8_t* ignore_row = c.has_bin_elimination ? c.bin_ignore_row.data() : nullptr;
  for (int32_t v = 0; v < n; ++v) {
    if (ignore_var && ignore_var[v]) continue;
    // Populated at climber init with integer_equal on [0,1] bounds.
    if (!is_binary_variable[v] && !fj_bin_fixed_binary(c, v)) {
      out.reject  = fj_binary_reject_t::non_binary_var;
      out.bad_var = v;
      return out;
    }
  }

  const auto& offsets             = c.problem->offsets;
  const auto& reverse_offsets     = c.problem->reverse_offsets;
  const auto& reverse_constraints = c.problem->reverse_constraints;
  const auto& coeffs              = c.problem->coefficients;
  const auto& cstr_lb             = c.problem->cstr_lb;
  const auto& cstr_ub             = c.problem->cstr_ub;

  cuopt_assert(thrust::all_of(thrust::host,
                              thrust::make_counting_iterator<int32_t>(0),
                              thrust::make_counting_iterator<int32_t>(n),
                              [&reverse_offsets, &reverse_constraints](int32_t v) {
                                const auto first = reverse_constraints.begin() + reverse_offsets[v];
                                const auto last =
                                  reverse_constraints.begin() + reverse_offsets[v + 1];
                                return std::adjacent_find(first, last) == last;
                              }),
               "duplicate variable in CSR row");

  double max_abs_coefficient = 0;
  std::vector<double> row_values;
  for (int32_t r = 0; r < m; ++r) {
    if (ignore_row && ignore_row[r]) continue;
    const double lb       = cstr_lb[r];
    const double ub       = cstr_ub[r];
    const bool lb_fin     = std::isfinite(lb);
    const bool ub_fin     = std::isfinite(ub);
    const double sides[2] = {lb, ub};
    const bool finite[2]  = {lb_fin, ub_fin};

    bool integral = true;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      if (!is_integer(coeffs[k], tol)) {
        integral = false;
        break;
      }
    }

    double row_s = 1.0;
    if (!integral) {
      row_values.clear();
      for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k)
        row_values.push_back(coeffs[k]);
      row_s = find_scaling_rational(row_values,
                                    /*maxscale=*/1.0 / tol,
                                    /*maxdnom=*/fj_bin_scale_cap,
                                    /*maxfinal=*/(double)fj_bin_scale_cap,
                                    /*intcheck_tol=*/tol);
      if (!std::isfinite(row_s) || row_s <= 0.0) {
        out.reject  = fj_binary_reject_t::fractional_coefficient;
        out.bad_row = r;
        return out;
      }
      if (out.row_scale.empty()) out.row_scale.assign(m, 1.0);
      out.row_scale[r] = row_s;
    }

    double row_abs_sum = 0;
    double row_lhs_min = 0;
    double row_lhs_max = 0;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      const double a = row_s * coeffs[k];
      cuopt_assert(is_integer(a, tol), "row scaling left a fractional coefficient");
      const double integral_a = std::round(a);
      const double abs_a      = std::fabs(integral_a);
      row_abs_sum += abs_a;
      if (integral_a < 0) {
        row_lhs_min += integral_a;
      } else {
        row_lhs_max += integral_a;
      }
      if (abs_a > max_abs_coefficient) max_abs_coefficient = abs_a;
    }

    // A binary assignment can drive lhs to sum|coef|; keep that inside the int32 accumulator with
    // room to spare. The int8-only reference engine never needed this bound.
    if (row_abs_sum > (double)(INT32_MAX / 2)) {
      out.reject  = fj_binary_reject_t::lhs_headroom;
      out.bad_row = r;
      return out;
    }

    for (int s = 0; s < 2; ++s) {
      if (!finite[s]) continue;
      const double scaled_side   = row_s * sides[s];
      const double integral_side = is_integer(scaled_side, tol)
                                     ? std::round(scaled_side)
                                     : (s == 0 ? std::ceil(scaled_side) : std::floor(scaled_side));
      if (!fj_bin_in_int32(integral_side)) {
        out.reject  = fj_binary_reject_t::row_bound_out_of_range;
        out.bad_row = r;
        return out;
      }
      const double min_slack = s == 0 ? row_lhs_min - integral_side : integral_side - row_lhs_max;
      const double max_slack = s == 0 ? row_lhs_max - integral_side : integral_side - row_lhs_min;
      if (!fj_bin_in_int32(min_slack) || !fj_bin_in_int32(max_slack)) {
        out.reject  = fj_binary_reject_t::lhs_headroom;
        out.bad_row = r;
        return out;
      }
    }
    // Free rows are dropped: trivially satisfied, contributing nothing to the search.
    out.n_split_constraints += (int32_t)lb_fin + (int32_t)ub_fin;
  }

  if (out.n_split_constraints <= 0) {
    out.reject = fj_binary_reject_t::empty_problem;
    return out;
  }

  if (max_abs_coefficient <= 127.0) {
    out.coefficient_bits = 8;
  } else if (max_abs_coefficient <= 32767.0) {
    out.coefficient_bits = 16;
  } else {
    out.reject = fj_binary_reject_t::coefficient_out_of_range;
  }
  return out;
}

// Build the narrowed, one-sided problem. Called only after fj_bin_scan cleared the instance, so a
// failing check here is a self-consistency bug and refuses the fast path rather than truncating.
template <typename i_t, typename f_t, typename coef_t>
bool fj_bin_narrow(const fj_cpu_climber_t<i_t, f_t>& c,
                   const fj_bin_scan_t& scan,
                   fj_bin_problem_t<coef_t>& pb,
                   fj_bin_setup_times_t& times)
{
  const int32_t n_split = scan.n_split_constraints;
  const int32_t n       = c.problem->n_variables;
  const int32_t m       = c.problem->n_constraints;
  const double tol      = c.problem->tolerances.integrality_tolerance;

  const auto& offsets   = c.problem->offsets;
  const auto& variables = c.problem->variables;
  const auto& coeffs    = c.problem->coefficients;
  const auto& cstr_lb   = c.problem->cstr_lb;
  const auto& cstr_ub   = c.problem->cstr_ub;
  const auto& left_w    = c.h_cstr_left_weights;
  const auto& right_w   = c.h_cstr_right_weights;
  const auto& obj       = c.problem->h_obj_coeffs;

  // Explicit stamps rather than scoped timers, so the three phases below can be delimited without
  // re-nesting them. A failure inside one drops its sample, which only happens on the
  // self-consistency paths that refuse the fast path outright.
  const double narrow_started = tic();

  const uint8_t* ignore_var = c.has_bin_elimination ? c.bin_ignore_var.data() : nullptr;

  pb.n_original = n;
  pb.var_offset.assign(n, 0.0);
  pb.orig_objective.assign(n, 0.0);
  pb.bit_owner.clear();
  pb.bit_owner.reserve(n);
  pb.original_to_bin_mapping.assign(n, -1);
  for (int32_t v = 0; v < n; ++v) {
    pb.orig_objective[v] = obj[v];
    if (ignore_var && ignore_var[v]) continue;
    if (!c.h_is_binary_variable[v] && fj_bin_fixed_binary(c, v)) {
      const auto bounds = c.h_var_bounds[v];
      pb.var_offset[v]  = (double)cuopt::get_lower(bounds) > 0.5 ? 1.0 : 0.0;
      continue;
    }
    pb.original_to_bin_mapping[v] = (int32_t)pb.bit_owner.size();
    pb.bit_owner.push_back(v);
  }
  const int32_t n_engine = (int32_t)pb.bit_owner.size();
  pb.bit_weight.assign(n_engine, 1.0);

  pb.n_variables   = n_engine;
  pb.n_constraints = n_split;
  pb.offsets.assign(1, 0);
  pb.offsets.reserve(n_split + 1);
  pb.bound.reserve(n_split);
  pb.cmax.reserve(n_split);
  pb.initial_weight.reserve(n_split);

  std::vector<double> incoming_weight;
  incoming_weight.reserve(n_split);

  // Each split row inherits the weight of the side it came from: left is the lower-bound side,
  // right the upper.
  //
  // Both sides are stored as a'x <= b. The lower-bound side is negated on the way in, which costs
  // nothing because each side already gets its own copy of the row, and it leaves the slack as
  // bound - lhs everywhere -- so no per-row sign reaches the engine at all. Negation is safe on
  // both fields: the scan admits |coef| up to 127 for int8 and 32767 for int16, and the bound is
  // checked for int32 range after negating.
  auto emit = [&](int32_t r, double side_bound, long side, double weight) -> bool {
    const double s  = scan.row_scale.empty() ? 1.0 : scan.row_scale[r];
    coef_t row_cmax = 1;
    long fixed_lhs  = 0;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      const double a = s * coeffs[k];
      const long ai  = side * std::lround(a);
      if (!is_integer(a, tol) || ai < std::numeric_limits<coef_t>::min() ||
          ai > std::numeric_limits<coef_t>::max()) {
        return false;
      }
      const int32_t v = variables[k];
      if (pb.original_to_bin_mapping[v] < 0) {
        cuopt_assert(!ignore_var || !ignore_var[v],
                     "an eliminated recourse column reached an emitted row");
        fixed_lhs += ai * (long)pb.var_offset[v];
        continue;
      }
      pb.variables.push_back(pb.original_to_bin_mapping[v]);
      pb.coefficients.push_back((coef_t)ai);
      const coef_t abs_a = (coef_t)std::labs(ai);
      if (abs_a > row_cmax) row_cmax = abs_a;
    }
    const double scaled_bound = (double)side * s * side_bound;
    const long b =
      (is_integer(scaled_bound, tol) ? std::lround(scaled_bound) : (long)std::floor(scaled_bound)) -
      fixed_lhs;
    if (!fj_bin_in_int32((double)b)) return false;
    pb.offsets.push_back((int32_t)pb.variables.size());
    pb.bound.push_back((int32_t)b);
    pb.cmax.push_back(row_cmax);
    incoming_weight.push_back(weight);
    return true;
  };

  const uint8_t* ignored_row = c.has_bin_elimination ? c.bin_ignore_row.data() : nullptr;
  for (int32_t r = 0; r < m; ++r) {
    if (ignored_row && ignored_row[r]) continue;
    const double lb = cstr_lb[r];
    const double ub = cstr_ub[r];
    if (std::isfinite(lb) && !emit(r, lb, -1, left_w[r])) return false;
    if (std::isfinite(ub) && !emit(r, ub, 1, right_w[r])) return false;
  }
  if ((int32_t)pb.bound.size() != n_split) return false;
  pb.nnz = (int32_t)pb.variables.size();

  cuopt_assert((int32_t)pb.bit_owner.size() == pb.n_variables,
               "engine column count disagrees with the owner map");
  for (int32_t j = 0; j < pb.n_variables; ++j) {
    cuopt_assert(pb.bit_owner[j] >= 0 && pb.bit_owner[j] < pb.n_original,
                 "owner map names a column outside the model");
    cuopt_assert(pb.bit_weight[j] == 1.0, "a narrowed engine column must carry unit weight");
  }
  for (int32_t k = 0; k < pb.nnz; ++k)
    cuopt_assert(pb.variables[k] >= 0 && pb.variables[k] < pb.n_variables,
                 "engine CSR holds a column outside engine space");

  // One vector of padding past nnz, so the row kernel can load and store whole vectors at the last
  // row without running off the end and can therefore mask its remainder rather than peeling it
  // into a scalar tail. The padding is never read as data: every lane past a row's end is excluded
  // from the gather, the scatter and the store by the row-length mask.
  pb.variables.resize(pb.nnz + fj_bin_simd_padding, 0);
  pb.coefficients.resize(pb.nnz + fj_bin_simd_padding, (coef_t)0);

  // Scale the incoming weights into the DDFW band by one global factor, so relative structure
  // survives while every row clears the donation floor. Capped so the largest scaled weight stays
  // clear of packed-score saturation; where the cap binds, the smallest rows sit below the floor.
  // TODO: bound the scaled weights by derivation instead of leaving them open. The packed score
  // holds while a variable's aggregate base stays under 2^16, and that aggregate is bounded by the
  // sum of weights over the rows the variable appears in, so 2^16 / max_var_degree gives a per-row
  // bound computable here from the transpose. Left uncapped for now, matching the reference
  // engine, which shipped with its weight cap disabled and relied on the end-of-solve saturation
  // report to say whether a bound was needed.
  double w_min = std::numeric_limits<double>::infinity();
  for (double w : incoming_weight) {
    if (w > 0 && w < w_min) w_min = w;
  }
  double scale = 1.0;
  if (std::isfinite(w_min) && w_min > 0) {
    scale = (double)fj_bin_ddfw_init / w_min;
    if (scale < 1.0) scale = 1.0;
  }
  for (double w : incoming_weight) {
    int32_t scaled = w > 0 ? (int32_t)std::lround(w * scale) : fj_bin_ddfw_init;
    if (scaled < 1) scaled = 1;
    pb.initial_weight.push_back(scaled);
  }
  times.narrow += toc(narrow_started);

  const double transpose_started = tic();
  // Transpose, plus the reverse-nnz to CSR-nnz map the apply path uses to store the flipped
  // variable's own score delta.
  pb.reverse_offsets.assign(n_engine + 1, 0);
  for (int32_t k = 0; k < pb.nnz; ++k)
    pb.reverse_offsets[pb.variables[k] + 1]++;
  for (int32_t v = 0; v < n_engine; ++v)
    pb.reverse_offsets[v + 1] += pb.reverse_offsets[v];
  pb.reverse_constraints.resize(pb.nnz);
  pb.reverse_coefficients.resize(pb.nnz);
  pb.reverse_to_csr.resize(pb.nnz);
  pb.incident_row_cmax.resize(pb.nnz);
  {
    std::vector<int32_t> cursor(pb.reverse_offsets.begin(), pb.reverse_offsets.begin() + n_engine);
    for (int32_t r = 0; r < n_split; ++r) {
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t slot            = cursor[pb.variables[k]]++;
        pb.reverse_constraints[slot]  = r;
        pb.reverse_coefficients[slot] = pb.coefficients[k];
        pb.reverse_to_csr[slot]       = k;
        pb.incident_row_cmax[slot]    = pb.cmax[r];
      }
    }
  }
  // Lookahead room for the row walk: a vector of overhang for the kernel's unit-stride loads, and
  // the prefetch distance the scalar path uses. Reads land on row 0, harmlessly, and every lane
  // past a variable's range is masked out of the gather, the scatter and the compress.
  const int32_t rpad = fj_bin_pf_dist > fj_bin_simd_padding ? fj_bin_pf_dist : fj_bin_simd_padding;
  pb.reverse_constraints.resize(pb.nnz + rpad, 0);
  pb.reverse_coefficients.resize(pb.nnz + rpad, (coef_t)0);
  pb.incident_row_cmax.resize(pb.nnz + rpad, (coef_t)1);

  pb.objective.resize(n_engine);
  for (int32_t j = 0; j < n_engine; ++j) {
    pb.objective[j] = obj[pb.bit_owner[j]];
    if (pb.objective[j] != 0.0) pb.objective_vars.push_back(j);
  }
  times.transpose += toc(transpose_started);

  return true;
}

// Bit budget for one general integer's domain.
constexpr int32_t fj_bin_encode_max_bits = 16;
// Cap on the bit-variable count relative to the model's variable count, bounding the SIMD sweep.
constexpr int64_t fj_bin_encode_max_growth = 6;

// Bits needed to represent the integers 0..W inclusive.
static inline int32_t fj_bin_encode_nbits(int64_t W)
{
  int32_t bits = 0;
  while (((int64_t)1 << bits) - 1 < W)
    ++bits;
  return bits;
}

// Encodes an all-integer model with bounded general integers into bits: x in [L,U] becomes
// x = L + sum_k w_k b_k over weights 1, 2, ..., 2^(nbits-2), R, with R closing the range at W =
// U-L.
template <typename i_t, typename f_t, typename coef_t>
bool fj_bin_encode(const fj_cpu_climber_t<i_t, f_t>& c,
                   fj_bin_problem_t<coef_t>& pb,
                   int& coefficient_bits,
                   fj_bin_setup_times_t& times)
{
  phase_timer_t timer(times.encode);
  const int32_t n = c.problem->n_variables;
  const int32_t m = c.problem->n_constraints;
  if (n <= 0 || m <= 0) return false;

  const double tol = c.problem->tolerances.integrality_tolerance;

  const auto& var_bounds = c.h_var_bounds;
  const auto& var_types  = c.problem->h_var_types;
  const auto& offsets    = c.problem->offsets;
  const auto& variables  = c.problem->variables;
  const auto& coeffs     = c.problem->coefficients;
  const auto& cstr_lb    = c.problem->cstr_lb;
  const auto& cstr_ub    = c.problem->cstr_ub;
  const auto& left_w     = c.h_cstr_left_weights;
  const auto& right_w    = c.h_cstr_right_weights;
  const auto& obj        = c.problem->h_obj_coeffs;

  std::vector<double> lower(n);
  std::vector<double> upper(n);
  std::vector<int32_t> nbits(n);
  std::vector<int32_t> bit_start(n);
  int64_t total_bits = 0;
  for (int32_t v = 0; v < n; ++v) {
    if (var_types[v] != var_t::INTEGER) return false;
    auto bounds    = var_bounds[v];
    const double x = (double)cuopt::get_lower(bounds);
    const double y = (double)cuopt::get_upper(bounds);
    if (!std::isfinite(x) || !std::isfinite(y) || y < x) return false;
    if (!is_integer(x, tol) || !is_integer(y, tol)) return false;

    lower[v]        = std::round(x);
    upper[v]        = std::round(y);
    const int64_t W = (int64_t)(upper[v] - lower[v]);

    nbits[v] = fj_bin_encode_nbits(W);
    if (nbits[v] > fj_bin_encode_max_bits) return false;
    bit_start[v] = (int32_t)total_bits;
    total_bits += nbits[v];
  }
  if (total_bits <= 0 || total_bits > (int64_t)INT32_MAX / 2) return false;
  if (total_bits > fj_bin_encode_max_growth * (int64_t)n) return false;

  const int32_t n_bits = (int32_t)total_bits;

  pb.encoded    = true;
  pb.n_original = n;
  pb.var_offset = lower;
  pb.orig_objective.assign(n, 0.0);
  pb.bit_owner.assign(n_bits, 0);
  pb.original_to_bin_mapping.assign(n, -1);
  pb.bit_weight.assign(n_bits, 0.0);
  for (int32_t v = 0; v < n; ++v) {
    int64_t covered = 0;
    const int64_t W = (int64_t)(upper[v] - lower[v]);
    for (int32_t k = 0; k < nbits[v]; ++k) {
      const int64_t w = k + 1 < nbits[v] ? (int64_t)1 << k : W - covered;
      covered += w;
      pb.bit_owner[bit_start[v] + k]  = v;
      pb.bit_weight[bit_start[v] + k] = (double)w;
    }
    if (nbits[v] == 1) pb.original_to_bin_mapping[v] = bit_start[v];
    cuopt_assert(covered == W, "bit weights do not close the domain exactly");
  }

  pb.n_variables = n_bits;
  pb.offsets.assign(1, 0);
  pb.bound.clear();
  pb.cmax.clear();
  pb.initial_weight.clear();
  pb.variables.clear();
  pb.coefficients.clear();

  std::vector<double> incoming_weight;
  std::vector<double> row_values;
  double max_abs_coefficient = 0;

  // One side of one row, as a'b <= bound in bit space with sum(a_j L_j) folded into the bound.
  auto emit = [&](int32_t r, double side_bound, long side, double weight) -> bool {
    double fixed = 0;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k)
      fixed += coeffs[k] * lower[variables[k]];
    const double folded_bound = side_bound - fixed;

    row_values.clear();
    bool integral = true;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      row_values.push_back(coeffs[k]);
      if (!is_integer(coeffs[k], tol)) integral = false;
    }

    double s = 1.0;
    if (!integral) {
      s = find_scaling_rational(
        row_values, 1.0 / tol, fj_bin_scale_cap, (double)fj_bin_scale_cap, tol);
      if (!std::isfinite(s) || s <= 0.0) return false;
    }

    coef_t row_cmax    = 1;
    double row_abs_sum = 0;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      const int32_t v = variables[k];
      const double a  = s * coeffs[k];
      if (!is_integer(a, tol)) return false;
      const long ai = std::lround(a);
      for (int32_t bk = 0; bk < nbits[v]; ++bk) {
        const int32_t bit = bit_start[v] + bk;
        const long scaled = side * ai * std::lround(pb.bit_weight[bit]);
        const long abs_a  = std::labs(scaled);
        // Bounded by magnitude, so cmax below and the negated side both stay representable.
        if (abs_a > (long)std::numeric_limits<coef_t>::max()) return false;
        pb.variables.push_back(bit);
        pb.coefficients.push_back((coef_t)scaled);

        if (abs_a > (long)row_cmax) row_cmax = (coef_t)abs_a;
        row_abs_sum += (double)abs_a;
        if ((double)abs_a > max_abs_coefficient) max_abs_coefficient = (double)abs_a;
      }
    }
    if (row_abs_sum > (double)(INT32_MAX / 2)) return false;

    const double scaled_bound = (double)side * s * folded_bound;
    const double bound =
      is_integer(scaled_bound, tol) ? std::round(scaled_bound) : std::floor(scaled_bound);
    if (!fj_bin_in_int32(bound)) return false;
    // A bit assignment can drive lhs anywhere in [-row_abs_sum, row_abs_sum].
    if (!fj_bin_in_int32(bound - row_abs_sum) || !fj_bin_in_int32(bound + row_abs_sum))
      return false;

    pb.offsets.push_back((int32_t)pb.variables.size());
    pb.bound.push_back((int32_t)bound);
    pb.cmax.push_back(row_cmax);
    incoming_weight.push_back(weight);
    return true;
  };

  const uint8_t* ignored_row = c.has_bin_elimination ? c.bin_ignore_row.data() : nullptr;
  for (int32_t r = 0; r < m; ++r) {
    if (ignored_row && ignored_row[r]) continue;
    const double lb = cstr_lb[r];
    const double ub = cstr_ub[r];
    if (std::isfinite(lb) && !emit(r, lb, -1, left_w[r])) return false;
    if (std::isfinite(ub) && !emit(r, ub, 1, right_w[r])) return false;
  }
  pb.n_constraints = (int32_t)pb.bound.size();
  if (pb.n_constraints <= 0) return false;
  pb.nnz = (int32_t)pb.variables.size();

  if (max_abs_coefficient <= 127.0) {
    coefficient_bits = 8;
  } else if (max_abs_coefficient <= 32767.0) {
    coefficient_bits = 16;
  } else {
    return false;
  }

  pb.variables.resize(pb.nnz + fj_bin_simd_padding, 0);
  pb.coefficients.resize(pb.nnz + fj_bin_simd_padding, (coef_t)0);

  double w_min = std::numeric_limits<double>::infinity();
  for (double w : incoming_weight) {
    if (w > 0 && w < w_min) w_min = w;
  }
  double scale = 1.0;
  if (std::isfinite(w_min) && w_min > 0) {
    scale = (double)fj_bin_ddfw_init / w_min;
    if (scale < 1.0) scale = 1.0;
  }
  for (double w : incoming_weight) {
    int32_t scaled = w > 0 ? (int32_t)std::lround(w * scale) : fj_bin_ddfw_init;
    if (scaled < 1) scaled = 1;
    pb.initial_weight.push_back(scaled);
  }

  pb.reverse_offsets.assign(n_bits + 1, 0);
  for (int32_t k = 0; k < pb.nnz; ++k)
    pb.reverse_offsets[pb.variables[k] + 1]++;
  for (int32_t v = 0; v < n_bits; ++v)
    pb.reverse_offsets[v + 1] += pb.reverse_offsets[v];
  pb.reverse_constraints.resize(pb.nnz);
  pb.reverse_coefficients.resize(pb.nnz);
  pb.reverse_to_csr.resize(pb.nnz);
  pb.incident_row_cmax.resize(pb.nnz);
  {
    std::vector<int32_t> cursor(pb.reverse_offsets.begin(), pb.reverse_offsets.begin() + n_bits);
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t slot            = cursor[pb.variables[k]]++;
        pb.reverse_constraints[slot]  = r;
        pb.reverse_coefficients[slot] = pb.coefficients[k];
        pb.reverse_to_csr[slot]       = k;
        pb.incident_row_cmax[slot]    = pb.cmax[r];
      }
    }
  }
  const int32_t rpad = fj_bin_pf_dist > fj_bin_simd_padding ? fj_bin_pf_dist : fj_bin_simd_padding;
  pb.reverse_constraints.resize(pb.nnz + rpad, 0);
  pb.reverse_coefficients.resize(pb.nnz + rpad, (coef_t)0);
  pb.incident_row_cmax.resize(pb.nnz + rpad, (coef_t)1);

  pb.objective.assign(n_bits, 0.0);
  pb.objective_vars.clear();
  for (int32_t v = 0; v < n; ++v) {
    pb.orig_objective[v] = obj[v];
    if (obj[v] == 0.0) continue;
    for (int32_t bk = 0; bk < nbits[v]; ++bk) {
      const int32_t bit = bit_start[v] + bk;
      pb.objective[bit] = obj[v] * pb.bit_weight[bit];
      if (pb.objective[bit] != 0.0) pb.objective_vars.push_back(bit);
    }
  }

  return true;
}

#if MIP_INSTANTIATE_FLOAT
template fj_bin_scan_t fj_bin_scan(const fj_cpu_climber_t<int, float>&, fj_bin_setup_times_t&);
template bool fj_bin_narrow(
  const fj_cpu_climber_t<int, float>&, const fj_bin_scan_t&, fj_bin_problem_t<int8_t>&, fj_bin_setup_times_t&);
template bool fj_bin_narrow(const fj_cpu_climber_t<int, float>&,
                            const fj_bin_scan_t&,
                            fj_bin_problem_t<int16_t>&,
                            fj_bin_setup_times_t&);
template bool fj_bin_encode(
  const fj_cpu_climber_t<int, float>&, fj_bin_problem_t<int8_t>&, int&, fj_bin_setup_times_t&);
template bool fj_bin_encode(
  const fj_cpu_climber_t<int, float>&, fj_bin_problem_t<int16_t>&, int&, fj_bin_setup_times_t&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template fj_bin_scan_t fj_bin_scan(const fj_cpu_climber_t<int, double>&, fj_bin_setup_times_t&);
template bool fj_bin_narrow(
  const fj_cpu_climber_t<int, double>&, const fj_bin_scan_t&, fj_bin_problem_t<int8_t>&, fj_bin_setup_times_t&);
template bool fj_bin_narrow(const fj_cpu_climber_t<int, double>&,
                            const fj_bin_scan_t&,
                            fj_bin_problem_t<int16_t>&,
                            fj_bin_setup_times_t&);
template bool fj_bin_encode(
  const fj_cpu_climber_t<int, double>&, fj_bin_problem_t<int8_t>&, int&, fj_bin_setup_times_t&);
template bool fj_bin_encode(
  const fj_cpu_climber_t<int, double>&, fj_bin_problem_t<int16_t>&, int&, fj_bin_setup_times_t&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
