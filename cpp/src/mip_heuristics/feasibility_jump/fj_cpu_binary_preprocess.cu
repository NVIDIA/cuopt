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
#include <mip_heuristics/utils.hpp>
#include <utilities/integer_scaling.hpp>

#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

constexpr int64_t fj_bin_scale_cap = std::numeric_limits<int16_t>::max();

template <typename i_t, typename f_t>
static bool fj_bin_fixed_binary(const fj_cpu_climber_t<i_t, f_t>& c, int32_t v)
{
  if (c.problem->h_var_types[v] != var_t::INTEGER) return false;
  const auto bounds = c.h_var_bounds[v];
  const double lb   = (double)cuopt::get_lower(bounds);
  const double ub   = (double)cuopt::get_upper(bounds);
  return lb == ub && (lb == 0.0 || lb == 1.0);
}

static void initialize_row_weights(const std::vector<double>& incoming_weight,
                                   std::vector<int32_t>& initial_weight)
{
  double w_min = std::numeric_limits<double>::infinity();
  for (double w : incoming_weight) {
    if (w > 0 && w < w_min) w_min = w;
  }
  double scale = 1.0;
  if (std::isfinite(w_min) && w_min > 0) {
    scale = (double)fj_bin_ddfw_init / w_min;
    if (scale < 1.0) scale = 1.0;
  }
  initial_weight.clear();
  initial_weight.reserve(incoming_weight.size());
  for (double w : incoming_weight) {
    int32_t scaled = w > 0 ? (int32_t)std::lround(w * scale) : fj_bin_ddfw_init;
    if (scaled < 1) scaled = 1;
    initial_weight.push_back(scaled);
  }
}

template <typename coef_t>
static void build_transpose(fj_bin_problem_t<coef_t>& pb)
{
  pb.reverse_offsets.assign(pb.n_variables + 1, 0);
  for (int32_t k = 0; k < pb.nnz; ++k)
    pb.reverse_offsets[pb.variables[k] + 1]++;
  for (int32_t v = 0; v < pb.n_variables; ++v)
    pb.reverse_offsets[v + 1] += pb.reverse_offsets[v];
  pb.reverse_constraints.resize(pb.nnz);
  pb.reverse_coefficients.resize(pb.nnz);
  pb.reverse_to_csr.resize(pb.nnz);
  pb.incident_row_cmax.resize(pb.nnz);
  {
    std::vector<int32_t> cursor(pb.reverse_offsets.begin(),
                                pb.reverse_offsets.begin() + pb.n_variables);
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
  pb.reverse_constraints.resize(pb.nnz + fj_bin_simd_padding, 0);
  pb.reverse_coefficients.resize(pb.nnz + fj_bin_simd_padding, (coef_t)0);
  pb.incident_row_cmax.resize(pb.nnz + fj_bin_simd_padding, (coef_t)1);
}

// go over each row and var and check if they are eligible for the binary fastpath.
template <typename i_t, typename f_t>
fj_bin_scan_t fj_bin_scan(const fj_cpu_climber_t<i_t, f_t>& c, fj_bin_setup_times_t& times)
{
  phase_timer_t timer(times.scan);
  fj_bin_scan_t out;
  const int32_t n_cols = c.problem->n_variables;
  const int32_t n_rows = c.problem->n_constraints;
  if (n_cols <= 0 || n_rows <= 0) {
    out.reject = fj_binary_reject_t::empty_problem;
    return out;
  }

  const double tol               = c.problem->tolerances.integrality_tolerance;
  const auto& is_binary_variable = c.h_is_binary_variable;
  cuopt_assert((int32_t)is_binary_variable.size() == n_cols, "is_binary_variable size mismatch");

  // special case equality rows that have a single continuous slack variable
  const uint8_t* ignore_var = c.has_bin_elimination ? c.bin_ignore_var.data() : nullptr;
  const uint8_t* ignore_row = c.has_bin_elimination ? c.bin_ignore_row.data() : nullptr;
  for (int32_t v = 0; v < n_cols; ++v) {
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

  double max_abs_coefficient = 0;
  std::vector<double> row_values;
  for (int32_t r = 0; r < n_rows; ++r) {
    if (ignore_row && ignore_row[r]) continue;
    const bool lb_fin     = std::isfinite(cstr_lb[r]);
    const bool ub_fin     = std::isfinite(cstr_ub[r]);
    const double sides[2] = {cstr_lb[r], cstr_ub[r]};
    const bool finite[2]  = {lb_fin, ub_fin};

    bool integral = true;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      if (!is_integer(coeffs[k], tol)) {
        integral = false;
        break;
      }
    }

    // if some rows arent integral - see if they can be scaled to integral coefficients
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
      if (out.row_scale.empty()) out.row_scale.assign(n_rows, 1.0);
      out.row_scale[r] = row_s;
    }

    // compute min/max activities
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

    if (!is_exactly_representable<int32_t>(row_lhs_min) ||
        !is_exactly_representable<int32_t>(row_lhs_max)) {
      out.reject  = fj_binary_reject_t::lhs_headroom;
      out.bad_row = r;
      return out;
    }

    // test each side of the row
    for (int s = 0; s < 2; ++s) {
      if (!finite[s]) continue;
      const double scaled_side   = row_s * sides[s];
      const double integral_side = is_integer(scaled_side, tol)
                                     ? std::round(scaled_side)
                                     : (s == 0 ? std::ceil(scaled_side) : std::floor(scaled_side));
      if (!is_exactly_representable<int32_t>(integral_side)) {
        out.reject  = fj_binary_reject_t::row_bound_out_of_range;
        out.bad_row = r;
        return out;
      }
      const double min_slack = s == 0 ? row_lhs_min - integral_side : integral_side - row_lhs_max;
      const double max_slack = s == 0 ? row_lhs_max - integral_side : integral_side - row_lhs_min;
      if (!is_exactly_representable<int32_t>(min_slack) ||
          !is_exactly_representable<int32_t>(max_slack)) {
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

  if (max_abs_coefficient <= INT8_MAX) {
    out.coefficient_bits = 8;
  } else if (max_abs_coefficient <= INT16_MAX) {
    out.coefficient_bits = 16;
  } else {
    out.reject = fj_binary_reject_t::coefficient_out_of_range;
  }
  return out;
}

// builds the problem in one-sided form with coef_t coefficients
template <typename i_t, typename f_t, typename coef_t>
void fj_bin_narrow(const fj_cpu_climber_t<i_t, f_t>& c,
                   const fj_bin_scan_t& scan,
                   fj_bin_problem_t<coef_t>& pb,
                   fj_bin_setup_times_t& times)
{
  const int32_t n_split = scan.n_split_constraints;
  const int32_t n_cols  = c.problem->n_variables;
  const int32_t n_rows  = c.problem->n_constraints;
  const double tol      = c.problem->tolerances.integrality_tolerance;

  const auto& offsets   = c.problem->offsets;
  const auto& variables = c.problem->variables;
  const auto& coeffs    = c.problem->coefficients;
  const auto& cstr_lb   = c.problem->cstr_lb;
  const auto& cstr_ub   = c.problem->cstr_ub;
  const auto& left_w    = c.h_cstr_left_weights;
  const auto& right_w   = c.h_cstr_right_weights;
  const auto& obj       = c.problem->h_obj_coeffs;

  // Explicit stamps rather than scoped timers, so narrow and transpose can be timed separately.
  const double narrow_started = tic();

  const uint8_t* ignore_var = c.has_bin_elimination ? c.bin_ignore_var.data() : nullptr;

  pb.n_original = n_cols;
  pb.var_offset.assign(n_cols, 0.0);
  pb.orig_objective.assign(n_cols, 0.0);
  pb.bit_owner.clear();
  pb.bit_owner.reserve(n_cols);
  pb.original_to_bin_mapping.assign(n_cols, -1);
  for (int32_t v = 0; v < n_cols; ++v) {
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

  // turn each two-sided row into a one-sided rows with narrowed integer coefficients
  auto emit = [&](int32_t r, double side_bound, long side, double weight) {
    const double s  = scan.row_scale.empty() ? 1.0 : scan.row_scale[r];
    coef_t row_cmax = 1;
    long fixed_lhs  = 0;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      const double a = s * coeffs[k];
      const long ai  = side * std::lround(a);
      cuopt_assert(is_integer(a, tol), "row scaling left a fractional coefficient");
      cuopt_assert(ai >= std::numeric_limits<coef_t>::min() &&
                     ai <= std::numeric_limits<coef_t>::max(),
                   "scaled coefficient exceeds selected width");
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
    cuopt_assert(is_exactly_representable<int32_t>((double)b),
                 "narrowed row bound is not an int32");
    pb.offsets.push_back((int32_t)pb.variables.size());
    pb.bound.push_back((int32_t)b);
    pb.cmax.push_back(row_cmax);
    incoming_weight.push_back(weight);
  };

  // iterate over all rows and convert to one-sided form
  const uint8_t* ignored_row = c.has_bin_elimination ? c.bin_ignore_row.data() : nullptr;
  for (int32_t r = 0; r < n_rows; ++r) {
    if (ignored_row && ignored_row[r]) continue;
    const double lb = cstr_lb[r];
    const double ub = cstr_ub[r];
    if (std::isfinite(lb)) emit(r, lb, -1, left_w[r]);
    if (std::isfinite(ub)) emit(r, ub, 1, right_w[r]);
  }
  cuopt_assert((int32_t)pb.bound.size() == n_split, "one-sided row count mismatch");
  pb.nnz = (int32_t)pb.variables.size();

  // a few sanity checks
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

  // usual padding to ensure SIMD loads/stores don't cause page faults
  pb.variables.resize(pb.nnz + fj_bin_simd_padding, 0);
  pb.coefficients.resize(pb.nnz + fj_bin_simd_padding, (coef_t)0);

  // Scale the incoming weights into the DDFW band by one global factor, so relative structure
  // survives while every row clears the donation floor.
  initialize_row_weights(incoming_weight, pb.initial_weight);
  times.narrow += toc(narrow_started);

  // compute the transpose now
  const double transpose_started = tic();
  build_transpose(pb);

  pb.objective.resize(n_engine);
  for (int32_t j = 0; j < n_engine; ++j) {
    pb.objective[j] = obj[pb.bit_owner[j]];
    if (pb.objective[j] != 0.0) pb.objective_vars.push_back(j);
  }
  times.transpose += toc(transpose_started);
}

constexpr int32_t fj_bin_encode_max_bits = 16;
constexpr int64_t fj_bin_encode_max_growth = 6;

// Encodes an all-integer model with bounded general integers into bits: x in [L,U] becomes
// x = L + sum_k w_k b_k over weights 1, 2, ..., 2^(nbits-2), R, with R closing the range at W =
// U-L.
// useful for mostly-binayr models with a few small-domain integers
template <typename i_t, typename f_t, typename coef_t>
bool fj_bin_encode(const fj_cpu_climber_t<i_t, f_t>& c,
                   fj_bin_problem_t<coef_t>& pb,
                   int& coefficient_bits,
                   fj_bin_setup_times_t& times)
{
  phase_timer_t timer(times.encode);
  const int32_t n_cols = c.problem->n_variables;
  const int32_t n_rows = c.problem->n_constraints;
  if (n_cols <= 0 || n_rows <= 0) return false;

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

  std::vector<double> lower(n_cols);
  std::vector<double> upper(n_cols);
  std::vector<int32_t> nbits(n_cols);
  std::vector<int32_t> bit_start(n_cols);
  int64_t total_bits = 0;
  // count the total bits that'd be required to encode this model as pure-binary
  for (int32_t v = 0; v < n_cols; ++v) {
    if (var_types[v] != var_t::INTEGER) return false;
    auto bounds    = var_bounds[v];
    const double x = (double)cuopt::get_lower(bounds);
    const double y = (double)cuopt::get_upper(bounds);
    if (!std::isfinite(x) || !std::isfinite(y) || y < x) return false;
    if (!is_integer(x, tol) || !is_integer(y, tol)) return false;

    lower[v]        = std::round(x);
    upper[v]        = std::round(y);
    const int64_t W = (int64_t)(upper[v] - lower[v]);

    nbits[v] = std::bit_width((uint64_t)W);
    if (nbits[v] > fj_bin_encode_max_bits) return false;
    bit_start[v] = (int32_t)total_bits;
    total_bits += nbits[v];
  }
  if (total_bits <= 0 || total_bits > (int64_t)INT32_MAX / 2) return false;
  if (total_bits > fj_bin_encode_max_growth * (int64_t)n_cols) return false;

  const int32_t n_bits = (int32_t)total_bits;

  pb.encoded    = true;
  pb.n_original = n_cols;
  pb.var_offset = lower;
  pb.orig_objective.assign(n_cols, 0.0);
  pb.bit_owner.assign(n_bits, 0);
  pb.original_to_bin_mapping.assign(n_cols, -1);
  pb.bit_weight.assign(n_bits, 0.0);
  for (int32_t v = 0; v < n_cols; ++v) {
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

  // emit onesided a row
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
    if (!is_exactly_representable<int32_t>(bound)) return false;
    // A bit assignment can drive lhs anywhere in [-row_abs_sum, row_abs_sum].
    if (!is_exactly_representable<int32_t>(bound - row_abs_sum) ||
        !is_exactly_representable<int32_t>(bound + row_abs_sum))
      return false;

    pb.offsets.push_back((int32_t)pb.variables.size());
    pb.bound.push_back((int32_t)bound);
    pb.cmax.push_back(row_cmax);
    incoming_weight.push_back(weight);
    return true;
  };

  const uint8_t* ignored_row = c.has_bin_elimination ? c.bin_ignore_row.data() : nullptr;
  for (int32_t r = 0; r < n_rows; ++r) {
    if (ignored_row && ignored_row[r]) continue;
    const double lb = cstr_lb[r];
    const double ub = cstr_ub[r];
    if (std::isfinite(lb) && !emit(r, lb, -1, left_w[r])) return false;
    if (std::isfinite(ub) && !emit(r, ub, 1, right_w[r])) return false;
  }
  pb.n_constraints = (int32_t)pb.bound.size();
  if (pb.n_constraints <= 0) return false;
  pb.nnz = (int32_t)pb.variables.size();

  if (max_abs_coefficient <= INT8_MAX) {
    coefficient_bits = 8;
  } else if (max_abs_coefficient <= INT16_MAX) {
    coefficient_bits = 16;
  } else {
    return false;
  }

  pb.variables.resize(pb.nnz + fj_bin_simd_padding, 0);
  pb.coefficients.resize(pb.nnz + fj_bin_simd_padding, (coef_t)0);

  initialize_row_weights(incoming_weight, pb.initial_weight);
  build_transpose(pb);

  pb.objective.assign(n_bits, 0.0);
  pb.objective_vars.clear();
  for (int32_t v = 0; v < n_cols; ++v) {
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
template void fj_bin_narrow(
  const fj_cpu_climber_t<int, float>&, const fj_bin_scan_t&, fj_bin_problem_t<int8_t>&, fj_bin_setup_times_t&);
template void fj_bin_narrow(const fj_cpu_climber_t<int, float>&,
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
template void fj_bin_narrow(
  const fj_cpu_climber_t<int, double>&, const fj_bin_scan_t&, fj_bin_problem_t<int8_t>&, fj_bin_setup_times_t&);
template void fj_bin_narrow(const fj_cpu_climber_t<int, double>&,
                            const fj_bin_scan_t&,
                            fj_bin_problem_t<int16_t>&,
                            fj_bin_setup_times_t&);
template bool fj_bin_encode(
  const fj_cpu_climber_t<int, double>&, fj_bin_problem_t<int8_t>&, int&, fj_bin_setup_times_t&);
template bool fj_bin_encode(
  const fj_cpu_climber_t<int, double>&, fj_bin_problem_t<int16_t>&, int&, fj_bin_setup_times_t&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
