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
#include <utilities/copy_helpers.hpp>
#include <utilities/integer_scaling.hpp>

#include <raft/random/rng_device.cuh>

#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

const char* fj_binary_reject_name(fj_binary_reject_t reason)
{
  switch (reason) {
    case fj_binary_reject_t::none: return "none";
    case fj_binary_reject_t::empty_problem: return "empty problem";
    case fj_binary_reject_t::non_binary_var: return "non-binary variable";
    case fj_binary_reject_t::fractional_coefficient: return "fractional coefficient";
    case fj_binary_reject_t::coefficient_out_of_range: return "coefficient wider than int16";
    case fj_binary_reject_t::fractional_row_bound: return "fractional row bound";
    case fj_binary_reject_t::row_bound_out_of_range: return "row bound outside int32";
    case fj_binary_reject_t::lhs_headroom: return "row sum|coef| exceeds int32 headroom";
    case fj_binary_reject_t::narrow_check_failed: return "narrowing check failed";
  }
  return "unknown";
}

// work unit proxy. will likely require a lot of tuning
constexpr double fj_bin_bytes_per_nnz = 16.0;

// Tabu for binary variables, expressed as a ring buffer
// There can be at most max_tenure tabu'd variables at any given time.
// since max_tenure << n_vars, it's cheaper to maintain a ring buffer than a full array
// and it allows smaller instances to become L1 resident
struct fj_bin_tabu_t {
  static constexpr int32_t ring_size  = 16;
  static constexpr int32_t max_tenure = ring_size;
  // Headroom so iter + tenure - iter_bias still fits uint16 when iter - iter_bias is at the rebase
  // threshold.
  static constexpr int32_t window =
    (int32_t)std::numeric_limits<uint16_t>::max() - max_tenure;

  std::vector<uint16_t> flip_until;
  std::vector<int32_t> last_flip;
  int32_t iter_bias{0};

  int32_t ring_var[ring_size];
  int32_t ring_expiry[ring_size];


  void resize(int32_t n)
  {
    flip_until.assign(n, 0);
    last_flip.assign(n, 0);
    clear_ring();
    iter_bias = 0;
  }

  void clear(int32_t iter)
  {
    std::fill(flip_until.begin(), flip_until.end(), (uint16_t)0);
    std::fill(last_flip.begin(), last_flip.end(), 0);
    clear_ring();
    iter_bias = iter;
  }

  void clear_ring()
  {
    for (int32_t i = 0; i < ring_size; ++i) {
      ring_var[i]    = -1;
      ring_expiry[i] = 0;
    }
  }

  void on_flip(int32_t v, int32_t iter, int32_t tenure)
  {
    flip_until[v] = (uint16_t)(iter + tenure - iter_bias);
    last_flip[v]  = iter;

    // keep only one tabu entry per var
    for (int32_t i = 0; i < ring_size; ++i) {
      if (ring_var[i] == v) ring_var[i] = -1;
    }

    const int32_t slot = iter & (ring_size - 1);
    ring_var[slot]     = v;
    ring_expiry[slot]  = iter + tenure;
  }

  // replace the scores of tabu'd variable with sentinel values
  int32_t block_tabu(int32_t iter,
                     int64_t* var_score,
                     int32_t (&saved_var)[ring_size],
                     int64_t (&saved_score)[ring_size]) const
  {
    int32_t k = 0;
    for (int32_t i = 0; i < ring_size; ++i) {
      const int32_t v = ring_var[i];
      if (v >= 0 && ring_expiry[i] > iter) {
        saved_var[k]   = v;
        saved_score[k] = var_score[v];
        var_score[v]   = fj_bin_score_invalid;
        ++k;
      }
    }
    return k;
  }

  // reverse the above operation.
  static void unblock_tabu(int32_t k,
                           int64_t* var_score,
                           const int32_t (&saved_var)[ring_size],
                           const int64_t (&saved_score)[ring_size])
  {
    for (int32_t i = k - 1; i >= 0; --i) var_score[saved_var[i]] = saved_score[i];
  }


  bool blocked(int32_t v, int32_t iter, bool localmin) const
  {
    return localmin ? (iter == last_flip[v] + 1)
                    : ((uint16_t)(iter - iter_bias) < flip_until[v]);
  }

  // rebase the iteration bias value every 64k iter
  void maybe_rebase(int32_t iter)
  {
    if ((int64_t)iter - iter_bias <= window) return;
    const uint16_t shift = (uint16_t)(iter - iter_bias);
    for (uint16_t& fu : flip_until) fu = (fu > shift) ? (uint16_t)(fu - shift) : (uint16_t)0;
    iter_bias = iter;
  }
};

// Narrowed problem: one-sided rows, integer coefficients, CSR plus its transpose.
template <typename coef_t>
struct fj_bin_problem_t {
  int32_t n_variables{0};
  int32_t n_constraints{0};
  int32_t nnz{0};

  std::vector<int32_t> offsets;
  std::vector<int32_t> variables;
  std::vector<coef_t> coefficients;

  std::vector<int32_t> reverse_offsets;
  std::vector<int32_t> reverse_constraints;
  std::vector<int32_t> reverse_to_csr;

  // Per incidence, for the vectorized row walk: the coefficient and the row's cmax, both replicated
  // in transpose order so the walk reads them at unit stride instead of gathering per row. Both are
  // structural.
  std::vector<coef_t> reverse_coefficients;
  std::vector<coef_t> incident_row_cmax;

  std::vector<int32_t> bound;
  std::vector<coef_t> cmax;
  std::vector<int32_t> initial_weight;

  std::vector<double> objective;
  std::vector<int32_t> objective_vars;
};

// Result of the width-independent eligibility scan.
struct fj_bin_scan_t {
  fj_binary_reject_t reject{fj_binary_reject_t::none};
  int coefficient_bits{0};
  int32_t n_split_constraints{0};
  int32_t bad_row{-1};
  int32_t bad_var{-1};
  std::vector<double> row_scale;
};

constexpr int64_t fj_bin_scale_cap = std::numeric_limits<int16_t>::max();

// DDFW and restart have no general-path equivalent, so their defaults live here until there is a
// reason to promote them alongside the other FJ knobs.
constexpr int32_t fj_bin_ddfw_init          = 10;  // initial weight, also the donation floor
constexpr int32_t fj_bin_ddfw_transfer      = 1;
constexpr int32_t fj_bin_ddfw_donor_samples = 4;
constexpr int32_t fj_bin_restart_period     = 5000000;

// Escalation threshold and step, in infeasible local minima without a severity improvement.
constexpr int32_t fj_bin_ddfw_escalate_after = 2000;
constexpr int32_t fj_bin_ddfw_escalate_max   = 100;

// The same, in feasible local minima without a best-objective improvement.
constexpr int32_t fj_bin_obj_stall_after   = 50;
constexpr int32_t fj_bin_obj_escalate_max  = 10;

// Infeasible-region kick: stall, cooldown, post-restart quiet window, rows drawn, flips per row.
constexpr int32_t fj_bin_kick_after         = 200;
constexpr int32_t fj_bin_kick_cooldown      = 200;
constexpr int32_t fj_bin_kick_restart_guard = 50;
constexpr int32_t fj_bin_kick_rows          = 3;
constexpr int32_t fj_bin_kick_vars_per_row  = 2;

// Candidate draws per 2-opt lift search.
constexpr int32_t fj_bin_2opt_candidates = 64;
// prefetch distance
// TODO: check if it actually matters at all for performance
constexpr int32_t fj_bin_pf_dist = 8;

constexpr int32_t fj_bin_base_limit  = 1 << 16;
constexpr int32_t fj_bin_bonus_limit = 1 << 14;

static inline bool fj_bin_in_int32(double v)
{
  return v >= (double)INT32_MIN && v <= (double)INT32_MAX;
}

// Tile width of the argmax sweep, in variables: min(algorithm target, L1-residency cap).
//
// The target is about the shape of the sweep rather than cache capacity -- it sets how often the
// running maximum is raised, which is what bounds the index re-scan -- and 256 is the measured
// optimum. The cap is a residency guard, and it is the reason this is not simply a constant: the
// re-scan pays off only because it revisits a tile that is still L1-hot, so the tile must not be
// wide enough to spill. It bites only on a small L1, where an unguarded 256 would push the re-scan
// out to L2 and cost more than the split saves.
//
// Bytes per variable is the score array alone. Tabu does not appear: the sweep reads var_score
// only, with the handful of tabu variables held at the invalid sentinel across it, so flip_until is
// never touched here.
constexpr int32_t fj_bin_argmax_tile_target = 256;
constexpr int32_t fj_bin_argmax_tile_cap_k  = 4;

static int32_t fj_bin_argmax_tile()
{
#ifdef _SC_LEVEL1_DCACHE_SIZE
  long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
#else
  long l1 = 0;
#endif
  if (l1 <= 0) l1 = 32768;  // fallback: 32 KiB, the common x86 L1d
  const int32_t bpv = (int32_t)sizeof(int32_t);
  const int32_t cap = (int32_t)(l1 / (fj_bin_argmax_tile_cap_k * bpv));
  int32_t t = fj_bin_argmax_tile_target < cap ? fj_bin_argmax_tile_target : cap;
  t &= ~15;  // whole vectors
  return t < 16 ? 16 : t;
}

// Width-independent eligibility scan over the climber's host mirrors. Mutates nothing.
template <typename i_t, typename f_t>
static fj_bin_scan_t fj_bin_scan(const fj_cpu_climber_t<i_t, f_t>& c)
{
  fj_bin_scan_t out;
  const int32_t n = c.view.pb.n_variables;
  const int32_t m = c.view.pb.n_constraints;
  if (n <= 0 || m <= 0) {
    out.reject = fj_binary_reject_t::empty_problem;
    return out;
  }

  const double tol             = c.view.pb.tolerances.integrality_tolerance;
  const auto& is_binary_variable = c.h_is_binary_variable;
  cuopt_assert((int32_t)is_binary_variable.size() == n, "is_binary_variable size mismatch");

  for (int32_t v = 0; v < n; ++v) {
    // Populated at climber init with integer_equal on [0,1] bounds.
    if (!is_binary_variable[v]) {
      out.reject  = fj_binary_reject_t::non_binary_var;
      out.bad_var = v;
      return out;
    }
  }

  const auto& offsets            = c.h_offsets;
  const auto& reverse_offsets    = c.h_reverse_offsets;
  const auto& reverse_constraints = c.h_reverse_constraints;
  const auto& coeffs             = c.h_coefficients;
  const auto& cstr_lb            = c.h_cstr_lb;
  const auto& cstr_ub            = c.h_cstr_ub;

  cuopt_assert(
    thrust::all_of(
      thrust::host,
      thrust::make_counting_iterator<int32_t>(0),
      thrust::make_counting_iterator<int32_t>(n),
      [&reverse_offsets, &reverse_constraints](int32_t v) {
        const auto first = reverse_constraints.begin() + reverse_offsets[v];
        const auto last  = reverse_constraints.begin() + reverse_offsets[v + 1];
        return thrust::adjacent_find(thrust::host, first, last) == last;
      }),
    "duplicate variable in CSR row");

  double max_abs_coefficient = 0;
  std::vector<double> row_values;
  for (int32_t r = 0; r < m; ++r) {
    const double lb    = cstr_lb[r];
    const double ub    = cstr_ub[r];
    const bool lb_fin  = std::isfinite(lb);
    const bool ub_fin  = std::isfinite(ub);
    const double sides[2] = {lb, ub};
    const bool finite[2]  = {lb_fin, ub_fin};

    bool fractional_coefficient_seen = false;
    bool integral                    = true;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      if (!is_integer(coeffs[k], tol)) {
        fractional_coefficient_seen = true;
        integral                    = false;
        break;
      }
    }
    for (int s = 0; s < 2 && integral; ++s) {
      if (finite[s] && !is_integer(sides[s], tol)) integral = false;
    }

    double row_s = 1.0;
    if (!integral) {
      row_values.clear();
      for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) row_values.push_back(coeffs[k]);
      for (int s = 0; s < 2; ++s) {
        if (finite[s]) row_values.push_back(sides[s]);
      }
      row_s = find_scaling_rational(row_values,
                                    /*maxscale=*/1.0 / tol,
                                    /*maxdnom=*/fj_bin_scale_cap,
                                    /*maxfinal=*/(double)fj_bin_scale_cap,
                                    /*intcheck_tol=*/tol);
      if (!std::isfinite(row_s) || row_s <= 0.0) {
        out.reject  = fractional_coefficient_seen ? fj_binary_reject_t::fractional_coefficient
                                                  : fj_binary_reject_t::fractional_row_bound;
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
      const double scaled_side = row_s * sides[s];
      cuopt_assert(is_integer(scaled_side, tol), "row scaling left a fractional row bound");
      if (!fj_bin_in_int32(std::round(scaled_side))) {
        out.reject  = fj_binary_reject_t::row_bound_out_of_range;
        out.bad_row = r;
        return out;
      }
      const double integral_side = std::round(scaled_side);
      const double min_slack =
        s == 0 ? row_lhs_min - integral_side : integral_side - row_lhs_max;
      const double max_slack =
        s == 0 ? row_lhs_max - integral_side : integral_side - row_lhs_min;
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
static bool fj_bin_narrow(const fj_cpu_climber_t<i_t, f_t>& c,
                   const fj_bin_scan_t& scan,
                   fj_bin_problem_t<coef_t>& pb)
{
  const int32_t n_split = scan.n_split_constraints;
  const int32_t n = c.view.pb.n_variables;
  const int32_t m = c.view.pb.n_constraints;
  const double tol = c.view.pb.tolerances.integrality_tolerance;

  const auto& offsets   = c.h_offsets;
  const auto& variables = c.h_variables;
  const auto& coeffs    = c.h_coefficients;
  const auto& cstr_lb   = c.h_cstr_lb;
  const auto& cstr_ub   = c.h_cstr_ub;
  const auto& left_w    = c.h_cstr_left_weights;
  const auto& right_w   = c.h_cstr_right_weights;
  const auto& obj       = c.h_obj_coeffs;

  pb.n_variables   = n;
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
  // bound - lhs everywhere -- so no per-row sign reaches the engine at all. Negation is safe on both
  // fields: the scan admits |coef| up to 127 for int8 and 32767 for int16, and the bound is checked
  // for int32 range after negating.
  auto emit = [&](int32_t r, double side_bound, long side, double weight) -> bool {
    const double s  = scan.row_scale.empty() ? 1.0 : scan.row_scale[r];
    coef_t row_cmax = 1;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      const double a = s * coeffs[k];
      const long ai  = side * std::lround(a);
      if (!is_integer(a, tol) || ai < std::numeric_limits<coef_t>::min() ||
          ai > std::numeric_limits<coef_t>::max()) {
        return false;
      }
      pb.variables.push_back(variables[k]);
      pb.coefficients.push_back((coef_t)ai);
      const coef_t abs_a = (coef_t)std::labs(ai);
      if (abs_a > row_cmax) row_cmax = abs_a;
    }
    const long b = side * std::lround(s * side_bound);
    if (!fj_bin_in_int32((double)b)) return false;
    pb.offsets.push_back((int32_t)pb.variables.size());
    pb.bound.push_back((int32_t)b);
    pb.cmax.push_back(row_cmax);
    incoming_weight.push_back(weight);
    return true;
  };

  for (int32_t r = 0; r < m; ++r) {
    const double lb = cstr_lb[r];
    const double ub = cstr_ub[r];
    if (std::isfinite(lb) && !emit(r, lb, -1, left_w[r])) return false;
    if (std::isfinite(ub) && !emit(r, ub, 1, right_w[r])) return false;
  }
  if ((int32_t)pb.bound.size() != n_split) return false;
  pb.nnz = (int32_t)pb.variables.size();

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

  // Transpose, plus the reverse-nnz to CSR-nnz map the apply path uses to store the flipped
  // variable's own score delta.
  pb.reverse_offsets.assign(n + 1, 0);
  for (int32_t k = 0; k < pb.nnz; ++k) pb.reverse_offsets[pb.variables[k] + 1]++;
  for (int32_t v = 0; v < n; ++v) pb.reverse_offsets[v + 1] += pb.reverse_offsets[v];
  pb.reverse_constraints.resize(pb.nnz);
  pb.reverse_coefficients.resize(pb.nnz);
  pb.reverse_to_csr.resize(pb.nnz);
  pb.incident_row_cmax.resize(pb.nnz);
  {
    std::vector<int32_t> cursor(pb.reverse_offsets.begin(), pb.reverse_offsets.begin() + n);
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
  // the prefetch distance the scalar path uses. Reads land on row 0, harmlessly, and every lane past
  // a variable's range is masked out of the gather, the scatter and the compress.
  const int32_t rpad =
    fj_bin_pf_dist > fj_bin_simd_padding ? fj_bin_pf_dist : fj_bin_simd_padding;
  pb.reverse_constraints.resize(pb.nnz + rpad, 0);
  pb.reverse_coefficients.resize(pb.nnz + rpad, (coef_t)0);
  pb.incident_row_cmax.resize(pb.nnz + rpad, (coef_t)1);

  pb.objective.resize(n);
  for (int32_t v = 0; v < n; ++v) {
    pb.objective[v] = obj[v];
    if (pb.objective[v] != 0.0) pb.objective_vars.push_back(v);
  }
  return true;
}


// The integer engine. Feasibility is an exact compare against one bound per row, so there is no
// tolerance arithmetic and no compensated summation anywhere below.
template <typename i_t, typename f_t, typename coef_t>
struct fj_bin_engine_t {
  fj_bin_problem_t<coef_t> pb;
  // The only mutable per-row state besides the slack. Everything else the apply path once read
  // per row now reaches it at unit stride: bound stayed in pb, where only the rebuild paths need
  // it, and cmax went to pb.incident_row_cmax, replicated per incidence.
  std::vector<int32_t> row_weight;

  // Per row, bound - lhs: negative exactly when the row is violated, and moved by a flip by exactly
  // -reverse_coefficients. The only mutable state the vectorized walk gathers.
  std::vector<int32_t> row_slack;

  std::vector<int8_t> assign;
  std::vector<int8_t> best_assign;
  std::vector<int8_t> seed_assign;   // restart target
  std::vector<int32_t> assign_i32;   // gather mirror for the SIMD patch (Batch B)

  std::vector<int8_t> best_infeasible_assign;
  int64_t best_infeasible_severity{std::numeric_limits<int64_t>::max()};
  int64_t checkpoint_severity{std::numeric_limits<int64_t>::max()};
  int32_t iters_since_infeasible_improve{0};
  int32_t restores_since_improvement{0};

  std::vector<int64_t> var_score;    // live feasibility score of flipping each variable
  std::vector<int64_t> nnz_score_delta;  // per CSR nnz: last score delta of variables[k] in its row

  // Objective half of the move score, held live so a weighted global scan can stay vectorized.
  // Its support is pb.objective_vars, so entries outside that set are zero for the whole solve.
  std::vector<int64_t> obj_base_score;
  std::vector<int64_t> combined_score;
  // Objective weight obj_base_score was built for; -1 marks it stale.
  int32_t obj_base_weight{-1};

  fj_bin_tabu_t tabu;

  std::vector<uint8_t> is_violated;
  std::vector<int32_t> violated_list;
  std::vector<int32_t> vpos;
  // Duplicate guard for find_move_in_rows, its only reader. Zero everywhere outside that function,
  // which clears what it set before returning.
  std::vector<char> var_bitmap;

  // One generator advanced across the whole search, rather than one re-seeded per call site per
  // iteration. Re-seeding from `seed + iters` gave every call site in an iteration the identical
  // stream, and a 624-word Mersenne state was being built and discarded on every move selection.
  raft::random::PCGenerator rng{0, 0, 0};
  std::vector<int32_t> sample_buf;  // move-selection row sample, reused to keep the loop allocation-free

  int32_t objective_weight{0};
  int32_t seed_objective_weight{0};
  // Feasible local minima since best_objective last moved, and the value it was last seen at.
  int32_t iterations_at_same_objective{0};
  double last_best_objective{std::numeric_limits<double>::infinity()};
  // Mean absolute nonzero objective coefficient; the unit of the objective score term.
  double obj_magnitude{1.0};
  double incumbent_objective{0};
  double best_objective{std::numeric_limits<double>::infinity()};
  int32_t max_weight{1};
  bool feasible_found{false};

  int32_t iters{0};
  int32_t last_feasible_entrance_iter{0};
  int32_t last_restart_iter{0};
  int32_t last_kick_iter{0};
  int64_t nnz_touched{0};

  // Denominator for the ops-per-nnz roofline: nonzeros the row kernel actually processes, and the
  // rows walked to find them. Unlike nnz_touched these are not mixed with the full-matrix rebuilds.
  int64_t nnz_patched{0};
  int64_t rows_walked{0};

  int64_t n_checkpoint_restores{0};
  int64_t n_checkpoint_snapshots{0};
  int32_t max_restores_since_improvement{0};

  // Tile width for the argmax sweep, in variables. Set at init from fj_bin_argmax_tile().
  int32_t argmax_tile{fj_bin_argmax_tile_target};

  // Settings read at solve entry, where the climber carries populated values.
  int32_t seed{0};
  int32_t tabu_tenure_min{3};
  int32_t tabu_tenure_max{13};
  int32_t perturb_interval{100};
  int32_t mtm_viol_samples{25};
  int32_t mtm_sat_samples{15};
  int32_t infeasible_restart_window{300};
  int32_t infeasible_restart_max_streak{20};
  double infeasible_restart_degrade_ratio{1.15};
  double infeasible_checkpoint_refresh_ratio{0.99};
  double breakthrough_margin{1e-4};

  int32_t max_aggregate_base{0};
  int32_t max_aggregate_bonus{0};

  int coefficient_bits() const { return 8 * (int)sizeof(coef_t); }

  // Largest per-variable aggregate base and bonus under the final weights and assignment, in raw
  // int32. The packed representation is only order-preserving while these stay inside their
  // limits, and weights grow without a cap, so this is the reading that says whether the packing
  // survived the run.
  // Independent audit of the incumbent at end of solve. Recomputes every row's lhs and the objective
  // from best_assign alone, trusting nothing the incremental path maintained: not the live lhs, not
  // violated_list, not the running incumbent_objective. Accumulates in int64 so an int32 lhs
  // overflow the eligibility scan was supposed to preclude would show up here rather than wrap
  // silently. Runs once per solve, so its cost is not on any hot path.
  void verify_incumbent(fj_cpu_climber_t<i_t, f_t>& climber) const
  {
    if (!feasible_found) return;

    int32_t n_violated = 0;
    int64_t worst      = 0;
    bool lhs_overflow  = false;
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      int64_t lhs = 0;
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        lhs += (int64_t)pb.coefficients[k] * (int64_t)best_assign[pb.variables[k]];
      }
      if (lhs < INT32_MIN || lhs > INT32_MAX) lhs_overflow = true;
      const int64_t slack = (int64_t)pb.bound[r] - lhs;
      if (slack < 0) {
        ++n_violated;
        if (-slack > worst) worst = -slack;
      }
    }

    double objective = 0;
    for (int32_t v = 0; v < pb.n_variables; ++v) objective += pb.objective[v] * (double)best_assign[v];
    const double drift = std::fabs(objective - best_objective);

    if (n_violated != 0 || lhs_overflow || drift > 1e-6) {
      CUOPT_LOG_ERROR(
        "%sCPUFJ[bin%d] incumbent audit FAILED: %d violated rows (worst %lld), lhs overflow %d, "
        "objective recomputed %.17g vs tracked %.17g (drift %g)",
        climber.log_prefix.c_str(),
        coefficient_bits(),
        n_violated,
        (long long)worst,
        (int)lhs_overflow,
        objective,
        best_objective,
        drift);
    } else {
      CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] incumbent audit ok: feasible, objective %.17g (drift %g)",
                      climber.log_prefix.c_str(),
                      coefficient_bits(),
                      objective,
                      drift);
    }
  }

  void compute_saturation()
  {
    int32_t peak_base = 0, peak_bonus = 0;
    for (int32_t v = 0; v < pb.n_variables; ++v) {
      const int8_t flip = (int8_t)(1 - 2 * assign[v]);
      int32_t agg_base = 0, agg_bonus = 0;
      for (int32_t i = pb.reverse_offsets[v]; i < pb.reverse_offsets[v + 1]; ++i) {
        const int32_t r  = pb.reverse_constraints[i];
        const int32_t os = row_slack[r];
        const int32_t ns = os - (int32_t)pb.reverse_coefficients[i] * flip;
        int32_t base = 0, bonus = 0;
        fj_bin_score_delta_parts(os, ns, row_weight[r], base, bonus);
        agg_base += base;
        agg_bonus += bonus;
      }
      const int32_t abs_base  = agg_base < 0 ? -agg_base : agg_base;
      const int32_t abs_bonus = agg_bonus < 0 ? -agg_bonus : agg_bonus;
      if (abs_base > peak_base) peak_base = abs_base;
      if (abs_bonus > peak_bonus) peak_bonus = abs_bonus;
    }
    max_aggregate_base  = peak_base;
    max_aggregate_bonus = peak_bonus;
  }

  void set_violated(int32_t r)
  {
    if (!is_violated[r]) {
      is_violated[r] = 1;
      vpos[r]        = (int32_t)violated_list.size();
      violated_list.push_back(r);
    }
  }

  void set_satisfied(int32_t r)
  {
    if (is_violated[r]) {
      is_violated[r]        = 0;
      const int32_t p       = vpos[r];
      const int32_t last    = violated_list.back();
      violated_list[p]      = last;
      vpos[last]            = p;
      violated_list.pop_back();
      vpos[r] = -1;
    }
  }

  void rebuild_scores()
  {
    std::fill(var_score.begin(), var_score.end(), 0);
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      const int32_t weight = row_weight[r];
      const int32_t os     = row_slack[r];
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t v    = pb.variables[k];
        const int32_t flip = 1 - 2 * assign[v];
        const int32_t ns   = os - (int32_t)pb.coefficients[k] * flip;
        const int64_t p    = fj_bin_packed_score_delta(os, ns, weight);
        nnz_score_delta[k] = p;
        var_score[v] += p;
      }
    }
    nnz_touched += pb.nnz;
  }

  void recompute_slack()
  {
    violated_list.clear();
    std::fill(is_violated.begin(), is_violated.end(), (uint8_t)0);
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      int32_t lhs = 0;
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k)
        lhs += (int32_t)pb.coefficients[k] * assign[pb.variables[k]];
      const int32_t slack = pb.bound[r] - lhs;
      row_slack[r]        = slack;
      if (slack < 0) set_violated(r);
    }
    incumbent_objective = 0;
    for (int32_t v = 0; v < pb.n_variables; ++v) incumbent_objective += pb.objective[v] * assign[v];
    nnz_touched += pb.nnz;
    rebuild_scores();
    // Every caller of this reached it by replacing the assignment wholesale, so the cached
    // per-variable flip directions no longer describe it.
    obj_base_weight = -1;
  }

  // Base field of the objective term: the weight, signed by the direction of the gain and scaled by
  // how large that gain is against the model's typical coefficient. Depends only on the variable's
  // own value and the weight, which is what lets a global scan cache it.
  int64_t objective_base(int32_t v, int8_t delta) const
  {
    const double obj_diff = pb.objective[v] * delta;
    if (obj_diff == 0) return 0;
    cuopt_assert(obj_magnitude > 0, "objective magnitude unit must be positive");
    const double rel  = std::fabs(obj_diff) / obj_magnitude;
    const double mult =
      rel < fj_obj_mult_min ? fj_obj_mult_min : (rel > fj_obj_mult_max ? fj_obj_mult_max : rel);
    const double raw = objective_weight * mult;
    cuopt_assert(fj_bin_in_int32(raw), "scaled objective weight out of int32 range");
    const int32_t scaled = (int32_t)std::lround(raw);
    return (int64_t)(obj_diff < 0 ? scaled : -scaled) * fj_bin_score_k;
  }

  int64_t objective_terms(int32_t v, int8_t delta) const
  {
    const double obj_diff = pb.objective[v] * delta;
    int32_t bonus         = 0;
    const bool old_better = incumbent_objective < best_objective;
    const bool new_better = incumbent_objective + obj_diff < best_objective;
    if (!old_better && new_better) {
      bonus += objective_weight;
    } else if (old_better && !new_better) {
      bonus -= objective_weight;
    }
    return objective_base(v, delta) + bonus;
  }

  int64_t flip_objective_base(int32_t v) const
  {
    return objective_base(v, (int8_t)(1 - 2 * assign[v]));
  }

  // Only the objective variables are written: the rest of the array is zero from init onwards.
  void ensure_objective_base()
  {
    if (obj_base_weight == objective_weight) return;
    for (int32_t v : pb.objective_vars) obj_base_score[v] = flip_objective_base(v);
    obj_base_weight = objective_weight;
  }

  int64_t full_score(int32_t v, int8_t delta) const
  {
    if (objective_weight == 0) return var_score[v];
    return var_score[v] + objective_terms(v, delta);
  }

  bool tabu_blocked(int32_t v, bool localmin) const { return tabu.blocked(v, iters, localmin); }

  void apply_move(int32_t var, int8_t delta, fj_cpu_climber_t<i_t, f_t>& climber)
  {
    const int8_t new_val  = (int8_t)(assign[var] + delta);
    const int8_t new_flip = (int8_t)(1 - 2 * new_val);
    const int32_t ob = pb.reverse_offsets[var], oe = pb.reverse_offsets[var + 1];
    const int32_t prev_violated = (int32_t)violated_list.size();
    int64_t own_score           = 0;

    // The tail writes a score delta through int32_t* and calls out to the patch, either of which may
    // alias a vector's internal pointer as far as the compiler can prove. Without these locals it
    // reloads every base pointer below out of `this` on each visit.
    int32_t* const weight_p            = row_weight.data();
    int32_t* const slack_p             = row_slack.data();
    const int32_t* const rcon_p        = pb.reverse_constraints.data();
    const coef_t* const skv_p          = pb.reverse_coefficients.data();
    const coef_t* const rcmax_p        = pb.incident_row_cmax.data();
    const int32_t* const rcsr_p        = pb.reverse_to_csr.data();
    const int32_t* const offsets_p     = pb.offsets.data();
    const int32_t* const vars_p        = pb.variables.data();
    const coef_t* const coefs_p        = pb.coefficients.data();
    int64_t* const var_score_p         = var_score.data();
    int64_t* const nnz_delta_p         = nnz_score_delta.data();
    const int32_t* const assign_p      = assign_i32.data();

    // Everything a visit still needs once its slack has been advanced. Shared by the two arms below
    // so the walk's shape is the only thing that differs between them.
    auto finish = [&](int32_t ii) {
      const int32_t r         = rcon_p[ii];
      const int32_t weight    = weight_p[r];
      const int32_t skv       = (int32_t)skv_p[ii];
      const int32_t new_slack = slack_p[r];
      const int32_t old_slack = new_slack + skv * delta;

      // A row can only cross its boundary if the flip moves it by at least the distance to it, so
      // every transition is inside this list and none was lost with the rows the walk absorbed.
      if (new_slack < 0 && old_slack >= 0) {
        set_violated(r);
      } else if (new_slack >= 0 && old_slack < 0) {
        set_satisfied(r);
      }

      // The mirror of the walk's deep_sat test. Kept here rather than there because it fires on
      // 0.02% of visits and guards the widest rows in the matrix: measured, moving it into the
      // vector loop costs more in the 85% case than it saves in the 0.02% one.
      const int32_t margin = (int32_t)rcmax_p[ii];
      if (!(old_slack < -margin && new_slack < -margin)) {
          const int32_t kb = offsets_p[r], ke = offsets_p[r + 1];
          // TODO: check that this may not cause AVX512 powerdown overheads if the AVX2 row/AVX512 row ratio is unbalanced
          fj_bin_patch_row(vars_p,
                           coefs_p,
                           kb,
                           ke,
                           var_score_p,
                           nnz_delta_p,
                           assign_p,
                           weight,
                           new_slack,
                           var);
        nnz_touched += ke - kb;
        nnz_patched += ke - kb;
      }

      // The flipped variable's own score delta. Zero on the rows the walk absorbed -- deeply
      // satisfied both ways -- and already stored as zero there.
      const int64_t pv = fj_bin_packed_score_delta(new_slack, new_slack - skv * new_flip, weight);
      own_score += pv;
      nnz_delta_p[rcsr_p[ii]] = pv;
    };

    // A tile at a time: the kernel advances every slack in the tile and reports back only the visits
    // that left the row within reach of its boundary, which on supportcase22 is 15.1% of them. The
    // buffer is a stack array rather than one sized to the widest reverse degree because the tail
    // runs between tiles, which is also what keeps the patch calls out of the vector loop.
    //
    // Unconditional: a scalar arm for short ranges was tried and never won. Sweeping the degree
    // below which apply_move walked the rows itself, bnatt400 degraded monotonically from 14.43M to
    // 14.19M iterations/s as the threshold rose from 0 to 64, and crypt16 and supportcase22 were
    // flat. At a median degree of 13 and 7 respectively, one gather still beats that many dependent
    // scalar load-modify-stores, because it breaks the dependence chain through row_slack rather
    // than following it.
    int32_t tile_incidence[fj_bin_walk_tile];
    for (int32_t t0 = ob; t0 < oe; t0 += fj_bin_walk_tile) {
      const int32_t t1 = (t0 + fj_bin_walk_tile < oe) ? t0 + fj_bin_walk_tile : oe;
      const int32_t n_tail =
        fj_bin_walk_rows(slack_p, rcon_p, skv_p, rcmax_p, t0, t1, delta, tile_incidence);
      for (int32_t j = 0; j < n_tail; ++j) finish(tile_incidence[j]);
    }
    nnz_touched += oe - ob;
    rows_walked += oe - ob;

    if (prev_violated > 0 && violated_list.empty()) last_feasible_entrance_iter = iters;

    assign[var]     = new_val;
    assign_i32[var] = new_val;
    var_score[var]  = own_score;
    incumbent_objective += pb.objective[var] * delta;
    // Only this variable's flip direction moved, so a live cache needs one entry rewritten.
    if (obj_base_weight == objective_weight && pb.objective[var] != 0)
      obj_base_score[var] = flip_objective_base(var);

    if (violated_list.empty() && incumbent_objective < best_objective) {
      best_objective = incumbent_objective;
      best_assign    = assign;
      feasible_found = true;
      report_incumbent(climber);
    }

    const int32_t tenure =
      tabu_tenure_min + (int32_t)(rng.next_u32() % (uint32_t)(tabu_tenure_max - tabu_tenure_min));
    tabu.on_flip(var, iters, tenure);
  }

  // Publish a new best into the climber, which owns the reporting contract.
  void report_incumbent(fj_cpu_climber_t<i_t, f_t>& climber)
  {
    auto& h_assign = climber.h_assignment;
    auto& h_best   = climber.h_best_assignment;
    for (int32_t v = 0; v < pb.n_variables; ++v) {
      h_assign[v] = (f_t)assign[v];
      h_best[v]   = (f_t)assign[v];
    }
    climber.h_incumbent_objective = (f_t)incumbent_objective;
    climber.h_best_objective      = (f_t)best_objective;
    climber.feasible_found        = true;
    // Emitted once per improvement so the benchmark harness can reconstruct the
    // incumbent trajectory exactly, rather than sampling it at log_interval.
    CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] new incumbent: objective %.17g",
                    climber.log_prefix.c_str(),
                    coefficient_bits(),
                    best_objective);
    if (climber.improvement_callback) {
      const double work_units = climber.work_units_elapsed.load(std::memory_order_acquire);
      climber.improvement_callback((f_t)best_objective, h_best, work_units);
    }
  }

  void reweight_constraint(int32_t r, int32_t new_weight)
  {
    if (new_weight == row_weight[r]) return;
    row_weight[r] = new_weight;
    if (new_weight > max_weight) max_weight = new_weight;
    // The slack is unchanged here, and no variable is excluded, so skip_var matches no index.
    const int32_t kb = pb.offsets[r], ke = pb.offsets[r + 1];
    fj_bin_patch_row(pb.variables.data(),
                     pb.coefficients.data(),
                     kb,
                     ke,
                     var_score.data(),
                     nnz_score_delta.data(),
                     assign_i32.data(),
                     new_weight,
                     row_slack[r],
                     -1);
    nnz_touched += ke - kb;
    nnz_patched += ke - kb;
  }

  // DDFW: every violated row gains weight taken from a satisfied neighbour above the donation
  // floor, so total weight is roughly conserved and differentiation stays local to the hard region.
  // Unit transfers stop moving the landscape on a long stall, so the amount grows with the stall.
  int32_t ddfw_transfer() const
  {
    if (iters_since_infeasible_improve <= fj_bin_ddfw_escalate_after) return fj_bin_ddfw_transfer;
    const int32_t over  = iters_since_infeasible_improve - fj_bin_ddfw_escalate_after;
    const int32_t steps = over / fj_bin_ddfw_escalate_after + 1;
    const int32_t scale = steps < fj_bin_ddfw_escalate_max ? steps : fj_bin_ddfw_escalate_max;
    return fj_bin_ddfw_transfer * scale;
  }

  void update_weights()
  {
    const int32_t transfer = ddfw_transfer();
    // Donors must stay above the floor, or weights go negative and every base score inverts.
    const int32_t donor_floor = fj_bin_ddfw_init + transfer - 1;

    for (int32_t cf : violated_list) {
      reweight_constraint(cf, row_weight[cf] + transfer);
      const int32_t vo = pb.offsets[cf], ve = pb.offsets[cf + 1];
      if (ve <= vo) continue;
      int32_t best_donor = -1, best_w = donor_floor;
      for (int32_t s = 0; s < fj_bin_ddfw_donor_samples; ++s) {
        const int32_t v  = pb.variables[vo + (int32_t)(rng.next_u32() % (uint32_t)(ve - vo))];
        const int32_t no = pb.reverse_offsets[v], ne = pb.reverse_offsets[v + 1];
        if (ne <= no) continue;
        const int32_t d =
          pb.reverse_constraints[no + (int32_t)(rng.next_u32() % (uint32_t)(ne - no))];
        if (d != cf && !is_violated[d] && row_weight[d] > best_w) {
          best_w     = row_weight[d];
          best_donor = d;
        }
      }
      if (best_donor >= 0) {
        const int32_t donated = row_weight[best_donor] - transfer;
        cuopt_assert(donated >= fj_bin_ddfw_init, "donation broke the weight floor");
        reweight_constraint(best_donor, donated);
      }
    }
    if (violated_list.empty()) {
      if (best_objective < last_best_objective) {
        iterations_at_same_objective = 0;
        last_best_objective          = best_objective;
      } else {
        ++iterations_at_same_objective;
      }
      objective_weight += objective_weight_increment();
    }
    track_infeasible_checkpoint();
  }

  // Stall-escalation for the objective weight, the feasible-region counterpart of ddfw_transfer:
  // a lane that keeps reaching local minima without moving its best objective needs more
  // objective pressure than one that is still improving.
  int32_t objective_weight_increment() const
  {
    if (iterations_at_same_objective <= fj_bin_obj_stall_after) return 1;
    const int32_t steps =
      1 + (iterations_at_same_objective - fj_bin_obj_stall_after) / fj_bin_obj_stall_after;
    return steps < fj_bin_obj_escalate_max ? steps : fj_bin_obj_escalate_max;
  }

  void reset_infeasible_checkpoint()
  {
    best_infeasible_assign.clear();
    best_infeasible_severity       = std::numeric_limits<int64_t>::max();
    checkpoint_severity            = std::numeric_limits<int64_t>::max();
    iters_since_infeasible_improve = 0;
  }

  void track_infeasible_checkpoint()
  {
    if (violated_list.empty()) {
      reset_infeasible_checkpoint();
      return;
    }

    int64_t severity = 0;
    for (int32_t r : violated_list) {
      cuopt_assert(row_slack[r] < 0, "row in violated_list is not violated");
      severity -= (int64_t)row_slack[r];
    }

    if (severity < best_infeasible_severity) {
      best_infeasible_severity       = severity;
      iters_since_infeasible_improve = 0;
      restores_since_improvement     = 0;
      if ((double)severity < (double)checkpoint_severity * infeasible_checkpoint_refresh_ratio) {
        best_infeasible_assign = assign;
        checkpoint_severity    = severity;
        ++n_checkpoint_snapshots;
      }
      return;
    }

    if (restores_since_improvement >= infeasible_restart_max_streak) return;
    if (++iters_since_infeasible_improve < infeasible_restart_window) return;
    if ((double)severity <= (double)best_infeasible_severity * infeasible_restart_degrade_ratio)
      return;
    if (best_infeasible_assign.empty()) return;

    cuopt_assert(checkpoint_severity >= best_infeasible_severity,
                 "checkpoint cannot beat the best severity seen");

    assign = best_infeasible_assign;
    for (int32_t v = 0; v < pb.n_variables; ++v)
      assign_i32[v] = assign[v];
    recompute_slack();

    ++n_checkpoint_restores;
    ++restores_since_improvement;
    if (restores_since_improvement > max_restores_since_improvement)
      max_restores_since_improvement = restores_since_improvement;
    iters_since_infeasible_improve = 0;
  }

  // Global argmax over every variable, affordable because var_score is maintained live. While the
  // objective weight is zero the full score is exactly var_score; above zero the sweep runs over
  // var_score plus the cached objective base. Only the local-minimum path falls to the scalar loop.
  std::pair<int32_t, int64_t> find_move_global(bool localmin)
  {
    if (!localmin && objective_weight == 0) {
      // The sweep reads var_score alone; the handful of tabu variables are held at the invalid
      // sentinel across it rather than tested per variable.
      int32_t saved_var[fj_bin_tabu_t::ring_size];
      int64_t saved_score[fj_bin_tabu_t::ring_size];
      const int32_t blocked = tabu.block_tabu(iters, var_score.data(), saved_var, saved_score);

      int32_t v = -1;
      int64_t s = fj_bin_score_invalid;
      fj_bin_argmax(var_score.data(), pb.n_variables, argmax_tile, v, s);

      fj_bin_tabu_t::unblock_tabu(blocked, var_score.data(), saved_var, saved_score);
      return {v, s};
    }

    if (!localmin) {
      // The breakthrough bonus is deliberately absent from the ranking: it depends on
      // incumbent_objective, so no per-variable form of it survives a move, and it occupies the low
      // field where it can only separate variables already tied on the base. The winner's score is
      // then taken from full_score so the caller sees the true value.
      ensure_objective_base();
      const int64_t* const obj_p = obj_base_score.data();
      const int64_t* const var_p = var_score.data();
      int64_t* const comb_p      = combined_score.data();
      for (int32_t v = 0; v < pb.n_variables; ++v)
        comb_p[v] = var_p[v] + obj_p[v];

      int32_t saved_var[fj_bin_tabu_t::ring_size];
      int64_t saved_score[fj_bin_tabu_t::ring_size];
      const int32_t blocked = tabu.block_tabu(iters, comb_p, saved_var, saved_score);

      int32_t v = -1;
      int64_t s = fj_bin_score_invalid;
      fj_bin_argmax(comb_p, pb.n_variables, argmax_tile, v, s);

      fj_bin_tabu_t::unblock_tabu(blocked, comb_p, saved_var, saved_score);
      if (v >= 0) s = full_score(v, (int8_t)(1 - 2 * assign[v]));
      return {v, s};
    }

    int32_t best_v = -1;
    int64_t best_s = fj_bin_score_invalid;
    for (int32_t v = 0; v < pb.n_variables; ++v) {
      if (tabu_blocked(v, localmin)) continue;
      const int64_t s = full_score(v, (int8_t)(1 - 2 * assign[v]));
      if (s > best_s) {
        best_s = s;
        best_v = v;
      }
    }
    return {best_v, best_s};
  }

  std::pair<int32_t, int64_t> find_move_in_rows(const std::vector<int32_t>& target_rows,
                                                bool localmin)
  {
    int32_t best_v = -1;
    int64_t best_s = fj_bin_score_invalid;
    for (int32_t r : target_rows) {
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t v = pb.variables[k];
        if (var_bitmap[v]) continue;
        var_bitmap[v] = 1;
        if (tabu_blocked(v, localmin)) continue;
        const int64_t s = full_score(v, (int8_t)(1 - 2 * assign[v]));
        if (s > best_s) {
          best_s = s;
          best_v = v;
        }
      }
    }
    // Restore the all-zero invariant by revisiting only what was set: the sampled rows hold a few
    // dozen variables against n in the thousands, so this is far cheaper than clearing the array.
    for (int32_t r : target_rows) {
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) var_bitmap[pb.variables[k]] = 0;
    }
    return {best_v, best_s};
  }

  std::pair<int32_t, int64_t> find_move_violated(int32_t sample_size, bool localmin)
  {
    // Draw the rows directly instead of reservoir-sampling the violated list: `std::sample` is
    // linear in the population, so it walked every violated row to keep a handful. Sampling with
    // replacement is what `find_move_satisfied` already does, and `find_move_in_rows` deduplicates
    // variables through `var_bitmap`, so a repeated row costs a bitmap sweep and no scoring.
    const int32_t n = (int32_t)violated_list.size();
    const std::vector<int32_t>* sampled = &violated_list;
    if (n > sample_size) {
      sample_buf.clear();
      for (int32_t i = 0; i < sample_size; ++i) {
        sample_buf.push_back(violated_list[rng.next_u32() % (uint32_t)n]);
      }
      sampled = &sample_buf;
    }
    auto move = find_move_in_rows(*sampled, localmin);

    // Breakthrough moves: once a feasible solution exists, allow objective-driven jumps.
    if (feasible_found && incumbent_objective >= best_objective + breakthrough_margin) {
      for (int32_t v : pb.objective_vars) {
        const double step = (best_objective - incumbent_objective) / pb.objective[v];
        double target     = pb.objective[v] > 0 ? std::floor(assign[v] + step)
                                                : std::ceil(assign[v] + step);
        if (target < 0) target = 0;
        if (target > 1) target = 1;
        if ((int8_t)target == assign[v]) continue;
        if (tabu_blocked(v, false)) continue;
        const int64_t s = full_score(v, (int8_t)((int8_t)target - assign[v]));
        if (s > move.second) move = {v, s};
      }
    }
    return move;
  }

  std::pair<int32_t, int64_t> find_move_satisfied(int32_t sample_size)
  {
    sample_buf.clear();
    for (int32_t tries = 0; (int32_t)sample_buf.size() < sample_size && tries < sample_size * 8;
         ++tries) {
      const int32_t r = (int32_t)(rng.next_u32() % (uint32_t)pb.n_constraints);
      if (!is_violated[r]) sample_buf.push_back(r);
    }
    return find_move_in_rows(sample_buf, false);
  }

  // True when flipping both variables leaves every row they touch satisfied. Both reverse ranges
  // are row-ascending, so shared rows are handled jointly by merging them.
  bool paired_flip_keeps_feasible(
    int32_t var1, int8_t delta1, int32_t var2, int8_t delta2) const
  {
    int32_t i = pb.reverse_offsets[var1], ie = pb.reverse_offsets[var1 + 1];
    int32_t j = pb.reverse_offsets[var2], je = pb.reverse_offsets[var2 + 1];

    while (i < ie || j < je) {
      const int32_t r1 = i < ie ? pb.reverse_constraints[i] : INT32_MAX;
      const int32_t r2 = j < je ? pb.reverse_constraints[j] : INT32_MAX;
      const int32_t r  = r1 < r2 ? r1 : r2;

      int32_t change = 0;
      if (r1 == r) change += (int32_t)pb.reverse_coefficients[i++] * delta1;
      if (r2 == r) change += (int32_t)pb.reverse_coefficients[j++] * delta2;
      if (row_slack[r] - change < 0) return false;
    }
    return true;
  }

  std::pair<std::pair<int32_t, int32_t>, int64_t> find_lift_2opt_move()
  {
    cuopt_assert(violated_list.empty(), "lift moves require a feasible incumbent");

    std::pair<int32_t, int32_t> best_pair = {-1, -1};
    int64_t best_s                        = 0;
    double best_improvement               = 0;
    if (pb.objective_vars.empty()) return {best_pair, best_s};

    const uint32_t n_obj  = (uint32_t)pb.objective_vars.size();
    const int32_t n_draws = n_obj < (uint32_t)fj_bin_2opt_candidates ? (int32_t)n_obj
                                                                    : fj_bin_2opt_candidates;

    for (int32_t t = 0; t < n_draws; ++t) {
      const int32_t var1  = pb.objective_vars[rng.next_u32() % n_obj];
      const int8_t delta1 = (int8_t)(1 - 2 * assign[var1]);
      if ((double)delta1 * pb.objective[var1] >= 0) continue;
      if (tabu_blocked(var1, false)) continue;

      // Only pairs are useful here: a flip breaking nothing is already the single-flip lift's job,
      // and one breaking several rows cannot be repaired by a single companion.
      int32_t broken   = -1;
      bool multiple    = false;
      for (int32_t i = pb.reverse_offsets[var1]; i < pb.reverse_offsets[var1 + 1] && !multiple;
           ++i) {
        const int32_t r = pb.reverse_constraints[i];
        if (row_slack[r] - (int32_t)pb.reverse_coefficients[i] * delta1 < 0) {
          if (broken >= 0)
            multiple = true;
          else
            broken = r;
        }
      }
      if (multiple || broken < 0) continue;

      for (int32_t k = pb.offsets[broken]; k < pb.offsets[broken + 1]; ++k) {
        const int32_t var2 = pb.variables[k];
        if (var2 == var1) continue;

        const int8_t delta2    = (int8_t)(1 - 2 * assign[var2]);
        const double combined  = (double)delta1 * pb.objective[var1] +
                                (double)delta2 * pb.objective[var2];
        if (combined >= 0) continue;
        if (tabu_blocked(var2, false)) continue;
        if (!paired_flip_keeps_feasible(var1, delta1, var2, delta2)) continue;

        // Both lift operators rank on the objective gain in its own units: the packed score counts
        // weights, and this engine requires an integral matrix but not integral objective terms.
        const double improvement = -combined;
        if (improvement > best_improvement) {
          best_improvement = improvement;
          best_s           = 1;  // sign only, never compared against another operator's score
          best_pair        = {var1, var2};
        }
      }
    }
    cuopt_assert((best_pair.first < 0) == (best_improvement <= 0),
                 "pair and score must agree on whether a move was found");
    return {best_pair, best_s};
  }

  std::pair<int32_t, int64_t> find_lift_move() const
  {
    cuopt_assert(violated_list.empty(), "lift moves require a feasible incumbent");

    int32_t best_v          = -1;
    int64_t best_s          = 0;
    double best_improvement = 0;
    for (int32_t v : pb.objective_vars) {
      const int8_t delta = (int8_t)(1 - 2 * assign[v]);
      if ((double)delta * pb.objective[v] >= 0) continue;
      if (tabu_blocked(v, false)) continue;
      // Base field is zero iff the flip breaks no row; K/2 splits it while |bonus| < 2^31.
      if (var_score[v] <= -(fj_bin_score_k / 2)) continue;
      const double improvement = -pb.objective[v] * (double)delta;
      if (improvement > best_improvement) {
        best_improvement = improvement;
        best_s           = 1;
        best_v           = v;
      }
    }
    cuopt_assert((best_v < 0) == (best_improvement <= 0),
                 "move and score must agree on whether a move was found");
    return {best_v, best_s};
  }

  // Flips a few variables drawn from violated rows, to leave a basin the weights cannot escape.
  void infeasible_region_kick()
  {
    const int32_t n_viol = (int32_t)violated_list.size();
    cuopt_assert(n_viol > 0, "kick requires a violated row");

    int32_t flipped[fj_bin_kick_rows * fj_bin_kick_vars_per_row];
    int32_t n_flipped = 0;

    for (int32_t i = 0; i < fj_bin_kick_rows; ++i) {
      const int32_t r        = violated_list[rng.next_u32() % (uint32_t)n_viol];
      const int32_t row_begin = pb.offsets[r];
      const int32_t row_end   = pb.offsets[r + 1];
      if (row_begin >= row_end) continue;

      for (int32_t j = 0; j < fj_bin_kick_vars_per_row; ++j) {
        const int32_t k = row_begin + (int32_t)(rng.next_u32() % (uint32_t)(row_end - row_begin));
        const int32_t v = pb.variables[k];

        bool already = false;
        for (int32_t f = 0; f < n_flipped && !already; ++f)
          already = flipped[f] == v;
        if (already) continue;

        cuopt_assert(n_flipped < fj_bin_kick_rows * fj_bin_kick_vars_per_row, "flip list overflow");
        flipped[n_flipped++] = v;
        assign[v]            = (int8_t)(1 - assign[v]);
        assign_i32[v]        = assign[v];
      }
    }
    recompute_slack();
  }

  void perturb()
  {
    if (pb.objective_vars.empty()) return;
    const uint32_t n = (uint32_t)pb.objective_vars.size();
    for (int i = 0; i < 2; ++i) {
      const int32_t v = pb.objective_vars[rng.next_u32() % n];
      assign[v]       = (int8_t)(rng.next_u32() & 1u);
      assign_i32[v]   = assign[v];
    }
    recompute_slack();
  }

  // Restart returns the assignment to the seed the climber was constructed with, leaving the
  // recorded best and the global iteration counter intact.
  void do_restart()
  {
    assign = seed_assign;
    for (int32_t v = 0; v < pb.n_variables; ++v) assign_i32[v] = assign[v];
    for (int32_t r = 0; r < pb.n_constraints; ++r) row_weight[r] = pb.initial_weight[r];
    max_weight       = fj_bin_ddfw_init;
    objective_weight = seed_objective_weight;
    reset_infeasible_checkpoint();
    tabu.clear(iters);
    recompute_slack();
    last_restart_iter           = iters;
    last_feasible_entrance_iter = iters;
  }

  void init(fj_cpu_climber_t<i_t, f_t>& climber)
  {
    const auto& params  = climber.settings.parameters;
    seed                = climber.settings.seed;
    rng                 = raft::random::PCGenerator((uint64_t)seed, 0, 0);
    tabu_tenure_min     = params.tabu_tenure_min;
    tabu_tenure_max     = params.tabu_tenure_max;
    breakthrough_margin = params.breakthrough_move_epsilon;
    perturb_interval    = climber.perturb_interval;
    mtm_viol_samples    = climber.mtm_viol_samples;
    mtm_sat_samples     = climber.mtm_sat_samples;

    infeasible_restart_window           = climber.infeasible_restart_window;
    infeasible_restart_max_streak       = climber.infeasible_restart_max_streak;
    infeasible_restart_degrade_ratio    = (double)climber.infeasible_restart_degrade_ratio;
    infeasible_checkpoint_refresh_ratio = (double)climber.infeasible_checkpoint_refresh_ratio;
    cuopt_assert(infeasible_restart_window > 0, "invalid infeasible restart window");
    cuopt_assert(infeasible_restart_max_streak > 0, "invalid infeasible restart streak cap");
    cuopt_assert(infeasible_restart_degrade_ratio >= 1.0, "degrade ratio should be at least one");
    cuopt_assert(
      infeasible_checkpoint_refresh_ratio > 0.0 && infeasible_checkpoint_refresh_ratio <= 1.0,
      "checkpoint refresh ratio should be in (0, 1]");

    if (tabu_tenure_max <= tabu_tenure_min) tabu_tenure_max = tabu_tenure_min + 1;

    // The tabu ring is indexed by iteration modulo its size, so a slot is reused after ring_size
    // iterations. A tenure that long would be overwritten while the variable is still tabu, and the
    // argmax would stop excluding it. Clamped as well as asserted: release builds compile the assert
    // out, and silently dropping tabu entries is worse than a shorter tenure.
    cuopt_assert(tabu_tenure_max <= fj_bin_tabu_t::max_tenure,
                 "tabu tenure exceeds the tabu ring, live entries would be evicted");
    if (tabu_tenure_max > fj_bin_tabu_t::max_tenure) tabu_tenure_max = fj_bin_tabu_t::max_tenure;

    const int32_t n = pb.n_variables, m = pb.n_constraints;
    const auto& h_assign = climber.h_assignment;
    assign.resize(n);
    for (int32_t v = 0; v < n; ++v) {
      const double val = (double)h_assign[v];
      assign[v]        = (int8_t)(val >= 0.5 ? 1 : 0);
    }
    seed_assign = assign;
    best_assign = assign;
    reset_infeasible_checkpoint();
    assign_i32.assign(n, 0);
    for (int32_t v = 0; v < n; ++v) assign_i32[v] = assign[v];

    row_weight.assign(pb.initial_weight.begin(), pb.initial_weight.end());
    row_slack.assign(m, 0);

    var_score.assign(n, 0);
    nnz_score_delta.assign(pb.nnz + fj_bin_simd_padding, 0);
    // Zeroed once: ensure_objective_base only ever rewrites the objective variables.
    obj_base_score.assign(n, 0);
    combined_score.assign(n, 0);
    obj_base_weight = -1;
    tabu.resize(n);
    is_violated.assign(m, 0);
    vpos.assign(m, -1);
    violated_list.clear();
    var_bitmap.assign(n, 0);

    const int32_t seeded_weight = (int32_t)std::lround(climber.h_objective_weight);
    cuopt_assert(seeded_weight >= 0, "objective weight should be positive or zero");

    double abs_obj_sum = 0;
    for (int32_t v : pb.objective_vars) abs_obj_sum += std::fabs(pb.objective[v]);
    obj_magnitude = abs_obj_sum > 0 ? abs_obj_sum / (double)pb.objective_vars.size() : 1.0;
    cuopt_assert(std::isfinite(obj_magnitude) && obj_magnitude > 0,
                 "objective magnitude unit must be finite and positive");

    argmax_tile                  = fj_bin_argmax_tile();
    objective_weight             = seeded_weight > 0 ? seeded_weight : 0;
    seed_objective_weight        = objective_weight;
    max_weight                   = fj_bin_ddfw_init;
    incumbent_objective          = 0;
    best_objective               = std::numeric_limits<double>::infinity();
    last_best_objective          = std::numeric_limits<double>::infinity();
    iterations_at_same_objective = 0;
    feasible_found               = false;
    iters                        = 0;
    last_restart_iter            = 0;
    last_kick_iter               = 0;
    recompute_slack();
  }

  void solve(fj_cpu_climber_t<i_t, f_t>& climber, f_t time_limit, double work_unit_limit)
  {
    init(climber);

    const auto loop_start = std::chrono::high_resolution_clock::now();
    const auto limit =
      std::chrono::milliseconds((int64_t)std::floor((double)time_limit * 1000.0));
    const bool bounded_time = std::isfinite((double)time_limit);

    while (!climber.halted && !climber.preemption_flag.load()) {
      if (bounded_time && std::chrono::high_resolution_clock::now() - loop_start > limit) break;
      if (iters >= climber.settings.iteration_limit) break;
      if (iters - last_restart_iter >= fj_bin_restart_period) do_restart();
      tabu.maybe_rebase(iters);

      int32_t move_var                 = -1;
      int64_t score                    = fj_bin_score_invalid;
      std::pair<int32_t, int32_t> pair2 = {-1, -1};
      if (violated_list.empty()) {
        std::tie(move_var, score) = find_lift_move();
        // Pairs are only reachable once no single improving flip preserves feasibility.
        if (score <= 0) {
          int64_t pair_score;
          std::tie(pair2, pair_score) = find_lift_2opt_move();
          if (pair_score > 0) score = pair_score;
        }
      }
      if (pair2.first < 0 && score <= 0) std::tie(move_var, score) = find_move_global(false);
      if (pair2.first < 0 && feasible_found && score <= 0)
        std::tie(move_var, score) = find_move_satisfied(mtm_sat_samples);

      bool perturb_now = false;
      if (violated_list.empty() && iters - last_feasible_entrance_iter > perturb_interval) {
        perturb_now                 = true;
        last_feasible_entrance_iter = iters;
      }

      if (pair2.first >= 0 && !perturb_now) {
        apply_move(pair2.first, (int8_t)(1 - 2 * assign[pair2.first]), climber);
        apply_move(pair2.second, (int8_t)(1 - 2 * assign[pair2.second]), climber);
      } else if (score > 0 && move_var >= 0 && !perturb_now) {
        apply_move(move_var, (int8_t)(1 - 2 * assign[move_var]), climber);
      } else {
        update_weights();
        const bool kick_ready = !violated_list.empty() &&
                                iters_since_infeasible_improve >= fj_bin_kick_after &&
                                iters - last_kick_iter >= fj_bin_kick_cooldown &&
                                iters - last_restart_iter >= fj_bin_kick_restart_guard;
        if (kick_ready) {
          infeasible_region_kick();
          last_kick_iter = iters;
        } else if (perturb_now) {
          perturb();
        }
        std::tie(move_var, score) = find_move_violated(1, true);
        const int32_t v           = move_var >= 0 ? move_var : 0;
        apply_move(v, (int8_t)(1 - 2 * assign[v]), climber);
      }

      if (iters % climber.log_interval == 0) {
        CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] iteration: %d, viol: %zu, best: %g, maxw: %d",
                        climber.log_prefix.c_str(),
                        coefficient_bits(),
                        iters,
                        violated_list.size(),
                        best_objective,
                        max_weight);
      }
      if (iters % climber.diversity_callback_interval == 0 && climber.diversity_callback) {
        auto& h_assign = climber.h_assignment;
        for (int32_t v = 0; v < pb.n_variables; ++v) h_assign[v] = (f_t)assign[v];
        climber.diversity_callback((f_t)incumbent_objective, h_assign);
      }

      // Work-unit proxy. nnz_touched is cumulative, reproducing the accumulation shape the general
      // path gets from its cumulative byte counters.
      if (iters % 100 == 0 && iters > 0) {
        const double work = (double)nnz_touched * fj_bin_bytes_per_nnz * climber.work_unit_bias / 1e10;
        climber.work_units_elapsed.store(work, std::memory_order_release);
        if (climber.producer_sync != nullptr) climber.producer_sync->notify_progress();
        if (work >= work_unit_limit) break;
      }

      ++iters;
    }

    compute_saturation();
    verify_incumbent(climber);
    climber.iterations = (i_t)iters;
    CUOPT_LOG_DEBUG(
      "%sCPUFJ[bin%d] done: %d iterations, best %g, max weight %d, aggregate base %d/%d, bonus %d/%d",
      climber.log_prefix.c_str(),
      coefficient_bits(),
      iters,
      best_objective,
      max_weight,
      max_aggregate_base,
      fj_bin_base_limit,
      max_aggregate_bonus,
      fj_bin_bonus_limit);
    CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] work: nnz_patched %lld, rows_walked %lld",
                    climber.log_prefix.c_str(),
                    coefficient_bits(),
                    (long long)nnz_patched,
                    (long long)rows_walked);
    CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] checkpoint: %lld restores, %lld snapshots, max streak %d",
                    climber.log_prefix.c_str(),
                    coefficient_bits(),
                    (long long)n_checkpoint_restores,
                    (long long)n_checkpoint_snapshots,
                    max_restores_since_improvement);
  }
};

template <typename i_t, typename f_t>
bool try_cpufj_binary_solve(fj_cpu_climber_t<i_t, f_t>& climber,
                            f_t time_limit,
                            double work_unit_limit)
{
  // Escape hatch for A/B against the general path on an instance the fast path would take. The two
  // paths are meant to search identically, so any divergence is a bug in this one; setting this is
  // how that gets bisected without editing the eligibility scan.
  static const bool disabled = std::getenv("CUOPT_NO_BINFJ") != nullptr;
  if (disabled) return false;

  const fj_bin_scan_t scan = fj_bin_scan(climber);
  if (scan.reject != fj_binary_reject_t::none) {
    CUOPT_LOG_DEBUG("%sCPUFJ binary fast path declined: %s (row %d, var %d)",
                    climber.log_prefix.c_str(),
                    fj_binary_reject_name(scan.reject),
                    scan.bad_row,
                    scan.bad_var);
    return false;
  }

  auto run = [&](auto& engine) -> bool {
    if (!fj_bin_narrow(climber, scan, engine.pb)) {
      CUOPT_LOG_DEBUG("%sCPUFJ binary fast path declined: %s",
                      climber.log_prefix.c_str(),
                      fj_binary_reject_name(fj_binary_reject_t::narrow_check_failed));
      return false;
    }
    CUOPT_LOG_DEBUG(
      "%sCPUFJ binary fast path enabled: int%d coefficients, %d rows after one-sided split",
      climber.log_prefix.c_str(),
      scan.coefficient_bits,
      scan.n_split_constraints);
    engine.solve(climber, time_limit, work_unit_limit);
    return true;
  };

  if (scan.coefficient_bits == 8) {
    fj_bin_engine_t<i_t, f_t, int8_t> engine;
    return run(engine);
  }
  fj_bin_engine_t<i_t, f_t, int16_t> engine;
  return run(engine);
}

#if MIP_INSTANTIATE_FLOAT
template bool try_cpufj_binary_solve(fj_cpu_climber_t<int, float>& climber,
                                     float time_limit,
                                     double work_unit_limit);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool try_cpufj_binary_solve(fj_cpu_climber_t<int, double>& climber,
                                     double time_limit,
                                     double work_unit_limit);
#endif

}  // namespace cuopt::mathematical_optimization::mip
