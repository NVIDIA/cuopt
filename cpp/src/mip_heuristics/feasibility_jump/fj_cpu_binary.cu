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

#include <raft/random/rng_device.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
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


// Work-unit proxy: bytes attributed per nnz touched. Stands in for the byte counters the general
// path reads off its instrumented vectors. Calibrated by the owner.
constexpr double fj_bin_bytes_per_nnz = 16.0;

// Tabu for binary variables. A binary variable's only move is a flip, so the direction is a
// function of the current assignment and the general four-array scheme collapses to two.
// flip_until is biased by a rolling base so it fits uint16.
struct fj_bin_tabu_t {
  static constexpr int32_t window = 65535 - 64;

  std::vector<uint16_t> flip_until;
  std::vector<int32_t> last_flip;
  int32_t base{0};

  // Ring of the last ring_size flips, indexed by iteration rather than by expiry. One variable
  // flips per iteration, so each slot takes exactly one entry and nothing is ever displaced while
  // still tabu; indexing by expiry instead would collide whenever two flips share a deadline, which
  // a simulation over the [3,12] tenure range puts at 35% of insertions.
  //
  // This is what lets the global argmax stop reading flip_until. Tenure is bounded by the ring, so
  // at most ring_size variables are tabu at once out of n: reading two bytes per variable to answer
  // a question that is almost always no costs a third of that scan's traffic. On crypt16 it is the
  // difference between a 31 KB working set that overflows a 32 KB L1 and a 21 KB one that does not.
  static constexpr int32_t ring_size = 16;
  int32_t ring_var[ring_size];
  int32_t ring_expiry[ring_size];


  void resize(int32_t n)
  {
    flip_until.assign(n, 0);
    last_flip.assign(n, 0);
    clear_ring();
    base = 0;
  }

  void clear(int32_t iter)
  {
    std::fill(flip_until.begin(), flip_until.end(), (uint16_t)0);
    std::fill(last_flip.begin(), last_flip.end(), 0);
    clear_ring();
    base = iter;
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
    flip_until[v] = (uint16_t)(iter + tenure - base);
    last_flip[v]  = iter;

    // Drop any earlier entry for v before inserting the new one. flip_until keeps one deadline per
    // variable and a reflip overwrites it, so without this the ring would hold a superseded, later
    // deadline and block v past the point the per-variable test would have released it. A variable
    // can be reflipped while still tabu: the local-minimum path tests the weaker
    // iter == last_flip + 1. With this the ring holds at most one live entry per variable carrying
    // its current deadline, which is exactly the invariant flip_until maintains.
    for (int32_t i = 0; i < ring_size; ++i) {
      if (ring_var[i] == v) ring_var[i] = -1;
    }

    const int32_t slot = iter & (ring_size - 1);
    ring_var[slot]     = v;
    ring_expiry[slot]  = iter + tenure;
  }

  // Blocks every currently-tabu variable by writing the invalid sentinel over its score, and reports
  // how many were touched. Paired with unblock around one argmax and nothing else: the score is
  // maintained incrementally, so it may only be disturbed across a window in which no patch runs.
  int32_t block_tabu(int32_t iter,
                     int32_t* var_score,
                     int32_t (&saved_var)[ring_size],
                     int32_t (&saved_score)[ring_size]) const
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

  // Reverse order on purpose: a variable flipped twice inside the ring appears twice, and its second
  // save holds the sentinel written by the first. Unwinding backwards restores the true score last.
  static void unblock_tabu(int32_t k,
                           int32_t* var_score,
                           const int32_t (&saved_var)[ring_size],
                           const int32_t (&saved_score)[ring_size])
  {
    for (int32_t i = k - 1; i >= 0; --i) var_score[saved_var[i]] = saved_score[i];
  }


  bool blocked(int32_t v, int32_t iter, bool localmin) const
  {
    return localmin ? (iter == last_flip[v] + 1) : ((uint16_t)(iter - base) < flip_until[v]);
  }

  // Rebase before iter - base can overflow the uint16 window. Expired entries saturate to 0,
  // which reads as not-tabu.
  void maybe_advance(int32_t iter)
  {
    if ((int64_t)iter - base <= window) return;
    const uint16_t shift = (uint16_t)(iter - base);
    for (uint16_t& fu : flip_until) fu = (fu > shift) ? (uint16_t)(fu - shift) : (uint16_t)0;
    base = iter;
  }
};

// One row of the narrowed problem. bound/sign encode the single finite bound the split leaves:
// sign +1 for lhs <= bound, -1 for lhs >= bound. cmax is max|coef| over the row, which bounds how
// far a single flip can move the slack.
template <typename coef_t>
struct fj_bin_row_t {
  int32_t lhs;
  int32_t weight;
  int32_t bound;
  coef_t cmax;
  int8_t sign;
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
  std::vector<coef_t> reverse_coefficients;
  std::vector<int32_t> reverse_to_csr;

  std::vector<int32_t> bound;
  std::vector<int8_t> sign;
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
};

// DDFW and restart have no general-path equivalent, so their defaults live here until there is a
// reason to promote them alongside the other FJ knobs.
constexpr int32_t fj_bin_ddfw_init          = 10;  // initial weight, also the donation floor
constexpr int32_t fj_bin_ddfw_transfer      = 1;
constexpr int32_t fj_bin_ddfw_donor_samples = 1;
constexpr int32_t fj_bin_restart_period     = 5000000;

// `[study]` Prefetch distance for the reverse-CSR row walk, in rows. Each move visits the rows of
// one variable through reverse_constraints, whose entries are effectively random indices into a row
// array far larger than L2, and the sequence is data-dependent so no hardware prefetcher can follow
// it. reverse_constraints is padded by this much so the lookahead needs no bounds test.
constexpr int32_t fj_bin_pf_dist = 8;

// Limits of the packed score. Breaching either corrupts the ordering, so compute_saturation
// reports the observed peaks against them at end of solve.
constexpr int32_t fj_bin_base_limit  = 1 << 16;
constexpr int32_t fj_bin_bonus_limit = 1 << 14;

static inline bool fj_bin_is_integral(double v, double tol) { return std::fabs(v - std::round(v)) <= tol; }

static inline bool fj_bin_in_int32(double v)
{
  return v >= (double)INT32_MIN && v <= (double)INT32_MAX;
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

  const double tol       = c.view.pb.tolerances.integrality_tolerance;
  const auto& var_bounds = c.h_var_bounds;
  const auto& var_types  = c.h_var_types;

  for (int32_t v = 0; v < n; ++v) {
    auto bounds = var_bounds[v];
    if (var_types[v] != var_t::INTEGER || std::fabs(get_lower(bounds)) > tol ||
        std::fabs(get_upper(bounds) - 1.0) > tol) {
      out.reject  = fj_binary_reject_t::non_binary_var;
      out.bad_var = v;
      return out;
    }
  }

  const auto& offsets = c.h_offsets;
  const auto& coeffs  = c.h_coefficients;
  const auto& cstr_lb = c.h_cstr_lb;
  const auto& cstr_ub = c.h_cstr_ub;

  double max_abs_coefficient = 0;
  for (int32_t r = 0; r < m; ++r) {
    double row_abs_sum = 0;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      const double a = coeffs[k];
      if (!fj_bin_is_integral(a, tol)) {
        out.reject  = fj_binary_reject_t::fractional_coefficient;
        out.bad_row = r;
        return out;
      }
      const double abs_a = std::fabs(std::round(a));
      row_abs_sum += abs_a;
      if (abs_a > max_abs_coefficient) max_abs_coefficient = abs_a;
    }

    // A binary assignment can drive lhs to sum|coef|; keep that inside the int32 accumulator with
    // room to spare. The int8-only reference engine never needed this bound.
    if (row_abs_sum > (double)(INT32_MAX / 2)) {
      out.reject  = fj_binary_reject_t::lhs_headroom;
      out.bad_row = r;
      return out;
    }

    const double lb    = cstr_lb[r];
    const double ub    = cstr_ub[r];
    const bool lb_fin  = std::isfinite(lb);
    const bool ub_fin  = std::isfinite(ub);
    const double sides[2] = {lb, ub};
    const bool finite[2]  = {lb_fin, ub_fin};
    for (int s = 0; s < 2; ++s) {
      if (!finite[s]) continue;
      if (!fj_bin_is_integral(sides[s], tol)) {
        out.reject  = fj_binary_reject_t::fractional_row_bound;
        out.bad_row = r;
        return out;
      }
      if (!fj_bin_in_int32(std::round(sides[s]))) {
        out.reject  = fj_binary_reject_t::row_bound_out_of_range;
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
                   int32_t n_split,
                   fj_bin_problem_t<coef_t>& pb)
{
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
  pb.sign.reserve(n_split);
  pb.cmax.reserve(n_split);
  pb.initial_weight.reserve(n_split);

  std::vector<double> incoming_weight;
  incoming_weight.reserve(n_split);

  // Each split row inherits the weight of the side it came from: left is the lower-bound side,
  // right the upper.
  auto emit = [&](int32_t r, double side_bound, int8_t sign, double weight) -> bool {
    coef_t row_cmax = 1;
    for (int32_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      const double a = coeffs[k];
      const long ai  = std::lround(a);
      if (!fj_bin_is_integral(a, tol) || ai < std::numeric_limits<coef_t>::min() ||
          ai > std::numeric_limits<coef_t>::max()) {
        return false;
      }
      pb.variables.push_back(variables[k]);
      pb.coefficients.push_back((coef_t)ai);
      const coef_t abs_a = (coef_t)std::labs(ai);
      if (abs_a > row_cmax) row_cmax = abs_a;
    }
    const long b = std::lround(side_bound);
    if (!fj_bin_in_int32((double)b)) return false;
    pb.offsets.push_back((int32_t)pb.variables.size());
    pb.bound.push_back((int32_t)b);
    pb.sign.push_back(sign);
    pb.cmax.push_back(row_cmax);
    incoming_weight.push_back(weight);
    return true;
  };

  for (int32_t r = 0; r < m; ++r) {
    const double lb = cstr_lb[r];
    const double ub = cstr_ub[r];
    if (std::isfinite(lb) && !emit(r, lb, (int8_t)-1, left_w[r])) return false;
    if (std::isfinite(ub) && !emit(r, ub, (int8_t)1, right_w[r])) return false;
  }
  if ((int32_t)pb.bound.size() != n_split) return false;
  pb.nnz = (int32_t)pb.variables.size();

  // One vector of padding past nnz, so the row kernel can load and store whole vectors at the last
  // row without running off the end and can therefore mask its remainder rather than peeling it
  // into a scalar tail. The padding is never read as data: every lane past a row's end is excluded
  // from the gather, the scatter and the store by the row-length mask.
  const int32_t pad = fj_bin_simd_padding();
  pb.variables.resize(pb.nnz + pad, 0);
  pb.coefficients.resize(pb.nnz + pad, (coef_t)0);

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
  {
    std::vector<int32_t> cursor(pb.reverse_offsets.begin(), pb.reverse_offsets.begin() + n);
    for (int32_t r = 0; r < n_split; ++r) {
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t slot           = cursor[pb.variables[k]]++;
        pb.reverse_constraints[slot] = r;
        pb.reverse_coefficients[slot] = pb.coefficients[k];
        pb.reverse_to_csr[slot]       = k;
      }
    }
  }
  // Lookahead room for the row-walk prefetch. Reads land on row 0, which is prefetched harmlessly.
  pb.reverse_constraints.resize(pb.nnz + fj_bin_pf_dist, 0);

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
struct fj_bin_engine_t : fj_binary_state_t<i_t, f_t> {
  fj_bin_problem_t<coef_t> pb;
  std::vector<fj_bin_row_t<coef_t>> rows;

  std::vector<int8_t> assign;
  std::vector<int8_t> best_assign;
  std::vector<int8_t> seed_assign;   // restart target
  std::vector<int32_t> assign_i32;   // gather mirror for the SIMD patch (Batch B)

  std::vector<int32_t> var_score;    // live feasibility score of flipping each variable
  std::vector<int32_t> nnz_score_delta;  // per CSR nnz: last score delta of variables[k] in its row

  fj_bin_tabu_t tabu;

  std::vector<uint8_t> is_violated;
  std::vector<int32_t> violated_list;
  std::vector<int32_t> vpos;
  std::vector<char> var_bitmap;

  // One generator advanced across the whole search, rather than one re-seeded per call site per
  // iteration. Re-seeding from `seed + iters` gave every call site in an iteration the identical
  // stream, and a 624-word Mersenne state was being built and discarded on every move selection.
  raft::random::PCGenerator rng{0, 0, 0};
  std::vector<int32_t> sample_buf;  // move-selection row sample, reused to keep the loop allocation-free

  int32_t objective_weight{0};
  double incumbent_objective{0};
  double best_objective{std::numeric_limits<double>::infinity()};
  int32_t max_weight{1};
  bool feasible_found{false};

  int32_t iters{0};
  int32_t last_feasible_entrance_iter{0};
  int32_t last_restart_iter{0};
  int64_t nnz_touched{0};

  // Denominator for the ops-per-nnz roofline: nonzeros the row kernel actually processes, and the
  // rows walked to find them. Unlike nnz_touched these are not mixed with the full-matrix rebuilds.
  int64_t nnz_patched{0};
  int64_t rows_walked{0};

  // Tile width for the argmax sweep. Governs how often the running maximum is raised, which is
  // what bounds the index re-scan, so it is about the shape of the sweep and not cache capacity.
  int32_t argmax_tile{256};

  // Rows at or below these lengths go to the 4- and 8-lane patch kernels; 0 disables that width.
  int32_t narrow4_max{0};
  int32_t narrow8_max{0};

  // Settings read at solve entry, where the climber carries populated values.
  int32_t seed{0};
  int32_t tabu_tenure_min{3};
  int32_t tabu_tenure_max{13};
  int32_t perturb_interval{100};
  int32_t mtm_viol_samples{25};
  int32_t mtm_sat_samples{15};
  double breakthrough_margin{1e-4};

  int32_t max_aggregate_base{0};
  int32_t max_aggregate_bonus{0};

  int coefficient_bits() const override { return 8 * (int)sizeof(coef_t); }
  int n_split_constraints() const override { return pb.n_constraints; }
  i_t iterations() const override { return (i_t)iters; }

  void saturation(int& base_peak, int& bonus_peak) const override
  {
    base_peak  = max_aggregate_base;
    bonus_peak = max_aggregate_bonus;
  }

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
      const int64_t slack = (int64_t)pb.sign[r] * ((int64_t)pb.bound[r] - lhs);
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
        const fj_bin_row_t<coef_t>& h = rows[pb.reverse_constraints[i]];
        const int32_t s  = h.sign;
        const int32_t os = s * (h.bound - h.lhs);
        const int32_t ns = os - s * ((int32_t)pb.reverse_coefficients[i] * flip);
        int32_t base = 0, bonus = 0;
        fj_bin_score_delta_parts(os, ns, h.weight, base, bonus);
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

  // Branchless score delta of flipping a variable, as seen by one row. base is the weighted change
  // in satisfaction; bonus is the weighted change in strict slack. When both states are violated
  // the improving direction earns half weight, matching excess_improvement_weight of 1/2.
  int32_t score_delta(const fj_bin_row_t<coef_t>& h, int32_t lhs, int8_t delta, coef_t k) const
  {
    const int32_t s  = h.sign;
    const int32_t os = s * (h.bound - lhs);
    const int32_t ns = os - s * ((int32_t)k * delta);
    return fj_bin_packed_score_delta(os, ns, h.weight);
  }

  void rebuild_scores()
  {
    std::fill(var_score.begin(), var_score.end(), 0);
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      const fj_bin_row_t<coef_t>& h = rows[r];
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t v = pb.variables[k];
        const int32_t p = score_delta(h, h.lhs, (int8_t)(1 - 2 * assign[v]), pb.coefficients[k]);
        nnz_score_delta[k] = p;
        var_score[v] += p;
      }
    }
    nnz_touched += pb.nnz;
  }

  void recompute_lhs()
  {
    violated_list.clear();
    std::fill(is_violated.begin(), is_violated.end(), (uint8_t)0);
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      int32_t lhs = 0;
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k)
        lhs += (int32_t)pb.coefficients[k] * assign[pb.variables[k]];
      rows[r].lhs = lhs;
      if (rows[r].sign * (rows[r].bound - lhs) < 0) set_violated(r);
    }
    incumbent_objective = 0;
    for (int32_t v = 0; v < pb.n_variables; ++v) incumbent_objective += pb.objective[v] * assign[v];
    nnz_touched += pb.nnz;
    rebuild_scores();
  }

  int32_t objective_terms(int32_t v, int8_t delta) const
  {
    const double obj_diff = pb.objective[v] * delta;
    const int32_t base = obj_diff < 0 ? objective_weight : (obj_diff > 0 ? -objective_weight : 0);
    int32_t bonus      = 0;
    const bool old_better = incumbent_objective < best_objective;
    const bool new_better = incumbent_objective + obj_diff < best_objective;
    if (!old_better && new_better) {
      bonus += objective_weight;
    } else if (old_better && !new_better) {
      bonus -= objective_weight;
    }
    return base * fj_bin_score_k + bonus;
  }

  int32_t full_score(int32_t v, int8_t delta) const
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
    int32_t own_score           = 0;

    for (int32_t ii = ob; ii < oe; ++ii) {
      // Write hint: the row's lhs is updated at the end of every iteration, so the line is wanted
      // exclusive. The padding on reverse_constraints makes the lookahead unconditional.
      __builtin_prefetch(&rows[pb.reverse_constraints[ii + fj_bin_pf_dist]], 1, 3);

      const int32_t r         = pb.reverse_constraints[ii];
      fj_bin_row_t<coef_t>& h = rows[r];
      const coef_t kv         = pb.reverse_coefficients[ii];
      const int32_t old_lhs   = h.lhs;
      const int32_t new_lhs   = old_lhs + (int32_t)kv * delta;
      const int32_t s         = h.sign;
      const int32_t old_slack = s * (h.bound - old_lhs);
      const int32_t new_slack = s * (h.bound - new_lhs);

      if (new_slack < 0 && old_slack >= 0) {
        set_violated(r);
      } else if (new_slack >= 0 && old_slack < 0) {
        set_satisfied(r);
      }

      // A row that stays clear of its boundary by more than max|coef| on both sides cannot change
      // any variable's satisfaction flags, so its patch is skipped entirely.
      const int32_t margin = h.cmax;
      const bool deep_sat  = old_slack > margin && new_slack > margin;
      const bool deep_viol = old_slack < -margin && new_slack < -margin;
      if (!(deep_sat || deep_viol)) {
        const int32_t kb = pb.offsets[r], ke = pb.offsets[r + 1];
        // The offsets are already loaded for the call, so the width choice is a compare rather
        // than a stored per-row flag.
        // TODO: check that this may not cause AVX512 powerdown overheads if the AVX2 row/AVX512 row ratio is unbalanced
        fj_bin_patch_row(fj_bin_patch_width_for(ke - kb, narrow4_max, narrow8_max),
                         pb.variables.data(),
                         pb.coefficients.data(),
                         kb,
                         ke,
                         var_score.data(),
                         nnz_score_delta.data(),
                         assign_i32.data(),
                         s,
                         h.weight,
                         new_slack,
                         var);
        nnz_touched += ke - kb;
        nnz_patched += ke - kb;
      }

      // The flipped variable's own score delta is provably zero when the row is deeply satisfied
      // both ways, and already stored as zero there.
      if (!deep_sat) {
        const int32_t pv = score_delta(h, new_lhs, new_flip, kv);
        own_score += pv;
        nnz_score_delta[pb.reverse_to_csr[ii]] = pv;
      }
      h.lhs = new_lhs;
    }
    nnz_touched += oe - ob;
    rows_walked += oe - ob;

    if (prev_violated > 0 && violated_list.empty()) last_feasible_entrance_iter = iters;

    assign[var]     = new_val;
    assign_i32[var] = new_val;
    var_score[var]  = own_score;
    incumbent_objective += pb.objective[var] * delta;

    if (violated_list.empty() && incumbent_objective < best_objective) {
      best_objective = incumbent_objective;
      best_assign    = assign;
      feasible_found = true;
      report_incumbent(climber);
    }

    const int32_t tenure =
      tabu_tenure_min + (int32_t)(rng.next_u32() % (uint32_t)(tabu_tenure_max - tabu_tenure_min));
    tabu.on_flip(var, iters, tenure);
    std::fill(var_bitmap.begin(), var_bitmap.end(), (char)0);
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
    if (climber.improvement_callback) {
      const double work_units = climber.work_units_elapsed.load(std::memory_order_acquire);
      climber.improvement_callback((f_t)best_objective, h_best, work_units);
    }
  }

  void reweight_constraint(int32_t r, int32_t new_weight)
  {
    fj_bin_row_t<coef_t>& h = rows[r];
    if (new_weight == h.weight) return;
    h.weight = new_weight;
    if (new_weight > max_weight) max_weight = new_weight;
    // lhs is unchanged here, and no variable is excluded, so skip_var matches no index.
    const int32_t kb = pb.offsets[r], ke = pb.offsets[r + 1];
    fj_bin_patch_row(fj_bin_patch_width_for(ke - kb, narrow4_max, narrow8_max),
                     pb.variables.data(),
                     pb.coefficients.data(),
                     kb,
                     ke,
                     var_score.data(),
                     nnz_score_delta.data(),
                     assign_i32.data(),
                     h.sign,
                     h.weight,
                     h.sign * (h.bound - h.lhs),
                     -1);
    nnz_touched += ke - kb;
    nnz_patched += ke - kb;
  }

  // DDFW: every violated row gains weight taken from a satisfied neighbour above the donation
  // floor, so total weight is roughly conserved and differentiation stays local to the hard region.
  void update_weights()
  {
    for (int32_t cf : violated_list) {
      reweight_constraint(cf, rows[cf].weight + fj_bin_ddfw_transfer);
      const int32_t vo = pb.offsets[cf], ve = pb.offsets[cf + 1];
      if (ve <= vo) continue;
      int32_t best_donor = -1, best_w = fj_bin_ddfw_init;
      for (int32_t s = 0; s < fj_bin_ddfw_donor_samples; ++s) {
        const int32_t v  = pb.variables[vo + (int32_t)(rng.next_u32() % (uint32_t)(ve - vo))];
        const int32_t no = pb.reverse_offsets[v], ne = pb.reverse_offsets[v + 1];
        if (ne <= no) continue;
        const int32_t d =
          pb.reverse_constraints[no + (int32_t)(rng.next_u32() % (uint32_t)(ne - no))];
        if (d != cf && !is_violated[d] && rows[d].weight > best_w) {
          best_w     = rows[d].weight;
          best_donor = d;
        }
      }
      if (best_donor >= 0) reweight_constraint(best_donor, rows[best_donor].weight - fj_bin_ddfw_transfer);
    }
    if (violated_list.empty()) objective_weight += 1;
  }

  // Global argmax over every variable, affordable because var_score is maintained live. While the
  // objective weight is zero the full score is exactly var_score, which is the vectorized sweep's
  // precondition; the objective and local-minimum paths fall to the scalar loop.
  std::pair<int32_t, int32_t> find_move_global(bool localmin)
  {
    if (!localmin && objective_weight == 0) {
      // The sweep reads var_score alone; the handful of tabu variables are held at the invalid
      // sentinel across it rather than tested per variable.
      int32_t saved_var[fj_bin_tabu_t::ring_size], saved_score[fj_bin_tabu_t::ring_size];
      const int32_t blocked = tabu.block_tabu(iters, var_score.data(), saved_var, saved_score);

      int32_t v = -1, s = fj_bin_score_invalid;
      fj_bin_argmax(var_score.data(), pb.n_variables, argmax_tile, v, s);

      fj_bin_tabu_t::unblock_tabu(blocked, var_score.data(), saved_var, saved_score);
      return {v, s};
    }

    int32_t best_v = -1, best_s = fj_bin_score_invalid;
    for (int32_t v = 0; v < pb.n_variables; ++v) {
      if (tabu_blocked(v, localmin)) continue;
      const int32_t s = full_score(v, (int8_t)(1 - 2 * assign[v]));
      if (s > best_s) {
        best_s = s;
        best_v = v;
      }
    }
    return {best_v, best_s};
  }

  std::pair<int32_t, int32_t> find_move_in_rows(const std::vector<int32_t>& target_rows,
                                                bool localmin)
  {
    int32_t best_v = -1, best_s = fj_bin_score_invalid;
    for (int32_t r : target_rows) {
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t v = pb.variables[k];
        if (var_bitmap[v]) continue;
        var_bitmap[v] = 1;
        if (tabu_blocked(v, localmin)) continue;
        const int32_t s = full_score(v, (int8_t)(1 - 2 * assign[v]));
        if (s > best_s) {
          best_s = s;
          best_v = v;
        }
      }
    }
    return {best_v, best_s};
  }

  std::pair<int32_t, int32_t> find_move_violated(int32_t sample_size, bool localmin)
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
        const int32_t s = full_score(v, (int8_t)((int8_t)target - assign[v]));
        if (s > move.second) move = {v, s};
      }
    }
    return move;
  }

  std::pair<int32_t, int32_t> find_move_satisfied(int32_t sample_size)
  {
    sample_buf.clear();
    for (int32_t tries = 0; (int32_t)sample_buf.size() < sample_size && tries < sample_size * 8;
         ++tries) {
      const int32_t r = (int32_t)(rng.next_u32() % (uint32_t)pb.n_constraints);
      if (!is_violated[r]) sample_buf.push_back(r);
    }
    return find_move_in_rows(sample_buf, false);
  }

  std::pair<int32_t, int32_t> find_lift_move() const
  {
    int32_t best_v = -1, best_s = 0;
    for (int32_t v : pb.objective_vars) {
      const int8_t delta = (int8_t)(1 - 2 * assign[v]);
      if ((double)delta * pb.objective[v] >= 0) continue;
      if (tabu_blocked(v, false)) continue;
      const int32_t s = (int32_t)(-std::llround(pb.objective[v] * delta)) * fj_bin_score_k;
      if (s > best_s) {
        best_s = s;
        best_v = v;
      }
    }
    return {best_v, best_s};
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
    recompute_lhs();
  }

  // Restart returns the assignment to the seed the climber was constructed with, leaving the
  // recorded best and the global iteration counter intact.
  void do_restart()
  {
    assign = seed_assign;
    for (int32_t v = 0; v < pb.n_variables; ++v) assign_i32[v] = assign[v];
    for (int32_t r = 0; r < pb.n_constraints; ++r) rows[r].weight = pb.initial_weight[r];
    max_weight       = fj_bin_ddfw_init;
    objective_weight = 0;
    tabu.clear(iters);
    recompute_lhs();
    last_restart_iter           = iters;
    last_feasible_entrance_iter = iters;
  }

  void init(fj_cpu_climber_t<i_t, f_t>& climber)
  {
    const auto& params  = climber.settings.parameters;
    seed                = climber.settings.seed;
    narrow4_max         = fj_bin_simd_narrow4_max();
    narrow8_max         = fj_bin_simd_narrow8_max();
    rng                 = raft::random::PCGenerator((uint64_t)seed, 0, 0);
    tabu_tenure_min     = params.tabu_tenure_min;
    tabu_tenure_max     = params.tabu_tenure_max;
    breakthrough_margin = params.breakthrough_move_epsilon;
    perturb_interval    = climber.perturb_interval;
    mtm_viol_samples    = climber.mtm_viol_samples;
    mtm_sat_samples     = climber.mtm_sat_samples;
    if (tabu_tenure_max <= tabu_tenure_min) tabu_tenure_max = tabu_tenure_min + 1;

    // The tabu ring is indexed by iteration modulo its size, so a slot is reused after ring_size
    // iterations. A tenure that long would be overwritten while the variable is still tabu, and the
    // argmax would stop excluding it. Clamped as well as asserted: release builds compile the assert
    // out, and silently dropping tabu entries is worse than a shorter tenure.
    cuopt_assert(tabu_tenure_max <= fj_bin_tabu_t::ring_size,
                 "tabu tenure exceeds the tabu ring, live entries would be evicted");
    if (tabu_tenure_max > fj_bin_tabu_t::ring_size) tabu_tenure_max = fj_bin_tabu_t::ring_size;

    const int32_t n = pb.n_variables, m = pb.n_constraints;
    const auto& h_assign = climber.h_assignment;
    assign.resize(n);
    for (int32_t v = 0; v < n; ++v) {
      const double val = (double)h_assign[v];
      assign[v]        = (int8_t)(val >= 0.5 ? 1 : 0);
    }
    seed_assign = assign;
    best_assign = assign;
    assign_i32.assign(n, 0);
    for (int32_t v = 0; v < n; ++v) assign_i32[v] = assign[v];

    rows.resize(m);
    for (int32_t r = 0; r < m; ++r)
      rows[r] = fj_bin_row_t<coef_t>{0, pb.initial_weight[r], pb.bound[r], pb.cmax[r], pb.sign[r]};

    var_score.assign(n, 0);
    nnz_score_delta.assign(pb.nnz + fj_bin_simd_padding(), 0);
    tabu.resize(n);
    is_violated.assign(m, 0);
    vpos.assign(m, -1);
    violated_list.clear();
    var_bitmap.assign(n, 0);

    objective_weight    = 0;
    max_weight          = fj_bin_ddfw_init;
    incumbent_objective = 0;
    best_objective      = std::numeric_limits<double>::infinity();
    feasible_found      = false;
    iters               = 0;
    last_restart_iter   = 0;
    recompute_lhs();
  }

  void solve(fj_cpu_climber_t<i_t, f_t>& climber, f_t time_limit, double work_unit_limit) override
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
      tabu.maybe_advance(iters);

      int32_t move_var = -1, score = fj_bin_score_invalid;
      if (violated_list.empty()) std::tie(move_var, score) = find_lift_move();
      if (score <= 0) std::tie(move_var, score) = find_move_global(false);
      if (feasible_found && score <= 0) std::tie(move_var, score) = find_move_satisfied(mtm_sat_samples);

      bool perturb_now = false;
      if (violated_list.empty() && iters - last_feasible_entrance_iter > perturb_interval) {
        perturb_now                 = true;
        last_feasible_entrance_iter = iters;
      }

      if (score > 0 && move_var >= 0 && !perturb_now) {
        apply_move(move_var, (int8_t)(1 - 2 * assign[move_var]), climber);
      } else {
        update_weights();
        if (perturb_now) perturb();
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
  }
};

template <typename i_t, typename f_t>
void fj_binary_state_deleter_t<i_t, f_t>::operator()(fj_binary_state_t<i_t, f_t>* ptr) const
{
  delete ptr;
}

template <typename i_t, typename f_t>
void try_build_binary_fastpath(fj_cpu_climber_t<i_t, f_t>& climber)
{
  const fj_bin_scan_t scan = fj_bin_scan(climber);
  if (scan.reject != fj_binary_reject_t::none) {
    CUOPT_LOG_DEBUG("%sCPUFJ binary fast path declined: %s (row %d, var %d)",
                    climber.log_prefix.c_str(),
                    fj_binary_reject_name(scan.reject),
                    scan.bad_row,
                    scan.bad_var);
    return;
  }

  bool built = false;
  if (scan.coefficient_bits == 8) {
    auto engine = std::make_unique<fj_bin_engine_t<i_t, f_t, int8_t>>();
    built       = fj_bin_narrow(climber, scan.n_split_constraints, engine->pb);
    if (built) climber.binary_fast.reset(engine.release());
  } else {
    auto engine = std::make_unique<fj_bin_engine_t<i_t, f_t, int16_t>>();
    built       = fj_bin_narrow(climber, scan.n_split_constraints, engine->pb);
    if (built) climber.binary_fast.reset(engine.release());
  }

  if (!built) {
    CUOPT_LOG_DEBUG("%sCPUFJ binary fast path declined: %s",
                    climber.log_prefix.c_str(),
                    fj_binary_reject_name(fj_binary_reject_t::narrow_check_failed));
    return;
  }

  CUOPT_LOG_DEBUG("%sCPUFJ binary fast path enabled: int%d coefficients, %d rows after one-sided split",
                  climber.log_prefix.c_str(),
                  scan.coefficient_bits,
                  scan.n_split_constraints);
}

#if MIP_INSTANTIATE_FLOAT
template struct fj_binary_state_deleter_t<int, float>;
template void try_build_binary_fastpath(fj_cpu_climber_t<int, float>& climber);
#endif

#if MIP_INSTANTIATE_DOUBLE
template struct fj_binary_state_deleter_t<int, double>;
template void try_build_binary_fastpath(fj_cpu_climber_t<int, double>& climber);
#endif

}  // namespace cuopt::mathematical_optimization::mip
