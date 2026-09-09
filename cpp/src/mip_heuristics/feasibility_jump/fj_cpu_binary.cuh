/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <math_optimization/tic_toc.hpp>

#include <cstdint>
#include <vector>

// The fast path applies to instances whose variables are all binary and whose rows carry integer
// coefficients within int8 or int16 range. On those it runs a SIMD integer engine: exact
// feasibility against a single row bound, a live per-variable score patched through stored per-nnz
// contributions, and a global argmax move selection.

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;

// Bounds on the per-variable objective multiplier, in units of the mean absolute nonzero objective
// coefficient, so one large coefficient cannot dominate the feasibility term.
inline constexpr double fj_obj_mult_min = 0.25;
inline constexpr double fj_obj_mult_max = 4.0;

// Wall seconds of each phase of the binary engine's setup. Charged whether or not the fast path is
// taken, so an attempt that scans the problem and then declines is visible rather than free.
struct fj_bin_setup_times_t {
  double scan{0};
  double narrow{0};
  double transpose{0};
  double encode{0};
  double engine_init{0};

  double total() const { return scan + narrow + transpose + encode + engine_init; }
};

// Adds one setup phase's wall time to a scalar on the climber. These phases run once per solve and
// only their accumulated total matters.
class phase_timer_t {
 public:
  explicit phase_timer_t(double& sink) : sink_(sink), started_(tic()) {}
  ~phase_timer_t() { sink_ += toc(started_); }

  phase_timer_t(const phase_timer_t&)            = delete;
  phase_timer_t& operator=(const phase_timer_t&) = delete;

 private:
  double& sink_;
  double started_;
};

enum class fj_binary_reject_t : uint8_t {
  none,
  empty_problem,
  non_binary_var,
  fractional_coefficient,
  coefficient_out_of_range,
  fractional_row_bound,
  row_bound_out_of_range,
  lhs_headroom,
  narrow_check_failed,
};
const char* fj_binary_reject_name(fj_binary_reject_t reason);

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

  // Empty unless encoded, when every engine variable is one bit of a bounded general integer and
  // original[j] = var_offset[j] + sum of bit_weight[b] * assign[b] over the bits b owned by j.
  bool encoded{false};
  int32_t n_original{0};
  std::vector<double> var_offset;
  std::vector<int32_t> bit_owner;
  std::vector<int32_t> original_to_bin_mapping;
  std::vector<double> bit_weight;
  std::vector<double> orig_objective;
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

constexpr int32_t fj_bin_ddfw_init = 10;  // initial weight, also the donation floor

static inline bool fj_bin_in_int32(double v)
{
  return v >= (double)INT32_MIN && v <= (double)INT32_MAX;
}

template <typename i_t, typename f_t>
fj_bin_scan_t fj_bin_scan(const fj_cpu_climber_t<i_t, f_t>& c, fj_bin_setup_times_t& times);

template <typename i_t, typename f_t, typename coef_t>
bool fj_bin_narrow(const fj_cpu_climber_t<i_t, f_t>& c,
                   const fj_bin_scan_t& scan,
                   fj_bin_problem_t<coef_t>& pb,
                   fj_bin_setup_times_t& times);

template <typename i_t, typename f_t, typename coef_t>
bool fj_bin_encode(const fj_cpu_climber_t<i_t, f_t>& c,
                   fj_bin_problem_t<coef_t>& pb,
                   int& coefficient_bits,
                   fj_bin_setup_times_t& times);

// Returns true if the fast path ran (eligible and narrowed); false if declined, in which case the
// caller should take the general path.
// TODO: worth revisiting if the same climber is solved repeatedly to cache the fastpath state
template <typename i_t, typename f_t>
bool try_cpufj_binary_solve(fj_cpu_climber_t<i_t, f_t>& climber,
                            f_t time_limit,
                            double work_unit_limit);

// Packed staged score: one int64 holding base * K + bonus, encoding the general path's
// lexicographic (base, bonus) comparison as a single arithmetic one.
//
// The width is what makes the encoding faithful. Both fields aggregate over the rows a variable
// appears in, so each is bounded by max_var_degree * max_weight -- unbounded above at build time,
// since DDFW grows the weights. At 15 bits the bonus field overflowed into the base on real
// instances (chromaticindex1024-7 reaches an aggregate bonus of 122880 against 16384), which
// silently corrupts the ordering the argmax depends on. 32 bits leaves the base free to use the
// whole int32 range before the encoding can break.
constexpr int32_t fj_bin_score_shift   = 32;
constexpr int64_t fj_bin_score_k       = (int64_t)1 << fj_bin_score_shift;
constexpr int64_t fj_bin_score_invalid = INT64_MIN;

// Change in one row's weighted score when one variable flips, from the row's signed slack before
// (os) and after (ns) that flip. base is the weighted change in satisfaction; bonus is the
// weighted change in strict slack. When both states are violated the improving direction earns
// half weight, matching excess_improvement_weight of 1/2.
// purpose: implements the scoring delta logic from feasibility_jump.cuh in a form easier to port to
// SIMD
static inline void fj_bin_score_delta_parts(
  int32_t os, int32_t ns, int32_t weight, int32_t& base, int32_t& bonus)
{
  const int32_t osat = os >= 0, nsat = ns >= 0;
  const int32_t ost = os > 0, nst = ns > 0;
  const int32_t improving = (os < ns) - (ns < os);
  base  = weight * (nsat - osat) + (1 - osat) * (1 - nsat) * improving * (weight / 2);
  bonus = weight * (nst - ost);
}

static inline int64_t fj_bin_packed_score_delta(int32_t os, int32_t ns, int32_t weight)
{
  int32_t base = 0, bonus = 0;
  fj_bin_score_delta_parts(os, ns, weight, base, bonus);
  return (int64_t)base * fj_bin_score_k + bonus;
}

// Padding margin to prevent faults on tail SIMD loads
constexpr int32_t fj_bin_simd_padding = 256;

// Patch every variable of one row against the row's current signed slack. The
// move case passes the post-move slack and the flipped variable's index; the reweight case passes
// the unchanged slack and -1, which matches no variable index.
template <typename coef_t>
void fj_bin_patch_row(const int32_t* variables,
                      const coef_t* coefficients,
                      int32_t kb,
                      int32_t ke,
                      int64_t* var_score,
                      int64_t* nnz_score_delta,
                      const int32_t* assign_i32,
                      int32_t weight,
                      int32_t os_new,
                      int32_t skip_var);

constexpr int32_t fj_bin_walk_tile = 256;

// Advance every row incident to one flipped variable within apply_move, and report which of those
// visits the caller must finish by hand (e.g. if the row needs patching)
//
// For every incidence i in the range this applies
//   row_slack[incident_row[i]] -= reverse_coefficients[i] * delta
// then writes to out_incidence, in increasing order, the subset of i whose row is not deeply
// satisfied on both sides of the flip and returns how many.
template <typename coef_t>
int32_t fj_bin_walk_rows(int32_t* row_slack,
                         const int32_t* incident_row,
                         const coef_t* reverse_coefficients,
                         const coef_t* incident_row_cmax,
                         int32_t incidence_begin,
                         int32_t incidence_end,
                         int32_t delta,
                         int32_t* out_incidence);

// Argmax over var_score, scanning all n variables. Valid while the objective weight is zero, where
// the full score is exactly var_score. Yields best_var of -1 only if n is 0.
// Tabu is handled by "blocking" the scores corresponding to the tabu vars, and restoring them after
// the argmax affordable since max_tenure is small
void fj_bin_argmax(
  const int64_t* var_score, int32_t n, int32_t tile, int32_t& best_var, int64_t& best_score);

// combined[v] = var_score[v] + obj_score[v] over n variables, which is the full score once the
// objective weight is nonzero. The three arrays must not overlap.
void fj_bin_add_scores(const int64_t* var_score,
                       const int64_t* obj_score,
                       int32_t n,
                       int64_t* combined);

}  // namespace cuopt::mathematical_optimization::mip
