/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>

// The fast path applies to instances whose variables are all binary and whose rows carry integer
// coefficients within int8 or int16 range. On those it runs a SIMD integer engine: exact feasibility
// against a single row bound, a live per-variable score patched through stored per-nnz
// contributions, and a global argmax move selection.

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;

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

// Returns true if the fast path ran (eligible and narrowed); false if declined, in which case the caller should take the general path.
// TODO: worth revisiting if the same climber is solved repeatedly to cache the fastpath state
template <typename i_t, typename f_t>
bool try_cpufj_binary_solve(fj_cpu_climber_t<i_t, f_t>& climber,
                            f_t time_limit,
                            double work_unit_limit);

// Packed staged score: one int32 holding base * K + bonus
constexpr int32_t fj_bin_score_shift   = 15;
constexpr int32_t fj_bin_score_k       = 1 << fj_bin_score_shift;
constexpr int32_t fj_bin_score_invalid = INT32_MIN;

// Change in one row's weighted score when one variable flips, from the row's signed slack before
// (os) and after (ns) that flip. base is the weighted change in satisfaction; bonus is the
// weighted change in strict slack. When both states are violated the improving direction earns
// half weight, matching excess_improvement_weight of 1/2.
// purpose: implements the scoring delta logic from feasibility_jump.cuh in a form easier to port to SIMD
static inline void fj_bin_score_delta_parts(
  int32_t os, int32_t ns, int32_t weight, int32_t& base, int32_t& bonus)
{
  const int32_t osat = os >= 0, nsat = ns >= 0;
  const int32_t ost = os > 0, nst = ns > 0;
  const int32_t improving = (os < ns) - (ns < os);
  base  = weight * (nsat - osat) + (1 - osat) * (1 - nsat) * improving * (weight / 2);
  bonus = weight * (nst - ost);
}

static inline int32_t fj_bin_packed_score_delta(int32_t os, int32_t ns, int32_t weight)
{
  int32_t base = 0, bonus = 0;
  fj_bin_score_delta_parts(os, ns, weight, base, bonus);
  return base * fj_bin_score_k + bonus;
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
                      int32_t* var_score,
                      int32_t* nnz_score_delta,
                      const int32_t* assign_i32,
                      int32_t weight,
                      int32_t os_new,
                      int32_t skip_var);

constexpr int32_t fj_bin_walk_tile = 256;

// Advance every row incident to one flipped variable within apply_move, and report which of those visits the caller
// must finish by hand (e.g. if the row needs patching)
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
// Tabu is handled by "blocking" the scores corresponding to the tabu vars, and restoring them after the argmax
// affordable since max_tenure is small
void fj_bin_argmax(const int32_t* var_score,
                   int32_t n,
                   int32_t tile,
                   int32_t& best_var,
                   int32_t& best_score);

}  // namespace cuopt::mathematical_optimization::mip
