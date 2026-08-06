/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <memory>

// Seam between the general CPU FJ path and the binary fast path. Only fj_cpu.cu includes this;
// fj_cpu.cuh forward-declares fj_binary_state_t so the climber can hold one without the general
// header depending on the fast path.
//
// The fast path applies to instances whose variables are all binary and whose rows carry integer
// coefficients within int8 or int16 range. On those it runs an integer engine: exact feasibility
// against a single row bound, a live per-variable score patched through stored per-nnz
// contributions, and a global argmax move selection.

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;

// Why an instance was refused, logged at DEBUG by the build entry.
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

// Built state for one eligible instance. The concrete engine is templated on coefficient width
// (int8_t or int16_t) and lives in fj_cpu_binary.cu; the width choice is made once at build time,
// so dispatching through this base costs one virtual call per solve.
template <typename i_t, typename f_t>
struct fj_binary_state_t {
  virtual ~fj_binary_state_t() = default;

  virtual void solve(fj_cpu_climber_t<i_t, f_t>& climber,
                     f_t time_limit,
                     double work_unit_limit) = 0;

  virtual int coefficient_bits() const   = 0;  // 8 or 16
  virtual int n_split_constraints() const = 0;
  virtual i_t iterations() const          = 0;

  // Largest per-variable aggregate base and bonus observed at end of solve. The packed score is
  // order-preserving only while these stay under 2^16 and 2^14 respectively.
  virtual void saturation(int& max_aggregate_base, int& max_aggregate_bonus) const = 0;
};

// Runs predicate -> one-sided split -> narrow. A pure function of the climber's host problem
// mirrors and its incoming constraint weights, so it is well defined wherever those are populated.
// Populates climber.binary_fast on success; leaves it empty and logs the reason otherwise.
template <typename i_t, typename f_t>
void try_build_binary_fastpath(fj_cpu_climber_t<i_t, f_t>& climber);

// ---------------------------------------------------------------------------------------------
// Hot kernels and the score-delta formula they share with the engine.
//
// The kernels live in fj_cpu_binary_kernels.cpp, built with Google Highway, which compiles the
// bodies once per SIMD target and dispatches at runtime. That file is host-compiled: nvcc's
// frontend rejects Highway's x86 headers, which reinterpret-cast intrinsic vectors to
// compiler-specific vector types (GCC vector extensions in the constant-folding path, __m128bh
// for bfloat16). Every argument below is a plain pointer or scalar, so the seam names no cuOpt or
// CUDA type.
// ---------------------------------------------------------------------------------------------

// Packed staged score: one int32 holding base * K + bonus. K exceeds twice the largest |bonus| the
// engine produces, so integer ordering on the packed word reproduces the lexicographic (base,
// bonus) ordering the general path gets from fj_staged_score_t.
constexpr int32_t fj_bin_score_shift   = 15;
constexpr int32_t fj_bin_score_k       = 1 << fj_bin_score_shift;
constexpr int32_t fj_bin_score_invalid = INT32_MIN;

// Change in one row's weighted score when one variable flips, from the row's signed slack before
// (os) and after (ns) that flip. base is the weighted change in satisfaction; bonus is the
// weighted change in strict slack. When both states are violated the improving direction earns
// half weight, matching excess_improvement_weight of 1/2.
//
// Single source of the formula: the engine scores moves with it, compute_saturation walks it, and
// the vector kernels reproduce it lane-wise.
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

// Elements of padding the per-nnz arrays must carry past nnz, in int32 units. A target that masks
// its row remainder loads and stores a whole vector at the last row of the matrix, and the padding
// is what keeps that off memory it does not own. A target that peels the remainder into a scalar
// tail stops at the row end and asks for nothing, so this returns 0 there.
int32_t fj_bin_simd_padding();

// Vector width the row patch runs at. A gather costs the same whether its lanes carry data or are
// masked off, so a row filling only part of a native vector is cheaper through a narrower one; past
// the crossover the extra vector and its extra full gather cost more than the idle lanes.
enum class fj_bin_patch_width_t : int32_t { narrow4 = 0, narrow8 = 1, native = 2 };

// Longest row for which each narrower width beats the native one, or 0 where that width is not
// worth offering on this target: a width is offered only when it is strictly narrower than the
// native vector, and scalable targets decline both.
int32_t fj_bin_simd_narrow4_max();
int32_t fj_bin_simd_narrow8_max();

static inline fj_bin_patch_width_t fj_bin_patch_width_for(int32_t row_len,
                                                          int32_t narrow4_max,
                                                          int32_t narrow8_max)
{
  if (row_len <= narrow4_max) return fj_bin_patch_width_t::narrow4;
  if (row_len <= narrow8_max) return fj_bin_patch_width_t::narrow8;
  return fj_bin_patch_width_t::native;
}

// Patch every variable of one row against the row's current signed slack, skipping skip_var. The
// move case passes the post-move slack and the flipped variable's index; the reweight case passes
// the unchanged slack and -1, which matches no variable index.
//
// Reads and writes up to fj_bin_simd_padding() elements from kb, so variables, coefficients and
// nnz_score_delta must carry that padding.
//
// Defined in the host-compiled kernels TU and explicitly instantiated there for int8_t and int16_t.
template <typename coef_t>
void fj_bin_patch_row(fj_bin_patch_width_t width,
                      const int32_t* variables,
                      const coef_t* coefficients,
                      int32_t kb,
                      int32_t ke,
                      int32_t* var_score,
                      int32_t* nnz_score_delta,
                      const int32_t* assign_i32,
                      int32_t sign,
                      int32_t weight,
                      int32_t os_new,
                      int32_t skip_var);

// Argmax over var_score with the tabu window folded in, scanning all n variables. Valid while the
// objective weight is zero, where the full score is exactly var_score. Yields best_var of -1 when
// every variable is tabu.
void fj_bin_argmax(const int32_t* var_score,
                   const uint16_t* flip_until,
                   int32_t n,
                   uint16_t iter_biased,
                   int32_t tile,
                   int32_t& best_var,
                   int32_t& best_score);

}  // namespace cuopt::mathematical_optimization::mip
