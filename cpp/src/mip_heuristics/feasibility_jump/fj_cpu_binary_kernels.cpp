/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Hot kernels of the binary CPU FJ fast path, vectorized with Google Highway. foreach_target.h
// re-includes this file once per SIMD target; HWY_EXPORT builds the dispatch table and
// HWY_DYNAMIC_DISPATCH picks at runtime. Host-compiled rather than nvcc-compiled: nvcc's frontend
// rejects Highway's x86 headers, which reinterpret-cast intrinsic vectors to compiler-specific
// vector types.

#include <mip_heuristics/feasibility_jump/fj_cpu_binary.cuh>

#include <cstddef>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mip_heuristics/feasibility_jump/fj_cpu_binary_kernels.cpp"
#include "hwy/foreach_target.h"  // must precede highway.h
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace cuopt::mathematical_optimization::mip {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Whether the row remainder is masked into the vector body or peeled into a scalar tail. AVX-512
// k-registers, SVE predicates and RVV masks make every operation maskable at no cost, so a row of
// three nonzeros is one masked iteration; peeling it would send most of the work to the tail, since
// row lengths are short and have nothing to do with the lane count. AVX2 and NEON have no mask
// registers: the mask becomes a vector, gather and scatter are emulated, and the tail is cheaper.
// Measured on AVX2, masking the remainder cost 6.8% on supportcase22 and 12.9% on bnatt400.
constexpr bool k_mask_remainder =
  (HWY_TARGET <= HWY_AVX3) || HWY_TARGET_IS_SVE || (HWY_TARGET == HWY_RVV);

// Whether the row walk below is worth vectorizing on this target. It needs a real gather to read the
// slacks and a real compress to emit the tail list; where either is emulated the emulation costs
// more than the scalar loop it replaces, since 85% of visits do nothing but subtract and compare.
// The scalar arm still returns the same list, so the caller needs no second code path -- it pays
// only one store per reported visit.
constexpr bool k_vector_walk =
  (HWY_TARGET <= HWY_AVX3) || HWY_TARGET_IS_SVE || (HWY_TARGET == HWY_RVV);

// One tile of a flipped variable's incidence range, vectorized. The caller tiles the range and runs
// each tile's tail before asking for the next; see fj_bin_walk_tile.
//
// Measured on supportcase22: 84.87% of row visits leave the row deeply satisfied on both sides of
// the flip, and those visits do nothing but update the slack. The remaining 15.13% need the row's
// weight, the flipped variable's own score delta, the violated-set transitions and usually a
// patch -- all indirect, all awkward in a vector. So this kernel does only the uniform part and
// hands back the indices of the visits that are not deep_sat, in increasing order, for the caller
// to finish scalar.
//
// The layout this assumes is what makes it worth doing. Storing the row's signed slack rather than
// its lhs collapses the update to
//
//   new_slack = old_slack - coef * delta
//
// so bound and lhs never appear, and the coefficient is the only per-incidence constant. It and
// cmax are replicated per incidence, which makes them unit-stride loads. What remains irregular is
// the slack itself: one gather and one scatter per vector, against four gathers and a scatter for a
// literal SoA split of the row record.
//
// Trajectory is preserved exactly. The slack update is per row and order-independent; the caller's
// tail visits its indices in the same order the scalar loop did; and a deep_sat row is never read by
// the tail, so updating it early is not observable.
template <typename coef_t>
int32_t WalkRowsImpl(int32_t* HWY_RESTRICT row_slack,
                     const int32_t* HWY_RESTRICT incident_row,
                     const coef_t* HWY_RESTRICT reverse_coefficients,
                     const coef_t* HWY_RESTRICT incident_row_cmax,
                     int32_t incidence_begin,
                     int32_t incidence_end,
                     int32_t delta,
                     int32_t* HWY_RESTRICT out_incidence)
{
  int32_t n_out = 0;
  int32_t ii    = incidence_begin;

  if constexpr (k_vector_walk) {
    const hn::ScalableTag<int32_t> d;
    const hn::Rebind<coef_t, decltype(d)> dc;  // same lane count, narrower lanes
    using V        = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);

    const V vdelta = hn::Set(d, delta);

    // The unit-stride loads always run whole and read into the per-incidence padding; FirstN keeps
    // the overhang out of the gather, the scatter and the compress.
    for (; ii < incidence_end; ii += (int32_t)N) {
      const auto active = hn::FirstN(d, (size_t)(incidence_end - ii));

      const V rows = hn::LoadU(d, incident_row + ii);
      const V skv  = hn::PromoteTo(d, hn::LoadU(dc, reverse_coefficients + ii));
      const V cmax = hn::PromoteTo(d, hn::LoadU(dc, incident_row_cmax + ii));

      const V os = hn::MaskedGatherIndex(active, d, row_slack, rows);
      // os - skv * vdelta
      const V ns = hn::NegMulAdd(skv, vdelta, os);

      // Only the satisfied side. deep_viol is the caller's business: it fires on 0.02% of visits but
      // guards the widest rows in the matrix, so it belongs where the row length is already known.
      const auto deep_sat = hn::And(hn::Gt(os, cmax), hn::Gt(ns, cmax));
      const auto to_tail  = hn::AndNot(deep_sat, active);

#if HWY_TARGET == HWY_AVX3_ZEN4
      // Same Zen 4 microcode argument as the score scatter in PatchRowBody: VPSCATTERDD is 89 uops
      // at ~24 CPI, against two vector stores and N scalar stores here. Unlike that one this is a
      // pure store with no read-modify-write, so it needs its own A/B before the arm is settled.
      HWY_ALIGN int32_t row_lane[hn::MaxLanes(d)], slack_lane[hn::MaxLanes(d)];
      hn::Store(rows, d, row_lane);
      hn::Store(ns, d, slack_lane);
      const size_t lanes = HWY_MIN(N, (size_t)(incidence_end - ii));
      for (size_t i = 0; i < lanes; ++i) row_slack[row_lane[i]] = slack_lane[i];
#else
      hn::MaskedScatterIndex(ns, active, d, row_slack, rows);
#endif

      // A variable meets each row at most once, so no two lanes carry the same row and neither the
      // scatter above nor the store loop needs conflict detection.
      n_out += (int32_t)hn::CompressStore(hn::Iota(d, ii), to_tail, d, out_incidence + n_out);
    }
    return n_out;
  }

  // Targets without a native gather or compress. Also the remainder is not reached here: the loop
  // above runs to oe under FirstN, and this arm replaces it wholesale rather than tailing it.
  for (; ii < incidence_end; ++ii) {
    const int32_t row  = incident_row[ii];
    const int32_t os   = row_slack[row];
    const int32_t ns   = os - (int32_t)reverse_coefficients[ii] * delta;
    row_slack[row]     = ns;
    const int32_t cmax = (int32_t)incident_row_cmax[ii];
    if (!(os > cmax && ns > cmax)) out_incidence[n_out++] = ii;
  }
  return n_out;
}

// Row remainder when it is peeled rather than masked, and the whole row on scalar targets.
template <typename coef_t>
void PatchRowScalar(const int32_t* HWY_RESTRICT variables,
                    const coef_t* HWY_RESTRICT coefficients,
                    int32_t kb,
                    int32_t ke,
                    int64_t* HWY_RESTRICT var_score,
                    int64_t* HWY_RESTRICT nnz_score_delta,
                    const int32_t* HWY_RESTRICT assign_i32,
                    int32_t weight,
                    int32_t os_new,
                    int32_t skip_var)
{
  for (int32_t k = kb; k < ke; ++k) {
    const int32_t v = variables[k];
    if (v == skip_var) continue;
    const int32_t flip = 1 - 2 * assign_i32[v];
    const int32_t ns   = os_new - (int32_t)coefficients[k] * flip;
    const int64_t nc   = fj_bin_packed_score_delta(os_new, ns, weight);
    var_score[v] += nc - nnz_score_delta[k];
    nnz_score_delta[k] = nc;
  }
}

// Templated on the vector tag so one body serves both the native-width kernel and the narrow one.
// Rows here average well under a native 512-bit vector, and a gather costs the same whether its
// lanes are used or discarded, so short rows are cheaper through a narrower vector.
template <typename coef_t, class D>
static HWY_INLINE void PatchRowBody(D d,
                                    const int32_t* HWY_RESTRICT variables,
                                    const coef_t* HWY_RESTRICT coefficients,
                                    int32_t kb,
                                    int32_t ke,
                                    int64_t* HWY_RESTRICT var_score,
                                    int64_t* HWY_RESTRICT nnz_score_delta,
                                    const int32_t* HWY_RESTRICT assign_i32,
                                    int32_t weight,
                                    int32_t os_new,
                                    int32_t skip_var)
{
  const hn::Rebind<coef_t, D> dc;        // same lane count, narrower lanes
  const hn::Repartition<int64_t, D> dw;  // half the lanes, twice as wide: the packed score
  using V         = hn::Vec<decltype(d)>;
  using VW        = hn::Vec<decltype(dw)>;
  const size_t N  = hn::Lanes(d);
  const size_t NW = hn::Lanes(dw);

  // When the remainder is peeled, a row below one vector never reaches the body, so it skips the
  // ten broadcasts below as well.
  if constexpr (!k_mask_remainder) {
    if ((size_t)(ke - kb) < N) {
      PatchRowScalar<coef_t>(variables, coefficients, kb, ke, var_score, nnz_score_delta,
                             assign_i32, weight, os_new, skip_var);
      return;
    }
  }

  const V vone = hn::Set(d, 1), vzero = hn::Zero(d);
  const V vskip = hn::Set(d, skip_var);
  const V vos = hn::Set(d, os_new);
  const V vw = hn::Set(d, weight), vw2 = hn::Set(d, weight / 2);

  // The row's own slack is uniform across lanes, so its flags are scalars. Broadcast negated to
  // match the new-state flags below, which come from VecFromMask and are 0 or -1.
  const int32_t osat = os_new >= 0, ost = os_new > 0;
  const V vneg_osat = hn::Set(d, -osat), vneg_ost = hn::Set(d, -ost);
  const V v_not_osat = hn::Set(d, 1 - osat);

  // The loads always run unmasked and read into the per-nnz padding; when the remainder is masked,
  // FirstN keeps the overhang out of the gather, the scatter and the store.
  const int32_t vec_end = k_mask_remainder ? ke : ke - (int32_t)N + 1;
  int32_t k             = kb;
  for (; k < vec_end; k += (int32_t)N) {
    const V v   = hn::LoadU(d, variables + k);
    auto active = hn::Ne(v, vskip);
    if constexpr (k_mask_remainder) {
      active = hn::And(active, hn::FirstN(d, (size_t)(ke - k)));
    }

    // Gathered in hardware even on Zen 4, unlike the score update below. Doing this one by lane
    // instead measured 8.2% slower: it must spill the index vector and reload it 4 bytes at a time,
    // which cannot store-to-load forward, and that cost 959 interlocks per iteration against 72.
    // The score update escapes this because it already needs the spill for its read-modify-write.
    const V a01  = hn::MaskedGatherIndex(active, d, assign_i32, v);
    const V flip = hn::Sub(vone, hn::ShiftLeft<1>(a01));
    const V coef = hn::PromoteTo(d, hn::LoadU(dc, coefficients + k));

    // vos - coef * flip
    const V ns = hn::NegMulAdd(coef, flip, vos);

    // -(ns >= 0)
    const V nsat_neg = hn::VecFromMask(d, hn::Ge(ns, vzero));
    // -(ns > 0)
    const V nst_neg = hn::VecFromMask(d, hn::Gt(ns, vzero));
    // (ns > vos) - (ns < vos)
    const V improving =
      hn::Sub(hn::VecFromMask(d, hn::Lt(ns, vos)), hn::VecFromMask(d, hn::Gt(ns, vos)));

    // (1 - osat) * (1 - nsat)
    const V both_violated = hn::Mul(v_not_osat, hn::Add(vone, nsat_neg));
    // vw * (nsat - osat) + both_violated * improving * vw2
    const V base =
      hn::MulAdd(vw, hn::Sub(vneg_osat, nsat_neg), hn::Mul(hn::Mul(both_violated, improving), vw2));
    // vw * (nst - ost)
    const V bonus = hn::Mul(vw, hn::Sub(vneg_ost, nst_neg));

    // The score is int64, so packing it costs two vectors where the fields took one. Both fields
    // are per-row here and fit int32, so they are computed at full lane count above and widened
    // only for the pack. Everything below stays in the vector: the pack, the old value, the
    // difference and the store back. What reaches the scalar loop is one add per nonzero, which is
    // what it was before the score widened -- that loop is 38% of all cycles, so work belongs
    // anywhere but there.
    const VW base_lo  = hn::PromoteLowerTo(dw, base);
    const VW base_hi  = hn::PromoteUpperTo(dw, base);
    const VW bonus_lo = hn::PromoteLowerTo(dw, bonus);
    const VW bonus_hi = hn::PromoteUpperTo(dw, bonus);

    const VW packed_lo = hn::Add(hn::ShiftLeft<fj_bin_score_shift>(base_lo), bonus_lo);
    const VW packed_hi = hn::Add(hn::ShiftLeft<fj_bin_score_shift>(base_hi), bonus_hi);

    const VW delta_lo = hn::Sub(packed_lo, hn::LoadU(dw, nnz_score_delta + k));
    const VW delta_hi = hn::Sub(packed_hi, hn::LoadU(dw, nnz_score_delta + k + NW));

    // The store mask is rebuilt at int64 width rather than narrowed from `active`: the same two
    // conditions, on the promoted indices. FirstN is applied on every target because where the
    // remainder is peeled the body never runs short, so it is all-true there anyway.
    const size_t rem  = (size_t)(ke - k);
    const VW v_lo     = hn::PromoteLowerTo(dw, v);
    const VW v_hi     = hn::PromoteUpperTo(dw, v);
    const VW vskip_w  = hn::Set(dw, skip_var);
    const auto act_lo = hn::And(hn::Ne(v_lo, vskip_w), hn::FirstN(dw, rem));
    const auto act_hi = hn::And(hn::Ne(v_hi, vskip_w), hn::FirstN(dw, rem > NW ? rem - NW : 0));
    hn::BlendedStore(packed_lo, act_lo, dw, nnz_score_delta + k);
    hn::BlendedStore(packed_hi, act_hi, dw, nnz_score_delta + k + NW);

#if HWY_TARGET == HWY_AVX3_ZEN4
    // zmm VSIB is microcode on Zen 4: VPGATHERDD ~76-80 uops / ~21 CPI and VPSCATTERDD 89 / 24,
    // against ~5 / ~10 and ~19 / ~11 on SPR-class Intel (Agner Fog, uops.info). So read-modify-write
    // by lane here; measured +5.8% over the arm below on an EPYC 9554 (supportcase22, 16 climbers).
    HWY_ALIGN int32_t idx[hn::MaxLanes(d)];
    HWY_ALIGN int64_t dl[hn::MaxLanes(d)];
    hn::Store(v, d, idx);
    hn::Store(delta_lo, dw, dl);
    hn::Store(delta_hi, dw, dl + NW);
    // Bounded by the row, not the vector: the lanes past it hold padding, whose zero index would
    // otherwise be applied to variable 0.
    const size_t lanes = HWY_MIN(N, (size_t)(ke - k));
    for (size_t i = 0; i < lanes; ++i) {
      if (idx[i] != skip_var) var_score[idx[i]] += dl[i];
    }
#else
    // The score is int64, so the gather and scatter run at the promoted width against the promoted
    // indices, in the two halves the pack already produced.
    const VW cur_lo = hn::MaskedGatherIndex(act_lo, dw, var_score, v_lo);
    const VW cur_hi = hn::MaskedGatherIndex(act_hi, dw, var_score, v_hi);
    hn::MaskedScatterIndex(hn::Add(cur_lo, delta_lo), act_lo, dw, var_score, v_lo);
    hn::MaskedScatterIndex(hn::Add(cur_hi, delta_hi), act_hi, dw, var_score, v_hi);
#endif
  }

  if constexpr (!k_mask_remainder) {
    PatchRowScalar<coef_t>(variables, coefficients, k, ke, var_score, nnz_score_delta, assign_i32,
                           weight, os_new, skip_var);
  }
}

// Native width, and the 8-lane variant for rows that would leave most of a native vector idle.
template <typename coef_t>
void PatchRowImpl(const int32_t* HWY_RESTRICT variables,
                  const coef_t* HWY_RESTRICT coefficients,
                  int32_t kb,
                  int32_t ke,
                  int64_t* HWY_RESTRICT var_score,
                  int64_t* HWY_RESTRICT nnz_score_delta,
                  const int32_t* HWY_RESTRICT assign_i32,
                  int32_t weight,
                  int32_t os_new,
                  int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::ScalableTag<int32_t>(), variables, coefficients, kb, ke, var_score,
                       nnz_score_delta, assign_i32, weight, os_new, skip_var);
}

template <typename coef_t>
void PatchRowNarrow8Impl(const int32_t* HWY_RESTRICT variables,
                         const coef_t* HWY_RESTRICT coefficients,
                         int32_t kb,
                         int32_t ke,
                         int64_t* HWY_RESTRICT var_score,
                         int64_t* HWY_RESTRICT nnz_score_delta,
                         const int32_t* HWY_RESTRICT assign_i32,
                         int32_t weight,
                         int32_t os_new,
                         int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::CappedTagIfFixed<int32_t, 8>(), variables, coefficients, kb, ke,
                       var_score, nnz_score_delta, assign_i32, weight, os_new, skip_var);
}

template <typename coef_t>
void PatchRowNarrow4Impl(const int32_t* HWY_RESTRICT variables,
                         const coef_t* HWY_RESTRICT coefficients,
                         int32_t kb,
                         int32_t ke,
                         int64_t* HWY_RESTRICT var_score,
                         int64_t* HWY_RESTRICT nnz_score_delta,
                         const int32_t* HWY_RESTRICT assign_i32,
                         int32_t weight,
                         int32_t os_new,
                         int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::CappedTagIfFixed<int32_t, 4>(), variables, coefficients, kb, ke,
                       var_score, nnz_score_delta, assign_i32, weight, os_new, skip_var);
}

// Longest row worth sending to each narrower kernel, or 0 where that width is not worth having.
// A gather costs the same whether its lanes carry data or are masked off, so a row that fills only
// part of a native vector is cheaper through a narrower one; past the crossover the extra vector
// and its extra full gather cost more than the wasted lanes. From the Zen 4 microcode ratio
// (VPGATHERDD ~78 uops at 512 bits, 48 at 256, 24 at 128) the crossovers land at 4 and 8.
//
// A width is offered only when it is strictly narrower than the native vector, so no target ever
// dispatches to a kernel identical to its own. Scalable targets opt out entirely: Highway notes
// that clamping Lanes() on RVV/SVE can cost more than the capping saves, which is why
// CappedTagIfFixed leaves them at native width above.
//
// These are per-target constants, so the width choice belongs here rather than at the call seam: a
// caller outside this file can only reach them through a dispatch pointer, which turns two
// immediates into two loads of runtime globals and puts an unpredictable branch directly in front
// of the indirect jump that follows it. Measured on supportcase22, that seam cost 2.4%.
constexpr size_t k_native_lanes  = HWY_MAX_LANES_D(hn::ScalableTag<int32_t>);
constexpr int32_t k_narrow4_max  = HWY_HAVE_SCALABLE ? 0 : (k_native_lanes > 4 ? 4 : 0);
constexpr int32_t k_narrow8_max  = HWY_HAVE_SCALABLE ? 0 : (k_native_lanes > 8 ? 8 : 0);

// Single entry point the seam dispatches to. On a scalable target both bounds are 0, so both
// compares fold away and the narrow arms are stripped.
template <typename coef_t>
void PatchRowDispatchImpl(const int32_t* HWY_RESTRICT variables,
                          const coef_t* HWY_RESTRICT coefficients,
                          int32_t kb,
                          int32_t ke,
                          int64_t* HWY_RESTRICT var_score,
                          int64_t* HWY_RESTRICT nnz_score_delta,
                          const int32_t* HWY_RESTRICT assign_i32,
                          int32_t weight,
                          int32_t os_new,
                          int32_t skip_var)
{
  const int32_t row_len = ke - kb;
  if (row_len <= k_narrow4_max) {
    PatchRowNarrow4Impl<coef_t>(variables, coefficients, kb, ke, var_score, nnz_score_delta,
                                assign_i32, weight, os_new, skip_var);
  } else if (row_len <= k_narrow8_max) {
    PatchRowNarrow8Impl<coef_t>(variables, coefficients, kb, ke, var_score, nnz_score_delta,
                                assign_i32, weight, os_new, skip_var);
  } else {
    PatchRowImpl<coef_t>(variables, coefficients, kb, ke, var_score, nnz_score_delta, assign_i32,
                         weight, os_new, skip_var);
  }
}

// Tiled sweep carrying a running maximum. The index re-scan fires only on a tile that raises it,
// and that tile is still cache-hot. The tabu window is uint16 against int32 scores, so the mask
// crosses a 2:1 width boundary through PromoteMaskTo.
void ArgmaxImpl(const int64_t* HWY_RESTRICT var_score,
                int32_t n,
                int32_t tile,
                int32_t* best_var,
                int64_t* best_score)
{
  const hn::ScalableTag<int64_t> d;
  using V = hn::Vec<decltype(d)>;

  const int32_t step = (int32_t)hn::Lanes(d);
  const V vmin       = hn::Set(d, fj_bin_score_invalid);

  // Whole vectors only; the remainder is scanned scalar below.
  const int32_t nblk = n - (n % step);
  int32_t tile_step  = tile - (tile % step);
  if (tile_step < step) tile_step = step;

  int32_t bv = -1;
  int64_t bs = fj_bin_score_invalid;

  for (int32_t t0 = 0; t0 < nblk; t0 += tile_step) {
    const int32_t t1 = (t0 + tile_step < nblk) ? t0 + tile_step : nblk;

    V tile_max = vmin;
    for (int32_t v = t0; v < t1; v += step) {
      tile_max = hn::Max(tile_max, hn::LoadU(d, var_score + v));
    }

    const int64_t peak = hn::ReduceMax(d, tile_max);
    if (peak > bs) {
      const V vpeak = hn::Set(d, peak);
      for (int32_t v = t0; v < t1; v += step) {
        const intptr_t lane = hn::FindFirstTrue(d, hn::Eq(hn::LoadU(d, var_score + v), vpeak));
        if (lane >= 0) {
          bv = v + (int32_t)lane;
          break;
        }
      }
      bs = peak;
    }
  }

  for (int32_t v = nblk; v < n; ++v) {
    if (var_score[v] > bs) {
      bs = var_score[v];
      bv = v;
    }
  }

  *best_var   = bv;
  *best_score = bs;
}

// combined[v] = var_score[v] + obj_score[v] over all n variables. Materialized rather than fused
// into the argmax because block_tabu writes sentinels into the result and restores them afterwards,
// so the array has to outlive the scan. None of the three has SIMD padding, hence the scalar tail.
void AddScoresImpl(const int64_t* HWY_RESTRICT var_score,
                   const int64_t* HWY_RESTRICT obj_score,
                   int32_t n,
                   int64_t* HWY_RESTRICT combined)
{
  const hn::ScalableTag<int64_t> d;
  const int32_t step = (int32_t)hn::Lanes(d);
  const int32_t nblk = n - (n % step);

  for (int32_t v = 0; v < nblk; v += step) {
    hn::StoreU(hn::Add(hn::LoadU(d, var_score + v), hn::LoadU(d, obj_score + v)), d, combined + v);
  }
  for (int32_t v = nblk; v < n; ++v) {
    combined[v] = var_score[v] + obj_score[v];
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace cuopt::mathematical_optimization::mip
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace cuopt::mathematical_optimization::mip {

// One dispatch table per (coefficient width, vector width). HWY_EXPORT_T names the table
// separately from the function, which lets the function be a template-id: only the table name goes
// through token pasting, so no hand-written non-template wrapper is needed. The template argument
// must stay comma-free, which is why the three tag-binding wrappers above take only coef_t.
HWY_EXPORT_T(PatchRowI8, PatchRowDispatchImpl<int8_t>);
HWY_EXPORT_T(PatchRowI16, PatchRowDispatchImpl<int16_t>);
HWY_EXPORT_T(WalkRowsI8, WalkRowsImpl<int8_t>);
HWY_EXPORT_T(WalkRowsI16, WalkRowsImpl<int16_t>);
HWY_EXPORT(ArgmaxImpl);
HWY_EXPORT(AddScoresImpl);

// HWY_DYNAMIC_DISPATCH resolves the target on every call, and the hwy::GetChosenTarget() call it
// expands to is a real out-of-line call: it clobbers the argument registers, so the compiler spills
// all eleven parameters to the stack and reloads them around it. These run once per row per move,
// so the pointers are resolved once instead.
//
// Entry 0 of a dispatch table is a trampoline that chooses the target and re-dispatches, and an
// unchosen target makes GetIndex() return 0. Caching then would pin that extra indirection for the
// process lifetime, so the target is chosen first. File scope rather than function scope keeps the
// guard variable of a magic static out of the call: its cold path can call __cxa_guard_acquire, so
// the compiler must preserve the arguments across it and cannot leave a bare tail jump. Nothing in
// cuOpt reaches feasibility jump during static initialization.
static void fj_bin_choose_target()
{
  if (!hwy::GetChosenTarget().IsInitialized()) {
    hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  }
}

template <typename coef_t>
using fj_bin_patch_fn_t = void (*)(const int32_t*,
                                   const coef_t*,
                                   int32_t,
                                   int32_t,
                                   int64_t*,
                                   int64_t*,
                                   const int32_t*,
                                   int32_t,
                                   int32_t,
                                   int32_t);

// The vector width is chosen inside the target (see PatchRowDispatchImpl), so the seam carries one
// pointer per coefficient width and nothing else.
static const auto fj_bin_patch_i8 =
  (fj_bin_choose_target(), (fj_bin_patch_fn_t<int8_t>)HWY_DYNAMIC_POINTER_T(PatchRowI8));
static const auto fj_bin_patch_i16 =
  (fj_bin_choose_target(), (fj_bin_patch_fn_t<int16_t>)HWY_DYNAMIC_POINTER_T(PatchRowI16));

// Overloaded rather than specialized, matching fj_bin_walk_fn below.
static fj_bin_patch_fn_t<int8_t> fj_bin_patch_fn(int8_t) { return fj_bin_patch_i8; }
static fj_bin_patch_fn_t<int16_t> fj_bin_patch_fn(int16_t) { return fj_bin_patch_i16; }

template <typename coef_t>
using fj_bin_walk_fn_t = int32_t (*)(
  int32_t*, const int32_t*, const coef_t*, const coef_t*, int32_t, int32_t, int32_t, int32_t*);

static const auto fj_bin_walk_i8 =
  (fj_bin_choose_target(), (fj_bin_walk_fn_t<int8_t>)HWY_DYNAMIC_POINTER_T(WalkRowsI8));
static const auto fj_bin_walk_i16 =
  (fj_bin_choose_target(), (fj_bin_walk_fn_t<int16_t>)HWY_DYNAMIC_POINTER_T(WalkRowsI16));

static fj_bin_walk_fn_t<int8_t> fj_bin_walk_fn(int8_t) { return fj_bin_walk_i8; }
static fj_bin_walk_fn_t<int16_t> fj_bin_walk_fn(int16_t) { return fj_bin_walk_i16; }

static const auto fj_bin_argmax_fn = (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(ArgmaxImpl));
static const auto fj_bin_add_scores_fn =
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(AddScoresImpl));

template <typename coef_t>
int32_t fj_bin_walk_rows(int32_t* row_slack,
                         const int32_t* incident_row,
                         const coef_t* reverse_coefficients,
                         const coef_t* incident_row_cmax,
                         int32_t incidence_begin,
                         int32_t incidence_end,
                         int32_t delta,
                         int32_t* out_incidence)
{
  return fj_bin_walk_fn(coef_t{})(row_slack,
                                  incident_row,
                                  reverse_coefficients,
                                  incident_row_cmax,
                                  incidence_begin,
                                  incidence_end,
                                  delta,
                                  out_incidence);
}

template int32_t fj_bin_walk_rows<int8_t>(
  int32_t*, const int32_t*, const int8_t*, const int8_t*, int32_t, int32_t, int32_t, int32_t*);
template int32_t fj_bin_walk_rows<int16_t>(
  int32_t*, const int32_t*, const int16_t*, const int16_t*, int32_t, int32_t, int32_t, int32_t*);

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
                      int32_t skip_var)
{
  fj_bin_patch_fn(coef_t{})(variables, coefficients, kb, ke, var_score, nnz_score_delta, assign_i32,
                            weight, os_new, skip_var);
}

template void fj_bin_patch_row<int8_t>(const int32_t*,
                                       const int8_t*,
                                       int32_t,
                                       int32_t,
                                       int64_t*,
                                       int64_t*,
                                       const int32_t*,
                                       int32_t,
                                       int32_t,
                                       int32_t);

template void fj_bin_patch_row<int16_t>(const int32_t*,
                                        const int16_t*,
                                        int32_t,
                                        int32_t,
                                        int64_t*,
                                        int64_t*,
                                        const int32_t*,
                                        int32_t,
                                        int32_t,
                                        int32_t);

void fj_bin_argmax(const int64_t* var_score,
                   int32_t n,
                   int32_t tile,
                   int32_t& best_var,
                   int64_t& best_score)
{
  fj_bin_argmax_fn(var_score, n, tile, &best_var, &best_score);
}

void fj_bin_add_scores(const int64_t* var_score,
                       const int64_t* obj_score,
                       int32_t n,
                       int64_t* combined)
{
  fj_bin_add_scores_fn(var_score, obj_score, n, combined);
}

}  // namespace cuopt::mathematical_optimization::mip
#endif  // HWY_ONCE
