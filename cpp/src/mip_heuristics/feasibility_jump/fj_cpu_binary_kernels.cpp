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
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mip_heuristics/feasibility_jump/fj_cpu_binary_kernels.cpp"
#include "hwy/foreach_target.h"  // must precede highway.h
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace cuopt::mathematical_optimization::mip {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// on ISAs with mask support these ops are really fast. However they need to be emulated on older
// ISAs so avoid emitting them on these (e.g. pre AVX-512/AVX10.2 on x86)
constexpr bool k_mask_remainder =
  (HWY_TARGET <= HWY_AVX3) || HWY_TARGET_IS_SVE || (HWY_TARGET == HWY_RVV);

// on arcs without gather/scatter this is too slow, fallback
constexpr bool k_vector_walk =
  (HWY_TARGET <= HWY_AVX3) || HWY_TARGET_IS_SVE || (HWY_TARGET == HWY_RVV);

// Updates slacks for rows incident to a flipped variable.
// Returns incidences whose old and new slacks are not both deeply satisfied for specialized
// handling by the caller We operate directly on row slacks to reduce arithmetic ops.
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

      const auto deep_sat = hn::And(hn::Gt(os, cmax), hn::Gt(ns, cmax));
      const auto to_tail  = hn::AndNot(deep_sat, active);

      // Zen4's VSIB ops are unfortunately heavily microcoded and slower than just a scalar
      // implementation which gets scheduled better
#if HWY_TARGET == HWY_AVX3_ZEN4
      // Same Zen 4 microcode argument as the score scatter in PatchRowBody: VPSCATTERDD is 89 uops
      // at ~24 CPI, against two vector stores and N scalar stores here. Unlike that one this is a
      // pure store with no read-modify-write, so it needs its own A/B before the arm is settled.
      HWY_ALIGN int32_t row_lane[hn::MaxLanes(d)], slack_lane[hn::MaxLanes(d)];
      hn::Store(rows, d, row_lane);
      hn::Store(ns, d, slack_lane);
      const size_t lanes = HWY_MIN(N, (size_t)(incidence_end - ii));
      for (size_t i = 0; i < lanes; ++i)
        row_slack[row_lane[i]] = slack_lane[i];
#else
      hn::MaskedScatterIndex(ns, active, d, row_slack, rows);
#endif

      // A variable meets each row at most once, so no two lanes carry the same row and neither the
      // scatter above nor the store loop needs conflict detection.
      n_out += (int32_t)hn::CompressStore(hn::Iota(d, ii), to_tail, d, out_incidence + n_out);
    }
    return n_out;
  }

  // scalar version serves as the tail
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

template <typename coef_t>
void PatchRowScalar(const int32_t* HWY_RESTRICT variables,
                    const coef_t* HWY_RESTRICT coefficients,
                    int32_t row_begin,
                    int32_t row_end,
                    int64_t* HWY_RESTRICT var_score,
                    int64_t* HWY_RESTRICT nnz_score_delta,
                    const int32_t* HWY_RESTRICT assign_i32,
                    int32_t weight,
                    int32_t current_slack,
                    int32_t skip_var)
{
  for (int32_t k = row_begin; k < row_end; ++k) {
    const int32_t v = variables[k];
    if (v == skip_var) continue;
    const int32_t flip = 1 - 2 * assign_i32[v];
    const int32_t ns   = current_slack - (int32_t)coefficients[k] * flip;
    const int64_t nc   = fj_bin_packed_score_delta(current_slack, ns, weight);
    var_score[v] += nc - nnz_score_delta[k];
    nnz_score_delta[k] = nc;
  }
}

// walk over a row after its slack has been updated
// to refersh the scores of incident variables, and the per-nnz contributions
template <typename coef_t, class D>
static HWY_INLINE void PatchRowBody(D d,
                                    const int32_t* HWY_RESTRICT variables,
                                    const coef_t* HWY_RESTRICT coefficients,
                                    int32_t row_begin,
                                    int32_t row_end,
                                    int64_t* HWY_RESTRICT var_score,
                                    int64_t* HWY_RESTRICT nnz_score_delta,
                                    const int32_t* HWY_RESTRICT assign_i32,
                                    int32_t weight,
                                    int32_t current_slack,
                                    int32_t skip_var)
{
  const hn::Rebind<coef_t, D> dc;        // same lane count, narrower lanes
  const hn::Repartition<int64_t, D> dw;  // half the lanes, twice as wide: the packed score
  using V         = hn::Vec<decltype(d)>;
  using VW        = hn::Vec<decltype(dw)>;
  const size_t N  = hn::Lanes(d);
  const size_t NW = hn::Lanes(dw);

  // just run the scalar version if this ISA doesn't have masked ops and the size is < the vector
  // width avoids unnecessary setup
  if constexpr (!k_mask_remainder) {
    if ((size_t)(row_end - row_begin) < N) {
      PatchRowScalar<coef_t>(variables,
                             coefficients,
                             row_begin,
                             row_end,
                             var_score,
                             nnz_score_delta,
                             assign_i32,
                             weight,
                             current_slack,
                             skip_var);
      return;
    }
  }

  const V vone = hn::Set(d, 1), vzero = hn::Zero(d);
  const V vskip = hn::Set(d, skip_var);
  const V vos   = hn::Set(d, current_slack);
  const V vw = hn::Set(d, weight), vw2 = hn::Set(d, weight / 2);

  // The row's own slack is uniform across lanes, so its flags are scalars. Broadcast negated to
  // match the new-state flags below, which come from VecFromMask and are 0 or -1.
  const int32_t osat = current_slack >= 0, ost = current_slack > 0;
  const V vneg_osat = hn::Set(d, -osat), vneg_ost = hn::Set(d, -ost);
  const V v_not_osat = hn::Set(d, 1 - osat);

  // The loads always run unmasked and read into the per-nnz padding; when the remainder is masked,
  // FirstN keeps the overhang out of the gather, the scatter and the store.
  const int32_t vec_end = k_mask_remainder ? row_end : row_end - (int32_t)N + 1;
  int32_t k             = row_begin;
  for (; k < vec_end; k += (int32_t)N) {
    const V v   = hn::LoadU(d, variables + k);
    auto active = hn::Ne(v, vskip);
    if constexpr (k_mask_remainder) {
      active = hn::And(active, hn::FirstN(d, (size_t)(row_end - k)));
    }

    // Native gather works best here, even on Zen4. Go figure (probably more favorable
    // scheduling/ports for this codepath)
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

    const VW base_lo  = hn::PromoteLowerTo(dw, base);
    const VW base_hi  = hn::PromoteUpperTo(dw, base);
    const VW bonus_lo = hn::PromoteLowerTo(dw, bonus);
    const VW bonus_hi = hn::PromoteUpperTo(dw, bonus);

    const VW packed_lo = hn::Add(hn::ShiftLeft<fj_bin_score_shift>(base_lo), bonus_lo);
    const VW packed_hi = hn::Add(hn::ShiftLeft<fj_bin_score_shift>(base_hi), bonus_hi);

    const VW delta_lo = hn::Sub(packed_lo, hn::LoadU(dw, nnz_score_delta + k));
    const VW delta_hi = hn::Sub(packed_hi, hn::LoadU(dw, nnz_score_delta + k + NW));

    const size_t rem  = (size_t)(row_end - k);
    const VW v_lo     = hn::PromoteLowerTo(dw, v);
    const VW v_hi     = hn::PromoteUpperTo(dw, v);
    const VW vskip_w  = hn::Set(dw, skip_var);
    const auto act_lo = hn::And(hn::Ne(v_lo, vskip_w), hn::FirstN(dw, rem));
    const auto act_hi = hn::And(hn::Ne(v_hi, vskip_w), hn::FirstN(dw, rem > NW ? rem - NW : 0));
    hn::BlendedStore(packed_lo, act_lo, dw, nnz_score_delta + k);
    hn::BlendedStore(packed_hi, act_hi, dw, nnz_score_delta + k + NW);

    // hardware gather/scatter pays off heavily only on Sapphire Rapids+
    // probably on Zen5 as well
#if HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX2
    // zmm VSIB is microcode on Zen 4: VPGATHERDD ~76-80 uops / ~21 CPI and VPSCATTERDD 89 / 24,
    // against ~5 / ~10 and ~19 / ~11 on SPR-class Intel
    HWY_ALIGN int32_t idx[hn::MaxLanes(d)];
    HWY_ALIGN int64_t dl[hn::MaxLanes(d)];
    hn::Store(v, d, idx);
    hn::Store(delta_lo, dw, dl);
    hn::Store(delta_hi, dw, dl + NW);
    // Bounded by the row, not the vector: the lanes past it hold padding, whose zero index would
    // otherwise be applied to variable 0.
    const size_t lanes = HWY_MIN(N, (size_t)(row_end - k));
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

  // scalar tail
  if constexpr (!k_mask_remainder) {
    PatchRowScalar<coef_t>(variables,
                           coefficients,
                           k,
                           row_end,
                           var_score,
                           nnz_score_delta,
                           assign_i32,
                           weight,
                           current_slack,
                           skip_var);
  }
}

template <typename coef_t>
void PatchRowImpl(const int32_t* HWY_RESTRICT variables,
                  const coef_t* HWY_RESTRICT coefficients,
                  int32_t row_begin,
                  int32_t row_end,
                  int64_t* HWY_RESTRICT var_score,
                  int64_t* HWY_RESTRICT nnz_score_delta,
                  const int32_t* HWY_RESTRICT assign_i32,
                  int32_t weight,
                  int32_t current_slack,
                  int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::ScalableTag<int32_t>(),
                       variables,
                       coefficients,
                       row_begin,
                       row_end,
                       var_score,
                       nnz_score_delta,
                       assign_i32,
                       weight,
                       current_slack,
                       skip_var);
}

template <typename coef_t>
void PatchRowNarrow8Impl(const int32_t* HWY_RESTRICT variables,
                         const coef_t* HWY_RESTRICT coefficients,
                         int32_t row_begin,
                         int32_t row_end,
                         int64_t* HWY_RESTRICT var_score,
                         int64_t* HWY_RESTRICT nnz_score_delta,
                         const int32_t* HWY_RESTRICT assign_i32,
                         int32_t weight,
                         int32_t current_slack,
                         int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::CappedTagIfFixed<int32_t, 8>(),
                       variables,
                       coefficients,
                       row_begin,
                       row_end,
                       var_score,
                       nnz_score_delta,
                       assign_i32,
                       weight,
                       current_slack,
                       skip_var);
}

template <typename coef_t>
void PatchRowNarrow4Impl(const int32_t* HWY_RESTRICT variables,
                         const coef_t* HWY_RESTRICT coefficients,
                         int32_t row_begin,
                         int32_t row_end,
                         int64_t* HWY_RESTRICT var_score,
                         int64_t* HWY_RESTRICT nnz_score_delta,
                         const int32_t* HWY_RESTRICT assign_i32,
                         int32_t weight,
                         int32_t current_slack,
                         int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::CappedTagIfFixed<int32_t, 4>(),
                       variables,
                       coefficients,
                       row_begin,
                       row_end,
                       var_score,
                       nnz_score_delta,
                       assign_i32,
                       weight,
                       current_slack,
                       skip_var);
}

// use lower-vector-width kernels for smaller rows if available on this target
// (e.g. AVX2 instead of AVX512)
// (works because AVX512 also brings masked ops and gather/scatter to 128/256bit vectors)
constexpr size_t k_native_lanes = HWY_MAX_LANES_D(hn::ScalableTag<int32_t>);
constexpr int32_t k_narrow4_max = HWY_HAVE_SCALABLE ? 0 : (k_native_lanes > 4 ? 4 : 0);
constexpr int32_t k_narrow8_max = HWY_HAVE_SCALABLE ? 0 : (k_native_lanes > 8 ? 8 : 0);

// Single entry point the seam dispatches to. On a scalable target both bounds are 0, so both
// compares fold away and the narrow arms are stripped.
template <typename coef_t>
void PatchRowDispatchImpl(const int32_t* HWY_RESTRICT variables,
                          const coef_t* HWY_RESTRICT coefficients,
                          int32_t row_begin,
                          int32_t row_end,
                          int64_t* HWY_RESTRICT var_score,
                          int64_t* HWY_RESTRICT nnz_score_delta,
                          const int32_t* HWY_RESTRICT assign_i32,
                          int32_t weight,
                          int32_t current_slack,
                          int32_t skip_var)
{
  const int32_t row_len = row_end - row_begin;
  if (row_len <= k_narrow4_max) {
    PatchRowNarrow4Impl<coef_t>(variables,
                                coefficients,
                                row_begin,
                                row_end,
                                var_score,
                                nnz_score_delta,
                                assign_i32,
                                weight,
                                current_slack,
                                skip_var);
  } else if (row_len <= k_narrow8_max) {
    PatchRowNarrow8Impl<coef_t>(variables,
                                coefficients,
                                row_begin,
                                row_end,
                                var_score,
                                nnz_score_delta,
                                assign_i32,
                                weight,
                                current_slack,
                                skip_var);
  } else {
    PatchRowImpl<coef_t>(variables,
                         coefficients,
                         row_begin,
                         row_end,
                         var_score,
                         nnz_score_delta,
                         assign_i32,
                         weight,
                         current_slack,
                         skip_var);
  }
}

// tiled two-pass argmax to first find a new max
// and then a second pass if a new max was found to then fetch its index
// works on the assu,ption the tile is L1 sized
// and that few tiles contain a new max compared to the entire set
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
      // second pass to find the index
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

  // scalar tail
  for (int32_t v = nblk; v < n; ++v) {
    if (var_score[v] > bs) {
      bs = var_score[v];
      bv = v;
    }
  }

  *best_var   = bv;
  *best_score = bs;
}

// combined[v] = var_score[v] + obj_score[v] over all n variables.
void AddScoresImpl(const int64_t* HWY_RESTRICT var_score,
                   const int64_t* HWY_RESTRICT obj_score,
                   int32_t n,
                   int64_t* HWY_RESTRICT combined)
{
  const hn::ScalableTag<int64_t> d;
  const int32_t step = (int32_t)hn::Lanes(d);
  const int32_t nblk = n - (n % step);
  for (int32_t v = 0; v < nblk; v += step)
    hn::StoreU(hn::Add(hn::LoadU(d, var_score + v), hn::LoadU(d, obj_score + v)), d, combined + v);
  for (int32_t v = nblk; v < n; ++v)
    combined[v] = var_score[v] + obj_score[v];
}

// row scoring for the general engine
template <typename T>
void ScoreRowsImpl(const int32_t* rows,
                   const T* coeff,
                   const T* state,
                   int32_t begin,
                   int32_t end,
                   T delta,
                   T tol,
                   T excess,
                   T* base_out,
                   T* bonus_out)
{
  const hn::ScalableTag<T> d;
  const hn::Rebind<int32_t, decltype(d)> d32;
  const hn::RebindToSigned<decltype(d)> di;
  const int32_t step = (int32_t)hn::Lanes(d);
  const int32_t nblk = end - ((end - begin) % step);
  const auto zero = hn::Zero(d), vtol = hn::Set(d, tol), negtol = hn::Neg(vtol);
  auto base = zero, bonus = zero;
  int32_t i = begin;
  for (; i < nblk; i += step) {
    const auto r32 = hn::LoadU(d32, rows + i);
    const auto r   = [&] {
      if constexpr (sizeof(T) == 4)
        return r32;
      else
        return hn::PromoteTo(di, r32);
    }();
    const auto si     = hn::Add(r, r);
    const auto old    = hn::GatherIndex(d, state, si);
    const auto weight = hn::GatherIndex(d, state, hn::Add(si, hn::Set(di, 1)));
    const auto next   = hn::NegMulAdd(hn::LoadU(d, coeff + i), hn::Set(d, delta), old);
    const auto os = hn::Gt(old, negtol), ns = hn::Gt(next, negtol);
    auto row_base = hn::Sub(hn::IfThenElse(ns, weight, zero), hn::IfThenElse(os, weight, zero));
    const auto partial   = hn::Trunc(hn::Mul(weight, hn::Set(d, excess)));
    const auto direction = hn::IfThenElse(
      hn::Gt(next, old), partial, hn::IfThenElse(hn::Lt(next, old), hn::Neg(partial), zero));
    row_base =
      hn::Add(row_base, hn::IfThenElse(hn::And(hn::Not(os), hn::Not(ns)), direction, zero));
    base  = hn::Add(base, row_base);
    bonus = hn::Add(bonus,
                    hn::Sub(hn::IfThenElse(hn::Gt(next, vtol), weight, zero),
                            hn::IfThenElse(hn::Gt(old, vtol), weight, zero)));
  }
  T bs = hn::ReduceSum(d, base), rs = hn::ReduceSum(d, bonus);

  // scalar tail
  for (; i < end; ++i) {
    const T a = coeff[i];
    if (a == 0) continue;
    const T old = state[2 * rows[i]], weight = state[2 * rows[i] + 1], next = old - a * delta;
    const bool os = old > -tol, ns = next > -tol;
    bs += (!os && ns)                  ? weight
          : (os && !ns)                ? -weight
          : (!os && !ns && next > old) ? (int32_t)(weight * excess)
          : (!os && !ns && next < old) ? -(int32_t)(weight * excess)
                                       : 0;
    rs += (old <= tol && next > tol) ? weight : (old > tol && next <= tol) ? -weight : 0;
  }
  *base_out  = bs;
  *bonus_out = rs;
}

void ScoreRowsF32(const int32_t* r,
                  const float* c,
                  const float* s,
                  int32_t b,
                  int32_t e,
                  float d,
                  float t,
                  float x,
                  float* bs,
                  float* rs)
{
  ScoreRowsImpl(r, c, s, b, e, d, t, x, bs, rs);
}
void ScoreRowsF64(const int32_t* r,
                  const double* c,
                  const double* s,
                  int32_t b,
                  int32_t e,
                  double d,
                  double t,
                  double x,
                  double* bs,
                  double* rs)
{
  ScoreRowsImpl(r, c, s, b, e, d, t, x, bs, rs);
}

}  // namespace HWY_NAMESPACE
}  // namespace cuopt::mathematical_optimization::mip
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace cuopt::mathematical_optimization::mip {

// highway dispatch table
HWY_EXPORT_T(PatchRowI8, PatchRowDispatchImpl<int8_t>);
HWY_EXPORT_T(PatchRowI16, PatchRowDispatchImpl<int16_t>);
HWY_EXPORT_T(WalkRowsI8, WalkRowsImpl<int8_t>);
HWY_EXPORT_T(WalkRowsI16, WalkRowsImpl<int16_t>);
HWY_EXPORT(ArgmaxImpl);
HWY_EXPORT(AddScoresImpl);
HWY_EXPORT(ScoreRowsF32);
HWY_EXPORT(ScoreRowsF64);

// resolve the highway target once instead of suffering indirection on every call
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

// little operator, trick to get highway to resolve the target on global static initialization
static const auto fj_bin_patch_i8 =
  (fj_bin_choose_target(), (fj_bin_patch_fn_t<int8_t>)HWY_DYNAMIC_POINTER_T(PatchRowI8));
static const auto fj_bin_patch_i16 =
  (fj_bin_choose_target(), (fj_bin_patch_fn_t<int16_t>)HWY_DYNAMIC_POINTER_T(PatchRowI16));

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
static const auto fj_score_rows_f32_fn =
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(ScoreRowsF32));
static const auto fj_score_rows_f64_fn =
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(ScoreRowsF64));

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
                      int32_t row_begin,
                      int32_t row_end,
                      int64_t* var_score,
                      int64_t* nnz_score_delta,
                      const int32_t* assign_i32,
                      int32_t weight,
                      int32_t current_slack,
                      int32_t skip_var)
{
  fj_bin_patch_fn(coef_t{})(variables,
                            coefficients,
                            row_begin,
                            row_end,
                            var_score,
                            nnz_score_delta,
                            assign_i32,
                            weight,
                            current_slack,
                            skip_var);
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

void fj_bin_argmax(
  const int64_t* var_score, int32_t n, int32_t tile, int32_t& best_var, int64_t& best_score)
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

template <typename T>
void fj_simd_score_rows(const int32_t* rows,
                        const T* coeff,
                        const T* state,
                        int32_t begin,
                        int32_t end,
                        T delta,
                        T tolerance,
                        T excess,
                        T& base,
                        T& bonus)
{
  if constexpr (std::is_same_v<T, float>)
    fj_score_rows_f32_fn(rows, coeff, state, begin, end, delta, tolerance, excess, &base, &bonus);
  else
    fj_score_rows_f64_fn(rows, coeff, state, begin, end, delta, tolerance, excess, &base, &bonus);
}
template void fj_simd_score_rows<float>(const int32_t*,
                                        const float*,
                                        const float*,
                                        int32_t,
                                        int32_t,
                                        float,
                                        float,
                                        float,
                                        float&,
                                        float&);
template void fj_simd_score_rows<double>(const int32_t*,
                                         const double*,
                                         const double*,
                                         int32_t,
                                         int32_t,
                                         double,
                                         double,
                                         double,
                                         double&,
                                         double&);

}  // namespace cuopt::mathematical_optimization::mip
#endif  // HWY_ONCE
