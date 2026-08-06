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

// Padding the caller must carry past nnz. Only a masking target reads past a row's end; a peeling
// target stops at ke, so it asks for nothing and its buffers keep their natural size.
int32_t PaddingImpl()
{
  return k_mask_remainder ? (int32_t)hn::Lanes(hn::ScalableTag<int32_t>()) : 0;
}

// Row remainder when it is peeled rather than masked, and the whole row on scalar targets.
template <typename coef_t>
void PatchRowScalar(const int32_t* HWY_RESTRICT variables,
                    const coef_t* HWY_RESTRICT coefficients,
                    int32_t kb,
                    int32_t ke,
                    int32_t* HWY_RESTRICT var_score,
                    int32_t* HWY_RESTRICT nnz_score_delta,
                    const int32_t* HWY_RESTRICT assign_i32,
                    int32_t sign,
                    int32_t weight,
                    int32_t os_new,
                    int32_t skip_var)
{
  for (int32_t k = kb; k < ke; ++k) {
    const int32_t v = variables[k];
    if (v == skip_var) continue;
    const int32_t flip = 1 - 2 * assign_i32[v];
    const int32_t ns   = os_new - sign * ((int32_t)coefficients[k] * flip);
    const int32_t nc   = fj_bin_packed_score_delta(os_new, ns, weight);
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
                                    int32_t* HWY_RESTRICT var_score,
                                    int32_t* HWY_RESTRICT nnz_score_delta,
                                    const int32_t* HWY_RESTRICT assign_i32,
                                    int32_t sign,
                                    int32_t weight,
                                    int32_t os_new,
                                    int32_t skip_var)
{
  const hn::Rebind<coef_t, D> dc;  // same lane count, narrower lanes
  using V        = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);

  // When the remainder is peeled, a row below one vector never reaches the body, so it skips the
  // ten broadcasts below as well.
  if constexpr (!k_mask_remainder) {
    if ((size_t)(ke - kb) < N) {
      PatchRowScalar<coef_t>(variables, coefficients, kb, ke, var_score, nnz_score_delta,
                             assign_i32, sign, weight, os_new, skip_var);
      return;
    }
  }

  const V vone = hn::Set(d, 1), vzero = hn::Zero(d);
  const V vsign = hn::Set(d, sign), vskip = hn::Set(d, skip_var);
  const V vos = hn::Set(d, os_new);
  const V vw = hn::Set(d, weight), vw2 = hn::Set(d, weight / 2);

  // The row's own slack is uniform across lanes, so its flags are scalars.
  const int32_t osat = os_new >= 0, ost = os_new > 0;
  const V vosat = hn::Set(d, osat), vost = hn::Set(d, ost);
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
    const V flip = vone - hn::ShiftLeft<1>(a01);
    const V coef = hn::PromoteTo(d, hn::LoadU(dc, coefficients + k));

    const V ns = vos - vsign * coef * flip;

    const V nsat = hn::IfThenElseZero(hn::Ge(ns, vzero), vone);
    const V nst  = hn::IfThenElseZero(hn::Gt(ns, vzero), vone);
    const V improving =
      hn::IfThenElseZero(hn::Gt(ns, vos), vone) - hn::IfThenElseZero(hn::Lt(ns, vos), vone);

    const V both_violated = v_not_osat * (vone - nsat);
    const V base          = vw * (nsat - vosat) + both_violated * improving * vw2;
    const V bonus         = vw * (nst - vost);
    const V packed_new    = hn::ShiftLeft<fj_bin_score_shift>(base) + bonus;

    const V delta = packed_new - hn::LoadU(d, nnz_score_delta + k);

#if HWY_TARGET == HWY_AVX3_ZEN4
    // zmm VSIB is microcode on Zen 4: VPGATHERDD ~76-80 uops / ~21 CPI and VPSCATTERDD 89 / 24,
    // against ~5 / ~10 and ~19 / ~11 on SPR-class Intel (Agner Fog, uops.info). So read-modify-write
    // by lane here; measured +5.8% over the arm below on an EPYC 9554 (supportcase22, 16 climbers).
    HWY_ALIGN int32_t idx[hn::MaxLanes(d)], dl[hn::MaxLanes(d)];
    hn::Store(v, d, idx);
    hn::Store(delta, d, dl);
    // Bounded by the row, not the vector: the lanes past it hold padding, whose zero index would
    // otherwise be applied to variable 0.
    const size_t lanes = HWY_MIN(N, (size_t)(ke - k));
    for (size_t i = 0; i < lanes; ++i) {
      if (idx[i] != skip_var) var_score[idx[i]] += dl[i];
    }
#else
    const V current = hn::MaskedGatherIndex(active, d, var_score, v);
    hn::MaskedScatterIndex(current + delta, active, d, var_score, v);
#endif

    hn::BlendedStore(packed_new, active, d, nnz_score_delta + k);
  }

  if constexpr (!k_mask_remainder) {
    PatchRowScalar<coef_t>(variables, coefficients, k, ke, var_score, nnz_score_delta, assign_i32,
                           sign, weight, os_new, skip_var);
  }
}

// Native width, and the 8-lane variant for rows that would leave most of a native vector idle.
template <typename coef_t>
void PatchRowImpl(const int32_t* HWY_RESTRICT variables,
                  const coef_t* HWY_RESTRICT coefficients,
                  int32_t kb,
                  int32_t ke,
                  int32_t* HWY_RESTRICT var_score,
                  int32_t* HWY_RESTRICT nnz_score_delta,
                  const int32_t* HWY_RESTRICT assign_i32,
                  int32_t sign,
                  int32_t weight,
                  int32_t os_new,
                  int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::ScalableTag<int32_t>(), variables, coefficients, kb, ke, var_score,
                       nnz_score_delta, assign_i32, sign, weight, os_new, skip_var);
}

template <typename coef_t>
void PatchRowNarrow8Impl(const int32_t* HWY_RESTRICT variables,
                         const coef_t* HWY_RESTRICT coefficients,
                         int32_t kb,
                         int32_t ke,
                         int32_t* HWY_RESTRICT var_score,
                         int32_t* HWY_RESTRICT nnz_score_delta,
                         const int32_t* HWY_RESTRICT assign_i32,
                         int32_t sign,
                         int32_t weight,
                         int32_t os_new,
                         int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::CappedTagIfFixed<int32_t, 8>(), variables, coefficients, kb, ke,
                       var_score, nnz_score_delta, assign_i32, sign, weight, os_new, skip_var);
}

template <typename coef_t>
void PatchRowNarrow4Impl(const int32_t* HWY_RESTRICT variables,
                         const coef_t* HWY_RESTRICT coefficients,
                         int32_t kb,
                         int32_t ke,
                         int32_t* HWY_RESTRICT var_score,
                         int32_t* HWY_RESTRICT nnz_score_delta,
                         const int32_t* HWY_RESTRICT assign_i32,
                         int32_t sign,
                         int32_t weight,
                         int32_t os_new,
                         int32_t skip_var)
{
  PatchRowBody<coef_t>(hn::CappedTagIfFixed<int32_t, 4>(), variables, coefficients, kb, ke,
                       var_score, nnz_score_delta, assign_i32, sign, weight, os_new, skip_var);
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
int32_t Narrow4MaxImpl()
{
  if (HWY_HAVE_SCALABLE) return 0;
  return hn::Lanes(hn::ScalableTag<int32_t>()) > 4 ? 4 : 0;
}

int32_t Narrow8MaxImpl()
{
  if (HWY_HAVE_SCALABLE) return 0;
  return hn::Lanes(hn::ScalableTag<int32_t>()) > 8 ? 8 : 0;
}

// Tiled sweep carrying a running maximum. The index re-scan fires only on a tile that raises it,
// and that tile is still cache-hot. The tabu window is uint16 against int32 scores, so the mask
// crosses a 2:1 width boundary through PromoteMaskTo.
void ArgmaxImpl(const int32_t* HWY_RESTRICT var_score,
                int32_t n,
                int32_t tile,
                int32_t* best_var,
                int32_t* best_score)
{
  const hn::ScalableTag<int32_t> d;
  using V = hn::Vec<decltype(d)>;

  const int32_t step = (int32_t)hn::Lanes(d);
  const V vmin       = hn::Set(d, fj_bin_score_invalid);

  // Whole vectors only; the remainder is scanned scalar below.
  const int32_t nblk = n - (n % step);
  int32_t tile_step  = tile - (tile % step);
  if (tile_step < step) tile_step = step;

  int32_t bv = -1, bs = fj_bin_score_invalid;

  for (int32_t t0 = 0; t0 < nblk; t0 += tile_step) {
    const int32_t t1 = (t0 + tile_step < nblk) ? t0 + tile_step : nblk;

    V tile_max = vmin;
    for (int32_t v = t0; v < t1; v += step) {
      tile_max = hn::Max(tile_max, hn::LoadU(d, var_score + v));
    }

    const int32_t peak = hn::ReduceMax(d, tile_max);
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

}  // namespace HWY_NAMESPACE
}  // namespace cuopt::mathematical_optimization::mip
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace cuopt::mathematical_optimization::mip {

// One dispatch table per (coefficient width, vector width). HWY_EXPORT_T names the table
// separately from the function, which lets the function be a template-id: only the table name goes
// through token pasting, so no hand-written non-template wrapper is needed. The template argument
// must stay comma-free, which is why the three tag-binding wrappers above take only coef_t.
HWY_EXPORT_T(PatchRowNatI8, PatchRowImpl<int8_t>);
HWY_EXPORT_T(PatchRowN8I8, PatchRowNarrow8Impl<int8_t>);
HWY_EXPORT_T(PatchRowN4I8, PatchRowNarrow4Impl<int8_t>);
HWY_EXPORT_T(PatchRowNatI16, PatchRowImpl<int16_t>);
HWY_EXPORT_T(PatchRowN8I16, PatchRowNarrow8Impl<int16_t>);
HWY_EXPORT_T(PatchRowN4I16, PatchRowNarrow4Impl<int16_t>);
HWY_EXPORT(ArgmaxImpl);
HWY_EXPORT(PaddingImpl);
HWY_EXPORT(Narrow4MaxImpl);
HWY_EXPORT(Narrow8MaxImpl);

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
                                   int32_t*,
                                   int32_t*,
                                   const int32_t*,
                                   int32_t,
                                   int32_t,
                                   int32_t,
                                   int32_t);

// Indexed by fj_bin_patch_width_t.
static const fj_bin_patch_fn_t<int8_t> fj_bin_patch_i8[3] = {
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER_T(PatchRowN4I8)),
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER_T(PatchRowN8I8)),
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER_T(PatchRowNatI8)),
};

static const fj_bin_patch_fn_t<int16_t> fj_bin_patch_i16[3] = {
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER_T(PatchRowN4I16)),
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER_T(PatchRowN8I16)),
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER_T(PatchRowNatI16)),
};

// Overloaded rather than specialized so the tables stay plain arrays.
static const fj_bin_patch_fn_t<int8_t>* fj_bin_patch_table(int8_t) { return fj_bin_patch_i8; }
static const fj_bin_patch_fn_t<int16_t>* fj_bin_patch_table(int16_t) { return fj_bin_patch_i16; }

static const auto fj_bin_argmax_fn = (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(ArgmaxImpl));
static const auto fj_bin_padding_fn = (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(PaddingImpl));
static const auto fj_bin_narrow4_max_fn =
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(Narrow4MaxImpl));
static const auto fj_bin_narrow8_max_fn =
  (fj_bin_choose_target(), HWY_DYNAMIC_POINTER(Narrow8MaxImpl));

int32_t fj_bin_simd_padding() { return fj_bin_padding_fn(); }
int32_t fj_bin_simd_narrow4_max() { return fj_bin_narrow4_max_fn(); }
int32_t fj_bin_simd_narrow8_max() { return fj_bin_narrow8_max_fn(); }

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
                      int32_t skip_var)
{
  fj_bin_patch_table(coef_t{})[(int)width](variables, coefficients, kb, ke, var_score,
                                           nnz_score_delta, assign_i32, sign, weight, os_new,
                                           skip_var);
}

template void fj_bin_patch_row<int8_t>(fj_bin_patch_width_t,
                                       const int32_t*,
                                       const int8_t*,
                                       int32_t,
                                       int32_t,
                                       int32_t*,
                                       int32_t*,
                                       const int32_t*,
                                       int32_t,
                                       int32_t,
                                       int32_t,
                                       int32_t);

template void fj_bin_patch_row<int16_t>(fj_bin_patch_width_t,
                                        const int32_t*,
                                        const int16_t*,
                                        int32_t,
                                        int32_t,
                                        int32_t*,
                                        int32_t*,
                                        const int32_t*,
                                        int32_t,
                                        int32_t,
                                        int32_t,
                                        int32_t);

void fj_bin_argmax(const int32_t* var_score,
                   int32_t n,
                   int32_t tile,
                   int32_t& best_var,
                   int32_t& best_score)
{
  fj_bin_argmax_fn(var_score, n, tile, &best_var, &best_score);
}

}  // namespace cuopt::mathematical_optimization::mip
#endif  // HWY_ONCE
