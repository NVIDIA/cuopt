/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"  // gtest + make_path_absolute (mip_utils.cuh deps)
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <linear_algebra/sort_csr.cuh>
#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/presolve/block_bve.cuh>
#include <mip_heuristics/presolve/bounds_presolve.cuh>
#include <mip_heuristics/presolve/probing_cache.cuh>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver.cuh>
#include <mip_heuristics/utils.cuh>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <rmm/device_uvector.hpp>

#include <utilities/integer_scaling.hpp>
#include <utilities/timer.hpp>

#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// ---- host enumeration projection (the differential oracle) ----

template <typename f_t>
inline bool bve_is_finite(f_t x)
{
  // finite iff it equals itself (rules out NaN) and is strictly within +/- inf
  return (x == x) && (x < INFINITY) && (x > -INFINITY);
}

// Feasibility of one packed row under a full local assignment `val` (length na+nb), with tolerance.
template <typename f_t>
inline bool bve_row_sat(const bve_block_t<f_t>& blk, int r, const int* val, f_t tol)
{
  f_t s = 0;
  for (int k = blk.row_off[r]; k < blk.row_off[r + 1]; ++k) {
    s += blk.row_coef[k] * (f_t)val[blk.row_var[k]];
  }
  if (bve_is_finite(blk.row_up[r]) && s > blk.row_up[r] + tol) return false;
  if (bve_is_finite(blk.row_lo[r]) && s < blk.row_lo[r] - tol) return false;
  return true;
}

// Project the block onto its boundary. `feas[m]` (length 2^nb) is set to 1 iff boundary pattern m
// (nb bits) admits SOME interior assignment satisfying every block row, and `witness[m]` receives
// the packed interior assignment (na bits) of the FIRST feasible completion. Both are left 0 for
// infeasible patterns. The GPU kernel must match this exactly.
template <typename f_t>
inline void bve_project(const bve_block_t<f_t>& blk, f_t tol, uint8_t* feas, uint32_t* witness)
{
  const int na = blk.na, nb = blk.nb;
  int val[BVE_MAX_SCOPE];
  for (uint32_t m = 0; m < (1u << nb); ++m) {
    for (int j = 0; j < nb; ++j)
      val[na + j] = (m >> j) & 1u;
    feas[m]    = 0;
    witness[m] = 0u;
    for (uint32_t am = 0; am < (1u << na); ++am) {
      for (int j = 0; j < na; ++j)
        val[j] = (am >> j) & 1u;
      bool ok = true;
      for (int r = 0; r < blk.n_rows && ok; ++r)
        ok = bve_row_sat(blk, r, val, tol);
      if (ok) {
        feas[m]    = 1;
        witness[m] = am;
        break;
      }
    }
  }
}

enum class bve_status_t : int {
  kReduced    = 0,  // sanity check passed; `clauses` is a sound replacement for the block rows
  kSkipCaps   = 1,  // block violates a bound cap (defensive; detector should pre-filter)
  kSkipGrowth = 2,  // |clauses| > |rows| + margin (would grow the row count)
  kSkipCheckFailed =
    3  // clauses did not reproduce feas (sanity check failed) => keep block verbatim
};

// Full per-block core on the host: project -> prime-implicate CNF -> growth gate -> inline sanity
// check. The production commit_projected does the same, but reads feas/witness from the GPU instead
// of the host bve_project above.
template <typename i_t, typename f_t>
inline bve_status_t bve_project_and_check(const bve_block_t<f_t>& blk,
                                          f_t tol,
                                          i_t margin,
                                          bve_clause_t* clauses,
                                          i_t* n_clauses,
                                          uint32_t* witness)
{
  *n_clauses = 0;
  if (blk.nb <= 0 || blk.nb > BVE_MAX_BOUNDARY) return bve_status_t::kSkipCaps;
  if (blk.na < 0 || blk.na + blk.nb > BVE_MAX_SCOPE) return bve_status_t::kSkipCaps;
  if (blk.n_rows < 0 || blk.n_rows > BVE_MAX_ROWS) return bve_status_t::kSkipCaps;

  uint8_t feas[BVE_MAX_PATTERNS];
  bve_project(blk, tol, feas, witness);
  bve_cover_scratch_t scratch;
  const int nc = bve_greedy_prime_cover(feas, blk.nb, clauses, BVE_MAX_CLAUSES, scratch);
  if (nc < 0) return bve_status_t::kSkipGrowth;  // clause explosion past cap
  if (nc > blk.n_rows + margin) return bve_status_t::kSkipGrowth;
  if (!bve_sanity_check(feas, blk.nb, clauses, nc)) return bve_status_t::kSkipCheckFailed;
  *n_clauses = nc;
  return bve_status_t::kReduced;
}

}  // namespace cuopt::mathematical_optimization::mip

namespace cuopt::mathematical_optimization::test {

namespace mip = cuopt::mathematical_optimization::mip;

// A minimal "a = b OR c, with b+c <= 1 forced" block. `a` is the only zero-objective binary aux
// (b and c carry objective, so they stay on the boundary and are never absorbed into the interior).
// Eliminating `a` by exact projection leaves exactly ONE prime-implicate clause: b + c <= 1 (the
// boundary pattern b=c=1 is infeasible because it would force a=1 and violate a+b+c<=2).
static constexpr const char* kBlockLp = R"LP(
Minimize
 obj: b + c
Subject To
 r0: a - b >= 0
 r1: a - c >= 0
 r2: a + b + c <= 2
Binaries
 a
 b
 c
End
)LP";

// Same gadget with every row scaled by 1/2, so the block coefficients and bounds are FRACTIONAL.
// The feasible region (hence the reduction: b + c <= 1, `a` eliminated) is identical, as positive
// row scaling preserves feasibility. This forces block-BVE's per-row integerization
// (row_int_scale) to recover integer coefficients before the exact tol-0 projection; a wrong
// integerization breaks either the reduction or its reconstruction.
static constexpr const char* kFractionalBlockLp = R"LP(
Minimize
 obj: b + c
Subject To
 r0: 0.5 a - 0.5 b >= 0
 r1: 0.5 a - 0.5 c >= 0
 r2: 0.5 a + 0.5 b + 0.5 c <= 1
Binaries
 a
 b
 c
End
)LP";

// solve_mip opens an OMP team before MIP internals that use taskloops; probing_cache sizes its
// pool from omp_get_num_threads()-1 (0 outside a parallel region → silent no-op).
template <typename F>
static void with_mip_omp_team(F&& f)
{
  const int num_threads             = std::max(2, omp_get_max_threads());
  const int saved_max_active_levels = omp_get_max_active_levels();
  if (saved_max_active_levels < 2) { omp_set_max_active_levels(2); }
#pragma omp parallel num_threads(num_threads)
  {
#pragma omp masked
    {
      f();
    }
  }
  if (saved_max_active_levels < 2) { omp_set_max_active_levels(saved_max_active_levels); }
}

// Production implication adjacency: bounds → probing cache → trivial compact → bve_build_impl_adj.
// If `out_infeasible` is non-null, probing infeasibility is reported there (empty adj returned);
// otherwise the caller is assumed to expect a feasible instance and we ASSERT that.
static std::vector<std::vector<int>> probing_impl_adj(mip::problem_t<int, double>& problem,
                                                      bool* out_infeasible = nullptr)
{
  mip_solver_settings_t<int, double> settings{};
  cuopt::timer_t timer(30.0);
  mip::mip_solver_t<int, double> solver(problem, settings, timer);
  problem.tolerances = settings.get_tolerances();
  mip::bound_presolve_t<int, double> bound_presolve(solver.context);

  bool infeasible = false;
  with_mip_omp_team([&]() {
    auto term_crit = bound_presolve.solve(problem);
    if (term_crit != mip::termination_criterion_t::NO_UPDATE) {
      bound_presolve.set_updated_bounds(problem);
    }
    cuopt::timer_t probing_timer(30.0);
    infeasible = mip::compute_probing_cache(bound_presolve, problem, probing_timer);
    if (!infeasible) {
      constexpr bool remap_cache_ids = true;
      mip::trivial_presolve(problem, remap_cache_ids);
    }
  });
  if (out_infeasible != nullptr) {
    *out_infeasible = infeasible;
  } else {
    EXPECT_FALSE(infeasible);
  }
  if (infeasible) { return {}; }
  return mip::bve_build_impl_adj(
    bound_presolve.probing_cache, problem.reverse_original_ids, problem.n_variables, timer);
}

// Build one block by hand for the projection-core tests. Local ids: a=0 (interior), b=1, c=2.
static mip::bve_block_t<double> make_block()
{
  const double INF = std::numeric_limits<double>::infinity();
  mip::bve_block_t<double> blk{};
  blk.na     = 1;
  blk.nb     = 2;
  blk.n_rows = 3;
  int nz     = 0;
  auto row = [&](int r, std::initializer_list<std::pair<int, double>> terms, double lo, double up) {
    blk.row_off[r] = nz;
    for (const auto& t : terms) {
      blk.row_var[nz]  = t.first;
      blk.row_coef[nz] = t.second;
      ++nz;
    }
    blk.row_lo[r] = lo;
    blk.row_up[r] = up;
  };
  row(0, {{0, 1.0}, {1, -1.0}}, 0.0, INF);            // a - b >= 0
  row(1, {{0, 1.0}, {2, -1.0}}, 0.0, INF);            // a - c >= 0
  row(2, {{0, 1.0}, {1, 1.0}, {2, 1.0}}, -INF, 2.0);  // a + b + c <= 2
  blk.row_off[blk.n_rows] = nz;
  return blk;
}

// --- 1. projection core: the block sanity checks, yields one clause and the right witness ---
TEST(block_bve_core, reduces_block_and_sanity_checks)
{
  auto blk = make_block();
  mip::bve_clause_t clauses[mip::BVE_MAX_CLAUSES];
  uint32_t witness[mip::BVE_MAX_PATTERNS];
  int n_clauses = 0;
  auto st       = mip::bve_project_and_check(blk, 1e-6, /*margin=*/0, clauses, &n_clauses, witness);

  EXPECT_EQ(st, mip::bve_status_t::kReduced);
  ASSERT_EQ(n_clauses, 1);
  // clause forbids boundary pattern b=1,c=1 (bits 0 and 1 both set): b + c <= 1
  EXPECT_EQ(clauses[0].lit_mask, 3u);
  EXPECT_EQ(clauses[0].bit_mask, 3u);
  // witness: (b=0,c=0)->a=0, (b=1,c=0)->a=1, (b=0,c=1)->a=1
  EXPECT_EQ(witness[0], 0u);
  EXPECT_EQ(witness[1], 1u);
  EXPECT_EQ(witness[2], 1u);
}

// --- 2. sanity check safety: the INDEPENDENT clause evaluator rejects any clause set that
// misrepresents
//        feas (the certifying-algorithm result check; not a machine-checkable certificate) ---
TEST(block_bve_core, sanity_check_rejects_corrupted_clauses)
{
  // feasible-pattern array for the block above (b=c=1 is the only infeasible pattern)
  const uint8_t feas[4]              = {1, 1, 1, 0};
  const mip::bve_clause_t correct[1] = {{3u, 3u}};  // b + c <= 1
  EXPECT_TRUE(mip::bve_sanity_check(feas, 2, correct, 1));

  // dropping the clause entirely: the CNF would accept b=c=1, but feas forbids it -> rejected
  EXPECT_FALSE(mip::bve_sanity_check(feas, 2, correct, 0));
  // a wrong clause (forbid b=1 only) makes a genuinely feasible pattern look infeasible -> rejected
  const mip::bve_clause_t wrong[1] = {{1u, 1u}};
  EXPECT_FALSE(mip::bve_sanity_check(feas, 2, wrong, 1));
}

// --- the row integerization GATE. block-BVE scales each block row to integers via
// find_scaling_rational (strict caps mirroring row_int_scale) so the projection is exact at
// tolerance 0; a row that will not integerize within the caps must be REJECTED (NaN), never rounded
// into a different model. This pins the accept/reject decision that keeps large / non-rational
// coefficients off the exact-projection path. ---
TEST(block_bve_core, integer_scaling_accepts_rational_rejects_pathological)
{
  // Strict caps matching row_int_scale (maxdnom/maxfinal = BVE_INT_SCALE_MAX = 1e6).
  const double kMaxScale  = 1e12;
  const int64_t kMaxDenom = 1000000;
  const double kMaxFinal  = 1e6;
  const double kIntTol    = 1e-9;
  auto all_integer        = [](double s, const std::vector<double>& v) {
    for (double c : v)
      if (std::abs(s * c - std::round(s * c)) >= 1e-9) return false;
    return true;
  };

  // Fractional-but-rational: {1/2, 1/4, -3/4, 1} integerize (expected multiplier 4).
  {
    std::vector<double> v{0.5, 0.25, -0.75, 1.0};
    double s = cuopt::find_scaling_rational(v, kMaxScale, kMaxDenom, kMaxFinal, kIntTol);
    ASSERT_TRUE(std::isfinite(s)) << "rational coefficients must integerize";
    EXPECT_GT(s, 0.0);
    EXPECT_TRUE(all_integer(s, v));
  }

  // Large integer coefficients stay exact (already integer -> multiplier 1, no rounding).
  {
    std::vector<double> v{1e9, -1e9, 3.0};
    double s = cuopt::find_scaling_rational(v, kMaxScale, kMaxDenom, kMaxFinal, kIntTol);
    ASSERT_TRUE(std::isfinite(s));
    EXPECT_TRUE(all_integer(s, v));
  }

  // Pathological: distinct prime reciprocals need lcm(11,13,17,19,23) = 1062347 > maxfinal (1e6),
  // so no bounded integer multiplier exists -> rejected (NaN), NOT silently rounded.
  {
    std::vector<double> v{1.0 / 11, 1.0 / 13, 1.0 / 17, 1.0 / 19, 1.0 / 23};
    double s = cuopt::find_scaling_rational(v, kMaxScale, kMaxDenom, kMaxFinal, kIntTol);
    EXPECT_TRUE(std::isnan(s)) << "un-integerizable coefficients must be rejected, got " << s;
  }
}

// A cached probe and a block projection are both valid, so they can only disagree when the
// antecedent they share is unsatisfiable. That fixes the variable to the opposite value; the model
// is infeasible only once both polarities are contradicted, which apply_bve_fixings derives from
// two fixings that disagree. An empty intersection on its own must not be reported as global
// infeasibility.
TEST(block_bve_core, cache_contradiction_fixes_the_variable_instead_of_failing)
{
  constexpr int var    = 7;
  constexpr int forced = 9;

  // Probing has x7 = 0 => x9 = 0; the exact projection has x7 = 0 => x9 = 1. Slot 1 is left
  // unpopulated, which also exercises the empty-bound-map guard.
  {
    mip::probing_cache_t<int, double> cache;
    std::array<mip::cache_entry_t<int, double>, 2> entries{};
    entries[0].val_interval                    = {0.0, mip::interval_type_t::EQUALS};
    entries[0].var_to_cached_bound_map[forced] = {0.0, 0.0};
    cache.probing_cache.insert({var, entries});

    std::vector<std::pair<int, bool>> fixings;
    cache.merge_forcings({{var, forced, false, true}}, fixings);

    ASSERT_EQ(fixings.size(), 1u) << "a contradicted probe yields one fixing, not infeasibility";
    EXPECT_EQ(fixings[0].first, var);
    EXPECT_TRUE(fixings[0].second) << "x7 = 0 is disproved, so x7 = 1";
  }

  // Both polarities contradicted: the two disagreeing fixings are what proves infeasibility.
  {
    mip::probing_cache_t<int, double> cache;
    std::array<mip::cache_entry_t<int, double>, 2> entries{};
    entries[0].val_interval                    = {0.0, mip::interval_type_t::EQUALS};
    entries[0].var_to_cached_bound_map[forced] = {0.0, 0.0};
    entries[1].val_interval                    = {1.0, mip::interval_type_t::EQUALS};
    entries[1].var_to_cached_bound_map[forced] = {0.0, 0.0};
    cache.probing_cache.insert({var, entries});

    std::vector<std::pair<int, bool>> fixings;
    cache.merge_forcings({{var, forced, false, true}, {var, forced, true, true}}, fixings);

    ASSERT_EQ(fixings.size(), 2u);
    std::sort(fixings.begin(), fixings.end());
    EXPECT_EQ(fixings[0], std::make_pair(var, false));
    EXPECT_EQ(fixings[1], std::make_pair(var, true));
  }

  // A forcing consistent with the cached interval tightens it and fixes nothing.
  {
    mip::probing_cache_t<int, double> cache;
    std::array<mip::cache_entry_t<int, double>, 2> entries{};
    entries[0].val_interval                    = {0.0, mip::interval_type_t::EQUALS};
    entries[0].var_to_cached_bound_map[forced] = {0.0, 1.0};
    cache.probing_cache.insert({var, entries});

    std::vector<std::pair<int, bool>> fixings;
    cache.merge_forcings({{var, forced, false, true}}, fixings);

    EXPECT_TRUE(fixings.empty()) << "a consistent forcing must not fix anything";
    const auto& bound = cache.probing_cache.at(var)[0].var_to_cached_bound_map.at(forced);
    EXPECT_EQ(bound.lb, 1.0);
    EXPECT_EQ(bound.ub, 1.0);
  }
}

// Build a random block LAYOUT (na/nb/n_rows + sparsity pattern), coefficients/bounds left unset.
// Reps of one shape reuse the SAME layout so they land in one GPU shape-bin (exercising the num>1
// path).
static mip::bve_block_t<double> make_block_layout(std::mt19937& rng, int na, int nb, int n_rows)
{
  const int scope = na + nb;
  mip::bve_block_t<double> blk{};
  blk.na     = na;
  blk.nb     = nb;
  blk.n_rows = n_rows;
  std::uniform_int_distribution<int> present(0, 1);  // is a var in this row
  int nz = 0;
  for (int r = 0; r < n_rows; ++r) {
    blk.row_off[r] = nz;
    for (int v = 0; v < scope; ++v)
      if (present(rng)) blk.row_var[nz++] = v;
    if (nz == blk.row_off[r]) blk.row_var[nz++] = r % scope;  // never leave an empty row
  }
  blk.row_off[n_rows] = nz;
  return blk;
}

// Fill a layout's coefficients (small integers) and bounds (randomly ±inf), leaving the pattern
// fixed.
static void randomize_block_data(std::mt19937& rng, mip::bve_block_t<double>& blk)
{
  const double INF      = std::numeric_limits<double>::infinity();
  const double coefs[4] = {-2.0, -1.0, 1.0, 2.0};
  std::uniform_int_distribution<int> coef_pick(0, 3);
  std::uniform_int_distribution<int> bnd_pick(0, 2);  // 0:[lo,inf] 1:[-inf,up] 2:[lo,up]
  for (int k = 0; k < blk.row_off[blk.n_rows]; ++k)
    blk.row_coef[k] = coefs[coef_pick(rng)];
  for (int r = 0; r < blk.n_rows; ++r) {
    const int terms = blk.row_off[r + 1] - blk.row_off[r];
    // Activity under 0/1 vars and coefs in {-2,-1,1,2} lies in [-2*terms, 2*terms]. Pick finite
    // uppers in [0, 2*terms] so they can bind (not always equal to the loose max activity).
    const double lo = -terms;
    std::uniform_int_distribution<int> up_pick(0, 2 * terms);
    const double up = up_pick(rng);
    const int kind  = bnd_pick(rng);
    blk.row_lo[r]   = (kind == 1) ? -INF : lo;
    blk.row_up[r]   = (kind == 0) ? INF : up;
  }
}

// --- projection correctness: the GPU batch projection must equal the host enumeration oracle on a
//     diverse batch (varied na/nb/rows, ±inf bounds, multiple distinct shapes, and >1-block bins).
//     This is what pins projection correctness; the inline sanity check cannot (it trusts feas).
//     Runs the same function two independent ways and asserts feas + witness agree everywhere.
TEST(block_bve_projection, gpu_batch_matches_host_oracle)
{
  const raft::handle_t handle_{};
  std::mt19937 rng(12345u);

  // several shapes, several blocks each; reps share a layout -> one shape-bin with num>1
  const int shapes[][3] = {{1, 2, 3}, {2, 2, 2}, {1, 3, 4}, {3, 3, 5}, {2, 4, 3}, {4, 2, 4}};
  std::vector<mip::bve_block_t<double>> blocks;
  for (const auto& s : shapes) {
    const mip::bve_block_t<double> layout = make_block_layout(rng, s[0], s[1], s[2]);
    for (int rep = 0; rep < 6; ++rep) {
      mip::bve_block_t<double> blk = layout;
      randomize_block_data(rng, blk);
      blocks.push_back(blk);
    }
  }

  std::vector<mip::bve_candidate_t<int, double>> cands(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i)
    cands[i].blk =
      blocks[i];  // the service reads only .blk; interior/boundary/rows are unused here

  cuopt::timer_t no_deadline(std::numeric_limits<double>::infinity());
  mip::bve_project_batch_gpu<int, double>(handle_, cands, 1e-6, no_deadline);

  for (size_t i = 0; i < blocks.size(); ++i) {
    uint8_t exp_feas[mip::BVE_MAX_PATTERNS];
    uint32_t exp_wit[mip::BVE_MAX_PATTERNS];
    mip::bve_project(blocks[i], 1e-6, exp_feas, exp_wit);
    const int patterns = 1 << blocks[i].nb;
    for (int m = 0; m < patterns; ++m) {
      EXPECT_EQ(cands[i].projection.feasible[m], exp_feas[m]) << "block " << i << " pattern " << m;
      if (exp_feas[m])  // witness only defined for feasible patterns
        EXPECT_EQ(cands[i].projection.witness[m], exp_wit[m]) << "block " << i << " pattern " << m;
    }
  }
}

// Fill a layout with LARGE integer coefficients and integer bounds (all exact fp64 integers, well
// under 2^53), leaving the pattern fixed. This is the shape block-BVE feeds the projection after
// integerization, and the magnitude range where a 1e-6-tolerance fp test would be marginal but
// exact integer arithmetic is not.
static void randomize_block_data_integer(std::mt19937& rng, mip::bve_block_t<double>& blk)
{
  const double INF     = std::numeric_limits<double>::infinity();
  const double coefs[] = {-2e6, -1e6, 1e6, 2e6, 5e6};
  std::uniform_int_distribution<int> coef_pick(0, 4);
  std::uniform_int_distribution<int> bnd_pick(0, 2);  // 0:[lo,inf] 1:[-inf,up] 2:[lo,up]
  for (int k = 0; k < blk.row_off[blk.n_rows]; ++k)
    blk.row_coef[k] = coefs[coef_pick(rng)];
  for (int r = 0; r < blk.n_rows; ++r) {
    const int terms = blk.row_off[r + 1] - blk.row_off[r];
    // Activity lies in [-5e6*terms, 5e6*terms]; pick finite integer bounds (multiples of 1e6) that
    // can bind.
    const double lo = -5e6 * terms;
    std::uniform_int_distribution<int> up_pick(0, 2 * terms);
    const double up = 1e6 * up_pick(rng);
    const int kind  = bnd_pick(rng);
    blk.row_lo[r]   = (kind == 1) ? -INF : lo;
    blk.row_up[r]   = (kind == 0) ? INF : up;
  }
}

// --- the EXACT projection path. Production integerizes each block and projects at tolerance 0;
//     the 1e-6 differential test above never exercises that. On large-integer-coefficient blocks
//     the GPU projection at tol 0 must still equal the host enumeration oracle at tol 0 everywhere.
//     ---
TEST(block_bve_projection, exact_projection_matches_host_at_tol0)
{
  const raft::handle_t handle_{};
  std::mt19937 rng(2024u);

  const int shapes[][3] = {{1, 2, 3}, {2, 2, 2}, {1, 3, 4}, {3, 3, 5}, {2, 4, 3}};
  std::vector<mip::bve_block_t<double>> blocks;
  for (const auto& s : shapes) {
    const mip::bve_block_t<double> layout = make_block_layout(rng, s[0], s[1], s[2]);
    for (int rep = 0; rep < 6; ++rep) {
      mip::bve_block_t<double> blk = layout;
      randomize_block_data_integer(rng, blk);
      blocks.push_back(blk);
    }
  }

  std::vector<mip::bve_candidate_t<int, double>> cands(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i)
    cands[i].blk = blocks[i];

  cuopt::timer_t no_deadline(std::numeric_limits<double>::infinity());
  mip::bve_project_batch_gpu<int, double>(handle_, cands, 0.0, no_deadline);  // exact: tol 0

  for (size_t i = 0; i < blocks.size(); ++i) {
    uint8_t exp_feas[mip::BVE_MAX_PATTERNS];
    uint32_t exp_wit[mip::BVE_MAX_PATTERNS];
    mip::bve_project(blocks[i], 0.0, exp_feas, exp_wit);
    const int patterns = 1 << blocks[i].nb;
    for (int m = 0; m < patterns; ++m) {
      EXPECT_EQ(cands[i].projection.feasible[m], exp_feas[m]) << "block " << i << " pattern " << m;
      if (exp_feas[m])
        EXPECT_EQ(cands[i].projection.witness[m], exp_wit[m]) << "block " << i << " pattern " << m;
    }
  }
}

// --- 3. end-to-end: run the pass on a problem_t, then reconstruct through postsolve ---
TEST(block_bve_presolve, end_to_end_reduction_and_reconstruction)
{
  const raft::handle_t handle_{};
  auto model      = io::read_lp_from_string<int, double>(kBlockLp);
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  problem.presolve_data.initialize_var_mapping(problem, problem.handle_ptr);
  const int n_before = problem.n_variables;

  auto impl_adj = probing_impl_adj(problem);

  cuopt::timer_t bve_timer(10.0);
  double bve_work_units = 0.0;
  const bool applied    = mip::block_bve_presolve(problem, impl_adj, bve_timer, bve_work_units);
  EXPECT_TRUE(applied);
  EXPECT_EQ(problem.n_variables, n_before - 1);  // exactly `a` eliminated

  // Set a reduced solution with the first surviving (boundary) variable = 1; whichever of b/c it
  // is, the block forces a = 1, so a correct reconstruction must satisfy the ORIGINAL constraints.
  std::vector<double> reduced(problem.n_variables, 0.0);
  if (!reduced.empty()) reduced[0] = 1.0;
  rmm::device_uvector<double> assignment(problem.n_variables, handle_.get_stream());
  raft::copy(assignment.data(), reduced.data(), reduced.size(), handle_.get_stream());
  problem.presolve_data.post_process_assignment(problem, assignment, /*resize_to_original=*/true);
  auto full = cuopt::host_copy(assignment, handle_.get_stream());
  handle_.sync_stream();

  ASSERT_EQ(full.size(), static_cast<size_t>(n_before));  // expanded back to all three variables
  // The reconstructed full assignment must satisfy EVERY original constraint. This is order-
  // independent (no assumption about which index is a/b/c): if the eliminated aux is reconstructed
  // wrongly, a - b >= 0 or a - c >= 0 is violated. Since one boundary variable is set to 1, a
  // correct reconstruction forces the aux to 1, so the feasibility check below is exactly that
  // correctness test.
  auto m_off = model.get_constraint_matrix_offsets();
  auto m_var = model.get_constraint_matrix_indices();
  auto m_val = model.get_constraint_matrix_values();
  auto m_rl  = model.get_constraint_lower_bounds();
  auto m_ru  = model.get_constraint_upper_bounds();
  for (size_t r = 0; r + 1 < m_off.size(); ++r) {
    double s = 0.0;
    for (int k = m_off[r]; k < m_off[r + 1]; ++k)
      s += m_val[k] * full[m_var[k]];
    EXPECT_GE(s, m_rl[r] - 1e-6);
    EXPECT_LE(s, m_ru[r] + 1e-6);
  }
}

// --- end-to-end with fractional block coefficients (the same gadget, rows scaled by 1/2). The
//     reduction and its reconstruction must be identical to the integer gadget: block-BVE has to
//     integerize the 0.5 coefficients before the exact projection and undo it correctly at
//     postsolve.
TEST(block_bve_presolve, fractional_gadget_reduces_and_reconstructs)
{
  const raft::handle_t handle_{};
  auto model      = io::read_lp_from_string<int, double>(kFractionalBlockLp);
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  problem.presolve_data.initialize_var_mapping(problem, problem.handle_ptr);
  const int n_before = problem.n_variables;

  auto impl_adj = probing_impl_adj(problem);

  cuopt::timer_t bve_timer(10.0);
  double bve_work_units = 0.0;
  const bool applied    = mip::block_bve_presolve(problem, impl_adj, bve_timer, bve_work_units);
  // Probing/trivial may already have eliminated the aux; either way exactly one variable is gone.
  EXPECT_TRUE(applied || problem.n_variables < n_before);
  ASSERT_EQ(problem.n_variables, n_before - 1) << "fractional gadget did not eliminate the aux";

  // Set the first surviving (boundary) variable to 1; a correct reconstruction forces the aux so
  // the full assignment satisfies every ORIGINAL (fractional) constraint.
  std::vector<double> reduced(problem.n_variables, 0.0);
  if (!reduced.empty()) reduced[0] = 1.0;
  rmm::device_uvector<double> assignment(problem.n_variables, handle_.get_stream());
  raft::copy(assignment.data(), reduced.data(), reduced.size(), handle_.get_stream());
  problem.presolve_data.post_process_assignment(problem, assignment, /*resize_to_original=*/true);
  auto full = cuopt::host_copy(assignment, handle_.get_stream());
  handle_.sync_stream();

  ASSERT_EQ(full.size(), static_cast<size_t>(n_before));
  auto m_off = model.get_constraint_matrix_offsets();
  auto m_var = model.get_constraint_matrix_indices();
  auto m_val = model.get_constraint_matrix_values();
  auto m_rl  = model.get_constraint_lower_bounds();
  auto m_ru  = model.get_constraint_upper_bounds();
  for (size_t r = 0; r + 1 < m_off.size(); ++r) {
    double s = 0.0;
    for (int k = m_off[r]; k < m_off[r + 1]; ++k)
      s += m_val[k] * full[m_var[k]];
    EXPECT_GE(s, m_rl[r] - 1e-6);
    EXPECT_LE(s, m_ru[r] + 1e-6);
  }
}

// Brute-force the (small, binary) reduced problem_t: enumerate all 2^n assignments, return whether
// any is feasible, the min solver-space objective, and its argmin.
struct bve_bf_t {
  bool found;
  double solver_obj;
  std::vector<double> x;
};
static bve_bf_t brute_force_binary(mip::problem_t<int, double>& problem)
{
  auto stream = problem.handle_ptr->get_stream();
  auto h_off  = cuopt::host_copy(problem.offsets, stream);
  auto h_var  = cuopt::host_copy(problem.variables, stream);
  auto h_coef = cuopt::host_copy(problem.coefficients, stream);
  auto h_clb  = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  auto h_cub  = cuopt::host_copy(problem.constraint_upper_bounds, stream);
  auto h_obj  = cuopt::host_copy(problem.objective_coefficients, stream);
  auto h_vb   = cuopt::host_copy(problem.variable_bounds, stream);
  problem.handle_ptr->sync_stream();

  const int nv = problem.n_variables;
  const int nr = problem.n_constraints;
  for (int v = 0; v < nv; ++v) {  // corpus is pure 0-1
    EXPECT_NEAR(get_lower(h_vb[v]), 0.0, 1e-9);
    EXPECT_NEAR(get_upper(h_vb[v]), 1.0, 1e-9);
  }

  bve_bf_t r{false, 0.0, {}};
  const double eps     = 1e-6;
  const uint64_t total = (nv >= 63) ? 0 : (uint64_t{1} << nv);
  std::vector<double> x(nv);
  for (uint64_t mask = 0; mask < total; ++mask) {
    for (int v = 0; v < nv; ++v)
      x[v] = (mask >> v) & 1u;
    bool ok = true;
    for (int rr = 0; rr < nr && ok; ++rr) {
      double s = 0.0;
      for (int k = h_off[rr]; k < h_off[rr + 1]; ++k)
        s += h_coef[k] * x[h_var[k]];
      if (s < h_clb[rr] - eps || s > h_cub[rr] + eps) ok = false;
    }
    if (!ok) continue;
    double obj = 0.0;
    for (int v = 0; v < nv; ++v)
      obj += h_obj[v] * x[v];
    if (!r.found || obj < r.solver_obj - eps) {
      r.found      = true;
      r.solver_obj = obj;
      r.x          = x;
    }
  }
  return r;
}

TEST(block_bve_regression, all_feasible_projection_can_remove_the_last_rows)
{
  const raft::handle_t handle_{};
  auto model = io::read_mps<int, double>(
    make_path_absolute("mip/block_bve/all_feasible_projection.mps"), /*fixed_format=*/false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  problem.presolve_data.initialize_var_mapping(problem, problem.handle_ptr);

  bool probing_infeasible = false;
  auto impl_adj           = probing_impl_adj(problem, &probing_infeasible);
  ASSERT_FALSE(probing_infeasible) << "probing must leave the all-feasible projection for BVE";
  ASSERT_TRUE(
    std::any_of(impl_adj.begin(), impl_adj.end(), [](const auto& adj) { return !adj.empty(); }))
    << "fixture must provide an implication edge for the zero-objective auxiliary";

  cuopt::timer_t bve_timer(10.0);
  double bve_work_units = 0.0;
  bool applied          = false;
  ASSERT_NO_THROW(applied = mip::block_bve_presolve(problem, impl_adj, bve_timer, bve_work_units));
  ASSERT_TRUE(applied) << "the all-feasible projection should eliminate its auxiliary";

  ASSERT_LE(problem.n_variables, 24);
  const auto bf = brute_force_binary(problem);
  ASSERT_TRUE(bf.found) << "eliminating a tautological projection changed feasibility";
  EXPECT_NEAR(bf.solver_obj, 0.0, 1e-6);
}

TEST(block_bve_regression, all_infeasible_projection_remains_infeasible)
{
  const raft::handle_t handle_{};
  auto model = io::read_mps<int, double>(
    make_path_absolute("mip/block_bve/all_infeasible_projection.mps"), /*fixed_format=*/false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  problem.presolve_data.initialize_var_mapping(problem, problem.handle_ptr);

  bool probing_infeasible = false;
  auto impl_adj           = probing_impl_adj(problem, &probing_infeasible);
  ASSERT_FALSE(probing_infeasible)
    << "the fixture must reach BVE instead of being discharged by probing";
  ASSERT_TRUE(
    std::any_of(impl_adj.begin(), impl_adj.end(), [](const auto& adj) { return !adj.empty(); }))
    << "fixture must provide an implication edge for the zero-objective auxiliary";

  cuopt::timer_t bve_timer(10.0);
  double bve_work_units = 0.0;
  ASSERT_NO_THROW((void)mip::block_bve_presolve(problem, impl_adj, bve_timer, bve_work_units));

  ASSERT_LE(problem.n_variables, 24);
  const auto bf = brute_force_binary(problem);
  EXPECT_FALSE(bf.found) << "BVE lost the empty clause produced by the projection";
}

// Corpus of small 0-1 instances whose optima were cross-checked OFFLINE by brute force AND HiGHS.
// MPS live in datasets/mip/block_bve/; optima inlined here. Mix: gadget-rich (block-BVE fires),
// no-op/soundness (aux-with-objective, random feasible ILPs), and infeasible.
struct bve_case_t {
  const char* file;
  bool feasible;
  double optimum;
  bool expect_reduce;  // gadget should shrink via probing and/or block-BVE
};
static const bve_case_t kBveCases[] = {
  {"mip/block_bve/or_used.mps", true, 1.0, true},
  {"mip/block_bve/and_used.mps", true, -2.0, true},
  {"mip/block_bve/neq_used.mps", true, -3.0, true},
  {"mip/block_bve/chain_or.mps", true, 1.0, true},
  {"mip/block_bve/two_gadgets.mps", true, 2.0, true},
  {"mip/block_bve/heavy_reduce.mps", true, 2.0, true},
  {"mip/block_bve/aux_with_obj.mps", true, 4.0, false},
  {"mip/block_bve/mixed.mps", true, -1.0, false},
  {"mip/block_bve/infeasible.mps", false, 0.0, false},
  {"mip/block_bve/random_a.mps", true, -3.0, false},
  {"mip/block_bve/random_b.mps", true, -5.0, false},
  {"mip/block_bve/random_c.mps", true, -1.0, false},
};

// End-to-end equivalence: for each corpus instance, run the pass, brute-force the reduced model,
// and assert block-BVE preserved the answer. block-BVE is a PRIMAL, optimum-preserving reduction,
// so the bar is: reduced optimum == known optimum, the reduced optimum reconstructs to an
// ORIGINAL-feasible point with that objective, and infeasibility is preserved. This stresses the
// full detect -> project
// -> commit -> install -> reconstruct chain (incl. variable_mapping + witness replay), which the
// component tests above don't.
TEST(block_bve_equivalence, preserves_optimum_and_reconstruction_on_corpus)
{
  const raft::handle_t handle_{};
  bool any_reduced = false;
  for (const auto& c : kBveCases) {
    SCOPED_TRACE(c.file);
    auto model      = io::read_mps<int, double>(make_path_absolute(c.file), /*fixed_format=*/false);
    auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
    mip::problem_t<int, double> problem(op_problem);
    problem.preprocess_problem();
    problem.presolve_data.initialize_var_mapping(problem, problem.handle_ptr);
    const int n_before = problem.n_variables;

    bool probing_infeas = false;
    auto impl_adj       = probing_impl_adj(problem, &probing_infeas);
    if (probing_infeas) {
      EXPECT_FALSE(c.feasible) << "probing proved infeasible on a feasible instance";
      continue;
    }

    cuopt::timer_t bve_timer(10.0);
    double bve_work_units = 0.0;
    const bool applied    = mip::block_bve_presolve(problem, impl_adj, bve_timer, bve_work_units);
    // Probing/trivial may already have eliminated the aux; BVE then correctly no-ops.
    if (applied || problem.n_variables < n_before) { any_reduced = true; }
    if (applied) {
      EXPECT_LT(problem.n_variables, n_before) << "applied but variable count unchanged";
    }
    if (c.expect_reduce) {
      EXPECT_LT(problem.n_variables, n_before)
        << "gadget fixture expected a reduction via probing and/or block-BVE";
    }

    ASSERT_LE(problem.n_variables, 24) << "brute force enumerates 2^n; keep the corpus small";
    auto bf = brute_force_binary(problem);
    if (!c.feasible) {
      // NOTE: if preprocess detects the infeasibility upstream and collapses the model, this may
      // need to become a problem-status check instead of a no-feasible-point check.
      EXPECT_FALSE(bf.found) << "reduced model is feasible but the instance is infeasible";
      continue;
    }
    ASSERT_TRUE(bf.found) << "reduced model is infeasible but the instance is feasible";

    // The reduced optimum must reconstruct to an ORIGINAL-feasible point whose ORIGINAL objective
    // equals the known optimum. This is offset/scaling-independent (evaluated directly on the
    // original model) and catches both directions: a cut optimum -> recon_obj > optimum; a spurious
    // better solution -> either the reconstruction is original-infeasible or recon_obj < optimum.
    rmm::device_uvector<double> assignment(problem.n_variables, handle_.get_stream());
    raft::copy(assignment.data(), bf.x.data(), bf.x.size(), handle_.get_stream());
    problem.presolve_data.post_process_assignment(problem, assignment, /*resize_to_original=*/true);
    auto full = cuopt::host_copy(assignment, handle_.get_stream());
    handle_.sync_stream();

    auto m_off = model.get_constraint_matrix_offsets();
    auto m_var = model.get_constraint_matrix_indices();
    auto m_val = model.get_constraint_matrix_values();
    auto m_rl  = model.get_constraint_lower_bounds();
    auto m_ru  = model.get_constraint_upper_bounds();
    for (size_t r = 0; r + 1 < m_off.size(); ++r) {
      double s = 0.0;
      for (int k = m_off[r]; k < m_off[r + 1]; ++k)
        s += m_val[k] * full[m_var[k]];
      EXPECT_GE(s, m_rl[r] - 1e-6);
      EXPECT_LE(s, m_ru[r] + 1e-6);
    }
    auto m_obj = model.get_objective_coefficients();
    ASSERT_EQ(full.size(), m_obj.size()) << "reconstruction is not in the original column frame";
    double recon_obj = 0.0;
    for (size_t j = 0; j < m_obj.size(); ++j)
      recon_obj += m_obj[j] * full[j];
    EXPECT_NEAR(recon_obj, c.optimum, 1e-6);
  }
  EXPECT_TRUE(any_reduced) << "corpus exercised no probing/block-BVE reduction path";
}

// Drive production MIP presolve (Papilo → cuOpt run_presolve) and optionally assert
// upper bounds on the reduced size. Pass std::numeric_limits<int>::max() for a
// dimension to skip that check.
static void run_presolve_size_check(const char* relative_mps_path,
                                    int max_vars = std::numeric_limits<int>::max(),
                                    int max_rows = std::numeric_limits<int>::max())
{
  const raft::handle_t handle_{};
  auto model      = io::read_mps<int, double>(make_path_absolute(relative_mps_path),
                                         /*fixed_format=*/false);
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
  sort_csr(op_problem);

  mip_solver_settings_t<int, double> settings{};
  settings.presolver = presolver_t::Papilo;
  settings.probing   = true;
  settings.block_bve = true;

  auto papilo = std::make_unique<mip::third_party_presolve_t<int, double>>();
  auto result = papilo->apply_presolve_from_op_problem(op_problem,
                                                       problem_category_t::MIP,
                                                       settings.presolver,
                                                       /*dual_postsolve=*/false,
                                                       settings.tolerances.absolute_tolerance,
                                                       settings.tolerances.relative_tolerance,
                                                       /*time_limit=*/60.0,
                                                       /*num_cpu_threads=*/0);
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::INFEASIBLE)
    << relative_mps_path << " infeasible after Papilo";
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::UNBNDORINFEAS)
    << relative_mps_path << " unbounded-or-infeasible after Papilo";
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::UNBOUNDED)
    << relative_mps_path << " unbounded after Papilo";

  mip::problem_t<int, double> problem(result.reduced_problem);
  problem.set_papilo_presolve_data(papilo.get(),
                                   result.reduced_to_original_map,
                                   result.original_to_reduced_map,
                                   op_problem.get_n_variables());
  problem.set_implied_integers(result.implied_integer_indices);
  problem.preprocess_problem();
  mip::trivial_presolve(problem, /*remap_cache_ids=*/true);  // mirrors solve.cu's setup

  cuopt::timer_t timer(120.0);
  mip::mip_solver_t<int, double> solver(problem, settings, timer);
  problem.tolerances = settings.get_tolerances();
  mip::diversity_manager_t<int, double> dm(solver.context);

  bool presolve_ok = false;
  with_mip_omp_team([&]() { presolve_ok = dm.run_presolve(/*time_limit=*/60.0, timer); });

  ASSERT_TRUE(presolve_ok) << relative_mps_path << " cuOpt run_presolve failed";
  if (max_vars != std::numeric_limits<int>::max()) {
    EXPECT_LT(problem.n_variables, max_vars)
      << relative_mps_path << " reduced n_variables=" << problem.n_variables;
  }
  if (max_rows != std::numeric_limits<int>::max()) {
    EXPECT_LT(problem.n_constraints, max_rows)
      << relative_mps_path << " reduced n_constraints=" << problem.n_constraints;
  }
}

TEST(block_bve_presolve, bnatt400_reduces_below_500_vars)
{
  run_presolve_size_check("mip/bnatt400.mps", /*max_vars=*/500);
}

TEST(block_bve_presolve, bnatt500_reduces_below_500_vars)
{
  run_presolve_size_check("mip/bnatt500.mps", /*max_vars=*/500);
}

}  // namespace cuopt::mathematical_optimization::test
