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
#include <mip_heuristics/presolve/block_bve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/utils.cuh>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// ============================================================================================
// TEST-ONLY reference/oracle for the block-BVE pass (NOT part of production — in the pass,
// projection runs on the GPU). Gives the tests a trusted, independent yardstick, reopening
// namespace ...::mip so the tests below can call e.g. mip::bve_project_and_check:
//   * bve_project / bve_project_and_check — the host ENUMERATION projection, ported bit-for-bit
//   from
//     the validated reference cpufj_sc22/bve_blocks.cpp (itself checked against `bveblk`). The
//     differential oracle for the GPU kernel: for any block, the GPU's feas/witness must equal
//     these. This is what pins projection correctness, which the inline sanity check
//     (bve_sanity_check) does NOT — the sanity check trusts feas and only verifies the clauses
//     reproduce it.
//   * bve_detect_closure / bve_detect_minfill — the sequential host-projection detectors: coverage
//   and
//     parity references for the production bve_detect_closure_batched (minfill reproduces bveblk's
//     block counts exactly). bve_host_try_commit is the host stage->project->commit they share.
// ============================================================================================
namespace cuopt::mathematical_optimization::mip {

// ---- host enumeration projection (the differential oracle) ----

template <typename f_t>
inline bool bve_is_finite(f_t x)
{
  // finite iff it equals itself (rules out NaN) and is strictly within +/- inf
  return (x == x) && (x < static_cast<f_t>(INFINITY)) && (x > static_cast<f_t>(-INFINITY));
}

// Feasibility of one packed row under a full local assignment `val` (length na+nb), with tolerance.
template <typename f_t>
inline bool bve_row_sat(const bve_block_t<f_t>& blk, int r, const int* val, f_t tol)
{
  f_t s = 0;
  for (int k = blk.row_off[r]; k < blk.row_off[r + 1]; ++k) {
    s += blk.row_coef[k] * static_cast<f_t>(val[blk.row_var[k]]);
  }
  if (bve_is_finite(blk.row_up[r]) && s > blk.row_up[r] + tol) return false;
  if (bve_is_finite(blk.row_lo[r]) && s < blk.row_lo[r] - tol) return false;
  return true;
}

// Project the block onto its boundary. `feas[m]` (length 2^nb) is set to 1 iff boundary pattern m
// (nb bits) admits SOME interior assignment satisfying every block row, and `witness[m]` receives
// the packed interior assignment (na bits) of the FIRST feasible completion. Both are left 0 for
// infeasible patterns. Mirrors the double loop in bve_blocks.cpp; the GPU kernel must match this.
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

// Full per-block core on the host: project -> prime-implicate CNF -> growth gate -> inline sanity
// check. The production commit_projected does the same, but reads feas/witness from the GPU instead
// of the host bve_project above.
template <typename f_t>
inline bve_status_t bve_project_and_check(const bve_block_t<f_t>& blk,
                                          f_t tol,
                                          int margin,
                                          bve_clause_t* clauses,
                                          int* n_clauses,
                                          uint32_t* witness)
{
  *n_clauses = 0;
  if (blk.nb <= 0 || blk.nb > BVE_MAX_BOUNDARY) return bve_status_t::kSkipCaps;
  if (blk.na < 0 || blk.na + blk.nb > BVE_MAX_SCOPE) return bve_status_t::kSkipCaps;
  if (blk.n_rows < 0 || blk.n_rows > BVE_MAX_ROWS) return bve_status_t::kSkipCaps;

  uint8_t feas[BVE_MAX_PATTERNS];
  bve_project(blk, tol, feas, witness);
  const int nc = bve_prime_implicates(feas, blk.nb, clauses, BVE_MAX_CLAUSES);
  if (nc < 0) return bve_status_t::kSkipGrowth;  // clause explosion past cap
  if (nc > blk.n_rows + margin) return bve_status_t::kSkipGrowth;
  if (!bve_sanity_check(feas, blk.nb, clauses, nc)) return bve_status_t::kSkipCheckFailed;
  *n_clauses = nc;
  return bve_status_t::kReduced;
}

// ---- host reference detectors ----

// Host stage -> host bve_project -> commit_projected. The monolithic path the reference detectors
// use (the production pass instead batches many stages into one GPU launch, then commit_projected
// each).
template <typename i_t, typename f_t>
inline bool bve_host_try_commit(bve_reducer_t<i_t, f_t>& R, const std::vector<i_t>& interior_in)
{
  bve_candidate_t<i_t, f_t> cand;
  if (!R.stage(interior_in, cand)) return false;
  bve_project(cand.blk, R.tol, cand.feas, cand.witness);
  return R.commit_projected(cand);
}

template <typename i_t, typename f_t>
inline std::vector<i_t> bve_seed_order(const bve_reducer_t<i_t, f_t>& R)
{
  std::vector<i_t> order;
  for (i_t c = 0; c < R.n_vars; ++c)
    if (R.is_bin[c] && !R.col2rows[c].empty() && !R.obj_nz[c]) order.push_back(c);
  std::sort(order.begin(), order.end(), [&](i_t a, i_t b) {
    return R.col2rows[a].size() < R.col2rows[b].size();
  });
  return order;
}

// PRODUCTION-EQUIVALENT reference (sequential host projection): implication-closure block growth
// over the probing-cache adjacency using min-fill's shrink criterion. Coverage/parity reference for
// the production bve_detect_closure_batched.
template <typename i_t, typename f_t>
bve_plan_t<i_t, f_t> bve_detect_closure(bve_reducer_t<i_t, f_t>& R,
                                        const std::vector<std::vector<i_t>>& impl_adj,
                                        double tbudget_s = 180.0)
{
  auto has_adj = [&](i_t v) {
    return static_cast<size_t>(v) < impl_adj.size() && !impl_adj[v].empty();
  };
  auto eligible = [&](i_t w) {
    return R.is_bin[w] && !R.obj_nz[w] && !R.done[w] && !R.col2rows[w].empty();
  };
  std::vector<i_t> order;
  for (i_t c = 0; c < R.n_vars; ++c)
    if (R.is_bin[c] && !R.obj_nz[c] && !R.col2rows[c].empty() && has_adj(c)) order.push_back(c);
  std::sort(order.begin(), order.end(), [&](i_t a, i_t b) {
    return R.col2rows[a].size() < R.col2rows[b].size();
  });

  auto t0 = std::chrono::steady_clock::now();
  for (i_t seed : order) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > tbudget_s)
      break;
    if (R.done[seed] || R.col2rows[seed].empty()) continue;

    std::unordered_set<i_t> A = {seed};
    // absorb the implication-connected candidate that most shrinks the boundary, until none does
    for (;;) {
      std::vector<i_t> Av(A.begin(), A.end());
      const int cur = R.boundary_size(Av);
      std::unordered_set<i_t> cands;
      for (i_t a : A)
        if (has_adj(a))
          for (i_t w : impl_adj[a])
            if (!A.count(w) && eligible(w)) cands.insert(w);
      i_t best    = static_cast<i_t>(-1);
      int best_nb = cur;
      for (i_t w : cands) {
        std::vector<i_t> cand = Av;
        cand.push_back(w);
        const int na = static_cast<int>(cand.size());
        const int nb = R.boundary_size(cand);
        if (nb < best_nb && na + nb <= R.enumcap && na <= BVE_MAX_INTERIOR) {
          best_nb = nb;
          best    = w;
        }
      }
      if (best < 0) break;
      A.insert(best);
    }
    std::vector<i_t> interior(A.begin(), A.end());
    bve_host_try_commit(R, interior);
  }
  return R.finalize();
}

// ORACLE / COVERAGE REFERENCE: faithful port of bve_blocks.cpp min-fill growth. Reproduces bveblk's
// block counts bit-for-bit; used to validate the projection core and as a coverage target.
template <typename i_t, typename f_t>
bve_plan_t<i_t, f_t> bve_detect_minfill(bve_reducer_t<i_t, f_t>& R, double tbudget_s = 180.0)
{
  std::vector<i_t> order = bve_seed_order(R);
  auto t0                = std::chrono::steady_clock::now();
  for (i_t seed : order) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > tbudget_s)
      break;
    if (R.done[seed] || R.col2rows[seed].empty()) continue;

    std::unordered_set<i_t> A = {seed};
    std::unordered_set<i_t> G(R.col2rows[seed].begin(), R.col2rows[seed].end());
    while (static_cast<int>(A.size()) < 40) {
      std::vector<i_t> bnd = R.boundary_of(G, A);
      int bestsz           = static_cast<int>(bnd.size());
      i_t bestv            = static_cast<i_t>(-1);
      std::unordered_set<i_t> bestG;
      for (i_t v : bnd) {
        if (!R.is_bin[v] || R.obj_nz[v]) continue;
        std::unordered_set<i_t> nG = G;
        for (i_t r : R.col2rows[v])
          nG.insert(r);
        std::unordered_set<i_t> nA = A;
        nA.insert(v);
        int nsz = static_cast<int>(R.boundary_of(nG, nA).size());
        if (nsz < bestsz) {
          bestsz = nsz;
          bestv  = v;
          bestG  = std::move(nG);
        }
      }
      if (bestv < 0) break;
      A.insert(bestv);
      G = std::move(bestG);
    }
    std::vector<i_t> interior(A.begin(), A.end());
    bve_host_try_commit(R, interior);
  }
  return R.finalize();
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
    const double lo = -static_cast<double>(terms);  // reachable given ±2 coeffs and 0/1 vars
    const double up = static_cast<double>(2 * terms);
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

  mip::bve_project_batch_gpu<int, double>(handle_, cands, 1e-6);

  for (size_t i = 0; i < blocks.size(); ++i) {
    uint8_t exp_feas[mip::BVE_MAX_PATTERNS];
    uint32_t exp_wit[mip::BVE_MAX_PATTERNS];
    mip::bve_project(blocks[i], 1e-6, exp_feas, exp_wit);
    const int patterns = 1 << blocks[i].nb;
    for (int m = 0; m < patterns; ++m) {
      EXPECT_EQ(cands[i].feas[m], exp_feas[m]) << "block " << i << " pattern " << m;
      if (exp_feas[m])  // witness only defined for feasible patterns
        EXPECT_EQ(cands[i].witness[m], exp_wit[m]) << "block " << i << " pattern " << m;
    }
  }
}

// helper: extract host CSR + bounds + types + obj from a parsed model
static void model_to_host(const io::mps_data_model_t<int, double>& m,
                          std::vector<int>& offsets,
                          std::vector<int>& variables,
                          std::vector<double>& coefficients,
                          std::vector<double>& row_lower,
                          std::vector<double>& row_upper,
                          std::vector<double>& col_lower,
                          std::vector<double>& col_upper,
                          std::vector<uint8_t>& is_integer,
                          std::vector<double>& obj)
{
  offsets      = m.get_constraint_matrix_offsets();
  variables    = m.get_constraint_matrix_indices();
  coefficients = m.get_constraint_matrix_values();
  row_lower    = m.get_constraint_lower_bounds();
  row_upper    = m.get_constraint_upper_bounds();
  col_lower    = m.get_variable_lower_bounds();
  col_upper    = m.get_variable_upper_bounds();
  obj          = m.get_objective_coefficients();
  auto types   = m.get_variable_types();  // mps_data_model uses 'I'/'C' chars, not var_t
  is_integer.resize(types.size());
  for (size_t i = 0; i < types.size(); ++i)
    is_integer[i] = (types[i] == 'I') ? 1 : 0;
}

// build a proxy implication adjacency: binary vars co-occurring in a row are connected
static std::vector<std::vector<int>> row_share_adjacency(int n_vars,
                                                         const std::vector<int>& offsets,
                                                         const std::vector<int>& variables,
                                                         const std::vector<uint8_t>& is_integer,
                                                         const std::vector<double>& col_lower,
                                                         const std::vector<double>& col_upper)
{
  std::vector<std::unordered_set<int>> adj(n_vars);
  const int n_rows = static_cast<int>(offsets.size()) - 1;
  for (int r = 0; r < n_rows; ++r) {
    std::vector<int> bins;
    for (int k = offsets[r]; k < offsets[r + 1]; ++k) {
      int c = variables[k];
      if (is_integer[c] && col_lower[c] == 0.0 && col_upper[c] == 1.0) bins.push_back(c);
    }
    for (size_t i = 0; i < bins.size(); ++i)
      for (size_t j = i + 1; j < bins.size(); ++j) {
        adj[bins[i]].insert(bins[j]);
        adj[bins[j]].insert(bins[i]);
      }
  }
  std::vector<std::vector<int>> out(n_vars);
  for (int v = 0; v < n_vars; ++v)
    out[v].assign(adj[v].begin(), adj[v].end());
  return out;
}

// --- 3. closure detector eliminates the aux and emits the one no-good ---
TEST(block_bve_detect, closure_eliminates_aux)
{
  auto model = io::read_lp_from_string<int, double>(kBlockLp);
  std::vector<int> offsets, variables;
  std::vector<double> coefficients, row_lower, row_upper, col_lower, col_upper, obj;
  std::vector<uint8_t> is_integer;
  model_to_host(model,
                offsets,
                variables,
                coefficients,
                row_lower,
                row_upper,
                col_lower,
                col_upper,
                is_integer,
                obj);
  const int n_vars = static_cast<int>(col_lower.size());
  const int n_rows = static_cast<int>(offsets.size()) - 1;
  auto impl_adj = row_share_adjacency(n_vars, offsets, variables, is_integer, col_lower, col_upper);

  mip::bve_reducer_t<int, double> reducer(n_vars,
                                          n_rows,
                                          offsets,
                                          variables,
                                          coefficients,
                                          row_lower,
                                          row_upper,
                                          col_lower,
                                          col_upper,
                                          is_integer,
                                          obj,
                                          1e-6,
                                          mip::BVE_MAX_BOUNDARY,
                                          mip::BVE_MAX_SCOPE,
                                          0);
  auto plan = mip::bve_detect_closure(reducer, impl_adj, 30.0);

  EXPECT_EQ(plan.n_blocks, 1);
  EXPECT_EQ(plan.n_elim_cols, 1);  // exactly `a`
  EXPECT_EQ(plan.reductions.size(), 1u);
  EXPECT_EQ(plan.reductions[0].interior.size(), 1u);
  EXPECT_EQ(plan.reductions[0].boundary.size(), 2u);  // b and c
  EXPECT_EQ(plan.added_rows.size(), 1u);              // the b + c <= 1 no-good
}

// --- 4. end-to-end: run the pass on a problem_t, then reconstruct through postsolve ---
TEST(block_bve_presolve, end_to_end_reduction_and_reconstruction)
{
  const raft::handle_t handle_{};
  auto model      = io::read_lp_from_string<int, double>(kBlockLp);
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  problem.presolve_data.initialize_var_mapping(problem, problem.handle_ptr);
  const int n_before = problem.n_variables;

  // proxy implication adjacency from the current CSR (bypasses the probing cache in the test)
  auto h_off = cuopt::host_copy(problem.offsets, handle_.get_stream());
  auto h_var = cuopt::host_copy(problem.variables, handle_.get_stream());
  auto h_vb  = cuopt::host_copy(problem.variable_bounds, handle_.get_stream());
  auto h_vt  = cuopt::host_copy(problem.variable_types, handle_.get_stream());
  handle_.sync_stream();
  std::vector<int> offsets(h_off.begin(), h_off.end()), variables(h_var.begin(), h_var.end());
  std::vector<uint8_t> is_integer(problem.n_variables);
  std::vector<double> col_lower(problem.n_variables), col_upper(problem.n_variables);
  for (int c = 0; c < problem.n_variables; ++c) {
    col_lower[c]  = get_lower(h_vb[c]);
    col_upper[c]  = get_upper(h_vb[c]);
    is_integer[c] = (h_vt[c] == var_t::INTEGER) ? 1 : 0;
  }
  auto impl_adj =
    row_share_adjacency(problem.n_variables, offsets, variables, is_integer, col_lower, col_upper);

  const bool applied = mip::block_bve_presolve(problem, impl_adj);
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
  // correct reconstruction forces the aux to 1 — the feasibility check below is exactly that
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

// proxy implication adjacency from the CURRENT problem_t CSR (bypasses the probing cache, as in the
// end-to-end test above)
static std::vector<std::vector<int>> proxy_impl_adj(mip::problem_t<int, double>& problem)
{
  auto stream = problem.handle_ptr->get_stream();
  auto h_off  = cuopt::host_copy(problem.offsets, stream);
  auto h_var  = cuopt::host_copy(problem.variables, stream);
  auto h_vb   = cuopt::host_copy(problem.variable_bounds, stream);
  auto h_vt   = cuopt::host_copy(problem.variable_types, stream);
  problem.handle_ptr->sync_stream();
  std::vector<int> offsets(h_off.begin(), h_off.end()), variables(h_var.begin(), h_var.end());
  std::vector<uint8_t> is_integer(problem.n_variables);
  std::vector<double> col_lower(problem.n_variables), col_upper(problem.n_variables);
  for (int c = 0; c < problem.n_variables; ++c) {
    col_lower[c]  = get_lower(h_vb[c]);
    col_upper[c]  = get_upper(h_vb[c]);
    is_integer[c] = (h_vt[c] == var_t::INTEGER) ? 1 : 0;
  }
  return row_share_adjacency(
    problem.n_variables, offsets, variables, is_integer, col_lower, col_upper);
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
  EXPECT_LE(nv, 24) << "brute force needs a small reduced model";
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
      x[v] = static_cast<double>((mask >> v) & 1u);
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

// Corpus of small 0-1 instances whose optima were cross-checked OFFLINE by brute force AND HiGHS.
// MPS live in datasets/mip/block_bve/ (generated by cpufj_sc22/bve_gen_fixtures.py); optima inlined
// here. Mix: gadget-rich (block-BVE fires), no-op/soundness (aux-with-objective, random feasible
// ILPs), and infeasible.
struct bve_case_t {
  const char* file;
  bool feasible;
  double optimum;
};
static const bve_case_t kBveCases[] = {
  {"mip/block_bve/or_used.mps", true, 1.0},
  {"mip/block_bve/and_used.mps", true, -2.0},
  {"mip/block_bve/neq_used.mps", true, -3.0},
  {"mip/block_bve/chain_or.mps", true, 1.0},
  {"mip/block_bve/two_gadgets.mps", true, 2.0},
  {"mip/block_bve/heavy_reduce.mps", true, 2.0},
  {"mip/block_bve/aux_with_obj.mps", true, 4.0},
  {"mip/block_bve/mixed.mps", true, -1.0},
  {"mip/block_bve/infeasible.mps", false, 0.0},
  {"mip/block_bve/random_a.mps", true, -3.0},
  {"mip/block_bve/random_b.mps", true, -5.0},
  {"mip/block_bve/random_c.mps", true, -1.0},
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
  for (const auto& c : kBveCases) {
    SCOPED_TRACE(c.file);
    auto model      = io::read_mps<int, double>(make_path_absolute(c.file), /*fixed_format=*/false);
    auto op_problem = mps_data_model_to_optimization_problem(&handle_, model);
    mip::problem_t<int, double> problem(op_problem);
    problem.preprocess_problem();
    problem.presolve_data.initialize_var_mapping(problem, problem.handle_ptr);

    auto impl_adj = proxy_impl_adj(problem);
    mip::block_bve_presolve(problem, impl_adj);

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
    auto m_obj       = model.get_objective_coefficients();
    double recon_obj = 0.0;
    for (size_t j = 0; j < m_obj.size() && j < full.size(); ++j)
      recon_obj += m_obj[j] * full[j];
    EXPECT_NEAR(recon_obj, c.optimum, 1e-6);
  }
}

}  // namespace cuopt::mathematical_optimization::test
