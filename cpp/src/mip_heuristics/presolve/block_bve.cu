/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "block_bve.cuh"
#include "trivial_presolve.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/presolve_data.cuh>
#include <mip_heuristics/utils.cuh>

#include <utilities/integer_scaling.hpp>

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/bit>

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <utilities/logger.hpp>
#include <utilities/scope_guard.hpp>
#include <utilities/timer.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

static constexpr int BVE_MAX_INTERIOR = BVE_MAX_SCOPE - 1;
// Cap closure probes over high-degree implication neighborhoods.
static constexpr int BVE_MAX_GROWTH_NBRS = 256;
// Cap peak device allocation for each projection chunk.
static constexpr size_t BVE_PROJECT_DEVICE_BUDGET = 64ull << 20;  // 64 MiB
// Cap the enumeration cost of one projection batch, summed as 2^(na+nb) * nnz over the candidates
// accepted into it.
static constexpr double BVE_BATCH_PROJECTION_BUDGET = 1e8;
static constexpr int BVE_MIN_COMMIT_RATIO           = 20;
// Outer rounds of the phase: each re-derives the implication graph from the model the previous one
// left behind.
static constexpr int BVE_MAX_ROUNDS = 3;
// Share of the model's columns a round must retire for another round's detect pass to be worth
// running.
static constexpr double BVE_MIN_ROUND_YIELD = 0.01;
// Seconds for the whole phase: implication graph build plus every round. Install and compact finish
// the round already committed, so a phase can exceed this by that tail.
static constexpr double BVE_STAGE_TIME_LIMIT = 1.5;

// Largest per-row rational multiplier / denominator we will apply. A row that would need a larger
// multiplier to become integer is treated as not exactly representable
static constexpr int64_t BVE_INT_SCALE_MAX = 1e6;

// Closed-form part of the commit_projected work estimate: prime-cube enumeration in
// bve_greedy_prime_cover is Θ(nb · 3^nb); sanity check is Θ(2^nb · #clauses) with #clauses bounded
// by the growth gate (n_rows + clause_growth_margin).
static double bve_commit_wall_ops(int nb, int clause_budget)
{
  cuopt_assert(nb >= 0 && nb <= BVE_MAX_BOUNDARY, "nb out of BVE range");
  double three_nb = 1.0;
  for (int i = 0; i < nb; ++i)
    three_nb *= 3.0;
  return nb * three_nb + (double)(1 << nb) * (clause_budget + 1);
}

bool bve_sanity_check(const uint8_t* feas, int nb, const bve_clause_t* clauses, int n_clauses)
{
  const uint32_t full_mask = (1u << nb) - 1u;
  for (int i = 0; i < n_clauses; ++i)
    if (clauses[i].lit_mask & ~full_mask) return false;  // literals must be on the boundary
  for (uint32_t m = 0; m <= full_mask; ++m) {
    bool crel = true;  // CNF value: AND over clauses of (clause satisfied by pattern m)
    for (int i = 0; i < n_clauses && crel; ++i) {
      const uint32_t lit = clauses[i].lit_mask;
      const uint32_t bit = clauses[i].bit_mask;
      // clause satisfied iff some literal position differs from its forbidden bit under m
      const bool satisfied = ((m ^ bit) & lit) != 0u;
      if (!satisfied) crel = false;
    }
    const bool feasible = feas[m] != 0;
    if (crel != feasible) return false;
  }
  return true;
}

// ===========================================================================================
//  Installed CNF: all prime forbidden cubes covered by max-gain greedy
// ===========================================================================================
//
// Two-level logic minimization in the shape of Quine, "The Problem of Simplifying Truth Functions"
// (Amer. Math. Monthly 1952) and McCluskey, "Minimization of Boolean Functions" (Bell System Tech.
// J. 1956): enumerate the prime implicants, then cover every minterm with a subset of them. Taking
// the primes of the infeasible patterns rather than the feasible ones makes each one a forbidden
// cube whose complement is a clause, so the cover comes out as a CNF instead of the usual DNF.
//
// The covering step is the greedy max-gain heuristic of Johnson, "Approximation Algorithms for
// Combinatorial Problems" (JCSS 1974), Lovász (Discrete Math. 1975) and Chvátal (Math. of OR 1979):
// repeatedly take the cube covering the most still-uncovered patterns, which lands within a factor
// 1 + ln m of the minimum cover for m infeasible patterns.

static size_t bve_mask_words(int nb) { return ((1u << nb) + 63u) / 64u; }

static int bve_mask_size(const bve_mask_t& m)
{
  int n = 0;
  for (uint64_t w : m)
    n += std::popcount(w);
  return n;
}

static void bve_mask_set(bve_mask_t& m, uint32_t pattern)
{
  cuopt_assert(size_t(pattern >> 6) < m.size(), "pattern outside mask width");
  m[pattern >> 6] |= uint64_t{1} << (pattern & 63);
}

static void bve_mask_subtract(bve_mask_t& m, const bve_mask_t& other)
{
  cuopt_assert(m.size() == other.size(), "mask width mismatch");
  for (size_t w = 0; w < m.size(); ++w)
    m[w] &= ~other[w];
}

static int bve_mask_overlap(const bve_mask_t& a, const bve_mask_t& b)
{
  cuopt_assert(a.size() == b.size(), "mask width mismatch");
  int n = 0;
  for (size_t w = 0; w < a.size(); ++w)
    n += std::popcount(a[w] & b[w]);
  return n;
}

// valid(lit, bit): every boundary pattern matching cube (lit, bit) is infeasible, so the
// complementary clause excludes no feasible pattern. Adding a literal SHRINKS the cube, so the
// table is filled from the minterms (lit == full_mask) downward in literal count:
//     valid(lit, bit) = valid(lit|j, bit) AND valid(lit|j, bit|j)   for any j not in lit
// A cube is already the bve_clause_t (lit_mask, bit_mask) encoding, so no separate ternary cube
// code is needed. The dense table is 4^nb bytes (16 MiB at nb = 12), so `valid` is caller-owned and
// grown once rather than reallocated per block. It is never re-initialized: only cells with
// bit subset of lit are ever addressed, the minterm seeding plus the recurrence below write every
// such cell, and each pass reads only cells an earlier pass already wrote.
static void bve_enumerate_prime_cubes(const uint8_t* feas,
                                      int nb,
                                      std::vector<uint8_t>& valid,
                                      std::vector<bve_clause_t>& primes)
{
  cuopt_assert(nb >= 1 && nb <= BVE_MAX_BOUNDARY, "nb out of BVE range");
  const uint32_t full_mask = (1u << nb) - 1u;
  const size_t stride      = size_t(full_mask) + 1;
  if (valid.size() < stride * stride) valid.resize(stride * stride);
  const auto at = [&](uint32_t lit, uint32_t bit) -> uint8_t& {
    cuopt_assert((bit & ~lit) == 0u, "cube bit_mask outside its lit_mask");
    return valid[size_t(lit) * stride + bit];
  };

  for (uint32_t m = 0; m <= full_mask; ++m)
    at(full_mask, m) = feas[m] ? 0 : 1;

  for (int n_lits = nb - 1; n_lits >= 0; --n_lits)
    for (uint32_t lit = 0; lit <= full_mask; ++lit) {
      if (std::popcount(lit) != n_lits) continue;
      const int j          = std::countr_zero(~lit & full_mask);
      const uint32_t child = lit | (1u << j);
      for (uint32_t bit = lit;; bit = (bit - 1u) & lit) {
        at(lit, bit) = at(child, bit) & at(child, bit | (1u << j));
        if (bit == 0u) break;
      }
    }

  primes.clear();
  for (uint32_t lit = 0; lit <= full_mask; ++lit)
    for (uint32_t bit = lit;; bit = (bit - 1u) & lit) {
      if (at(lit, bit)) {
        bool prime = true;
        for (int j = 0; j < nb && prime; ++j)
          if ((lit & (1u << j)) != 0u && at(lit ^ (1u << j), bit & ~(1u << j))) prime = false;
        if (prime) primes.push_back(bve_clause_t{lit, bit});
      }
      if (bit == 0u) break;
    }
}

// Boundary patterns matching the cube; every one of them is infeasible when the cube is valid.
static void bve_cube_cover(
  uint32_t lit, uint32_t bit, uint32_t full_mask, size_t n_words, bve_mask_t& cover)
{
  cover.assign(n_words, 0u);
  const uint32_t free_positions = full_mask & ~lit;
  for (uint32_t s = free_positions;; s = (s - 1u) & free_positions) {
    bve_mask_set(cover, bit | s);
    if (s == 0u) break;
  }
}

int bve_greedy_prime_cover(const uint8_t* feas,
                           int nb,
                           bve_clause_t* out,
                           int cap,
                           bve_cover_scratch_t& scratch,
                           int64_t* ops_out)
{
  cuopt_assert(nb >= 1 && nb <= BVE_MAX_BOUNDARY, "nb out of BVE range");
  cuopt_assert(cap >= 1, "clause cap leaves no room for a cover");
  int64_t ops    = 0;
  auto ops_guard = cuopt::scope_guard([&]() {
    if (ops_out != nullptr) *ops_out += ops;
  });

  const uint32_t full_mask  = (1u << nb) - 1u;
  const uint32_t n_patterns = 1u << nb;
  const size_t n_words      = bve_mask_words(nb);

  bve_enumerate_prime_cubes(feas, nb, scratch.valid, scratch.primes);
  const std::vector<bve_clause_t>& primes = scratch.primes;

  bve_mask_t& uncovered = scratch.uncovered;
  uncovered.assign(n_words, 0u);
  for (uint32_t m = 0; m < n_patterns; ++m)
    if (!feas[m]) bve_mask_set(uncovered, m);
  if (bve_mask_size(uncovered) == 0) return 0;  // nothing to forbid
  cuopt_assert(!primes.empty(), "infeasible patterns exist but no prime cube was enumerated");

  scratch.cover.resize(primes.size());
  for (size_t q = 0; q < primes.size(); ++q) {
    bve_cube_cover(primes[q].lit_mask, primes[q].bit_mask, full_mask, n_words, scratch.cover[q]);
    // Zeroing the words, then one set-bit per pattern the cube matches.
    ops += n_words + (int64_t{1} << (nb - std::popcount(primes[q].lit_mask)));
  }

  int n = 0;
  while (bve_mask_size(uncovered) > 0) {
    // Per pick: the size test above, one bve_mask_overlap per prime, then the subtract below.
    ops += (primes.size() + 2) * n_words;
    int best_q    = -1;
    int best_gain = 0;
    for (size_t q = 0; q < primes.size(); ++q) {
      const int gain = bve_mask_overlap(uncovered, scratch.cover[q]);
      if (gain > best_gain) {
        best_gain = gain;
        best_q    = q;
      }
    }
    cuopt_assert(best_q >= 0, "prime cubes do not cover the infeasible patterns");
    if (n >= cap) return -1;
    out[n++] = primes[best_q];
    bve_mask_subtract(uncovered, scratch.cover[best_q]);
  }
  ops += n_words;  // the size test that ended the loop
  cuopt_assert(n >= 1, "non-empty infeasible set covered by zero clauses");
  return n;
}

// Committed elimination in commit order. `witness[pattern]` packs interior values for the boundary
// pattern; reductions are replayed in reverse order during postsolve.
template <typename i_t>
struct bve_reduction_t {
  std::vector<i_t> interior;
  std::vector<i_t> boundary;
  std::vector<uint32_t> witness;  // size 2^boundary.size()
};

// A surviving clause row to append to problem_t (a set-covering no-good over boundary columns).
// Always in >= form.
template <typename i_t, typename f_t>
struct bve_added_row_t {
  std::vector<std::pair<i_t, f_t>> terms;
  f_t lower;
};

template <typename i_t, typename f_t>
struct bve_plan_t {
  std::vector<bve_reduction_t<i_t>> reductions;       // commit order
  std::vector<i_t> removed_rows;                      // original row ids to drop
  std::vector<bve_added_row_t<i_t, f_t>> added_rows;  // surviving clause rows
};

// Working model and accumulated reduction plan. Candidates are staged without mutation and
// committed only after projection and clause validation.
template <typename i_t, typename f_t>
struct bve_reducer_t {
  struct work_row_t {
    std::vector<std::pair<i_t, f_t>> terms;
    f_t lo, up;
    bool active;
  };

  i_t n_vars, n_rows_orig;
  f_t tol;
  i_t boundary_cap, scope_cap, clause_growth_margin;
  std::vector<work_row_t> rows;
  std::vector<std::unordered_set<i_t>> col2rows;
  std::vector<uint8_t> is_bin, obj_nz, done;
  bve_plan_t<i_t, f_t> plan;
  bve_cover_scratch_t cover_scratch;

  bve_reducer_t(i_t n_vars_,
                i_t n_rows_orig_,
                const std::vector<i_t>& offsets,
                const std::vector<i_t>& variables,
                const std::vector<f_t>& coefficients,
                const std::vector<f_t>& row_lower,
                const std::vector<f_t>& row_upper,
                const std::vector<f_t>& col_lower,
                const std::vector<f_t>& col_upper,
                const std::vector<uint8_t>& is_integer,
                const std::vector<f_t>& obj,
                f_t tol_,
                i_t boundary_cap_,
                i_t scope_cap_,
                i_t clause_growth_margin_);

  // Rows spanned by `interior` and the boundary columns of those rows, both unsorted, with op
  // accounting. Single traversal behind both the growth probe (which needs only the boundary size)
  // and stage(); outputs are overwritten, so a caller in a loop can reuse them.
  void scope_of(const std::vector<i_t>& interior,
                std::vector<i_t>& rows_out,
                std::vector<i_t>& boundary_out,
                int64_t& ops) const;

  // Gather and pack a candidate without projecting or mutating the working model.
  bool stage(const std::vector<i_t>& interior_in,
             bve_candidate_t<i_t, f_t>& out,
             int64_t* ops_out = nullptr);

  // Validate and commit an already-projected candidate; return true iff reduced.
  bool commit_projected(const bve_candidate_t<i_t, f_t>& cand, int64_t* ops_out = nullptr);

  bve_plan_t<i_t, f_t> finalize();
};

template <typename i_t, typename f_t>
bve_reducer_t<i_t, f_t>::bve_reducer_t(i_t n_vars_,
                                       i_t n_rows_orig_,
                                       const std::vector<i_t>& offsets,
                                       const std::vector<i_t>& variables,
                                       const std::vector<f_t>& coefficients,
                                       const std::vector<f_t>& row_lower,
                                       const std::vector<f_t>& row_upper,
                                       const std::vector<f_t>& col_lower,
                                       const std::vector<f_t>& col_upper,
                                       const std::vector<uint8_t>& is_integer,
                                       const std::vector<f_t>& obj,
                                       f_t tol_,
                                       i_t boundary_cap_,
                                       i_t scope_cap_,
                                       i_t clause_growth_margin_)
  : n_vars(n_vars_),
    n_rows_orig(n_rows_orig_),
    tol(tol_),
    boundary_cap(boundary_cap_),
    scope_cap(scope_cap_),
    clause_growth_margin(clause_growth_margin_),
    col2rows(n_vars_),
    is_bin(n_vars_),
    obj_nz(n_vars_),
    done(n_vars_, 0)
{
  const f_t INF = std::numeric_limits<f_t>::infinity();
  for (i_t c = 0; c < n_vars; ++c) {
    is_bin[c] =
      (is_integer[c] && std::abs(col_lower[c]) < tol && std::abs(col_upper[c] - f_t(1)) < tol) ? 1
                                                                                               : 0;
    obj_nz[c] = (obj[c] != f_t(0)) ? 1 : 0;
  }
  rows.reserve(n_rows_orig * 2);
  for (i_t r = 0; r < n_rows_orig; ++r) {
    work_row_t R;
    R.active = true;
    R.lo     = scaling_bound_finite(row_lower[r]) ? row_lower[r] : -INF;
    R.up     = scaling_bound_finite(row_upper[r]) ? row_upper[r] : INF;
    for (i_t k = offsets[r]; k < offsets[r + 1]; ++k)
      R.terms.emplace_back(variables[k], coefficients[k]);
    i_t id = rows.size();
    rows.push_back(std::move(R));
    for (auto& p : rows[id].terms)
      col2rows[p.first].insert(id);
  }
}

template <typename i_t, typename f_t>
void bve_reducer_t<i_t, f_t>::scope_of(const std::vector<i_t>& interior,
                                       std::vector<i_t>& rows_out,
                                       std::vector<i_t>& boundary_out,
                                       int64_t& ops) const
{
  ops += interior.size();
  std::unordered_set<i_t> interior_set(interior.begin(), interior.end());
  std::unordered_set<i_t> affected_rows;
  for (i_t a : interior)
    for (i_t r : col2rows[a]) {
      ++ops;
      affected_rows.insert(r);
    }
  std::unordered_set<i_t> b;
  for (i_t r : affected_rows)
    for (const auto& p : rows[r].terms) {
      ++ops;
      if (!interior_set.count(p.first)) b.insert(p.first);
    }
  rows_out.assign(affected_rows.begin(), affected_rows.end());
  boundary_out.assign(b.begin(), b.end());
}

// Rescale every row to integer coefficients and bounds so the projection can run at tolerance 0.
// Returns false if any row does not scale to bounded integers.
template <typename f_t>
static bool integerize_projection_rows(bve_block_t<f_t>& block)
{
  for (int rr = 0; rr < block.n_rows; ++rr) {
    const int rb   = block.row_off[rr];
    const int re   = block.row_off[rr + 1];
    const double s = row_int_scale<f_t>(block.row_coef + rb,
                                        re - rb,
                                        block.row_lo[rr],
                                        block.row_up[rr],
                                        BVE_MAX_ROW_LEN,
                                        BVE_INT_SCALE_MAX);
    if (s == 0.0) return false;
    for (int k = rb; k < re; ++k)
      block.row_coef[k] = std::llround((double)block.row_coef[k] * s);
    if (scaling_bound_finite(block.row_lo[rr]))
      block.row_lo[rr] = std::llround((double)block.row_lo[rr] * s);
    if (scaling_bound_finite(block.row_up[rr]))
      block.row_up[rr] = std::llround((double)block.row_up[rr] * s);
  }
  return true;
}

template <typename i_t, typename f_t>
bool bve_reducer_t<i_t, f_t>::stage(const std::vector<i_t>& interior_in,
                                    bve_candidate_t<i_t, f_t>& out,
                                    int64_t* ops_out)
{
  int64_t ops    = 0;
  auto ops_guard = cuopt::scope_guard([&]() {
    if (ops_out != nullptr) *ops_out += ops;
  });

  std::vector<i_t> interior(interior_in.begin(), interior_in.end());
  std::sort(interior.begin(), interior.end());
  std::vector<i_t> affected_rows, boundary;
  scope_of(interior, affected_rows, boundary, ops);
  // sorting improves GPU shape-binning
  std::sort(affected_rows.begin(), affected_rows.end());
  ops += affected_rows.size();
  std::sort(boundary.begin(), boundary.end());
  ops += boundary.size();

  const i_t nb = boundary.size();
  const i_t na = interior.size();
  if (nb == 0 || nb > boundary_cap || na + nb > scope_cap) return false;
  for (i_t v : boundary)
    if (!is_bin[v]) return false;
  if (na > BVE_MAX_INTERIOR || nb > BVE_MAX_BOUNDARY || na + nb > BVE_MAX_SCOPE) return false;
  if (affected_rows.size() > BVE_MAX_ROWS) return false;

  bve_block_t<f_t>& blk = out.blk;
  blk.na                = na;
  blk.nb                = nb;
  blk.n_rows            = affected_rows.size();
  std::unordered_map<i_t, i_t> local;
  for (i_t j = 0; j < na; ++j)
    local[interior[j]] = j;
  for (i_t j = 0; j < nb; ++j)
    local[boundary[j]] = na + j;
  ops += na + nb;
  i_t nzc           = 0;
  bool row_overflow = false;
  for (i_t rr = 0; rr < blk.n_rows && !row_overflow; ++rr) {
    const i_t r     = affected_rows[rr];
    blk.row_off[rr] = nzc;
    if (rows[r].terms.size() > BVE_MAX_ROW_LEN || nzc + rows[r].terms.size() > BVE_MAX_NNZ) {
      row_overflow = true;
      break;
    }
    for (auto& p : rows[r].terms) {
      blk.row_var[nzc]  = local[p.first];
      blk.row_coef[nzc] = p.second;
      ++nzc;
      ++ops;
    }
    blk.row_lo[rr] = rows[r].lo;
    blk.row_up[rr] = rows[r].up;
  }
  if (row_overflow) return false;
  blk.row_off[blk.n_rows] = nzc;

  if (!integerize_projection_rows(blk)) return false;

  out.interior = std::move(interior);
  out.boundary = std::move(boundary);
  out.rows     = std::move(affected_rows);
  out.projection.feasible.assign(size_t(1) << nb, 0);
  out.projection.witness.assign(size_t(1) << nb, 0u);
  ops += 1 << nb;
  return true;
}

template <typename i_t, typename f_t>
bool bve_reducer_t<i_t, f_t>::commit_projected(const bve_candidate_t<i_t, f_t>& cand,
                                               int64_t* ops_out)
{
  const int nb            = cand.blk.nb;
  const uint8_t* feasible = cand.projection.feasible.data();
  cuopt_assert(cand.projection.feasible.size() == (size_t(1) << nb), "projection table unsized");
  bve_clause_t clauses[BVE_MAX_CLAUSES];
  const int n_clauses =
    bve_greedy_prime_cover(feasible, nb, clauses, BVE_MAX_CLAUSES, cover_scratch, ops_out);
  if (n_clauses < 0) return false;  // clause explosion past cap
  if (n_clauses > cand.blk.n_rows + clause_growth_margin) return false;  // growth gate
  if (!bve_sanity_check(feasible, nb, clauses, n_clauses))
    return false;  // sanity check failed => keep block

  bve_reduction_t<i_t> red;
  red.interior = cand.interior;
  red.boundary = cand.boundary;
  red.witness  = cand.projection.witness;
  plan.reductions.push_back(std::move(red));

  for (i_t r : cand.rows) {
    for (auto& p : rows[r].terms)
      col2rows[p.first].erase(r);
    rows[r].active = false;
    rows[r].terms.clear();
  }
  const f_t INF = std::numeric_limits<f_t>::infinity();
  for (i_t ci = 0; ci < n_clauses; ++ci) {
    const uint32_t lit = clauses[ci].lit_mask;
    const uint32_t bit = clauses[ci].bit_mask;
    cuopt_assert(lit != 0u, "empty clause reached the row builder");
    work_row_t R;
    R.active = true;
    R.up     = INF;
    i_t n1   = 0;
    for (i_t j = 0; j < nb; ++j)
      if (lit & (1u << j)) {
        const i_t b = (bit >> j) & 1u;
        R.terms.emplace_back(cand.boundary[j], b ? f_t(-1) : f_t(1));
        n1 += b;
      }
    R.lo   = f_t(1 - n1);
    i_t id = rows.size();
    rows.push_back(std::move(R));
    for (auto& p : rows[id].terms)
      col2rows[p.first].insert(id);
  }
  for (i_t a : cand.interior) {
    col2rows[a].clear();
    done[a] = 1;
  }
  return true;
}

template <typename i_t, typename f_t>
bve_plan_t<i_t, f_t> bve_reducer_t<i_t, f_t>::finalize()
{
  for (i_t r = 0; r < n_rows_orig; ++r)
    if (!rows[r].active) plan.removed_rows.push_back(r);
  for (size_t r = n_rows_orig; r < rows.size(); ++r)
    if (rows[r].active) {
      cuopt_assert(rows[r].up == std::numeric_limits<f_t>::infinity(),
                   "clause rows carry no upper bound");
      bve_added_row_t<i_t, f_t> ar;
      ar.terms = std::move(rows[r].terms);
      ar.lower = rows[r].lo;
      plan.added_rows.push_back(std::move(ar));
    }
  return plan;
}

// ===========================================================================================
//  GPU enumeration projection kernel
// ===========================================================================================

//
//   grid : one CTA per assignment (block, boundary pattern m, interior pattern am),
//          grid-strided over CTAs   ( for assignment = blockIdx.x; ...; += gridDim.x )
//   CTA  : one warp per row          ( blockDim.x == min(nrows,32)*32; warps loop if nrows > 32 )
//   warp : reduces  sum = Σ coeff * value  over the row's entries, tests sum in [lower, upper]
//
// The CTA ANDs the per-row satisfied bits into a single "assignment feasible" bit. For each
// boundary pattern m, feasibility is the OR over its interior patterns am and the witness is the
// first feasible am; both are encoded by a single atomicMin into `out_witness` (sentinel 0xFFFFFFFF
// = no feasible interior), so downstream:
//     feasible[block][m] == (out_witness[block][m] != 0xFFFFFFFF)
//     witness [block][m] ==  out_witness[block][m]        // the smallest feasible interior
// `out_witness` must be initialized to 0xFFFFFFFF by the caller before launch.

template <typename i_t, typename f_t>
__global__ void bve_enumerate_kernel(
  i_t num_blocks,
  i_t nb,
  i_t na,
  i_t nrows,
  f_t tolerance,
  const f_t* block_coeffs,        // [num_blocks * nnz]
  const i_t* local_var_of_entry,  // [nnz]        (shared by the bin)
  const i_t* row_start,           // [nrows + 1]  (shared by the bin)
  const f_t* block_row_lower,     // [num_blocks * nrows]
  const f_t* block_row_upper,     // [num_blocks * nrows]
  uint32_t* out_witness)          // [num_blocks * (1<<nb)]
{
  extern __shared__ uint8_t row_satisfied[];  // [nrows]

  const i_t nnz          = row_start[nrows];
  const i_t num_patterns = i_t(1) << nb;
  // Layout of assignment: [block | boundary_pattern | interior_pattern]
  //                         high    mid (nb bits)       low (na bits)
  const int64_t num_assignments = (int64_t)num_blocks << (na + nb);

  const int lane_id   = threadIdx.x % 32;
  const int warp_id   = threadIdx.x / 32;
  const int num_warps = blockDim.x / 32;

  // one CTA per assignment (block, m, am), grid-strided over CTAs
  for (int64_t assignment = blockIdx.x; assignment < num_assignments; assignment += gridDim.x) {
    const auto a               = (uint64_t)assignment;
    const i_t interior_pattern = cuda::bitfield_extract(a, 0, na);
    const i_t boundary_pattern = cuda::bitfield_extract(a, na, nb);
    const i_t block            = a >> (na + nb);

    const f_t* coeffs = block_coeffs + block * nnz;
    const f_t* lower  = block_row_lower + block * nrows;
    const f_t* upper  = block_row_upper + block * nrows;

    for (i_t row = warp_id; row < nrows; row += num_warps) {
      f_t partial = 0;
      for (i_t entry = row_start[row] + lane_id; entry < row_start[row + 1]; entry += 32) {
        const i_t var = local_var_of_entry[entry];
        const f_t value =
          (var < na) ? ((interior_pattern >> var) & 1) : ((boundary_pattern >> (var - na)) & 1);
        partial += coeffs[entry] * value;
      }
      const f_t sum = raft::warpReduce(partial);
      if (lane_id == 0) {
        row_satisfied[row] =
          (sum <= upper[row] + tolerance && sum >= lower[row] - tolerance) ? 1 : 0;
      }
    }
    __syncthreads();

    // AND the per-row bits; if this assignment is feasible, offer its interior as a witness
    if (threadIdx.x == 0) {
      uint8_t feasible = 1;
      for (i_t row = 0; row < nrows; ++row) {
        feasible &= row_satisfied[row];
      }
      if (feasible) {
        atomicMin(&out_witness[block * num_patterns + boundary_pattern],
                  (uint32_t)interior_pattern);
      }
    }
    __syncthreads();
  }
}

// ---- GPU batch projection: one enumeration-kernel launch per shape-bin ----
// Returns raw work for the enumerations (sum over bins of assignments · nnz).
template <typename i_t, typename f_t>
double bve_project_batch_gpu(const raft::handle_t& handle,
                             std::vector<bve_candidate_t<i_t, f_t>>& cands,
                             f_t tol,
                             const timer_t& timer)
{
  if (cands.empty()) return 0.0;
  auto stream       = handle.get_stream();
  double work_units = 0.0;

  // Bin candidates by identical shape so every CTA in a launch runs the same loop structure.
  struct shape_key_hash {
    size_t operator()(const std::vector<i_t>& key) const
    {
      size_t h = 0;
      for (i_t x : key) {
        h ^= std::hash<i_t>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
      }
      return h;
    }
  };
  std::unordered_map<std::vector<i_t>, std::vector<size_t>, shape_key_hash> bins;
  for (size_t i = 0; i < cands.size(); ++i) {
    const auto& blk = cands[i].blk;
    const i_t nnz   = blk.row_off[blk.n_rows];
    std::vector<i_t> key;
    key.reserve(4 + (blk.n_rows + 1) + nnz);
    key.push_back(blk.na);
    key.push_back(blk.nb);
    key.push_back(blk.n_rows);
    key.push_back(nnz);
    for (i_t r = 0; r <= blk.n_rows; ++r)
      key.push_back(blk.row_off[r]);
    for (i_t k = 0; k < nnz; ++k)
      key.push_back(blk.row_var[k]);
    bins[std::move(key)].push_back(i);
  }

  for (const auto& kv : bins) {
    if (timer.check_time_limit()) return work_units;
    const std::vector<size_t>& idxs = kv.second;
    const auto& proto               = cands[idxs[0]].blk;
    const i_t na                    = proto.na;
    const i_t nb                    = proto.nb;
    const i_t nrows                 = proto.n_rows;
    const i_t nnz                   = proto.row_off[nrows];
    const i_t patterns              = i_t(1) << nb;

    // Shared layout is O(nnz) and identical for every candidate in the bin.
    std::vector<i_t> h_row_start(proto.row_off, proto.row_off + nrows + 1);
    std::vector<i_t> h_local_var(proto.row_var, proto.row_var + nnz);
    rmm::device_uvector<i_t> d_row_start(h_row_start.size(), stream);
    rmm::device_uvector<i_t> d_local_var(h_local_var.size(), stream);
    raft::copy(d_row_start.data(), h_row_start.data(), h_row_start.size(), stream);
    raft::copy(d_local_var.data(), h_local_var.data(), h_local_var.size(), stream);

    // Per-block device cost: coeffs + row bounds + witness table.
    const size_t bytes_per_block = size_t(nnz) * sizeof(f_t) + 2 * size_t(nrows) * sizeof(f_t) +
                                   size_t(patterns) * sizeof(uint32_t);

    const size_t chunk =
      std::max<size_t>(1,
                       std::min(size_t(std::numeric_limits<i_t>::max()),
                                BVE_PROJECT_DEVICE_BUDGET / std::max<size_t>(1, bytes_per_block)));

    const int num_warps = std::min<i_t>(nrows, 32);
    const int cta_dim   = num_warps * 32;
    const size_t shmem  = size_t(nrows) * sizeof(uint8_t);

    for (size_t offset = 0; offset < idxs.size(); offset += chunk) {
      if (timer.check_time_limit()) return work_units;
      const size_t num_sz = std::min(chunk, idxs.size() - offset);
      const i_t num       = num_sz;

      std::vector<f_t> h_coeffs(num_sz * size_t(nnz));
      std::vector<f_t> h_lower(num_sz * size_t(nrows));
      std::vector<f_t> h_upper(num_sz * size_t(nrows));
      for (size_t g = 0; g < num_sz; ++g) {
        const auto& blk = cands[idxs[offset + g]].blk;
        std::copy(blk.row_coef, blk.row_coef + nnz, h_coeffs.begin() + g * nnz);
        std::copy(blk.row_lo, blk.row_lo + nrows, h_lower.begin() + g * nrows);
        std::copy(blk.row_up, blk.row_up + nrows, h_upper.begin() + g * nrows);
      }

      rmm::device_uvector<f_t> d_coeffs(h_coeffs.size(), stream);
      rmm::device_uvector<f_t> d_lower(h_lower.size(), stream);
      rmm::device_uvector<f_t> d_upper(h_upper.size(), stream);
      rmm::device_uvector<uint32_t> d_witness(num_sz * size_t(patterns), stream);
      raft::copy(d_coeffs.data(), h_coeffs.data(), h_coeffs.size(), stream);
      raft::copy(d_lower.data(), h_lower.data(), h_lower.size(), stream);
      raft::copy(d_upper.data(), h_upper.data(), h_upper.size(), stream);
      // sentinel 0xFFFFFFFF (every byte 0xFF) marks a boundary pattern with no feasible interior
      // yet
      RAFT_CUDA_TRY(
        cudaMemsetAsync(d_witness.data(), 0xFF, d_witness.size() * sizeof(uint32_t), stream.get()));

      // one warp per row, one CTA per (block, m, am) assignment, grid-strided
      const int64_t total = (int64_t)num * (int64_t)patterns * ((int64_t)1 << na);
      const int grid      = std::min(total, int64_t{65535});
      bve_enumerate_kernel<i_t, f_t><<<grid, cta_dim, shmem, stream.get()>>>(num,
                                                                             nb,
                                                                             na,
                                                                             nrows,
                                                                             tol,
                                                                             d_coeffs.data(),
                                                                             d_local_var.data(),
                                                                             d_row_start.data(),
                                                                             d_lower.data(),
                                                                             d_upper.data(),
                                                                             d_witness.data());
      RAFT_CUDA_TRY(cudaGetLastError());

      // Unscaled op counts: host pack/unpack touches + one coeff read per assignment.
      work_units += (double)num_sz * (nnz + 2 * nrows + patterns);
      work_units += (double)total * nnz;

      std::vector<uint32_t> h_witness(num_sz * size_t(patterns));
      raft::copy(h_witness.data(), d_witness.data(), h_witness.size(), stream);
      handle.sync_stream();
      for (size_t g = 0; g < num_sz; ++g) {
        auto& cand = cands[idxs[offset + g]];
        // No-op for anything stage() produced; sizes a caller that assembled `blk` by hand.
        cand.projection.feasible.resize(patterns);
        cand.projection.witness.resize(patterns);
        for (i_t m = 0; m < patterns; ++m) {
          const uint32_t w            = h_witness[g * patterns + m];
          const bool feasible         = (w != 0xFFFFFFFFu);
          cand.projection.feasible[m] = feasible ? 1 : 0;
          cand.projection.witness[m]  = feasible ? w : 0u;
        }
        cand.projection.projected = true;
      }
    }
  }
  return work_units;
}

// ---- harvest unary-conditioned implications from an exactly projected block ----
//
// `feas` is the block's exact existential projection onto its nb boundary columns, so for boundary
// position j and value a the feasible patterns agreeing with (j == a) describe every completion the
// block admits. Intersecting them (AND) gives the positions forced to 1 and the complement of their
// union (OR) gives those forced to 0; the same reasoning with no condition gives unconditional
// fixings. This is complete for the block's rows, where the probing cache only holds what bound
// propagation could prove, so these forcings can be strictly stronger. It holds whether or not the
// block is eventually eliminated, hence the call site harvests before the growth gate can reject.
//
// Ids are emitted in the current-problem frame; the caller maps them to original ids.
template <typename i_t, typename f_t>
static void bve_extract_forcings(const bve_candidate_t<i_t, f_t>& cand, probe_findings_t<i_t>& out)
{
  const i_t nb = cand.blk.nb;
  cuopt_assert(nb > 0 && nb <= BVE_MAX_BOUNDARY, "boundary width out of range");
  cuopt_assert((i_t)cand.boundary.size() == nb, "boundary id count disagrees with block width");
  const uint32_t n_patterns = 1u << nb;

  // Accumulators for condition s = 2*j + a; slot 2*nb holds the unconditional case.
  constexpr i_t n_slots   = 2 * BVE_MAX_BOUNDARY + 1;
  const i_t unconditional = 2 * nb;
  const uint32_t all_ones = n_patterns - 1u;
  uint32_t and_acc[n_slots];
  uint32_t or_acc[n_slots];
  std::fill_n(and_acc, unconditional + 1, all_ones);
  std::fill_n(or_acc, unconditional + 1, 0u);

  uint32_t n_feasible = 0;
  for (uint32_t m = 0; m < n_patterns; ++m) {
    if (!cand.projection.feasible[m]) continue;
    ++n_feasible;
    and_acc[unconditional] &= m;
    or_acc[unconditional] |= m;
    for (i_t j = 0; j < nb; ++j) {
      const i_t s = 2 * j + ((m >> j) & 1u);
      and_acc[s] &= m;
      or_acc[s] |= m;
    }
  }
  // Vacuous accumulators would otherwise read as "every position forced to 1".
  if (n_feasible == 0u) return;  // the caller turns this block into an infeasibility proof

  // Positions the block fixes outright.
  const uint32_t fixed_mask = and_acc[unconditional] | (~or_acc[unconditional] & all_ones);
  for (i_t j = 0; j < nb; ++j) {
    if (!(fixed_mask & (1u << j))) continue;
    out.fixings.emplace_back(cand.boundary[j], ((and_acc[unconditional] >> j) & 1u) != 0u);
  }

  for (i_t j = 0; j < nb; ++j) {
    if (fixed_mask & (1u << j)) continue;  // condition never binds
    for (i_t a = 0; a < 2; ++a) {
      const i_t s = 2 * j + a;
      for (i_t k = 0; k < nb; ++k) {
        const uint32_t bit = 1u << k;
        if (k == j || (fixed_mask & bit)) continue;
        if (and_acc[s] & bit) {
          out.forcings.push_back({cand.boundary[j], cand.boundary[k], a != 0, true});
        } else if (!(or_acc[s] & bit)) {
          out.forcings.push_back({cand.boundary[j], cand.boundary[k], a != 0, false});
        }
      }
    }
  }
}

template <typename i_t>
struct bve_growth_result_t {
  std::vector<i_t> interior;  // sorted current-problem column ids, always contains the seed
  int64_t ops = 0;            // work performed, for the deterministic wall estimate
};

// Grows one seed into a block interior: starting from {seed}, repeatedly absorb the eligible
// implication-neighbor that shrinks the boundary the most, stopping when no neighbor strictly
// improves it or a cap is hit. Read-only on `reducer`, which is what lets the round run this across
// seeds under OpenMP against a frozen model.
template <typename i_t, typename f_t>
static bve_growth_result_t<i_t> grow_seed_interior(
  i_t seed,
  const bve_reducer_t<i_t, f_t>& reducer,
  const std::vector<std::vector<i_t>>& implication_adjacency,
  const timer_t& timer)
{
  auto has_adj = [&](i_t v) {
    return v >= 0 && v < (i_t)implication_adjacency.size() && !implication_adjacency[v].empty();
  };

  bve_growth_result_t<i_t> result;
  std::unordered_set<i_t> interior_set = {seed};
  std::vector<i_t> probe_rows, probe_bnd;  // scope_of scratch, reused across probes
  bool timed_out = false;
  for (;;) {
    if (timer.check_time_limit()) break;
    // Hub fast-path: raw implication degree upper-bounds |cands_w|.
    if (interior_set.size() == 1) {
      const i_t s   = *interior_set.begin();
      const i_t deg = has_adj(s) ? (i_t)implication_adjacency[s].size() : 0;
      if (deg > BVE_MAX_GROWTH_NBRS) break;
    }
    std::vector<i_t> candidate_interior(interior_set.begin(), interior_set.end());
    reducer.scope_of(candidate_interior, probe_rows, probe_bnd, result.ops);
    const i_t cur = probe_bnd.size();
    // Implication-neighbors of the interior that are still eligible to enter it.
    std::unordered_set<i_t> cands_w;
    bool gated = false;
    for (i_t a : interior_set) {
      if (!has_adj(a)) continue;
      for (i_t w : implication_adjacency[a]) {
        ++result.ops;
        const bool eligible = reducer.is_bin[w] && !reducer.obj_nz[w] && !reducer.done[w] &&
                              !reducer.col2rows[w].empty();
        if (interior_set.count(w) || !eligible) continue;
        cands_w.insert(w);
        if ((i_t)cands_w.size() > BVE_MAX_GROWTH_NBRS) {
          gated = true;
          break;
        }
      }
      if (gated) break;
    }
    // Hub neighborhoods: full probe is Θ(|cands_w|) boundary walks and rarely absorbs.
    if (gated) break;
    // Pick the neighbor with the smallest boundary
    i_t best    = -1;
    i_t best_nb = cur;
    i_t probes  = 0;
    for (i_t w : cands_w) {
      // Each probe is a boundary walk, so the deadline is honoured within a single seed's growth.
      if ((++probes & 0xF) == 0 && timer.check_time_limit()) {
        timed_out = true;
        break;
      }
      candidate_interior.push_back(w);  // probe interior ∪ {w}; the pop below restores it
      const i_t na = candidate_interior.size();
      reducer.scope_of(candidate_interior, probe_rows, probe_bnd, result.ops);
      const i_t nb = probe_bnd.size();
      candidate_interior.pop_back();
      if (nb < best_nb && na + nb <= reducer.scope_cap && na <= BVE_MAX_INTERIOR) {
        best_nb = nb;
        best    = w;
      }
    }
    if (timed_out || best < 0) break;
    interior_set.insert(best);
  }
  result.interior.assign(interior_set.begin(), interior_set.end());
  return result;
}

// Implication-closure block growth over the probing-cache adjacency: each seed absorbs the
// implication-neighbor that most shrinks its boundary (subject to enum/interior caps) until no
// such neighbor remains. Within a round the working model is frozen, so every seed grows its
// interior against the same model. Because that growth is read-only on the model, it runs in an
// OpenMP parallel-for across the round's seeds; the results are deterministic per seed and
// acceptance is then applied serially in seed order, so the committed plan is identical to a serial
// run of the same frozen growth. Candidates are staged and only mutually scope-disjoint ones (no
// shared interior or boundary column, which also forbids a shared row) are accepted into the batch.
// The batch is projected on the device (bve_project_batch_gpu), then committed on the host; because
// the accepted candidates touch disjoint columns/rows, commit order is irrelevant and each block's
// staged projection is still valid at commit time. Candidates deferred for overlap are retried in
// later rounds; the loop stops when a round accepts nothing or commits nothing (each committing
// round retires >= 1 column, hence terminates).
//
// A block whose projection admits no boundary assignment sets `out_infeasible` and abandons the
// round with an empty plan, so nothing this call staged is ever installed.
template <typename i_t, typename f_t>
static bve_plan_t<i_t, f_t> bve_detect_closure_batched(
  const raft::handle_t& handle,
  bve_reducer_t<i_t, f_t>& reducer,
  const std::vector<std::vector<i_t>>& impl_adj,
  timer_t& timer,
  double& work_units,
  probe_findings_t<i_t>* findings,
  bool* out_infeasible)
{
  std::vector<i_t> order;
  for (i_t c = 0; c < reducer.n_vars; ++c) {
    // grow_seed_interior's hub fast path refuses to grow a seed whose implication degree is past
    // the probe cap, so such a seed only ever reaches stage() as a singleton interior.
    const i_t degree    = c < (i_t)impl_adj.size() ? (i_t)impl_adj[c].size() : 0;
    const bool growable = degree > 0 && degree <= BVE_MAX_GROWTH_NBRS;
    if (reducer.is_bin[c] && !reducer.obj_nz[c] && !reducer.col2rows[c].empty() && growable)
      order.push_back(c);
  }
  std::sort(order.begin(), order.end(), [&](i_t a, i_t b) {
    return reducer.col2rows[a].size() < reducer.col2rows[b].size();
  });

  std::vector<char> attempted(reducer.n_vars, 0);
  std::vector<char> growth_done(reducer.n_vars, 0);
  std::vector<std::vector<i_t>> growth_interior(reducer.n_vars);
  for (;;) {
    if (timer.check_time_limit()) break;

    // This round's live seeds, in the deterministic growth order.
    std::vector<i_t> round_seeds;
    for (i_t seed : order)
      if (!attempted[seed] && !reducer.done[seed] && !reducer.col2rows[seed].empty())
        round_seeds.push_back(seed);
    if (round_seeds.empty()) break;

    // Grow each seed against the frozen model (read-only on reducer → OMP-safe). Acceptance below
    // is serial in round_seeds order, so the plan matches a serial frozen-growth run.
    std::vector<std::vector<i_t>> interiors(round_seeds.size());
    std::vector<int64_t> growth_ops(round_seeds.size(), 0);
#pragma omp taskloop default(shared) priority(CUOPT_DEFAULT_TASK_PRIORITY)
    for (i_t k = 0; k < (i_t)round_seeds.size(); ++k) {
      const i_t seed = round_seeds[k];
      if (growth_done[seed]) {
        interiors[k] = growth_interior[seed];
        continue;
      }
      bve_growth_result_t<i_t> grown = grow_seed_interior(seed, reducer, impl_adj, timer);
      growth_ops[k]                  = grown.ops;
      interiors[k]                   = std::move(grown.interior);
      growth_interior[seed]          = interiors[k];
      growth_done[seed]              = 1;
    }
    // OMP growth: wall ≈ critical-path seed (max), not sum across threads.
    int64_t max_growth_ops = 0;
    for (int64_t ops : growth_ops)
      max_growth_ops = std::max(max_growth_ops, ops);
    work_units += max_growth_ops;

    if (timer.check_time_limit()) break;

    // Serial: stage each grown interior and greedily accept mutually SCOPE-DISJOINT candidates, in
    // round_seeds order. Nothing mutates the model until commit, so this stays serial.
    std::vector<bve_candidate_t<i_t, f_t>> cands;
    std::unordered_set<i_t> claimed;  // interior+boundary columns of already-accepted candidates
    double batch_projection_ops = 0.0;
    for (size_t k = 0; k < round_seeds.size(); ++k) {
      if (timer.check_time_limit()) break;
      const i_t seed = round_seeds[k];
      bve_candidate_t<i_t, f_t> cand;
      int64_t stage_ops = 0;
      if (!reducer.stage(interiors[k], cand, &stage_ops)) {
        work_units += stage_ops;
        attempted[seed] = 1;  // failed the caps against this model; one attempt per seed
        continue;
      }
      work_units += stage_ops;
      bool overlap = false;
      for (i_t c : cand.interior)
        if (claimed.count(c)) {
          overlap = true;
          break;
        }
      if (!overlap)
        for (i_t c : cand.boundary)
          if (claimed.count(c)) {
            overlap = true;
            break;
          }
      if (overlap) continue;  // scope collides; retry stage later from cached interior

      attempted[seed] = 1;
      for (i_t c : cand.interior)
        claimed.insert(c);
      for (i_t c : cand.boundary)
        claimed.insert(c);
      cuopt_assert(cand.blk.na + cand.blk.nb <= BVE_MAX_SCOPE, "staged scope past enumeration cap");
      batch_projection_ops +=
        (double)(1u << (cand.blk.na + cand.blk.nb)) * cand.blk.row_off[cand.blk.n_rows];
      cands.push_back(std::move(cand));

      if (batch_projection_ops >= BVE_BATCH_PROJECTION_BUDGET) break;
    }

    if (cands.empty() || timer.check_time_limit()) break;
    // Staged blocks are integerized (integerize_projection_rows), so the subset-sum feasibility
    // test is exact at tolerance 0.
    work_units += bve_project_batch_gpu<i_t, f_t>(handle, cands, f_t(0), timer);
    if (timer.check_time_limit()) break;
    i_t committed = 0;
    for (auto& cand : cands) {
      if (timer.check_time_limit()) break;
      cuopt_assert(cand.projection.projected, "commit loop reached an unprojected candidate");
      const bool admits_nothing =
        cand.projection.projected && std::none_of(cand.projection.feasible.begin(),
                                                  cand.projection.feasible.end(),
                                                  [](uint8_t feasible) { return feasible != 0; });
      if (admits_nothing) {
        if (out_infeasible != nullptr) *out_infeasible = true;
        return {};
      }
      // Valid for the block's rows regardless of the clause gates below, so harvest before them.
      if (findings != nullptr) {
        bve_extract_forcings<i_t, f_t>(cand, *findings);
        work_units += (double)(1u << cand.blk.nb) * cand.blk.nb;
      }
      work_units +=
        bve_commit_wall_ops(cand.blk.nb, cand.blk.n_rows + reducer.clause_growth_margin);
      int64_t commit_ops = 0;
      if (reducer.commit_projected(cand, &commit_ops)) ++committed;
      work_units += commit_ops;
    }
    if (committed == 0) break;
    cuopt_assert(committed <= (i_t)cands.size(), "committed more candidates than were projected");
    if ((i_t)cands.size() > committed * BVE_MIN_COMMIT_RATIO) break;
  }
  return reducer.finalize();
}

// ---- implication adjacency from the probing cache (original-id -> current column) ----
template <typename i_t, typename f_t>
std::vector<std::vector<i_t>> bve_build_impl_adj(
  const probing_cache_t<i_t, f_t>& cache,
  const std::vector<i_t>& reverse_original_ids,
  i_t n_vars,
  const timer_t& timer,
  const probe_findings_t<i_t>* prior_original_id_findings)
{
  // original-id -> current column index (or -1 if the column no longer exists)
  auto to_current = [&](i_t original_id) -> i_t {
    if (original_id < 0 || original_id >= (i_t)reverse_original_ids.size()) return -1;
    return reverse_original_ids[original_id];
  };
  std::vector<std::unordered_set<i_t>> adj(n_vars);
  auto add_edge = [&](i_t original_x, i_t original_y) {
    const i_t x = to_current(original_x);
    if (x < 0 || x >= n_vars) return;
    const i_t y = to_current(original_y);
    if (y < 0 || y >= n_vars || y == x) return;
    adj[x].insert(y);
    adj[y].insert(x);
  };
  // An abandoned build returns no edges rather than a partial graph, so which reductions exist
  // never depends on where the clock landed.
  i_t entries_seen = 0;
  for (const auto& kv : cache.probing_cache) {
    if ((++entries_seen & 0x3F) == 0 && timer.check_time_limit()) {
      return std::vector<std::vector<i_t>>(n_vars);
    }
    for (int p = 0; p < 2; ++p) {
      for (const auto& yb : kv.second[p].var_to_cached_bound_map)
        add_edge(kv.first, yb.first);
    }
  }
  // Forcings mined from earlier projections. Pairs the cache never held become seed/absorb
  // candidates, so a later round can grow blocks the first round could not see.
  if (prior_original_id_findings != nullptr) {
    i_t forcings_seen = 0;
    for (const auto& forcing : prior_original_id_findings->forcings) {
      if ((++forcings_seen & 0x3FF) == 0 && timer.check_time_limit()) {
        return std::vector<std::vector<i_t>>(n_vars);
      }
      add_edge(forcing.var, forcing.forced_var);
    }
  }
  std::vector<std::vector<i_t>> out(n_vars);
  for (i_t v = 0; v < n_vars; ++v)
    out[v].assign(adj[v].begin(), adj[v].end());
  return out;
}

template <typename i_t, typename f_t>
static void append_bve_reconstructions(const bve_plan_t<i_t, f_t>& plan,
                                       const std::vector<i_t>& current_to_post_papilo,
                                       presolve_data_t<i_t, f_t>& presolve_data,
                                       double& work_units)
{
  auto to_post_papilo = [&](i_t column) {
    cuopt_assert(column >= 0 && column < (i_t)current_to_post_papilo.size(),
                 "block column out of variable_mapping range");
    return current_to_post_papilo[column];
  };

  auto& reconstructions = presolve_data.postsolve_reconstructions;
  reconstructions.reserve(reconstructions.size() + plan.reductions.size());
  for (const auto& red : plan.reductions) {
    work_units += red.interior.size() + red.boundary.size() + red.witness.size();
    postsolve_reconstruction_t<i_t, f_t> reconstruction;
    reconstruction.kind = reconstruction_kind_t::BlockBve;
    reconstruction.bve.interior.reserve(red.interior.size());
    for (i_t c : red.interior)
      reconstruction.bve.interior.push_back(to_post_papilo(c));
    reconstruction.bve.boundary.reserve(red.boundary.size());
    for (i_t c : red.boundary)
      reconstruction.bve.boundary.push_back(to_post_papilo(c));
    reconstruction.bve.witness = red.witness;
    reconstructions.push_back(std::move(reconstruction));
  }
}

template <typename i_t, typename f_t>
bool bve_has_stageable_row(const problem_t<i_t, f_t>& problem)
{
  const i_t n_rows = problem.n_constraints;
  if (problem.empty || n_rows == 0) { return false; }
  const i_t* offsets    = problem.offsets.data();
  constexpr i_t max_len = BVE_MAX_ROW_LEN;
  return thrust::any_of(problem.handle_ptr->get_thrust_policy(),
                        thrust::make_counting_iterator<i_t>(0),
                        thrust::make_counting_iterator<i_t>(n_rows),
                        [offsets, max_len] __device__(i_t r) -> bool {
                          return offsets[r + 1] - offsets[r] <= max_len;
                        });
}

// Pin variables that a BVE projection table showed to have a single admissible value. Ids arrive in
// the original frame and may repeat across blocks and rounds. Returns false when two blocks
// disagree on a variable, which proves infeasibility since each fixing is a consequence of its
// block alone.
template <typename i_t, typename f_t>
static bool apply_bve_fixings(problem_t<i_t, f_t>& problem,
                              const std::vector<std::pair<i_t, bool>>& fixings,
                              i_t& n_applied)
{
  n_applied = 0;
  if (fixings.empty()) { return true; }
  std::vector<std::pair<i_t, bool>> sorted(fixings);
  std::sort(sorted.begin(), sorted.end());

  const std::vector<i_t>& reverse_original_ids = problem.reverse_original_ids;
  std::vector<i_t> var_indices;
  std::vector<f_t> lb_values;
  std::vector<f_t> ub_values;
  for (size_t k = 0; k < sorted.size(); ++k) {
    const auto [original_id, value] = sorted[k];
    if (k > 0 && original_id == sorted[k - 1].first) {
      if (value != sorted[k - 1].second) { return false; }
      continue;
    }
    cuopt_assert(original_id >= 0 && original_id < (i_t)reverse_original_ids.size(),
                 "fixings are keyed by original id");
    const i_t column = reverse_original_ids[original_id];
    if (column < 0 || column >= problem.n_variables) { continue; }  // already eliminated
    var_indices.push_back(column);
    lb_values.push_back(value ? f_t(1) : f_t(0));
    ub_values.push_back(value ? f_t(1) : f_t(0));
  }
  n_applied = var_indices.size();
  problem.update_variable_bounds(var_indices, lb_values, ub_values);
  return true;
}

// ---- the pass: detect (GPU-projected) -> install reduced model -> record reconstructions ----
template <typename i_t, typename f_t>
bool block_bve_presolve(problem_t<i_t, f_t>& problem,
                        const std::vector<std::vector<i_t>>& impl_adj,
                        timer_t& timer,
                        double& work_units,
                        probe_findings_t<i_t>* out_findings,
                        bool* out_infeasible,
                        i_t boundary_cap,
                        i_t scope_cap,
                        i_t clause_growth_margin)
{
  work_units = 0.0;
  timer_t wall(std::numeric_limits<double>::infinity());
  [[maybe_unused]] double t_setup = 0.0, t_detect = 0.0, t_install = 0.0, t_compact = 0.0;
  auto timer_raii_guard = cuopt::scope_guard([&]() {
    CUOPT_LOG_DEBUG(
      "Block-BVE phases: setup=%.2fs detect=%.2fs install=%.2fs compact=%.2fs total=%.2fs "
      "work units: %.6g",
      t_setup,
      t_detect,
      t_install,
      t_compact,
      wall.elapsed_time(),
      work_units);
  });

  const raft::handle_t* handle = problem.handle_ptr;
  auto stream                  = handle->get_stream();
  const i_t n_vars             = problem.n_variables;
  const i_t n_rows             = problem.n_constraints;
  const f_t tol                = problem.tolerances.presolve_absolute_tolerance;
  if (problem.empty || n_vars == 0 || n_rows == 0) return false;

  auto h_off   = cuopt::host_copy(problem.offsets, stream);
  auto h_var   = cuopt::host_copy(problem.variables, stream);
  auto h_coef  = cuopt::host_copy(problem.coefficients, stream);
  auto h_clb   = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  auto h_cub   = cuopt::host_copy(problem.constraint_upper_bounds, stream);
  auto h_vb    = cuopt::host_copy(problem.variable_bounds, stream);
  auto h_vtype = cuopt::host_copy(problem.variable_types, stream);
  auto h_obj   = cuopt::host_copy(problem.objective_coefficients, stream);
  auto h_vmap  = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  handle->sync_stream();

  const i_t nnz0 = h_off.back();
  work_units     = 2.0 * nnz0 + 2.0 * n_vars + n_rows;

  if (timer.check_time_limit()) return false;

  std::vector<i_t> offsets(h_off.begin(), h_off.end());
  std::vector<i_t> variables(h_var.begin(), h_var.end());
  std::vector<f_t> coefficients(h_coef.begin(), h_coef.end());
  std::vector<f_t> row_lower(h_clb.begin(), h_clb.end());
  std::vector<f_t> row_upper(h_cub.begin(), h_cub.end());
  std::vector<f_t> col_lower(n_vars), col_upper(n_vars);
  std::vector<uint8_t> is_integer(n_vars);
  for (i_t c = 0; c < n_vars; ++c) {
    col_lower[c]  = get_lower(h_vb[c]);
    col_upper[c]  = get_upper(h_vb[c]);
    is_integer[c] = (h_vtype[c] == var_t::INTEGER) ? 1 : 0;
  }
  std::vector<f_t> obj(h_obj.begin(), h_obj.end());

  if (timer.check_time_limit()) return false;  // the reducer below walks the CSR again

  // ---- detect + sanity check (probing-cache implication closure). Projection of each candidate
  // block runs on the GPU: the batched detector stages scope-disjoint candidates per round and
  // hands the whole batch to bve_project_batch_gpu (one enumeration-kernel launch per shape-bin),
  // which fills feas/witness; commit (prime-implicate CNF + inline sanity check) then runs on the
  // host. ----
  bve_reducer_t<i_t, f_t> reducer(n_vars,
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
                                  tol,
                                  boundary_cap,
                                  scope_cap,
                                  clause_growth_margin);
  t_setup = wall.elapsed_time();
  probe_findings_t<i_t> current_id_findings;
  bool detected_infeasible = false;
  bve_plan_t<i_t, f_t> plan =
    bve_detect_closure_batched<i_t, f_t>(*handle,
                                         reducer,
                                         impl_adj,
                                         timer,
                                         work_units,
                                         out_findings != nullptr ? &current_id_findings : nullptr,
                                         &detected_infeasible);
  t_detect = wall.elapsed_time() - t_setup;

  if (detected_infeasible) {
    cuopt_assert(plan.reductions.empty(), "an infeasibility proof must abandon the round's plan");
    if (out_infeasible != nullptr) *out_infeasible = true;
    return false;
  }

  // Projection findings hold for the block's rows whether or not the block was eliminated, so they
  // are exported before the no-reduction exit; the rejected blocks are often the interesting ones.
  if (out_findings != nullptr) {
    auto to_original = [&](i_t column) {
      cuopt_assert(column >= 0 && column < (i_t)h_vmap.size(), "column outside variable_mapping");
      return (i_t)h_vmap[column];
    };
    out_findings->forcings.reserve(out_findings->forcings.size() +
                                   current_id_findings.forcings.size());
    for (const auto& forcing : current_id_findings.forcings) {
      out_findings->forcings.push_back({to_original(forcing.var),
                                        to_original(forcing.forced_var),
                                        forcing.value,
                                        forcing.forced_value});
    }
    for (const auto& [column, value] : current_id_findings.fixings)
      out_findings->fixings.emplace_back(to_original(column), value);
  }

  if (plan.reductions.empty()) return false;

  // ---- build the reduced forward CSR, append clause rows ----
  const double t_install_begin = wall.elapsed_time();
  std::vector<char> removed(n_rows, 0);
  for (i_t r : plan.removed_rows)
    removed[r] = 1;
  std::vector<i_t> new_off, new_var;
  std::vector<f_t> new_coef, new_clb, new_cub;
  new_off.reserve(n_rows + plan.added_rows.size() + 1);
  new_off.push_back(0);
  for (i_t r = 0; r < n_rows; ++r) {
    if (removed[r]) continue;
    for (i_t k = offsets[r]; k < offsets[r + 1]; ++k) {
      new_var.push_back(variables[k]);
      new_coef.push_back(coefficients[k]);
    }
    new_off.push_back(new_var.size());
    new_clb.push_back(row_lower[r]);
    new_cub.push_back(row_upper[r]);
  }
  for (const auto& ar : plan.added_rows) {
    cuopt_assert(!ar.terms.empty(), "installing a term-free row loses whatever it constrained");
    for (const auto& [var, coef] : ar.terms) {
      new_var.push_back(var);
      new_coef.push_back(coef);
    }
    new_off.push_back(new_var.size());
    new_clb.push_back(ar.lower);  // eliminated interior cols become empty (only in removed rows)
    // clause rows are >= no-goods; upper is +inf (problem_t convention)
    new_cub.push_back(std::numeric_limits<f_t>::infinity());
  }

  work_units += new_var.size() + new_clb.size();
  problem.set_constraints_from_host_csr(new_off, new_var, new_coef, new_clb, new_cub, {});

  append_bve_reconstructions(plan, h_vmap, problem.presolve_data, work_units);
  t_install = wall.elapsed_time() - t_install_begin;

  const double t_compact_begin = wall.elapsed_time();
  work_units += n_vars + new_var.size();
  trivial_presolve(problem, /*remap_cache_ids=*/true);
  handle->sync_stream();
  t_compact              = wall.elapsed_time() - t_compact_begin;
  const i_t reduced_cols = n_vars - problem.n_variables;
  const i_t reduced_rows = n_rows - problem.n_constraints;
  if (reduced_cols > 0 || reduced_rows > 0) {
    CUOPT_LOG_DEBUG("Block-BVE reduced %d columns, %d rows", reduced_cols, reduced_rows);
  }
#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)
  const i_t fractional_coefs =
    thrust::count_if(handle->get_thrust_policy(),
                     problem.coefficients.begin(),
                     problem.coefficients.end(),
                     [] __device__(f_t v) -> bool { return floor(v) != v; });
  CUOPT_LOG_DEBUG("Block-BVE: %d fractional coefficients in A", fractional_coefs);
#endif
  return true;
}

template <typename i_t, typename f_t>
bool block_bve_phase(bound_presolve_t<i_t, f_t>& bound_presolve,
                     problem_t<i_t, f_t>& problem,
                     const timer_t& deadline)
{
  if (const char* disabled = std::getenv("CUOPT_DISABLE_BLOCK_BVE");
      disabled != nullptr && std::atoi(disabled) != 0) {
    CUOPT_LOG_DEBUG("Block-BVE disabled via CUOPT_DISABLE_BLOCK_BVE");
    return true;
  }

  if (bound_presolve.probing_cache.probing_cache.empty()) {
    CUOPT_LOG_DEBUG("Block-BVE skipped: the probing cache is empty");
    return true;
  }

  const i_t n_vars_before_phase = problem.n_variables;
  const i_t n_rows_before_phase = problem.n_constraints;

  // Implications read off the projection tables, accumulated across rounds. They feed the next
  // round's adjacency (pairs the cache never held) and are folded back into the cache afterwards.
  probe_findings_t<i_t> findings;
  timer_t stage_timer(deadline.clamp_remaining_time(BVE_STAGE_TIME_LIMIT));
  for (i_t round = 0; round < BVE_MAX_ROUNDS; ++round) {
    if (problem.empty || deadline.check_time_limit() || stage_timer.check_time_limit()) { break; }

    if (!bve_has_stageable_row(problem)) {
      CUOPT_LOG_DEBUG("Block-BVE skipped: every row exceeds the %d-nonzero block row cap",
                      BVE_MAX_ROW_LEN);
      break;
    }

    const i_t n_vars_before = problem.n_variables;
    const i_t n_rows_before = problem.n_constraints;
    auto impl_adj           = bve_build_impl_adj(bound_presolve.probing_cache,
                                       problem.reverse_original_ids,
                                       problem.n_variables,
                                       stage_timer,
                                       &findings);
    if (stage_timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Block-BVE hit its %.2fs phase limit building the implication graph",
                      stage_timer.get_time_limit());
      break;
    }

    double work_units      = 0.0;
    bool proved_infeasible = false;
    timer_t round_timer(stage_timer.clamp_remaining_time(deadline.remaining_time()));
    const bool reduced =
      block_bve_presolve(problem, impl_adj, round_timer, work_units, &findings, &proved_infeasible);
    if (proved_infeasible) {
      CUOPT_LOG_DEBUG("Block-BVE proved the problem infeasible");
      return false;
    }
    CUOPT_LOG_DEBUG("Block-BVE outer round %d/%d: reduced=%d vars %d->%d rows %d->%d",
                    round + 1,
                    BVE_MAX_ROUNDS,
                    (int)reduced,
                    n_vars_before,
                    problem.n_variables,
                    n_rows_before,
                    problem.n_constraints);
    if (!reduced) { break; }
    if (problem.n_variables >= n_vars_before) { break; }
    if (n_vars_before - problem.n_variables < n_vars_before * BVE_MIN_ROUND_YIELD) { break; }
  }

  // Harvest the projections: tighten the cache in place, pin the variables the blocks left with a
  // single value, then propagate.
  bound_presolve.probing_cache.merge_forcings(findings.forcings, findings.fixings);
  i_t n_fixings = 0;
  if (!deadline.check_time_limit()) {
    if (!apply_bve_fixings(problem, findings.fixings, n_fixings)) { return false; }
    if (n_fixings > 0) { trivial_presolve(problem, /*remap_cache_ids=*/true); }
  }
  const bool changed_model = problem.n_variables != n_vars_before_phase ||
                             problem.n_constraints != n_rows_before_phase || n_fixings > 0;
  if (changed_model) {
    CUOPT_LOG_DEBUG("Block-BVE projections fixed %d variables", n_fixings);
    if (!problem.empty && !deadline.check_time_limit()) {
      bound_presolve.resize(problem);
      auto term_crit = bound_presolve.solve(problem);
      if (bound_presolve.infeas_constraints_count > 0) { return false; }
      if (termination_criterion_t::NO_UPDATE != term_crit) {
        bound_presolve.set_updated_bounds(problem);
      }
    }
  }
  return true;
}

#define INSTANTIATE(F_TYPE)                                                                     \
  template double bve_project_batch_gpu<int, F_TYPE>(                                           \
    const raft::handle_t&, std::vector<bve_candidate_t<int, F_TYPE>>&, F_TYPE, const timer_t&); \
  template std::vector<std::vector<int>> bve_build_impl_adj<int, F_TYPE>(                       \
    const probing_cache_t<int, F_TYPE>&,                                                        \
    const std::vector<int>&,                                                                    \
    int,                                                                                        \
    const timer_t&,                                                                             \
    const probe_findings_t<int>*);                                                              \
  template bool bve_has_stageable_row<int, F_TYPE>(const problem_t<int, F_TYPE>&);              \
  template bool block_bve_phase<int, F_TYPE>(                                                   \
    bound_presolve_t<int, F_TYPE>&, problem_t<int, F_TYPE>&, const timer_t&);                   \
  template bool block_bve_presolve<int, F_TYPE>(problem_t<int, F_TYPE>&,                        \
                                                const std::vector<std::vector<int>>&,           \
                                                timer_t&,                                       \
                                                double&,                                        \
                                                probe_findings_t<int>*,                         \
                                                bool*,                                          \
                                                int,                                            \
                                                int,                                            \
                                                int)

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float);
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double);
#endif

#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
