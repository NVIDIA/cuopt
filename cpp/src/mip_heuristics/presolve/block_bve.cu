/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "block_bve.cuh"
#include "trivial_presolve.cuh"

#include <mip_heuristics/problem/presolve_data.cuh>
#include <mip_heuristics/utils.cuh>

#include <utilities/integer_scaling.hpp>  // find_scaling_rational (exact row integerization)

#include <raft/util/cuda_utils.cuh>  // raft::warpReduce
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/bit>  // cuda::bitfield_extract

#include <utilities/logger.hpp>
#include <utilities/scope_guard.hpp>
#include <utilities/timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// ===========================================================================================
//  Clause core (projection re-encoding + sanity check) + host detector (declarations in
//  block_bve.cuh)
// ===========================================================================================

// A constraint bound is "infinite" if non-finite or at/above the solver's large-bound sentinel.
template <typename f_t>
static bool bve_bound_finite(f_t x)
{
  return std::isfinite(x) && std::abs(x) < f_t(1e30);
}

// Largest per-row rational multiplier / denominator we will apply. A row that would need a larger
// multiplier to become integer is treated as not exactly representable (passed to
// find_scaling_rational as its maxdnom/maxfinal caps).
static constexpr int64_t BVE_INT_SCALE_MAX = 1000000;  // 1e6
// The exact subset sum (<= BVE_MAX_ROW_LEN integer terms) plus the bound compare must stay below
// 2^53 so the fp64 projection arithmetic never rounds.
static constexpr double BVE_EXACT_SUM_BUDGET = 9007199254740992.0;  // 2^53

// Scale one block row (coefficients + finite bounds) to integers by a single positive rational
// multiplier so the projection's subset-sum feasibility test is EXACT in fp64: enumerated values
// are binary, so Sigma coeff*value is a subset sum; once every coefficient and finite bound is an
// exactly representable integer of bounded magnitude, that sum (<= BVE_MAX_ROW_LEN terms) never
// rounds and feasibility is an exact integer comparison (projection tol 0). +/-inf bounds are
// ignored (they stay infinite). Returns the multiplier, or 0 if the row does not integerize within
// the caps -- the caller then rejects the whole block (leaves it un-eliminated) rather than risk a
// tolerance-sensitive misclassification on large or non-rational coefficients.
//
// The rationalization reuses find_scaling_rational (utilities/integer_scaling.hpp), the same
// continued-fraction vector->integer scaling used for objective integer-scaling. A strict tolerance
// is passed so only genuinely small-rational coefficients integerize; anything noisier yields NaN
// and the block is rejected, never silently rounded into a different model.
template <typename f_t>
static double bve_row_int_scale(const f_t* coef, int n, f_t lo, f_t up)
{
  std::vector<double> vals;
  vals.reserve(n + 2);
  for (int k = 0; k < n; ++k)
    vals.push_back((double)coef[k]);
  if (bve_bound_finite(lo)) vals.push_back((double)lo);
  if (bve_bound_finite(up)) vals.push_back((double)up);

  const double scale = find_scaling_rational(vals,
                                             /*maxscale=*/1e12,
                                             /*maxdnom=*/BVE_INT_SCALE_MAX,
                                             /*maxfinal=*/(double)BVE_INT_SCALE_MAX,
                                             /*intcheck_tol=*/1e-9);
  if (!std::isfinite(scale) || scale <= 0.0) return 0.0;

  // find_scaling_rational bounds the multiplier, not the resulting magnitude: guard the exactness
  // budget so the subset sum (<= BVE_MAX_ROW_LEN integer terms) stays below 2^53 (no fp rounding).
  double maxabs = 0.0;
  for (double v : vals)
    maxabs = std::max(maxabs, std::abs(v * scale));
  if (maxabs * (double)BVE_MAX_ROW_LEN >= BVE_EXACT_SUM_BUDGET) return 0.0;

  return scale;
}

// Closed-form work estimate for commit_projected: Quine-style literal dropping in
// bve_prime_implicates is Θ(nb · 3^nb); sanity check is Θ(2^nb · #clauses) with #clauses bounded
// by the growth gate (n_rows + margin).
static double bve_commit_wall_ops(int nb, int clause_budget)
{
  cuopt_assert(nb >= 0 && nb <= BVE_MAX_BOUNDARY, "nb out of BVE range");
  double three_nb = 1.0;
  for (int i = 0; i < nb; ++i)
    three_nb *= 3.0;
  return double(nb) * three_nb + double(1 << nb) * double(clause_budget + 1);
}

template <typename i_t, typename f_t>
i_t bve_prime_implicates(const uint8_t* feas, i_t nb, bve_clause_t* out, i_t cap)
{
  const uint32_t full_mask = (1u << nb) - 1u;
  i_t n                    = 0;
  for (uint32_t m = 0; m <= full_mask; ++m) {
    if (feas[m]) continue;  // feasible pattern: not forbidden
    uint32_t active = full_mask;
    bool changed    = true;
    while (changed) {
      changed = false;
      for (i_t j = 0; j < nb; ++j) {
        if (!(active & (1u << j))) continue;
        // positions free to vary if we drop j: everything not currently active, plus j
        const uint32_t dropped = (~active | (1u << j)) & full_mask;
        // active-minus-j positions held at pattern m's bits
        const uint32_t fixed_bits = (active & ~(1u << j)) & m;
        bool all_forbidden        = true;
        for (uint32_t sub = dropped;; sub = (sub - 1u) & dropped) {
          const uint32_t full = fixed_bits | sub;
          if (feas[full]) {
            all_forbidden = false;
            break;
          }
          if (sub == 0u) break;
        }
        if (all_forbidden) {
          active &= ~(1u << j);
          changed = true;
          break;
        }
      }
    }
    bve_clause_t c;
    c.lit_mask = active;
    c.bit_mask = m & active;
    bool dup   = false;
    for (i_t i = 0; i < n; ++i)
      if (out[i].lit_mask == c.lit_mask && out[i].bit_mask == c.bit_mask) {
        dup = true;
        break;
      }
    if (dup) continue;
    if (n >= cap) return -1;
    out[n++] = c;
  }
  return n;
}

template <typename i_t, typename f_t>
bool bve_sanity_check(const uint8_t* feas, i_t nb, const bve_clause_t* clauses, i_t n_clauses)
{
  const uint32_t full_mask = (1u << nb) - 1u;
  for (i_t i = 0; i < n_clauses; ++i)
    if (clauses[i].lit_mask & ~full_mask) return false;  // literals must be on the boundary
  for (uint32_t m = 0; m <= full_mask; ++m) {
    bool crel = true;  // CNF value: AND over clauses of (clause satisfied by pattern m)
    for (i_t i = 0; i < n_clauses && crel; ++i) {
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

// ---- host detector: working model, staged candidates, accumulated plan (all TU-local) ----
namespace {

// Committed elimination in commit order. `witness[pattern]` packs interior values for the boundary
// pattern; reductions are replayed in reverse order during postsolve.
template <typename i_t>
struct bve_reduction_t {
  std::vector<i_t> interior;
  std::vector<i_t> boundary;
  std::vector<uint32_t> witness;  // size 2^boundary.size()
};

// A surviving clause row to append to problem_t (a set-covering no-good over boundary columns).
template <typename i_t, typename f_t>
struct bve_added_row_t {
  std::vector<i_t> vars;
  std::vector<f_t> coeffs;
  f_t lower;
  f_t upper;
};

template <typename i_t, typename f_t>
struct bve_plan_t {
  std::vector<bve_reduction_t<i_t>> reductions;       // commit order
  std::vector<i_t> removed_rows;                      // original row ids to drop
  std::vector<bve_added_row_t<i_t, f_t>> added_rows;  // surviving clause rows
  i_t n_blocks = 0;
};

// Working model and accumulated reduction plan. Candidates are staged without mutation and
// committed only after projection and clause validation.
template <typename i_t, typename f_t>
struct bve_reducer_t {
  struct work_row_t {
    std::vector<std::pair<i_t, f_t>> terms;
    f_t lo, up;
    bool active;
    bool original;
  };

  i_t n_vars, n_rows_orig;
  f_t tol;
  i_t Bcap, enumcap, margin;
  std::vector<work_row_t> rows;
  std::vector<std::unordered_set<i_t>> col2rows;
  std::vector<uint8_t> is_bin, obj_nz, done;
  bve_plan_t<i_t, f_t> plan;

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
                i_t Bcap_,
                i_t enumcap_,
                i_t margin_);

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
  bool commit_projected(const bve_candidate_t<i_t, f_t>& cand);

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
                                       i_t Bcap_,
                                       i_t enumcap_,
                                       i_t margin_)
  : n_vars(n_vars_),
    n_rows_orig(n_rows_orig_),
    tol(tol_),
    Bcap(Bcap_),
    enumcap(enumcap_),
    margin(margin_),
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
    R.active   = true;
    R.original = true;
    R.lo       = bve_bound_finite(row_lower[r]) ? row_lower[r] : -INF;
    R.up       = bve_bound_finite(row_upper[r]) ? row_upper[r] : INF;
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
  ops += (int64_t)interior.size();
  std::unordered_set<i_t> A(interior.begin(), interior.end());
  std::unordered_set<i_t> G;
  for (i_t a : interior)
    for (i_t r : col2rows[a]) {
      ++ops;
      G.insert(r);
    }
  std::unordered_set<i_t> b;
  for (i_t r : G)
    for (const auto& p : rows[r].terms) {
      ++ops;
      if (!A.count(p.first)) b.insert(p.first);
    }
  rows_out.assign(G.begin(), G.end());
  boundary_out.assign(b.begin(), b.end());
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
  std::vector<i_t> Gl, bnd;
  scope_of(interior, Gl, bnd, ops);
  // row order is result-invariant; sorting improves GPU shape-binning
  std::sort(Gl.begin(), Gl.end());
  ops += (int64_t)Gl.size();
  std::sort(bnd.begin(), bnd.end());
  ops += (int64_t)bnd.size();

  const i_t nb = bnd.size();
  const i_t na = interior.size();
  if (nb == 0 || nb > Bcap || na + nb > enumcap) return false;
  for (i_t v : bnd)
    if (!is_bin[v]) return false;
  if (na > BVE_MAX_INTERIOR || nb > BVE_MAX_BOUNDARY || na + nb > BVE_MAX_SCOPE) return false;
  if (Gl.size() > BVE_MAX_ROWS) return false;

  bve_block_t<f_t>& blk = out.blk;
  blk.na                = na;
  blk.nb                = nb;
  blk.n_rows            = Gl.size();
  std::unordered_map<i_t, i_t> local;
  for (i_t j = 0; j < na; ++j)
    local[interior[j]] = j;
  for (i_t j = 0; j < nb; ++j)
    local[bnd[j]] = na + j;
  ops += (int64_t)(na + nb);
  i_t nzc           = 0;
  bool row_overflow = false;
  for (i_t rr = 0; rr < blk.n_rows && !row_overflow; ++rr) {
    const i_t r     = Gl[rr];
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

  // Integerize every row so the GPU projection is exact (tol 0). A row whose coefficients/bounds do
  // not scale to bounded integers is not exactly representable: reject the whole block (leave it
  // un-eliminated) rather than risk a tolerance-sensitive feasibility misclassification on large or
  // non-rational coefficients. Only the projection's internal copy is scaled -- the block rows are
  // dropped from the model and the appended no-goods are scale-independent +/-1 clauses, so this
  // never perturbs the installed model.
  for (int rr = 0; rr < blk.n_rows; ++rr) {
    const int rb = blk.row_off[rr];
    const int re = blk.row_off[rr + 1];
    const double s =
      bve_row_int_scale<f_t>(blk.row_coef + rb, re - rb, blk.row_lo[rr], blk.row_up[rr]);
    if (s == 0.0) return false;
    for (int k = rb; k < re; ++k)
      blk.row_coef[k] = (f_t)std::llround((double)blk.row_coef[k] * s);
    if (bve_bound_finite(blk.row_lo[rr]))
      blk.row_lo[rr] = (f_t)std::llround((double)blk.row_lo[rr] * s);
    if (bve_bound_finite(blk.row_up[rr]))
      blk.row_up[rr] = (f_t)std::llround((double)blk.row_up[rr] * s);
  }

  out.interior = std::move(interior);
  out.boundary = std::move(bnd);
  out.rows     = std::move(Gl);
  for (uint32_t m = 0; m < (1u << nb); ++m) {
    out.feas[m]    = 0;
    out.witness[m] = 0u;
  }
  ops += (int64_t)(1 << nb);
  return true;
}

template <typename i_t, typename f_t>
bool bve_reducer_t<i_t, f_t>::commit_projected(const bve_candidate_t<i_t, f_t>& cand)
{
  const i_t nb = cand.blk.nb;
  bve_clause_t clauses[BVE_MAX_CLAUSES];
  const i_t n_clauses = bve_prime_implicates<i_t, f_t>(cand.feas, nb, clauses, BVE_MAX_CLAUSES);
  if (n_clauses < 0) return false;                         // clause explosion past cap
  if (n_clauses > cand.blk.n_rows + margin) return false;  // growth gate
  if (!bve_sanity_check<i_t, f_t>(cand.feas, nb, clauses, n_clauses))
    return false;  // sanity check failed => keep block

  bve_reduction_t<i_t> red;
  red.interior = cand.interior;
  red.boundary = cand.boundary;
  red.witness.assign(cand.witness, cand.witness + (size_t(1) << nb));
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
    work_row_t R;
    R.active   = true;
    R.original = false;
    R.up       = INF;
    i_t n1     = 0;
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
  plan.n_blocks += 1;
  return true;
}

template <typename i_t, typename f_t>
bve_plan_t<i_t, f_t> bve_reducer_t<i_t, f_t>::finalize()
{
  for (i_t r = 0; r < n_rows_orig; ++r)
    if (!rows[r].active) plan.removed_rows.push_back(r);
  for (size_t r = n_rows_orig; r < rows.size(); ++r)
    if (rows[r].active) {
      bve_added_row_t<i_t, f_t> ar;
      for (auto& p : rows[r].terms) {
        ar.vars.push_back(p.first);
        ar.coeffs.push_back(p.second);
      }
      ar.lower = rows[r].lo;
      ar.upper = rows[r].up;
      plan.added_rows.push_back(std::move(ar));
    }
  return plan;
}

}  // namespace

// ===========================================================================================
//  GPU enumeration projection kernel
// ===========================================================================================

// Exact-enumeration projection kernel, laid out to fill the GPU:
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
//
// Shape (nb, na, nrows, and the row layout) is passed at RUNTIME, not as template parameters: it
// would otherwise need one instantiation per distinct shape. All blocks in a single launch share
// the shape (they are pre-binned), so every CTA still runs the identical loop structure.
// `row_start` and `local_var_of_entry` describe that shared layout; `nnz == row_start[nrows]`.
// `row_satisfied` uses dynamic shared memory of `nrows` bytes.
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
    const i_t interior_pattern = (i_t)cuda::bitfield_extract(a, 0, na);
    const i_t boundary_pattern = (i_t)cuda::bitfield_extract(a, na, nb);
    const i_t block            = (i_t)(a >> (na + nb));

    const f_t* coeffs = block_coeffs + block * nnz;
    const f_t* lower  = block_row_lower + block * nrows;
    const f_t* upper  = block_row_upper + block * nrows;

    // one warp per row (a warp loops over multiple rows when nrows > num_warps)
    for (i_t row = warp_id; row < nrows; row += num_warps) {
      f_t partial = 0;
      for (i_t entry = row_start[row] + lane_id; entry < row_start[row + 1]; entry += 32) {
        const i_t var   = local_var_of_entry[entry];
        const f_t value = (var < na) ? (f_t)((interior_pattern >> var) & 1)
                                     : (f_t)((boundary_pattern >> (var - na)) & 1);
        partial += coeffs[entry] * value;
      }
      // Lane 0 holds the result for both XOR-butterfly and typical down-sweep warp reduces.
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
    __syncthreads();  // guard row_satisfied before the next assignment overwrites it
  }
}

// ---- GPU batch projection: one enumeration-kernel launch per shape-bin ----
// Returns raw work for the enumerations (sum over bins of assignments · nnz).
template <typename i_t, typename f_t>
double bve_project_batch_gpu(const raft::handle_t& handle,
                             std::vector<bve_candidate_t<i_t, f_t>>& cands,
                             f_t tol)
{
  if (cands.empty()) return 0.0;
  auto stream       = handle.get_stream();
  double work_units = 0.0;

  // Bin candidates by identical shape so every CTA in a launch runs the same loop structure. The
  // key is (na, nb, n_rows, nnz, row_off[...], row_var[...]) — everything the kernel reads as
  // shared; only the coefficients and row bounds differ per block. Hash map avoids O(key_len ·
  // log n_bins) tree compares on long keys (up to ~1605 ints at the BVE caps).
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
    // Also clamp to i_t range: the kernel takes num_blocks as i_t.
    const size_t chunk =
      std::max<size_t>(1,
                       std::min(size_t(std::numeric_limits<i_t>::max()),
                                BVE_PROJECT_DEVICE_BUDGET / std::max<size_t>(1, bytes_per_block)));

    // Launch dims are CUDA `int` by API.
    const int num_warps = std::min<i_t>(nrows, 32);
    const int cta_dim   = num_warps * 32;
    const size_t shmem  = size_t(nrows) * sizeof(uint8_t);

    for (size_t offset = 0; offset < idxs.size(); offset += chunk) {
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
        cudaMemsetAsync(d_witness.data(), 0xFF, d_witness.size() * sizeof(uint32_t), stream));

      // one warp per row, one CTA per (block, m, am) assignment, grid-strided
      const int64_t total = (int64_t)num * (int64_t)patterns * ((int64_t)1 << na);
      const int grid      = std::min(total, int64_t{65535});
      bve_enumerate_kernel<i_t, f_t><<<grid, cta_dim, shmem, stream>>>(num,
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
      work_units += double(num_sz) * double(nnz + 2 * nrows + patterns);
      work_units += double(total) * double(nnz);

      std::vector<uint32_t> h_witness(num_sz * size_t(patterns));
      raft::copy(h_witness.data(), d_witness.data(), h_witness.size(), stream);
      handle.sync_stream();
      for (size_t g = 0; g < num_sz; ++g) {
        auto& cand = cands[idxs[offset + g]];
        for (i_t m = 0; m < patterns; ++m) {
          const uint32_t w    = h_witness[g * patterns + m];
          const bool feasible = (w != 0xFFFFFFFFu);
          cand.feas[m]        = feasible ? 1 : 0;
          cand.witness[m]     = feasible ? w : 0u;
        }
      }
    }
  }
  return work_units;
}

// ---- production detector: round-based, scope-disjoint, one GPU projection launch per round ----
//
// Implication-closure block growth over the probing-cache adjacency: each seed absorbs the
// implication-neighbor that most shrinks its boundary (subject to enum/interior caps) until no
// such neighbor remains. Restructured so many candidate blocks are projected in ONE GPU launch.
// Within a round the working model is FROZEN — every seed grows its interior against the same
// model. Because that growth is read-only on the model, it runs in an OpenMP parallel-for across
// the round's seeds; the results are deterministic per seed and acceptance is then applied
// serially in seed order, so the committed plan is identical to a serial run of the same frozen
// growth. Candidates are staged and only mutually SCOPE-DISJOINT ones (no shared interior or
// boundary column, which also forbids a shared row) are accepted into the batch. The batch is
// projected on the device (bve_project_batch_gpu), then committed on the host; because the accepted
// candidates touch disjoint columns/rows, commit order is irrelevant and each block's staged
// projection is still valid at commit time. Candidates deferred for overlap are retried in later
// rounds; the loop stops when a round accepts nothing or commits nothing (each committing round
// retires >= 1 column => terminates). The scope-disjoint rule is deliberately conservative (it also
// rejects candidates that merely share a boundary column, which would be safe); relax it if
// per-round batch sizes prove too small. TU-local (only the pass uses it).
template <typename i_t, typename f_t>
static bve_plan_t<i_t, f_t> bve_detect_closure_batched(
  const raft::handle_t& handle,
  bve_reducer_t<i_t, f_t>& R,
  const std::vector<std::vector<i_t>>& impl_adj,
  timer_t& timer,
  double& work_units)
{
  auto has_adj  = [&](i_t v) { return v >= 0 && v < (i_t)impl_adj.size() && !impl_adj[v].empty(); };
  auto eligible = [&](i_t w) {
    return R.is_bin[w] && !R.obj_nz[w] && !R.done[w] && !R.col2rows[w].empty();
  };
  std::vector<i_t> order;
  for (i_t c = 0; c < R.n_vars; ++c)
    if (R.is_bin[c] && !R.obj_nz[c] && !R.col2rows[c].empty() && has_adj(c)) order.push_back(c);
  std::sort(order.begin(), order.end(), [&](i_t a, i_t b) {
    return R.col2rows[a].size() < R.col2rows[b].size();
  });

  std::vector<char> attempted(R.n_vars, 0);  // a seed is attempted once (whether or not it commits)
  // Grow each seed at most once; overlap-deferred seeds only re-stage from the cached interior.
  // Re-growing hubs every round dominated wall; retiring them on first overlap killed reductions.
  std::vector<char> growth_done(R.n_vars, 0);
  std::vector<std::vector<i_t>> growth_interior(R.n_vars);
  for (;;) {
    if (timer.check_time_limit()) break;

    // This round's live seeds, in the deterministic growth order.
    std::vector<i_t> round_seeds;
    for (i_t seed : order)
      if (!attempted[seed] && !R.done[seed] && !R.col2rows[seed].empty())
        round_seeds.push_back(seed);
    if (round_seeds.empty()) break;

    // Grow each seed against the frozen model (read-only on R → OMP-safe). Acceptance below is
    // serial in round_seeds order, so the plan matches a serial frozen-growth run.
    std::vector<std::vector<i_t>> interiors(round_seeds.size());
    std::vector<int64_t> growth_ops(round_seeds.size(), 0);
#pragma omp parallel for schedule(dynamic)
    for (i_t k = 0; k < (i_t)round_seeds.size(); ++k) {
      const i_t seed = round_seeds[k];
      if (growth_done[seed]) {
        interiors[k] = growth_interior[seed];
        continue;
      }
      // Interior A starts as {seed}; greedily absorb neighbors that shrink the boundary.
      std::unordered_set<i_t> A = {seed};
      int64_t ops               = 0;
      std::vector<i_t> probe_rows, probe_bnd;  // scope_of scratch, reused across probes
      for (;;) {
        // Hub fast-path: raw implication degree upper-bounds |cands_w|. Skip boundary walks
        // and adj materialization when the neighborhood is past the probe cap.
        if (A.size() == 1) {
          const i_t s   = *A.begin();
          const i_t deg = has_adj(s) ? (i_t)impl_adj[s].size() : 0;
          if (deg > BVE_MAX_GROWTH_NBRS) break;
        }
        std::vector<i_t> Av(A.begin(), A.end());
        R.scope_of(Av, probe_rows, probe_bnd, ops);
        const i_t cur = probe_bnd.size();
        // Implication-neighbors of A that are still eligible to enter the interior.
        std::unordered_set<i_t> cands_w;
        bool gated = false;
        for (i_t a : A) {
          if (!has_adj(a)) continue;
          for (i_t w : impl_adj[a]) {
            ++ops;
            if (A.count(w) || !eligible(w)) continue;
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
        // Pick the neighbor with the smallest boundary; stop when none strictly improves.
        i_t best    = -1;
        i_t best_nb = cur;
        for (i_t w : cands_w) {
          Av.push_back(w);  // probe A ∪ {w}; pop restores Av
          const i_t na = Av.size();
          R.scope_of(Av, probe_rows, probe_bnd, ops);
          const i_t nb = probe_bnd.size();
          Av.pop_back();
          if (nb < best_nb && na + nb <= R.enumcap && na <= BVE_MAX_INTERIOR) {
            best_nb = nb;
            best    = w;
          }
        }
        if (best < 0) break;
        A.insert(best);
      }
      interiors[k].assign(A.begin(), A.end());
      growth_ops[k]         = ops;
      growth_interior[seed] = interiors[k];
      growth_done[seed]     = 1;
    }
    // OMP growth: wall ≈ critical-path seed (max), not sum across threads.
    int64_t max_growth_ops = 0;
    for (int64_t ops : growth_ops)
      max_growth_ops = std::max(max_growth_ops, ops);
    work_units += double(max_growth_ops);

    if (timer.check_time_limit()) break;

    // Serial: stage each grown interior and greedily accept mutually SCOPE-DISJOINT candidates, in
    // round_seeds order. Nothing mutates the model until commit, so this stays serial.
    std::vector<bve_candidate_t<i_t, f_t>> cands;
    std::unordered_set<i_t> claimed;  // interior+boundary columns of already-accepted candidates
    for (size_t k = 0; k < round_seeds.size(); ++k) {
      if (timer.check_time_limit()) break;
      const i_t seed = round_seeds[k];
      bve_candidate_t<i_t, f_t> cand;
      int64_t stage_ops = 0;
      if (!R.stage(interiors[k], cand, &stage_ops)) {
        work_units += double(stage_ops);
        attempted[seed] =
          1;  // failed the caps against this model; treat as one touch, like sequential
        continue;
      }
      work_units += double(stage_ops);
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
      cands.push_back(std::move(cand));
    }

    if (cands.empty() || timer.check_time_limit()) break;
    // Staged blocks are integerized (bve_row_int_scale), so the subset-sum feasibility test is
    // exact: project with tolerance 0 rather than R.tol.
    work_units += bve_project_batch_gpu<i_t, f_t>(handle, cands, f_t(0));
    if (timer.check_time_limit()) break;
    i_t committed = 0;
    for (auto& cand : cands) {
      if (timer.check_time_limit()) break;
      work_units += bve_commit_wall_ops(cand.blk.nb, cand.blk.n_rows + R.margin);
      if (R.commit_projected(cand)) ++committed;
    }
    if (committed == 0) break;
  }
  return R.finalize();
}

// ---- implication adjacency from the probing cache (original-id -> current column) ----
template <typename i_t, typename f_t>
std::vector<std::vector<i_t>> bve_build_impl_adj(const probing_cache_t<i_t, f_t>& cache,
                                                 const std::vector<i_t>& reverse_original_ids,
                                                 i_t n_vars)
{
  // original-id -> current column index (or -1 if the column no longer exists)
  auto to_current = [&](i_t original_id) -> i_t {
    if (original_id < 0 || original_id >= (i_t)reverse_original_ids.size()) return -1;
    return reverse_original_ids[original_id];
  };
  std::vector<std::unordered_set<i_t>> adj(n_vars);
  for (const auto& kv : cache.probing_cache) {
    const i_t x = to_current(kv.first);
    if (x < 0 || x >= n_vars) continue;
    for (int p = 0; p < 2; ++p) {
      for (const auto& yb : kv.second[p].var_to_cached_bound_map) {
        const i_t y = to_current(yb.first);
        if (y < 0 || y >= n_vars || y == x) continue;
        adj[x].insert(y);
        adj[y].insert(x);
      }
    }
  }
  std::vector<std::vector<i_t>> out(n_vars);
  for (i_t v = 0; v < n_vars; ++v)
    out[v].assign(adj[v].begin(), adj[v].end());
  return out;
}

// ---- the pass: detect (GPU-projected) -> install reduced model -> record reconstructions ----
template <typename i_t, typename f_t>
bool block_bve_presolve(problem_t<i_t, f_t>& problem,
                        const std::vector<std::vector<i_t>>& impl_adj,
                        timer_t& timer,
                        double& work_units,
                        i_t Bcap,
                        i_t enumcap,
                        i_t margin)
{
  work_units = 0.0;
  // Local wall clock for the DEBUG total; `timer` is the caller's stage deadline.
  timer_t wall(std::numeric_limits<double>::infinity());
  double t_setup = 0.0, t_detect = 0.0, t_install = 0.0, t_compact = 0.0;
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

  // ---- 1. host copy of the current (post-Papilo, post-initial-trivial-presolve) model ----
  auto h_off   = cuopt::host_copy(problem.offsets, stream);
  auto h_var   = cuopt::host_copy(problem.variables, stream);
  auto h_coef  = cuopt::host_copy(problem.coefficients, stream);
  auto h_clb   = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  auto h_cub   = cuopt::host_copy(problem.constraint_upper_bounds, stream);
  auto h_vb    = cuopt::host_copy(problem.variable_bounds, stream);
  auto h_vtype = cuopt::host_copy(problem.variable_types, stream);
  auto h_obj   = cuopt::host_copy(problem.objective_coefficients, stream);
  // variable_mapping maps current-space column -> post-Papilo index (the frame postsolve uses)
  auto h_vmap = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  handle->sync_stream();

  // Host mirror + reducer construction (each walks the CSR once).
  const i_t nnz0 = (i_t)h_off.back();
  work_units     = double(2 * nnz0) + double(2 * n_vars) + double(n_rows);

  if (timer.check_time_limit()) return false;

  // ---- 2. detector inputs (i_t CSR, f_t bounds/coeffs) ----
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

  // ---- 3. detect + sanity check (probing-cache implication closure). Projection of each candidate
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
                                  Bcap,
                                  enumcap,
                                  margin);
  t_setup = wall.elapsed_time();
  bve_plan_t<i_t, f_t> plan =
    bve_detect_closure_batched<i_t, f_t>(*handle, reducer, impl_adj, timer, work_units);
  t_detect = wall.elapsed_time() - t_setup;
  if (plan.n_blocks == 0) return false;

  // ---- 4. build the reduced forward CSR: keep original rows not removed, append clause rows ----
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
    for (size_t t = 0; t < ar.vars.size(); ++t) {
      new_var.push_back(ar.vars[t]);
      new_coef.push_back(ar.coeffs[t]);
    }
    new_off.push_back(new_var.size());
    new_clb.push_back(ar.lower);  // eliminated interior cols become empty (only in removed rows)
    new_cub.push_back(
      ar.upper);  // clause rows are >= no-goods; upper is +inf (problem_t convention)
  }
  // ---- 5. install the rewritten rows into problem_t (matrix + derived state) ----
  work_units += double(new_var.size()) + double(new_clb.size());
  problem.set_constraints_from_host_csr(new_off, new_var, new_coef, new_clb, new_cub, {});

  // ---- 6. record reconstructions on the unified append-only log (detection-space ids ->
  // post-Papilo variable_mapping frame). Commit order preserved; postsolve replays reverse. ----
  auto& recs = problem.presolve_data.var_postsolve;
  recs.reserve(recs.size() + plan.reductions.size());
  for (const auto& red : plan.reductions) {
    work_units += double(red.interior.size() + red.boundary.size() + red.witness.size());
    var_postsolve_t<i_t, f_t> rec;
    rec.kind = reconstruction_kind_t::BlockBve;
    rec.bve.interior.reserve(red.interior.size());
    for (i_t c : red.interior) {
      cuopt_assert(c >= 0 && c < (i_t)h_vmap.size(), "interior col out of variable_mapping range");
      rec.bve.interior.push_back(h_vmap[c]);
    }
    rec.bve.boundary.reserve(red.boundary.size());
    for (i_t c : red.boundary) {
      cuopt_assert(c >= 0 && c < (i_t)h_vmap.size(), "boundary col out of variable_mapping range");
      rec.bve.boundary.push_back(h_vmap[c]);
    }
    rec.bve.witness = red.witness;
    recs.push_back(std::move(rec));
  }
  t_install = wall.elapsed_time() - t_install_begin;

  // ---- 7. compact the now-empty interior columns and update variable_mapping ----
  const double t_compact_begin = wall.elapsed_time();
  work_units += double(n_vars) + double(new_var.size());
  trivial_presolve(problem, /*remap_cache_ids=*/true);
  handle->sync_stream();
  t_compact              = wall.elapsed_time() - t_compact_begin;
  const i_t reduced_cols = n_vars - problem.n_variables;
  const i_t reduced_rows = n_rows - problem.n_constraints;
  if (reduced_cols > 0 || reduced_rows > 0) {
    CUOPT_LOG_DEBUG("Block-BVE reduced %d columns, %d rows", reduced_cols, reduced_rows);
  }
  return true;
}

#define INSTANTIATE(F_TYPE)                                                                   \
  template int bve_prime_implicates<int, F_TYPE>(const uint8_t*, int, bve_clause_t*, int);    \
  template bool bve_sanity_check<int, F_TYPE>(const uint8_t*, int, const bve_clause_t*, int); \
  template double bve_project_batch_gpu<int, F_TYPE>(                                         \
    const raft::handle_t&, std::vector<bve_candidate_t<int, F_TYPE>>&, F_TYPE);               \
  template std::vector<std::vector<int>> bve_build_impl_adj<int, F_TYPE>(                     \
    const probing_cache_t<int, F_TYPE>&, const std::vector<int>&, int);                       \
  template bool block_bve_presolve<int, F_TYPE>(problem_t<int, F_TYPE>&,                      \
                                                const std::vector<std::vector<int>>&,         \
                                                timer_t&,                                     \
                                                double&,                                      \
                                                int,                                          \
                                                int,                                          \
                                                int)

INSTANTIATE(double);
#ifdef MIP_INSTANTIATE_FLOAT
INSTANTIATE(float);
#endif
#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
