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

#include <cooperative_groups.h>      // cg::invoke_one (elect one thread of a group)
#include <raft/util/cuda_utils.cuh>  // raft::warpReduce
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <utilities/logger.hpp>
#include <utilities/scope_guard.hpp>
#include <utilities/timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cg = cooperative_groups;

namespace cuopt::mathematical_optimization::mip {

// ===========================================================================================
//  Clause core (projection re-encoding + sanity check) + host detector (declarations in
//  block_bve.cuh)
// ===========================================================================================

// A constraint bound is "infinite" if non-finite or at/above the solver's large-bound sentinel.
template <typename f_t>
static bool bve_bound_finite(f_t x)
{
  return std::isfinite(x) && std::abs(x) < static_cast<f_t>(1e30);
}

int bve_prime_implicates(const uint8_t* feas, int nb, bve_clause_t* out, int cap)
{
  const uint32_t full_mask = (1u << nb) - 1u;
  int n                    = 0;
  for (uint32_t m = 0; m <= full_mask; ++m) {
    if (feas[m]) continue;  // feasible pattern: not forbidden
    uint32_t active = full_mask;
    bool changed    = true;
    while (changed) {
      changed = false;
      for (int j = 0; j < nb; ++j) {
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
    if (n >= cap) return -1;
    bve_clause_t c;
    c.lit_mask = active;
    c.bit_mask = m & active;
    bool dup   = false;
    for (int i = 0; i < n; ++i)
      if (out[i].lit_mask == c.lit_mask && out[i].bit_mask == c.bit_mask) {
        dup = true;
        break;
      }
    if (!dup) out[n++] = c;
  }
  return n;
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
                                       int Bcap_,
                                       int enumcap_,
                                       int margin_)
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
    is_bin[c] = (is_integer[c] && std::abs(col_lower[c]) < tol &&
                 std::abs(col_upper[c] - static_cast<f_t>(1)) < tol)
                  ? 1
                  : 0;
    obj_nz[c] = (obj[c] != f_t(0)) ? 1 : 0;
  }
  rows.reserve(static_cast<size_t>(n_rows_orig) * 2);
  for (i_t r = 0; r < n_rows_orig; ++r) {
    work_row_t R;
    R.active   = true;
    R.original = true;
    R.lo       = bve_bound_finite(row_lower[r]) ? row_lower[r] : -INF;
    R.up       = bve_bound_finite(row_upper[r]) ? row_upper[r] : INF;
    for (i_t k = offsets[r]; k < offsets[r + 1]; ++k)
      R.terms.emplace_back(variables[k], coefficients[k]);
    i_t id = static_cast<i_t>(rows.size());
    rows.push_back(std::move(R));
    for (auto& p : rows[id].terms)
      col2rows[p.first].insert(id);
  }
}

template <typename i_t, typename f_t>
std::unordered_set<i_t> bve_reducer_t<i_t, f_t>::rows_of(const std::vector<i_t>& interior) const
{
  std::unordered_set<i_t> G;
  for (i_t a : interior)
    for (i_t r : col2rows[a])
      G.insert(r);
  return G;
}

template <typename i_t, typename f_t>
std::vector<i_t> bve_reducer_t<i_t, f_t>::boundary_of(const std::unordered_set<i_t>& G,
                                                      const std::unordered_set<i_t>& A) const
{
  std::unordered_set<i_t> b;
  for (i_t r : G)
    for (auto& p : rows[r].terms)
      if (!A.count(p.first)) b.insert(p.first);
  return std::vector<i_t>(b.begin(), b.end());
}

template <typename i_t, typename f_t>
int bve_reducer_t<i_t, f_t>::boundary_size(const std::vector<i_t>& interior) const
{
  std::unordered_set<i_t> A(interior.begin(), interior.end());
  return static_cast<int>(boundary_of(rows_of(interior), A).size());
}

template <typename i_t, typename f_t>
bool bve_reducer_t<i_t, f_t>::stage(const std::vector<i_t>& interior_in,
                                    bve_candidate_t<i_t, f_t>& out)
{
  std::vector<i_t> interior(interior_in.begin(), interior_in.end());
  std::sort(interior.begin(), interior.end());
  std::unordered_set<i_t> A(interior.begin(), interior.end());
  std::unordered_set<i_t> Gset = rows_of(interior);
  std::vector<i_t> Gl(Gset.begin(), Gset.end());
  std::sort(Gl.begin(),
            Gl.end());  // row order is result-invariant; sorting improves GPU shape-binning
  std::vector<i_t> bnd = boundary_of(Gset, A);
  std::sort(bnd.begin(), bnd.end());
  const int nb = static_cast<int>(bnd.size());
  const int na = static_cast<int>(interior.size());
  if (nb == 0 || nb > Bcap || na + nb > enumcap) return false;
  for (i_t v : bnd)
    if (!is_bin[v]) return false;
  if (na > BVE_MAX_INTERIOR || nb > BVE_MAX_BOUNDARY || na + nb > BVE_MAX_SCOPE) return false;
  if (static_cast<int>(Gl.size()) > BVE_MAX_ROWS) return false;

  bve_block_t<f_t>& blk = out.blk;
  blk.na                = na;
  blk.nb                = nb;
  blk.n_rows            = static_cast<int>(Gl.size());
  std::unordered_map<i_t, int> local;
  for (int j = 0; j < na; ++j)
    local[interior[j]] = j;
  for (int j = 0; j < nb; ++j)
    local[bnd[j]] = na + j;
  int nzc           = 0;
  bool row_overflow = false;
  for (int rr = 0; rr < blk.n_rows && !row_overflow; ++rr) {
    const i_t r     = Gl[rr];
    blk.row_off[rr] = nzc;
    if (static_cast<int>(rows[r].terms.size()) > BVE_MAX_ROW_LEN ||
        nzc + static_cast<int>(rows[r].terms.size()) > BVE_MAX_NNZ) {
      row_overflow = true;
      break;
    }
    for (auto& p : rows[r].terms) {
      blk.row_var[nzc]  = local[p.first];
      blk.row_coef[nzc] = p.second;
      ++nzc;
    }
    blk.row_lo[rr] = rows[r].lo;
    blk.row_up[rr] = rows[r].up;
  }
  if (row_overflow) return false;
  blk.row_off[blk.n_rows] = nzc;

  out.interior = std::move(interior);
  out.boundary = std::move(bnd);
  out.rows     = std::move(Gl);
  for (uint32_t m = 0; m < (1u << nb); ++m) {
    out.feas[m]    = 0;
    out.witness[m] = 0u;
  }
  return true;
}

template <typename i_t, typename f_t>
bool bve_reducer_t<i_t, f_t>::commit_projected(const bve_candidate_t<i_t, f_t>& cand)
{
  const int nb = cand.blk.nb;
  const int na = cand.blk.na;
  bve_clause_t clauses[BVE_MAX_CLAUSES];
  const int n_clauses = bve_prime_implicates(cand.feas, nb, clauses, BVE_MAX_CLAUSES);
  if (n_clauses < 0) return false;                         // clause explosion past cap
  if (n_clauses > cand.blk.n_rows + margin) return false;  // growth gate
  if (!bve_sanity_check(cand.feas, nb, clauses, n_clauses))
    return false;  // sanity check failed => keep block

  bve_reduction_t<i_t> red;
  red.interior = cand.interior;
  red.boundary = cand.boundary;
  red.witness.assign(cand.witness, cand.witness + (static_cast<size_t>(1) << nb));
  plan.reductions.push_back(std::move(red));

  for (i_t r : cand.rows) {
    for (auto& p : rows[r].terms)
      col2rows[p.first].erase(r);
    rows[r].active = false;
    rows[r].terms.clear();
  }
  const f_t INF = std::numeric_limits<f_t>::infinity();
  for (int ci = 0; ci < n_clauses; ++ci) {
    const uint32_t lit = clauses[ci].lit_mask;
    const uint32_t bit = clauses[ci].bit_mask;
    work_row_t R;
    R.active   = true;
    R.original = false;
    R.up       = INF;
    int n1     = 0;
    for (int j = 0; j < nb; ++j)
      if (lit & (1u << j)) {
        const int b = (bit >> j) & 1u;
        R.terms.emplace_back(cand.boundary[j], b ? static_cast<f_t>(-1) : static_cast<f_t>(1));
        n1 += b;
      }
    R.lo   = static_cast<f_t>(1 - n1);
    i_t id = static_cast<i_t>(rows.size());
    rows.push_back(std::move(R));
    for (auto& p : rows[id].terms)
      col2rows[p.first].insert(id);
  }
  for (i_t a : cand.interior) {
    col2rows[a].clear();
    done[a] = 1;
    plan.eliminated_cols.push_back(a);
  }
  plan.n_blocks += 1;
  plan.n_elim_cols += na;
  return true;
}

template <typename i_t, typename f_t>
bve_plan_t<i_t, f_t> bve_reducer_t<i_t, f_t>::finalize()
{
  for (i_t r = 0; r < n_rows_orig; ++r)
    if (!rows[r].active) plan.removed_rows.push_back(r);
  for (size_t r = static_cast<size_t>(n_rows_orig); r < rows.size(); ++r)
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
  for (i_t c = 0; c < n_vars; ++c)
    if (!col2rows[c].empty()) plan.final_cols += 1;
  for (const auto& R : rows)
    if (R.active) plan.final_rows += 1;
  return plan;
}

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

  const i_t nnz           = row_start[nrows];
  const i_t num_patterns  = static_cast<i_t>(1) << nb;
  const i_t num_interiors = static_cast<i_t>(1) << na;
  // num_blocks * 2^nb * 2^na can exceed 2^31 for a large shape-bin, so the assignment index is
  // 64-bit
  const long long num_assignments =
    static_cast<long long>(num_blocks) * num_patterns * num_interiors;

  const int lane_id   = threadIdx.x % 32;
  const int warp_id   = threadIdx.x / 32;
  const int num_warps = blockDim.x / 32;

  const auto cta  = cg::this_thread_block();  // the CUDA thread block (blockIdx/blockDim); a BVE
  const auto warp = cg::tiled_partition<32>(cta);  // "block" below is one candidate BVE block

  // one CTA per assignment (block, m, am), grid-strided over CTAs
  for (long long assignment = blockIdx.x; assignment < num_assignments; assignment += gridDim.x) {
    const i_t interior_pattern = static_cast<i_t>(assignment % num_interiors);
    const i_t boundary_pattern = static_cast<i_t>((assignment / num_interiors) % num_patterns);
    const i_t block =
      static_cast<i_t>(assignment / (static_cast<long long>(num_interiors) * num_patterns));

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
      // butterfly reduce: `sum` is broadcast to every lane, so the elected lane holds it
      const f_t sum = raft::warpReduce(partial);
      cg::invoke_one(warp, [&]() {
        row_satisfied[row] =
          (sum <= upper[row] + tolerance && sum >= lower[row] - tolerance) ? 1 : 0;
      });
    }
    __syncthreads();

    // AND the per-row bits; if this assignment is feasible, offer its interior as a witness
    cg::invoke_one(cta, [&]() {
      uint8_t feasible = 1;
      for (i_t row = 0; row < nrows; ++row) {
        feasible &= row_satisfied[row];
      }
      if (feasible) {
        atomicMin(&out_witness[block * num_patterns + boundary_pattern],
                  static_cast<uint32_t>(interior_pattern));
      }
    });
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
  // shared; only the coefficients and row bounds differ per block.
  std::map<std::vector<i_t>, std::vector<size_t>> bins;
  for (size_t i = 0; i < cands.size(); ++i) {
    const auto& blk = cands[i].blk;
    const i_t nnz   = blk.row_off[blk.n_rows];
    std::vector<i_t> key;
    key.reserve(4 + (blk.n_rows + 1) + nnz);
    key.push_back(blk.na);
    key.push_back(blk.nb);
    key.push_back(blk.n_rows);
    key.push_back(nnz);
    for (int r = 0; r <= blk.n_rows; ++r)
      key.push_back(blk.row_off[r]);
    for (int k = 0; k < nnz; ++k)
      key.push_back(blk.row_var[k]);
    bins[key].push_back(i);
  }

  for (const auto& kv : bins) {
    const std::vector<size_t>& idxs = kv.second;
    const auto& proto               = cands[idxs[0]].blk;
    const i_t na                    = proto.na;
    const i_t nb                    = proto.nb;
    const i_t nrows                 = proto.n_rows;
    const i_t nnz                   = proto.row_off[nrows];
    const i_t num                   = static_cast<i_t>(idxs.size());
    const i_t patterns              = static_cast<i_t>(1) << nb;

    // ---- host staging: shared layout once, per-block coeffs/bounds concatenated ----
    std::vector<i_t> h_row_start(proto.row_off, proto.row_off + nrows + 1);
    std::vector<i_t> h_local_var(proto.row_var, proto.row_var + nnz);
    std::vector<f_t> h_coeffs(static_cast<size_t>(num) * nnz);
    std::vector<f_t> h_lower(static_cast<size_t>(num) * nrows);
    std::vector<f_t> h_upper(static_cast<size_t>(num) * nrows);
    for (size_t g = 0; g < idxs.size(); ++g) {
      const auto& blk = cands[idxs[g]].blk;
      std::copy(blk.row_coef, blk.row_coef + nnz, h_coeffs.begin() + g * nnz);
      std::copy(blk.row_lo, blk.row_lo + nrows, h_lower.begin() + g * nrows);
      std::copy(blk.row_up, blk.row_up + nrows, h_upper.begin() + g * nrows);
    }

    // ---- device upload ----
    rmm::device_uvector<i_t> d_row_start(h_row_start.size(), stream);
    rmm::device_uvector<i_t> d_local_var(h_local_var.size(), stream);
    rmm::device_uvector<f_t> d_coeffs(h_coeffs.size(), stream);
    rmm::device_uvector<f_t> d_lower(h_lower.size(), stream);
    rmm::device_uvector<f_t> d_upper(h_upper.size(), stream);
    rmm::device_uvector<uint32_t> d_witness(static_cast<size_t>(num) * patterns, stream);
    raft::copy(d_row_start.data(), h_row_start.data(), h_row_start.size(), stream);
    raft::copy(d_local_var.data(), h_local_var.data(), h_local_var.size(), stream);
    raft::copy(d_coeffs.data(), h_coeffs.data(), h_coeffs.size(), stream);
    raft::copy(d_lower.data(), h_lower.data(), h_lower.size(), stream);
    raft::copy(d_upper.data(), h_upper.data(), h_upper.size(), stream);
    // sentinel 0xFFFFFFFF (every byte 0xFF) marks a boundary pattern with no feasible interior yet
    RAFT_CUDA_TRY(
      cudaMemsetAsync(d_witness.data(), 0xFF, d_witness.size() * sizeof(uint32_t), stream));

    // ---- launch: one warp per row, one CTA per (block, m, am) assignment, grid-strided ----
    const int num_warps   = std::min<i_t>(nrows, 32);
    const int cta_dim     = num_warps * 32;
    const size_t shmem    = static_cast<size_t>(nrows) * sizeof(uint8_t);
    const long long total = static_cast<long long>(num) * patterns * (static_cast<i_t>(1) << na);
    const int grid        = static_cast<int>(std::min<long long>(total, 65535));
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

    // Closed-form work: one assignment evaluates nnz coefficient multiplies.
    work_units += total * nnz;

    // ---- readback: witness sentinel -> feas; smallest feasible interior -> witness ----
    std::vector<uint32_t> h_witness(static_cast<size_t>(num) * patterns);
    raft::copy(h_witness.data(), d_witness.data(), h_witness.size(), stream);
    handle.sync_stream();
    for (size_t g = 0; g < idxs.size(); ++g) {
      auto& cand = cands[idxs[g]];
      for (i_t m = 0; m < patterns; ++m) {
        const uint32_t w    = h_witness[g * patterns + m];
        const bool feasible = (w != 0xFFFFFFFFu);
        cand.feas[m]        = feasible ? 1 : 0;
        cand.witness[m]     = feasible ? w : 0u;
      }
    }
  }
  return work_units;
}

// ---- production detector: round-based, scope-disjoint, one GPU projection launch per round ----
//
// Implication-closure block growth over the probing-cache adjacency (same shrink rule as the host
// reference bve_detect_closure), restructured so many candidate blocks are projected in ONE GPU
// launch. Within a round the working model is FROZEN — every seed grows its interior against the
// same model. Because that growth is read-only on the model, it runs in an OpenMP parallel-for
// across the round's seeds; the results are deterministic per seed and acceptance is then applied
// serially in seed order, so the committed plan is identical to a serial run. Candidates are staged
// and only mutually SCOPE-DISJOINT ones (no shared interior or boundary column, which also forbids
// a shared row) are accepted into the batch. The batch is projected on the device
// (bve_project_batch_gpu), then committed on the host; because the accepted candidates touch
// disjoint columns/rows, commit order is irrelevant and each block's staged projection is still
// valid at commit time. Candidates deferred for overlap are retried in later rounds; the loop stops
// when a round accepts nothing or commits nothing (each committing round retires >= 1 column =>
// terminates).
//
// Coverage is NOT bit-for-bit identical to the sequential bve_detect_closure: there, a later seed
// grows against the model already mutated by earlier commits, whereas here all growth in a round
// sees the frozen pre-round model. Both are sound (every committed block passes the same inline
// sanity check) and both process each seed once; the set of blocks found can differ. The
// scope-disjoint rule is deliberately conservative (it also rejects candidates that merely share a
// boundary column, which would be safe); relax it if per-round batch sizes prove too small.
// TU-local (only the pass uses it).
template <typename i_t, typename f_t>
static bve_plan_t<i_t, f_t> bve_detect_closure_batched(
  const raft::handle_t& handle,
  bve_reducer_t<i_t, f_t>& R,
  const std::vector<std::vector<i_t>>& impl_adj,
  timer_t& timer,
  double& work_units)
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

  std::vector<char> attempted(R.n_vars, 0);  // a seed is attempted once (whether or not it commits)
  for (;;) {
    if (timer.check_time_limit()) break;

    // This round's live seeds, in the deterministic growth order.
    std::vector<i_t> round_seeds;
    for (i_t seed : order)
      if (!attempted[seed] && !R.done[seed] && !R.col2rows[seed].empty())
        round_seeds.push_back(seed);
    if (round_seeds.empty()) break;

    // Grow each seed's interior against the FROZEN model (same shrink rule as bve_detect_closure).
    // This is read-only on R -- boundary_size / rows_of / boundary_of are const and only allocate
    // thread-local scratch -- so it parallelizes across seeds. Growth is deterministic per seed and
    // acceptance below runs in round_seeds order, so the committed plan is identical to the serial
    // version: this is a pure speedup, not a behaviour change.
    std::vector<std::vector<i_t>> interiors(round_seeds.size());
    std::vector<int64_t> growth_ops(round_seeds.size(), 0);
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < static_cast<int>(round_seeds.size()); ++k) {
      std::unordered_set<i_t> A = {round_seeds[k]};
      int64_t ops               = 0;
      for (;;) {
        std::vector<i_t> Av(A.begin(), A.end());
        const int cur = R.boundary_size(Av);
        ops += Av.size();
        std::unordered_set<i_t> cands_w;
        for (i_t a : A)
          if (has_adj(a))
            for (i_t w : impl_adj[a])
              if (!A.count(w) && eligible(w)) cands_w.insert(w);
        ops += cands_w.size();
        i_t best    = static_cast<i_t>(-1);
        int best_nb = cur;
        for (i_t w : cands_w) {
          Av.push_back(
            w);  // test interior ∪ {w}, then pop to reuse the buffer (no per-candidate copy)
          const int na = static_cast<int>(Av.size());
          const int nb = R.boundary_size(Av);
          ops += Av.size();
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
      growth_ops[k] = ops;
    }
    int64_t max_growth_ops = 0;
    for (int64_t ops : growth_ops)
      max_growth_ops = std::max(max_growth_ops, ops);
    work_units += max_growth_ops;

    if (timer.check_time_limit()) break;

    // Serial: stage each grown interior and greedily accept mutually SCOPE-DISJOINT candidates, in
    // round_seeds order. Nothing mutates the model until commit, so this stays serial.
    std::vector<bve_candidate_t<i_t, f_t>> cands;
    std::unordered_set<i_t> claimed;  // interior+boundary columns of already-accepted candidates
    for (size_t k = 0; k < round_seeds.size(); ++k) {
      if (timer.check_time_limit()) break;
      const i_t seed = round_seeds[k];
      bve_candidate_t<i_t, f_t> cand;
      if (!R.stage(interiors[k], cand)) {
        attempted[seed] =
          1;  // failed the caps against this model; treat as one touch, like sequential
        continue;
      }
      work_units += cand.blk.row_off[cand.blk.n_rows];
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
      if (overlap) continue;  // scope collides with an accepted candidate; defer to a later round

      attempted[seed] = 1;
      for (i_t c : cand.interior)
        claimed.insert(c);
      for (i_t c : cand.boundary)
        claimed.insert(c);
      cands.push_back(std::move(cand));
    }

    if (cands.empty() || timer.check_time_limit()) break;
    work_units += bve_project_batch_gpu<i_t, f_t>(handle, cands, R.tol);
    if (timer.check_time_limit()) break;
    int committed = 0;
    for (auto& cand : cands) {
      if (timer.check_time_limit()) break;
      // Prime-implicate generation + sanity check scale with the feasibility table size.
      work_units += (1 << cand.blk.nb);
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
    if (original_id < 0 || original_id >= static_cast<i_t>(reverse_original_ids.size())) return -1;
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
                        int Bcap,
                        int enumcap,
                        int margin)
{
  work_units = 0.0;
  // Local wall clock for the DEBUG total; `timer` is the caller's stage deadline.
  timer_t wall(std::numeric_limits<double>::infinity());
  auto timer_raii_guard = cuopt::scope_guard([&]() {
    CUOPT_LOG_DEBUG(
      "Block-BVE presolve time: %.2f work units: %.6g", wall.elapsed_time(), work_units);
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
  bve_plan_t<i_t, f_t> plan =
    bve_detect_closure_batched<i_t, f_t>(*handle, reducer, impl_adj, timer, work_units);
  if (plan.n_blocks == 0) return false;

  // ---- 4. build the reduced forward CSR: keep original rows not removed, append clause rows ----
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
    new_off.push_back(static_cast<i_t>(new_var.size()));
    new_clb.push_back(row_lower[r]);
    new_cub.push_back(row_upper[r]);
  }
  for (const auto& ar : plan.added_rows) {
    for (size_t t = 0; t < ar.vars.size(); ++t) {
      new_var.push_back(ar.vars[t]);
      new_coef.push_back(ar.coeffs[t]);
    }
    new_off.push_back(static_cast<i_t>(new_var.size()));
    new_clb.push_back(ar.lower);  // eliminated interior cols become empty (only in removed rows)
    new_cub.push_back(
      ar.upper);  // clause rows are >= no-goods; upper is +inf (problem_t convention)
  }
  // ---- 5. install the rewritten rows into problem_t. set_constraint_matrix_from_host does the
  // full constraint-side rebuild (matrix, bounds, transpose, combined bounds, and the
  // n_constraints-sized auxiliary buffers); recompute_auxilliary_data then refreshes the
  // variable/constraint-graph tables (the column set is unchanged here, but the constraint graph
  // is). ----
  problem.set_constraint_matrix_from_host(new_off, new_var, new_coef, new_clb, new_cub);
  problem.recompute_auxilliary_data(false);

  // ---- 6. record reconstructions, translating detection-space ids -> post-Papilo
  // (variable_mapping value) frame, which is the frame post_process_assignment replays in. Commit
  // order preserved. ----
  auto& recs = problem.presolve_data.block_reconstructions;
  recs.reserve(recs.size() + plan.reductions.size());
  for (const auto& red : plan.reductions) {
    block_reconstruction_t<i_t> rec;
    rec.interior.reserve(red.interior.size());
    for (i_t c : red.interior)
      rec.interior.push_back(h_vmap[c]);
    rec.boundary.reserve(red.boundary.size());
    for (i_t c : red.boundary)
      rec.boundary.push_back(h_vmap[c]);
    rec.witness = red.witness;
    recs.push_back(std::move(rec));
  }

  // ---- 7. compact the now-empty interior columns and update variable_mapping ----
  trivial_presolve(problem, /*remap_cache_ids=*/true);
  handle->sync_stream();
  return true;
}

#define INSTANTIATE(F_TYPE)                                                           \
  template struct bve_reducer_t<int, F_TYPE>;                                         \
  template double bve_project_batch_gpu<int, F_TYPE>(                                 \
    const raft::handle_t&, std::vector<bve_candidate_t<int, F_TYPE>>&, F_TYPE);       \
  template std::vector<std::vector<int>> bve_build_impl_adj<int, F_TYPE>(             \
    const probing_cache_t<int, F_TYPE>&, const std::vector<int>&, int);               \
  template bool block_bve_presolve<int, F_TYPE>(problem_t<int, F_TYPE>&,              \
                                                const std::vector<std::vector<int>>&, \
                                                timer_t&,                             \
                                                double&,                              \
                                                int,                                  \
                                                int,                                  \
                                                int)

INSTANTIATE(double);
#ifdef MIP_INSTANTIATE_FLOAT
INSTANTIATE(float);
#endif
#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
