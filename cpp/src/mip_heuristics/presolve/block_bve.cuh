/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <vector>

#include "probing_cache.cuh"

#include <mip_heuristics/problem/problem.cuh>

#include <raft/core/handle.hpp>
#include <utilities/timer.hpp>

// Eliminates small blocks of zero-objective binary variables by enumerating their existential
// projection onto the remaining boundary variables. Infeasible boundary assignments are encoded as
// prime-implicate no-goods; one feasible interior witness per accepted boundary assignment is
// stored for postsolve. This preserves feasibility and objective value.
//
// Candidate interiors are grown from the probing implication graph and committed only when the
// projected CNF satisfies the bounded-elimination growth limit of Eén and Biere, "Effective
// Preprocessing in SAT through Variable and Clause Elimination" (SAT 2005). Before commit, the
// emitted clauses are checked against the GPU-computed boundary feasibility table.

namespace cuopt::mathematical_optimization::mip {

// Caps for a single enumerated block.
static constexpr int BVE_MAX_BOUNDARY = 12;  // nb  <= 12  => 2^nb <= 4096 feasibility patterns
static constexpr int BVE_MAX_SCOPE    = 16;  // na + nb <=
static constexpr int BVE_MAX_ROWS     = 64;  // |G| (rows spanned by the block); clauses <= |G|
static constexpr int BVE_MAX_ROW_LEN  = 24;  // nnz within one block row (interior+boundary entries)
static constexpr int BVE_MAX_NNZ      = BVE_MAX_ROWS * BVE_MAX_ROW_LEN;
static constexpr int BVE_MAX_CLAUSES  = 64;  // <= |rows| for any committed block
static constexpr int BVE_MAX_PATTERNS = 1 << BVE_MAX_BOUNDARY;

// Packed projection block. Local ids [0, na) are interior and [na, na+nb) are boundary; rows use
// CSR layout and missing bounds are +/- infinity.
template <typename f_t>
struct bve_block_t {
  // Plain int (not i_t): this packed layout is not i_t-templated; all fields are bounded by
  // BVE_MAX_*.
  int na;      // number of interior variables
  int nb;      // number of boundary variables (all must be binary; caller guarantees)
  int n_rows;  // |G|
  int row_off[BVE_MAX_ROWS + 1];
  int row_var[BVE_MAX_NNZ];  // local var id in [0, na+nb)
  f_t row_coef[BVE_MAX_NNZ];
  f_t row_lo[BVE_MAX_ROWS];  // -inf if no lower bound
  f_t row_up[BVE_MAX_ROWS];  // +inf if no upper bound
};

// Boundary clause forbidding patterns that match `bit_mask` at every position in `lit_mask`.
// It is emitted as sum_j (bit_j == 0 ? x_j : -x_j) >= 1 - popcount(bit_mask & lit_mask).
struct bve_clause_t {
  uint32_t lit_mask;
  uint32_t bit_mask;
};

// One bit per boundary pattern. The width tracks the block's own 2^nb, not BVE_MAX_PATTERNS, so
// raising BVE_MAX_BOUNDARY costs nothing on narrower blocks.
using bve_mask_t = std::vector<uint64_t>;

// Buffers the CNF construction reuses across blocks: `valid` alone is 4^nb bytes, so per-block
// allocation would dominate at wide boundaries.
struct bve_cover_scratch_t {
  std::vector<uint8_t> valid;  // prime-cube validity table, grow-only
  std::vector<bve_clause_t> primes;
  std::vector<bve_mask_t> cover;  // patterns matched by each prime
  bve_mask_t uncovered;
};

// Derive a prime-implicate CNF from the boundary feasibility table by covering the infeasible
// patterns with a max-gain greedy over every prime forbidden cube; return -1 on cap overflow.
// `ops_out` accumulates a deterministic unscaled estimate of the cover build and the greedy scan,
// both of which scale with a prime count the caller cannot know in advance.
template <typename i_t>
i_t bve_greedy_prime_cover(const uint8_t* feas,
                           i_t nb,
                           bve_clause_t* out,
                           i_t cap,
                           bve_cover_scratch_t& scratch,
                           int64_t* ops_out = nullptr);

// Verify that the emitted clauses reproduce the boundary feasibility table exactly.
template <typename i_t, typename f_t>
bool bve_sanity_check(const uint8_t* feas, i_t nb, const bve_clause_t* clauses, i_t n_clauses);

// Exact existential projection of one block onto its boundary, filled by the projection backend.
// Both tables are sized to the block's own 2^nb rather than BVE_MAX_PATTERNS, so a narrow block
// does not carry the cost of raising BVE_MAX_BOUNDARY.
struct bve_projection_t {
  std::vector<uint8_t> feasible;  // [2^nb] 1 iff the boundary pattern admits some interior
  std::vector<uint32_t> witness;  // [2^nb] smallest feasible interior, 0 where infeasible
};

// Staged candidate. Vector fields use sorted current-problem ids; `blk` uses local ids.
template <typename i_t, typename f_t>
struct bve_candidate_t {
  std::vector<i_t> interior;    // sorted global column ids (to be eliminated)
  std::vector<i_t> boundary;    // sorted global column ids (kept)
  std::vector<i_t> rows;        // sorted global row ids spanned by the block
  bve_block_t<f_t> blk;         // gathered block, local ids, for the projection
  bve_projection_t projection;  // sized and zeroed by stage(), filled by the projection backend
};

// Project shape-binned candidate batches on the GPU and return a deterministic work estimate.
template <typename i_t, typename f_t>
double bve_project_batch_gpu(const raft::handle_t& handle,
                             std::vector<bve_candidate_t<i_t, f_t>>& cands,
                             f_t tol);

// Build symmetric current-problem implication adjacency from the original-id keyed probing cache,
// optionally unioned with forcings harvested from earlier block projections (also original-id).
template <typename i_t, typename f_t>
std::vector<std::vector<i_t>> bve_build_impl_adj(
  const probing_cache_t<i_t, f_t>& cache,
  const std::vector<i_t>& reverse_original_ids,
  i_t n_vars,
  const probe_findings_t<i_t>* prior_original_id_findings = nullptr);

// Run block BVE using caller-provided implication adjacency and deadline. Returns true iff at least
// one validated reduction was installed; `work_units` receives a deterministic unscaled estimate.
// `out_findings`, when given, is appended with the implications read off every projected block
// (original-id frame) -- including blocks that were not eliminated.
template <typename i_t, typename f_t>
bool block_bve_presolve(problem_t<i_t, f_t>& problem,
                        const std::vector<std::vector<i_t>>& impl_adj,
                        timer_t& timer,
                        double& work_units,
                        probe_findings_t<i_t>* out_findings = nullptr,
                        i_t boundary_cap                    = BVE_MAX_BOUNDARY,
                        i_t scope_cap                       = BVE_MAX_SCOPE,
                        i_t clause_growth_margin            = 0);

}  // namespace cuopt::mathematical_optimization::mip
