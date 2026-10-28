/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <vector>

#include "bounds_presolve.cuh"
#include "probing_cache.cuh"

#include <mip_heuristics/problem/problem.cuh>

#include <raft/core/handle.hpp>
#include <utilities/timer.hpp>

// Eliminates small blocks of zero-objective binary variables. A block is a set of columns to remove
// (the interior, na columns) together with every row they appear in; the other columns of those
// rows are the boundary (nb columns), which stays in the model and must also be binary.
//
// For each of the 2^nb boundary assignments the projection decides whether some interior
// assignment satisfies the block's rows. The ruled-out assignments are everything the block still
// forces on the rest of the model, so emitting them as prime-implicate no-goods over the boundary
// carries that force without the interior. Committing therefore deletes the interior columns and
// every block row, installing the no-goods in their place: interior variables disappear and the row
// count drops whenever the no-goods are fewer than the rows they replace, which the growth gate
// below requires. One feasible interior witness per surviving assignment is stored so postsolve
// can rebuild the deleted columns; since the interior carries no objective coefficients, any
// witness preserves the objective as well as feasibility.
//
// Candidate interiors are grown from the probing implication graph and committed only when the
// projected CNF satisfies the bounded-elimination growth limit of Eén and Biere, "Effective
// Preprocessing in SAT through Variable and Clause Elimination" (SAT 2005). Before commit, the
// emitted clauses are checked against the GPU-computed boundary feasibility table.

namespace cuopt::mathematical_optimization::mip {

// Caps for a single enumerated block.
static constexpr int BVE_MAX_BOUNDARY = 12;  // nb  <= 12  => 2^nb <= 4096 feasibility patterns
static constexpr int BVE_MAX_SCOPE    = 16;  // na + nb <= 16
static constexpr int BVE_MAX_ROWS     = 64;  // rows spanned by the block; #clauses <= #rows
static constexpr int BVE_MAX_ROW_LEN  = 24;  // nnz within one block row (interior+boundary entries)
static constexpr int BVE_MAX_NNZ      = BVE_MAX_ROWS * BVE_MAX_ROW_LEN;
static constexpr int BVE_MAX_CLAUSES  = 64;  // <= |rows| for any committed block
static constexpr int BVE_MAX_PATTERNS = 1 << BVE_MAX_BOUNDARY;

// Packed projection block. Local ids [0, na) are interior and [na, na+nb) are boundary; rows use
// CSR layout and missing bounds are +/- infinity.
template <typename f_t>
struct bve_block_t {
  int na;      // number of interior variables
  int nb;      // number of boundary variables
  int n_rows;  // rows spanned by the block
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
// Untemplated: the CNF is a Boolean computation over the feasibility table, and every dimension it
// touches is capped by the BVE_MAX_* constants above.
int bve_greedy_prime_cover(const uint8_t* feas,
                           int nb,
                           bve_clause_t* out,
                           int cap,
                           bve_cover_scratch_t& scratch,
                           int64_t* ops_out = nullptr);

// Verify that the emitted clauses reproduce the boundary feasibility table exactly.
bool bve_sanity_check(const uint8_t* feas, int nb, const bve_clause_t* clauses, int n_clauses);

// Exact existential projection of one block onto its boundary, filled by the projection backend.
// Both tables are sized to the block's own 2^nb rather than BVE_MAX_PATTERNS, so a narrow block
// does not carry the cost of raising BVE_MAX_BOUNDARY.
struct bve_projection_t {
  std::vector<uint8_t> feasible;  // [2^nb] 1 iff the boundary pattern admits some interior
  std::vector<uint32_t> witness;  // [2^nb] smallest feasible interior, 0 where infeasible
  bool projected{false};          // set by the backend once both tables hold its result
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
                             f_t tol,
                             const timer_t& timer);

// Build symmetric current-problem implication adjacency from the original-id keyed probing cache,
// optionally unioned with forcings harvested from earlier block projections (also original-id).
// Returns an edgeless adjacency when `timer` expires, which leaves the pass with no seeds.
template <typename i_t, typename f_t>
std::vector<std::vector<i_t>> bve_build_impl_adj(
  const probing_cache_t<i_t, f_t>& cache,
  const std::vector<i_t>& reverse_original_ids,
  i_t n_vars,
  const timer_t& timer,
  const probe_findings_t<i_t>* prior_original_id_findings = nullptr);

// True when some row is short enough to appear in a staged block. A block's scope spans whole rows,
// so a model whose every row exceeds BVE_MAX_ROW_LEN has no stageable candidate no matter what the
// implication graph holds, and the pass can be skipped before that graph is built.
template <typename i_t, typename f_t>
bool bve_has_stageable_row(const problem_t<i_t, f_t>& problem);

// cuOpt's block-BVE presolve phase, and the only entry point production code needs: bounded rounds
// of detect and install against the probing implication graph, followed by the projection harvest
// (cache tightening, single-value fixings, bound propagation). Mutates `problem` in place and
// returns false when the phase proved it infeasible. Requires a populated probing cache in
// `bound_presolve`; the caller decides whether the phase runs at all.
template <typename i_t, typename f_t>
bool block_bve_phase(bound_presolve_t<i_t, f_t>& bound_presolve,
                     problem_t<i_t, f_t>& problem,
                     const timer_t& deadline);

// Run block BVE using caller-provided implication adjacency and deadline. Returns true iff at least
// one validated reduction was installed; `work_units` receives a deterministic unscaled estimate.
// `out_findings`, when given, is appended with the implications read off every projected block
// (original-id frame), including blocks that were not eliminated.
template <typename i_t, typename f_t>
bool block_bve_presolve(problem_t<i_t, f_t>& problem,
                        const std::vector<std::vector<i_t>>& impl_adj,
                        timer_t& timer,
                        double& work_units,
                        probe_findings_t<i_t>* out_findings = nullptr,
                        bool* out_infeasible                = nullptr,
                        i_t boundary_cap                    = BVE_MAX_BOUNDARY,
                        i_t scope_cap                       = BVE_MAX_SCOPE,
                        i_t clause_growth_margin            = 0);

}  // namespace cuopt::mathematical_optimization::mip
