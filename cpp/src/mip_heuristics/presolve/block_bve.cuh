/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

// CUDA-only dependencies (types named by the pass/service declarations). Guarded so a host tool
// could include this header for just the block/plan structs and the reducer/clause declarations
// without pulling in CUDA / problem_t / the probing cache.
#ifdef __CUDACC__
#include "probing_cache.cuh"

#include <mip_heuristics/problem/problem.cuh>

#include <raft/core/handle.hpp>
#include <utilities/timer.hpp>
#endif

// Post-Papilo GPU block-BVE presolve pass. This header DECLARES the public surface; all function
// bodies live in block_bve.cu (explicit-instantiation style, matching the rest of the codebase).
//
// WHAT IT IS, in standard vocabulary. A block of zero-objective binary auxiliary variables (the
// block "interior") is eliminated by EXISTENTIAL PROJECTION onto the remaining "boundary" columns
// (∃interior. block_rows), and the projected feasible region is re-encoded over the boundary as the
// prime-implicate CNF of that projection (a set of set-covering no-goods). This is a PRIMAL,
// feasibility- and optimality-preserving reformulation in the sense of Achterberg et al., "Presolve
// Reductions in MIP" (INFORMS JoC 2020): a boundary assignment is feasible in the reduced model iff
// some interior completes it in the original, and eliminating only zero-objective aux leaves the
// objective untouched. It is MORE GENERAL than affine substitution/aggregation (the interior is
// removed via a general Boolean function, not an affine equality) and is adjacent to
// gate/definitional variable elimination in SAT (Ostrowski et al. 2002, "Recovering Structural
// Knowledge from CNF"). The growth gate |clauses| <= |rows| + margin is the bounded-elimination
// criterion of Een & Biere's SatELite (SAT'05) — hence "BVE" — though the mechanism here is block
// enumeration/projection, not the pairwise resolution of classic SAT BVE. The reconstruction table
// (`witness`, below) plays the role of the witness substitution w in VeriPB's redundance-based
// strengthening rule: for a certified pipeline it maps each eliminated interior column to its value
// given the boundary. See Hoen, Oertel, Gleixner & Nordstrom, "Certifying MIP-Based Presolve
// Reductions for 0-1 ILPs" (CPAIOR 2024), which certifies exactly this class of PaPILO reductions
// with machine-checkable pseudo-Boolean proofs.
//
// TRUST MODEL (read the honest caveat in bve_sanity_check). We do NOT emit a machine-checkable
// certificate. commit_projected runs an inline SANITY CHECK (certifying-algorithm / result-checking
// style): the emitted clauses are re-evaluated by an independent evaluator and must reproduce the
// projected feasibility array, else the block is kept verbatim. This guards against
// detector/encoder bugs; it is not a proof a third party can verify. The certified variant would be
// VeriPB proof logging (Hoen et al. 2024) atop PaPILO, which cuOpt already runs.
//
// Layers:
//   1. Clause core (bve_block_t / bve_prime_implicates / bve_sanity_check). The projection
//   ENUMERATION runs
//      on the GPU (layer 3); on the host, commit_projected derives the prime-implicate CNF from the
//      GPU-computed feas and re-checks it with the inline sanity check. Ported bit-for-bit from the
//      validated host reference cpufj_sc22/bve_blocks.cpp.
//   2. Host detector working model (bve_reducer_t): `stage` gathers a candidate block from the
//   model,
//      the GPU projection backend fills its feas/witness, `commit_projected` sanity checks +
//      rewrites the model. The round-based driver over the probing-cache implication closure
//      (bve_detect_closure_batched) is private to block_bve.cu — only the pass uses it.
//   3. GPU enumeration kernel (bve_enumerate_kernel, defined in block_bve.cu) + the pass/service
//      declarations. The kernel projects a whole batch of shape-identical candidate blocks in one
//      launch (CTA per assignment, warp per row); the pass driver detects, projects on the GPU,
//      installs the reduced model, and records the reconstruction data replayed by
//      presolve_data_t::post_process_assignment.
//
// The host enumeration projection and the reference detectors (bve_project / bve_project_and_check
// / bve_detect_closure / bve_detect_minfill) are NOT part of this header — they are the trusted
// differential oracle / coverage reference and live in tests/mip/block_bve_test.cu.
//
// fp64 + 1e-6 tol throughout; integrality is a value property, never a storage type; fractional
// coefficients are handled natively (no scaling). All column/row ids are in the CURRENT problem_t
// space at detection time (post-Papilo, before this pass's trivial_presolve).

namespace cuopt::mathematical_optimization::mip {

// ===========================================================================================
//  1. Clause core: projection -> prime-implicate CNF -> inline sanity check
// ===========================================================================================

// Bounded caps for a single block. Mirror the host reference's gates:
//   nb  <= BVE_MAX_BOUNDARY  (Bcap)
//   na + nb <= BVE_MAX_SCOPE (enumcap)
//   |clauses| <= |rows| + margin  (bounded-elimination growth gate a la SatELite, Een & Biere
//   SAT'05;
//                                  margin 0 by default -- only commit if the CNF is no larger than
//                                  the block it replaces)
static constexpr int BVE_MAX_BOUNDARY = 8;   // nb  <= 8  => 2^nb <= 256 feasibility patterns
static constexpr int BVE_MAX_SCOPE    = 16;  // na + nb <= 16
static constexpr int BVE_MAX_INTERIOR = BVE_MAX_SCOPE - 1;
static constexpr int BVE_MAX_ROWS     = 64;  // |G| (rows spanned by the block); clauses <= |G|
static constexpr int BVE_MAX_ROW_LEN  = 24;  // nnz within one block row (interior+boundary entries)
static constexpr int BVE_MAX_NNZ      = BVE_MAX_ROWS * BVE_MAX_ROW_LEN;
static constexpr int BVE_MAX_CLAUSES  = 64;                     // <= |rows| for any committed block
static constexpr int BVE_MAX_PATTERNS = 1 << BVE_MAX_BOUNDARY;  // 256

// One block handed to the projection core. All variable references are LOCAL to the block: local id
// v in [0, na) is an interior (to-be-eliminated) variable; v in [na, na+nb) is boundary variable
// (v-na). Rows are packed CSR-style: row r spans [row_off[r], row_off[r+1)) in row_var / row_coef.
// A missing bound is encoded as +/- infinity (the kernel handles it directly in the row test).
template <typename f_t>
struct bve_block_t {
  int na;      // number of interior variables
  int nb;      // number of boundary variables (all must be binary; caller guarantees)
  int n_rows;  // |G|
  int row_off[BVE_MAX_ROWS + 1];
  int row_var[BVE_MAX_NNZ];  // local var id in [0, na+nb)
  f_t row_coef[BVE_MAX_NNZ];
  f_t row_lo[BVE_MAX_ROWS];  // -inf if no lower bound
  f_t row_up[BVE_MAX_ROWS];  // +inf if no upper bound
};

// One prime-implicate clause over the boundary. Bit j (0-based over the block's boundary variables)
// of `lit_mask` is set iff boundary var j is a literal of the clause; `bit_mask` bit j is the
// FORBIDDEN value of that literal. The clause forbids exactly the boundary patterns that match
// `bit_mask` on every `lit_mask` position (i.e. it asserts OR_j (x_j != bit_mask_j)).
//
// Row encoding used by the transform (kept here so producer and consumer agree): for each literal j
// coefficient is (bit==0 ? +1 : -1) on boundary var j, and the row is `sum >= 1 - popcount(bit_mask
// & lit_mask)` (a <= row is never needed — these are pure set-covering no-goods).
struct bve_clause_t {
  uint32_t lit_mask;
  uint32_t bit_mask;
};

enum class bve_status_t : int {
  kReduced    = 0,  // sanity check passed; `clauses` is a sound replacement for the block rows
  kSkipCaps   = 1,  // block violates a bound cap (defensive; detector should pre-filter)
  kSkipGrowth = 2,  // |clauses| > |rows| + margin (would grow the row count)
  kSkipCheckFailed =
    3  // clauses did not reproduce feas (sanity check failed) => keep block verbatim
};

// The host enumeration projection (bve_project / bve_project_and_check) is NOT here — in this pass
// projection runs on the GPU (bve_enumerate_kernel); the host versions are the test-only oracle
// (tests/mip/block_bve_test.cu). bve_prime_implicates + bve_sanity_check DO run in production
// (commit_projected derives + sanity checks the clauses from the GPU-computed feas on the host).

// Prime-implicate CNF over the boundary from the feasible-pattern array (feas[m] over 2^nb
// patterns). This IS the projection ∃interior. block_rows expressed in CNF: prime-implicate
// generation by literal dropping (Quine's consensus/expansion). For each infeasible pattern we
// start from the full nb-literal clause and greedily drop literals while the reduced clause still
// forbids only infeasible patterns (a prime implicate), then de-duplicate. Returns clause count, or
// -1 if `cap` would be exceeded. Faithful port of bve_blocks.cpp ~170-213.
int bve_prime_implicates(const uint8_t* feas, int nb, bve_clause_t* out, int cap);

// Inline SANITY CHECK (certifying-algorithm / result-checking style; NOT a machine-checkable
// certificate). An INDEPENDENT boolean evaluator of the emitted clauses must reproduce `feas` on
// every boundary pattern, and every clause literal must live on the boundary. If it does, the CNF
// is provably equivalent to the projection this block computed, so replacing the block rows with
// the CNF is a sound reformulation; if not, the caller keeps the block verbatim. This catches
// detector/encoder bugs but is not a proof a third party can verify — the certified variant would
// emit VeriPB pseudo-Boolean proof steps (redundance-based strengthening with the witness
// substitution + checked deletion of the replaced rows; Hoen et al., CPAIOR 2024). Faithful port of
// bve_blocks.cpp ~219-246.
bool bve_sanity_check(const uint8_t* feas, int nb, const bve_clause_t* clauses, int n_clauses);

// ===========================================================================================
//  2. Host detector (working model + plan types)
// ===========================================================================================

// One committed elimination, in commit order. `witness` is the reconstruction table (the witness
// substitution w of VeriPB's redundance rule): given the boundary `pattern`, `witness[pattern]`
// packs the eliminated interior columns' values. `interior[k]` is the k-th eliminated column and
// bit k of `witness[pattern]` is its reconstructed value; `boundary[j]` is the j-th boundary column
// and bit j of `pattern` is its value. Replayed in REVERSE commit order at postsolve (a boundary
// column may be a later block's interior).
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
  std::vector<i_t> eliminated_cols;                   // interior columns (become empty)
  i_t n_blocks    = 0;
  i_t n_elim_cols = 0;
  i_t final_cols  = 0;  // columns still appearing in an active row (oracle parity / logging)
  i_t final_rows  = 0;  // active rows after commit
};

// A candidate block, gathered from the working model but NOT yet projected or committed. Produced
// by `bve_reducer_t::stage`, projected by a backend (host `bve_project` oracle or the GPU batch
// service), then consumed by `bve_reducer_t::commit_projected`. Decoupling gather from projection
// is what lets many candidates be projected in one batched GPU launch instead of one host call per
// block. `interior`, `boundary`, `rows` are global ids in the current problem_t space (sorted);
// `blk` holds the same block with LOCAL ids for the projection; `feas`/`witness` are filled by the
// projection.
template <typename i_t, typename f_t>
struct bve_candidate_t {
  std::vector<i_t> interior;           // sorted global column ids (to be eliminated)
  std::vector<i_t> boundary;           // sorted global column ids (kept)
  std::vector<i_t> rows;               // sorted global row ids spanned by the block (|G|)
  bve_block_t<f_t> blk;                // gathered block, local ids, for the projection
  uint8_t feas[BVE_MAX_PATTERNS];      // [2^nb]  filled by projection: 1 iff pattern is feasible
  uint32_t witness[BVE_MAX_PATTERNS];  // [2^nb]  filled by projection: smallest feasible interior
};

// Working model: original + appended clause rows, the column->active-rows adjacency, and the
// growing reduction plan. A finder proposes an interior set; `stage` computes its
// rows/boundary/block, `commit_projected` derives + sanity checks the clauses for the projected
// block and — only if the sanity check passes — deactivates the block rows, appends the clause
// rows, retires the interior columns, and records the reduction. Sequential/order-dependent by
// design. Methods in block_bve.cu.
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
  int Bcap, enumcap, margin;
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
                int Bcap_,
                int enumcap_,
                int margin_);

  std::unordered_set<i_t> rows_of(const std::vector<i_t>& interior) const;
  std::vector<i_t> boundary_of(const std::unordered_set<i_t>& G,
                               const std::unordered_set<i_t>& A) const;
  // boundary size of a candidate interior (used by the growth heuristics)
  int boundary_size(const std::vector<i_t>& interior) const;

  // Gather one candidate block from the working model WITHOUT projecting or mutating it (sorts
  // interior/boundary/rows so the local bit-ordering is deterministic, applies the caps, packs the
  // rows into out.blk with local ids). Returns false if any cap is violated; out.feas/out.witness
  // are zeroed and must be filled by a projection backend before commit_projected.
  bool stage(const std::vector<i_t>& interior_in, bve_candidate_t<i_t, f_t>& out);

  // Derive the prime-implicate CNF from an already-projected candidate, apply the growth gate and
  // the inline sanity check (bve_sanity_check), and — only if the sanity check passes — mutate the
  // working model (deactivate block rows, append no-good clause rows over the boundary, retire
  // interior columns, record the reduction). Returns true iff reduced. Callers guarantee a batch's
  // candidates have disjoint scope, so commit order is irrelevant and each block's staged
  // projection is valid at commit.
  bool commit_projected(const bve_candidate_t<i_t, f_t>& cand);

  bve_plan_t<i_t, f_t> finalize();
};

// ===========================================================================================
//  3. GPU pass/service declarations (CUDA only; bodies in block_bve.cu)
// ===========================================================================================
#ifdef __CUDACC__

// GPU batch-projection backend: bin `cands` by identical shape (nb, na, n_rows, row layout), upload
// the per-block coefficients/bounds, launch bve_enumerate_kernel once per shape-bin, and fill each
// candidate's feas/witness from the returned witness table. Replaces the per-block host
// bve_project. Returns a deterministic raw work estimate for the enumerations performed
// (assignments · nnz).
template <typename i_t, typename f_t>
double bve_project_batch_gpu(const raft::handle_t& handle,
                             std::vector<bve_candidate_t<i_t, f_t>>& cands,
                             f_t tol);

// Build the symmetric implication adjacency (in CURRENT problem-space) from the probing cache:
// x ~ y iff probing x moves y's bound (y in probing_cache[x][0/1].var_to_cached_bound_map) or vice
// versa. The cache is keyed in ORIGINAL-id space, so every key and neighbor is translated through
// `reverse_original_ids[original_id] -> current column index` (-1 if the column was removed). This
// is the candidate pool the closure detector grows over.
template <typename i_t, typename f_t>
std::vector<std::vector<i_t>> bve_build_impl_adj(const probing_cache_t<i_t, f_t>& cache,
                                                 const std::vector<i_t>& reverse_original_ids,
                                                 i_t n_vars);

// The pass. `impl_adj` is built by the caller from the probing cache (bve_build_impl_adj).
// `timer` is the caller's deadline clock (typically the solve-wide global_timer).
// `work_units` is set to a deterministic raw estimate of work performed (operation counts;
// not yet scaled into the shared deterministic work-unit clock). Wall-clock `timer` remains the
// stop condition today.
// Returns true iff at least one sanity checked reduction was applied (and the model was rewritten +
// a trivial_presolve compaction run). tol/Bcap/enumcap/margin mirror the host reference.
template <typename i_t, typename f_t>
bool block_bve_presolve(problem_t<i_t, f_t>& problem,
                        const std::vector<std::vector<i_t>>& impl_adj,
                        timer_t& timer,
                        double& work_units,
                        f_t tol     = static_cast<f_t>(1e-6),
                        int Bcap    = BVE_MAX_BOUNDARY,
                        int enumcap = BVE_MAX_SCOPE,
                        int margin  = 0);

#endif  // __CUDACC__

}  // namespace cuopt::mathematical_optimization::mip
