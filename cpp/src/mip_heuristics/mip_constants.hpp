/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>

#include <utilities/pcgenerator.hpp>
#include <utilities/splitmix64.hpp>

#include <cstdint>
#include <random>

#define MIP_INSTANTIATE_FLOAT  CUOPT_INSTANTIATE_FLOAT
#define MIP_INSTANTIATE_DOUBLE CUOPT_INSTANTIATE_DOUBLE

#define PDLP_INSTANTIATE_FLOAT 1

/* @brief Minimimum number of threads to enable each part of the MIP Solver */
#define CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT          8
#define CUOPT_MIP_EARLY_GPUFJ_REQUIRED_THREAD_COUNT 3
#define CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT 2
#define CUOPT_MIP_BATCH_PDLP_REQUIRED_THREAD_COUNT  3
#define CUOPT_MIP_CLIQUE_CUTS_REQUIRED_THREAD_COUNT 3

// MIP-only gate: skip the concurrent barrier when fewer threads are available than this
// (1 PDLP + 1 dual simplex + 1 barrier). Stand-alone LP always runs all three.
#define CUOPT_CONCURRENT_LP_BARRIER_REQUIRED_THREAD_COUNT 3

/* @brief Priority classes for the omp tasks. Highest value = higher priority.
 * Note that this only gives a hint to the runtime, such that the high priority
 * is not guarantee to be executed before a low priority one (i.e., do not rely on
 * these values for correctness).
 */
#define CUOPT_CRITICAL_TASK_PRIORITY 1000
#define CUOPT_HIGH_TASK_PRIORITY     100
#define CUOPT_MEDIUM_TASK_PRIORITY   10
#define CUOPT_DEFAULT_TASK_PRIORITY  1

// Default values for work stealing in B&B
#define MIP_DEFAULT_STEAL_CHANCE       0.05
#define MIP_DEFAULT_NODES_PER_STEAL    10
#define MIP_DEFAULT_MAX_STEAL_ATTEMPTS 3

namespace cuopt::mathematical_optimization::mip {

// Fixed logical identity used to seed each singleton heuristics component's own RNG stream from
// mip_solver_context_t::base_seed (see cpp/src/mip_heuristics/solver_context.cuh). These are NOT
// runtime thread/task ids -- OMP tasks can migrate between OS threads, so identity must be a
// compile-time-fixed constant to keep seeding reproducible across runs. Components that spawn a
// variable number of parallel workers (e.g. local-search CPU-FJ climbers) additionally offset by
// their own fixed slot index on top of the relevant id below.
enum class mip_rng_component_id_t : uint64_t {
  diversity_manager = 10000,
  population,
  local_search,
  feasibility_pump,
  constraint_prop,
  lb_constraint_prop,
  recombiner_bound_prop,
  recombiner_fp,
  recombiner_line_segment,
  recombiner_default,
  recombiner_sub_mip,
  local_search_cpu_fj,
  early_cpufj,
  early_gpufj,
  line_segment_search,
};

// Resolves the solve-wide base seed: the user's requested seed if non-negative, otherwise a
// fresh random one. Called wherever a base seed is needed before/independently of
// mip_solver_context_t (which resolves it the same way for its own base_seed field) -- when
// requested_seed >= 0 both resolve to the identical value, which is what deterministic mode
// requires; when requested_seed < 0 each call draws independently, which is fine since no
// reproducibility is promised in that case.
inline uint64_t mip_resolve_base_seed(int64_t requested_seed)
{
  if (requested_seed >= 0) { return requested_seed; }
  return std::random_device{}();
}

// Derives a well-mixed, reproducible 64-bit seed from the solve's base seed plus a fixed logical
// identity, for owners that need a raw seed value (e.g. to hand to std::mt19937 or a multi-armed
// bandit) rather than owning a PCG-family generator built from (seed, stream) -- see
// mip_derive_stream for the matching stream value. `index` further distinguishes multiple
// independent draws made by the same component (e.g. one per parallel worker slot).
inline uint64_t mip_derive_seed(uint64_t base_seed,
                                mip_rng_component_id_t component_id,
                                uint64_t index = 0)
{
  splitmix64_t seed_gen(base_seed + static_cast<uint64_t>(component_id), index);
  return seed_gen.next_u64();
}

// Derives a well-mixed, reproducible 64-bit stream/subsequence value from the solve's base seed
// plus a fixed logical identity, for owners that construct their own PCG-family generator
// (raft::PCGenerator, cuopt::pcgenerator_t) and need both a seed (mip_derive_seed) and an
// independent stream. `index` further distinguishes multiple independent draws made by the same
// component (e.g. one per parallel worker slot); pass the same `index` to mip_derive_seed and
// mip_derive_stream to get the matching seed/stream pair for one generator instance.
inline uint64_t mip_derive_stream(uint64_t base_seed,
                                  mip_rng_component_id_t component_id,
                                  uint64_t index = 0)
{
  splitmix64_t seed_gen(base_seed + static_cast<uint64_t>(component_id), index);
  return seed_gen.generate_stream();
}

}  // namespace cuopt::mathematical_optimization::mip
