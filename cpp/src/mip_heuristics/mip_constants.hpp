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
#define CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT               8
#define CUOPT_MIP_EARLY_GPUFJ_REQUIRED_THREAD_COUNT      3
#define CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT      2
#define CUOPT_MIP_EARLY_STRUCTURAL_REQUIRED_THREAD_COUNT 2
#define CUOPT_MIP_ROOT_STRUCTURAL_REQUIRED_THREAD_COUNT  3
#define CUOPT_MIP_BATCH_PDLP_REQUIRED_THREAD_COUNT       3
#define CUOPT_MIP_CLIQUE_CUTS_REQUIRED_THREAD_COUNT      3

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

enum class rng_id_t : uint64_t {
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

inline uint64_t get_base_seed(int64_t requested_seed)
{
  if (requested_seed >= 0) { return requested_seed; }
  return std::random_device{}();
}

inline uint64_t derive_seed(uint64_t base_seed, rng_id_t component_id, uint64_t index = 0)
{
  splitmix64_t seed_gen(base_seed + static_cast<uint64_t>(component_id), index);
  return seed_gen.next_u64();
}

inline uint64_t derive_stream(uint64_t base_seed, rng_id_t component_id, uint64_t index = 0)
{
  splitmix64_t seed_gen(base_seed + static_cast<uint64_t>(component_id), index);
  return seed_gen.generate_stream();
}

}  // namespace cuopt::mathematical_optimization::mip
