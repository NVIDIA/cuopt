/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <raft/random/rng_device.cuh>
#include <utilities/cuda_helpers.cuh>

#include <atomic>
#include <cstdint>
#include <random>

namespace cuopt {

namespace detail {

// Folds several values into one seed. Shared by the instance and the legacy static API.
template <typename seed_t>
inline int64_t fold_seed(seed_t seed)
{
  return static_cast<int64_t>(seed);
}

template <typename arg0, typename arg1, typename... args>
inline int64_t fold_seed(arg0 seed0, arg1 seed1, args... seeds)
{
  return fold_seed(seed1 + ((seed0 + seed1) * (seed0 + seed1 + 1) / 2), seeds...);
}

}  // namespace detail

/**
 * @brief Source of deterministic seeds, owned by the solver that uses it.
 *
 * Each solver holds its own generator, seeded from its own settings, so that two solvers
 * running in the same process cannot overwrite each other's seed.
 *
 * @note Thread-safe. get_seed() hands out distinct values to concurrent callers, but the
 *       order in which they are handed out is not deterministic; reproducibility therefore
 *       still requires a deterministic call order.
 */
class seed_generator_t {
  // Mutable so that a solver reachable only through a const pointer can still draw seeds;
  // drawing a seed does not change the solver's logical state.
  mutable std::atomic<int64_t> seed_{0};

 public:
  seed_generator_t() = default;
  explicit seed_generator_t(int64_t initial) : seed_(initial) {}

  seed_generator_t(seed_generator_t const&) = delete;
  void operator=(seed_generator_t const&)   = delete;

  template <typename... args>
  void set_seed(args... seeds)
  {
#ifdef BENCHMARK
    seed_.store(std::random_device{}(), std::memory_order_relaxed);
#else
    seed_.store(detail::fold_seed(seeds...), std::memory_order_relaxed);
#endif
  }

  int64_t get_seed() const { return seed_.fetch_add(1, std::memory_order_relaxed); }
};

/**
 * @brief Legacy process-wide seed source.
 *
 * @deprecated Being replaced by seed_generator_t owned by each solver. Call sites are
 *             migrating; do not add new uses.
 */
class seed_generator {
  static inline std::atomic<int64_t> seed_{0};

 public:
  template <typename... args>
  static void set_seed(args... seeds)
  {
#ifdef BENCHMARK
    seed_.store(std::random_device{}(), std::memory_order_relaxed);
#else
    seed_.store(detail::fold_seed(seeds...), std::memory_order_relaxed);
#endif
  }

  static int64_t get_seed() { return seed_.fetch_add(1, std::memory_order_relaxed); }

 public:
  seed_generator(seed_generator const&) = delete;
  void operator=(seed_generator const&) = delete;
};

}  // namespace cuopt
