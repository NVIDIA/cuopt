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

// Folds several values into one seed using the Cantor pairing function.
//
// The arithmetic is done in uint64_t: routing folds `int` problem dimensions, and the
// product overflows a 32-bit int once two equal dimensions reach 181. Signed overflow is
// undefined behaviour, so widen first and let the unsigned type wrap deterministically.
template <typename seed_t>
inline int64_t fold_seed(seed_t seed)
{
  return static_cast<int64_t>(static_cast<uint64_t>(seed));
}

template <typename arg0, typename arg1, typename... args>
inline int64_t fold_seed(arg0 seed0, arg1 seed1, args... seeds)
{
  const uint64_t a   = static_cast<uint64_t>(seed0);
  const uint64_t b   = static_cast<uint64_t>(seed1);
  const uint64_t sum = a + b;
  return fold_seed(b + sum * (sum + 1) / 2, seeds...);
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

  // std::atomic is not copyable, so the counter value is transferred explicitly. Without
  // these, declaring the copy operations deleted would also suppress the implicit move
  // assignment of any class holding a generator (problem_t is move-assigned).
  seed_generator_t(seed_generator_t const& other)
    : seed_(other.seed_.load(std::memory_order_relaxed))
  {
  }
  seed_generator_t& operator=(seed_generator_t const& other)
  {
    seed_.store(other.seed_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

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

}  // namespace cuopt
