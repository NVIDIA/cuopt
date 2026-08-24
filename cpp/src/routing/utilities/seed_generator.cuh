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
namespace routing {

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
 * @brief Routing's source of deterministic seeds, owned by the problem that uses it.
 *
 * `problem_t` holds one of these, seeded from the user's `solver_settings_t::set_seed` or,
 * when none was given, from the problem's own dimensions. Routing previously drew from a
 * process-wide counter shared with the MIP heuristics, so whichever solver constructed its
 * problem last overwrote the other's seed.
 *
 * The counter is `mutable` and atomic so that `get_seed()` can be `const`: `solution_t`
 * reaches its problem through a `const` pointer, and drawing a seed does not change the
 * problem's logical state. Concurrent callers are handed distinct values, but the order in
 * which they receive them is not fixed, so reproducibility still requires a deterministic
 * call order.
 */
class seed_generator_t {
  mutable std::atomic<int64_t> counter_{0};

 public:
  seed_generator_t() = default;
  explicit seed_generator_t(int64_t initial) : counter_(initial) {}

  // std::atomic is neither copyable nor movable, which would delete problem_t's defaulted
  // move constructor. Transfer the value instead so the owning problem stays movable.
  seed_generator_t(seed_generator_t&& other) noexcept
    : counter_(other.counter_.load(std::memory_order_relaxed))
  {
  }

  seed_generator_t& operator=(seed_generator_t&& other) noexcept
  {
    counter_.store(other.counter_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

  template <typename... args>
  void set_seed(args... seeds)
  {
#ifdef BENCHMARK
    counter_.store(static_cast<int64_t>(std::random_device{}()), std::memory_order_relaxed);
#else
    counter_.store(detail::fold_seed(seeds...), std::memory_order_relaxed);
#endif
  }

  int64_t get_seed() const { return counter_.fetch_add(1, std::memory_order_relaxed); }
};

}  // namespace routing
}  // namespace cuopt
