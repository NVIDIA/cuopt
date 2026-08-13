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

/**
 * @brief Source of deterministic seeds for a single cuOpt solver library.
 *
 * The counter is defined inline, so each library that links this header keeps its own.
 * That matches how the seed is actually supplied: routing derives it from the problem
 * geometry while mathematical optimization takes it from the user's solver settings.
 * Those are independent inputs, and a single shared counter meant whichever solver ran
 * last silently overwrote the other's seed.
 *
 * @note Thread-safe. get_seed() hands out distinct values to concurrent callers, but the
 *       order in which they are handed out is not deterministic; reproducibility across
 *       runs therefore still requires a deterministic call order.
 */
class seed_generator {
  static inline std::atomic<int64_t> seed_{0};

 public:
  template <typename seed_t>
  static void set_seed(seed_t seed)
  {
#ifdef BENCHMARK
    seed_.store(std::random_device{}(), std::memory_order_relaxed);
#else
    seed_.store(static_cast<int64_t>(seed), std::memory_order_relaxed);
#endif
  }
  template <typename arg0, typename arg1, typename... args>
  static void set_seed(arg0 seed0, arg1 seed1, args... seeds)
  {
    set_seed(seed1 + ((seed0 + seed1) * (seed0 + seed1 + 1) / 2), seeds...);
  }

  static int64_t get_seed() { return seed_.fetch_add(1, std::memory_order_relaxed); }

 public:
  seed_generator(seed_generator const&) = delete;
  void operator=(seed_generator const&) = delete;
};

}  // namespace cuopt
