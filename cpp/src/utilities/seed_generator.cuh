/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <raft/random/rng_device.cuh>
#include <utilities/cuda_helpers.cuh>

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
 * The counter that hands out seeds is thread-local and is rebased whenever the owning
 * solver's base seed changes. Each thread therefore walks its own deterministic sequence
 * from that base, so the order in which concurrent workers happen to ask for seeds does
 * not change which seed any of them receives. A shared counter would hand out values in a
 * nondeterministic order and break reproducibility across synchronisation points.
 *
 * Two solvers configured with the *same* base seed and used from one thread continue a
 * single sequence rather than restarting, since the rebase is triggered by a change of
 * base.
 */
class seed_generator_t {
  int64_t base_seed_{0};

  struct thread_state_t {
    int64_t counter{0};
    int64_t last_base{0};
    bool initialized{false};
  };

  // Shared by every generator on this thread; the base check rebases when the caller
  // switches to a solver seeded differently.
  static thread_state_t& local_state()
  {
    thread_local thread_state_t state;
    return state;
  }

 public:
  seed_generator_t() = default;
  explicit seed_generator_t(int64_t initial) : base_seed_(initial) {}

  template <typename... args>
  void set_seed(args... seeds)
  {
#ifdef BENCHMARK
    base_seed_ = static_cast<int64_t>(std::random_device{}());
#else
    base_seed_ = detail::fold_seed(seeds...);
#endif
  }

  int64_t get_seed() const
  {
    auto& state = local_state();
    if (!state.initialized || state.last_base != base_seed_) {
      state.counter     = base_seed_;
      state.last_base   = base_seed_;
      state.initialized = true;
    }
    return state.counter++;
  }
};

}  // namespace cuopt
