/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <atomic>

namespace cuopt::linear_programming::dual_simplex {

/** True if caller requested stop (any non-zero value). Uses acquire for pairing with release stores. */
inline bool concurrent_halt_is_set(std::atomic<int> const* halt)
{
  return halt != nullptr && std::atomic_load_explicit(halt, std::memory_order_acquire) != 0;
}

/** Signal peer solvers to stop. No-op if halt is null. Uses release for pairing with acquire loads. */
inline void concurrent_halt_signal(std::atomic<int>* halt)
{
  if (halt != nullptr) { std::atomic_store_explicit(halt, 1, std::memory_order_release); }
}

/** Clear halt after concurrent threads have joined; no peers are reading the flag. */
inline void concurrent_halt_reset(std::atomic<int>* halt)
{
  if (halt != nullptr) { std::atomic_store_explicit(halt, 0, std::memory_order_relaxed); }
}

}  // namespace cuopt::linear_programming::dual_simplex
