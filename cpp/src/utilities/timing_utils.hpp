/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CUOPT_HAS_RDTSC 1
#else
#define CUOPT_HAS_RDTSC 0
#endif

#if CUOPT_HAS_RDTSC

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

namespace cuopt {

inline uint64_t rdtsc()
{
  uint32_t lo, hi;
  __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

}  // namespace cuopt

// clang-format off
#define CYCLE_TIMING_PROLOGUE(name)                                                      \
  static constexpr size_t timing_buffer_size_##name = 1024;                              \
  static thread_local std::array<uint64_t, timing_buffer_size_##name> timing_buffer_##name; \
  static thread_local size_t timing_idx_##name = 0;                                      \
  uint64_t t_start_##name = cuopt::rdtsc();

#define CYCLE_TIMING_EPILOGUE(name)                                                      \
  do {                                                                                   \
    uint64_t t_end_##name = cuopt::rdtsc();                                              \
    timing_buffer_##name[timing_idx_##name++] = t_end_##name - t_start_##name;           \
    if (timing_idx_##name == timing_buffer_size_##name) {                                \
      uint64_t sum_##name = 0;                                                           \
      for (size_t i = 0; i < timing_buffer_size_##name; ++i)                             \
        sum_##name += timing_buffer_##name[i];                                           \
      uint64_t avg_##name = sum_##name / timing_buffer_size_##name;                      \
      std::array<uint64_t, timing_buffer_size_##name> sorted_##name = timing_buffer_##name; \
      std::nth_element(sorted_##name.begin(),                                            \
                       sorted_##name.begin() + timing_buffer_size_##name / 2,            \
                       sorted_##name.end());                                             \
      uint64_t median_##name = sorted_##name[timing_buffer_size_##name / 2];             \
      printf(#name ": avg=%lu cycles, median=%lu cycles (n=%zu)\n",                      \
             avg_##name, median_##name, timing_buffer_size_##name);                      \
      timing_idx_##name = 0;                                                             \
    }                                                                                    \
  } while (0)
// clang-format on

#else  // !CUOPT_HAS_RDTSC

#define CYCLE_TIMING_PROLOGUE(name) \
  do {                              \
  } while (0)
#define CYCLE_TIMING_EPILOGUE(name) \
  do {                              \
  } while (0)

#endif  // CUOPT_HAS_RDTSC
