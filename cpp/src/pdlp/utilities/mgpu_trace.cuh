/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

// Lightweight env-gated tracing for multi-GPU PDLP diagnosis.
//
// Enable by setting CUOPT_MGPU_TRACE=1 in the environment.
// All prints go to stderr (line-buffered + explicit flush) so they survive
// a CUDA hang and interleave with cuOpt's normal output.
//
// Usage:
//   MGPU_TRACE("entering compute_At_y");
//   MGPU_TRACE_FMT("shard %d nnz=%lld", r, (long long)nnz);
//
// The guard reads the env var once on first use (thread-safe via static
// initialization) and the cost when disabled is a single load + branch.

#include <cstdio>
#include <cstdlib>

namespace cuopt::linear_programming::detail {

inline bool mgpu_trace_enabled()
{
  static const bool enabled = []() {
    const char* v = std::getenv("CUOPT_MGPU_TRACE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

}  // namespace cuopt::linear_programming::detail

#define MGPU_TRACE(msg)                                                     \
  do {                                                                      \
    if (::cuopt::linear_programming::detail::mgpu_trace_enabled()) {        \
      std::fprintf(stderr, "[mgpu %s:%d] %s\n", __func__, __LINE__, (msg)); \
      std::fflush(stderr);                                                  \
    }                                                                       \
  } while (0)

#define MGPU_TRACE_FMT(fmt, ...)                                                       \
  do {                                                                                 \
    if (::cuopt::linear_programming::detail::mgpu_trace_enabled()) {                   \
      std::fprintf(stderr, "[mgpu %s:%d] " fmt "\n", __func__, __LINE__, __VA_ARGS__); \
      std::fflush(stderr);                                                             \
    }                                                                                  \
  } while (0)
