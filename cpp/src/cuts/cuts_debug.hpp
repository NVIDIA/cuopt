/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdio>

#define DEBUG_CLIQUE_CUTS    0
#define DEBUG_ZERO_HALF_CUTS 0
#define CHECK_WORKSPACE      0

// Shared crash-tolerant debug logger: writes a prefixed line to stderr and
// flushes immediately so the last line is visible even if the process
// aborts/terminates right after. Each channel below enables it through its own
// DEBUG_* flag and supplies its own prefix; when the flag is 0 the call expands
// to a no-op that still consumes its arguments.
#define CUTS_DEBUG_LOG(prefix, ...)    \
  do {                                 \
    std::fprintf(stderr, prefix " ");  \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fprintf(stderr, "\n");        \
    std::fflush(stderr);               \
  } while (0)
#define CUTS_DEBUG_NOOP(...) \
  do {                       \
  } while (0)

#if DEBUG_CLIQUE_CUTS
#define CLIQUE_CUTS_DEBUG(...) CUTS_DEBUG_LOG("[DEBUG_CLIQUE_CUTS]", __VA_ARGS__)
#else
#define CLIQUE_CUTS_DEBUG(...) CUTS_DEBUG_NOOP(__VA_ARGS__)
#endif

#if DEBUG_ZERO_HALF_CUTS
#define ZERO_HALF_DEBUG(...) CUTS_DEBUG_LOG("[zero_half]", __VA_ARGS__)
#else
#define ZERO_HALF_DEBUG(...) CUTS_DEBUG_NOOP(__VA_ARGS__)
#endif
