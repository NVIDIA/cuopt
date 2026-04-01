/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <csignal>
#include <cstdlib>
#include <cstring>

namespace {

bool coredump_env_enabled()
{
  const char* v = std::getenv("CUOPT_TEST_COREDUMP");
  if (v == nullptr || v[0] == '\0') { return false; }
  return std::strcmp(v, "0") != 0;
}

}  // namespace

// Opt-in crash to validate CI core dump collection (see ci/cuopt_coredumps.sh).
// Normal CI: skipped. To reproduce locally or in a one-off job:
//   CUOPT_TEST_COREDUMP=1 ulimit -c unlimited ./COREDUMP_SANITY_TEST
TEST(CoredumpSanity, IntentionalSegfaultWhenEnvSet)
{
  if (!coredump_env_enabled()) {
    GTEST_SKIP() << "Set CUOPT_TEST_COREDUMP=1 to intentionally SIGSEGV for core dump checks.";
  }
  std::raise(SIGSEGV);
}
