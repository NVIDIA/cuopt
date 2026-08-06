// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuopt/mathematical_optimization/remote_solve_registry.hpp>
#include <utilities/logger.hpp>

#include <dlfcn.h>

namespace cuopt::mathematical_optimization {

std::atomic<solve_lp_remote_fn_t> g_solve_lp_remote_fn{nullptr};
std::atomic<solve_mip_remote_fn_t> g_solve_mip_remote_fn{nullptr};
std::atomic<bool> g_remote_solvers_ready{false};

void register_remote_solvers(solve_lp_remote_fn_t lp_fn, solve_mip_remote_fn_t mip_fn)
{
  g_solve_lp_remote_fn.store(lp_fn, std::memory_order_relaxed);
  g_solve_mip_remote_fn.store(mip_fn, std::memory_order_relaxed);
  // Published last with release ordering: a reader that observes the ready flag is
  // guaranteed to observe both callbacks. Using a separate flag rather than one of the
  // slots keeps the readiness condition independent of how many callbacks there are.
  g_remote_solvers_ready.store(true, std::memory_order_release);
}

void ensure_remote_solvers_loaded()
{
  if (g_remote_solvers_ready.load(std::memory_order_acquire)) { return; }
  // The constructor in libcuopt_grpc.so calls register_remote_solvers(). dlopen is
  // itself thread-safe and refcounted, so a concurrent second call is harmless.
  if (dlopen("libcuopt_grpc.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) {
    const char* err = dlerror();
    CUOPT_LOG_DEBUG("Could not load libcuopt_grpc.so: %s", err != nullptr ? err : "unknown error");
  }
}

}  // namespace cuopt::mathematical_optimization
