// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuopt/mathematical_optimization/remote_solve_registry.hpp>

#include <dlfcn.h>

namespace cuopt::mathematical_optimization {

std::atomic<solve_lp_remote_fn_t> g_solve_lp_remote_fn{nullptr};
std::atomic<solve_mip_remote_fn_t> g_solve_mip_remote_fn{nullptr};

void register_remote_solvers(solve_lp_remote_fn_t lp_fn, solve_mip_remote_fn_t mip_fn)
{
  g_solve_lp_remote_fn.store(lp_fn, std::memory_order_release);
  g_solve_mip_remote_fn.store(mip_fn, std::memory_order_release);
}

void ensure_remote_solvers_loaded()
{
  if (g_solve_lp_remote_fn.load(std::memory_order_acquire) != nullptr) { return; }
  // The constructor in libcuopt_grpc.so calls register_remote_solvers(). dlopen is
  // itself thread-safe and refcounted, so a concurrent second call is harmless.
  dlopen("libcuopt_grpc.so", RTLD_NOW | RTLD_GLOBAL);
}

}  // namespace cuopt::mathematical_optimization
