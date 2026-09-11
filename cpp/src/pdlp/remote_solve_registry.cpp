// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuopt/mathematical_optimization/remote_solve_registry.hpp>
#include <utilities/logger.hpp>

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
  // Nothing to load. grpc_registration.cpp builds into this library, so its ELF
  // constructor has already called register_remote_solvers() by the time any code here
  // runs. The function stays because callers read better for it, and because a future
  // arrangement that moves registration back out would need somewhere to hook.
  //
  // It previously dlopen'd libcuopt_grpc.so, which held the constructor as a separate
  // component. That component was 18 KB exporting nothing, so it was folded in here.
}

}  // namespace cuopt::mathematical_optimization
