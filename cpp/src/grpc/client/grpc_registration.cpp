// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Registers the gRPC-based remote solve implementations with the registry at
// dynamic-link time, before any user code runs.
//
// This builds into libcuopt_mathopt.so, which owns the registry. It cannot build into
// libcuopt_client.so, where solve_lp_remote and solve_mip_remote live: that would leave
// the client library with an undefined register_remote_solvers and cost it the standalone
// property that lets it ship without CUDA.

#include <cuopt/mathematical_optimization/remote_solve_registry.hpp>
#include <cuopt/mathematical_optimization/solve_remote.hpp>

namespace {
__attribute__((constructor)) void register_grpc_remote_solvers()
{
  cuopt::mathematical_optimization::register_remote_solvers(
    &cuopt::mathematical_optimization::solve_lp_remote<int, double>,
    &cuopt::mathematical_optimization::solve_mip_remote<int, double>);
}
}  // namespace
