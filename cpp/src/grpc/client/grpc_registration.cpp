// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Registers the gRPC-based remote solve implementations with libcuopt_mathopt.so
// at dynamic-link time (before any user code runs).  This breaks the circular
// dependency: libcuopt_mathopt.so holds nullable function pointers rather than a
// hard reference to symbols in libcuopt_grpc.so.

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
