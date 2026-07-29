// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuopt/mathematical_optimization/remote_solve_registry.hpp>

namespace cuopt::mathematical_optimization {

solve_lp_remote_fn_t g_solve_lp_remote_fn   = nullptr;
solve_mip_remote_fn_t g_solve_mip_remote_fn = nullptr;

void register_remote_solvers(solve_lp_remote_fn_t lp_fn, solve_mip_remote_fn_t mip_fn)
{
  g_solve_lp_remote_fn  = lp_fn;
  g_solve_mip_remote_fn = mip_fn;
}

}  // namespace cuopt::mathematical_optimization
