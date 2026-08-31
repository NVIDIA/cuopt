/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>

#include <vector>

namespace cuopt::mathematical_optimization {

/**
 * @brief Whether a feature combination the remote server cannot honour must be dropped.
 *
 * Some client-side requests cannot be forwarded to cuopt_grpc_server. Rather than failing
 * the solve, solve_mip_remote() drops the unsupported part and warns. This predicate is the
 * decision, kept separate from the RPC plumbing so it is unit-testable without a live
 * connection.
 *
 * Currently one rule: MIP get/set callbacks are not supported for semi-continuous models.
 * Additional rules belong here as further arguments rather than as new call-site branches.
 *
 * @param var_types     Problem variable types (host).
 * @param has_callbacks Whether the caller registered MIP incumbent callbacks.
 * @return true when the callbacks must be dropped before submitting.
 */
bool should_disable_unsupported(const std::vector<var_t>& var_types, bool has_callbacks);

}  // namespace cuopt::mathematical_optimization
