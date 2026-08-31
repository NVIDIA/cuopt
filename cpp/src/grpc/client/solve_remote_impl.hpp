/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>

namespace cuopt::mathematical_optimization {

/**
 * @brief Whether a feature combination the remote server cannot honour must be dropped.
 *
 * Some client-side requests cannot be forwarded to cuopt_grpc_server. Rather than failing
 * the solve, solve_mip_remote() drops the unsupported part and warns. This predicate is that
 * decision, kept out of the RPC plumbing so it is unit-testable without a live connection.
 *
 * Takes the problem and settings themselves rather than pre-extracted fields, so new rules
 * can consult anything either object exposes without changing this signature or adding
 * branches at the call site.
 *
 * Currently one rule: MIP get/set callbacks are not supported for semi-continuous models.
 *
 * @param problem  The problem being submitted.
 * @param settings The MIP settings the caller configured.
 * @return true when the unsupported request (today, the callbacks) must be dropped.
 */
template <typename i_t, typename f_t>
bool should_disable_unsupported(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                const mip_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::mathematical_optimization
