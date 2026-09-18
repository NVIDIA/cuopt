/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "state.hpp"

namespace cuopt::mathematical_optimization::mip {

inline constexpr bool fj_audit_every_iteration = false;

template <typename i_t, typename f_t>
void audit_assignment_bounds(fj_cpu_climber_t<i_t, f_t>&, const char*);
template <typename i_t, typename f_t>
f_t fresh_row_slack(fj_cpu_climber_t<i_t, f_t>&, i_t, const f_t*);
template <typename i_t, typename f_t>
void audit_objective_update(fj_cpu_climber_t<i_t, f_t>&, i_t, f_t, f_t, f_t, f_t);
template <typename i_t, typename f_t>
void audit_row_updates(fj_cpu_climber_t<i_t, f_t>&, i_t, f_t, f_t, i_t, i_t);
template <typename i_t, typename f_t>
void audit_incremental_state(fj_cpu_climber_t<i_t, f_t>&, const char*);
template <typename i_t, typename f_t>
bool check_variable_feasibility(fj_cpu_climber_t<i_t, f_t>&, bool check_integer = true);
template <typename i_t, typename f_t>
void sanity_checks(fj_cpu_climber_t<i_t, f_t>&);
}  // namespace cuopt::mathematical_optimization::mip
