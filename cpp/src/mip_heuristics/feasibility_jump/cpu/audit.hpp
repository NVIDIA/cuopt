/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "state.hpp"

namespace cuopt::mathematical_optimization::mip {

inline constexpr bool fj_audit_every_iteration = false;

// Recheck the raw model using explicit bounds and types, which may precede CPUFJ's domain caps
// and implied-integrality strengthening.
template <typename i_t, typename f_t>
bool verify_cpufj_lns_feasible(const fj_cpu_problem_t<i_t, f_t>& problem,
                               const std::vector<typename type_2<f_t>::type>& bounds,
                               const std::vector<var_t>& types,
                               const std::vector<f_t>& assignment);
template <typename i_t, typename f_t>
bool verify_cpufj_lns_feasible(const fj_cpu_problem_t<i_t, f_t>& problem,
                               const std::vector<typename type_2<f_t>::type>& bounds,
                               const std::vector<f_t>& assignment);
template <typename i_t, typename f_t>
bool clamp_cpufj_lns_seed_to_domain(const fj_cpu_problem_t<i_t, f_t>& problem,
                                    const std::vector<typename type_2<f_t>::type>& bounds,
                                    const std::vector<var_t>& types,
                                    std::vector<f_t>& assignment,
                                    bool* changed = nullptr);
template <typename i_t, typename f_t>
bool clamp_and_validate_cpufj_lns_seed(const fj_cpu_problem_t<i_t, f_t>& problem,
                                       const std::vector<typename type_2<f_t>::type>& bounds,
                                       const std::vector<var_t>& types,
                                       std::vector<f_t>& assignment);
template <typename i_t, typename f_t>
bool clamp_and_validate_cpufj_lns_seed(const fj_cpu_problem_t<i_t, f_t>& problem,
                                       const std::vector<typename type_2<f_t>::type>& bounds,
                                       std::vector<f_t>& assignment);

template <typename i_t, typename f_t>
void audit_assignment_bounds(fj_cpu_climber_t<i_t, f_t>&, const char*);
template <typename i_t, typename f_t>
f_t fresh_row_slack(fj_cpu_climber_t<i_t, f_t>&, i_t, const f_t*, f_t&);
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
