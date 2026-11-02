/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/audit.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::mip {
// Population seeds belong to the solver model, before CPUFJ caps domains or strengthens types.
template <typename i_t, typename f_t, typename model_validator_t>
bool clamp_and_validate_cpufj_lns_seed(const fj_cpu_climber_t<i_t, f_t>& climber,
                                       std::vector<f_t>& assignment,
                                       const model_validator_t& model_feasible)
{
  if (!model_feasible(assignment)) return false;
  bool changed = false;
  return clamp_cpufj_lns_seed_to_domain(*climber.problem,
                                        climber.h_var_bounds.underlying(),
                                        climber.problem->h_var_types,
                                        assignment,
                                        &changed) &&
         (!changed || model_feasible(assignment));
}
}  // namespace cuopt::mathematical_optimization::mip
