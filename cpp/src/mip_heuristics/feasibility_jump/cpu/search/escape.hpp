/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../state.hpp"

namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
void perturb(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void reset_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void infeasible_kick(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void track_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>&);
}  // namespace cuopt::mathematical_optimization::mip
