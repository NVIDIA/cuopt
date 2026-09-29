/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../state.hpp"
namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
void cap_integer_domains(fj_cpu_climber_t<i_t, f_t>&, i_t);
template <typename i_t, typename f_t>
void clamp_start_magnitude(fj_cpu_climber_t<i_t, f_t>&, i_t);
template <typename i_t, typename f_t>
void apply_bound_propagation(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void apply_lock_weighted_start(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void apply_ambiguous_lock_start(fj_cpu_climber_t<i_t, f_t>&);
}  // namespace cuopt::mathematical_optimization::mip
