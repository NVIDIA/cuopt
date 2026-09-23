/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../state.hpp"

namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
void perturb(fj_cpu_climber_t<i_t, f_t>&);
}  // namespace cuopt::mathematical_optimization::mip
