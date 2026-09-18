/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../setup/bounds.hpp"
#include "starts.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void apply_structural_completion_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  apply_lock_weighted_start<i_t, f_t>(fj_cpu);
  apply_exact_k_start<i_t, f_t>(fj_cpu);
  apply_greedy_covering_start<i_t, f_t>(fj_cpu);
  repair_difficult_anchor<i_t, f_t>(fj_cpu);
}

#if MIP_INSTANTIATE_FLOAT
template void apply_structural_completion_start<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void apply_structural_completion_start<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
