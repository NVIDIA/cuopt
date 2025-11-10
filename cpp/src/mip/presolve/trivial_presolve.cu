/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip/mip_constants.hpp>
#include <mip/presolve/trivial_presolve.cuh>

namespace cuopt::linear_programming::detail {

#if MIP_INSTANTIATE_FLOAT
template void trivial_presolve(problem_t<int, float>& problem);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void trivial_presolve(problem_t<int, double>& problem);
#endif

}  // namespace cuopt::linear_programming::detail
