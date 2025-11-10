/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../solution/solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
void initialize_incompatible(problem_t<i_t, f_t>& problem,
                             solution_t<i_t, f_t, REQUEST>* sol_ptr = nullptr);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
