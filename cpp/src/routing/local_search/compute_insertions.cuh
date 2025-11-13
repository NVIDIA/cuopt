/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../solution/solution.cuh"
#include "move_candidates/move_candidates.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
void find_insertions(solution_t<i_t, f_t, REQUEST>& sol,
                     move_candidates_t<i_t, f_t>& move_candidates,
                     search_type_t search_type = search_type_t::IMPROVE);

template <typename i_t, typename f_t, request_t REQUEST>
void find_unserviced_insertions(solution_t<i_t, f_t, REQUEST>& sol,
                                move_candidates_t<i_t, f_t>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
