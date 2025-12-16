/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../../solution/solution.cuh"
#include "../move_candidates/move_candidates.cuh"

namespace cuopt {
namespace routing {
namespace detail {

constexpr auto const max_n_best_route_pair_moves = 5000;

template <typename i_t, typename f_t, request_t REQUEST>
bool select_and_execute_vrp_move(solution_t<i_t, f_t, REQUEST>& sol,
                                 move_candidates_t<i_t, f_t>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
