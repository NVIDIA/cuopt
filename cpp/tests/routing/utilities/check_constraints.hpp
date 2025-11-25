/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/solver.hpp>

#include <vector>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t>
void check_route(data_model_view_t<i_t, f_t> const& data_model,
                 host_assignment_t<i_t> const& h_routing_solution);

template <typename i_t, typename f_t>
void check_route(data_model_view_t<i_t, f_t> const& data_model,
                 assignment_t<i_t> const& routing_solution);

}  // namespace test
}  // namespace routing
}  // namespace cuopt
