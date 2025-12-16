/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/fleet_info.hpp>
#include <routing/order_info.hpp>
namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
void populate_demand_container(data_model_view_t<i_t, f_t> const& data_model,
                               fleet_info_t<i_t, f_t>& fleet_info,
                               order_info_t<i_t, f_t>& order_info);
}
}  // namespace routing
}  // namespace cuopt
