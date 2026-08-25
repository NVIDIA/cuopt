/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class vehicle_fixed_cost_node_t {
 public:
  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward([[maybe_unused]] vehicle_fixed_cost_node_t& next,
                             [[maybe_unused]] f_t vehicle_fixed_cost_between) const noexcept
  {
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward([[maybe_unused]] vehicle_fixed_cost_node_t& prev,
                              [[maybe_unused]] f_t vehicle_fixed_cost_between) const noexcept
  {
  }

  HDI double forward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] const double weight       = 1.,
                            [[maybe_unused]] const double excess_limit = 0.) const noexcept
  {
    return true;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next }*/
  static HDI double combine([[maybe_unused]] const vehicle_fixed_cost_node_t& prev,
                            [[maybe_unused]] const vehicle_fixed_cost_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] double vehicle_fixed_cost_between) noexcept
  {
    return 0.;
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                             [[maybe_unused]] const double weight       = 1.,
                             [[maybe_unused]] const double excess_limit = 0.) const noexcept
  {
    return true;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const vehicle_fixed_cost_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    [[maybe_unused]] const vehicle_fixed_cost_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
  {
    // This get_cost is only invoked when the VEHICLE_FIXED_COST dimension is enabled (via
    // loop_over_dimensions). A fragment that has reached this point represents a route that
    // will be active after the move (at least one service node), so it carries this vehicle's
    // fixed cost — mirroring vehicle_fixed_cost_route_t::compute_cost on a populated route.
    obj_cost[objective_t::VEHICLE_FIXED_COST] = vehicle_info.fixed_cost;
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
