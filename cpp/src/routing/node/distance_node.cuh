/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class distance_node_t {
 public:
  //! Distance gathered to node
  double distance_forward = 0.0;
  //! Distance gathered after node
  double distance_backward = 0.0;

  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward(distance_node_t& next, double distance_between) const noexcept
  {
    next.distance_forward = distance_forward + distance_between;
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward(distance_node_t& prev, double distance_between) const noexcept
  {
    prev.distance_backward = distance_backward + distance_between;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0.f, distance_forward - vehicle_info.max_cost);
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0.f, distance_backward - vehicle_info.max_cost);
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next }*/
  static HDI double combine(const distance_node_t& prev,
                            const distance_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            f_t distance_between) noexcept
  {
    double total_distance = prev.distance_forward + next.distance_backward + distance_between;
    return max(0., total_distance - vehicle_info.max_cost);
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight    = 1.,
                             const f_t excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const distance_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const cost_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double total_distance = ((double)distance_forward + (double)distance_backward);

    obj_cost[objective_t::COST] = total_distance;
    if (dim_info.has_max_constraint) {
      inf_cost[dim_t::DIST] = max(0., total_distance - vehicle_info.max_cost);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
