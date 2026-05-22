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
class cost_node_t {
 public:
  //! Cost gathered to node
  double cost_forward = 0.0;
  //! Cost gathered after node
  double cost_backward = 0.0;

  /*! \brief { Calculate next node forward gathered cost data based on actual node} */
  void HDI calculate_forward(cost_node_t& next, double cost_between) const noexcept
  {
    next.cost_forward = cost_forward + cost_between;
  }

  /*! \brief { Calculate prev node gathered cost backward data based on actual node} */
  void HDI calculate_backward(cost_node_t& prev, double cost_between) const noexcept
  {
    prev.cost_backward = cost_backward + cost_between;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0.f, cost_forward - vehicle_info.max_cost);
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0.f, cost_backward - vehicle_info.max_cost);
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Cost excess of route represented by nodes prev and next }*/
  static HDI double combine(const cost_node_t& prev,
                            const cost_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            f_t cost_between) noexcept
  {
    double total_cost = prev.cost_forward + next.cost_backward + cost_between;
    return max(0., total_cost - vehicle_info.max_cost);
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight    = 1.,
                             const f_t excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const cost_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const cost_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double total_cost = ((double)cost_forward + (double)cost_backward);

    obj_cost[objective_t::COST] = total_cost;
    if (dim_info.has_max_constraint) {
      inf_cost[dim_t::COST] = max(0., total_cost - vehicle_info.max_cost);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
