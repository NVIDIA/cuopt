/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class service_time_node_t {
 public:
  //! Distance gathered to node
  double service_time_forward = 0.0;
  //! Distance gathered after node
  double service_time_backward = 0.0;

  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward(service_time_node_t& next, f_t service_time_between) const noexcept
  {
    next.service_time_forward = service_time_forward + service_time_between;
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward(service_time_node_t& prev, f_t service_time_between) const noexcept
  {
    prev.service_time_backward = service_time_backward + service_time_between;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept { return 0.; }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept { return 0.; }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight       = 1.,
                            const double excess_limit = 0.) const noexcept
  {
    return true;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next }*/
  static HDI double combine(const service_time_node_t& prev,
                            const service_time_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            double service_time_between) noexcept
  {
    return 0.;
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight       = 1.,
                             const double excess_limit = 0.) const noexcept
  {
    return true;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const service_time_node_t& prev_node,
                    [[maybe_unused]] const VehicleInfo<f_t, is_device>& vehicle_info,
                    const service_time_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
  {
    double total_service_time = ((double)service_time_forward + (double)service_time_backward);
    double tmp                = total_service_time - dim_info.mean_service_time;
    obj_cost[objective_t::VARIANCE_ROUTE_SERVICE_TIME] = tmp * tmp;
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
