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
class tasks_node_t {
 public:
  //! Distance gathered to node
  i_t tasks_forward = 0;
  //! Distance gathered after node
  i_t tasks_backward = 0;

  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward(tasks_node_t& next, [[maybe_unused]] f_t dummy = 0.) const noexcept
  {
    next.tasks_forward = tasks_forward + 1;
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward(tasks_node_t& prev, [[maybe_unused]] f_t dummy = 0.) const noexcept
  {
    prev.tasks_backward = tasks_backward + 1;
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
                            [[maybe_unused]] const double weight    = 1.,
                            [[maybe_unused]] const f_t excess_limit = 0.) const noexcept
  {
    return true;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next }*/
  static HDI double combine([[maybe_unused]] const tasks_node_t& prev,
                            [[maybe_unused]] const tasks_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] f_t dummy = 0.) noexcept
  {
    return 0.;
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                             [[maybe_unused]] const double weight    = 1.,
                             [[maybe_unused]] const f_t excess_limit = 0.) const noexcept
  {
    return true;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const tasks_node_t& prev_node,
                    [[maybe_unused]] const VehicleInfo<f_t, is_device>& vehicle_info,
                    const tasks_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
  {
    double diff = ((double)tasks_forward + (double)tasks_backward) - dim_info.mean_tasks;
    obj_cost[objective_t::VARIANCE_ROUTE_SIZE] = diff * diff;
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
