/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "routing/dimensions.cuh"
#include "routing/vehicle_info.hpp"

#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class break_node_t {
 public:
  //! Number of breaks gathered to node
  i_t breaks_forward = 0;
  //! Number of breaks gathered after node
  i_t breaks_backward = 0;

  /*! \brief { Calculate next node forward gathered breaks data based on actual node} */
  void HDI calculate_forward(break_node_t& next, i_t breaks_in_between) const noexcept
  {
    next.breaks_forward = breaks_forward + breaks_in_between;
  }

  /*! \brief { Calculate prev node gathered breaks backward data based on actual node} */
  void HDI calculate_backward(break_node_t& prev, double breaks_in_between) const noexcept
  {
    prev.breaks_backward = breaks_backward + breaks_in_between;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0., (double)(breaks_forward - vehicle_info.num_breaks()));
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0., (double)(breaks_backward - vehicle_info.num_breaks()));
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next }*/
  static HDI double combine(const break_node_t& prev,
                            const break_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            f_t breaks_in_between) noexcept
  {
    double total_breaks = prev.breaks_forward + next.breaks_backward + breaks_in_between;
    return max(0., (double)(total_breaks - vehicle_info.num_breaks()));
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight    = 1.,
                             const f_t excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const break_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    [[maybe_unused]] const break_dimension_info_t& dim_info,
                    [[maybe_unused]] objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double total_breaks    = ((double)breaks_forward + (double)breaks_backward);
    inf_cost[dim_t::BREAK] = max(0., (double)(total_breaks - vehicle_info.num_breaks()));
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
