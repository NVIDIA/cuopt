/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "routing/dimensions.cuh"
#include "routing/routing_structures.hpp"
#include "routing/vehicle_info.hpp"

#include <algorithm>
#include <limits>

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
  //! Number of load-window violations in the realized prefix ending at this node
  i_t break_window_violations_forward = 0;
  //! Realized active load after this node for each capacity dimension
  i_t load_at_node[max_break_load_window_dim] = {0};
  //! Admissible incoming load interval before this node for the suffix
  i_t incoming_min[max_break_load_window_dim] = {std::numeric_limits<i_t>::lowest(),
                                                 std::numeric_limits<i_t>::lowest(),
                                                 std::numeric_limits<i_t>::lowest()};
  i_t incoming_max[max_break_load_window_dim] = {std::numeric_limits<i_t>::max(),
                                                 std::numeric_limits<i_t>::max(),
                                                 std::numeric_limits<i_t>::max()};
  //! Optional fixed load window on this break node
  bool load_window_active[max_break_load_window_dim] = {false};
  i_t load_window_min[max_break_load_window_dim]     = {0};
  i_t load_window_max[max_break_load_window_dim]     = {0};

  /*! \brief { Calculate next node forward gathered breaks data based on actual node} */
  void HDI calculate_forward(break_node_t& next, i_t breaks_in_between) const noexcept
  {
    next.breaks_forward = breaks_forward + breaks_in_between;
    next.break_window_violations_forward = break_window_violations_forward;
  }

  /*! \brief { Calculate prev node gathered breaks backward data based on actual node} */
  void HDI calculate_backward(break_node_t& prev, double breaks_in_between) const noexcept
  {
    prev.breaks_backward = breaks_backward + breaks_in_between;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0., (double)(breaks_forward - vehicle_info.num_breaks())) +
           (double)break_window_violations_forward;
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0., (double)(breaks_backward - vehicle_info.num_breaks())) +
           (double)!has_non_empty_incoming_interval();
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Break
   * excess of route represented by nodes prev and next }*/
  static HDI double combine(const break_node_t& prev,
                            const break_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            f_t breaks_in_between) noexcept
  {
    double total_breaks = prev.breaks_forward + next.breaks_backward + breaks_in_between;
    double excess = max(0., (double)(total_breaks - vehicle_info.num_breaks()));
    excess += prev.break_window_violations_forward;
    excess += next.boundary_load_is_admissible(prev.load_at_node) ? 0. : 1.;
    return excess;
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
    double total_breaks = ((double)breaks_forward + (double)breaks_backward);
    inf_cost[dim_t::BREAK] =
      max(0., (double)(total_breaks - vehicle_info.num_breaks())) +
      (double)prev_node.break_window_violations_forward +
      (boundary_load_is_admissible(prev_node.load_at_node) ? 0. : 1.);
  }

  HDI bool has_non_empty_incoming_interval() const noexcept
  {
    for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
      if (incoming_min[dim] > incoming_max[dim]) { return false; }
    }
    return true;
  }

  HDI bool boundary_load_is_admissible(const i_t boundary_load[max_break_load_window_dim]) const
    noexcept
  {
    for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
      if (boundary_load[dim] < incoming_min[dim] || boundary_load[dim] > incoming_max[dim]) {
        return false;
      }
    }
    return true;
  }

  HDI bool satisfies_load_window() const noexcept
  {
    for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
      if (!load_window_active[dim]) { continue; }
      if (load_at_node[dim] < load_window_min[dim] || load_at_node[dim] > load_window_max[dim]) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
