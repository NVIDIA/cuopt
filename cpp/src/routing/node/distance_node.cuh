/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "routing/dimensions.cuh"
#include "routing/vehicle_info.hpp"

#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

// Distance dimension. Tracks both the cumulative route distance (for vehicle.max_cost) and
// per-node distance windows used by distance-based charging breaks. The window fields parallel
// time_node_t; distance_forward / distance_backward keep raw-sum semantics for CVRP/TSP and
// fragment kernels that read them directly.
template <typename i_t, typename f_t>
class distance_node_t {
 public:
  double distance_forward  = 0.0;
  double distance_backward = 0.0;
  // Window-clamped forward cumulative and constraint-propagation backward bounds.
  double distance_window_forward      = 0.0;
  double distance_window_backward     = 1e18;  // latest allowable cumulative-from-start
  double distance_window_backward_min = 0.0;   // earliest required cumulative-from-start
  // [0, 1e18] means unconstrained (non-break node).
  double window_start    = 0.0;
  double window_end      = 1e18;
  double excess_forward  = 0.0;
  double excess_backward = 0.0;

  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward(distance_node_t& next, double distance_between) const noexcept
  {
    next.distance_forward        = distance_forward + distance_between;
    next.distance_window_forward = distance_window_forward + distance_between;
    next.excess_forward          = excess_forward;
    if (next.distance_window_forward < next.window_start) {
      next.excess_forward += next.window_start - next.distance_window_forward;
      next.distance_window_forward = next.window_start;
    } else if (next.distance_window_forward > next.window_end) {
      next.excess_forward += next.distance_window_forward - next.window_end;
      next.distance_window_forward = next.window_end;
    }
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward(distance_node_t& prev, double distance_between) const noexcept
  {
    prev.distance_backward            = distance_backward + distance_between;
    prev.distance_window_backward     = distance_window_backward - distance_between;
    prev.distance_window_backward_min = distance_window_backward_min - distance_between;
    prev.excess_backward              = excess_backward;
    // Latest-allowable propagation: lower clamp = suffix can't reach prev within its window.
    if (prev.distance_window_backward > prev.window_end) {
      prev.distance_window_backward = prev.window_end;
    } else if (prev.distance_window_backward < prev.window_start) {
      prev.excess_backward += prev.window_start - prev.distance_window_backward;
      prev.distance_window_backward = prev.window_start;
    }
    // Earliest-required propagation: upper clamp = suffix forces prev past its window_end.
    if (prev.distance_window_backward_min < prev.window_start) {
      prev.distance_window_backward_min = prev.window_start;
    } else if (prev.distance_window_backward_min > prev.window_end) {
      prev.excess_backward += prev.distance_window_backward_min - prev.window_end;
      prev.distance_window_backward_min = prev.window_end;
    }
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0., distance_forward - vehicle_info.max_cost) + excess_forward;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return excess_backward;
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next } */
  static HDI double combine(const distance_node_t& prev,
                            const distance_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            f_t distance_between) noexcept
  {
    double arrival_window_f = prev.distance_window_forward + distance_between;
    double upper_excess     = max(0., arrival_window_f - next.distance_window_backward);
    double lower_excess     = max(0., next.distance_window_backward_min - arrival_window_f);
    double total_distance   = prev.distance_forward + distance_between + next.distance_backward;
    double max_cost_excess  = max(0., total_distance - vehicle_info.max_cost);
    return prev.excess_forward + next.excess_backward + upper_excess + lower_excess +
           max_cost_excess;
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
    double total_distance       = distance_forward + distance_backward;
    obj_cost[objective_t::COST] = total_distance;
    if (dim_info.has_distance_window) {
      double upper_boundary  = max(0., distance_window_forward - distance_window_backward);
      double lower_boundary  = max(0., distance_window_backward_min - distance_window_forward);
      double max_cost_excess = max(0., total_distance - vehicle_info.max_cost);
      inf_cost[dim_t::DIST] =
        excess_forward + excess_backward + upper_boundary + lower_boundary + max_cost_excess;
    } else if (dim_info.has_max_constraint) {
      inf_cost[dim_t::DIST] = max(0., total_distance - vehicle_info.max_cost);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
