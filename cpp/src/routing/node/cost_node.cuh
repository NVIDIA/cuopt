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

constexpr double DISTANCE_WINDOW_INFINITY = 1e18;

// Cost dimension. Tracks the cumulative route cost (for vehicle.max_cost) and, when
// distance-based charging breaks are configured, per-node distance windows. The upper-bound
// state and lower-bound state propagate independently: arriving after window_end is hard
// infeasibility, while arriving before window_start contributes to a separately weighted
// objective. cost_forward / cost_backward keep raw-sum semantics for CVRP/TSP and
// fragment kernels that read them directly.
template <typename i_t, typename f_t>
class cost_node_t {
 public:
  //! Cost gathered to node
  double cost_forward = 0.0;
  //! Cost gathered after node
  double cost_backward = 0.0;
  // Upper-bound propagation: clamped cumulative-from-start (forward) and latest-allowable
  // cumulative-from-start (backward).
  // [window_start, window_end] = [0, DISTANCE_WINDOW_INFINITY] means unconstrained
  // (non-break node).
  double distance_window_forward  = 0.0;
  double distance_window_backward = DISTANCE_WINDOW_INFINITY;
  double window_start             = 0.0;
  double window_end               = DISTANCE_WINDOW_INFINITY;
  double excess_forward           = 0.0;
  double excess_backward          = 0.0;
  // Lower-bound propagation: maximum raw-distance shortfall in the prefix and the earliest
  // cumulative distance required by the suffix.
  double distance_window_backward_min = 0.0;
  double distance_break_cost_forward  = 0.0;

  /*! \brief { Calculate next node forward gathered cost data based on actual node} */
  void HDI calculate_forward(cost_node_t& next, double cost_between) const noexcept
  {
    next.cost_forward = cost_forward + cost_between;

    next.distance_window_forward = distance_window_forward + cost_between;
    next.excess_forward          = excess_forward;
    if (next.distance_window_forward > next.window_end) {
      next.excess_forward += next.distance_window_forward - next.window_end;
      next.distance_window_forward = next.window_end;
    }

    next.distance_break_cost_forward =
      max(distance_break_cost_forward, next.window_start - next.cost_forward);
  }

  /*! \brief { Calculate prev node gathered cost backward data based on actual node} */
  void HDI calculate_backward(cost_node_t& prev, double cost_between) const noexcept
  {
    prev.cost_backward = cost_backward + cost_between;

    prev.distance_window_backward = distance_window_backward - cost_between;
    prev.excess_backward          = excess_backward;
    if (prev.distance_window_backward > prev.window_end) {
      prev.distance_window_backward = prev.window_end;
    } else if (prev.distance_window_backward < 0.) {
      prev.excess_backward -= prev.distance_window_backward;
      prev.distance_window_backward = 0.;
    }

    prev.distance_window_backward_min = distance_window_backward_min - cost_between;
    if (prev.distance_window_backward_min < prev.window_start) {
      prev.distance_window_backward_min = prev.window_start;
    }
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return excess_forward + max(0., cost_forward - vehicle_info.max_cost);
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return excess_backward + max(0., cost_backward - vehicle_info.max_cost);
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
    double arrival_f  = prev.distance_window_forward + cost_between;
    return prev.excess_forward + next.excess_backward +
           max(0., arrival_f - next.distance_window_backward) +
           max(0., total_cost - vehicle_info.max_cost);
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
    double total_cost           = cost_forward + cost_backward;
    obj_cost[objective_t::COST] = total_cost;

    if (dim_info.has_distance_window && dim_info.has_distance_break_cost) {
      obj_cost[objective_t::DISTANCE_BREAK_COST] =
        max(distance_break_cost_forward, distance_window_backward_min - cost_forward);
    }

    inf_cost[dim_t::COST] = 0.;
    if (dim_info.has_max_constraint) {
      inf_cost[dim_t::COST] = max(0., total_cost - vehicle_info.max_cost);
    }
    if (dim_info.has_distance_window) {
      inf_cost[dim_t::COST] += excess_forward + excess_backward +
                               max(0., distance_window_forward - distance_window_backward);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
