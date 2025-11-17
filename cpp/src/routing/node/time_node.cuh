/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../routing_helpers.cuh"

#include <routing/arc_value.hpp>
#include <routing/fleet_info.hpp>
#include <routing/routing_details.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class time_node_t {
 public:
  //! Hashimoto Yagiura [hy] forward departure time
  double departure_forward = 0.0;
  //! [hy] forward time excess
  double excess_forward = 0.0;
  //! [hy] backward departure time
  double departure_backward = 0.0;
  //! [hy] backward time excess
  double excess_backward = 0.0;
  //! Copied from problem data time window start/end for convenience
  double window_start = 0.0;
  double window_end   = 0.0;

  //! Time gathered to node and after node.
  //! These are needed when we use TIME as objective function, and max times as constraints
  double transit_time_forward     = 0.0;
  double latest_arrival_forward   = 0.;
  double unavoidable_wait_forward = 0.;

  double transit_time_backward     = 0.0;
  double earliest_arrival_backward = 0.;
  double unavoidable_wait_backward = 0.;

  /*! \brief { Calculate next node forward time data based on actual node} */
  void HDI calculate_forward(time_node_t& next, double time_between) const noexcept
  {
    next.departure_forward = departure_forward + time_between;
    next.excess_forward    = excess_forward;

    if (next.departure_forward < next.window_start) {
      next.departure_forward = next.window_start;
    } else if (next.departure_forward > next.window_end) {
      next.excess_forward += next.departure_forward - next.window_end;
      next.departure_forward = next.window_end;
    }

    next.latest_arrival_forward   = latest_arrival_forward + time_between;
    next.unavoidable_wait_forward = unavoidable_wait_forward;
    if (next.latest_arrival_forward < next.window_start) {
      next.unavoidable_wait_forward += (next.window_start - next.latest_arrival_forward);
      next.latest_arrival_forward = next.window_start;
    } else if (next.latest_arrival_forward > next.window_end) {
      next.latest_arrival_forward = next.window_end;
    }
    next.transit_time_forward = transit_time_forward + time_between;
  }

  /*! \brief { Calculate prev node time backward data based on actual node} */
  void HDI calculate_backward(time_node_t& prev, double time_between) const noexcept
  {
    prev.departure_backward = departure_backward - time_between;
    prev.excess_backward    = excess_backward;

    if (prev.departure_backward > prev.window_end)
      prev.departure_backward = prev.window_end;
    else if (prev.departure_backward < prev.window_start) {
      prev.excess_backward += prev.window_start - prev.departure_backward;
      prev.departure_backward = prev.window_start;
    }

    prev.earliest_arrival_backward = earliest_arrival_backward - time_between;
    prev.unavoidable_wait_backward = unavoidable_wait_backward;
    if (prev.earliest_arrival_backward > prev.window_end) {
      prev.unavoidable_wait_backward += (prev.earliest_arrival_backward - prev.window_end);
      prev.earliest_arrival_backward = prev.window_end;
    } else if (prev.earliest_arrival_backward < prev.window_start) {
      prev.earliest_arrival_backward = prev.window_start;
    }

    prev.transit_time_backward = transit_time_backward + time_between;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Time excess of route represented by nodes prev and next }*/
  static HDI double combine(const time_node_t& prev,
                            const time_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            double time_between) noexcept
  {
    double wait_arrival    = prev.latest_arrival_forward + time_between;
    double total_wait_time = prev.unavoidable_wait_forward + next.unavoidable_wait_backward +
                             max(0., next.earliest_arrival_backward - wait_arrival);
    double total_transit_time =
      prev.transit_time_forward + next.transit_time_backward + time_between;
    double total_time = total_transit_time + total_wait_time;

    double arrival_f = prev.departure_forward + time_between;
    return prev.excess_forward + next.excess_backward +
           max(0.0, arrival_f - next.departure_backward) +
           max(0.0, total_time - vehicle_info.max_time);
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return excess_forward +
           max(0.f, transit_time_forward + unavoidable_wait_forward - vehicle_info.max_time);
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return excess_backward +
           max(0.f, transit_time_backward + unavoidable_wait_backward - vehicle_info.max_time);
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            double weight       = 1.0,
                            double excess_limit = 0.) const
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight = 1.0,
                             double excess_limit = 0.) const
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const time_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const time_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    inf_cost[dim_t::TIME] =
      (excess_forward + excess_backward + max(0., departure_forward - departure_backward));
    if (dim_info.should_compute_travel_time()) {
      const double total_wait_time = unavoidable_wait_forward + unavoidable_wait_backward +
                                     max(0., earliest_arrival_backward - latest_arrival_forward);
      const double total_transit_time = transit_time_forward + transit_time_backward;

      const double total_time = total_wait_time + total_transit_time;

      if (dim_info.has_travel_time_obj) { obj_cost[objective_t::TRAVEL_TIME] = total_time; }
      if (dim_info.has_max_constraint) {
        inf_cost[dim_t::TIME] += max(0., total_time - (double)vehicle_info.max_time);
      }
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
