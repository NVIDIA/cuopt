/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "break_node.cuh"
#include "capacity_node.cuh"
#include "distance_node.cuh"
#include "mismatch_node.cuh"
#include "pdp_node.cuh"
#include "prize_node.cuh"
#include "service_time_node.cuh"
#include "tasks_node.cuh"
#include "time_node.cuh"
#include "vehicle_fixed_cost_node.cuh"

#include "../routing_helpers.cuh"

#include <routing/fleet_info.hpp>
#include <routing/routing_details.hpp>

#include <routing/arc_value.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
class node_t {
 public:
  DI node_t() = delete;

  DI node_t(const enabled_dimensions_t& dimensions_info_)
    : dimensions_info(dimensions_info_), capacity_dim(dimensions_info_.capacity_dim)
  {
  }

  node_t(const node_t<i_t, f_t, REQUEST>& other) = default;

  node_t& operator=(const node_t<i_t, f_t, REQUEST>& other) = default;

  template <int I>
  constexpr auto& get_dimension() const noexcept
  {
    using my_type = std::decay_t<decltype((*this))>;
    return detail::get_dimension_of<I, my_type>(*this);
  }

  template <int I>
  constexpr auto& get_dimension() noexcept
  {
    using my_type = std::decay_t<decltype((*this))>;
    return detail::get_dimension_of<I, my_type>(*this);
  }

  HDI i_t id() const { return request.info.node(); }
  HDI NodeInfo<i_t> node_info() const { return request.info; }

  // calculates and stores forward data
  template <bool is_device>
  constexpr void calculate_forward_all(node_t& next_node,
                                       const VehicleInfo<f_t, is_device>& vehicle_info) const
  {
    loop_over_dimensions(dimensions_info, [&](auto I) {
      double arc_value = get_arc_of_dimension<i_t, f_t, I, is_device>(
        request.info, next_node.request.info, vehicle_info);
      get_dimension<I>().calculate_forward(next_node.get_dimension<I>(), arc_value);
    });
  }

  // returns the cost delta of the new route after if we combine this node and next_node
  // this does not return the forward data but just the total cost of new route minus the old route
  template <bool is_device>
  HDI double calculate_forward_all_and_delta(node_t& next_node,
                                             VehicleInfo<f_t, is_device> const& vehicle_info,
                                             bool include_objective,
                                             infeasible_cost_t const& weights,
                                             objective_cost_t const& old_obj_cost,
                                             infeasible_cost_t const& old_inf_cost) const
  {
    calculate_forward_all(next_node, vehicle_info);

    objective_cost_t new_obj_cost;
    infeasible_cost_t new_inf_cost;
    loop_over_dimensions(dimensions_info, [&](auto I) {
      const auto& curr_dim = get_dimension<I>();
      next_node.get_dimension<I>().get_cost(
        curr_dim, vehicle_info, dimensions_info.get_dimension<I>(), new_obj_cost, new_inf_cost);
    });

    double delta =
      infeasible_cost_t::dot(weights, infeasible_cost_t::nominal_diff(new_inf_cost, old_inf_cost));
    if (include_objective) {
      // it's a copy
      auto obj_weights = dimensions_info.objective_weights;
      // In moves evalutation this function compares fragments (nodes) with routes. Resulting
      // in a corrupted delta because fragments do not have a vehicle cost. This leads to non
      // improving moves being picked.
      obj_weights[objective_t::VEHICLE_FIXED_COST] = 0.;
      delta += objective_cost_t::dot(obj_weights, new_obj_cost - old_obj_cost);
    }

    return delta;
  }

  // calculates and stores backward data
  DI void calculate_backward_all(node_t& prev_node, const VehicleInfo<f_t>& vehicle_info) const
  {
    loop_over_dimensions(dimensions_info, [&](auto I) {
      double arc_value =
        get_arc_of_dimension<i_t, f_t, I>(prev_node.request.info, request.info, vehicle_info);
      get_dimension<I>().calculate_backward(prev_node.get_dimension<I>(), arc_value);
    });
  }

  // returns the cost delta of the new route after if we combine this node and next_node
  // this does not return the forward data but just the total cost of new route minus the old route
  DI double calculate_backward_all_and_delta(node_t& prev_node,
                                             VehicleInfo<f_t> const& vehicle_info,
                                             bool include_objective,
                                             infeasible_cost_t const& weights,
                                             objective_cost_t const& old_obj_cost,
                                             infeasible_cost_t const& old_inf_cost) const
  {
    calculate_backward_all(prev_node, vehicle_info);

    objective_cost_t new_obj_cost;
    infeasible_cost_t new_inf_cost;

    loop_over_dimensions(dimensions_info, [&](auto I) {
      const auto& curr_dim = get_dimension<I>();
      prev_node.get_dimension<I>().get_cost(
        curr_dim, vehicle_info, dimensions_info.get_dimension<I>(), new_obj_cost, new_inf_cost);
    });

    double delta =
      infeasible_cost_t::dot(weights, infeasible_cost_t::nominal_diff(new_inf_cost, old_inf_cost));
    if (include_objective) {
      // it's a copy
      auto obj_weights = dimensions_info.objective_weights;
      // In moves evalutation this function compares fragments (nodes) with routes. Resulting
      // in a corrupted delta because fragments do not have a vehicle cost. This leads to non
      // improving moves being picked.
      obj_weights[objective_t::VEHICLE_FIXED_COST] = 0.;
      delta += objective_cost_t::dot(obj_weights, new_obj_cost - old_obj_cost);
    }

    return delta;
  }

  static double DI total_excess_of_combine(const node_t& prev,
                                           const node_t& next,
                                           const VehicleInfo<f_t>& vehicle_info,
                                           infeasible_cost_t weights = d_default_weights,
                                           double time_between       = -1.0)
  {
    double time_excess = 0.;
    if (prev.dimensions_info.has_dimension(dim_t::TIME)) {
      if (time_between == -1.0) {
        time_between = get_transit_time(prev.request.info, next.request.info, vehicle_info, true);
      }
      time_excess =
        time_node_t<i_t, f_t>::combine(prev.time_dim, next.time_dim, vehicle_info, time_between);
    }
    double total_excess = time_excess * weights[dim_t::TIME];
    loop_over_dimensions(prev.dimensions_info, [&] __device__(auto I) {
      // time dimension is already included
      if constexpr (I != (size_t)dim_t::TIME) {
        double arc_value =
          get_arc_of_dimension<i_t, f_t, I>(prev.request.info, next.request.info, vehicle_info);
        auto& dim_node    = prev.get_dimension<I>();
        double dim_excess = std::decay_t<decltype(dim_node)>::combine(
          prev.get_dimension<I>(), next.get_dimension<I>(), vehicle_info, arc_value);
        total_excess += dim_excess * weights[I];
      }
    });

    return total_excess;
  }

  // set the default weights to 1 so that this function works in the feasible case
  static bool DI combine(const node_t& prev,
                         const node_t& next,
                         const VehicleInfo<f_t>& vehicle_info,
                         infeasible_cost_t const& weights,
                         double excess_limit,
                         double time_between = -1.0)
  {
    return total_excess_of_combine(prev, next, vehicle_info, weights, time_between) <= excess_limit;
  }

  static bool DI feasible_combine(const node_t& prev,
                                  const node_t& next,
                                  const VehicleInfo<f_t>& vehicle_info,
                                  double time_between = -1.0)
  {
    return total_excess_of_combine(prev, next, vehicle_info, d_default_weights, time_between) <
           std::numeric_limits<double>::epsilon();
  }

  // Makes a copy of next to use calculate_forward/backward_all without modifying the actual node
  template <bool is_device>
  static double HDI cost_combine(const node_t& prev,
                                 node_t const& next,
                                 VehicleInfo<f_t, is_device> const& vehicle_info,
                                 bool include_objective,
                                 infeasible_cost_t const& weights,
                                 objective_cost_t const& old_objective_cost,
                                 infeasible_cost_t const& old_infeasibility_cost)
  {
    auto next_copy = next;
    return prev.calculate_forward_all_and_delta(next_copy,
                                                vehicle_info,
                                                include_objective,
                                                weights,
                                                old_objective_cost,
                                                old_infeasibility_cost);
  }

  static bool DI time_combine(const node_t& prev,
                              const node_t& next,
                              VehicleInfo<f_t> const& vehicle_info,
                              infeasible_cost_t const& weights,
                              double excess_limit = 0.)
  {
    if (!prev.dimensions_info.has_dimension(dim_t::TIME)) { return true; }
    auto time_between = get_transit_time(prev.request.info, next.request.info, vehicle_info, true);
    return time_node_t<i_t, f_t>::combine(
             prev.time_dim, next.time_dim, vehicle_info, time_between) *
             weights[dim_t::TIME] <=
           excess_limit;
  }

  static bool DI feasible_time_combine(const node_t& prev,
                                       const node_t& next,
                                       const VehicleInfo<f_t>& vehicle_info)
  {
    if (!prev.dimensions_info.has_dimension(dim_t::TIME)) { return true; }
    return time_combine(prev, next, vehicle_info, d_default_weights, 0.);
  }

  DI double forward_excess(const VehicleInfo<f_t>& vehicle_info,
                           infeasible_cost_t weights = d_default_weights) const
  {
    double excess = 0.;
    loop_over_dimensions(dimensions_info, [&](auto I) {
      excess += get_dimension<I>().forward_excess(vehicle_info) * weights[I];
    });
    return excess;
  }

  DI double backward_excess(const VehicleInfo<f_t>& vehicle_info,
                            infeasible_cost_t weights = d_default_weights) const
  {
    double excess = 0.;
    loop_over_dimensions(dimensions_info, [&](auto I) {
      excess += get_dimension<I>().backward_excess(vehicle_info) * weights[I];
    });
    return excess;
  }

  DI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                           infeasible_cost_t weights = d_default_weights,
                           double excess_limit       = 0.) const
  {
    return forward_excess(vehicle_info, weights) <= excess_limit;
  }

  DI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            infeasible_cost_t weights = d_default_weights,
                            double excess_limit       = 0.) const
  {
    return backward_excess(vehicle_info, weights) <= excess_limit;
  }

  DI bool feasible(const VehicleInfo<f_t>& vehicle_info,
                   infeasible_cost_t weights = d_default_weights,
                   double excess_limit       = 0.) const
  {
    return forward_excess(vehicle_info, weights) + backward_excess(vehicle_info, weights) <=
           excess_limit;
  }

  enabled_dimensions_t dimensions_info;
  request_info_t<i_t, REQUEST> request;
  time_node_t<i_t, f_t> time_dim;
  capacity_node_t<i_t, f_t> capacity_dim;
  distance_node_t<i_t, f_t> distance_dim;
  prize_node_t<i_t, f_t> prize_dim;
  tasks_node_t<i_t, f_t> tasks_dim;
  service_time_node_t<i_t, f_t> service_time_dim;
  mismatch_node_t<i_t, f_t> mismatch_dim;
  break_node_t<i_t, f_t> break_dim;
  vehicle_fixed_cost_node_t<i_t, f_t> vehicle_fixed_cost_dim;

  static constexpr int max_capacity_dim = decltype(capacity_dim)::max_capacity_dim;
};

template <typename i_t, typename f_t, request_t REQUEST, typename Enable = void>
struct request_node_t;

template <typename i_t, typename f_t, request_t REQUEST>
struct request_node_t<i_t, f_t, REQUEST, std::enable_if_t<REQUEST == request_t::PDP>> {
  request_node_t() = delete;
  HDI request_node_t(const node_t<i_t, f_t, REQUEST>& pickup_,
                     const node_t<i_t, f_t, REQUEST>& delivery_)
    : pickup(pickup_), delivery(delivery_)
  {
  }
  HDI request_node_t& operator=(const request_node_t& other)
  {
    pickup   = other.pickup;
    delivery = other.delivery;
    return *this;
  }
  HDI request_node_t(request_node_t const& other) : pickup(other.pickup), delivery(other.delivery)
  {
  }

  HDI auto& node() { return pickup; }
  HDI auto node() const { return pickup; }
  node_t<i_t, f_t, REQUEST> pickup;
  node_t<i_t, f_t, REQUEST> delivery;
};

template <typename i_t, typename f_t, request_t REQUEST>
struct request_node_t<i_t, f_t, REQUEST, std::enable_if_t<REQUEST == request_t::VRP>> {
  request_node_t() = delete;
  HDI request_node_t(const node_t<i_t, f_t, REQUEST>& pickup_or_delivery_)
    : pickup_or_delivery(pickup_or_delivery_)
  {
  }
  HDI request_node_t& operator=(const request_node_t& other)
  {
    pickup_or_delivery = other.pickup_or_delivery;
    return *this;
  }
  HDI auto& node() { return pickup_or_delivery; }
  HDI auto node() const { return pickup_or_delivery; }
  node_t<i_t, f_t, REQUEST> pickup_or_delivery;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
