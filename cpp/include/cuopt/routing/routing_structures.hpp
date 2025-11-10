/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/device_span.hpp>

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
namespace cuopt {
namespace routing {
/**
 * @brief Enumerated representation of supported objective function types
 *
 */
enum class objective_t {
  COST,         // Cost of all the routes according to the cost matrix
  TRAVEL_TIME,  // Driving time (excludes the wait time) of all the routes according to the travel
                // time matirx
  VARIANCE_ROUTE_SIZE,          // Variance in route sizes
  VARIANCE_ROUTE_SERVICE_TIME,  // Variance in route service times
  PRIZE,                        // Sum of prizes of all orders that are served
  VEHICLE_FIXED_COST,           // Used when fixed vehicle cost are enabled
  SIZE  // Helper enum to keep track of number of supported objective functions
};

enum class node_type_t : uint8_t { DEPOT = 0, PICKUP, DELIVERY, BREAK };

using demand_i_t = int32_t;
using cap_i_t    = int32_t;

namespace detail {

template <typename i_t, typename f_t>
class break_dimension_t {
 public:
  break_dimension_t(i_t const* break_earliest, i_t const* break_latest, i_t const* break_duration)
    : break_earliest_(break_earliest), break_latest_(break_latest), break_duration_(break_duration)
  {
  }

  std::tuple<i_t const*, i_t const*, i_t const*> get_breaks() const
  {
    return std::make_tuple(break_earliest_, break_latest_, break_duration_);
  }

 private:
  i_t const* break_earliest_;
  i_t const* break_latest_;
  i_t const* break_duration_;
};

template <typename i_t>
class vehicle_break_t {
 public:
  vehicle_break_t(i_t earliest, i_t latest, i_t duration, raft::device_span<const i_t> locations)
    : earliest_(earliest), latest_(latest), duration_(duration), locations_(locations)
  {
  }

  i_t earliest_;
  i_t latest_;
  i_t duration_;
  raft::device_span<const i_t> locations_{};
};

template <typename i_t, typename f_t>
class vehicle_time_window_t {
 public:
  vehicle_time_window_t(i_t const* earliest, i_t const* latest)
    : earliest_(earliest), latest_(latest)
  {
  }
  vehicle_time_window_t() = default;
  i_t const* get_earliest_time() const { return earliest_; }
  i_t const* get_latest_time() const { return latest_; }

 private:
  i_t const* earliest_{nullptr};
  i_t const* latest_{nullptr};
};

template <typename i_t, typename f_t>
class capacity_t {
 public:
  capacity_t(std::string const& name, i_t const* demands, i_t const* vehicle_capacities)
    : name_(name), demands_(demands), vehicle_capacities_(vehicle_capacities)
  {
  }
  i_t const* get_demands() const { return demands_; }
  i_t const* get_vehicle_capacities() const { return vehicle_capacities_; }

 private:
  std::string name_{nullptr};
  i_t const* demands_{nullptr};
  i_t const* vehicle_capacities_{nullptr};
};

// internal
template <typename i_t, typename f_t>
class order_time_window_t {
 public:
  order_time_window_t(i_t const* earliest, i_t const* latest) : earliest_(earliest), latest_(latest)
  {
  }

  order_time_window_t() = default;

  i_t const* get_earliest_time() const { return earliest_; }
  i_t const* get_latest_time() const { return latest_; }

 private:
  i_t const* earliest_{nullptr};
  i_t const* latest_{nullptr};
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
