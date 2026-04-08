/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/routing/data_model_view.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <raft/core/span.hpp>

#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <vector>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t>
struct fleet_order_constraints_t {
  fleet_order_constraints_t(raft::handle_t const* handle_ptr_, i_t n_vehicles_, i_t n_orders_)
    : handle_ptr(handle_ptr_),
      n_vehicles(n_vehicles_),
      n_orders(n_orders_),
      order_service_times(n_vehicles * n_orders, handle_ptr_->get_stream()),
      order_costs(0, handle_ptr_->get_stream())
  {
  }

  void fill(i_t val)
  {
    thrust::uninitialized_fill(
      handle_ptr->get_thrust_policy(), order_service_times.begin(), order_service_times.end(), val);
  }

  void resize(i_t n_vehicles_, i_t n_orders_)
  {
    n_vehicles = n_vehicles_;
    n_orders   = n_orders_;
    order_service_times.resize(n_vehicles * n_orders, order_service_times.stream());
  }

  struct host_t {
    static constexpr bool is_device = false;
    constexpr auto get_order_service_times(i_t truck_id) const
    {
      return raft::span<i_t const, is_device>(order_service_times.data() + truck_id * n_orders,
                                              n_orders);
    }

    constexpr auto get_order_costs(i_t truck_id) const
    {
      if (order_costs.empty()) { return raft::span<double const, is_device>{}; }
      cuopt_assert(order_costs.size() == (size_t)(n_orders * n_vehicles),
                   "size mismatch of order_costs vector");
      return raft::span<double const, is_device>(order_costs.data() + truck_id * n_orders,
                                                 n_orders);
    }

    std::vector<i_t> order_service_times;
    std::vector<double> order_costs;
    i_t n_orders;
    i_t n_vehicles;
  };

  host_t to_host(rmm::cuda_stream_view stream)
  {
    host_t h;
    h.order_service_times = host_copy(order_service_times, stream);
    h.order_costs         = host_copy(order_costs, stream);
    h.n_orders            = n_orders;
    h.n_vehicles          = n_vehicles;
    return h;
  }

  struct view_t {
    constexpr raft::device_span<i_t const> get_order_service_times(i_t truck_id) const
    {
      return raft::device_span<i_t const>(order_service_times.data() + truck_id * n_orders,
                                          n_orders);
    }

    raft::device_span<double const> get_order_costs(i_t truck_id) const
    {
      if (order_costs.empty()) { return raft::device_span<double const>{}; }
      cuopt_assert(order_costs.size() == (size_t)(n_orders * n_vehicles),
                   "size mismatch of order_costs vector");
      return raft::device_span<double const>(order_costs.data() + truck_id * n_orders, n_orders);
    }

    i_t n_vehicles{};
    i_t n_orders{};
    raft::device_span<i_t const> order_service_times{};
    raft::device_span<double const> order_costs{};
  };

  constexpr raft::device_span<i_t const> get_order_service_times(i_t truck_id) const
  {
    return raft::device_span<i_t const>(order_service_times.data() + truck_id * n_orders, n_orders);
  }

  raft::device_span<double const> get_order_costs(i_t truck_id) const
  {
    if (order_costs.is_empty()) { return raft::device_span<double const>{}; }
    cuopt_assert(order_costs.size() == (size_t)(n_orders * n_vehicles),
                 "size mismatch of order_costs vector");
    return raft::device_span<double const>(order_costs.data() + truck_id * n_orders, n_orders);
  }

  view_t view() const
  {
    view_t v;
    v.order_service_times =
      raft::device_span<i_t const>(order_service_times.data(), order_service_times.size());
    v.order_costs = raft::device_span<double const>(order_costs.data(), order_costs.size());
    v.n_vehicles  = n_vehicles;
    v.n_orders    = n_orders;
    return v;
  }

  raft::handle_t const* handle_ptr{nullptr};
  i_t n_vehicles{};
  i_t n_orders{};
  rmm::device_uvector<i_t> order_service_times;
  rmm::device_uvector<double> order_costs;
};

/**
 * @brief helper function to convert vehicle order match constraints to appropriate service times
 *
 * @tparam i_t
 * @tparam f_t
 * @param data_model
 * @param fleet_order_constraints_
 */
template <typename i_t, typename f_t>
void populate_vehicle_order_match(data_model_view_t<i_t, f_t> const& data_model,
                                  fleet_order_constraints_t<i_t>& fleet_order_constraints,
                                  bool& is_homogenous);

template <typename i_t, typename f_t>
void populate_vehicle_order_cost(data_model_view_t<i_t, f_t> const& data_model,
                                 fleet_order_constraints_t<i_t>& fleet_order_constraints);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
