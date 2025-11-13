/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/device_span.hpp>
#include <routing/routing_details.hpp>
#include <routing/structures.hpp>
#include <routing/utilities/md_utils.hpp>

namespace cuopt {
namespace routing {

namespace detail {

template <typename i_t, typename f_t>
class order_info_t {
 public:
  order_info_t(raft::handle_t const* handle_ptr, i_t num_orders)
    : handle_ptr_(handle_ptr),
      v_demand_(num_orders, handle_ptr->get_stream()),
      v_order_locations_(num_orders, handle_ptr->get_stream()),
      v_pair_indices_(num_orders, handle_ptr->get_stream()),
      v_is_pickup_index_(num_orders, handle_ptr->get_stream()),
      v_earliest_time_(num_orders, handle_ptr->get_stream()),
      v_latest_time_(num_orders, handle_ptr->get_stream()),
      v_prizes_(num_orders, handle_ptr->get_stream())
  {
  }

  constexpr i_t get_num_orders() const { return v_earliest_time_.size(); }
  constexpr i_t get_num_depot_excluded_orders() const
  {
    return get_num_orders() - (i_t)depot_included_;
  }

  constexpr i_t get_num_requests() const
  {
    return is_pdp() ? get_num_depot_excluded_orders() / 2 : get_num_depot_excluded_orders();
  }

  void resize(i_t size, bool is_pickup, rmm::cuda_stream_view stream)
  {
    v_demand_.resize(size, stream);
    v_earliest_time_.resize(size, stream);
    v_latest_time_.resize(size, stream);
    if (is_pickup) {
      v_pair_indices_.resize(size, stream);
      v_is_pickup_index_.resize(size, stream);
    }
    v_prizes_.resize(size, stream);
  }

  bool is_pdp() const { return !v_pair_indices_.is_empty(); }

  auto to_host()
  {
    host_t h;
    h.earliest_time   = cuopt::host_copy(v_earliest_time_);
    h.latest_time     = cuopt::host_copy(v_latest_time_);
    h.demand          = cuopt::host_copy(v_demand_);
    h.prizes          = cuopt::host_copy(v_prizes_);
    h.order_locations = cuopt::host_copy(v_order_locations_);
    h.depot_included  = depot_included_;
    return h;
  }

  struct host_t {
    constexpr i_t get_order_location(i_t order_id) const
    {
      assert(order_locations.empty() || order_id < (i_t)order_locations.size());
      return order_locations.empty() ? order_id : order_locations[order_id];
    }
    std::vector<i_t> earliest_time;
    std::vector<i_t> latest_time;
    std::vector<demand_i_t> demand;
    std::vector<f_t> prizes;
    std::vector<i_t> order_locations;
    bool depot_included;
  };

  struct view_t {
    constexpr i_t get_order_location(i_t order_id) const
    {
      assert(order_locations.empty() || order_id < (i_t)order_locations.size());
      return order_locations.empty() ? order_id : order_locations[order_id];
    }
    constexpr i_t get_num_nodes() const
    {
      return order_locations.empty() ? norders : order_locations.size();
    }
    constexpr i_t get_num_orders() const { return norders; }
    constexpr i_t get_num_depot_excluded_orders() const { return (norders - (i_t)depot_included); }

    constexpr i_t get_num_requests() const { return nrequests; }

    constexpr bool is_pdp() const { return !pair_indices.empty(); }

    i_t nrequests{0};
    i_t norders{0};
    bool depot_included = true;
    raft::device_span<const i_t> order_locations;
    raft::device_span<const i_t> pair_indices;
    raft::device_span<const bool> is_pickup_index;
    raft::device_span<const demand_i_t> demand;
    raft::device_span<const i_t> earliest_time;
    raft::device_span<const i_t> latest_time;
    raft::device_span<const f_t> prizes;
  };

  view_t view() const
  {
    view_t v;
    v.norders        = v_earliest_time_.size();
    v.depot_included = depot_included_;
    v.order_locations =
      raft::device_span<const i_t>{v_order_locations_.data(), v_order_locations_.size()};
    v.pair_indices = raft::device_span<const i_t>{v_pair_indices_.data(), v_pair_indices_.size()};
    v.is_pickup_index =
      raft::device_span<const bool>{v_is_pickup_index_.data(), v_is_pickup_index_.size()};
    v.demand = raft::device_span<const demand_i_t>{v_demand_.data(), v_demand_.size()};
    v.earliest_time =
      raft::device_span<const i_t>{v_earliest_time_.data(), v_earliest_time_.size()};
    v.latest_time = raft::device_span<const i_t>{v_latest_time_.data(), v_latest_time_.size()};
    v.prizes      = raft::device_span<const f_t>{v_prizes_.data(), v_prizes_.size()};
    v.nrequests   = get_num_requests();
    return v;
  }

  raft::handle_t const* handle_ptr_{nullptr};
  bool depot_included_{true};
  // FIXME: Now that we have node and location separated out, we can use node = -1 for breaks, or
  // introduce NodeType
  rmm::device_uvector<i_t> v_order_locations_;
  rmm::device_uvector<demand_i_t> v_demand_;
  rmm::device_uvector<i_t> v_pair_indices_;
  rmm::device_uvector<bool> v_is_pickup_index_;
  rmm::device_uvector<i_t> v_earliest_time_;
  rmm::device_uvector<i_t> v_latest_time_;
  rmm::device_uvector<f_t> v_prizes_;
};

/**
 * @brief Helper function to construct order_info object from data model
 *
 * @tparam i_t
 * @tparam f_t
 * @param data_model
 * @param order_info
 */
template <typename i_t, typename f_t>
void populate_order_info(data_model_view_t<i_t, f_t> const& data_model,
                         detail::order_info_t<i_t, f_t>& order_info);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
