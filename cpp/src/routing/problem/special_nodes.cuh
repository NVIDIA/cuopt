/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/structures.hpp>

#include <raft/core/span.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t>
class special_nodes_t {
 public:
  special_nodes_t() = delete;

  special_nodes_t(raft::handle_t const* _handle_ptr)
    : handle_ptr(_handle_ptr),
      num_breaks_offset(0, handle_ptr->get_stream()),
      break_nodes_offset(0, handle_ptr->get_stream()),
      node_infos(0, handle_ptr->get_stream()),
      earliest_time(0, handle_ptr->get_stream()),
      latest_time(0, handle_ptr->get_stream()),
      break_loc_to_idx(0, handle_ptr->get_stream())
  {
  }

  struct view_t {
    DI i_t size() const { return node_infos.size(); }

    DI bool empty() const { return size() == 0; }

    constexpr i_t get_break_loc_idx(NodeInfo<> const& break_node) const
    {
      return break_loc_to_idx[break_node.location()];
    }

    DI view_t subset(const int vehicle_id, const int break_dim) const
    {
      view_t v;
      v.num_vehicles = num_vehicles;
      // v.num_break_dimensions            = num_break_dimensions;
      // v.nodes_per_dimension_per_vehicle = nodes_per_dimension_per_vehicle;

      i_t break_offset = num_breaks_offset[vehicle_id] + break_dim;
      i_t offset       = break_nodes_offset[break_offset];
      i_t sz           = break_nodes_offset[break_offset + 1] - offset;
      v.node_infos     = raft::device_span<const NodeInfo<>>(node_infos.data() + offset, sz);
      v.earliest_time  = raft::device_span<const i_t>(earliest_time.data() + offset, sz);
      v.latest_time    = raft::device_span<const i_t>(latest_time.data() + offset, sz);

      return v;
    }

    DI i_t get_break_dimensions(const i_t vehicle_id) const
    {
      if (!empty()) { return num_breaks_offset[vehicle_id + 1] - num_breaks_offset[vehicle_id]; }
      return 0;
    }

    i_t num_vehicles{0};
    i_t num_max_break_dimensions{0};
    // i_t num_break_dimensions{0};
    // i_t nodes_per_dimension_per_vehicle{0};
    raft::device_span<const i_t> num_breaks_offset;
    raft::device_span<const i_t> break_nodes_offset;
    raft::device_span<const NodeInfo<>> node_infos;
    raft::device_span<const i_t> earliest_time;
    raft::device_span<const i_t> latest_time;
    raft::device_span<const i_t> break_loc_to_idx;
  };

  view_t view() const
  {
    view_t v;
    v.num_vehicles             = num_vehicles;
    v.num_max_break_dimensions = num_max_break_dimensions;
    // v.num_break_dimensions            = num_break_dimensions;
    // v.nodes_per_dimension_per_vehicle = nodes_per_dimension_per_vehicle;

    v.num_breaks_offset  = cuopt::make_span(num_breaks_offset);
    v.break_nodes_offset = cuopt::make_span(break_nodes_offset);
    v.node_infos         = cuopt::make_span(node_infos);
    v.earliest_time      = cuopt::make_span(earliest_time);
    v.latest_time        = cuopt::make_span(latest_time);
    v.break_loc_to_idx   = cuopt::make_span(break_loc_to_idx);

    return v;
  }

  size_t size() const { return node_infos.size(); }

  bool is_empty() const { return size() == 0; }

  // handle
  raft::handle_t const* handle_ptr{nullptr};

  i_t num_vehicles{0};
  i_t num_max_break_dimensions{0};
  // FIXME:: Use mdarray

  rmm::device_uvector<i_t> num_breaks_offset;
  rmm::device_uvector<i_t> break_nodes_offset;
  rmm::device_uvector<NodeInfo<>> node_infos;
  rmm::device_uvector<i_t> earliest_time;
  rmm::device_uvector<i_t> latest_time;
  rmm::device_uvector<i_t> break_loc_to_idx;
};
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
