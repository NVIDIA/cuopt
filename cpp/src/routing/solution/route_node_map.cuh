/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <thrust/pair.h>
#include <raft/core/device_span.hpp>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t>
class route_node_map_t {
 public:
  route_node_map_t(const int num_orders, rmm::cuda_stream_view stream)
    : route_id_per_node(num_orders, stream), intra_route_idx_per_node(num_orders, stream)
  {
    thrust::fill(rmm::exec_policy(stream), route_id_per_node.begin(), route_id_per_node.end(), -1);
    thrust::fill(rmm::exec_policy(stream),
                 intra_route_idx_per_node.begin(),
                 intra_route_idx_per_node.end(),
                 -1);
  }

  route_node_map_t(const route_node_map_t& other, rmm::cuda_stream_view stream)
    : route_id_per_node(other.route_id_per_node, stream),
      intra_route_idx_per_node(other.intra_route_idx_per_node, stream)
  {
  }

  void copy_from(const route_node_map_t& other, rmm::cuda_stream_view stream)
  {
    raft::copy(intra_route_idx_per_node.data(),
               other.intra_route_idx_per_node.data(),
               other.intra_route_idx_per_node.size(),
               stream);
    raft::copy(route_id_per_node.data(),
               other.route_id_per_node.data(),
               other.route_id_per_node.size(),
               stream);
  }

  std::pair<i_t, i_t> get_route_id_and_intra_idx(const NodeInfo<i_t>& node) const
  {
    if (node.is_service_node()) {
      i_t node_id   = node.node();
      i_t route_id  = route_id_per_node.element(node_id, route_id_per_node.stream());
      i_t intra_idx = intra_route_idx_per_node.element(node_id, intra_route_idx_per_node.stream());

      return {route_id, intra_idx};
    }
    return {-1, -1};
  }

  i_t get_route_id(const NodeInfo<i_t>& node) const
  {
    if (node.is_service_node()) {
      i_t node_id = node.node();
      return route_id_per_node.element(node_id, route_id_per_node.stream());
    }
    return -1;
  }

  struct view_t {
    DI void set_route_id(const NodeInfo<i_t>& node, const i_t route_id)
    {
      if (node.is_service_node()) {
        i_t node_id                = node.node();
        route_id_per_node[node_id] = route_id;
      }
    }

    DI void set_intra_route_idx(const NodeInfo<i_t>& node, const i_t intra_route_idx)
    {
      if (node.is_service_node()) {
        i_t node_id                       = node.node();
        intra_route_idx_per_node[node_id] = intra_route_idx;
        cuopt_assert(route_id_per_node[node_id] != -1,
                     "Route id should be set before intra route idx");
      }
    }

    DI void set_route_id_and_intra_idx(const NodeInfo<i_t>& node,
                                       const i_t route_id,
                                       const i_t intra_route_idx)
    {
      if (node.is_service_node()) {
        i_t node_id                       = node.node();
        route_id_per_node[node_id]        = route_id;
        intra_route_idx_per_node[node_id] = intra_route_idx;
      }
    }

    DI void reset_node(const NodeInfo<i_t>& node)
    {
      if (node.is_service_node()) {
        i_t node_id                       = node.node();
        route_id_per_node[node_id]        = -1;
        intra_route_idx_per_node[node_id] = -1;
      }
    }

    DI void reset_node(const i_t node_id)
    {
      route_id_per_node[node_id]        = -1;
      intra_route_idx_per_node[node_id] = -1;
    }

    DI i_t get_route_id(const i_t node_id) const
    {
      cuopt_assert(node_id < route_id_per_node.size(),
                   "node_id should be less than number of orders");
      return route_id_per_node[node_id];
    }

    DI i_t get_route_id(const NodeInfo<>& node) const
    {
      if (node.is_service_node()) { return get_route_id(node.node()); }
      return -1;
    }

    DI i_t get_intra_route_idx(const i_t node_id) const
    {
      return intra_route_idx_per_node[node_id];
    }

    DI i_t get_intra_route_idx(const NodeInfo<i_t>& node) const
    {
      if (node.is_service_node()) {
        i_t node_id = node.node();
        return get_intra_route_idx(node_id);
      }
      return -1;
    }

    DI thrust::pair<i_t, i_t> get_route_id_and_intra_idx(const i_t node_id) const
    {
      return thrust::make_pair(route_id_per_node[node_id], intra_route_idx_per_node[node_id]);
    }

    DI thrust::pair<i_t, i_t> get_route_id_and_intra_idx(const NodeInfo<i_t>& node) const
    {
      if (node.is_service_node()) {
        i_t node_id = node.node();
        return get_route_id_and_intra_idx(node_id);
      }
      return thrust::make_pair(-1, -1);
    }

    DI void copy_from(const view_t& other)
    {
      block_copy(route_id_per_node, other.route_id_per_node);
      block_copy(intra_route_idx_per_node, other.intra_route_idx_per_node);
    }

    DI i_t size() const { return route_id_per_node.size(); }

    DI bool is_node_served(const i_t node_id) const
    {
      return route_id_per_node[node_id] != -1 && intra_route_idx_per_node[node_id] != -1;
    }

    DI bool is_node_served(const NodeInfo<i_t>& node) const
    {
      if (node.is_service_node()) { return is_node_served(node.node()); }

      return true;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared(i_t* shmem, i_t sz)
    {
      view_t v;
      i_t* sh_ptr = shmem;

      thrust::tie(v.route_id_per_node, sh_ptr)        = wrap_ptr_as_span<i_t>(sh_ptr, sz);
      thrust::tie(v.intra_route_idx_per_node, sh_ptr) = wrap_ptr_as_span<i_t>(sh_ptr, sz);
      return thrust::make_tuple(v, sh_ptr);
    }

    raft::device_span<i_t> route_id_per_node;
    raft::device_span<i_t> intra_route_idx_per_node;
  };

  view_t view()
  {
    view_t v;
    v.route_id_per_node =
      raft::device_span<i_t>{route_id_per_node.data(), route_id_per_node.size()};
    v.intra_route_idx_per_node =
      raft::device_span<i_t>{intra_route_idx_per_node.data(), intra_route_idx_per_node.size()};
    return v;
  }

  // route ids per node
  rmm::device_uvector<i_t> route_id_per_node;
  // intra route idx per node
  rmm::device_uvector<i_t> intra_route_idx_per_node;
};
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
