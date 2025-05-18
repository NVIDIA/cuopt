/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/node.cuh"
#include "../solution/solution_handle.cuh"
#include "tsp_route.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST, typename Enable = void>
class request_route_t;

template <typename i_t, typename f_t, request_t REQUEST>
class request_route_t<i_t, f_t, REQUEST, std::enable_if_t<REQUEST == request_t::PDP>> {
 public:
  request_route_t(solution_handle_t<i_t, f_t> const* sol_handle_)
    : node_info(0, sol_handle_->get_stream()),
      brother_info(0, sol_handle_->get_stream()),
      tsp_requests(sol_handle_)
  {
    raft::common::nvtx::range fun_scope("zero pdp_route_t copy_ctr");
  }

  request_route_t(const request_route_t& request_route,
                  solution_handle_t<i_t, f_t> const* sol_handle_)
    : node_info(request_route.node_info, sol_handle_->get_stream()),
      brother_info(request_route.brother_info, sol_handle_->get_stream()),
      tsp_requests(request_route.tsp_requests, sol_handle_)
  {
    raft::common::nvtx::range fun_scope("pdp route copy_ctr");
  }

  request_route_t& operator=(request_route_t&& request_route) = default;

  void print(i_t n_nodes) const
  {
    std::cout << "[";
    for (i_t i = 0; i < n_nodes; ++i) {
      auto info_i      = node_info.element(i, node_info.stream());
      auto id          = info_i.node();
      auto type_string = NodeInfo<i_t>::get_string(info_i.node_type());
      std::cout << "(" << type_string << ", " << id << "); ";
    }
    std::cout << "]\n";
  }

  void resize(i_t max_nodes_per_route, bool is_tsp, rmm::cuda_stream_view stream)
  {
    node_info.resize(max_nodes_per_route, stream);
    brother_info.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return node_info.empty(); }

    DI request_info_t<i_t, REQUEST> get_node(i_t idx) const
    {
      request_info_t<i_t, REQUEST> request_node;
      request_node.info         = node_info[idx];
      request_node.brother_info = brother_info[idx];

      return request_node;
    }

    DI void set_node(i_t idx, const request_info_t<i_t, REQUEST>& node)
    {
      node_info[idx]    = node.info;
      brother_info[idx] = node.brother_info;
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      auto size = to_idx - from_idx;
      block_copy(node_info.subspan(write_start), orig_route.node_info.subspan(from_idx), size);
      block_copy(
        brother_info.subspan(write_start), orig_route.brother_info.subspan(from_idx), size);
    }

    DI bool is_pickup_node(i_t idx) const { return node_info[idx].is_pickup(); }

    DI i_t node_id(i_t idx) const { return node_info[idx].node(); };

    DI i_t brother_id(i_t idx) const { return brother_info[idx].node(); };

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              i_t n_nodes_route,
                                                              bool is_tsp = false)
    {
      view_t v;

      v.node_info =
        raft::device_span<NodeInfo<i_t>>{(NodeInfo<i_t>*)shmem, (size_t)n_nodes_route + 1};
      v.brother_info = raft::device_span<NodeInfo<i_t>>{&v.node_info.data()[n_nodes_route + 1],
                                                        (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.brother_info.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    raft::device_span<NodeInfo<i_t>> node_info;
    raft::device_span<NodeInfo<i_t>> brother_info;
    typename tsp_route_t<i_t, f_t>::view_t tsp_requests;
  };

  view_t view()
  {
    view_t v;
    v.node_info    = raft::device_span<NodeInfo<i_t>>{node_info.data(), node_info.size()};
    v.brother_info = raft::device_span<NodeInfo<i_t>>{brother_info.data(), brother_info.size()};
    v.tsp_requests = tsp_requests.view();
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a request route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size, bool is_tsp = false)
  {
    // node, brother
    size_t byte_size = request_info_t<i_t, REQUEST>::size() * route_size * sizeof(NodeInfo<i_t>);
    return raft::alignTo(byte_size, sizeof(double));
  }

  // node ids
  rmm::device_uvector<NodeInfo<i_t>> node_info;

  // brother ids
  rmm::device_uvector<NodeInfo<i_t>> brother_info;

  // PDP is instantiated so this variable is needed but not the implementation
  tsp_route_t<i_t, f_t> tsp_requests;
};

template <typename i_t, typename f_t, request_t REQUEST>
class request_route_t<i_t, f_t, REQUEST, std::enable_if_t<REQUEST == request_t::VRP>> {
 public:
  request_route_t(solution_handle_t<i_t, f_t> const* sol_handle_)
    : node_info(0, sol_handle_->get_stream()), tsp_requests(sol_handle_)
  {
  }

  request_route_t(const request_route_t& request_route,
                  solution_handle_t<i_t, f_t> const* sol_handle_)
    : node_info(request_route.node_info, sol_handle_->get_stream()),
      tsp_requests(request_route.tsp_requests, sol_handle_)
  {
  }

  request_route_t& operator=(request_route_t&& request_route) = default;

  void print(i_t n_nodes) const
  {
    std::cout << "[";
    for (i_t i = 0; i < n_nodes; ++i) {
      auto info_i      = node_info.element(i, node_info.stream());
      auto id          = info_i.node();
      auto type_string = NodeInfo<i_t>::get_string(info_i.node_type());
      std::cout << "(" << type_string << ", " << id << "); ";
    }
    std::cout << "]\n";
  }

  void resize(i_t max_nodes_per_route, bool is_tsp, rmm::cuda_stream_view stream)
  {
    node_info.resize(max_nodes_per_route, stream);
    if (is_tsp) { tsp_requests.resize(max_nodes_per_route, stream); }
  }

  struct view_t {
    bool is_empty() const { return node_info.empty(); }

    DI request_info_t<i_t, REQUEST> get_node(i_t idx) const
    {
      request_info_t<i_t, REQUEST> request_node;
      request_node.info = node_info[idx];
      return request_node;
    }

    DI void set_node(i_t idx, const request_info_t<i_t, REQUEST>& node)
    {
      node_info[idx] = node.info;
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      auto size = to_idx - from_idx;
      block_copy(node_info.subspan(write_start), orig_route.node_info.subspan(from_idx), size);
    }

    DI i_t node_id(i_t idx) const { return node_info[idx].node(); };

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              i_t n_nodes_route,
                                                              bool is_tsp)
    {
      view_t v;

      v.node_info =
        raft::device_span<NodeInfo<i_t>>{(NodeInfo<i_t>*)shmem, (size_t)n_nodes_route + 1};
      size_t sz_aligned =
        raft::alignTo((size_t)(n_nodes_route + 1), sizeof(double) / sizeof(NodeInfo<i_t>));
      i_t* sh_ptr = (i_t*)&v.node_info.data()[sz_aligned];
      if (is_tsp) {
        thrust::tie(v.tsp_requests, sh_ptr) =
          tsp_route_t<i_t, f_t>::view_t::create_shared_route(sh_ptr, n_nodes_route);
      }
      return thrust::make_tuple(v, sh_ptr);
    }

    raft::device_span<NodeInfo<i_t>> node_info;
    typename tsp_route_t<i_t, f_t>::view_t tsp_requests;
  };

  view_t view()
  {
    view_t v;
    v.node_info    = raft::device_span<NodeInfo<i_t>>{node_info.data(), node_info.size()};
    v.tsp_requests = tsp_requests.view();
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a request route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size, bool is_tsp = false)
  {
    // node, brother
    size_t byte_size = request_info_t<i_t, REQUEST>::size() * route_size * sizeof(NodeInfo<i_t>);
    if (is_tsp) { byte_size += tsp_route_t<i_t, f_t>::get_shared_size(route_size); }
    return raft::alignTo(byte_size, sizeof(double));
  }

  rmm::device_uvector<NodeInfo<i_t>> node_info;
  tsp_route_t<i_t, f_t> tsp_requests;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
