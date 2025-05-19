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

#include "break_route.cuh"
#include "capacity_route.cuh"
#include "distance_route.cuh"
#include "mismatch_route.cuh"
#include "pdp_route.cuh"
#include "prize_route.cuh"
#include "service_time_route.cuh"
#include "tasks_route.cuh"
#include "time_route.cuh"
#include "vehicle_fixed_cost_route.cuh"

#include "../node/node.cuh"
#include "../solution/route_node_map.cuh"
#include "../solution/solution_handle.cuh"

#include "../routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {
template <typename i_t, typename f_t, request_t REQUEST>
class route_t;

template <size_t I, typename i_t, typename f_t>
using route_from_dim = typename std::conditional<
  ((dim_t)I == dim_t::TIME),
  time_route_t<i_t, f_t>,
  typename std::conditional<
    ((dim_t)I == dim_t::DIST),
    distance_route_t<i_t, f_t>,
    typename std::conditional<
      ((dim_t)I == dim_t::CAP),
      capacity_route_t<i_t, f_t>,
      typename std::conditional<
        ((dim_t)I == dim_t::PRIZE),
        prize_route_t<i_t, f_t>,
        typename std::conditional<
          ((dim_t)I == dim_t::TASKS),
          tasks_route_t<i_t, f_t>,
          typename std::conditional<
            ((dim_t)I == dim_t::SERVICE_TIME),
            service_time_route_t<i_t, f_t>,
            typename std::conditional<
              ((dim_t)I == dim_t::MISMATCH),
              mismatch_route_t<i_t, f_t>,
              typename std::conditional<((dim_t)I == dim_t::BREAK),
                                        break_route_t<i_t, f_t>,
                                        vehicle_fixed_cost_route_t<i_t, f_t>>::type>::type>::type>::
          type>::type>::type>::type>::type;
template <typename i_t, typename f_t, request_t REQUEST>
class dimensions_route_t {
 public:
  dimensions_route_t() = delete;
  dimensions_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                     enabled_dimensions_t dimensions_info_)
    : sol_handle(sol_handle_),
      time_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::TIME>()),
      capacity_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::CAP>()),
      distance_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::DIST>()),
      prize_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::PRIZE>()),
      tasks_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::TASKS>()),
      service_time_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::SERVICE_TIME>()),
      mismatch_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::MISMATCH>()),
      break_dim(sol_handle_, dimensions_info_.get_dimension<dim_t::BREAK>()),
      vehicle_fixed_cost_dim(sol_handle_,
                             dimensions_info_.get_dimension<dim_t::VEHICLE_FIXED_COST>()),
      requests(sol_handle_),
      dimensions_info(dimensions_info_)
  {
    raft::common::nvtx::range fun_scope("zero dimensions_route_t copy_ctr");
  }

  dimensions_route_t(const dimensions_route_t& dim_route)
    : sol_handle(dim_route.sol_handle),
      time_dim(dim_route.time_dim, dim_route.sol_handle),
      capacity_dim(dim_route.capacity_dim, dim_route.sol_handle),
      distance_dim(dim_route.distance_dim, dim_route.sol_handle),
      prize_dim(dim_route.prize_dim, dim_route.sol_handle),
      tasks_dim(dim_route.tasks_dim, dim_route.sol_handle),
      service_time_dim(dim_route.service_time_dim, dim_route.sol_handle),
      mismatch_dim(dim_route.mismatch_dim, dim_route.sol_handle),
      break_dim(dim_route.break_dim, dim_route.sol_handle),
      vehicle_fixed_cost_dim(dim_route.vehicle_fixed_cost_dim, dim_route.sol_handle),
      requests(dim_route.requests, dim_route.sol_handle),
      dimensions_info(dim_route.dimensions_info)
  {
    raft::common::nvtx::range fun_scope("dimensions_route_t copy_ctr");
  }

  dimensions_route_t& operator=(dimensions_route_t&& route) = default;

  // this will be used in device code
  struct view_t {
    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
    DI auto get_request_node(typename route_node_map_t<i_t>::view_t const& route_node_map,
                             request_id_t<r_t> const& request) const
    {
      i_t intra_idx = route_node_map.get_intra_route_idx(request.id());
      auto node     = get_node(intra_idx);
      return request_node_t<i_t, f_t, r_t>(node);
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
    DI auto get_request_node(typename route_node_map_t<i_t>::view_t const& route_node_map,
                             request_id_t<r_t> const& request) const
    {
      i_t pickup_intra   = route_node_map.get_intra_route_idx(request.pickup);
      i_t delivery_intra = route_node_map.get_intra_route_idx(request.delivery);
      auto pickup_node   = get_node(pickup_intra);
      auto delivery_node = get_node(delivery_intra);
      return request_node_t<i_t, f_t, r_t>(pickup_node, delivery_node);
    }

    /**
     * @brief Get the node object from the index
     *
     * @param idx Intra route index
     * @return node_t
     */
    DI node_t<i_t, f_t, REQUEST> get_node(i_t idx) const
    {
      cuopt_assert(idx >= 0, "Get_node should receive a positive index");
      node_t<i_t, f_t, REQUEST> node(dimensions_info);
      node.request = requests.get_node(idx);
      loop_over_dimensions(dimensions_info, [&](auto I) {
        auto& node_dim = get_dimension_of<I>(node);
        node_dim       = get_dimension_of<I>(*this).get_node(idx);
      });
      return node;
    }
    DI void set_node(i_t idx, const node_t<i_t, f_t, REQUEST>& node)
    {
      requests.set_node(idx, node.request);
      loop_over_dimensions(dimensions_info, [&](auto I) {
        auto& node_dim = get_dimension_of<I>(node);
        get_dimension_of<I>(*this).set_node(idx, node_dim);
      });
    }

    DI void parallel_copy_nodes_from(i_t dst_start,
                                     const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                                     i_t src_start,
                                     i_t size,
                                     bool reverse_copy = false)
    {
      cuopt_assert(src_start + size - 1 <= route.get_num_nodes(), "Invalid start_idx or size");
      cuopt_assert(src_start >= 0, "Invalid start_idx");
      for (i_t i = threadIdx.x; i < size; i += blockDim.x) {
        i_t idx = reverse_copy ? size - i - 1 : i;
        set_node(dst_start + idx, route.get_node(src_start + i));
      }
    }

    DI void copy_nodes_from(i_t dst_start,
                            const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                            i_t src_start,
                            i_t size,
                            bool reverse_copy = false)
    {
      cuopt_assert(src_start + size - 1 <= route.get_num_nodes(), "Invalid start_idx or size");
      cuopt_assert(src_start >= 0, "Invalid start_idx");
      for (i_t i = 0; i < size; ++i) {
        i_t idx = reverse_copy ? size - i - 1 : i;
        set_node(dst_start + idx, route.get_node(src_start + i));
      }
    }

    DI i_t node_id(i_t idx) const { return requests.node_id(idx); }
    DI i_t brother_id(i_t idx) const
    {
      if constexpr (REQUEST == request_t::PDP) {
        return requests.brother_id(idx);
      } else {
        return -1;
      }
    }
    DI NodeInfo<i_t> node_info(i_t idx) const { return requests.node_info[idx]; };

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* sh_ptr, const enabled_dimensions_t dimensions_info_, i_t size, bool is_tsp = false)
    {
      view_t v;
      v.dimensions_info = dimensions_info_;
      thrust::tie(v.requests, sh_ptr) =
        std::decay_t<decltype(requests)>::create_shared_route(sh_ptr, size, is_tsp);
      loop_over_dimensions(dimensions_info_, [&] __device__(auto I) {
        auto& dim_route                = get_dimension_of<I>(v);
        thrust::tie(dim_route, sh_ptr) = std::decay_t<decltype(dim_route)>::create_shared_route(
          sh_ptr, get_dimension_of<I>(dimensions_info_), size);
      });
      return thrust::make_tuple(v, sh_ptr);
    }

    typename request_route_t<i_t, f_t, REQUEST>::view_t requests;
    typename distance_route_t<i_t, f_t>::view_t distance_dim;
    typename time_route_t<i_t, f_t>::view_t time_dim;
    typename capacity_route_t<i_t, f_t>::view_t capacity_dim;
    typename prize_route_t<i_t, f_t>::view_t prize_dim;
    typename tasks_route_t<i_t, f_t>::view_t tasks_dim;
    typename service_time_route_t<i_t, f_t>::view_t service_time_dim;
    typename mismatch_route_t<i_t, f_t>::view_t mismatch_dim;
    typename break_route_t<i_t, f_t>::view_t break_dim;
    typename vehicle_fixed_cost_route_t<i_t, f_t>::view_t vehicle_fixed_cost_dim;
    enabled_dimensions_t dimensions_info{};
  };

  view_t view()
  {
    view_t v;
    v.dimensions_info = dimensions_info;
    v.requests        = requests.view();
    loop_over_dimensions(dimensions_info, [&](auto I) {
      auto& dim_view = get_dimension_of<I>(v);
      dim_view       = get_dimension_of<I>(*this).view();
    });
    return v;
  }

  void resize(i_t new_size)
  {
    requests.resize(new_size, dimensions_info.is_tsp, sol_handle->get_stream());
    loop_over_dimensions(dimensions_info, [&](auto I) {
      get_dimension_of<I>(*this).resize(new_size, sol_handle->get_stream());
    });
  }

  /**
   * @brief Get the shared memory size got raw dimension vectors
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size, enabled_dimensions_t dimensions_info)
  {
    size_t sz =
      request_route_t<i_t, f_t, REQUEST>::get_shared_size(route_size, dimensions_info.is_tsp);
    loop_over_dimensions(dimensions_info, [&](auto I) {
      sz += route_from_dim<I, i_t, f_t>::get_shared_size(route_size,
                                                         get_dimension_of<I>(dimensions_info));
    });
    return sz;
  }

  // solution handle
  solution_handle_t<i_t, f_t> const* sol_handle;

  // pdp route
  request_route_t<i_t, f_t, REQUEST> requests;

  // distance route
  distance_route_t<i_t, f_t> distance_dim;

  // time route
  time_route_t<i_t, f_t> time_dim;

  // capacity route
  capacity_route_t<i_t, f_t> capacity_dim;

  // prize route
  prize_route_t<i_t, f_t> prize_dim;

  // tasks route
  tasks_route_t<i_t, f_t> tasks_dim;

  // service time route
  service_time_route_t<i_t, f_t> service_time_dim;

  // mismatch route
  mismatch_route_t<i_t, f_t> mismatch_dim;

  // break route
  break_route_t<i_t, f_t> break_dim;

  // vehicle cost route
  vehicle_fixed_cost_route_t<i_t, f_t> vehicle_fixed_cost_dim;

  // encoded struct to get enabled dimensions info
  enabled_dimensions_t dimensions_info;
};
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
