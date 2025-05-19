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
#include "../node/distance_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class distance_route_t {
 public:
  distance_route_t(solution_handle_t<i_t, f_t> const* sol_handle_, cost_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      distance_forward(0, sol_handle_->get_stream()),
      distance_backward(0, sol_handle_->get_stream()),
      reverse_distance(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero distance_route_t copy_ctr");
  }

  distance_route_t(const distance_route_t& distance_route,
                   solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(distance_route.dim_info),
      distance_forward(distance_route.distance_forward, sol_handle_->get_stream()),
      distance_backward(distance_route.distance_backward, sol_handle_->get_stream()),
      reverse_distance(distance_route.reverse_distance, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("distance route copy_ctr");
  }

  distance_route_t& operator=(distance_route_t&& distance_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    distance_forward.resize(max_nodes_per_route, stream);
    distance_backward.resize(max_nodes_per_route, stream);
    reverse_distance.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return distance_forward.empty(); }
    DI distance_node_t<i_t, f_t> get_node(i_t idx) const
    {
      distance_node_t<i_t, f_t> distance_node;
      distance_node.distance_forward  = distance_forward[idx];
      distance_node.distance_backward = distance_backward[idx];
      return distance_node;
    }

    DI void set_node(i_t idx, const distance_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const distance_node_t<i_t, f_t>& node)
    {
      distance_forward[idx] = node.distance_forward;
    }

    DI void set_backward_data(i_t idx, const distance_node_t<i_t, f_t>& node)
    {
      distance_backward[idx] = node.distance_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(distance_forward.subspan(write_start),
                 orig_route.distance_forward.subspan(start_idx),
                 size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(distance_backward.subspan(write_start),
                 orig_route.distance_backward.subspan(start_idx),
                 size);
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      // there is no fixed route data associated with distance
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      double objective_cost     = distance_forward[n_nodes_route];
      double infeasibility_cost = 0.;
      if (dim_info.has_max_constraint) {
        infeasibility_cost = max(0., distance_forward[n_nodes_route] - vehicle_info.max_cost);
      }

      obj_cost[objective_t::COST] = objective_cost;
      inf_cost[dim_t::DIST]       = infeasibility_cost;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const cost_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      v.dim_info          = dim_info;
      v.distance_forward  = raft::device_span<double>{(double*)shmem, (size_t)n_nodes_route + 1};
      v.distance_backward = raft::device_span<double>{
        (double*)&v.distance_forward.data()[n_nodes_route + 1], (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.distance_backward.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    cost_dimension_info_t dim_info;
    raft::device_span<double> distance_forward;
    raft::device_span<double> distance_backward;
    raft::device_span<double> reverse_distance;
  };

  view_t view()
  {
    view_t v;
    v.dim_info = dim_info;
    v.distance_forward =
      raft::device_span<double>{distance_forward.data(), distance_forward.size()};
    v.distance_backward =
      raft::device_span<double>{distance_backward.data(), distance_backward.size()};
    v.reverse_distance =
      raft::device_span<double>{reverse_distance.data(), reverse_distance.size()};
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a distance route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] cost_dimension_info_t dim_info,
                                    [[maybe_unused]] bool is_tsp = false)
  {
    // forward, backward
    return 2 * route_size * sizeof(double);
  }

  cost_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<double> distance_forward;
  // backward data
  rmm::device_uvector<double> distance_backward;
  // The info is not updated with the other dimension buffers.
  // It is only used for cvrp/tsp and populated in global memory.
  rmm::device_uvector<double> reverse_distance;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
