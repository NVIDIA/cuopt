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
#include "../node/tasks_node.cuh"
#include "../solution/solution_handle.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class tasks_route_t {
 public:
  tasks_route_t(solution_handle_t<i_t, f_t> const* sol_handle_, tasks_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      tasks_forward(0, sol_handle_->get_stream()),
      tasks_backward(0, sol_handle_->get_stream())
  {
  }

  tasks_route_t(const tasks_route_t& tasks_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(tasks_route.dim_info),
      tasks_forward(tasks_route.tasks_forward, sol_handle_->get_stream()),
      tasks_backward(tasks_route.tasks_backward, sol_handle_->get_stream())
  {
  }

  tasks_route_t& operator=(tasks_route_t&& tasks_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    tasks_forward.resize(max_nodes_per_route, stream);
    tasks_backward.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return tasks_forward.empty(); }
    DI tasks_node_t<i_t, f_t> get_node(i_t idx) const
    {
      tasks_node_t<i_t, f_t> tasks_node;
      tasks_node.tasks_forward  = tasks_forward[idx];
      tasks_node.tasks_backward = tasks_backward[idx];
      return tasks_node;
    }

    DI void set_node(i_t idx, const tasks_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const tasks_node_t<i_t, f_t>& node)
    {
      tasks_forward[idx] = node.tasks_forward;
    }

    DI void set_backward_data(i_t idx, const tasks_node_t<i_t, f_t>& node)
    {
      tasks_backward[idx] = node.tasks_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        tasks_forward.subspan(write_start), orig_route.tasks_forward.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        tasks_backward.subspan(write_start), orig_route.tasks_backward.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      // there is no fixed route data associated with tasks
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      double diff = tasks_forward[n_nodes_route] - dim_info.mean_tasks;
      obj_cost[objective_t::VARIANCE_ROUTE_SIZE] = diff * diff;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const tasks_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      v.dim_info       = dim_info;
      v.tasks_forward  = raft::device_span<i_t>{shmem, (size_t)n_nodes_route + 1};
      v.tasks_backward = raft::device_span<i_t>{&v.tasks_forward.data()[n_nodes_route + 1],
                                                (size_t)n_nodes_route + 1};

      i_t* sh_ptr = &v.tasks_backward.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    tasks_dimension_info_t dim_info;
    raft::device_span<i_t> tasks_forward;
    raft::device_span<i_t> tasks_backward;
  };

  view_t view()
  {
    view_t v;
    v.dim_info       = dim_info;
    v.tasks_forward  = raft::device_span<i_t>{tasks_forward.data(), tasks_forward.size()};
    v.tasks_backward = raft::device_span<i_t>{tasks_backward.data(), tasks_backward.size()};
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a distance route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] tasks_dimension_info_t dim_info)
  {
    // forward, backward
    return 2 * route_size * sizeof(i_t);
  }

  tasks_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<i_t> tasks_forward;
  // backward data
  rmm::device_uvector<i_t> tasks_backward;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
