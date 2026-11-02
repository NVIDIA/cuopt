/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/cost_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <cuda/stream>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class cost_route_t {
 public:
  cost_route_t(solution_handle_t<i_t, f_t> const* sol_handle_, cost_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      cost_forward(0, sol_handle_->get_stream()),
      cost_backward(0, sol_handle_->get_stream()),
      reverse_cost(0, sol_handle_->get_stream()),
      distance_window_forward(0, sol_handle_->get_stream()),
      distance_window_backward(0, sol_handle_->get_stream()),
      distance_window_backward_min(0, sol_handle_->get_stream()),
      window_start(0, sol_handle_->get_stream()),
      window_end(0, sol_handle_->get_stream()),
      excess_forward(0, sol_handle_->get_stream()),
      excess_backward(0, sol_handle_->get_stream()),
      distance_break_cost_forward(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero cost_route_t copy_ctr");
  }

  cost_route_t(const cost_route_t& cost_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(cost_route.dim_info),
      cost_forward(cost_route.cost_forward, sol_handle_->get_stream()),
      cost_backward(cost_route.cost_backward, sol_handle_->get_stream()),
      reverse_cost(cost_route.reverse_cost, sol_handle_->get_stream()),
      distance_window_forward(cost_route.distance_window_forward, sol_handle_->get_stream()),
      distance_window_backward(cost_route.distance_window_backward, sol_handle_->get_stream()),
      distance_window_backward_min(cost_route.distance_window_backward_min,
                                   sol_handle_->get_stream()),
      window_start(cost_route.window_start, sol_handle_->get_stream()),
      window_end(cost_route.window_end, sol_handle_->get_stream()),
      excess_forward(cost_route.excess_forward, sol_handle_->get_stream()),
      excess_backward(cost_route.excess_backward, sol_handle_->get_stream()),
      distance_break_cost_forward(cost_route.distance_break_cost_forward, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("cost route copy_ctr");
  }

  cost_route_t& operator=(cost_route_t&& cost_route) = default;

  void resize(i_t max_nodes_per_route, cuda::stream_ref stream)
  {
    cost_forward.resize(max_nodes_per_route, stream);
    cost_backward.resize(max_nodes_per_route, stream);
    reverse_cost.resize(max_nodes_per_route, stream);
    if (dim_info.has_distance_window) {
      distance_window_forward.resize(max_nodes_per_route, stream);
      distance_window_backward.resize(max_nodes_per_route, stream);
      window_start.resize(max_nodes_per_route, stream);
      window_end.resize(max_nodes_per_route, stream);
      excess_forward.resize(max_nodes_per_route, stream);
      excess_backward.resize(max_nodes_per_route, stream);
      if (dim_info.has_distance_break_cost) {
        distance_window_backward_min.resize(max_nodes_per_route, stream);
        distance_break_cost_forward.resize(max_nodes_per_route, stream);
      }
    }
  }

  struct view_t {
    bool is_empty() const { return cost_forward.empty(); }
    DI cost_node_t<i_t, f_t> get_node(i_t idx) const
    {
      cost_node_t<i_t, f_t> cost_node;
      cost_node.cost_forward  = cost_forward[idx];
      cost_node.cost_backward = cost_backward[idx];
      if (dim_info.has_distance_window) {
        cost_node.distance_window_forward  = distance_window_forward[idx];
        cost_node.distance_window_backward = distance_window_backward[idx];
        cost_node.window_start             = window_start[idx];
        cost_node.window_end               = window_end[idx];
        cost_node.excess_forward           = excess_forward[idx];
        cost_node.excess_backward          = excess_backward[idx];
        if (dim_info.has_distance_break_cost) {
          cost_node.distance_window_backward_min = distance_window_backward_min[idx];
          cost_node.distance_break_cost_forward  = distance_break_cost_forward[idx];
        }
      }
      return cost_node;
    }

    DI void set_node(i_t idx, const cost_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
      if (dim_info.has_distance_window) {
        window_start[idx] = node.window_start;
        window_end[idx]   = node.window_end;
      }
    }

    DI void set_forward_data(i_t idx, const cost_node_t<i_t, f_t>& node)
    {
      cost_forward[idx] = node.cost_forward;
      if (dim_info.has_distance_window) {
        distance_window_forward[idx] = node.distance_window_forward;
        excess_forward[idx]          = node.excess_forward;
        if (dim_info.has_distance_break_cost) {
          distance_break_cost_forward[idx] = node.distance_break_cost_forward;
        }
      }
    }

    DI void set_backward_data(i_t idx, const cost_node_t<i_t, f_t>& node)
    {
      cost_backward[idx] = node.cost_backward;
      if (dim_info.has_distance_window) {
        distance_window_backward[idx] = node.distance_window_backward;
        excess_backward[idx]          = node.excess_backward;
        if (dim_info.has_distance_break_cost) {
          distance_window_backward_min[idx] = node.distance_window_backward_min;
        }
      }
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        cost_forward.subspan(write_start), orig_route.cost_forward.subspan(start_idx), size);
      if (dim_info.has_distance_window) {
        block_copy(distance_window_forward.subspan(write_start),
                   orig_route.distance_window_forward.subspan(start_idx),
                   size);
        block_copy(
          excess_forward.subspan(write_start), orig_route.excess_forward.subspan(start_idx), size);
        if (dim_info.has_distance_break_cost) {
          block_copy(distance_break_cost_forward.subspan(write_start),
                     orig_route.distance_break_cost_forward.subspan(start_idx),
                     size);
        }
      }
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        cost_backward.subspan(write_start), orig_route.cost_backward.subspan(start_idx), size);
      if (dim_info.has_distance_window) {
        block_copy(distance_window_backward.subspan(write_start),
                   orig_route.distance_window_backward.subspan(start_idx),
                   size);
        block_copy(excess_backward.subspan(write_start),
                   orig_route.excess_backward.subspan(start_idx),
                   size);
        if (dim_info.has_distance_break_cost) {
          block_copy(distance_window_backward_min.subspan(write_start),
                     orig_route.distance_window_backward_min.subspan(start_idx),
                     size);
        }
      }
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      if (dim_info.has_distance_window) {
        auto size = to_idx - from_idx;
        block_copy(
          window_start.subspan(write_start), orig_route.window_start.subspan(from_idx), size);
        block_copy(window_end.subspan(write_start), orig_route.window_end.subspan(from_idx), size);
      }
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      obj_cost[objective_t::COST] = cost_forward[n_nodes_route];
      if (dim_info.has_distance_window && dim_info.has_distance_break_cost) {
        obj_cost[objective_t::DISTANCE_BREAK_COST] = distance_break_cost_forward[n_nodes_route];
      }

      inf_cost[dim_t::COST] = 0.;
      if (dim_info.has_max_constraint) {
        inf_cost[dim_t::COST] = max(0., cost_forward[n_nodes_route] - vehicle_info.max_cost);
      }
      if (dim_info.has_distance_window) { inf_cost[dim_t::COST] += excess_forward[n_nodes_route]; }
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const cost_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      size_t sz                            = n_nodes_route + 1;
      i_t* sh_ptr                          = shmem;
      v.dim_info                           = dim_info;
      thrust::tie(v.cost_forward, sh_ptr)  = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.cost_backward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
      if (dim_info.has_distance_window) {
        thrust::tie(v.distance_window_forward, sh_ptr)  = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.distance_window_backward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.window_start, sh_ptr)             = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.window_end, sh_ptr)               = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.excess_forward, sh_ptr)           = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.excess_backward, sh_ptr)          = wrap_ptr_as_span<double>(sh_ptr, sz);
        if (dim_info.has_distance_break_cost) {
          thrust::tie(v.distance_window_backward_min, sh_ptr) =
            wrap_ptr_as_span<double>(sh_ptr, sz);
          thrust::tie(v.distance_break_cost_forward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
        }
      }
      return thrust::make_tuple(v, sh_ptr);
    }

    cost_dimension_info_t dim_info;
    raft::device_span<double> cost_forward;
    raft::device_span<double> cost_backward;
    raft::device_span<double> reverse_cost;
    raft::device_span<double> distance_window_forward;
    raft::device_span<double> distance_window_backward;
    raft::device_span<double> distance_window_backward_min;
    raft::device_span<double> window_start;
    raft::device_span<double> window_end;
    raft::device_span<double> excess_forward;
    raft::device_span<double> excess_backward;
    raft::device_span<double> distance_break_cost_forward;
  };

  view_t view()
  {
    view_t v;
    v.dim_info      = dim_info;
    v.cost_forward  = raft::device_span<double>{cost_forward.data(), cost_forward.size()};
    v.cost_backward = raft::device_span<double>{cost_backward.data(), cost_backward.size()};
    v.reverse_cost  = raft::device_span<double>{reverse_cost.data(), reverse_cost.size()};
    if (dim_info.has_distance_window) {
      v.distance_window_forward =
        raft::device_span<double>{distance_window_forward.data(), distance_window_forward.size()};
      v.distance_window_backward =
        raft::device_span<double>{distance_window_backward.data(), distance_window_backward.size()};
      v.window_start    = raft::device_span<double>{window_start.data(), window_start.size()};
      v.window_end      = raft::device_span<double>{window_end.data(), window_end.size()};
      v.excess_forward  = raft::device_span<double>{excess_forward.data(), excess_forward.size()};
      v.excess_backward = raft::device_span<double>{excess_backward.data(), excess_backward.size()};
      if (dim_info.has_distance_break_cost) {
        v.distance_window_backward_min = raft::device_span<double>{
          distance_window_backward_min.data(), distance_window_backward_min.size()};
        v.distance_break_cost_forward = raft::device_span<double>{
          distance_break_cost_forward.data(), distance_break_cost_forward.size()};
      }
    }
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a cost route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] cost_dimension_info_t dim_info,
                                    [[maybe_unused]] bool is_tsp = false)
  {
    return (2 + 6 * dim_info.has_distance_window +
            2 * (dim_info.has_distance_window && dim_info.has_distance_break_cost)) *
           route_size * sizeof(double);
  }

  cost_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<double> cost_forward;
  // backward data
  rmm::device_uvector<double> cost_backward;
  // The info is not updated with the other dimension buffers.
  // It is only used for cvrp/tsp and populated in global memory.
  rmm::device_uvector<double> reverse_cost;
  // Allocated only when has_distance_window.
  rmm::device_uvector<double> distance_window_forward;
  rmm::device_uvector<double> distance_window_backward;
  // Allocated only when has_distance_break_cost.
  rmm::device_uvector<double> distance_window_backward_min;
  rmm::device_uvector<double> window_start;
  rmm::device_uvector<double> window_end;
  rmm::device_uvector<double> excess_forward;
  rmm::device_uvector<double> excess_backward;
  rmm::device_uvector<double> distance_break_cost_forward;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
