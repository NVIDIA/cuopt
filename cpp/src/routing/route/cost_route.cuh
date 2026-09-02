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
      reverse_cost(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero cost_route_t copy_ctr");
  }

  cost_route_t(const cost_route_t& cost_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(cost_route.dim_info),
      cost_forward(cost_route.cost_forward, sol_handle_->get_stream()),
      cost_backward(cost_route.cost_backward, sol_handle_->get_stream()),
      reverse_cost(cost_route.reverse_cost, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("cost route copy_ctr");
  }

  cost_route_t& operator=(cost_route_t&& cost_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    cost_forward.resize(max_nodes_per_route, stream);
    cost_backward.resize(max_nodes_per_route, stream);
    reverse_cost.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return cost_forward.empty(); }
    DI cost_node_t<i_t, f_t> get_node(i_t idx) const
    {
      cost_node_t<i_t, f_t> cost_node;
      cost_node.cost_forward  = cost_forward[idx];
      cost_node.cost_backward = cost_backward[idx];
      return cost_node;
    }

    DI void set_node(i_t idx, const cost_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const cost_node_t<i_t, f_t>& node)
    {
      cost_forward[idx] = node.cost_forward;
    }

    DI void set_backward_data(i_t idx, const cost_node_t<i_t, f_t>& node)
    {
      cost_backward[idx] = node.cost_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        cost_forward.subspan(write_start), orig_route.cost_forward.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        cost_backward.subspan(write_start), orig_route.cost_backward.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      // there is no fixed route data associated with cost
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      double objective_cost     = cost_forward[n_nodes_route];
      double infeasibility_cost = 0.;
      if (dim_info.has_max_constraint) {
        infeasibility_cost = max(0., cost_forward[n_nodes_route] - vehicle_info.max_cost);
      }

      obj_cost[objective_t::COST] = objective_cost;
      inf_cost[dim_t::COST]       = infeasibility_cost;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const cost_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      v.dim_info      = dim_info;
      v.cost_forward  = raft::device_span<double>{(double*)shmem, (size_t)n_nodes_route + 1};
      v.cost_backward = raft::device_span<double>{
        (double*)&v.cost_forward.data()[n_nodes_route + 1], (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.cost_backward.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    cost_dimension_info_t dim_info;
    raft::device_span<double> cost_forward;
    raft::device_span<double> cost_backward;
    raft::device_span<double> reverse_cost;
  };

  view_t view()
  {
    view_t v;
    v.dim_info      = dim_info;
    v.cost_forward  = raft::device_span<double>{cost_forward.data(), cost_forward.size()};
    v.cost_backward = raft::device_span<double>{cost_backward.data(), cost_backward.size()};
    v.reverse_cost  = raft::device_span<double>{reverse_cost.data(), reverse_cost.size()};
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
    // forward, backward
    return 2 * route_size * sizeof(double);
  }

  cost_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<double> cost_forward;
  // backward data
  rmm::device_uvector<double> cost_backward;
  // The info is not updated with the other dimension buffers.
  // It is only used for cvrp/tsp and populated in global memory.
  rmm::device_uvector<double> reverse_cost;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
