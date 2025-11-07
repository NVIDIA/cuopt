/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/prize_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class prize_route_t {
 public:
  prize_route_t(solution_handle_t<i_t, f_t> const* sol_handle_, prize_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      prize(0, sol_handle_->get_stream()),
      prize_forward(0, sol_handle_->get_stream()),
      prize_backward(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero prize_route_t copy_ctr");
  }

  prize_route_t(const prize_route_t& prize_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(prize_route.dim_info),
      prize(prize_route.prize, sol_handle_->get_stream()),
      prize_forward(prize_route.prize_forward, sol_handle_->get_stream()),
      prize_backward(prize_route.prize_backward, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("distance route copy_ctr");
  }

  prize_route_t& operator=(prize_route_t&& prize_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    prize.resize(max_nodes_per_route, stream);
    prize_forward.resize(max_nodes_per_route, stream);
    prize_backward.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return prize_forward.empty(); }
    DI prize_node_t<i_t, f_t> get_node(i_t idx) const
    {
      prize_node_t<i_t, f_t> prize_node;
      prize_node.prize          = prize[idx];
      prize_node.prize_forward  = prize_forward[idx];
      prize_node.prize_backward = prize_backward[idx];
      return prize_node;
    }

    DI void set_node(i_t idx, const prize_node_t<i_t, f_t>& node)
    {
      prize[idx] = node.prize;
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const prize_node_t<i_t, f_t>& node)
    {
      prize_forward[idx] = node.prize_forward;
    }

    DI void set_backward_data(i_t idx, const prize_node_t<i_t, f_t>& node)
    {
      prize_backward[idx] = node.prize_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        prize_forward.subspan(write_start), orig_route.prize_forward.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        prize_backward.subspan(write_start), orig_route.prize_backward.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t start_idx,
                                  i_t end_idx,
                                  i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(prize.subspan(write_start), orig_route.prize.subspan(start_idx), size);
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      obj_cost[objective_t::PRIZE] = -prize_forward[n_nodes_route];
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const prize_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      v.dim_info       = dim_info;
      v.prize          = raft::device_span<double>{(double*)shmem, (size_t)n_nodes_route + 1};
      v.prize_forward  = raft::device_span<double>{(double*)&v.prize.data()[n_nodes_route + 1],
                                                   (size_t)n_nodes_route + 1};
      v.prize_backward = raft::device_span<double>{
        (double*)&v.prize_forward.data()[n_nodes_route + 1], (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.prize_backward.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    prize_dimension_info_t dim_info;
    raft::device_span<double> prize;
    raft::device_span<double> prize_forward;
    raft::device_span<double> prize_backward;
  };

  view_t view()
  {
    view_t v;
    v.dim_info       = dim_info;
    v.prize          = raft::device_span<double>{prize.data(), prize.size()};
    v.prize_forward  = raft::device_span<double>{prize_forward.data(), prize_forward.size()};
    v.prize_backward = raft::device_span<double>{prize_backward.data(), prize_backward.size()};
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a distance route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] prize_dimension_info_t dim_info)
  {
    // prize, prize_forward, prize_backward
    return 3 * route_size * sizeof(double);
  }

  prize_dimension_info_t dim_info;

  // prize data of nodes in route
  rmm::device_uvector<double> prize;
  // forward data
  rmm::device_uvector<double> prize_forward;
  // backward data
  rmm::device_uvector<double> prize_backward;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
