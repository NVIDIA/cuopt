/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/break_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class break_route_t {
 public:
  break_route_t(solution_handle_t<i_t, f_t> const* sol_handle_, break_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      breaks_forward(0, sol_handle_->get_stream()),
      breaks_backward(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero break_route_t copy_ctr");
  }

  break_route_t(const break_route_t& break_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(break_route.dim_info),
      breaks_forward(break_route.breaks_forward, sol_handle_->get_stream()),
      breaks_backward(break_route.breaks_backward, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("break route copy_ctr");
  }

  break_route_t& operator=(break_route_t&& break_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    breaks_forward.resize(max_nodes_per_route, stream);
    breaks_backward.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return breaks_forward.empty(); }
    DI break_node_t<i_t, f_t> get_node(i_t idx) const
    {
      break_node_t<i_t, f_t> break_node;
      break_node.breaks_forward  = breaks_forward[idx];
      break_node.breaks_backward = breaks_backward[idx];
      return break_node;
    }

    DI void set_node(i_t idx, const break_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const break_node_t<i_t, f_t>& node)
    {
      breaks_forward[idx] = node.breaks_forward;
    }

    DI void set_backward_data(i_t idx, const break_node_t<i_t, f_t>& node)
    {
      breaks_backward[idx] = node.breaks_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        breaks_forward.subspan(write_start), orig_route.breaks_forward.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        breaks_backward.subspan(write_start), orig_route.breaks_backward.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      // there is no fixed route data associated with break
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      double infeasibility_cost = 0.;
      if (dim_info.has_breaks) {
        infeasibility_cost =
          max(0., (double)(breaks_forward[n_nodes_route] - vehicle_info.num_breaks()));
      }
      inf_cost[dim_t::BREAK] = infeasibility_cost;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const break_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      v.dim_info        = dim_info;
      v.breaks_forward  = raft::device_span<int>{(int*)shmem, (size_t)n_nodes_route + 1};
      v.breaks_backward = raft::device_span<int>{(int*)&v.breaks_forward.data()[n_nodes_route + 1],
                                                 (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.breaks_backward.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    break_dimension_info_t dim_info;
    raft::device_span<int> breaks_forward;
    raft::device_span<int> breaks_backward;
  };

  view_t view()
  {
    view_t v;
    v.dim_info        = dim_info;
    v.breaks_forward  = raft::device_span<int>{breaks_forward.data(), breaks_forward.size()};
    v.breaks_backward = raft::device_span<int>{breaks_backward.data(), breaks_backward.size()};
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a break route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] break_dimension_info_t dim_info)
  {
    // forward, backward
    return 2 * route_size * sizeof(int);
  }

  break_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<int> breaks_forward;
  // backward data
  rmm::device_uvector<int> breaks_backward;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
