/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/mismatch_node.cuh"
#include "../solution/solution_handle.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class mismatch_route_t {
 public:
  mismatch_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                   mismatch_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      mismatch_forward(0, sol_handle_->get_stream()),
      mismatch_backward(0, sol_handle_->get_stream()),
      cost_forward(0, sol_handle_->get_stream()),
      cost_backward(0, sol_handle_->get_stream())
  {
  }

  mismatch_route_t(const mismatch_route_t& mismatch_route,
                   solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(mismatch_route.dim_info),
      mismatch_forward(mismatch_route.mismatch_forward, sol_handle_->get_stream()),
      mismatch_backward(mismatch_route.mismatch_backward, sol_handle_->get_stream()),
      cost_forward(mismatch_route.cost_forward, sol_handle_->get_stream()),
      cost_backward(mismatch_route.cost_backward, sol_handle_->get_stream())
  {
  }

  mismatch_route_t& operator=(mismatch_route_t&& mismatch_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    mismatch_forward.resize(max_nodes_per_route, stream);
    mismatch_backward.resize(max_nodes_per_route, stream);
    cost_forward.resize(max_nodes_per_route, stream);
    cost_backward.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return mismatch_forward.empty(); }
    DI mismatch_node_t<i_t, f_t> get_node(i_t idx) const
    {
      mismatch_node_t<i_t, f_t> mismatch_node;
      mismatch_node.mismatch_forward  = mismatch_forward[idx];
      mismatch_node.mismatch_backward = mismatch_backward[idx];
      mismatch_node.cost_forward      = cost_forward[idx];
      mismatch_node.cost_backward     = cost_backward[idx];
      return mismatch_node;
    }

    DI void set_node(i_t idx, const mismatch_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const mismatch_node_t<i_t, f_t>& node)
    {
      mismatch_forward[idx] = node.mismatch_forward;
      cost_forward[idx]     = node.cost_forward;
    }

    DI void set_backward_data(i_t idx, const mismatch_node_t<i_t, f_t>& node)
    {
      mismatch_backward[idx] = node.mismatch_backward;
      cost_backward[idx]     = node.cost_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(mismatch_forward.subspan(write_start),
                 orig_route.mismatch_forward.subspan(start_idx),
                 size);
      block_copy(
        cost_forward.subspan(write_start), orig_route.cost_forward.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(mismatch_backward.subspan(write_start),
                 orig_route.mismatch_backward.subspan(start_idx),
                 size);
      block_copy(
        cost_backward.subspan(write_start), orig_route.cost_backward.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      // there is no fixed route data associated with mismatch
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      inf_cost[dim_t::MISMATCH] = mismatch_forward[n_nodes_route];
      if (dim_info.has_vehicle_order_cost) {
        obj_cost[objective_t::VEHICLE_ORDER_COST] = cost_forward[n_nodes_route];
      }
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const mismatch_dimension_info_t dim_info_, i_t n_nodes_route)
    {
      view_t v;
      i_t* sh_ptr                              = shmem;
      v.dim_info                               = dim_info_;
      thrust::tie(v.mismatch_forward, sh_ptr)  = wrap_ptr_as_span<i_t>(sh_ptr, n_nodes_route + 1);
      thrust::tie(v.mismatch_backward, sh_ptr) = wrap_ptr_as_span<i_t>(sh_ptr, n_nodes_route + 1);
      thrust::tie(v.cost_forward, sh_ptr)  = wrap_ptr_as_span<double>(sh_ptr, n_nodes_route + 1);
      thrust::tie(v.cost_backward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, n_nodes_route + 1);
      return thrust::make_tuple(v, sh_ptr);
    }

    mismatch_dimension_info_t dim_info;
    raft::device_span<i_t> mismatch_forward;
    raft::device_span<i_t> mismatch_backward;
    raft::device_span<double> cost_forward;
    raft::device_span<double> cost_backward;
  };

  view_t view()
  {
    view_t v;
    v.dim_info         = dim_info;
    v.mismatch_forward = raft::device_span<i_t>{mismatch_forward.data(), mismatch_forward.size()};
    v.mismatch_backward =
      raft::device_span<i_t>{mismatch_backward.data(), mismatch_backward.size()};
    v.cost_forward  = raft::device_span<double>{cost_forward.data(), cost_forward.size()};
    v.cost_backward = raft::device_span<double>{cost_backward.data(), cost_backward.size()};
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a mismatch route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] mismatch_dimension_info_t dim_info_)
  {
    // 2 i_t arrays (mismatch_forward, mismatch_backward) + 2 double arrays (cost_forward,
    // cost_backward)
    return 2 * (route_size + 1) * sizeof(i_t) + 2 * (route_size + 1) * sizeof(double);
  }

  mismatch_dimension_info_t dim_info;

  rmm::device_uvector<i_t> mismatch_forward;
  rmm::device_uvector<i_t> mismatch_backward;
  rmm::device_uvector<double> cost_forward;
  rmm::device_uvector<double> cost_backward;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
