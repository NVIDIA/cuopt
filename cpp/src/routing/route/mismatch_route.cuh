/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/mismatch_node.cuh"
#include "../solution/solution_handle.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

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
      mismatch_backward(0, sol_handle_->get_stream())
  {
  }

  mismatch_route_t(const mismatch_route_t& mismatch_route,
                   solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(mismatch_route.dim_info),
      mismatch_forward(mismatch_route.mismatch_forward, sol_handle_->get_stream()),
      mismatch_backward(mismatch_route.mismatch_backward, sol_handle_->get_stream())
  {
  }

  mismatch_route_t& operator=(mismatch_route_t&& mismatch_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    mismatch_forward.resize(max_nodes_per_route, stream);
    mismatch_backward.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return mismatch_forward.empty(); }
    DI mismatch_node_t<i_t, f_t> get_node(i_t idx) const
    {
      mismatch_node_t<i_t, f_t> mismatch_node;
      mismatch_node.mismatch_forward  = mismatch_forward[idx];
      mismatch_node.mismatch_backward = mismatch_backward[idx];
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
    }

    DI void set_backward_data(i_t idx, const mismatch_node_t<i_t, f_t>& node)
    {
      mismatch_backward[idx] = node.mismatch_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(mismatch_forward.subspan(write_start),
                 orig_route.mismatch_forward.subspan(start_idx),
                 size);
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
      inf_cost[dim_t::MISMATCH] = mismatch_forward[n_nodes_route];
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const mismatch_dimension_info_t dim_info, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info          = dim_info;
      v.mismatch_forward  = raft::device_span<i_t>{(i_t*)shmem, (size_t)n_nodes_route + 1};
      v.mismatch_backward = raft::device_span<i_t>{
        (i_t*)&v.mismatch_forward.data()[n_nodes_route + 1], (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.mismatch_backward.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    mismatch_dimension_info_t dim_info;
    raft::device_span<i_t> mismatch_forward;
    raft::device_span<i_t> mismatch_backward;
  };

  view_t view()
  {
    view_t v;
    v.dim_info         = dim_info;
    v.mismatch_forward = raft::device_span<i_t>{mismatch_forward.data(), mismatch_forward.size()};
    v.mismatch_backward =
      raft::device_span<i_t>{mismatch_backward.data(), mismatch_backward.size()};
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a distance route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] mismatch_dimension_info_t dim_info)
  {
    // forward, backward
    return 2 * route_size * sizeof(i_t);
  }

  mismatch_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<i_t> mismatch_forward;
  // backward data
  rmm::device_uvector<i_t> mismatch_backward;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
