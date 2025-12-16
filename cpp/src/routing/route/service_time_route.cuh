/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/service_time_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class service_time_route_t {
 public:
  service_time_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                       service_time_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      service_time_forward(0, sol_handle_->get_stream()),
      service_time_backward(0, sol_handle_->get_stream())
  {
  }

  service_time_route_t(const service_time_route_t& service_time_route,
                       solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(service_time_route.dim_info),
      service_time_forward(service_time_route.service_time_forward, sol_handle_->get_stream()),
      service_time_backward(service_time_route.service_time_backward, sol_handle_->get_stream())
  {
  }

  service_time_route_t& operator=(service_time_route_t&& service_time_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    service_time_forward.resize(max_nodes_per_route, stream);
    service_time_backward.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return service_time_forward.empty(); }
    DI service_time_node_t<i_t, f_t> get_node(i_t idx) const
    {
      service_time_node_t<i_t, f_t> service_time_node;
      service_time_node.service_time_forward  = service_time_forward[idx];
      service_time_node.service_time_backward = service_time_backward[idx];
      return service_time_node;
    }

    DI void set_node(i_t idx, const service_time_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const service_time_node_t<i_t, f_t>& node)
    {
      service_time_forward[idx] = node.service_time_forward;
    }

    DI void set_backward_data(i_t idx, const service_time_node_t<i_t, f_t>& node)
    {
      service_time_backward[idx] = node.service_time_backward;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(service_time_forward.subspan(write_start),
                 orig_route.service_time_forward.subspan(start_idx),
                 size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(service_time_backward.subspan(write_start),
                 orig_route.service_time_backward.subspan(start_idx),
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
      double tmp = service_time_forward[n_nodes_route] - dim_info.mean_service_time;
      obj_cost[objective_t::VARIANCE_ROUTE_SERVICE_TIME] = tmp * tmp;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const service_time_dimension_info_t dim_info, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info             = dim_info;
      v.service_time_forward = raft::device_span<double>{(double*)shmem, (size_t)n_nodes_route + 1};
      v.service_time_backward = raft::device_span<double>{
        (double*)&v.service_time_forward.data()[n_nodes_route + 1], (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.service_time_backward.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    service_time_dimension_info_t dim_info;
    raft::device_span<double> service_time_forward;
    raft::device_span<double> service_time_backward;
  };

  view_t view()
  {
    view_t v;
    v.dim_info = dim_info;
    v.service_time_forward =
      raft::device_span<double>{service_time_forward.data(), service_time_forward.size()};
    v.service_time_backward =
      raft::device_span<double>{service_time_backward.data(), service_time_backward.size()};
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a distance route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] service_time_dimension_info_t dim_info)
  {
    // forward, backward
    return 2 * route_size * sizeof(double);
  }

  service_time_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<double> service_time_forward;
  // backward data
  rmm::device_uvector<double> service_time_backward;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
