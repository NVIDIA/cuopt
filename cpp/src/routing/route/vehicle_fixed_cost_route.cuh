/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/vehicle_fixed_cost_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class vehicle_fixed_cost_route_t {
 public:
  vehicle_fixed_cost_route_t([[maybe_unused]] solution_handle_t<i_t, f_t> const* sol_handle_,
                             vehicle_fixed_cost_dimension_info_t& dim_info_)
    : dim_info(dim_info_)
  {
  }

  vehicle_fixed_cost_route_t(const vehicle_fixed_cost_route_t& vehicle_fixed_cost_route,
                             [[maybe_unused]] solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(vehicle_fixed_cost_route.dim_info)
  {
  }

  vehicle_fixed_cost_route_t& operator=(vehicle_fixed_cost_route_t&& vehicle_fixed_cost_route) =
    default;

  void resize([[maybe_unused]] i_t max_nodes_per_route,
              [[maybe_unused]] rmm::cuda_stream_view stream)
  {
  }

  struct view_t {
    DI vehicle_fixed_cost_node_t<i_t, f_t> get_node(i_t idx) const
    {
      return vehicle_fixed_cost_node_t<i_t, f_t>{};
    }

    DI void set_node(i_t idx, const vehicle_fixed_cost_node_t<i_t, f_t>& node) {}

    DI void set_forward_data(i_t idx, const vehicle_fixed_cost_node_t<i_t, f_t>& node) {}

    DI void set_backward_data(i_t idx, const vehicle_fixed_cost_node_t<i_t, f_t>& node) {}

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      // FIXME: Use break_route_t info
      auto break_dim =
        vehicle_info.break_durations.empty() ? 0 : vehicle_info.break_durations.size();
      obj_cost[objective_t::VEHICLE_FIXED_COST] =
        n_nodes_route > 1 + break_dim ? vehicle_info.fixed_cost : 0.;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const vehicle_fixed_cost_dimension_info_t dim_info, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info = dim_info;
      return thrust::make_tuple(v, shmem);
    }

    vehicle_fixed_cost_dimension_info_t dim_info;
  };

  view_t view()
  {
    view_t v;
    v.dim_info = dim_info;
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a distance route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size([[maybe_unused]] i_t route_size,
                                    [[maybe_unused]] vehicle_fixed_cost_dimension_info_t dim_info)
  {
    //
    return 0;
  }

  vehicle_fixed_cost_dimension_info_t dim_info;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
