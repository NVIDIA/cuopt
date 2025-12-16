/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/capacity_node.cuh"
#include "../routing_helpers.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>
namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class capacity_route_t {
 public:
  capacity_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                   const capacity_dimension_info_t& dim_info_)
    : demand(0, sol_handle_->get_stream()),
      gathered(0, sol_handle_->get_stream()),
      max_to_node(0, sol_handle_->get_stream()),
      max_after(0, sol_handle_->get_stream()),
      dim_info(dim_info_)
  {
    cuopt_assert(dim_info.n_capacity_dimensions <= default_max_capacity_dim,
                 "Supplied capacity dimension exceeds limit");
    raft::common::nvtx::range fun_scope("zero capacity route copy_ctr");
  }

  capacity_route_t(const capacity_route_t& capacity_route,
                   solution_handle_t<i_t, f_t> const* sol_handle_)
    : demand(capacity_route.demand, sol_handle_->get_stream()),
      gathered(capacity_route.gathered, sol_handle_->get_stream()),
      max_to_node(capacity_route.max_to_node, sol_handle_->get_stream()),
      max_after(capacity_route.max_after, sol_handle_->get_stream()),
      dim_info(capacity_route.dim_info)
  {
    raft::common::nvtx::range fun_scope("capacity route copy_ctr");
  }

  capacity_route_t& operator=(capacity_route_t&& capacity_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    demand.resize(dim_info.n_capacity_dimensions * max_nodes_per_route, stream);
    gathered.resize(dim_info.n_capacity_dimensions * max_nodes_per_route, stream);
    max_to_node.resize(dim_info.n_capacity_dimensions * max_nodes_per_route, stream);
    max_after.resize(dim_info.n_capacity_dimensions * max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return demand.is_empty(); }

    DI capacity_node_t<i_t, f_t> get_node(i_t idx) const
    {
      capacity_node_t<i_t, f_t> capacity_node(dim_info);
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) {
          capacity_node.gathered[i]    = gathered[i * stride + idx];
          capacity_node.max_to_node[i] = max_to_node[i * stride + idx];
          capacity_node.max_after[i]   = max_after[i * stride + idx];
          capacity_node.demand[i]      = demand[i * stride + idx];
        }
      });
      capacity_node.n_capacity_dimensions = dim_info.n_capacity_dimensions;
      return capacity_node;
    }

    DI void set_node(i_t idx, const capacity_node_t<i_t, f_t>& node)
    {
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) { demand[i * stride + idx] = node.demand[i]; }
      });
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const capacity_node_t<i_t, f_t>& node)
    {
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) {
          gathered[i * stride + idx]    = node.gathered[i];
          max_to_node[i * stride + idx] = node.max_to_node[i];
        }
      });
    }

    DI void set_backward_data(i_t idx, const capacity_node_t<i_t, f_t>& node)
    {
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) { max_after[i * stride + idx] = node.max_after[i]; }
      });
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) {
          block_copy(gathered.subspan(write_start), orig_route.gathered.subspan(start_idx), size);
          block_copy(
            max_to_node.subspan(write_start), orig_route.max_to_node.subspan(start_idx), size);
          write_start += this->stride;
          start_idx += orig_route.stride;
        }
      });
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) {
          block_copy(max_after.subspan(write_start), orig_route.max_after.subspan(start_idx), size);
          write_start += this->stride;
          start_idx += orig_route.stride;
        }
      });
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      auto size = to_idx - from_idx;
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) {
          block_copy(demand.subspan(write_start), orig_route.demand.subspan(from_idx), size);
          write_start += this->stride;
          from_idx += orig_route.stride;
        }
      });
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      double infeasibility_cost = 0.;
      constexpr_for<capacity_node_t<i_t, f_t>::max_capacity_dim>([&](auto i) {
        if (i < dim_info.n_capacity_dimensions) {
          infeasibility_cost +=
            max(0, max_to_node[n_nodes + stride * i] - vehicle_info.capacities[i]);
        }
      });

      inf_cost[dim_t::CAP] = infeasibility_cost;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const capacity_dimension_info_t dim_info, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info = dim_info;
      v.stride   = n_nodes_route + 1;

      size_t sz   = static_cast<size_t>(v.stride * dim_info.n_capacity_dimensions);
      i_t* sh_ptr = shmem;

      thrust::tie(v.gathered, sh_ptr)    = wrap_ptr_as_span<i_t>(sh_ptr, sz);
      thrust::tie(v.max_to_node, sh_ptr) = wrap_ptr_as_span<i_t>(sh_ptr, sz);
      thrust::tie(v.max_after, sh_ptr)   = wrap_ptr_as_span<i_t>(sh_ptr, sz);
      thrust::tie(v.demand, sh_ptr)      = wrap_ptr_as_span<i_t>(sh_ptr, sz);

      return thrust::make_tuple(v, sh_ptr);
    }

    raft::device_span<i_t> gathered;
    raft::device_span<i_t> max_to_node;
    raft::device_span<i_t> max_after;
    raft::device_span<i_t> demand;
    capacity_dimension_info_t dim_info;
    i_t stride;
  };

  view_t view()
  {
    view_t v;
    v.demand      = raft::device_span<i_t>{demand.data(), demand.size()};
    v.gathered    = raft::device_span<i_t>{gathered.data(), gathered.size()};
    v.max_to_node = raft::device_span<i_t>{max_to_node.data(), max_to_node.size()};
    v.max_after   = raft::device_span<i_t>{max_after.data(), max_after.size()};
    v.dim_info    = dim_info;
    v.stride      = static_cast<i_t>(demand.size() / dim_info.n_capacity_dimensions);
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a capacity route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size, capacity_dimension_info_t dim_info)
  {
    // demand, gathered, max_to_node, max_after
    return dim_info.n_capacity_dimensions * 4 * route_size * sizeof(i_t);
  }

  //! Data copied from problem
  rmm::device_uvector<i_t> demand;
  //! Total commodity gathered from begining up to considered node (incl. the node) - forward
  //! calculation
  rmm::device_uvector<i_t> gathered;
  //! Max load of the vehicle before the node (incl. immediatly after the node) - forward
  //! calculation
  rmm::device_uvector<i_t> max_to_node;
  //! Max load after the node (considering only the final fragment) - backward calculation
  rmm::device_uvector<i_t> max_after;

  capacity_dimension_info_t dim_info;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
