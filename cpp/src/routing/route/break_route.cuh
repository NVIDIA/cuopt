/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/break_node.cuh"
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
class break_route_t {
 public:
  break_route_t(solution_handle_t<i_t, f_t> const* sol_handle_, break_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      breaks_forward(0, sol_handle_->get_stream()),
      breaks_backward(0, sol_handle_->get_stream()),
      break_window_violations_forward(0, sol_handle_->get_stream()),
      load_at_node(0, sol_handle_->get_stream()),
      incoming_min(0, sol_handle_->get_stream()),
      incoming_max(0, sol_handle_->get_stream()),
      load_window_active(0, sol_handle_->get_stream()),
      load_window_min(0, sol_handle_->get_stream()),
      load_window_max(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero break_route_t copy_ctr");
  }

  break_route_t(const break_route_t& break_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(break_route.dim_info),
      breaks_forward(break_route.breaks_forward, sol_handle_->get_stream()),
      breaks_backward(break_route.breaks_backward, sol_handle_->get_stream()),
      break_window_violations_forward(break_route.break_window_violations_forward,
                                      sol_handle_->get_stream()),
      load_at_node(break_route.load_at_node, sol_handle_->get_stream()),
      incoming_min(break_route.incoming_min, sol_handle_->get_stream()),
      incoming_max(break_route.incoming_max, sol_handle_->get_stream()),
      load_window_active(break_route.load_window_active, sol_handle_->get_stream()),
      load_window_min(break_route.load_window_min, sol_handle_->get_stream()),
      load_window_max(break_route.load_window_max, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("break route copy_ctr");
  }

  break_route_t& operator=(break_route_t&& break_route) = default;

  void resize(i_t max_nodes_per_route, cuda::stream_ref stream)
  {
    breaks_forward.resize(max_nodes_per_route, stream);
    breaks_backward.resize(max_nodes_per_route, stream);
    break_window_violations_forward.resize(max_nodes_per_route, stream);
    auto sz = static_cast<size_t>(max_nodes_per_route) * max_break_load_window_dim;
    load_at_node.resize(sz, stream);
    incoming_min.resize(sz, stream);
    incoming_max.resize(sz, stream);
    load_window_active.resize(sz, stream);
    load_window_min.resize(sz, stream);
    load_window_max.resize(sz, stream);
  }

  struct view_t {
    bool is_empty() const { return breaks_forward.empty(); }
    DI break_node_t<i_t, f_t> get_node(i_t idx) const
    {
      break_node_t<i_t, f_t> break_node;
      break_node.breaks_forward  = breaks_forward[idx];
      break_node.breaks_backward = breaks_backward[idx];
      break_node.break_window_violations_forward = break_window_violations_forward[idx];
      for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
        auto offset                      = dim * stride + idx;
        break_node.load_at_node[dim]    = load_at_node[offset];
        break_node.incoming_min[dim]    = incoming_min[offset];
        break_node.incoming_max[dim]    = incoming_max[offset];
        break_node.load_window_active[dim] = load_window_active[offset];
        break_node.load_window_min[dim] = load_window_min[offset];
        break_node.load_window_max[dim] = load_window_max[offset];
      }
      return break_node;
    }

    DI void set_node(i_t idx, const break_node_t<i_t, f_t>& node)
    {
      set_forward_data(idx, node);
      set_backward_data(idx, node);
      for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
        auto offset               = dim * stride + idx;
        load_window_active[offset] = node.load_window_active[dim];
        load_window_min[offset]   = node.load_window_min[dim];
        load_window_max[offset]   = node.load_window_max[dim];
      }
    }

    DI void set_forward_data(i_t idx, const break_node_t<i_t, f_t>& node)
    {
      breaks_forward[idx]                 = node.breaks_forward;
      break_window_violations_forward[idx] = node.break_window_violations_forward;
      for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
        load_at_node[dim * stride + idx] = node.load_at_node[dim];
      }
    }

    DI void set_backward_data(i_t idx, const break_node_t<i_t, f_t>& node)
    {
      breaks_backward[idx] = node.breaks_backward;
      for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
        incoming_min[dim * stride + idx] = node.incoming_min[dim];
        incoming_max[dim * stride + idx] = node.incoming_max[dim];
      }
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        breaks_forward.subspan(write_start), orig_route.breaks_forward.subspan(start_idx), size);
      block_copy(break_window_violations_forward.subspan(write_start),
                 orig_route.break_window_violations_forward.subspan(start_idx),
                 size);
      for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
        block_copy(load_at_node.subspan(dim * stride + write_start),
                   orig_route.load_at_node.subspan(dim * orig_route.stride + start_idx),
                   size);
      }
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(
        breaks_backward.subspan(write_start), orig_route.breaks_backward.subspan(start_idx), size);
      for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
        block_copy(incoming_min.subspan(dim * stride + write_start),
                   orig_route.incoming_min.subspan(dim * orig_route.stride + start_idx),
                   size);
        block_copy(incoming_max.subspan(dim * stride + write_start),
                   orig_route.incoming_max.subspan(dim * orig_route.stride + start_idx),
                   size);
      }
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      i_t size = to_idx - from_idx;
      for (int dim = 0; dim < max_break_load_window_dim; ++dim) {
        block_copy(load_window_active.subspan(dim * stride + write_start),
                   orig_route.load_window_active.subspan(dim * orig_route.stride + from_idx),
                   size);
        block_copy(load_window_min.subspan(dim * stride + write_start),
                   orig_route.load_window_min.subspan(dim * orig_route.stride + from_idx),
                   size);
        block_copy(load_window_max.subspan(dim * stride + write_start),
                   orig_route.load_window_max.subspan(dim * orig_route.stride + from_idx),
                   size);
      }
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
        infeasibility_cost += break_window_violations_forward[n_nodes_route];
      }
      inf_cost[dim_t::BREAK] = infeasibility_cost;
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const break_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      v.dim_info        = dim_info;
      v.stride          = n_nodes_route + 1;
      v.breaks_forward  = raft::device_span<int>{(int*)shmem, (size_t)v.stride};
      v.breaks_backward =
        raft::device_span<int>{(int*)&v.breaks_forward.data()[v.stride], (size_t)v.stride};
      v.break_window_violations_forward =
        raft::device_span<int>{(int*)&v.breaks_backward.data()[v.stride], (size_t)v.stride};
      auto* dim_ptr = (i_t*)&v.break_window_violations_forward.data()[v.stride];
      v.load_at_node = raft::device_span<i_t>{dim_ptr,
                                              (size_t)v.stride * max_break_load_window_dim};
      dim_ptr += v.stride * max_break_load_window_dim;
      v.incoming_min = raft::device_span<i_t>{dim_ptr,
                                              (size_t)v.stride * max_break_load_window_dim};
      dim_ptr += v.stride * max_break_load_window_dim;
      v.incoming_max = raft::device_span<i_t>{dim_ptr,
                                              (size_t)v.stride * max_break_load_window_dim};
      dim_ptr += v.stride * max_break_load_window_dim;
      v.load_window_active = raft::device_span<i_t>{dim_ptr,
                                                    (size_t)v.stride * max_break_load_window_dim};
      dim_ptr += v.stride * max_break_load_window_dim;
      v.load_window_min = raft::device_span<i_t>{dim_ptr,
                                                 (size_t)v.stride * max_break_load_window_dim};
      dim_ptr += v.stride * max_break_load_window_dim;
      v.load_window_max = raft::device_span<i_t>{dim_ptr,
                                                 (size_t)v.stride * max_break_load_window_dim};

      i_t* sh_ptr = (i_t*)&v.load_window_max.data()[v.stride * max_break_load_window_dim];
      return thrust::make_tuple(v, sh_ptr);
    }

    break_dimension_info_t dim_info;
    i_t stride{0};
    raft::device_span<int> breaks_forward;
    raft::device_span<int> breaks_backward;
    raft::device_span<int> break_window_violations_forward;
    raft::device_span<i_t> load_at_node;
    raft::device_span<i_t> incoming_min;
    raft::device_span<i_t> incoming_max;
    raft::device_span<i_t> load_window_active;
    raft::device_span<i_t> load_window_min;
    raft::device_span<i_t> load_window_max;
  };

  view_t view()
  {
    view_t v;
    v.dim_info        = dim_info;
    v.stride          = breaks_forward.size();
    v.breaks_forward  = raft::device_span<int>{breaks_forward.data(), breaks_forward.size()};
    v.breaks_backward = raft::device_span<int>{breaks_backward.data(), breaks_backward.size()};
    v.break_window_violations_forward = raft::device_span<int>{break_window_violations_forward.data(),
                                                               break_window_violations_forward.size()};
    v.load_at_node = raft::device_span<i_t>{load_at_node.data(), load_at_node.size()};
    v.incoming_min = raft::device_span<i_t>{incoming_min.data(), incoming_min.size()};
    v.incoming_max = raft::device_span<i_t>{incoming_max.data(), incoming_max.size()};
    v.load_window_active = raft::device_span<i_t>{load_window_active.data(), load_window_active.size()};
    v.load_window_min = raft::device_span<i_t>{load_window_min.data(), load_window_min.size()};
    v.load_window_max = raft::device_span<i_t>{load_window_max.data(), load_window_max.size()};
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
    return 3 * route_size * sizeof(int) +
           (6 * route_size * max_break_load_window_dim) * sizeof(i_t);
  }

  break_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<int> breaks_forward;
  // backward data
  rmm::device_uvector<int> breaks_backward;
  rmm::device_uvector<int> break_window_violations_forward;
  rmm::device_uvector<i_t> load_at_node;
  rmm::device_uvector<i_t> incoming_min;
  rmm::device_uvector<i_t> incoming_max;
  rmm::device_uvector<i_t> load_window_active;
  rmm::device_uvector<i_t> load_window_min;
  rmm::device_uvector<i_t> load_window_max;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
