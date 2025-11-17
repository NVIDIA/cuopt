/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/vector_helpers.cuh>
#include "../../solution/solution_handle.cuh"

#include <utilities/copy_helpers.hpp>
#include <utilities/vector_helpers.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/device_span.hpp>
#include <raft/util/cuda_utils.cuh>

#include <vector>

namespace cuopt {
namespace routing {
namespace detail {

constexpr int max_graph_nodes_per_row = 1024;

template <typename i_t, typename f_t>
struct graph_t {
  graph_t(i_t size, rmm::cuda_stream_view stream)
    : row_sizes(size, stream),
      route_ids(size, stream),
      // allocate with the max size
      indices(size * max_graph_nodes_per_row, stream),
      weights(size * max_graph_nodes_per_row, stream)
  {
  }

  struct host_t {
    std::vector<int> row_sizes;
    std::vector<int> route_ids;
    std::vector<int> indices;
    std::vector<double> weights;
  };

  host_t to_host()
  {
    host_t h;
    h.row_sizes = host_copy(row_sizes);
    h.route_ids = host_copy(route_ids);
    h.indices   = host_copy(indices);
    h.weights   = host_copy(weights);
    return h;
  }

  size_t get_num_vertices() const { return special_index + 1; }

  void reset(solution_handle_t<int, float> const* sol_handle)
  {
    async_fill(row_sizes, 0, sol_handle->get_stream());
    async_fill(indices, -1, sol_handle->get_stream());
    async_fill(weights, std::numeric_limits<double>::max(), sol_handle->get_stream());
    async_fill(route_ids, -1, sol_handle->get_stream());
  }

  struct view_t {
    constexpr size_t get_num_vertices() const { return special_index + 1; }

    raft::device_span<i_t> row_sizes;
    raft::device_span<i_t> route_ids;
    raft::device_span<i_t> indices;
    raft::device_span<double> weights;
    i_t special_index;
  };

  view_t view()
  {
    view_t v;
    v.row_sizes     = raft::device_span<i_t>{row_sizes.data(), row_sizes.size()};
    v.route_ids     = raft::device_span<i_t>{route_ids.data(), route_ids.size()};
    v.indices       = raft::device_span<i_t>{indices.data(), indices.size()};
    v.weights       = raft::device_span<double>{weights.data(), weights.size()};
    v.special_index = special_index;
    return v;
  }

  rmm::device_uvector<i_t> row_sizes;
  rmm::device_uvector<i_t> route_ids;
  rmm::device_uvector<i_t> indices;
  rmm::device_uvector<double> weights;
  i_t special_index;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
