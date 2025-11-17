/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
struct ox_graph_t {
  ox_graph_t(i_t n_buckets_, i_t size, i_t max_nodes_per_row, rmm::cuda_stream_view stream)
    : row_sizes(n_buckets_ * size, stream),
      route_ids(n_buckets_ * size, stream),
      // allocate with the max size
      indices(n_buckets_ * size * max_nodes_per_row, stream),
      weights(n_buckets_ * size * max_nodes_per_row, stream),
      buckets(n_buckets_ * size * max_nodes_per_row, stream),
      n_buckets(n_buckets_)
  {
  }

  static auto get_allocated_bytes(i_t n_buckets, i_t size, i_t max_nodes_per_row)
  {
    return sizeof(i_t) * (n_buckets * size) * 2 +
           sizeof(i_t) * (n_buckets * size * max_nodes_per_row) * 2 +
           sizeof(double) * (n_buckets * size * max_nodes_per_row);
  }

  struct host_t {
    std::vector<int> row_sizes;
    std::vector<int> route_ids;
    std::vector<int> indices;
    std::vector<double> weights;
    std::vector<int> buckets;
  };

  host_t to_host()
  {
    host_t h;
    h.row_sizes = host_copy(row_sizes);
    h.route_ids = host_copy(route_ids);
    h.indices   = host_copy(indices);
    h.weights   = host_copy(weights);
    h.buckets   = host_copy(buckets);
    return h;
  }

  void resize(i_t n_buckets_, i_t size, i_t max_nodes_per_row, rmm::cuda_stream_view stream)
  {
    n_buckets = n_buckets_;
    row_sizes.resize(n_buckets * size, stream);
    route_ids.resize(n_buckets * size, stream);
    indices.resize(n_buckets * size * max_nodes_per_row, stream);
    weights.resize(n_buckets * size * max_nodes_per_row, stream);
    buckets.resize(n_buckets * size * max_nodes_per_row, stream);
  }

  auto get_num_vertices() const { return row_sizes.size() / n_buckets; }

  auto get_max_nodes_per_row() const { return indices.size() / row_sizes.size(); }

  void reset(solution_handle_t<int, float> const* sol_handle)
  {
    async_fill(row_sizes, 0, sol_handle->get_stream());
    async_fill(indices, std::numeric_limits<int>::max(), sol_handle->get_stream());
    async_fill(weights, std::numeric_limits<double>::max(), sol_handle->get_stream());
    async_fill(buckets, -1, sol_handle->get_stream());
    async_fill(route_ids, -1, sol_handle->get_stream());
  }

  struct view_t {
    constexpr auto get_num_vertices() const { return row_sizes.size() / n_buckets; }
    constexpr auto get_max_nodes_per_row() const { return indices.size() / row_sizes.size(); }

    raft::device_span<i_t> row_sizes;
    raft::device_span<i_t> route_ids;
    raft::device_span<i_t> indices;
    raft::device_span<double> weights;
    raft::device_span<i_t> buckets;
    i_t n_buckets;
  };

  view_t view()
  {
    view_t v;
    v.row_sizes = raft::device_span<i_t>{row_sizes.data(), row_sizes.size()};
    v.route_ids = raft::device_span<i_t>{route_ids.data(), route_ids.size()};
    v.indices   = raft::device_span<i_t>{indices.data(), indices.size()};
    v.weights   = raft::device_span<double>{weights.data(), weights.size()};
    v.buckets   = raft::device_span<i_t>{buckets.data(), buckets.size()};
    v.n_buckets = n_buckets;
    return v;
  }

  i_t n_buckets;
  rmm::device_uvector<i_t> row_sizes;
  rmm::device_uvector<i_t> route_ids;
  rmm::device_uvector<i_t> indices;
  rmm::device_uvector<double> weights;
  rmm::device_uvector<i_t> buckets;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
