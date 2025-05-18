/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuopt/error.hpp>
#include <cuopt/routing/distance_engine/waypoint_matrix.hpp>

#include <routing/utilities/check_input.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stack>
#include <vector>

namespace cuopt {
namespace distance_engine {

#define dispatch(func, ...)                     \
  do {                                          \
    if (is_int16_)                              \
      func(predecessor_matrix16_, __VA_ARGS__); \
    else                                        \
      func(predecessor_matrix32_, __VA_ARGS__); \
  } while (false)

template <typename pm_t, typename i_t>
static void add_path(pm_t& predecessor_matrix,
                     typename pm_t::value_type::value_type src_matrix_id,
                     typename pm_t::value_type::value_type dst_graph_id,
                     std::vector<i_t>& paths_list,
                     i_t i,
                     std::vector<i_t>& paths_offsets)
{
  std::stack<i_t> s;
  std::size_t path_size;

  do {
    s.emplace(dst_graph_id);
    dst_graph_id = predecessor_matrix[src_matrix_id][dst_graph_id];
  } while (dst_graph_id != static_cast<typename pm_t::value_type::value_type>(-1));

  path_size = s.size();

  while (!s.empty()) {
    paths_list.emplace_back(s.top());
    s.pop();
  }

  paths_offsets[i] = path_size + paths_offsets[i - 1];
}

template <typename i_t, typename f_t>
static void write_cost_matrix(std::vector<f_t>& cost_matrix,
                              i_t id_src,
                              std::vector<f_t> const& dist,
                              i_t const* target_locations,
                              i_t n_target_locations)
{
  for (std::size_t i = 0; i != n_target_locations; ++i)
    cost_matrix[id_src * n_target_locations + i] = dist[target_locations[i]];
}

template <typename i_t, typename f_t>
template <typename pm_t>
void waypoint_matrix_t<i_t, f_t>::dijkstra(pm_t& predecessor_matrix,
                                           std::vector<f_t>& cost_matrix,
                                           i_t src,
                                           i_t const* target_locations,
                                           i_t n_target_locations,
                                           i_t id_src)
{
  using node_t = std::pair<f_t, i_t>;

  constexpr f_t unset_val = 1.0e+30;

  std::vector<f_t> dist(n_vertices_, unset_val);
  std::priority_queue<node_t, std::vector<node_t>, std::greater<node_t>> min_q;

  // Init src in dist array and in priority queue
  min_q.emplace(static_cast<f_t>(0), src);
  dist[src] = static_cast<f_t>(0);

  while (!min_q.empty()) {
    // Get node with minimum distance node out of the priority queue
    const auto [distance, u] = min_q.top();
    min_q.pop();

    if (distance > dist[u]) continue;

    const auto nbr_offsets     = offsets_[u];
    const auto nbr_offset_last = offsets_[u + 1];

    // Loop through neighbor vertices
    for (auto nbr_offset = nbr_offsets; nbr_offset != nbr_offset_last; ++nbr_offset) {
      const auto v            = indices_[nbr_offset];
      const auto new_distance = distance + weights_[nbr_offset];
      // Update dist array and priority queue structure if new path is smaller
      if (new_distance < dist[v]) {
        // Store offset & edge id for easier time matrix computation
        predecessor_matrix[id_src][v] = u;
        dist[v]                       = new_distance;
        min_q.emplace(new_distance, v);
      }
    }
  }

  // Write in cost matrix
  write_cost_matrix(cost_matrix, id_src, dist, target_locations, n_target_locations);
}

template <typename i_t, typename f_t>
std::vector<f_t> waypoint_matrix_t<i_t, f_t>::mpsp(i_t const* target_locations,
                                                   i_t n_target_locations)
{
  // TODO : data is not pinned, passing buffer to vector is not really easy so at worst just use
  // regular ptr
  std::vector<f_t> cost_matrix(n_target_locations * n_target_locations);

  if (n_vertices_ < std::numeric_limits<uint16_t>::max()) {
    // -1 gets round up to uint16_t::max
    predecessor_matrix16_ = std::vector<std::vector<uint16_t>>(
      n_target_locations, std::vector<uint16_t>(n_vertices_, -1));
    is_int16_ = true;
  } else {
    predecessor_matrix32_ =
      std::vector<std::vector<int32_t>>(n_target_locations, std::vector<int32_t>(n_vertices_, -1));
  }

// Run n_target_locations dijkstras in parallel with each target as source
#pragma omp parallel for
  for (std::size_t i = 0; i < n_target_locations; ++i)
    dispatch(dijkstra, cost_matrix, target_locations[i], target_locations, n_target_locations, i);

  return cost_matrix;
}

// Negative values or not sorted or out of bounds (more than edges)
template <typename i_t>
static void check_offsets(i_t const* offsets, i_t n_vertices)
{
  const i_t nb_edges = offsets[n_vertices];
  i_t prev           = offsets[0];
  cuopt_expects(prev >= 0, error_type_t::ValidationError, "Offsets values must be positive");
  cuopt_expects(prev <= nb_edges,
                error_type_t::ValidationError,
                "Offsets values must be lower than the number of edges");

  for (i_t i = 1; i < n_vertices; ++i) {
    const i_t curr = offsets[i];
    cuopt_expects(curr >= 0, error_type_t::ValidationError, "Offsets values must be positive.");
    cuopt_expects(curr <= nb_edges,
                  error_type_t::ValidationError,
                  "Offsets values must be lower than the number of edges.");
    cuopt_expects(
      curr >= prev, error_type_t::ValidationError, "Offsets values must in an increasing order.");
    prev = curr;
  }
}

// Negative values, values out of bounds (more than vertices)
template <typename i_t>
static void check_indices(i_t const* indices, i_t n_vertices, i_t n_edges)
{
  for (i_t i = 0; i < n_edges; ++i) {
    const i_t curr = indices[i];
    cuopt_expects(curr >= 0, error_type_t::ValidationError, "Indices values must be positive.");
    cuopt_expects(curr < n_vertices,
                  error_type_t::ValidationError,
                  "Indices values must be lower than the number of vertices.");
  }
}

// Negative values
template <typename i_t, typename f_t>
static void check_weights(f_t const* weights, i_t n_edges)
{
  if (std::any_of(weights, weights + n_edges, [](f_t val) { return val < 0; }))
    cuopt_expects(false, error_type_t::ValidationError, "Weights values must be positive.");
}

template <typename i_t, typename f_t>
waypoint_matrix_t<i_t, f_t>::waypoint_matrix_t(raft::handle_t const& handle,
                                               i_t const* offsets,
                                               i_t n_vertices,
                                               i_t const* indices,
                                               f_t const* weights)
  : handle_ptr_(&handle), stream_view_(handle_ptr_->get_stream())
{
  cuopt_expects(offsets != nullptr, error_type_t::ValidationError, "Offsets input cannot be null.");
  cuopt_expects(
    n_vertices > 0, error_type_t::ValidationError, "Number of indices should be positive.");
  cuopt_expects(indices != nullptr, error_type_t::ValidationError, "Indices input cannot be null.");
  cuopt_expects(weights != nullptr, error_type_t::ValidationError, "Weights input cannot be null.");

  // Graph validity checks
  // If multiple problems occur, only one will be displayed
  check_offsets(offsets, n_vertices);
  check_indices(indices, n_vertices, offsets[n_vertices]);
  check_weights(weights, offsets[n_vertices]);

  offsets_    = offsets;
  n_vertices_ = n_vertices;
  indices_    = indices;
  weights_    = weights;
}

// Negative values, out of bounds (more than vertices)
template <typename i_t>
static void check_target_locations(i_t const* targets, i_t n_targets, i_t n_vertices)
{
  for (i_t i = 0; i < n_targets; ++i) {
    const i_t val = targets[i];
    cuopt_expects(
      val >= 0, error_type_t::ValidationError, "Target locations values must be positive.");
    cuopt_expects(val < n_vertices,
                  error_type_t::ValidationError,
                  "Target locations values must be lower than the number of vertices.");
  }
}

template <typename i_t, typename f_t>
void waypoint_matrix_t<i_t, f_t>::compute_cost_matrix(f_t* d_cost_matrix,
                                                      i_t const* target_locations,
                                                      i_t n_target_locations)
{
  cuopt_expects(
    d_cost_matrix != nullptr, error_type_t::ValidationError, "Cost matrix input cannot be null.");
  cuopt_expects(
    target_locations != nullptr, error_type_t::ValidationError, "Target locations cannot be null.");
  cuopt_expects(n_target_locations > 0,
                error_type_t::ValidationError,
                "Number of target locations should be positive.");

  // Target locations validity checks
  check_target_locations(target_locations, n_target_locations, n_vertices_);

  std::vector<f_t> cost_matrix = mpsp(target_locations, n_target_locations);

  raft::copy(d_cost_matrix, cost_matrix.data(), cost_matrix.size(), stream_view_);
  stream_view_.synchronize();
}

// Location values are greater or equal to n_target_locations
template <typename i_t>
static void check_locations(i_t const* locations, i_t n_locations, i_t n_target_locations)
{
  if (std::any_of(locations, locations + n_locations, [n_target_locations](i_t index) {
        return index >= n_target_locations;
      }))
    cuopt_expects(false,
                  error_type_t::ValidationError,
                  "Locations values must be lower than the number of target locations.");
}

template <typename i_t, typename f_t>
std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>
waypoint_matrix_t<i_t, f_t>::compute_waypoint_sequence(i_t const* target_locations,
                                                       i_t n_target_locations,
                                                       i_t const* locations,
                                                       i_t n_locations)
{
  cuopt_expects(
    target_locations != nullptr, error_type_t::ValidationError, "Offset input cannot be null.");
  cuopt_expects(n_target_locations > 0,
                error_type_t::ValidationError,
                "Number of target locations should be positive.");
  cuopt_expects(
    locations != nullptr, error_type_t::ValidationError, "Location input cannot be null.");
  cuopt_expects(
    n_locations > 0, error_type_t::ValidationError, "Number of locations should be positive.");
  if (is_int16_)
    cuopt_expects(predecessor_matrix16_.size() > 0,
                  error_type_t::ValidationError,
                  "compute_waypoint_sequence cannot be called before compute_cost_matrix.");
  else
    cuopt_expects(predecessor_matrix32_.size() > 0,
                  error_type_t::ValidationError,
                  "compute_waypoint_sequence cannot be called before compute_cost_matrix.");

  // Target locations validity checks
  check_target_locations(target_locations, n_target_locations, n_vertices_);

  std::vector<i_t> h_locations(n_locations);
  raft::copy(h_locations.data(), locations, n_locations, stream_view_);
  stream_view_.synchronize();

  // Locations validity checks
  check_locations(h_locations.data(), n_locations, n_target_locations);

  std::vector<i_t> paths_offsets(n_locations);
  std::vector<i_t> paths_list;

  /* Full path could be computed in parallel :
  ** Compute each path in parallel and store the path length in an array
  ** Do an exclusive sum scan over this array to the each final offset
  ** Create the array knowing its full size (thanks to the final offset)
  ** Insert in parallel nodes in the array
  **
  ** Currently the computation is fast enough
  */
  paths_offsets[0] = 0;
  for (i_t i = 1; i != n_locations; ++i) {
    const auto src_matrix_id = h_locations[i - 1];
    const auto dst_graph_id  = target_locations[h_locations[i]];
    dispatch(add_path, src_matrix_id, dst_graph_id, paths_list, i, paths_offsets);
  }

  rmm::device_uvector<i_t> paths_offsets_out(paths_offsets.size(), stream_view_);
  rmm::device_uvector<i_t> paths_list_out(paths_list.size(), stream_view_);

  raft::copy(paths_offsets_out.data(), paths_offsets.data(), paths_offsets.size(), stream_view_);
  raft::copy(paths_list_out.data(), paths_list.data(), paths_list.size(), stream_view_);
  stream_view_.synchronize();

  return {std::make_unique<rmm::device_buffer>(paths_offsets_out.release()),
          std::make_unique<rmm::device_buffer>(paths_list_out.release())};
}

template <typename i_t, typename f_t>
template <typename pm_t>
void waypoint_matrix_t<i_t, f_t>::compute_secondary_cost(
  pm_t& predecessor_matrix,
  typename pm_t::value_type::value_type src_matrix_id,
  typename pm_t::value_type::value_type dst_graph_id,
  f_t const* weights,
  f_t& out_cost)
{
  f_t cost = 0;

  auto previous = predecessor_matrix[src_matrix_id][dst_graph_id];
  do {
    const auto nbr_offsets     = offsets_[previous];
    const auto nbr_offset_last = offsets_[previous + 1];

    // Loop through neighbor vertices
    for (auto nbr_offset = nbr_offsets; nbr_offset != nbr_offset_last; ++nbr_offset) {
      const auto v = indices_[nbr_offset];
      if (v == dst_graph_id)  // Found next node in neighbor list
      {
        cost += weights[nbr_offset];
        break;
      }
    }

    // Climb up
    dst_graph_id = previous;
    previous     = predecessor_matrix[src_matrix_id][dst_graph_id];
  } while (previous != static_cast<typename pm_t::value_type::value_type>(-1));

  out_cost = cost;
}

template <typename i_t, typename f_t>
std::vector<f_t> waypoint_matrix_t<i_t, f_t>::_compute_shortest_path_costs(
  i_t const* target_locations, i_t n_target_locations, f_t const* weights)
{
  std::vector<f_t> shortest_path_matrix(n_target_locations * n_target_locations);

#pragma omp parallel for
  for (std::size_t i = 0; i < n_target_locations * n_target_locations; ++i) {
    const auto src = i / n_target_locations;
    const auto dst = i % n_target_locations;
    if (src == dst)
      shortest_path_matrix[i] = 0.0f;
    else
      dispatch(
        compute_secondary_cost, src, target_locations[dst], weights, shortest_path_matrix[i]);
  }

  return shortest_path_matrix;
}

template <typename i_t, typename f_t>
void waypoint_matrix_t<i_t, f_t>::compute_shortest_path_costs(f_t* d_custom_matrix,
                                                              i_t const* target_locations,
                                                              i_t n_target_locations,
                                                              f_t const* weights)
{
  cuopt_expects(d_custom_matrix != nullptr,
                error_type_t::ValidationError,
                "Custom matrix input cannot be null.");
  cuopt_expects(target_locations != nullptr,
                error_type_t::ValidationError,
                "Target locations input cannot be null.");
  cuopt_expects(n_target_locations > 0,
                error_type_t::ValidationError,
                "Number of target locations should be positive.");
  cuopt_expects(weights != nullptr, error_type_t::ValidationError, "Weights input cannot be null.");

  // Target locations validity checks
  check_target_locations(target_locations, n_target_locations, n_vertices_);

  std::vector<f_t> shortest_path_matrix =
    _compute_shortest_path_costs(target_locations, n_target_locations, weights);

  raft::copy(
    d_custom_matrix, shortest_path_matrix.data(), shortest_path_matrix.size(), stream_view_);
  stream_view_.synchronize();
}

template class waypoint_matrix_t<int, float>;

}  // namespace distance_engine
}  // namespace cuopt
