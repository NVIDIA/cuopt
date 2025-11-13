/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/handle.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

namespace cuopt {
namespace distance_engine {

/**
 * @brief A waypoint matrix.
 *
 * This class has two purposes:
 *
 * When the cost matrix is not acquirable due to an
 * incomplete graph, the latter can be passed to waypoint matrix.
 * The waypoint matrix can then return a cost matrix that can be used by the solver.
 *
 * The waypoint matrix can also generate the full path between target locations.
 * The full path represents the set of all vertices (with respect to the graph)
 * to go from one target location to another.
 *
 * @tparam i_t Integer type. int (32bit) is expected at the moment. Please
 * open an issue if other type are needed.
 * @tparam f_t Floating point type. float (32bit) is expected at the moment.
 * Please open an issue if other type are needed.
 */
template <typename i_t, typename f_t>
class waypoint_matrix_t {
 public:
  /**
   * @brief Construct an empty waypoint_matrix_t object
   */
  waypoint_matrix_t() = default;

  /**
   * @brief Construct a new waypoint matrix_t object
   *
   * @throws cuopt::logic_error when an error occurs.
   *
   * @param[in] handle Library handle (RAFT) containing hardware resources
   * informations. A default handle is valid.
   * @param[in] offsets Host memory pointer of size V + 1 (V: number of vertices).
   * It contains the offsets for the vertices in this graph. cuOpt does not own
   * or copy this data.
   * Offsets must be in the range [0, E] (E: number of edges).
   * @param n_vertices Number of vertices
   * @param[in] indices Host memory pointer of size E (E: number of edges).
   * It contains the destination index for each edge. cuOpt does not own or copy this data.
   * Destination indices must be in the range [0, V) (V: number of vertices).
   * @param[in] weights Host memory pointer of size E (E: number of edges).
   * It contains the weight value for each edge.
   * cuOpt does not own or copy this data.
   * The expected type is floating point number.
   */
  waypoint_matrix_t(raft::handle_t const& handle,
                    i_t const* offsets,
                    i_t n_vertices,
                    i_t const* indices,
                    f_t const* weights);

  /**
   * @brief Compute the cost matrix over the passed graph and target locations.
   *
   * This function can be used when the cost matrix is not acquirable due
   * to an incomplete graph. The cost matrix is computed then returned.
   * It can later be used for the Solver (see data_model_view_t::set_cost_matrix).
   *
   * @throws cuopt::logic_error when an error occurs.
   *
   * @param[out] d_cost_matrix Device memory pointer of size T*T (T: n_target_locations)
   * where the cost matrix will be written.
   * @param[in] target_locations Host memory pointer of size T (T: n_target_locations)
   * representing the target locations indices with respect to the graph.
   * Target locations indices must be in the range [0, V) (V: number of vertices).
   * @param n_target_locations Number of target locations.
   */
  void compute_cost_matrix(f_t* d_cost_matrix, i_t const* target_locations, i_t n_target_locations);

  /**
   * @brief Compute the waypoint sequence over the whole route.
   *
   * The waypoint sequence is an extend version of the route.
   * Between each route target locations, all the intermediate waypoints
   * are added.
   * Waypoints & target locations ids are based on the graph.
   *
   * @note Calling this function before compute_cost_matrix is an error.
   *
   * @throws cuopt::logic_error when an error occurs.
   *
   * @param[in] target_locations Host memory pointer of size T (T: n_target_locations)
   * representing the target locations indices with respect to the graph.
   * Target locations indices must be in the range [0, V) (V: number of vertices).
   * @param n_target_locations Number of target locations
   * @param[in] locations Device memory pointer of size L (L: n_locations) containing the location
   * of orders
   * Locations indices my be in the range [0, T) (T: n_target_locations)]
   * @param n_locations Number of locations
   * @return std::pair<rmm::device_uvector<int>, rmm::device_uvector<int>> First is a device buffer
   * of size L (L : n_locations) containing an array of offsets. Second is a device buffer
   * containing the full path for all the route. For each element in the route, the corresponding
   * full route can accessed in the full path through the offsets array.
   */
  std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>
  compute_waypoint_sequence(i_t const* target_locations,
                            i_t n_target_locations,
                            i_t const* locations,
                            i_t n_locations);

  /**
   * @brief Compute a custom matrix over the passed weights and target locations applied on shortest
   * paths found during previous compute_cost_matrix call.
   *
   * This function allows setting a custom cost between waypoints (for
   * example time) and then getting the total cost it takes to go from one target
   * location to all the others. The shortest paths are **not** recomputed.
   * The path found from compute_cost_matrix between target locations stays the
   * same but the new weight set is used to compute the output matrix.
   *
   * @note Giving an edge ordering for weights different from the one given
   * during waypoint matrix instanciation will lead to incorrect results.
   *
   * @throws cuopt::logic_error when an error occurs.
   *
   * @param[out] d_custom_matrix Device memory pointer of size T*T (T: n_target_locations)
   * where the custom matrix will be written.
   * @param[in] target_locations Host memory pointer of size T (T: n_target_locations)
   * representing the target locations indices with respect to the graph.
   * Target locations indices must be in the range [0, V) (V: number of vertices).
   * @param n_target_locations Number of target locations.
   * @param[in] weights Host memory pointer of size E (E: number of edges).
   * It contains the weight value for each edge.
   * cuOpt does not own or copy this data.
   * The expected type is floating point number.
   */
  void compute_shortest_path_costs(f_t* d_custom_matrix,
                                   i_t const* target_locations,
                                   i_t n_target_locations,
                                   f_t const* weights);

 private:
  std::vector<f_t> mpsp(i_t const* target_locations, i_t n_target_locations);
  template <typename pm_t>
  void dijkstra(pm_t& predecessor_matrix,
                std::vector<f_t>& cost_matrix,
                i_t src,
                i_t const* target_locations,
                i_t n_target_locations,
                i_t id_src);
  std::vector<f_t> _compute_shortest_path_costs(i_t const* target_locations,
                                                i_t n_target_locations,
                                                f_t const* weights);
  template <typename pm_t>
  void compute_secondary_cost(pm_t& predecessor_matrix,
                              typename pm_t::value_type::value_type src_matrix_id,
                              typename pm_t::value_type::value_type dst_graph_id,
                              f_t const* weights,
                              f_t& out_cost);
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_{};
  i_t const* offsets_;
  i_t n_vertices_;
  i_t const* indices_;
  f_t const* weights_;
  // Optimize allocation time based on number of vertices
  bool is_int16_{false};
  std::vector<std::vector<int32_t>> predecessor_matrix32_{};
  std::vector<std::vector<uint16_t>> predecessor_matrix16_{};
};
}  // namespace distance_engine
}  // namespace cuopt
