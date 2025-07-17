/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include "../adapters/adapted_nodes.cuh"
#include "../adapters/adapted_sol.cuh"
#include "../dimensions.cuh"
#include "../diversity/helpers.hpp"
#include "ox_graph.hpp"
#include "ox_kernels.cuh"

#include <random>
#include <vector>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

namespace cuopt {
namespace routing {
namespace detail {

auto constexpr const max_vehicle_increase = 2;
auto constexpr const gb_limit             = 2;

/*! \brief { Recombine two solutions. One of input solutions may be overwritten and child may
 * contain lower number of nodes. } */
template <class Solution>
struct OX {
  std::random_device rd;
  std::mt19937 mt;

  size_t problem_size;

  costs rcosts;
  costs weights;

  // Costs of edges which correspond to routes in the VRP solution
  std::vector<std::vector<std::tuple<int, double, int>>> graph;
  // Costs of paths of given length path_cost[k][n]  -> min cost path of length 'k' to node 'n'
  std::vector<std::vector<double>> path_cost;
  // Vector of parents to recreate optimal path
  std::vector<std::vector<int>> predcessor;
  std::vector<std::vector<int>> predcessor_vehicle;

  std::vector<int> genome_A;
  std::vector<int> genome_B;

  std::vector<int> offspring;
  std::vector<int> offspring_tmp;

  int max_route_len;
  int routes_number;
  int optimal_routes_number;
  int n_buckets;

  std::vector<std::pair<size_t, double>> distances;
  std::unordered_set<int> helper;
  // TODO: use these for heterogenous/route cost/non fixed route

  //! Vector index := vehicle type. For each vehicle type the number of available vehicles of a
  //! given type is provided. If the vector is empty the fixed vehicle bellman_ford is performed.
  std::vector<int> vehicle_availability;
  std::vector<int> veh_cpy;
  std::vector<std::vector<int>> available_buckets;
  std::vector<int> vehicle_id_per_bucket;
  std::unordered_set<int> helper_set;
  bool fixed_route{false};
  bool run_heuristic{true};
  bool optimal_routes_search{false};

  // GPU
  rmm::device_uvector<double> d_path_cost;
  rmm::device_uvector<int> d_predecessor;
  rmm::device_uvector<int> d_predecessor_vehicle;
  rmm::device_uvector<int> d_offspring;
  rmm::device_uvector<int> d_helper_nodes;
  rmm::device_uvector<int> d_vehicle_id_per_bucket;
  rmm::device_uvector<int> val_map;
  rmm::device_uvector<int> row_offsets;
  rmm::device_uvector<std::byte> d_tmp_storage_bytes;
  rmm::device_uvector<double> gather_double;
  rmm::device_uvector<int> gather_int;
  rmm::device_uvector<int> d_vehicle_availability;
  ox_graph_t<int, float> d_graph;
  ox_graph_t<int, float> transpose_graph;

  explicit OX(size_t nodes_number, const costs& weight, rmm::cuda_stream_view stream_view)
    : mt(rd()),
      problem_size(nodes_number),
      graph(problem_size),
      path_cost(problem_size + 1),
      predcessor(problem_size + 1),
      predcessor_vehicle(problem_size + 1),
      distances(128),
      d_path_cost(0, stream_view),
      d_predecessor(0, stream_view),
      d_predecessor_vehicle(0, stream_view),
      d_offspring(problem_size, stream_view),
      d_helper_nodes(3, stream_view),
      d_vehicle_id_per_bucket(0, stream_view),
      d_graph(0, 0, 0, stream_view),
      transpose_graph(0, 0, 0, stream_view),
      val_map(0, stream_view),
      row_offsets(0, stream_view),
      d_tmp_storage_bytes(0, stream_view),
      gather_double(0, stream_view),
      gather_int(0, stream_view),
      d_vehicle_availability(0, stream_view)
  {
    weights = weight;
    genome_A.reserve(problem_size);
    genome_B.reserve(problem_size);
    offspring.reserve(problem_size);
    offspring_tmp.reserve(problem_size);

    for (auto& a : graph)
      a.reserve(problem_size);
    for (auto& a : path_cost)
      a.reserve(problem_size);
    for (auto& a : predcessor)
      a.reserve(problem_size);
  }

  void set_weight(const costs& weight) { weights = weight; }

  void adjust_vehicles(Solution& A, std::vector<int>& recombined_routes)
  {
    helper_set.clear();
    for (auto a : recombined_routes)
      helper_set.insert(a);

    for (size_t i = 0; i < A.get_routes().size(); i++)
      if (helper_set.count(i) == 0) vehicle_availability[A.get_routes()[i].bucket_id]--;
  }

  /*! \author { Piotr Sielski, psielski@nvidia.com}
   * \brief { OX recombiner: Christian Prins, "A simple and effective evolutionary algorithm for the
   * vehicle routing problem." Improvements: 1. When injecting part of the genome from B to A we
   * limit the genome fragment not to destroy the solution too much.
   *                        2. When creating the offspring: after injecting genome from B we start
   * iterating genome of A from the next node in A instead of next with respect to array order.
   *                        3. We order the routes in A and B in a way that close node ends from one
   * route are followed by close node starts in another.
   *                        4. We solve the fixed routes number version.
   *                        5. We limit the route length in the offspring to improve shortest path
   * speed.
   *           }
   */

  auto get_allocated_bytes(int n_buckets, int offspring_size, int max_route_len)
  {
    auto graphs_size =
      ox_graph_t<int, float>::get_allocated_bytes(n_buckets, offspring_size, max_route_len) * 2;
    auto sol_arrays_size = 2 * sizeof(int) * (offspring_size * (problem_size + 1)) +
                           sizeof(double) * (offspring_size * (problem_size + 1));
    auto helper_arrays_size = sizeof(int) * (offspring_size * offspring_size) * 2 +
                              sizeof(int) * offspring_size +
                              sizeof(double) * (offspring_size * offspring_size);
    return graphs_size + sol_arrays_size + helper_arrays_size;
  }

  bool recombine(Solution& A, Solution& B)
  {
    raft::common::nvtx::range fun_scope("ox");

    if (check_if_routes_empty(A) || check_if_routes_empty(B)) return false;

    n_buckets = A.problem->get_vehicle_buckets().size();
    compute_available_buckets(A);
    compute_vehicle_id_per_bucket(A);

    std::vector<int> routesA;
    std::vector<int> routesB;

    A.different_route_ids(routesA, B);
    B.different_route_ids(routesB, A);

    if (routesA.empty() || routesB.empty()) { return false; }

    if (routesA.size() > 2 && routesB.size() > 2) {
      std::swap(routesA[0], routesA[1 + next_random() % (routesA.size() - 1)]);
      std::swap(routesB[0], routesB[1 + next_random() % (routesB.size() - 1)]);

      prepare_route_order(routesA, A);
      prepare_route_order(routesB, B);
    }

    fixed_route = A.problem->get_fleet_size() == A.problem->data_view_ptr->get_min_vehicles();

    if (fixed_route && routesA.size() != routesB.size()) { return false; }
    adjust_vehicles(A, routesA);

    // minimization and fixed route
    routes_number = std::min(routesA.size(), routesB.size());

    const auto& dimensions_info = A.problem->dimensions_info;
    if (dimensions_info.has_dimension(dim_t::VEHICLE_FIXED_COST)) {
      routes_number         = max_vehicle_increase + std::max(routesA.size(), routesB.size());
      optimal_routes_search = true;
    }

    routes_number = std::min(routes_number, (int)problem_size);

    optimal_routes_number = routes_number;

    if (A.problem->fleet_info.is_homogenous_ && !optimal_routes_search) { run_heuristic = false; }

    max_route_len = 1;
    for (auto& b : routesA) {
      auto& a       = A.get_routes()[b];
      max_route_len = std::max<int>(max_route_len, a.length);
    }
    max_route_len += (max_route_len / 2);

    fill_genome(A, genome_A, routesA);
    fill_genome(B, genome_B, routesB);

    // For prize collecting we need common nodes in routes first
    std::unordered_set<int> set_genome_A(genome_A.begin(), genome_A.end());
    std::unordered_set<int> set_genome_B(genome_B.begin(), genome_B.end());
    if (set_genome_A != set_genome_B) { return false; }

    if (genome_A.size() < 2) { return false; }

    fill_offspring(A, routesA);

    // FIXME: Guard to avoid crash on large sizes.
    if (get_allocated_bytes(n_buckets, offspring.size(), max_route_len) * 1e-9 >= gb_limit) {
      return false;
    }

    d_graph.resize(n_buckets, offspring.size(), max_route_len, A.sol.sol_handle->get_stream());
    transpose_graph.resize(
      n_buckets, offspring.size(), max_route_len, A.sol.sol_handle->get_stream());
    calculate_edge_costs(A);
    bellman_ford(A);

    return recreate_solution(A);
  }

  /*! \brief { Arrange routes in a way that starts are close to the ends of previous route. } */
  void prepare_route_order(std::vector<int>& routesS, Solution& S)
  {
    auto const vehicle_buckets = S.problem->get_vehicle_buckets();
    for (size_t i = 1; i < routesS.size() - 1; i++) {
      distances.clear();

      for (size_t j = i; j < routesS.size(); j++) {
        auto vehicle_id = S.get_routes()[routesS[i - 1]].vehicle_id;
        distances.emplace_back(
          j,
          S.problem->distance_between(
            S.get_routes()[routesS[i - 1]].end, S.get_routes()[routesS[j]].start, vehicle_id));
      }
      std::normal_distribution<double> d(0, 0.25);
      double ind   = d(mt);
      size_t index = std::min<size_t>(distances.size() - 1, std::round(std::fabs(ind)));

      std::partial_sort(distances.begin(),
                        distances.begin() + index + 1,
                        distances.end(),
                        [](std::pair<size_t, double>& a, std::pair<size_t, double>& b) {
                          return a.second < b.second;
                        });

      if (distances[index].first != i) { std::swap(routesS[i], routesS[distances[index].first]); }
    }
  }

  /**
   * @brief Find disjoint set between current problem vehicle buckets and used buckets in solution
   *
   */
  void compute_available_buckets(Solution const& A)
  {
    available_buckets.clear();
    auto const problem_vehicle_buckets = A.problem->get_vehicle_buckets();
    std::vector<std::vector<int>> sol_vehicle_buckets(problem_vehicle_buckets.size());

    for (auto const& route : A.routes) {
      cuopt_assert(A.bucket_id(route.vehicle_id) == route.bucket_id, "Vehicle bucket mismatch");
      sol_vehicle_buckets[route.bucket_id].push_back(route.vehicle_id);
    }

    for (size_t i = 0; i < problem_vehicle_buckets.size(); ++i) {
      std::set<int> set_problem_bucket(problem_vehicle_buckets[i].begin(),
                                       problem_vehicle_buckets[i].end());
      std::set<int> set_sol_bucket(sol_vehicle_buckets[i].begin(), sol_vehicle_buckets[i].end());
      std::vector<int> diff;
      std::set_difference(set_problem_bucket.begin(),
                          set_problem_bucket.end(),
                          set_sol_bucket.begin(),
                          set_sol_bucket.end(),
                          std::inserter(diff, diff.begin()));
      available_buckets.push_back(diff);
    }
  }

  void compute_vehicle_id_per_bucket(Solution const& A)
  {
    vehicle_availability = A.problem->fleet_info_h.vehicle_availability;
    vehicle_id_per_bucket.clear();

    auto const& problem_vehicle_buckets = A.problem->get_vehicle_buckets();
    for (size_t i = 0; i < problem_vehicle_buckets.size(); ++i) {
      auto veh_id = problem_vehicle_buckets[i][0];
      vehicle_id_per_bucket.push_back(veh_id);
    }
    cuopt::device_copy(
      d_vehicle_id_per_bucket, vehicle_id_per_bucket, A.sol.sol_handle->get_stream());
  }

  //! \brief { Recreate the solution from the shortest path algorithm output.
  //!          TODO: return the changed routes
  //! }
  bool recreate_solution(Solution& A)
  {
    if (path_cost[optimal_routes_number].back() == std::numeric_limits<double>::max()) {
      return false;
    }

    std::vector<std::tuple<int, std::vector<uint32_t>>> tmp_routes;
    std::unordered_set<int> routes_to_remove;
    std::unordered_set<int> vehicle_ids_to_remove;
    const auto& dimensions_info = A.problem->dimensions_info;
    int i                       = routes_number;
    if (optimal_routes_search) { i = optimal_routes_number; }
    int end_index = offspring.size() - 1;
    double cost_n, cost_p, total_delta = 0.;

    std::vector<std::pair<int, std::vector<NodeInfo<>>>> routes_to_add;
    std::vector<uint32_t> tmp_route;

    while (i > 0) {
      int start_index = predcessor[i][end_index];
      int bucket      = predcessor_vehicle[i][end_index];
      cost_n          = path_cost[i][end_index];
      cost_p          = path_cost[i - 1][start_index];
      for (int k = start_index + 1; k <= end_index; k++) {
        tmp_route.push_back(offspring[k]);
      }

      if (tmp_route.size() < (size_t)request_info_t<int, Solution::request_type>::size()) {
        return false;
      }

      bool same_route = true;
      if (!A.succ[tmp_route.back()].is_depot() || !A.pred[tmp_route[0]].is_depot()) {
        same_route = false;
      } else if (A.bucket_id(A.get_node(tmp_route[0]).v_id) != bucket) {
        same_route = false;
      } else {
        for (size_t l = 0; l < tmp_route.size() - 1; l++) {
          auto& node = tmp_route[l];
          if ((uint32_t)A.succ[node].node() != tmp_route[l + 1]) {
            same_route = false;
            break;
          }
        }
      }
      // Insert new route / Omit the route that didn't change
      if (!same_route) {
        for (auto& node : tmp_route) {
          // First eject the routes that contain nodes from tmp_route
          if (!A.unserviced(node)) {
            routes_to_remove.insert(A.get_node(node).r_id);
            auto v_id     = A.get_node(node).v_id;
            auto fill_veh = A.bucket_id(v_id);
            if (!vehicle_ids_to_remove.count(v_id)) {
              available_buckets[fill_veh].push_back(v_id);
              vehicle_ids_to_remove.insert(v_id);
            }
          }
        }
        total_delta += (cost_n - cost_p);
        tmp_routes.push_back({bucket, tmp_route});
      }
      tmp_route.clear();
      end_index = start_index;
      --i;
    }

    if (fixed_route) {
      cuopt_assert(routes_to_remove.size() == tmp_routes.size(),
                   "number of routes removed and routes added should be same");
    }
    if (routes_to_remove.size() == 0 || tmp_routes.size() == 0) { return false; }

    A.remove_routes(std::vector(routes_to_remove.begin(), routes_to_remove.end()));

    std::vector<NodeInfo<>> tmp_node_info;
    auto vehicle_ids_to_remove_vec =
      std::vector(vehicle_ids_to_remove.begin(), vehicle_ids_to_remove.end());

    for (auto const& [bucket, tmp_route] : tmp_routes) {
      for (auto const& node : tmp_route) {
        tmp_node_info.push_back(A.problem->get_node_info_of_node(node));
      }

      auto& pop_vec = available_buckets[bucket];
      cuopt_assert(pop_vec.size() > 0, "OX pop vec is empty");
      auto vehicle_id = pop_random(pop_vec);

      routes_to_add.push_back({vehicle_id, tmp_node_info});
      tmp_node_info.clear();
    }
    cuopt_func_call(double cost_before = A.get_cost(weights));
    A.add_new_routes(routes_to_add);
    cuopt_func_call(double cost_after = A.get_cost(weights));
    //                Coherence test (check if graph path cost is the same as the inserted
    //                route):
    if (A.problem->data_view_ptr->get_vehicle_locations().first == nullptr) {
      cuopt_assert(abs((cost_after - cost_before) - total_delta) < MOVE_EPSILON,
                   "Cost mismatch on graph and solution");
    }
    return true;
  }

  void test_bellman_ford(Solution const& A)
  {
    std::vector<std::vector<std::tuple<int, double, int>>> h_graph(offspring.size());
    for (size_t i = 0; i < h_graph.size(); ++i) {
      h_graph[i].reserve(problem_size);
    }

    adj_to_host(h_graph);

    std::vector<std::vector<double>> h_path_cost(problem_size + 1);
    // Vector of parents to recreate optimal path
    std::vector<std::vector<int>> h_predcessor(problem_size + 1);
    std::vector<std::vector<int>> h_predcessor_vehicle(problem_size + 1);
    for (size_t i = 0; i < h_path_cost.size(); ++i) {
      h_path_cost[i].reserve(problem_size);
    }
    for (size_t i = 0; i < h_predcessor.size(); ++i) {
      h_predcessor[i].reserve(problem_size);
    }
    for (size_t i = 0; i < h_predcessor_vehicle.size(); ++i) {
      h_predcessor_vehicle[i].reserve(problem_size);
    }

    // ! If the path is limited and not fixed a simpler algorithm can be used
    h_path_cost[0].assign(1, 0.0);
    h_predcessor[0].assign(1, 0);
    h_predcessor_vehicle[0].assign(1, 0);

    for (int i = 0; i < routes_number; i++) {
      h_path_cost[i + 1].assign(offspring.size(), std::numeric_limits<double>::max());
      h_predcessor[i + 1].assign(offspring.size(), -1);
      h_predcessor_vehicle[i + 1].assign(offspring.size(), -1);

      //  ------------- Those two loops must be performed in parallel: ------------
      for (size_t j = i; j < h_path_cost[i].size(); j++) {
        if (h_path_cost[i][j] != std::numeric_limits<double>::max()) {
          veh_cpy = vehicle_availability;
          if (run_heuristic && i != 0) {
            // Traverse the path to 'i' and mark vehicles used in the path to i
            int tmp   = j;
            int i_tmp = i;
            while (tmp > 0) {
              veh_cpy[h_predcessor_vehicle[i_tmp][tmp]]--;

              tmp = h_predcessor[i_tmp][tmp];
              --i_tmp;
            }
          }

          for (size_t k = 0; k < h_graph[j].size(); k++) {
            if (!run_heuristic || veh_cpy[std::get<2>(h_graph[j][k])] > 0) {
              if (h_path_cost[i][j] + std::get<1>(h_graph[j][k]) <
                  h_path_cost[i + 1][std::get<0>(h_graph[j][k])]) {
                h_path_cost[i + 1][std::get<0>(h_graph[j][k])] =
                  h_path_cost[i][j] + std::get<1>(h_graph[j][k]);
                h_predcessor[i + 1][std::get<0>(h_graph[j][k])] = j;
                h_predcessor_vehicle[i + 1][std::get<0>(h_graph[j][k])] =
                  std::get<2>(h_graph[j][k]);
              }
            }
          }
        }
      }
    }

    if (A.problem->fleet_info.is_homogenous_) {
      for (size_t i = 0; i < h_path_cost.size(); ++i) {
        for (size_t j = 0; j < h_path_cost[i].size(); ++j) {
          cuopt_assert(std::abs(h_path_cost[i][j] - path_cost[i][j]) < 0.01, "Path cost mismatch");
        }
      }
    }
  }

  void test_transpose_graph()
  {
    std::vector<std::vector<std::tuple<int, double, int>>> h_transpose_graph(offspring.size());
    for (size_t i = 0; i < h_transpose_graph.size(); ++i) {
      h_transpose_graph[i].reserve(problem_size);
    }

    std::vector<std::vector<std::tuple<int, double, int>>> tmp_graph(offspring.size());
    for (size_t i = 0; i < tmp_graph.size(); ++i) {
      tmp_graph[i].reserve(problem_size);
    }

    adj_to_host(tmp_graph);

    for (size_t i = 0; i < tmp_graph.size(); ++i) {
      for (size_t j = 0; j < tmp_graph[i].size(); ++j) {
        h_transpose_graph[std::get<0>(tmp_graph[i][j])].emplace_back(
          i, std::get<1>(tmp_graph[i][j]), std::get<2>(tmp_graph[i][j]));
      }
    }

    auto tmp_transpose = transpose_graph.to_host();

    for (size_t i = 0; i < h_transpose_graph.size(); ++i) {
      auto transpose_offset =
        i * transpose_graph.n_buckets * transpose_graph.get_max_nodes_per_row();
      cuopt_assert(h_transpose_graph[i].size() == tmp_transpose.row_sizes[i],
                   "Mismatch number of edges");
      for (size_t j = 0; j < h_transpose_graph[i].size(); ++j) {
        auto [ref_edge, ref_weight, ref_veh] = h_transpose_graph[i][j];
        bool found                           = false;
        for (int x = 0; x < tmp_transpose.row_sizes[i]; ++x) {
          auto edge = tmp_transpose.indices[transpose_offset + x];
          auto veh  = tmp_transpose.buckets[transpose_offset + x];
          if (edge == ref_edge && veh == ref_veh) {
            found       = true;
            auto weight = tmp_transpose.weights[transpose_offset + x];
            cuopt_assert(std::abs(ref_weight - weight) < 0.01, "Mismatch weights");
          }
        }
        cuopt_assert(found, "Edge not found");
      }
    }
  }

  // Sort edges for coalesced accesses in bellman ford
  template <typename i_t, typename f_t>
  void sort_graph_edges(Solution const& A, ox_graph_t<i_t, f_t>& graph)
  {
    raft::common::nvtx::range fun_scope("ox_sort_graph");

    auto stream_view = A.sol.sol_handle->get_stream();
    auto policy      = A.sol.sol_handle->get_thrust_policy();
    auto row_size    = graph.n_buckets * graph.get_max_nodes_per_row();
    val_map.resize(row_size * graph.get_num_vertices(), stream_view);
    row_offsets.resize(graph.get_num_vertices() + 1, stream_view);
    gather_double.resize(row_size * graph.get_num_vertices(), stream_view);
    gather_int.resize(row_size * graph.get_num_vertices(), stream_view);
    thrust::sequence(policy, val_map.begin(), val_map.end());
    thrust::sequence(policy, row_offsets.begin(), row_offsets.end(), 0, static_cast<int>(row_size));

    size_t tmp_storage_bytes{0};
    auto num_segments = graph.get_num_vertices();
    auto num_items    = row_size * graph.get_num_vertices();
    cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                        tmp_storage_bytes,
                                        graph.indices.data(),
                                        graph.indices.data(),
                                        val_map.data(),
                                        val_map.data(),
                                        num_items,
                                        num_segments,
                                        row_offsets.data(),
                                        row_offsets.data() + 1,
                                        stream_view);
    d_tmp_storage_bytes.resize(tmp_storage_bytes, stream_view);
    cub::DeviceSegmentedSort::SortPairs(d_tmp_storage_bytes.data(),
                                        tmp_storage_bytes,
                                        graph.indices.data(),
                                        graph.indices.data(),
                                        val_map.data(),
                                        val_map.data(),
                                        num_items,
                                        num_segments,
                                        row_offsets.data(),
                                        row_offsets.data() + 1,
                                        stream_view);
    RAFT_CHECK_CUDA(stream_view);

    thrust::gather(policy, val_map.begin(), val_map.end(), graph.buckets.data(), gather_int.data());
    thrust::gather(
      policy, val_map.begin(), val_map.end(), graph.weights.data(), gather_double.data());

    raft::copy(graph.weights.data(), gather_double.data(), gather_double.size(), stream_view);
    raft::copy(graph.buckets.data(), gather_int.data(), gather_int.size(), stream_view);
  }

  void compute_transpose_graph(Solution const& A)
  {
    raft::common::nvtx::range fun_scope("transpose_graph");
    constexpr auto const TPB = 128;
    auto const n_blocks      = n_buckets * d_graph.get_num_vertices();

    transpose_graph.reset(A.sol.sol_handle);
    transpose_graph_kernel<int, float><<<n_blocks, TPB, 0, A.sol.sol_handle->get_stream()>>>(
      d_graph.view(), transpose_graph.view(), max_route_len);
    RAFT_CHECK_CUDA(A.sol.sol_handle->get_stream());
    sort_graph_edges<int, float>(A, transpose_graph);
  }

  //! \brief{  path_cost[k][l] contains the min cost of a path  0->..->l of length at most k
  //! (std::numeric_limits<double>::max() if doesn't exist). Input is a topologically sorted DAG
  //!          }
  void bellman_ford(Solution& A)
  {
    raft::common::nvtx::range fun_scope("bellman_ford");

    compute_transpose_graph(A);
    cuopt_func_call(test_transpose_graph());

    auto row_size = offspring.size();
    d_path_cost.resize((problem_size + 1) * row_size, A.sol.sol_handle->get_stream());
    d_predecessor.resize((problem_size + 1) * row_size, A.sol.sol_handle->get_stream());
    d_predecessor_vehicle.resize((problem_size + 1) * row_size, A.sol.sol_handle->get_stream());
    auto max_val = std::numeric_limits<int>::max();
    async_fill(d_path_cost, std::numeric_limits<double>::max(), A.sol.sol_handle->get_stream());
    async_fill(d_predecessor, -1, A.sol.sol_handle->get_stream());
    async_fill(d_predecessor_vehicle, -1, A.sol.sol_handle->get_stream());
    bellman_ford_init<int, double><<<1, 1, 0, A.sol.sol_handle->get_stream()>>>(
      raft::device_span<double>(d_path_cost.data(), d_path_cost.size()),
      raft::device_span<int>(d_predecessor.data(), d_predecessor.size()),
      raft::device_span<int>(d_predecessor_vehicle.data(), d_predecessor_vehicle.size()));
    RAFT_CHECK_CUDA(A.sol.sol_handle->get_stream());

    constexpr auto const TPB     = 128;
    auto min_cost_of_last_column = std::numeric_limits<double>::max();
    auto cost_of_last_column     = std::numeric_limits<double>::max();
    const auto& dimensions_info  = A.problem->dimensions_info;

    cuopt::device_copy(
      d_vehicle_availability, vehicle_availability, A.sol.sol_handle->get_stream());

    for (int i = 1; i <= routes_number; ++i) {
      auto n_blocks = transpose_graph.get_num_vertices() - i;
      // routes number exceeds num nodes. Stop the search here
      if (n_blocks == 0) { break; }
      bellman_ford_kernel<int, float, Solution::request_type>
        <<<n_blocks, TPB, 0, A.sol.sol_handle->get_stream()>>>(
          A.sol.view(),
          transpose_graph.view(),
          raft::device_span<double>(d_path_cost.data(), d_path_cost.size()),
          raft::device_span<int>(d_predecessor.data(), d_predecessor.size()),
          raft::device_span<int>(d_predecessor_vehicle.data(), d_predecessor_vehicle.size()),
          raft::device_span<int>(d_vehicle_availability.data(), d_vehicle_availability.size()),
          row_size,
          i,
          run_heuristic);
      RAFT_CHECK_CUDA(A.sol.sol_handle->get_stream());

      if (optimal_routes_search) {
        raft::copy(&cost_of_last_column,
                   d_path_cost.data() + ((i + 1) * row_size) - 1,
                   1,
                   A.sol.sol_handle->get_stream());
        if (cost_of_last_column < min_cost_of_last_column) {
          min_cost_of_last_column = cost_of_last_column;
          optimal_routes_number   = i;
        }
      }
    }
    A.sol.sol_handle->sync_stream();

    for (size_t i = 0; i < path_cost.size(); ++i) {
      path_cost[i].resize(row_size);
    }
    for (size_t i = 0; i < predcessor.size(); ++i) {
      predcessor[i].resize(row_size);
    }
    for (size_t i = 0; i < predcessor_vehicle.size(); ++i) {
      predcessor_vehicle[i].resize(row_size);
    }
    // Fill cpu structure
    for (int i = 0; i <= routes_number; ++i) {
      raft::copy(path_cost[i].data(),
                 d_path_cost.data() + i * row_size,
                 row_size,
                 A.sol.sol_handle->get_stream());
    }

    for (int i = 0; i <= routes_number; ++i) {
      raft::copy(predcessor[i].data(),
                 d_predecessor.data() + i * row_size,
                 row_size,
                 A.sol.sol_handle->get_stream());
    }

    for (int i = 0; i <= routes_number; ++i) {
      raft::copy(predcessor_vehicle[i].data(),
                 d_predecessor_vehicle.data() + i * row_size,
                 row_size,
                 A.sol.sol_handle->get_stream());
    }

    cuopt_func_call(test_bellman_ford(A));
  }

  //! \brief{  Fill the nodes from changed routes}
  void fill_genome(Solution& S, std::vector<int>& genome, std::vector<int>& routes_id)
  {
    genome.clear();
    for (auto& b : routes_id) {
      auto& a   = S.get_routes()[b];
      int start = a.start.node();
      int n_end = S.succ[a.end.node()].node();

      while (start != n_end) {
        genome.push_back(start);
        start = S.succ[start].node();
      }
    }
  }

  //! \brief{ Create offspring according to improved OX logic }
  void fill_offspring(Solution const& S,
                      std::vector<int> const& routesS,
                      bool cycle_rotate_to_start = true)
  {
    helper.clear();
    offspring.clear();
    offspring.assign(genome_A.size(), 0);
    offspring_tmp.clear();
    offspring_tmp.assign(genome_A.size(), 0);

    // Choose random fragment from genome B and inject to genome A
    int i = 0;
    if (S.routes.size() == 1) { i = next_random() % (genome_A.size() - 1); }
    int j = i + 1 + (next_random() % (3 * max_route_len));
    if ((size_t)j >= genome_A.size() - 1) {
      j = i + next_random() % std::max(1, (((int)genome_A.size() - 1 - i) / 2));
    }

    for (int k = i; k <= j; k++) {
      offspring[k] = genome_B[k];
      helper.insert(genome_B[k]);
    }

    int last_node_ind = find(genome_A.begin(), genome_A.end(), genome_B[j]) - genome_A.begin();
    int start_ind     = (1 + last_node_ind) % genome_A.size();

    int offspring_ind = (j + 1) % offspring.size();
    for (int k = start_ind; k != last_node_ind; k = (k + 1) % genome_A.size()) {
      if (helper.count(genome_A[k]) == 0) {
        offspring[offspring_ind] = genome_A[k];
        helper.insert(genome_A[k]);
        offspring_ind = (offspring_ind + 1) % offspring.size();
      }
    }

    // Cycle the offspring until we find some route start
    j = 0;
    if (cycle_rotate_to_start) {
      helper.clear();
      for (auto& a : routesS) {
        auto& b = S.get_routes()[a];
        helper.insert(b.start.node());
      }

      for (size_t i = 0; i < offspring.size(); i++) {
        if (helper.count(offspring[i]) > 0) {
          j = i;
          break;
        }
      }
    }
    int tmp = 0;
    // Copy rotated offspring
    for (size_t k = j;; k = (k + 1) % offspring.size()) {
      offspring_tmp[tmp++] = offspring[k];
      if ((j == 0 && k == offspring.size() - 1) || (int)k == j - 1) { break; }
    }

    offspring.push_back(0);
    // First node set to dummy -> calculate_edges will fill it with the depo
    for (size_t k = 0; k < offspring_tmp.size(); k++) {
      offspring[k + 1] = offspring_tmp[k];
    }
    offspring[0] = 0;
  }

  void adj_to_host(std::vector<std::vector<std::tuple<int, double, int>>>& h_graph)
  {
    auto tmp_graph = d_graph.to_host();
    for (int veh = 0; veh < n_buckets; ++veh) {
      for (size_t i = 0; i < d_graph.get_num_vertices(); ++i) {
        auto row_size      = tmp_graph.row_sizes[veh * d_graph.get_num_vertices() + i];
        auto global_offset = veh * d_graph.get_num_vertices() * d_graph.get_max_nodes_per_row() +
                             i * d_graph.get_max_nodes_per_row();
        for (int j = 0; j < row_size; ++j) {
          auto edge   = tmp_graph.indices[global_offset + j];
          auto weight = tmp_graph.weights[global_offset + j];
          auto veh    = tmp_graph.buckets[global_offset + j];
          h_graph[i].emplace_back(edge, weight, veh);
        }
      }
    }
  }

  void test_fill_edges(Solution& A)
  {
    graph.resize(offspring.size());
    for (size_t i = 0; i < graph.size(); ++i) {
      graph[i].clear();
    }

    std::vector<std::vector<std::tuple<int, double, int>>> h_graph(offspring.size());
    for (size_t i = 0; i < h_graph.size(); ++i) {
      h_graph[i].reserve(max_route_len);
    }
    adj_to_host(h_graph);

    const auto& dimensions_info = A.problem->dimensions_info;

    // Helper nodes for forward calculation of paths
    node_t<int, float, Solution::request_type> HelperNodes[3] = {
      node_t<int, float, Solution::request_type>(dimensions_info),
      node_t<int, float, Solution::request_type>(dimensions_info),
      node_t<int, float, Solution::request_type>(dimensions_info)};
    // TODO: here we should find optimal DEPOT for each path. This will require storing tuples
    // instead of pairs that contain start_depot, end_depot
    int depot    = 0;
    offspring[0] = depot;

    // ------------- The loop below must be performed in parallel ---------------
    for (int veh = 0; veh < n_buckets; veh++) {
      // Here we have to use the vehicle type 'veh'
      auto vehicle_id             = vehicle_id_per_bucket[veh];
      auto vehicle_info           = A.problem->get_vehicle_info(vehicle_id);
      auto start_depot_node_info  = A.problem->start_depot_node_infos_h[vehicle_id];
      auto return_depot_node_info = A.problem->return_depot_node_infos_h[vehicle_id];
      auto cuopt_weights          = get_cuopt_cost(weights);
      double optimal_vehicle_fixed_cost =
        dimensions_info.has_dimension(dim_t::VEHICLE_FIXED_COST) ? vehicle_info.fixed_cost : 0.;

      // Here we have to use the vehicle type 'veh'
      HelperNodes[2] =
        create_depot_node<int, float, Solution::request_type>(A.problem,
                                                              start_depot_node_info,
                                                              return_depot_node_info,
                                                              vehicle_id);  // fill depot node

      for (size_t i = 0; i < offspring.size() - 1; i++) {
        double cost = 0;

        HelperNodes[0] =
          create_node<int, float, Solution::request_type>(A.problem, offspring[i + 1]);
        HelperNodes[2].calculate_forward_all(HelperNodes[0], vehicle_info);
        bool b = 0;
        for (size_t j = i + 1; j < offspring.size(); j++) {
          if ((int)(j - i) > max_route_len) break;
          // here iterate over vehicles for hetero
          cost = node_t<int, float, Solution::request_type>::cost_combine(HelperNodes[b],
                                                                          HelperNodes[2],
                                                                          vehicle_info,
                                                                          true,
                                                                          cuopt_weights,
                                                                          objective_cost_t{},
                                                                          infeasible_cost_t{});
          graph[i].emplace_back(j, optimal_vehicle_fixed_cost + cost, veh);
          if (j + 1 == offspring.size()) break;

          HelperNodes[!b] =
            create_node<int, float, Solution::request_type>(A.problem, offspring[j + 1]);
          HelperNodes[b].calculate_forward_all(HelperNodes[!b], vehicle_info);

          b = !b;
        }
      }
    }

    for (size_t i = 0; i < graph.size(); ++i) {
      cuopt_assert(graph[i].size() == h_graph[i].size(), "Mismatch number of edges");
      for (size_t j = 0; j < graph[i].size(); ++j) {
        auto [ref_edge, ref_weight, ref_veh] = graph[i][j];
        auto [edge, weight, veh]             = h_graph[i][j];
        cuopt_assert(ref_edge == edge, "Edge mismatch");
        cuopt_assert(std::abs(ref_weight - weight) < 0.01, "Weight mismatch");
        cuopt_assert(ref_veh == veh, "Vehicle type mismatch");
      }
    }
  }

  //! \brief{  Calculates the edges costs. TODO: Include vehicle costs and different possible
  //! depots. }
  void calculate_edge_costs(Solution& A)
  {
    raft::common::nvtx::range fun_scope("calculate_edge_costs");

    d_graph.reset(A.sol.sol_handle);

    auto shmem      = max_route_len * sizeof(int) + max_route_len * sizeof(double);
    auto gpu_weight = get_cuopt_cost(weights);

    d_offspring.resize(offspring.size(), A.sol.sol_handle->get_stream());
    d_offspring.shrink_to_fit(A.sol.sol_handle->get_stream());
    raft::copy(
      d_offspring.data(), offspring.data(), offspring.size(), A.sol.sol_handle->get_stream());

    auto const n_blocks = n_buckets * (d_offspring.size() - 1);

    if (!set_shmem_of_kernel(calculate_edge_costs_kernel<int, float, Solution::request_type>,
                             shmem)) {
      cuopt_assert(false, "Not enough shared memory in recombiner");
      return;
    }
    calculate_edge_costs_kernel<int, float, Solution::request_type>
      <<<n_blocks, 128, shmem, A.sol.sol_handle->get_stream()>>>(
        A.sol.view(),
        d_graph.view(),
        raft::device_span<int>(d_offspring.data(), d_offspring.size()),
        raft::device_span<int>(d_vehicle_id_per_bucket.data(), d_vehicle_id_per_bucket.size()),
        max_route_len,
        gpu_weight);
    RAFT_CHECK_CUDA(A.sol.sol_handle->get_stream());
    A.sol.sol_handle->sync_stream();

    if (A.problem->data_view_ptr->get_vehicle_locations().first == nullptr) {
      cuopt_func_call(test_fill_edges(A));
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
