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

#include "nodes_to_search.cuh"

#include "../../solution/solution.cuh"
#include "../move_candidates/move_candidates.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void extract_nodes_to_search_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename nodes_to_search_t<i_t, f_t>::view_t nodes_to_search,
  bool restore_phase)
{
  i_t route_id = blockIdx.x;
  if (!sol.routes_to_search[route_id]) { return; }
  const auto& route = sol.routes[route_id];
  if (route.get_num_nodes() == 1) { return; }
  if (!restore_phase || nodes_to_search.active_nodes_impacted[route_id + sol.get_num_orders()]) {
    // add more nodes for the after depot insertion
    for (i_t i = threadIdx.x; i < after_depot_insertion_multiplier; i += blockDim.x) {
      auto start_depot_node_info = route.get_node(0).node_info();
      auto node_info =
        NodeInfo<i_t>(route_id * after_depot_insertion_multiplier + i + sol.get_num_orders(),
                      start_depot_node_info.location(),
                      node_type_t::DEPOT);
      i_t offset                              = atomicAdd(nodes_to_search.n_nodes_to_search, 1);
      nodes_to_search.nodes_to_search[offset] = node_info;
    }
  }

  for (i_t i = threadIdx.x + 1; i < route.get_num_nodes(); i += blockDim.x) {
    auto node_info = route.node_info(i);
    if (node_info.is_service_node()) {
      if (!restore_phase || nodes_to_search.active_nodes_impacted[node_info.node()]) {
        i_t offset                              = atomicAdd(nodes_to_search.n_nodes_to_search, 1);
        nodes_to_search.nodes_to_search[offset] = node_info;
      }
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void run_extract_kernel(solution_t<i_t, f_t, REQUEST>& sol,
                        nodes_to_search_t<i_t, f_t>& nodes_to_search,
                        bool restore_phase)
{
  i_t TPB      = 256;
  i_t n_blocks = sol.get_n_routes();
  extract_nodes_to_search_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, 0, sol.sol_handle->get_stream()>>>(
      sol.view(), nodes_to_search.view(), restore_phase);
}

template <typename i_t>
i_t get_sample_size_vrp(i_t n_of_changed_nodes)
{
  i_t num = 40;
  if (n_of_changed_nodes < num)
    num = n_of_changed_nodes;
  else if (n_of_changed_nodes < num * 2)
    num = n_of_changed_nodes / 2;
  return num;
}

template <typename i_t, typename f_t>
void nodes_to_search_t<i_t, f_t>::restore_found_nodes(solution_t<i_t, f_t, request_t::VRP>& sol)
{
  raft::common::nvtx::range fun_scope("restore_found_nodes");
  constexpr bool restore_phase = true;
  // here we are doing a reuse of n_nodes_to_search for the extracted nodes
  // all other valid nodes to search are already in host side
  // we will append the resulting values to the host vector
  reset(sol.sol_handle);
  run_extract_kernel(sol, *this, restore_phase);
  i_t n_nodes_extracted = n_nodes_to_search.value(sol.sol_handle->get_stream());
  i_t prev_size         = h_nodes_to_search.size();
  h_nodes_to_search.resize(prev_size + n_nodes_extracted);
  // start writing after the current valid values
  raft::copy(h_nodes_to_search.data() + prev_size,
             nodes_to_search.data(),
             n_nodes_extracted,
             sol.sol_handle->get_stream());
  std::sort(h_nodes_to_search.begin(), h_nodes_to_search.end(), [](const auto& a, const auto& b) {
    return a.node() < b.node();
  });
  auto last = std::unique(h_nodes_to_search.begin(), h_nodes_to_search.end());
  h_nodes_to_search.erase(last, h_nodes_to_search.end());
  sol.sol_handle->sync_stream();
}

template <typename i_t, typename f_t>
bool nodes_to_search_t<i_t, f_t>::sample_nodes_for_recycle(
  const solution_t<i_t, f_t, request_t::VRP>& sol, move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("sample_nodes_for_recycle");

  raft::copy(h_best_id_per_node.data(),
             move_candidates.vrp_move_candidates.best_id_per_node.data(),
             h_best_id_per_node.size(),
             sol.sol_handle->get_stream());

  h_recycled_node_pairs.clear();
  for (i_t i = 0; i < (i_t)h_best_id_per_node.size(); ++i) {
    if (h_best_id_per_node[i] != -1) {
      h_recycled_node_pairs.push_back(int2{i, h_best_id_per_node[i]});
    }
  }
  n_sampled_nodes = h_recycled_node_pairs.size();
  if (n_sampled_nodes == 0) { return false; }
  raft::copy(recycled_node_pairs.data(),
             h_recycled_node_pairs.data(),
             n_sampled_nodes,
             sol.sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t>
bool nodes_to_search_t<i_t, f_t>::sample_nodes_to_search(
  const solution_t<i_t, f_t, request_t::VRP>& sol, std::mt19937& rng, bool full_set)
{
  raft::common::nvtx::range fun_scope("sample_nodes_to_search");
  i_t curr_n_nodes_to_search = h_nodes_to_search.size();
  if (curr_n_nodes_to_search == 0) return false;
  if (!full_set) {
    n_sampled_nodes = get_sample_size_vrp<i_t>(curr_n_nodes_to_search);
  } else {
    n_sampled_nodes = curr_n_nodes_to_search;
  }
  cuopt_assert(n_sampled_nodes > 0, "There must be at least one operator!");
  cuopt_assert(curr_n_nodes_to_search > 0, "There must be at least one operator!");
  n_sampled_nodes = std::min(n_sampled_nodes, curr_n_nodes_to_search);
  h_sampled_nodes.clear();
  for (i_t i = 0; i < n_sampled_nodes; ++i) {
    std::uniform_int_distribution<i_t> rng_dist(0, h_nodes_to_search.size() - 1);
    i_t node_idx   = rng_dist(rng);
    auto node_info = h_nodes_to_search[node_idx];
    h_sampled_nodes.push_back(node_info);
    h_nodes_to_search.erase(h_nodes_to_search.begin() + node_idx);
  }
  sample_nodes_graph.start_capture(sol.sol_handle->get_stream());
  raft::copy(sampled_nodes_to_search.data(),
             h_sampled_nodes.data(),
             n_sampled_nodes,
             sol.sol_handle->get_stream());
  reset_active_nodes(sol.sol_handle);
  sample_nodes_graph.end_capture(sol.sol_handle->get_stream());
  sample_nodes_graph.launch_graph(sol.sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
void extract_nodes_to_search(solution_t<i_t, f_t, REQUEST>& sol,
                             move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("extract_nodes_to_search");
  auto& nodes_to_search = move_candidates.nodes_to_search;
  nodes_to_search.extract_nodes_graph.start_capture(sol.sol_handle->get_stream());
  nodes_to_search.reset(sol.sol_handle);
  constexpr bool restore_phase = false;
  run_extract_kernel(sol, nodes_to_search, restore_phase);
  nodes_to_search.extract_nodes_graph.end_capture(sol.sol_handle->get_stream());
  nodes_to_search.extract_nodes_graph.launch_graph(sol.sol_handle->get_stream());
  i_t n_nodes_extracted = nodes_to_search.n_nodes_to_search.value(sol.sol_handle->get_stream());
  nodes_to_search.h_nodes_to_search.resize(n_nodes_extracted);
  raft::copy(nodes_to_search.h_nodes_to_search.data(),
             nodes_to_search.nodes_to_search.data(),
             n_nodes_extracted,
             sol.sol_handle->get_stream());
  sol.sol_handle->sync_stream();
}

template bool nodes_to_search_t<int, float>::sample_nodes_for_recycle(
  const solution_t<int, float, request_t::VRP>& sol,
  move_candidates_t<int, float>& move_candidates);

template bool nodes_to_search_t<int, float>::sample_nodes_to_search(
  const solution_t<int, float, request_t::VRP>& sol, std::mt19937& rng, bool full_set);

template void nodes_to_search_t<int, float>::restore_found_nodes(
  solution_t<int, float, request_t::VRP>& sol);

template void extract_nodes_to_search<int, float, request_t::VRP>(
  solution_t<int, float, request_t::VRP>& sol, move_candidates_t<int, float>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
