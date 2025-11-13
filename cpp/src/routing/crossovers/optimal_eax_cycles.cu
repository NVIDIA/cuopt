/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cfloat>
#include <cmath>

#include "optimal_eax_cycles.cuh"

namespace cuopt::routing::detail {

template <typename i_t, typename f_t, request_t REQUEST>
optimal_cycles_t<i_t, f_t, REQUEST>::optimal_cycles_t(allocator& pool_allocator_)
  : pool_allocator(pool_allocator_),
    d_cycle(0, pool_allocator_.sol_handles[0]->get_stream()),
    eax_cycle_delta(0, pool_allocator_.sol_handles[0]->get_stream()),
    d_cub_storage_bytes(0, pool_allocator_.sol_handles[0]->get_stream()),
    index_delta_pair(pool_allocator_.sol_handles[0]->get_stream()),
    eax_fragment(pool_allocator_.sol_handles[0].get(), pool_allocator_.problem.dimensions_info)
{
}

template <typename i_t, typename f_t>
__global__ void create_rotations_kernel(
  typename solution_t<i_t, f_t, request_t::VRP>::view_t const sol,
  const raft::device_span<NodeInfo<>> d_cycle,
  typename dimensions_route_t<i_t, f_t, request_t::VRP>::view_t eax_fragment,
  i_t n_rotations)
{
  for (i_t i = threadIdx.x; i < n_rotations; i += blockDim.x) {
    auto node_info = d_cycle[i];
    auto node      = create_node<i_t, f_t, request_t::VRP>(sol.problem, node_info, node_info);
    eax_fragment.set_node(i, node);
  }
}

template <typename i_t, typename f_t>
__global__ void find_optimal_position_kernel(
  const typename solution_t<i_t, f_t, request_t::VRP>::view_t sol,
  const typename move_candidates_t<i_t, f_t>::view_t move_candidates,
  const typename dimensions_route_t<i_t, f_t, request_t::VRP>::view_t eax_fragment,
  i_t n_rotations,
  raft::device_span<double> eax_cycle_delta)
{
  i_t th_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (th_id >= (sol.get_num_orders() + sol.n_routes) * n_rotations) { return; }
  i_t insertion_node  = th_id / n_rotations;
  i_t rotation        = th_id % n_rotations;
  const i_t frag_size = n_rotations;
  if (insertion_node == 0) {
    eax_cycle_delta[th_id] = std::numeric_limits<double>::max();
    return;
  }

  i_t route_id;
  i_t insertion_idx;
  if (insertion_node >= sol.get_num_orders()) {
    route_id      = insertion_node - sol.get_num_orders();
    insertion_idx = 0;
  } else {
    thrust::tie(route_id, insertion_idx) =
      sol.route_node_map.get_route_id_and_intra_idx(insertion_node);
  }
  // return for unrouted nodes
  if (route_id < 0) {
    eax_cycle_delta[th_id] = std::numeric_limits<double>::max();
    return;
  }
  const auto& route = sol.routes[route_id];
  auto temp_node    = route.get_node(insertion_idx);
  for (i_t i = 0; i < frag_size; ++i) {
    // so that reads are coalesced
    i_t rotated_idx = (rotation + i) % n_rotations;
    auto next_node  = eax_fragment.get_node(rotated_idx);
    temp_node.calculate_forward_all(next_node, route.vehicle_info());
    temp_node = next_node;
  }
  auto end_node = route.get_node(insertion_idx + 1);
  double delta  = temp_node.calculate_forward_all_and_delta(end_node,
                                                           route.vehicle_info(),
                                                           move_candidates.include_objective,
                                                           move_candidates.weights,
                                                           route.get_objective_cost(),
                                                           route.get_infeasibility_cost());
  // store the best delta
  eax_cycle_delta[th_id] = delta;
}

template <typename i_t, typename f_t>
__global__ void insert_optimal_rotation_kernel(
  typename solution_t<i_t, f_t, request_t::VRP>::view_t sol,
  const cub::KeyValuePair<i_t, double>* index_delta_pair,
  const typename dimensions_route_t<i_t, f_t, request_t::VRP>::view_t eax_fragment,
  i_t n_rotations)
{
  extern __shared__ i_t shmem[];
  i_t index          = index_delta_pair->key;
  i_t insertion_node = index / n_rotations;
  i_t rotation       = index % n_rotations;
  i_t route_id;
  i_t insertion_idx;
  if (insertion_node >= sol.get_num_orders()) {
    route_id      = insertion_node - sol.get_num_orders();
    insertion_idx = 0;
  } else {
    route_id      = sol.route_node_map.route_id_per_node[insertion_node];
    insertion_idx = sol.route_node_map.intra_route_idx_per_node[insertion_node];
  }
  cuopt_assert(route_id >= 0, "Route id should not be negative");
  auto& route = sol.routes[route_id];
  // load fragments from the routes
  i_t depot_excluded_max_route_size = sol.get_max_active_nodes_for_all_routes() + n_rotations - 1;
  auto s_route = route_t<i_t, f_t, request_t::VRP>::view_t::create_shared_route(
    (i_t*)shmem, route, depot_excluded_max_route_size);
  // copy until/including insertion_idx, copy fragment, copy rest of the route
  __syncthreads();
  // copy including insertion index
  s_route.copy_from(route, 0, insertion_idx + 1, 0);
  __syncthreads();
  for (i_t i = threadIdx.x; i < n_rotations; i += blockDim.x) {
    i_t rotated_idx = (rotation + i) % n_rotations;
    s_route.set_node(i + insertion_idx + 1, eax_fragment.get_node(rotated_idx));
  }
  __syncthreads();
  s_route.copy_from(
    route, insertion_idx + 1, route.get_num_nodes() + 1, insertion_idx + 1 + n_rotations);
  __syncthreads();
  if (threadIdx.x == 0) {
    s_route.set_num_nodes(route.get_num_nodes() + n_rotations);
    sol.routes_to_copy[s_route.get_id()]   = 1;
    sol.routes_to_search[s_route.get_id()] = 1;
  }
  __syncthreads();
  route_t<i_t, f_t, request_t::VRP>::view_t::compute_forward_backward_cost(s_route);
  __syncthreads();
  route.copy_from(s_route);
}

template <typename i_t, typename f_t, request_t REQUEST>
void optimal_cycles_t<i_t, f_t, REQUEST>::get_min_delta_and_index(
  adapted_sol_t<i_t, f_t, REQUEST>& sol, i_t num_items)
{
  raft::common::nvtx::range fun_scope("get_min_delta_and_index");
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMin(static_cast<void*>(nullptr),
                            temp_storage_bytes,
                            eax_cycle_delta.data(),
                            index_delta_pair.data(),
                            num_items,
                            sol.sol.sol_handle->get_stream());
  // Allocate temporary storage
  if (d_cub_storage_bytes.size() < temp_storage_bytes) {
    d_cub_storage_bytes.resize(temp_storage_bytes, sol.sol.sol_handle->get_stream());
  }
  // Run argmin-reduction
  cub::DeviceReduce::ArgMin(d_cub_storage_bytes.data(),
                            temp_storage_bytes,
                            eax_cycle_delta.data(),
                            index_delta_pair.data(),
                            num_items,
                            sol.sol.sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
bool optimal_cycles_t<i_t, f_t, REQUEST>::insert_cycle_to_found_position(
  adapted_sol_t<i_t, f_t, REQUEST>& sol, i_t n_rotations)
{
  raft::common::nvtx::range fun_scope("insert_cycle_to_found_position");
  auto& solution    = sol.sol;
  size_t sh_size    = solution.check_routes_can_insert_and_get_sh_size(n_rotations);
  constexpr i_t TPB = 128;

  if (!set_shmem_of_kernel(insert_optimal_rotation_kernel<i_t, f_t>, sh_size)) {
    cuopt_assert(false, "Not enough shared memory in insert_cycle_to_found_position");
    return false;
  }
  // prepare the rotations once and copy them to respective device arrays
  insert_optimal_rotation_kernel<i_t, f_t><<<1, TPB, sh_size, solution.sol_handle->get_stream()>>>(
    solution.view(), index_delta_pair.data(), eax_fragment.view(), n_rotations);
  solution.compute_route_id_per_node();
  solution.compute_cost();
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool>>
bool optimal_cycles_t<i_t, f_t, REQUEST>::add_cycles_request(
  adapted_sol_t<i_t, f_t, REQUEST>& sol,
  std::vector<std::vector<NodeInfo<>>>& cycles,
  costs final_weight)
{
  raft::common::nvtx::range fun_scope("add_cycles_request_vrp");
  auto [resource, index] = pool_allocator.resource_pool->acquire();
  auto gpu_weight        = get_cuopt_cost(final_weight);
  resource.ls.set_active_weights(gpu_weight, std::numeric_limits<f_t>::max());
  auto& solution      = sol.sol;
  i_t total_positions = (solution.get_num_orders() + solution.n_routes);
  for (auto& cycle : cycles) {
    // dynamic resizing
    if (d_cycle.size() < cycle.size()) {
      d_cycle.resize(cycle.size(), solution.sol_handle->get_stream());
      eax_fragment.resize(cycle.size());
    }
    // Number of routes in sol can change after recombination
    eax_cycle_delta.resize(total_positions * cycle.size(), solution.sol_handle->get_stream());

    // copy cycle to device
    raft::copy(d_cycle.data(), cycle.data(), cycle.size(), solution.sol_handle->get_stream());

    const i_t n_rotations = cycle.size();
    const i_t n_positions = solution.get_num_orders() + solution.n_routes;

    constexpr i_t TPB = 128;
    // prepare the rotations once and copy them to respective device arrays
    create_rotations_kernel<i_t, f_t><<<1, TPB, 0, solution.sol_handle->get_stream()>>>(
      solution.view(),
      raft::device_span<NodeInfo<>>(d_cycle.data(), d_cycle.size()),
      eax_fragment.view(),
      n_rotations);

    i_t n_blocks = (n_rotations * n_positions + TPB - 1) / TPB;
    find_optimal_position_kernel<i_t, f_t><<<n_blocks, TPB, 0, solution.sol_handle->get_stream()>>>(
      solution.view(),
      resource.ls.move_candidates.view(),
      eax_fragment.view(),
      n_rotations,
      raft::device_span<double>(eax_cycle_delta.data(), eax_cycle_delta.size()));
    get_min_delta_and_index(sol, n_rotations * n_positions);
    bool success = insert_cycle_to_found_position(sol, n_rotations);

    if (!success) {
      solution.sol_handle->sync_stream();
      pool_allocator.resource_pool->release(index);
      return false;
    }
  }
  solution.sol_handle->sync_stream();
  pool_allocator.resource_pool->release(index);
  sol.populate_host_data();

  return true;
}

/*! \brief { Find best cycle order based on number of order violations for the PDP variant }*/
template <typename i_t, typename f_t, request_t REQUEST>
void optimal_cycles_t<i_t, f_t, REQUEST>::find_best_rotate_cycle(
  std::vector<NodeInfo<>>& cycle, adapted_sol_t<i_t, f_t, REQUEST>& s)
{
  raft::common::nvtx::range fun_scope("find_best_rotate_cycle");
  cuopt_assert(REQUEST == request_t::PDP, "A runtime error occurred in find_best_rotate_cycle!");
  cuopt_expects(REQUEST == request_t::PDP, error_type_t::RuntimeError, "A runtime error occurred!");
  cycle_helper.resize(cycle.size(), NodeInfo<>());
  best_so_far.resize(cycle.size(), NodeInfo<>());
  std::copy(cycle.begin(), cycle.end(), cycle_helper.begin());

  int random_cnt = 1;
  int best_score = INT_MAX;
  for (int k = 0; k < (int)cycle.size(); k++) {
    in_cycle.clear();
    for (size_t index = 0; index < cycle_helper.size(); index++)
      in_cycle[cycle_helper[index].node()] = index;

    // Find the score of a given cycle order
    int score = 0;
    for (size_t index = 0; index < cycle_helper.size(); index++) {
      auto node    = cycle_helper[index];
      auto brother = s.problem->get_brother_node_info(node);

      if (node.is_pickup() && in_cycle.count(brother.node()) > 0 &&
          in_cycle[brother.node()] < index) {
        score++;
      }
    }

    bool record = false;
    if (score < best_score) {
      record     = true;
      random_cnt = 1;
    } else if (score == best_score) {
      random_cnt++;
      if (next_random() % random_cnt == 0) record = true;
    }
    if (record) {
      best_score  = score;
      best_so_far = cycle_helper;
    }

    auto val = cycle_helper.back();
    cycle_helper.pop_back();
    cycle_helper.push_front(val);
  }
  std::copy(best_so_far.begin(), best_so_far.end(), cycle.begin());
}

/*! \brief { Find best insert position of cycle to route in a solution. First criterion of
 * minimization is lowest order violation, second is distance }*/
template <typename i_t, typename f_t, request_t REQUEST>
void optimal_cycles_t<i_t, f_t, REQUEST>::insert_cycle_to_route_request(
  std::vector<NodeInfo<>>& cycle, size_t route_id, adapted_sol_t<i_t, f_t, REQUEST>& s)
{
  raft::common::nvtx::range fun_scope("insert_cycle_to_route_PDP");
  cuopt_assert(REQUEST == request_t::PDP,
               "A runtime error occurred in insert_cycle_to_route_request!");
  cuopt_expects(REQUEST == request_t::PDP, error_type_t::RuntimeError, "A runtime error occurred!");
  int best_score        = INT_MAX;
  double best_sec_score = DBL_MAX;

  std::pair<NodeInfo<>, NodeInfo<>> between;
  auto& route = s.get_route(route_id);

  int vehicle_id                = route.vehicle_id;
  detail::NodeInfo<> prev_start = s.problem->get_start_depot_node_info(vehicle_id);
  auto start                    = route.start;
  int node_index                = 0;
  do {
    int score = 0;
    for (auto node : cycle) {
      auto brother = s.problem->get_brother_node_info(node);
      if (!s.unserviced(brother.node()) && s.nodes[brother.node()].r_id == route_id) {
        if (s.problem->is_pickup(node.node()) &&
            (int)s.nodes[brother.node()].r_index <= node_index) {
          score++;
        }
        if (!s.problem->is_pickup(node.node()) &&
            (int)s.nodes[brother.node()].r_index > node_index) {
          score++;
        }
      }
    }
    double sec_core = s.problem->distance_between(prev_start, cycle[0], vehicle_id) +
                      s.problem->distance_between(cycle.back(), start, vehicle_id) -
                      s.problem->distance_between(prev_start, start, vehicle_id);
    if (score < best_score || (score == best_score && sec_core < best_sec_score)) {
      between.first  = prev_start;
      between.second = start;
      best_score     = score;
      best_sec_score = sec_core;
    }
    prev_start = start;
    start      = s.succ[start.node()];
    node_index++;
  } while (!prev_start.is_depot());
  // insert a whole cycle after the between.first
  s.add_nodes_to_route(cycle, between.first, between.second);
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool>>
bool optimal_cycles_t<i_t, f_t, REQUEST>::add_cycles_request(
  adapted_sol_t<i_t, f_t, REQUEST>& a,
  std::vector<std::vector<NodeInfo<>>>& cycles,
  [[maybe_unused]] costs final_weight)
{
  auto& routes = a.get_routes();
  for (auto& cycle : cycles) {
    int min_sized_route        = 0;
    int size_of_smallest_route = routes[0].length;
    for (size_t route_index = 1; route_index < routes.size(); route_index++) {
      if (routes[route_index].length < size_of_smallest_route) {
        min_sized_route        = route_index;
        size_of_smallest_route = routes[route_index].length;
      }
    }
    // index contains best route to insert the cycle, now find optimal cut ( by minimising order
    // violation of that cycle )
    find_best_rotate_cycle(cycle, a);
    // Now insert the cycle in optimal orded to optimal (in the PDP sense) route.
    insert_cycle_to_route_request(cycle, min_sized_route, a);
  }

  return true;
}

// Don't instantiate the struct as there are many exclusive functions for PDP VRP
// only instantiate the ctr and the main functino add_cycles_request

template optimal_cycles_t<int, float, request_t::PDP>::optimal_cycles_t(allocator& pool_allocator_);
template optimal_cycles_t<int, float, request_t::VRP>::optimal_cycles_t(allocator& pool_allocator_);

template bool optimal_cycles_t<int, float, request_t::VRP>::add_cycles_request(
  adapted_sol_t<int, float, request_t::VRP>& sol,
  std::vector<std::vector<NodeInfo<>>>& cycles,
  costs final_weight);

template bool optimal_cycles_t<int, float, request_t::PDP>::add_cycles_request(
  adapted_sol_t<int, float, request_t::PDP>& sol,
  std::vector<std::vector<NodeInfo<>>>& cycles,
  costs final_weight);

}  // namespace cuopt::routing::detail
