/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/vector_helpers.cuh>
#include "../solution/solution.cuh"

#include <utilities/vector_helpers.cuh>

#include <thrust/logical.h>
#include <functional>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void time_window_check(typename solution_t<i_t, f_t, REQUEST>::view_t solution)
{
  i_t route_id     = blockIdx.x;
  const auto route = solution.routes[route_id];
  cuopt_assert(route.get_num_nodes() > 0, "Invalid route size");
  for (int i = 0; i < route.get_num_nodes() + 1; ++i) {
    if (route.template get_dim<dim_t::TIME>().window_end[i] == 0)
      printf("Window end is 0 is route %d at pos %d\n", route_id, i);
    cuopt_assert(route.template get_dim<dim_t::TIME>().window_end[i] > 0,
                 "Window end can't be 0 or negative");
    cuopt_assert(route.template get_dim<dim_t::TIME>().window_start[i] >= 0,
                 "Window start can't be negative");
    cuopt_assert(route.template get_dim<dim_t::TIME>().window_start[i] <
                     route.template get_dim<dim_t::TIME>().window_end[0] &&
                   route.template get_dim<dim_t::TIME>().window_start[i] <
                     route.template get_dim<dim_t::TIME>().window_end[route.get_num_nodes()],
                 "Window start can't be greater than depot window end");
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void feasibility_check(typename solution_t<i_t, f_t, REQUEST>::view_t solution)
{
  i_t route_id     = blockIdx.x;
  const auto route = solution.routes[route_id];
  i_t vehicle_id   = route.get_vehicle_id();
  i_t n_nodes      = route.get_num_nodes();

  cuopt_assert(route.is_valid(), "Invalid route");

  for (int i = 0; i < n_nodes + 1; ++i) {
    auto node = route.get_node(i);
    if (!node.forward_feasible(route.vehicle_info())) {
      printf("i %d n_nodes %d f_excess :%f\n", i, n_nodes, node.time_dim.excess_forward);
    }
    cuopt_assert(node.forward_feasible(route.vehicle_info()), "All nodes should be feasible");
    if (i < n_nodes) {
      bool res = node_t<i_t, f_t, REQUEST>::combine(
        node, route.get_node(i + 1), route.vehicle_info(), d_default_weights, 0.001);
      if (!res) {
        *solution.sol_found = 0;
        cuopt_assert(true,
                     printf("Failed node:%d f_excess:%f b_excess:%f route:%d, vehicle:%d "
                            "n_nodes:%d\n",
                            node.request.info.node(),
                            node.time_dim.excess_forward,
                            node.time_dim.excess_backward,
                            route_id,
                            vehicle_id,
                            n_nodes));
        cuopt_assert(false, "Node should be feasible combine");
        return;
      }
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void node_global_coherence_check(typename solution_t<i_t, f_t, REQUEST>::view_t solution)
{
  i_t route_id     = blockIdx.x;
  const auto route = solution.routes[route_id];

  auto& route_node_map = solution.route_node_map;

  cuopt_assert(route.get_num_nodes() >= 1, "Invalid route size");
  if (route.get_id() != route_id) {
    printf("var route_id %d index route_id %d\n", route.get_id(), route_id);
    return;
  }
  cuopt_assert(route.get_id() == route_id, "Route id mismatch!");
  // Not checking first / last depot
  for (int i = 1; i < route.get_num_nodes(); ++i) {
    auto node                  = route.get_node(i);
    auto node_info             = node.node_info();
    const bool is_service_node = node_info.is_service_node();

    if (!is_service_node) { continue; }

    auto [my_route_id, my_intra_route_idx] = route_node_map.get_route_id_and_intra_idx(node_info);
    if (my_route_id != route_id) {
      printf("Route %d node %d at pos %d appears as being %d, node type: %d\n",
             route_id,
             node.id(),
             i,
             my_route_id,
             (int)node_info.node_type());
    }
    cuopt_assert(my_route_id == route_id, "route_id_per_node is incorrect");
    if (my_intra_route_idx != i) {
      printf(
        "Incorrect intra_route_idx_per_node in route %d for node %d at pos %d, has %d and should "
        "have %d\n",
        route_id,
        node.id(),
        i,
        my_intra_route_idx,
        i);
    }

    cuopt_assert(my_intra_route_idx == i, "intra_route_idx_per_node is incorrect");
    if constexpr (REQUEST == request_t::PDP) {
      bool found = false;
      for (int j = 0; j < solution.get_num_requests(); ++j) {
        if (node.request.is_pickup()) {
          if (solution.problem.pickup_indices[j] == node.id()) found = true;
        } else if (node.request.is_delivery()) {
          if (solution.problem.delivery_indices[j] == node.id()) found = true;
        }
      }
      if (!found) {
        printf(
          "Wrong pickup/delivery info node id for %d in pos %d of route %d.Pickup val of node %d "
          "\n",
          node.id(),
          i,
          route_id,
          node.request.is_pickup());
        cuopt_assert(found, "Incorrect is pickup value");
      }

      i_t node_route_id, node_intra_idx, brother_route_id, brother_intra_idx;

      thrust::tie(brother_route_id, brother_intra_idx) =
        solution.route_node_map.get_route_id_and_intra_idx(node.request.brother_info);

      thrust::tie(node_route_id, node_intra_idx) =
        solution.route_node_map.get_route_id_and_intra_idx(node.node_info());

      cuopt_assert(brother_intra_idx != -1, "Brother should be served");
      cuopt_assert(brother_route_id == route_id, "Node brother should be on the same route");
      cuopt_assert(
        node.request.is_pickup() || route.get_node(brother_intra_idx).request.is_pickup(),
        "Either one or the other node must be a pickup");

      cuopt_assert(node.request.is_pickup() ^ (brother_intra_idx < node_intra_idx),
                   "Pickup should come before the delivery!");
      cuopt_func_call((constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto i) {
        if (i < node.capacity_dim.n_capacity_dimensions) {
          cuopt_assert(node.capacity_dim.demand[i] ==
                         -route.get_node(brother_intra_idx).capacity_dim.demand[i],
                       "The demands should be negation of each other");
        }
      })));
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void fill_histo(typename solution_t<i_t, f_t, REQUEST>::view_t solution, i_t* histo)
{
  i_t route_id              = blockIdx.x;
  const bool depot_included = solution.problem.order_info.depot_included;
  const auto route          = solution.routes[route_id];
  for (int i = 0; i < route.get_num_nodes() + 1; ++i) {
    auto node = route.get_node(i);
    if ((depot_included || !node.node_info().is_depot()) && !node.node_info().is_break()) {
      atomicAdd(&histo[node.id()], 1);
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void check_histogram(const i_t* histogram,
                                i_t size,
                                bool all_nodes_served,
                                bool depot_included)
{
  const i_t id = threadIdx.x + (i_t)depot_included + blockIdx.x * blockDim.x;
  if (id < size) {
    if (all_nodes_served) {
      cuopt_assert(histogram[id] > 0, "All nodes should be served");
      cuopt_assert(histogram[id] < 2, "No node should appear twice");
    } else {
      cuopt_assert(histogram[id] <= 1, "Node should be served at most once!");
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void check_breaks(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                             bool all_nodes_should_be_served)
{
  i_t route_id     = blockIdx.x;
  const auto route = solution.routes[route_id];

  if (route.get_num_breaks() == 0) { return; }

  i_t max_break_dims = solution.problem.get_max_break_dimensions();

  __shared__ i_t shmem_num_breaks;
  if (threadIdx.x == 0) { shmem_num_breaks = 0; }
  __syncthreads();
  // Get the number of breaks in the route
  for (i_t tid = threadIdx.x; tid < route.get_num_nodes(); tid += blockDim.x) {
    auto node = route.get_node(tid);
    if (node.node_info().is_break()) { atomicAdd_block(&shmem_num_breaks, 1); }
  }
  __syncthreads();

  cuopt_assert(shmem_num_breaks == route.get_num_breaks(),
               "Mismatch between get_num_breaks and actual breaks");

  extern __shared__ i_t break_dim_counters[];
  for (i_t tid = threadIdx.x; tid < max_break_dims; tid += blockDim.x) {
    break_dim_counters[tid] = 0;
  }
  __syncthreads();

  for (i_t tid = threadIdx.x; tid < route.get_num_nodes(); tid += blockDim.x) {
    auto node = route.get_node(tid);
    if (node.node_info().is_break()) {
      atomicAdd_block(&break_dim_counters[node.node_info().node()], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (i_t i = 0; i < max_break_dims; ++i) {
      if (break_dim_counters[i] > 1) {
        cuopt_assert(false, "There should be at most one break per dimension!");
      }
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
bool global_runtime_checks_(solution_t<i_t, f_t, REQUEST>& solution,
                            rmm::cuda_stream_view stream,
                            bool all_nodes_should_be_served,
                            bool check_feasible)
{
  if (solution.get_n_routes() < 1) { return true; }
  if (check_feasible) { solution.run_feasibility_check(); }

  solution.run_coherence_check();
  async_fill(solution.runtime_check_histo, 0, solution.sol_handle->get_stream());
  fill_histo<i_t, f_t, REQUEST><<<solution.get_n_routes(), 1, 0, stream>>>(
    solution.view(), solution.runtime_check_histo.data());

  const bool depot_included = solution.problem_ptr->order_info.depot_included_;
  check_histogram<i_t, f_t, REQUEST>
    <<<(solution.get_num_depot_excluded_orders() + 32 - 1) / 32, 32, 0, stream>>>(
      solution.runtime_check_histo.data(),
      solution.get_num_orders(),
      all_nodes_should_be_served,
      depot_included);

  if (solution.problem_ptr->get_max_break_dimensions() > 0) {
    auto sh_size = solution.problem_ptr->get_max_break_dimensions() * sizeof(i_t);
    check_breaks<i_t, f_t, REQUEST><<<solution.get_n_routes(), 32, sh_size, stream>>>(
      solution.view(), all_nodes_should_be_served);
  }

  // We can just return true because cuopt_assert are inside those functions
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::global_runtime_checks(
  [[maybe_unused]] bool all_nodes_should_be_served,
  [[maybe_unused]] bool check_feasible,
  [[maybe_unused]] const std::string_view where)
{
  // #define DEBUG_RUNTIME_CHECKS 1

#ifdef DEBUG_RUNTIME_CHECKS
  // Enable this print for debugging
  std::cout << "Starting checks in: " << where << std::endl;
  cudaDeviceSynchronize();
#endif

  cuopt_assert(global_runtime_checks_(
                 *this, sol_handle->get_stream(), all_nodes_should_be_served, check_feasible),
               "_");

#ifdef DEBUG_RUNTIME_CHECKS
  cudaDeviceSynchronize();
  std::cout << "Done checks in: " << where << std::endl;
#endif
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::run_feasibility_check()
{
  cuopt_func_call((feasibility_check<i_t, f_t, REQUEST>
                   <<<get_n_routes(), 1, 0, sol_handle->get_stream()>>>(view())));
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::run_coherence_check()
{
  cuopt_func_call((node_global_coherence_check<i_t, f_t, REQUEST>
                   <<<get_n_routes(), 1, 0, sol_handle->get_stream()>>>(view())));
}

template void solution_t<int, float, request_t::PDP>::global_runtime_checks(
  bool all_nodes_should_be_served, bool, const std::string_view where);
template void solution_t<int, float, request_t::PDP>::run_feasibility_check();
template void solution_t<int, float, request_t::PDP>::run_coherence_check();

template void solution_t<int, float, request_t::VRP>::global_runtime_checks(
  bool all_nodes_should_be_served, bool, const std::string_view where);
template void solution_t<int, float, request_t::VRP>::run_feasibility_check();
template void solution_t<int, float, request_t::VRP>::run_coherence_check();

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
