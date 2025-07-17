/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <thrust/count.h>
#include <utilities/copy_helpers.hpp>
#include "solution.cuh"
#include "solution_kernels.cuh"
namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
route_t<i_t, f_t, REQUEST>& solution_t<i_t, f_t, REQUEST>::get_route(i_t route_id)
{
  cuopt_assert(route_id < (i_t)route_id_to_idx.size(),
               "route_id should be less than total number of routes");
  cuopt_assert(route_id < (i_t)routes.size(),
               "route_id should be less than total number of routes");
  raft::common::nvtx::range fun_scope("get_route");
  auto idx = route_id_to_idx[route_id];
  return routes[idx];
}

// const overload for const copy ctr
template <typename i_t, typename f_t, request_t REQUEST>
const route_t<i_t, f_t, REQUEST>& solution_t<i_t, f_t, REQUEST>::get_route(i_t route_id) const
{
  cuopt_assert(route_id < (i_t)route_id_to_idx.size(),
               "route_id should be less than total number of routes");
  raft::common::nvtx::range fun_scope("get_route_const");
  auto idx = route_id_to_idx[route_id];
  return routes[idx];
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::check_and_allocate_routes(i_t total_num_routes)
{
  cuopt_assert(routes.size() == route_id_to_idx.size(),
               "routes and route_id_to_idx should have same size");
  cuopt_assert(total_num_routes <= (i_t)problem_ptr->get_fleet_size(),
               "Can not allocate more routes than fleet size!");
  if (total_num_routes > (i_t)routes.size()) {
    i_t num_old_routes         = routes.size();
    i_t num_routes_to_allocate = total_num_routes - num_old_routes;
    for (i_t i = 0; i < num_routes_to_allocate; ++i) {
      // don't initialize vehicle_id
      i_t vehicle_id = -1;
      i_t route_id   = i + num_old_routes;
      route_id_to_idx.push_back(route_id);

      routes.emplace_back(route_t<i_t, f_t, REQUEST>{sol_handle,
                                                     route_id,
                                                     vehicle_id,
                                                     &(problem_ptr->fleet_info),
                                                     problem_ptr->dimensions_info});

      routes[route_id].resize(base_route_size);
      max_nodes_per_route = std::max(max_nodes_per_route, base_route_size);
    }
    cuopt_assert(routes.size() == (size_t)total_num_routes,
                 "routes size should be equal to total_num_routes");
    cuopt_assert(route_id_to_idx.size() == (size_t)total_num_routes,
                 "route_id_to_idx size should be equal to total_num_routes");
    set_route_views();
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::add_route(route_t<i_t, f_t, REQUEST>&& route,
                                              i_t route_id,
                                              i_t n_nodes,
                                              bool check_size)
{
  raft::common::nvtx::range fun_scope("add_route");
  cuopt_assert(!check_size || (n_nodes > request_info_t<i_t, REQUEST>::size()),
               "There should be at least one request in the route!");
  cuopt_expects(!check_size || (n_nodes > request_info_t<i_t, REQUEST>::size()),
                error_type_t::RuntimeError,
                "A runtime error occurred!");
  if constexpr (REQUEST == request_t::PDP) {
    cuopt_assert(!check_size || n_nodes % 2 == 1, "Node count should be odd!");
  }
  cuopt_assert(n_routes < get_num_requests(),
               "Route count cannot be bigger that get_num_requests()!");
  cuopt_assert(route_id == n_routes, "Route ids should be strictly increasing!");
  ++n_routes;
  check_and_allocate_routes(n_routes);

  route.n_nodes.set_value_async(n_nodes, sol_handle->get_stream());
  route.route_id.set_value_async(route_id, sol_handle->get_stream());
  sol_handle->sync_stream();
  i_t route_slot        = route_id_to_idx[route_id];
  routes[route_slot]    = std::move(route);
  const auto route_view = routes[route_slot].view();
  cuopt_assert(route_id < (int)routes_view.size(), "route id should be in range");
  routes_view.set_element_async(route_id, route_view, sol_handle->get_stream());
  if (max_nodes_per_route < get_route(route_id).max_nodes_per_route()) {
    resize_routes(raft::alignTo(get_route(route_id).max_nodes_per_route(), base_route_size));
  }
  sol_handle->sync_stream();
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::add_routes(
  const std::vector<std::pair<int, std::vector<NodeInfo<>>>>& new_routes)
{
  raft::common::nvtx::range fun_scope("add_routes");
  cuopt_assert(new_routes.size() > 0, "There should be at least one route in the vector!");
  thrust::fill(sol_handle->get_thrust_policy(),
               routes_to_search.data() + n_routes,
               routes_to_search.data() + n_routes + new_routes.size(),
               1);

  i_t prev_route_size = n_routes;
  i_t route_id        = n_routes;
  i_t added_routes    = new_routes.size();

  n_routes += added_routes;

  check_and_allocate_routes(n_routes);
  for (const auto& [vehicle_id, route] : new_routes) {
    // depot for the beginning and the end
    i_t new_route_size = route.size() + 2;
    auto& d_route      = get_route(route_id);
    if (new_route_size > d_route.max_nodes_per_route()) {
      new_route_size =
        std::max(max_nodes_per_route, raft::alignTo(new_route_size, base_route_size));
      resize_routes(new_route_size);
    }
    std::vector<NodeInfo<i_t>> node_info_h;
    for (size_t x = 0; x < route.size(); ++x) {
      // Dummy, will be overriden in set_nodes_data_of_route
      node_info_h.emplace_back(route[x]);
    }
    // skip depot
    raft::copy(d_route.dimensions.requests.node_info.data() + 1,
               node_info_h.data(),
               node_info_h.size(),
               sol_handle->get_stream());
    const i_t n_nodes = route.size() + 1;
    d_route.n_nodes.set_value_async(n_nodes, sol_handle->get_stream());
    d_route.route_id.set_value_async(route_id, sol_handle->get_stream());
    d_route.vehicle_id.set_value_async(vehicle_id, sol_handle->get_stream());
    i_t route_slot        = route_id_to_idx[route_id];
    const auto route_view = routes[route_slot].view();
    cuopt_assert(route_id < (int)routes_view.size(), "route id should be in range");
    routes_view.set_element_async(route_id, route_view, sol_handle->get_stream());
    ++route_id;
  }
  set_nodes_data_of_new_routes(added_routes, prev_route_size);
  sol_handle->sync_stream();
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::add_nodes_to_route(
  const std::vector<NodeInfo<>>& nodes_to_insert, i_t route_id, i_t intra_idx)
{
  raft::common::nvtx::range fun_scope("add_nodes_to_route");
  cuopt_assert(route_id > -1, "Route id should be valid");
  const auto n_nodes_to_insert = nodes_to_insert.size();
  raft::copy(
    temp_nodes.data(), nodes_to_insert.data(), n_nodes_to_insert, sol_handle->get_stream());
  size_t sh_size = check_routes_can_insert_and_get_sh_size(n_nodes_to_insert);
  bool is_set    = set_shmem_of_kernel(insert_nodes_to_route_kernel<i_t, f_t, REQUEST>, sh_size);
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  i_t TPB = 256;
  insert_nodes_to_route_kernel<i_t, f_t, REQUEST><<<1, TPB, sh_size, sol_handle->get_stream()>>>(
    view(), route_id, intra_idx, n_nodes_to_insert, temp_nodes.data());
  thrust::fill(sol_handle->get_thrust_policy(),
               routes_to_search.data() + route_id,
               routes_to_search.data() + route_id + 1,
               1);
  sol_handle->sync_stream();
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::add_nodes_to_best(
  const std::vector<NodeInfo<>>& nodes_to_insert, const infeasible_cost_t& weights)
{
  raft::common::nvtx::range fun_scope("add_nodes_to_best");
  i_t TPB                          = 256;
  constexpr bool include_objective = true;
  // TODO check perf and implement parallel insertion
  for (auto node : nodes_to_insert) {
    size_t sh_size = check_routes_can_insert_and_get_sh_size(1);
    bool is_set    = set_shmem_of_kernel(insert_node_to_best_kernel<i_t, f_t, REQUEST>, sh_size);
    cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
    insert_node_to_best_kernel<i_t, f_t, REQUEST>
      <<<1, TPB, sh_size, sol_handle->get_stream()>>>(view(), node, include_objective, weights);
    sol_handle->sync_stream();
  }
  this->global_runtime_checks(false, false, "add_nodes_to_best");
  sol_handle->sync_stream();
}

// removes nodes from the solution
template <typename i_t, typename f_t, request_t REQUEST>
bool solution_t<i_t, f_t, REQUEST>::remove_nodes(const std::vector<NodeInfo<>>& nodes_to_eject)
{
  raft::common::nvtx::range fun_scope("remove_nodes");
  const auto n_nodes_to_eject = nodes_to_eject.size();
  rmm::device_scalar<i_t> empty_route_produced(sol_handle->get_stream());
  raft::copy(temp_nodes.data(), nodes_to_eject.data(), n_nodes_to_eject, sol_handle->get_stream());
  compute_max_active();
  size_t sh_size = std::max(get_temp_route_shared_size(), n_routes * sizeof(i_t));
  bool is_set    = set_shmem_of_kernel(remove_nodes_kernel<i_t, f_t, REQUEST>, sh_size);
  cuopt_assert(is_set, "Not enough shared memory on device for remove_nodes!");
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  i_t TPB = 256;
  remove_nodes_kernel<i_t, f_t, REQUEST><<<1, TPB, sh_size, sol_handle->get_stream()>>>(
    view(), n_nodes_to_eject, temp_nodes.data(), empty_route_produced.data());
  sol_handle->sync_stream();
  return !empty_route_produced.value(sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::clear_routes(std::vector<i_t> vehicle_ids)
{
  raft::common::nvtx::range fun_scope("clear_routes");
  i_t given_n_routes;
  if (vehicle_ids.empty()) {
    given_n_routes = problem_ptr->get_fleet_size();
    vehicle_ids.resize(given_n_routes);
    std::iota(std::begin(vehicle_ids), std::end(vehicle_ids), 0);
  } else {
    given_n_routes = vehicle_ids.size();
  }

  // Do a permutation based on random number
  auto rng = std::default_random_engine(solution_id);
  std::shuffle(vehicle_ids.begin(), vehicle_ids.end(), rng);

  if (given_n_routes > n_routes) {
    std::iota(route_id_to_idx.begin(), route_id_to_idx.end(), 0);
    set_route_views();
  }
  n_routes = given_n_routes;
  check_and_allocate_routes(n_routes);
  cuopt_assert(routes.size() >= given_n_routes, "routes size error!");
  cuopt_assert(routes_view.size() >= given_n_routes, "routes size error!");

  for (i_t i = 0; i < n_routes; ++i) {
    get_route(i).vehicle_id.set_value_async(vehicle_ids[i], sol_handle->get_stream());
    get_route(i).route_id.set_value_async(i, sol_handle->get_stream());
  }

  set_routes_to_copy();
  thrust::fill(sol_handle->get_thrust_policy(),
               route_node_map.route_id_per_node.begin(),
               route_node_map.route_id_per_node.end(),
               -1);
  thrust::fill(sol_handle->get_thrust_policy(),
               route_node_map.intra_route_idx_per_node.begin(),
               route_node_map.intra_route_idx_per_node.end(),
               -1);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::unset_routes_to_copy()
{
  thrust::fill(sol_handle->get_thrust_policy(), routes_to_copy.begin(), routes_to_copy.end(), 0);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::set_routes_to_copy()
{
  thrust::fill(sol_handle->get_thrust_policy(), routes_to_copy.begin(), routes_to_copy.end(), 1);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::unset_routes_to_search()
{
  thrust::fill(
    sol_handle->get_thrust_policy(), routes_to_search.begin(), routes_to_search.end(), 0);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::set_routes_to_search()
{
  thrust::fill(
    sol_handle->get_thrust_policy(), routes_to_search.begin(), routes_to_search.end(), 1);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::set_route_views()
{
  raft::common::nvtx::range fun_scope("set_route_views");
  if (routes.size() > routes_view.size()) {
    // reserve with the max size
    routes_view.resize(routes.size(), sol_handle->get_stream());
  }

  for (size_t i = 0; i < routes.size(); ++i) {
    const auto route_view = get_route(i).view();
    routes_view.set_element_async(i, route_view, sol_handle->get_stream());
  }
}

// init the rotues with the desired number of routes
template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::random_init_routes()
{
  raft::common::nvtx::range fun_scope("random_init_routes");
  auto stream = sol_handle->get_stream();
  stream.synchronize();
  const i_t one = 1;
  d_sol_found.set_value_async(one, stream);
  std::vector<i_t> indices(get_num_requests());
  rmm::device_uvector<i_t> d_indices(get_num_requests(), stream);
  thrust::sequence(indices.begin(), indices.end());
  std::random_shuffle(indices.begin(), indices.end());
  raft::copy(d_indices.data(), indices.data(), indices.size(), stream);
  // Do the updates to n_nodes in host side and device side
  // Other option of copying might be better
  const int n_initial_nodes = 1 + request_info_t<i_t, REQUEST>::size();
  // assign one request per route, for remaining routes don't assign any requests
  for (i_t route_id = 0; route_id < n_routes; ++route_id) {
    if (route_id < get_num_requests()) {
      routes[route_id].n_nodes.set_value_async(n_initial_nodes, stream);
    } else {
      routes[route_id].n_nodes.set_value_async(one, stream);
    }
  }
  set_initial_nodes(d_indices, n_routes);
  stream.synchronize();
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::compute_initial_data(bool check_feasibility)
{
  raft::common::nvtx::range fun_scope("compute_initial_data");
  compute_backward_forward();
  compute_cost();
  eject_until_feasible();
  if (check_feasibility) run_feasibility_check();
  // initialize sol_found here. it is possible that one request per route will result in
  // infeasible solution
  sol_found = d_sol_found.value(sol_handle->get_stream());
  if (check_feasibility) this->global_runtime_checks(false, true, "compute_initial_data");
}

template <typename i_t, typename f_t, request_t REQUEST>
size_t solution_t<i_t, f_t, REQUEST>::get_temp_route_shared_size(i_t added_size) const
{
  const i_t max_route_size = get_max_active_nodes_for_all_routes();
  return route_t<i_t, f_t, REQUEST>::get_shared_size(max_route_size + added_size,
                                                     problem_ptr->dimensions_info);
}

// call this function before each kernel that changes route sizes
template <typename i_t, typename f_t, request_t REQUEST>
i_t solution_t<i_t, f_t, REQUEST>::get_max_active_nodes_for_all_routes() const
{
  raft::common::nvtx::range fun_scope("get_max_active_nodes_for_all_routes");
  return max_active_nodes;
}

// call this function before each kernel that changes route sizes
template <typename i_t, typename f_t, request_t REQUEST>
size_t solution_t<i_t, f_t, REQUEST>::check_routes_can_insert_and_get_sh_size(i_t added_nodes)
{
  raft::common::nvtx::range fun_scope("check_routes_can_insert_and_get_sh_size");
  compute_max_active();
  resize_routes(raft::alignTo(max_active_nodes + added_nodes, base_route_size));
  return get_temp_route_shared_size(added_nodes);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::resize_routes(i_t new_size)
{
  raft::common::nvtx::range fun_scope("resize_routes");

  for (i_t i = 0; i < n_routes; ++i) {
    auto& route_i = get_route(i);
    if (route_i.max_nodes_per_route() < new_size) {
      route_i.resize(new_size);
      const auto route_view = get_route(i).view();
      routes_view.set_element_async(i, route_view, sol_handle->get_stream());
    }
  }
  max_nodes_per_route = std::max(max_nodes_per_route, new_size);
}

// returns the initial number of routes
template <typename i_t, typename f_t, request_t REQUEST>
i_t solution_t<i_t, f_t, REQUEST>::get_n_routes() const noexcept
{
  return n_routes;
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t solution_t<i_t, f_t, REQUEST>::get_num_empty_vehicles()
{
  return thrust::count_if(
    rmm::exec_policy(sol_handle->get_stream()),
    routes_view.begin(),
    routes_view.begin() + n_routes,
    [] __device__(const auto& route) { return (0 == route.get_num_service_nodes()); });
}

template <typename i_t, typename f_t, request_t REQUEST>
std::vector<i_t> solution_t<i_t, f_t, REQUEST>::get_used_vehicle_ids() const noexcept
{
  std::vector<i_t> vehicle_ids(n_routes);
  for (int iroute = 0; iroute < n_routes; ++iroute) {
    vehicle_ids[iroute] = get_route(iroute).vehicle_id.value(sol_handle->get_stream());
  }

  sol_handle->sync_stream();
  return vehicle_ids;
}

template <typename i_t, typename f_t, request_t REQUEST>
std::vector<i_t> solution_t<i_t, f_t, REQUEST>::get_unused_vehicles() const noexcept
{
  std::vector<i_t> vehicle_ids(problem_ptr->get_fleet_size());
  std::iota(vehicle_ids.begin(), vehicle_ids.end(), 0);
  for (const auto& id : get_used_vehicle_ids()) {
    vehicle_ids.erase(std::remove(vehicle_ids.begin(), vehicle_ids.end(), id), vehicle_ids.end());
  }
  return vehicle_ids;
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t solution_t<i_t, f_t, REQUEST>::get_num_orders() const noexcept
{
  return problem_ptr->order_info.get_num_orders();
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t solution_t<i_t, f_t, REQUEST>::get_num_requests() const noexcept
{
  return problem_ptr->order_info.get_num_requests();
}

template <typename i_t, typename f_t, request_t REQUEST>
double solution_t<i_t, f_t, REQUEST>::get_cost(const bool include_objective,
                                               const infeasible_cost_t weights) const
{
  raft::common::nvtx::range fun_scope("get_cost");

  double total_cost =
    infeasible_cost_t::dot(weights, infeasibility_cost.value(sol_handle->get_stream()));

  if (include_objective) {
    auto obj_weights = problem_ptr->dimensions_info.objective_weights;
    total_cost +=
      objective_cost_t::dot(obj_weights, objective_cost.value(sol_handle->get_stream()));
  }

  return total_cost;
}

template <typename i_t, typename f_t, request_t REQUEST>
double solution_t<i_t, f_t, REQUEST>::get_total_cost(const infeasible_cost_t weights) const
{
  return get_cost(true, weights);
}

template <typename i_t, typename f_t, request_t REQUEST>
objective_cost_t solution_t<i_t, f_t, REQUEST>::get_objective_cost() const
{
  return objective_cost.value(sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
infeasible_cost_t solution_t<i_t, f_t, REQUEST>::get_infeasibility_cost() const
{
  return infeasibility_cost.value(sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
bool solution_t<i_t, f_t, REQUEST>::is_feasible() const
{
  raft::common::nvtx::range fun_scope("is_feasible");
  auto n_cost      = infeasibility_cost.value(sol_handle->get_stream());
  bool is_feasible = true;
  for (size_t dim = 0; dim < (size_t)dim_t::SIZE; ++dim) {
    is_feasible &= n_cost[dim] <= std::numeric_limits<double>::epsilon();
  }
  return is_feasible;
}

// this should be called when we want to execute feasible functions
template <typename i_t, typename f_t, request_t REQUEST>
f_t solution_t<i_t, f_t, REQUEST>::get_total_excess(const infeasible_cost_t weights) const
{
  raft::common::nvtx::range fun_scope("get_total_excess");
  auto n_cost = infeasibility_cost.value(sol_handle->get_stream());
  return infeasible_cost_t::dot(n_cost, weights);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::print() const
{
  for (i_t i = 0; i < get_n_routes(); ++i) {
    get_route(i).print();
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::copy_device_solution(solution_t<i_t, f_t, REQUEST>& src_sol)
{
  raft::common::nvtx::range fun_scope("copy_device_solution");
  sol_handle->sync_stream();
  src_sol.sol_handle->sync_stream();

  if (!src_sol.n_routes) { return; }
  check_and_allocate_routes(src_sol.n_routes);

  for (i_t i = n_routes; i < src_sol.n_routes; ++i) {
    const auto route_view = get_route(i).view();
    routes_view.set_element_async(i, route_view, sol_handle->get_stream());
  }

  n_routes = src_sol.n_routes;

  const auto common_max_size = std::max(src_sol.max_nodes_per_route, max_nodes_per_route);
  resize_routes(common_max_size);
  const i_t TPB       = 256;
  const auto n_blocks = n_routes;
  copy_routes<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, 0, sol_handle->get_stream()>>>(view(), src_sol.view());
  RAFT_CHECK_CUDA(sol_handle->get_stream());

  cuopt_assert(route_node_map.intra_route_idx_per_node.size() == (size_t)get_num_orders(),
               "Intra route size mismatch!");
  cuopt_expects(route_node_map.intra_route_idx_per_node.size() == (size_t)get_num_orders(),
                error_type_t::RuntimeError,
                "A runtime error occurred!");

  cuopt_assert(route_node_map.route_id_per_node.size() == (size_t)get_num_orders(),
               "Route id size mismatch!");
  cuopt_expects(route_node_map.route_id_per_node.size() == (size_t)get_num_orders(),
                error_type_t::RuntimeError,
                "A runtime error occurred!");
  route_node_map.copy_from(src_sol.route_node_map, sol_handle->get_stream());

  raft::copy(
    infeasibility_cost.data(), src_sol.infeasibility_cost.data(), 1, sol_handle->get_stream());
  raft::copy(objective_cost.data(), src_sol.objective_cost.data(), 1, sol_handle->get_stream());
  raft::copy(max_active_nodes_for_all_routes.data(),
             src_sol.max_active_nodes_for_all_routes.data(),
             1,
             sol_handle->get_stream());
  raft::copy(
    n_infeasible_routes.data(), src_sol.n_infeasible_routes.data(), 1, sol_handle->get_stream());
  unset_routes_to_copy();
  sol_handle->sync_stream();
  RAFT_CHECK_CUDA(sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::compute_cost()
{
  raft::common::nvtx::range fun_scope("compute_cost");
  const i_t TPB       = 256;
  const auto n_blocks = (n_routes + TPB - 1) / TPB;

  infeasible_cost_t zero_inf;
  objective_cost_t zero_obj;
  infeasibility_cost.set_value_async(zero_inf, sol_handle->get_stream());
  objective_cost.set_value_async(zero_obj, sol_handle->get_stream());
  n_infeasible_routes.set_value_to_zero_async(sol_handle->get_stream());
  if (get_n_routes() < 1) return;
  compute_cost_kernel<i_t, f_t, REQUEST><<<n_blocks, TPB, 0, sol_handle->get_stream()>>>(view());
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::check_cost_coherence(const infeasible_cost_t& weights)
{
  raft::common::nvtx::range fun_scope("check_cost_coherence");
  double excess_before = get_total_excess(weights);
  double cost_before   = get_cost(true, weights);
  compute_backward_forward();
  compute_cost();
  double excess_after = get_total_excess(weights);
  double cost_after   = get_cost(true, weights);
  cuopt_assert(abs(excess_before - excess_after) < 0.00001, "Excess mismatch!");
  cuopt_assert(abs(cost_before - cost_after) < 0.00001, "Cost mismatch!");
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::shift_move_routes(
  const std::vector<i_t>& route_ids, rmm::device_uvector<i_t>& route_ids_device_copy)

{
  raft::common::nvtx::range fun_scope("shift_move_routes");
  // shift and swap route slots
  i_t remove_counter = 0;
  // this is not super efficient but not significant
  for (auto route_id : route_ids) {
    cuopt_assert(route_id < n_routes, "Route does not exist!");
    i_t released_slot = route_id_to_idx[route_id - remove_counter];
    for (i_t i = route_id - remove_counter; i < n_routes - remove_counter - 1; ++i) {
      route_id_to_idx[i] = route_id_to_idx[i + 1];
    }
    route_id_to_idx[n_routes - remove_counter - 1] = released_slot;
    ++remove_counter;
  }

  constexpr auto threads_per_block = 64;
  auto n_blocks                    = n_routes - route_ids[0];
  // if there is a block to launch
  if (n_blocks > 0) {
    // Decrement route_id_per_node for this route
    remap_route_nodes<i_t, f_t, REQUEST>
      <<<n_blocks, threads_per_block, 0, sol_handle->get_stream()>>>(
        routes_view.data(), route_node_map.view(), route_ids_device_copy.data(), route_ids.size());
    RAFT_CHECK_CUDA(sol_handle->get_stream());
    shift_routes_kernel<i_t, f_t, REQUEST><<<1, 1, 0, sol_handle->get_stream()>>>(
      view(), route_ids_device_copy.data(), route_ids.size());
    RAFT_CHECK_CUDA(sol_handle->get_stream());
  }
  sol_handle->sync_stream();
  n_routes -= route_ids.size();
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::remove_empty_routes()
{
  std::vector<i_t> routes_to_remove;
  routes_to_remove.reserve(get_n_routes());
  for (i_t id = 0; id < get_n_routes(); ++id) {
    if (get_route(id).is_empty()) { routes_to_remove.push_back(id); }
  }
  if (!routes_to_remove.empty()) { remove_routes(routes_to_remove); }
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::remove_routes(
  ejection_pool_t<request_info_t<i_t, REQUEST>>& ejection_pool,
  const std::vector<i_t>& routes_to_remove)
{
  raft::common::nvtx::range fun_scope("remove_routes");

  if (routes_to_remove.empty()) { return; }

  i_t ep_size = 0;
  for (auto route_id : routes_to_remove) {
    cuopt_assert(route_id >= 0 && route_id < n_routes, "route to remove should be in range");
    ep_size += (get_route(route_id).get_num_service_nodes() / request_info_t<i_t, REQUEST>::size());
  }

  // Update index (- 1 for depot, / 2 for requests and -1 because it's index not size)
  ejection_pool.index_ = ep_size - 1;

  // reuse temp_int_vector
  temp_int_vector.resize(routes_to_remove.size(), sol_handle->get_stream());
  raft::copy(temp_int_vector.data(),
             routes_to_remove.data(),
             routes_to_remove.size(),
             sol_handle->get_stream());

  if (ep_size) {
    temp_stack_counter.set_value_to_zero_async(sol_handle->get_stream());

    cuopt_assert(ejection_pool.index_ >= 0, "Index should be at least 0");
    set_deleted_routes_kernel<i_t, f_t, REQUEST>
      <<<routes_to_remove.size(), 1, 0, sol_handle->get_stream()>>>(
        view(),
        cuopt::make_span(routes_view),
        cuopt::make_span(temp_int_vector),
        cuopt::make_span(ejection_pool.stack_),
        temp_stack_counter.data());
  }
  shift_move_routes(routes_to_remove, temp_int_vector);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::remove_routes(const std::vector<i_t>& routes_to_remove)
{
  raft::common::nvtx::range fun_scope("remove_routes");
  cuopt_assert(routes_to_remove.size() > 0, "There should be at least one route in the vector!");
  // reuse temp_int_vector
  temp_int_vector.resize(routes_to_remove.size(), sol_handle->get_stream());
  raft::copy(temp_int_vector.data(),
             routes_to_remove.data(),
             routes_to_remove.size(),
             sol_handle->get_stream());

  for (size_t i = 0; i < routes_to_remove.size(); ++i) {
    cuopt_assert(routes_to_remove[i] >= 0 && routes_to_remove[i] < n_routes,
                 "route to remove should be in range");
  }
  set_deleted_routes_kernel<i_t, f_t, REQUEST>
    <<<routes_to_remove.size(), 1, 0, sol_handle->get_stream()>>>(
      view(), cuopt::make_span(routes_view), cuopt::make_span(temp_int_vector));
  shift_move_routes(routes_to_remove, temp_int_vector);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::keep_only_vehicles(const std::vector<i_t>& vehicles_to_keep)
{
  raft::common::nvtx::range fun_scope("keep_only_vehicles");

  // Figure out which routes to remove
  std::vector<i_t> routes_to_remove;
  std::set<i_t> vehicles(vehicles_to_keep.begin(), vehicles_to_keep.end());
  for (i_t iroute = 0; iroute < n_routes; ++iroute) {
    i_t vehicle_id = get_route(iroute).vehicle_id.value(sol_handle->get_stream());
    if (!vehicles.count(vehicle_id)) { routes_to_remove.push_back(iroute); }
  }

  if (!routes_to_remove.empty()) { remove_routes(routes_to_remove); }
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t solution_t<i_t, f_t, REQUEST>::compute_max_active()
{
  raft::common::nvtx::range fun_scope("compute_max_active");
  i_t TPB = 1024;
  compute_max_active_kernel<i_t, f_t, REQUEST><<<1, TPB, 0, sol_handle->get_stream()>>>(view());
  max_active_nodes = max_active_nodes_for_all_routes.value(sol_handle->get_stream());
  return max_active_nodes;
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::compute_route_id_per_node()
{
  raft::common::nvtx::range fun_scope("compute_route_id_per_node");
  i_t TPB = 256;
  compute_route_id_kernel<i_t, f_t, REQUEST>
    <<<n_routes, TPB, 0, sol_handle->get_stream()>>>(routes_view.data(), route_node_map.view());
  global_runtime_checks(false, false, "compute_route_id_per_node");
}

template <typename i_t, typename f_t, request_t REQUEST>
std::vector<i_t> solution_t<i_t, f_t, REQUEST>::get_unserviced_nodes() const
{
  std::vector<i_t> unserviced_nodes;
  unserviced_nodes.reserve(get_num_orders());
  const bool depot_included = problem_ptr->order_info.depot_included_;
  auto h_route_id_per_node  = host_copy(route_node_map.route_id_per_node);
  for (size_t i = 0; i < h_route_id_per_node.size(); ++i) {
    if (h_route_id_per_node[i] == -1) {
      if (i > 0 || !depot_included) { unserviced_nodes.push_back(i); }
    }
  }

  sol_handle->sync_stream();
  return unserviced_nodes;
}

template class solution_t<int, float, request_t::PDP>;
template class solution_t<int, float, request_t::VRP>;

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
