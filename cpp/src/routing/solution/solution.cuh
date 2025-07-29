/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../ges/ejection_pool.cuh"
#include "../node/node.cuh"
#include "../problem/problem.cuh"
#include "../route/route.cuh"
#include "../routing_helpers.cuh"
#include "route_node_map.cuh"
#include "solution_handle.cuh"

#include <cuopt/routing/data_model_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <map>
#include <random>
#include <tuple>

namespace cuopt {
namespace routing {
namespace detail {

constexpr int base_route_size = 64;

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI request_info_t<i_t, REQUEST> create_request(const typename problem_t<i_t, f_t>::view_t& problem,
                                               request_id_t<REQUEST> const& request_id)
{
  const auto& order_info = problem.order_info;
  auto node_info         = NodeInfo<i_t>(
    request_id.id(), order_info.get_order_location(request_id.id()), node_type_t::DELIVERY);
  return request_info_t<i_t, REQUEST>(node_info);
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI request_info_t<i_t, REQUEST> create_request(const typename problem_t<i_t, f_t>::view_t& problem,
                                               request_id_t<REQUEST> const& request_id)
{
  const auto& order_info = problem.order_info;
  auto node_info         = NodeInfo<i_t>(
    request_id.id(), order_info.get_order_location(request_id.id()), node_type_t::PICKUP);
  auto brother_info = NodeInfo<i_t>(
    request_id.delivery, order_info.get_order_location(request_id.delivery), node_type_t::DELIVERY);
  return request_info_t<i_t, REQUEST>(node_info, brother_info);
}

template <typename i_t, typename f_t, request_t REQUEST>
DI node_t<i_t, f_t, REQUEST> create_node(const typename problem_t<i_t, f_t>::view_t& problem,
                                         const NodeInfo<i_t> node_info,
                                         const NodeInfo<i_t> brother_info)
{
  i_t node_idx = node_info.node();
  node_t<i_t, f_t, REQUEST> node(problem.dimensions_info);

  cuopt_assert(!node_info.is_depot(), "create_node should not be called with depot nodes");

  double earliest = problem.order_info.earliest_time[node_idx];

  double latest = problem.order_info.latest_time[node_idx];

  node.time_dim.window_start       = earliest;
  node.time_dim.window_end         = latest;
  node.time_dim.departure_forward  = node.time_dim.window_start;
  node.time_dim.departure_backward = node.time_dim.window_end;

  constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto i) {
    if (i < node.capacity_dim.n_capacity_dimensions) {
      node.capacity_dim.demand[i] =
        problem.order_info.demand[node_idx + i * problem.order_info.get_num_orders()];
    }
  });

  node.prize_dim.prize = problem.order_info.prizes[node_idx];

  node.request = request_info_t<i_t, REQUEST>(node_info, brother_info);
  return node;
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI node_t<i_t, f_t, REQUEST> create_node(const typename problem_t<i_t, f_t>::view_t& problem,
                                         i_t node_id)
{
  bool is_pickup       = problem.order_info.is_pickup_index[node_id];
  auto this_node_type  = is_pickup ? node_type_t::PICKUP : node_type_t::DELIVERY;
  auto other_node_type = is_pickup ? node_type_t::DELIVERY : node_type_t::PICKUP;
  i_t other_node_id    = problem.order_info.pair_indices[node_id];

  const auto this_node_info =
    NodeInfo<i_t>(node_id, problem.order_info.get_order_location(node_id), this_node_type);

  const auto other_node_info = NodeInfo<i_t>(
    other_node_id, problem.order_info.get_order_location(other_node_id), other_node_type);

  return create_node<i_t, f_t, REQUEST>(problem, this_node_info, other_node_info);
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI node_t<i_t, f_t, REQUEST> create_node(const typename problem_t<i_t, f_t>::view_t& problem,
                                         i_t node_id)
{
  auto this_node_type = node_type_t::DELIVERY;
  const auto this_node_info =
    NodeInfo<i_t>(node_id, problem.order_info.get_order_location(node_id), this_node_type);

  return create_node<i_t, f_t, REQUEST>(problem, this_node_info, this_node_info);
}

// FIXME: template on problem
template <typename i_t, typename f_t, request_t REQUEST>
constexpr node_t<i_t, f_t, REQUEST> create_node(const problem_t<i_t, f_t>* problem,
                                                const NodeInfo<i_t> node_info,
                                                const NodeInfo<i_t> brother_info)
{
  i_t node_idx = node_info.node();
  node_t<i_t, f_t, REQUEST> node(problem->dimensions_info);

  cuopt_assert(!node_info.is_depot(), "create_node should not be called with depot nodes");

  double earliest = problem->order_info_h.earliest_time[node_idx];

  double latest = problem->order_info_h.latest_time[node_idx];

  node.time_dim.window_start       = earliest;
  node.time_dim.window_end         = latest;
  node.time_dim.departure_forward  = node.time_dim.window_start;
  node.time_dim.departure_backward = node.time_dim.window_end;

  constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto i) {
    if (i < node.capacity_dim.n_capacity_dimensions) {
      node.capacity_dim.demand[i] =
        problem->order_info_h.demand[node_idx + i * problem->order_info.get_num_orders()];
    }
  });

  node.prize_dim.prize = problem->order_info_h.prizes[node_idx];

  node.request = request_info_t<i_t, REQUEST>(node_info, brother_info);
  return node;
}

template <typename i_t, typename f_t, request_t REQUEST>
constexpr node_t<i_t, f_t, REQUEST> create_node(const problem_t<i_t, f_t>* problem, i_t node_id)
{
  auto this_node_type = node_type_t::DELIVERY;
  const auto this_node_info =
    NodeInfo<i_t>(node_id, problem->order_info_h.get_order_location(node_id), this_node_type);

  return create_node<i_t, f_t, REQUEST>(problem, this_node_info, this_node_info);
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI request_node_t<i_t, f_t, REQUEST> create_request_node(
  const typename problem_t<i_t, f_t>::view_t& problem, request_id_t<REQUEST> request_id)
{
  i_t node_id          = request_id.pickup;
  i_t other_node_id    = request_id.delivery;
  auto this_node_type  = node_type_t::PICKUP;
  auto other_node_type = node_type_t::DELIVERY;

  const auto this_node_info =
    NodeInfo<i_t>(node_id, problem.order_info.get_order_location(node_id), this_node_type);

  const auto other_node_info = NodeInfo<i_t>(
    other_node_id, problem.order_info.get_order_location(other_node_id), other_node_type);

  return request_node_t<i_t, f_t, REQUEST>(
    create_node<i_t, f_t, REQUEST>(problem, this_node_info, other_node_info),
    create_node<i_t, f_t, REQUEST>(problem, other_node_info, this_node_info));
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI request_node_t<i_t, f_t, REQUEST> create_request_node(
  const typename problem_t<i_t, f_t>::view_t& problem, request_id_t<REQUEST> request_id)
{
  auto this_node_type = node_type_t::DELIVERY;
  i_t node_id         = request_id.id();
  const auto this_node_info =
    NodeInfo<i_t>(node_id, problem.order_info.get_order_location(node_id), this_node_type);

  return request_node_t<i_t, f_t, REQUEST>(
    create_node<i_t, f_t, REQUEST>(problem, this_node_info, this_node_info));
}

template <typename i_t, typename f_t, request_t REQUEST>
DI node_t<i_t, f_t, REQUEST> create_depot_node(const typename problem_t<i_t, f_t>::view_t& problem,
                                               const NodeInfo<i_t> node_info,
                                               const NodeInfo<i_t> brother_info,
                                               const i_t vehicle_id)
{
  node_t<i_t, f_t, REQUEST> node(problem.dimensions_info);

  cuopt_assert(node_info.is_depot(), "create_depot_node should be only called with depot nodes");

  double earliest =
    !problem.order_info.depot_included
      ? problem.fleet_info.earliest_time[vehicle_id]
      : max(problem.order_info.earliest_time[DEPOT], problem.fleet_info.earliest_time[vehicle_id]);

  double latest =
    !problem.order_info.depot_included
      ? problem.fleet_info.latest_time[vehicle_id]
      : min(problem.order_info.latest_time[DEPOT], problem.fleet_info.latest_time[vehicle_id]);

  node.time_dim.window_start              = earliest;
  node.time_dim.window_end                = latest;
  node.time_dim.departure_forward         = node.time_dim.window_start;
  node.time_dim.departure_backward        = node.time_dim.window_end;
  node.time_dim.latest_arrival_forward    = latest;
  node.time_dim.earliest_arrival_backward = earliest;

  constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto i) {
    if (i < node.capacity_dim.n_capacity_dimensions) { node.capacity_dim.demand[i] = 0; }
  });

  node.prize_dim.prize = 0.;
  node.request         = request_info_t<i_t, REQUEST>(node_info, brother_info);
  return node;
}

template <typename i_t, typename f_t, request_t REQUEST>
constexpr node_t<i_t, f_t, REQUEST> create_depot_node(const problem_t<i_t, f_t>* problem,
                                                      const NodeInfo<i_t> node_info,
                                                      const NodeInfo<i_t> brother_info,
                                                      const i_t vehicle_id)
{
  node_t<i_t, f_t, REQUEST> node(problem->dimensions_info);

  cuopt_assert(node_info.is_depot(), "create_depot_node should be only called with depot nodes");

  double earliest = !problem->order_info_h.depot_included
                      ? problem->fleet_info_h.earliest_time[vehicle_id]
                      : max(problem->order_info_h.earliest_time[DEPOT],
                            problem->fleet_info_h.earliest_time[vehicle_id]);

  double latest = !problem->order_info_h.depot_included
                    ? problem->fleet_info_h.latest_time[vehicle_id]
                    : min(problem->order_info_h.latest_time[DEPOT],
                          problem->fleet_info_h.latest_time[vehicle_id]);

  node.time_dim.window_start              = earliest;
  node.time_dim.window_end                = latest;
  node.time_dim.departure_forward         = node.time_dim.window_start;
  node.time_dim.departure_backward        = node.time_dim.window_end;
  node.time_dim.latest_arrival_forward    = latest;
  node.time_dim.earliest_arrival_backward = earliest;

  constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto i) {
    if (i < node.capacity_dim.n_capacity_dimensions) { node.capacity_dim.demand[i] = 0; }
  });

  node.prize_dim.prize = 0.;
  node.request         = request_info_t<i_t, REQUEST>(node_info, brother_info);
  return node;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI node_t<i_t, f_t, REQUEST> create_break_node(
  const typename special_nodes_t<i_t>::view_t& special_nodes,
  const i_t index,
  const enabled_dimensions_t& dimensions_info)
{
  auto node_info = special_nodes.node_infos[index];

  node_t<i_t, f_t, REQUEST> node(dimensions_info);

  cuopt_assert(node_info.is_break(), "special nodes should only contain breaks");

  node.time_dim.window_start = special_nodes.earliest_time[index];
  node.time_dim.window_end   = special_nodes.latest_time[index];

  // FIXME:: setting the prize to zero for now.
  // When we support breaks through prize collection mechanism, this will change
  node.prize_dim.prize = 0.;

  node.request = request_info_t<i_t, REQUEST>(node_info, node_info);
  return node;
}

template <typename i_t, typename f_t, request_t REQUEST>
class solution_t {
 public:
  static const request_t request_type = REQUEST;

  solution_t(const problem_t<i_t, f_t>& problem_,
             i_t solution_id_,
             solution_handle_t<i_t, f_t> const* sol_handle_,
             // i_t desired_n_routes = -1,
             std::vector<i_t> desired_vehicle_ids = {},
             i_t route_max_size                   = -1)
    : problem_ptr(&problem_),
      sol_handle(sol_handle_),
      solution_id(solution_id_),
      n_routes(problem_.get_fleet_size()),
      routes_view(0, sol_handle_->get_stream()),
      route_node_map(problem_.get_num_orders(), sol_handle_->get_stream()),
      routes_to_copy(problem_.get_fleet_size(), sol_handle->get_stream()),
      routes_to_search(problem_.get_fleet_size(), sol_handle->get_stream()),
      runtime_check_histo(problem_.get_num_orders(), sol_handle->get_stream()),
      // TODO populate fleet info or directly get it from the main solver class
      // even though fleet info is created with the main sol_handle_->get_stream() this will be only
      // constructed once and not copied back and forth, so it is not an issue to pass handle it
      // might be a better idea to create it with a sol_handle_->get_stream() though
      route_id_to_idx(0),
      objective_cost(sol_handle_->get_stream()),
      infeasibility_cost(sol_handle_->get_stream()),
      d_sol_found(sol_handle_->get_stream()),
      d_lock(sol_handle_->get_stream()),
      d_lock_per_route(problem_.get_fleet_size(), sol_handle->get_stream()),
      d_lock_per_order(problem_.get_num_orders(), sol_handle->get_stream()),
      n_infeasible_routes(sol_handle_->get_stream()),
      max_active_nodes_for_all_routes(sol_handle_->get_stream()),
      temp_nodes(problem_.get_num_orders(), sol_handle_->get_stream()),
      temp_stack_counter(sol_handle_->get_stream()),
      temp_int_vector(std::max(problem_.get_num_orders(), problem_.get_fleet_size()),
                      sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("solution_t");

    thrust::fill(
      sol_handle_->get_thrust_policy(), d_lock_per_route.begin(), d_lock_per_route.end(), 0);
    thrust::fill(
      sol_handle_->get_thrust_policy(), d_lock_per_order.begin(), d_lock_per_order.end(), 0);

    set_routes_to_copy();
    set_routes_to_search();
    n_routes = desired_vehicle_ids.size();
    route_id_to_idx.resize(n_routes);
    std::iota(std::begin(route_id_to_idx), std::end(route_id_to_idx), 0);
    if (route_max_size == -1) { route_max_size = base_route_size; }
    cuopt_assert((route_max_size >= request_info_t<i_t, REQUEST>::size()),
                 "Initial max route size should be at least 2");
    cuopt_assert(n_routes <= problem_.get_fleet_size(),
                 "Number of routes should be less than the fleet size");

    i_t fleet_size = problem_.get_fleet_size();
    std::vector<i_t> unused_vehicle_ids;

    if (n_routes > 0) {
      unused_vehicle_ids.reserve(fleet_size - desired_vehicle_ids.size());
      std::set<i_t> used_vehicles(desired_vehicle_ids.begin(), desired_vehicle_ids.end());
      for (i_t i = 0; i < fleet_size; ++i) {
        if (!used_vehicles.count(i)) { unused_vehicle_ids.push_back(i); }
      }
      n_routes = desired_vehicle_ids.size();

      std::vector<i_t> vehicle_ids = desired_vehicle_ids;
      // Do a permutation based on random number
      auto rng = std::default_random_engine(solution_id_);
      std::shuffle(vehicle_ids.begin(), vehicle_ids.end(), rng);
      // make sure all the routes have same route sizes
      routes.reserve(n_routes);
      for (i_t i = 0; i < n_routes; ++i) {
        i_t vehicle_id =
          i < (i_t)vehicle_ids.size() ? vehicle_ids[i] : unused_vehicle_ids[i - vehicle_ids.size()];
        routes.emplace_back(route_t<i_t, f_t, REQUEST>{
          sol_handle_, i, vehicle_id, &(problem_ptr->fleet_info), problem_ptr->dimensions_info});
      }

      set_route_views();
      resize_routes(raft::alignTo(route_max_size, base_route_size));
      random_init_routes();
      compute_initial_data();
    }
  }

  // Solution only containing information modified during ges loop (route)
  // To be able to restore the state if we need to leave the loop early
  // TODO get rid of copy constructor

  solution_t(const solution_t& sol)
    : sol_found(sol.sol_found),
      solution_id(sol.solution_id),
      sol_handle(sol.sol_handle),
      problem_ptr(sol.problem_ptr),
      routes_view(0, sol.sol_handle->get_stream()),  // Empty construction
      route_node_map(sol.route_node_map, sol.sol_handle->get_stream()),
      routes_to_copy(sol.routes_to_copy, sol.sol_handle->get_stream()),
      routes_to_search(sol.routes_to_copy, sol.sol_handle->get_stream()),
      runtime_check_histo(sol.runtime_check_histo, sol.sol_handle->get_stream()),
      objective_cost(sol.objective_cost, sol.sol_handle->get_stream()),
      infeasibility_cost(sol.infeasibility_cost, sol.sol_handle->get_stream()),
      d_sol_found(sol.d_sol_found, sol.sol_handle->get_stream()),
      d_lock(sol.d_lock, sol.sol_handle->get_stream()),
      d_lock_per_route(sol.d_lock_per_route, sol.sol_handle->get_stream()),
      d_lock_per_order(sol.d_lock_per_order, sol.sol_handle->get_stream()),
      n_routes(sol.n_routes),
      n_infeasible_routes(sol.n_infeasible_routes, sol.sol_handle->get_stream()),
      max_nodes_per_route(sol.max_nodes_per_route),
      max_active_nodes_for_all_routes(sol.max_active_nodes_for_all_routes,
                                      sol.sol_handle->get_stream()),
      temp_nodes(sol.temp_nodes, sol.sol_handle->get_stream()),
      temp_stack_counter(sol.temp_stack_counter, sol.sol_handle->get_stream()),
      temp_int_vector(sol.temp_int_vector, sol.sol_handle->get_stream())
  {
    raft::common::nvtx::range fun_scope("copy ctr solution_t");
    cuopt_assert(n_routes <= problem_ptr->get_fleet_size(),
                 "Number of routes should be less than the fleet size");
    set_routes_to_copy();
    set_routes_to_search();
    // reserve with the max size
    routes.reserve(n_routes);
    route_id_to_idx.reserve(n_routes);

    cuopt_assert(routes.size() == route_id_to_idx.size(), "route and ids should have same size!");

    {
      raft::common::nvtx::range fun_scope("copy_routes");
      for (i_t i = 0; i < sol.n_routes; ++i) {
        routes.emplace_back(sol.get_route(i));
        route_id_to_idx.emplace_back(i);
      }
    }

    set_route_views();
    copy_device_solution(const_cast<solution_t&>(sol));
  }

  // forward decleration, definition is in different files
  void print() const;
  void copy_device_solution(solution_t<i_t, f_t, REQUEST>& src_sol);
  i_t get_max_route_size() const { return max_nodes_per_route; }
  void set_initial_nodes(const rmm::device_uvector<i_t>& d_indices, i_t desired_n_routes);
  void set_nodes_data_of_solution();
  void set_nodes_data_of_route(i_t route_id);
  void set_nodes_data_of_new_routes(i_t added_routes, i_t prev_route_size);
  void compute_cost();
  void populate_ep_with_unserved(ejection_pool_t<request_info_t<i_t, REQUEST>>& EP);
  void populate_ep_with_selected_unserved(ejection_pool_t<request_info_t<i_t, REQUEST>>& EP,
                                          const std::vector<i_t>& unserviced);
  void eject_until_feasible(bool add_slack_to_sol = false);
  void global_runtime_checks(bool all_nodes_should_be_served,
                             bool check_feasible,
                             const std::string_view where);
  void run_feasibility_check();
  void run_coherence_check();
  void compute_backward_forward();
  void compute_actual_arrival_times();
  void shift_move_routes(const std::vector<i_t>& route_ids,
                         rmm::device_uvector<i_t>& route_ids_device_copy);
  void remove_empty_routes();
  void remove_route(ejection_pool_t<request_info_t<i_t, REQUEST>>& ejection_pool, i_t route_id);
  void remove_routes(const std::vector<i_t>& routes_to_remove);
  void remove_routes(ejection_pool_t<request_info_t<i_t, REQUEST>>& ejection_pool,
                     const std::vector<i_t>& routes_to_remove);
  void keep_only_vehicles(const std::vector<i_t>& vehicle_ids_to_keep);
  void compute_route_id_per_node();
  i_t compute_max_active();
  f_t get_total_excess(const infeasible_cost_t weights) const;
  bool is_feasible() const;
  double get_total_cost(const infeasible_cost_t weights) const;
  double get_cost(const bool include_objective, const infeasible_cost_t weights) const;
  objective_cost_t get_objective_cost() const;
  infeasible_cost_t get_infeasibility_cost() const;
  void check_cost_coherence(const infeasible_cost_t& weights);
  i_t get_n_routes() const noexcept;
  i_t get_num_empty_vehicles();
  std::vector<i_t> get_used_vehicle_ids() const noexcept;
  std::vector<i_t> get_unused_vehicles() const noexcept;
  void check_route_assignment(std::string msg) const;
  size_t check_routes_can_insert_and_get_sh_size(
    i_t added_nodes = request_info_t<i_t, REQUEST>::size());
  size_t get_temp_route_shared_size(i_t added_size = 0) const;
  void compute_initial_data(bool check_feasibility = true);
  void random_init_routes();
  void set_route_views();
  void expand_route(i_t route_id);
  void resize_route(i_t route_id, i_t new_route_size);
  void clear_routes(std::vector<i_t> vehicle_ids);
  bool remove_nodes(const std::vector<NodeInfo<>>& nodes_to_eject);
  void add_nodes_to_route(const std::vector<NodeInfo<>>& nodes_to_insert,
                          i_t route_id,
                          i_t intra_idx);
  void add_nodes_to_best(const std::vector<NodeInfo<>>& nodes_to_insert,
                         const infeasible_cost_t& weights);
  void add_routes(const std::vector<std::pair<int, std::vector<NodeInfo<>>>>& new_routes);
  void add_route(route_t<i_t, f_t, REQUEST>&& route,
                 i_t route_id,
                 i_t n_nodes,
                 bool check_size = true);
  const route_t<i_t, f_t, REQUEST>& get_route(i_t route_id) const;
  route_t<i_t, f_t, REQUEST>& get_route(i_t route_id);
  void check_and_allocate_routes(i_t total_num_routes);
  void resize_routes(i_t new_size);
  void unset_routes_to_copy();
  void set_routes_to_copy();
  // for now, the default for routes to search will be all, as it is used in GES and other places
  // too routes to search will only explicity used for routes that should not be searched this will
  // be managed within the LS and adapted_solution interface functions
  void unset_routes_to_search();
  void set_routes_to_search();
  i_t get_max_active_nodes_for_all_routes() const;
  i_t get_num_orders() const noexcept;

  i_t get_num_requests() const noexcept;

  i_t get_num_depot_excluded_orders() const noexcept
  {
    return problem_ptr->order_info.get_num_depot_excluded_orders();
  }

  // this will be used in device code
  struct view_t {
    DI i_t get_max_active_nodes_for_all_routes() const { return *max_active_nodes_for_all_routes; }

    DI i_t get_num_orders() const { return problem.order_info.norders; }

    DI i_t get_num_depot_excluded_orders() const
    {
      return problem.order_info.get_num_depot_excluded_orders();
    }

    DI i_t get_num_requests() const { return problem.order_info.get_num_requests(); }

    DI i_t is_depot_included() const { return (i_t)problem.order_info.depot_included; }

    DI node_t<i_t, f_t, REQUEST> get_node(const i_t node_id) const
    {
      const i_t route_id  = route_node_map.route_id_per_node[node_id];
      const i_t intra_idx = route_node_map.intra_route_idx_per_node[node_id];
      const auto node     = routes[route_id].get_node(intra_idx);
      return node;
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
    DI request_id_t<REQUEST> get_request(i_t idx) const
    {
      const auto& order_info = problem.order_info;
      return request_id_t<r_t>(idx + (i_t)order_info.depot_included);
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
    DI request_id_t<r_t> get_request(i_t idx) const
    {
      return request_id_t<r_t>(problem.pickup_indices[idx], problem.delivery_indices[idx]);
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
    DI auto get_request(const request_info_t<i_t, r_t>* request_id) const
    {
      const auto& order_info = problem.order_info;
      // TODO: optimize in a way
      const request_info_t<i_t, r_t> request = *request_id;
      i_t delivery_id                        = request.info.node();
      cuopt_assert(delivery_id >= 0, "Request ids must be positive");
      cuopt_assert(delivery_id < get_num_orders(),
                   "Request ids must be lower than the amount of nodes");
      NodeInfo<i_t> node_info = NodeInfo<i_t>(
        delivery_id, order_info.get_order_location(delivery_id), node_type_t::DELIVERY);
      const auto node = create_node<i_t, f_t, REQUEST>(problem, node_info, node_info);
      return request_node_t<i_t, f_t, REQUEST>(node);
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
    DI auto get_request(const request_info_t<i_t, r_t>* request_id) const
    {
      const auto& order_info = problem.order_info;
      // TODO: optimize in a way
      const request_info_t<i_t, r_t> request = *request_id;
      i_t pickup_id                          = request.info.node();
      i_t delivery_id                        = request.brother_info.node();
      const auto pickup_info =
        NodeInfo<i_t>(pickup_id, order_info.get_order_location(pickup_id), node_type_t::PICKUP);
      const auto delivery_info = NodeInfo<i_t>(
        delivery_id, order_info.get_order_location(delivery_id), node_type_t::DELIVERY);
      cuopt_assert(pickup_id >= 0 && delivery_id >= 0, "Request ids must be positive");
      cuopt_assert(pickup_id < get_num_orders() && delivery_id < get_num_orders(),
                   "Request ids must be lower than the amount of nodes");
      const auto pickup_node = create_node<i_t, f_t, REQUEST>(problem, pickup_info, delivery_info);
      const auto delivery_node =
        create_node<i_t, f_t, REQUEST>(problem, delivery_info, pickup_info);
      return request_node_t<i_t, f_t, REQUEST>(pickup_node, delivery_node);
    }

    // device spans
    raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t> routes;
    typename route_node_map_t<i_t>::view_t route_node_map;
    raft::device_span<i_t> routes_to_copy;
    raft::device_span<i_t> routes_to_search;

    typename problem_t<i_t, f_t>::view_t problem;
    objective_cost_t* objective_cost;
    infeasible_cost_t* infeasibility_cost;
    i_t* sol_found;
    i_t* lock;
    raft::device_span<i_t> lock_per_route;
    raft::device_span<i_t> lock_per_order;
    i_t* max_active_nodes_for_all_routes;
    i_t n_routes;
    i_t solution_id;
    i_t* n_infeasible_routes;
  };

  view_t view()
  {
    view_t v;
    v.routes  = raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t>{routes_view.data(),
                                                                               routes_view.size()};
    v.problem = problem_ptr->view();
    v.solution_id                     = solution_id;
    v.n_routes                        = get_n_routes();
    v.n_infeasible_routes             = n_infeasible_routes.data();
    v.objective_cost                  = objective_cost.data();
    v.infeasibility_cost              = infeasibility_cost.data();
    v.sol_found                       = d_sol_found.data();
    v.lock                            = d_lock.data();
    v.lock_per_route                  = cuopt::make_span(d_lock_per_route);
    v.lock_per_order                  = cuopt::make_span(d_lock_per_order);
    v.max_active_nodes_for_all_routes = max_active_nodes_for_all_routes.data();
    v.route_node_map                  = route_node_map.view();
    v.routes_to_copy   = raft::device_span<i_t>{routes_to_copy.data(), routes_to_copy.size()};
    v.routes_to_search = raft::device_span<i_t>{routes_to_search.data(), routes_to_search.size()};
    return v;
  }

  std::vector<i_t> get_unserviced_nodes() const;

 private:
  // we shouldn't access this directly as route ids map differently
  std::vector<route_t<i_t, f_t, REQUEST>> routes;

 public:
  static constexpr i_t fragment_step = 1;

  // solution resource handle
  solution_handle_t<i_t, f_t> const* sol_handle;
  // whether a valid feasible solution is found
  bool sol_found = false;
  // solution id
  const i_t solution_id;
  // problem definition
  const problem_t<i_t, f_t>* problem_ptr;
  // route_id to route idx map
  std::vector<i_t> route_id_to_idx;
  // route view for device
  rmm::device_uvector<typename route_t<i_t, f_t, REQUEST>::view_t> routes_view;

  route_node_map_t<i_t> route_node_map;

  rmm::device_scalar<objective_cost_t> objective_cost;
  rmm::device_scalar<infeasible_cost_t> infeasibility_cost;

  rmm::device_uvector<i_t> d_lock_per_route;
  rmm::device_uvector<i_t> d_lock_per_order;

  // whether a solution has been found
  rmm::device_scalar<i_t> d_sol_found;
  // a lock to update global best values
  rmm::device_scalar<i_t> d_lock;
  // max active node size
  rmm::device_scalar<i_t> max_active_nodes_for_all_routes;
  // temporary nodes
  rmm::device_uvector<NodeInfo<>> temp_nodes;

  // temp counter
  rmm::device_scalar<i_t> temp_stack_counter;
  rmm::device_uvector<i_t> temp_int_vector;
  // routes that will be copied to the host(that are modified)
  rmm::device_uvector<i_t> routes_to_copy;
  // routes that will be searched
  rmm::device_uvector<i_t> routes_to_search;
  // histogram for global runtime checks
  rmm::device_uvector<i_t> runtime_check_histo;
  // Inital number of routes
  i_t max_nodes_per_route = base_route_size;
  // Inital number of routes
  i_t n_routes;
  // max active nodes across all routes
  i_t max_active_nodes;
  // Number of infeasible routes
  rmm::device_scalar<i_t> n_infeasible_routes;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
