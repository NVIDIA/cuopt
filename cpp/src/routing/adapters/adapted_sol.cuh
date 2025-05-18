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

#include "../diversity/helpers.hpp"
#include "../diversity/macros.hpp"
#include "../local_search/compute_compatible.cuh"
#include "../problem/problem.cuh"
#include "../routing_helpers.cuh"
#include "../solution/pool_allocator.cuh"
#include "../solution/solution.cuh"
#include "adapted_nodes.cuh"

#include "adapted_nodes.cuh"

#include <raft/util/cudart_utils.hpp>

namespace cuopt::routing::detail {

static inline infeasible_cost_t get_cuopt_cost(costs cpu_cost)
{
  double cost_array[(size_t)dim_t::SIZE];
  for (size_t i = 0; i < (size_t)dim_t::SIZE; ++i) {
    cost_array[i] = cpu_cost[i];
  }
  return infeasible_cost_t(cost_array);
}

static inline costs get_cpu_cost(infeasible_cost_t ges_cost)
{
  costs cpu_cost;
  for (size_t i = 0; i < (size_t)dim_t::SIZE; ++i) {
    cpu_cost[i] = ges_cost[i];
  }
  return cpu_cost;
}

template <typename i_t, typename f_t>
struct adapted_route_t {
  NodeInfo<> start;
  NodeInfo<> end;
  //! Id in the routes vector in solution
  size_t id;
  int length{0};
  int vehicle_id;
  int bucket_id;

  bool is_empty() const { return start.is_depot(); }

  void initialize(
    NodeInfo<> start_, NodeInfo<> end_, size_t id_, int length_, int vehicle_id_, int bucket_id_)
  {
    start      = start_;
    end        = end_;
    id         = id_;
    length     = length_;
    vehicle_id = vehicle_id_;
    bucket_id  = bucket_id_;
  }
};

template <typename i_t, typename f_t>
bool operator==(const adapted_route_t<i_t, f_t>& lhs, const adapted_route_t<i_t, f_t>& rhs)
{
  bool equal = true;
  equal      = equal && (lhs.start == rhs.start);
  equal      = equal && (lhs.end == rhs.end);
  equal      = equal && (lhs.id == rhs.id);
  equal      = equal && (lhs.length == rhs.length);
  equal      = equal && (lhs.vehicle_id == rhs.vehicle_id);
  equal      = equal && (lhs.bucket_id == rhs.bucket_id);
  return equal;
}

// get it from solution pool solutions
template <typename i_t, typename f_t, request_t REQUEST>
struct adapted_sol_t {
  using i_type                        = i_t;
  static const request_t request_type = REQUEST;

  costs infeasibility_cost;
  solution_t<i_t, f_t, REQUEST> sol;
  const problem_t<i_t, f_t>* problem;
  std::vector<NodeInfo<>> succ;
  std::vector<NodeInfo<>> pred;
  std::vector<adapted_node_t<i_t, f_t>> nodes;
  std::vector<adapted_route_t<i_t, f_t>> routes;
  bool has_unserviced_nodes = false;

  adapted_sol_t(solution_t<i_t, f_t, REQUEST> sol_, const problem_t<i_t, f_t>* problem_)
    : sol(sol_), problem(problem_)
  {
    initialize_host_data();
    populate_host_data(true);
  }

  // handle this glue constructor later with a better integration
  // currently the solution handle is created with a new stream
  // this solution will have the same stream as raft::handle
  adapted_sol_t(const problem_t<i_t, f_t>* problem_,
                solution_handle_t<i_t, f_t>* sol_handle_,
                std::vector<i_t> desired_vehicles = {})
    : sol(*problem_, 0, sol_handle_, desired_vehicles), problem(problem_)
  {
    raft::common::nvtx::range fun_scope("adapted_sol_t ctr");
    initialize_host_data();
    populate_host_data(true);
  }

  adapted_sol_t(adapted_sol_t& other_sol)       = default;
  adapted_sol_t(const adapted_sol_t& other_sol) = default;

  adapted_sol_t& operator=(const adapted_sol_t& other_sol)
  {
    raft::common::nvtx::range fun_scope("adapted_sol_t assignment");
    if (this == &other_sol) return *this;
    infeasibility_cost   = other_sol.infeasibility_cost;
    problem              = other_sol.problem;
    succ                 = other_sol.succ;
    pred                 = other_sol.pred;
    nodes                = other_sol.nodes;
    routes               = other_sol.routes;
    has_unserviced_nodes = other_sol.has_unserviced_nodes;
    sol.copy_device_solution(const_cast<solution_t<i_t, f_t, REQUEST>&>(other_sol.sol));
    cuopt_assert(routes.size() == sol.n_routes, "Route count mismatch!");
    return *this;
  }

  auto bucket_id(int vehicle_id) const { return problem->fleet_info_h.buckets[vehicle_id]; }

  auto get_node(i_t i) const { return nodes[i]; }

  void reset_viable_of_problem()
  {
    raft::common::nvtx::range fun_scope("reset_viable_of_problem");
    initialize_incompatible<i_t, f_t, REQUEST>(const_cast<problem_t<i_t, f_t>&>(*sol.problem_ptr),
                                               &sol);
  }

  bool inline check_device_host_coherence_()
  {
    [[maybe_unused]] const auto copy_infeasibility_cost = infeasibility_cost;
    const auto copy_succ                                = succ;
    const auto copy_pred                                = pred;
    const auto copy_nodes                               = nodes;
    const auto copy_routes                              = routes;
    sol.compute_backward_forward();
    populate_host_data(true);
    cuopt_assert(copy_infeasibility_cost == infeasibility_cost, "Host/device coherence error!");
    cuopt_assert(copy_succ == succ, "Host/device coherence error!");
    cuopt_assert(copy_pred == pred, "Host/device coherence error!");
    cuopt_assert(copy_nodes == nodes, "Host/device coherence error!");
    cuopt_assert(copy_routes == routes, "Host/device coherence error!");
    return true;
  }
  // used to check device and host coherence in assert mode
  void check_device_host_coherence() { cuopt_assert(check_device_host_coherence_(), ""); }

  bool is_feasible() const { return sol.is_feasible(); }

  double get_cost(costs& weight)
  {
    raft::common::nvtx::range fun_scope("get_cost");
    auto cuopt_weight = get_cuopt_cost(weight);
    sol.compute_cost();
    return sol.get_total_cost(cuopt_weight);
  }

  const std::vector<adapted_route_t<i_t, f_t>>& get_routes() const { return routes; }
  const adapted_route_t<i_t, f_t>& get_route(const int route_id) const { return routes[route_id]; }

  bool unserviced(int node) const
  {
    raft::common::nvtx::range fun_scope("unserviced");
    return (!pred[node].is_valid() && !succ[node].is_valid());
  }

  // adds nodes to best position that minimizes cluster violations
  void add_nodes_to_best(const std::vector<NodeInfo<>>& nodes_to_insert, costs& weight)
  {
    raft::common::nvtx::range fun_scope("add_nodes_to_best");
    sol.add_nodes_to_best(nodes_to_insert, get_cuopt_cost(weight));
    populate_host_data();
    check_device_host_coherence();
  }

  // unsets the routes to search
  // NOTE: all the functions that modify the solution state in adapted_sol should change the
  // routes_to_search array
  void unset_routes_to_search()
  {
    raft::common::nvtx::range fun_scope("unset_routes_to_search");
    sol.unset_routes_to_search();
  }

  void set_routes_to_search()
  {
    raft::common::nvtx::range fun_scope("set_routes_to_search");
    sol.set_routes_to_search();
  }

  // insert a set of nodes(a cycle or a path) to the route
  void add_nodes_to_route(const std::vector<NodeInfo<>>& nodes_to_insert,
                          NodeInfo<> prev_node,
                          NodeInfo<> next_node)
  {
    raft::common::nvtx::range fun_scope("add_nodes_to_route");
    i_t route_id, intra_idx;
    if (!prev_node.is_depot()) {
      std::tie(route_id, intra_idx) = sol.route_node_map.get_route_id_and_intra_idx(prev_node);
    } else {
      route_id  = sol.route_node_map.get_route_id(next_node);
      intra_idx = 0;
    }
    thrust::fill(sol.sol_handle->get_thrust_policy(),
                 sol.routes_to_copy.data() + route_id,
                 sol.routes_to_copy.data() + route_id + 1,
                 1);
    sol.add_nodes_to_route(nodes_to_insert, route_id, intra_idx);
    populate_host_data();
    check_device_host_coherence();
  }

  // removes the given nodes, returns true if successful
  // returns false if we encounter an empty route
  bool remove_nodes(const std::vector<NodeInfo<>>& nodes_to_eject)
  {
    raft::common::nvtx::range fun_scope("remove_nodes");
    bool success = sol.remove_nodes(nodes_to_eject);
    populate_host_data();
    check_device_host_coherence();
    return success;
  }

  double calculate_similarity_radius(const adapted_sol_t<i_t, f_t, REQUEST>& second) const
  {
    // always do symmetric measure if it is a CVRP or if there are unserviced nodes
    if (problem->is_tsp || problem->is_cvrp() || this->has_unserviced_nodes ||
        second.has_unserviced_nodes) {
      return 0.5 * (this->calculate_similarity_radius_asymetric(second) +
                    second.calculate_similarity_radius_asymetric(*this));
    }

    return calculate_similarity_radius_asymetric(second);
  }

  double calculate_similarity_radius_asymetric(const adapted_sol_t<i_t, f_t, REQUEST>& second) const
  {
    raft::common::nvtx::range fun_scope("calculate_similarity_radius");
    // check_device_host_coherence();
    int common_edges    = 0;
    int nodes           = problem->get_num_orders();
    bool depot_included = problem->order_info.depot_included_;

    for (int i = (int)depot_included; i < nodes; i++) {
      if (succ[i] == second.succ[i] || succ[i] == second.pred[i]) {
        common_edges++;
      } else if (problem->is_cvrp() &&
                 (!succ[i].is_depot() &&
                  second.nodes[i].r_id == second.nodes[succ[i].node()].r_id)) {
        // For edge ij in solution A, if both i and j belong to same route, we consider it to be
        // similar in case of CVRP
        common_edges++;
      }
    }

    common_edges *= 2;

    for (auto& a : routes) {
      if (second.pred[a.start.node()].is_depot() || second.succ[a.start.node()].is_depot()) {
        common_edges++;
      }
    }

    for (auto& a : second.routes) {
      if (second.pred[a.start.node()].is_depot() || second.succ[a.start.node()].is_depot()) {
        common_edges++;
      }
    }

    double max_symmetrical_diference =
      (double)(sol.n_routes + second.sol.n_routes + 2 * (nodes - (int)depot_included));
    return (double)common_edges / max_symmetrical_diference;
  }

  void add_new_routes(const std::vector<std::pair<int, std::vector<NodeInfo<>>>>& routes)
  {
    raft::common::nvtx::range fun_scope("add_new_routes");
    cuopt_assert(routes.size() > 0, "Indices array cannot be empty");
    thrust::fill(sol.sol_handle->get_thrust_policy(),
                 sol.routes_to_copy.data() + sol.n_routes,
                 sol.routes_to_copy.data() + sol.n_routes + routes.size(),
                 1);
    sol.global_runtime_checks(false, false, "add_new_routes_begin");
    sol.add_routes(routes);
    sol.compute_backward_forward();
    // In case of PDP, we may add only the pickup nodes but not delivery nodes
    if constexpr (REQUEST != request_t::PDP) {
      sol.global_runtime_checks(false, false, "add_new_routes_end");
    }
    sol.sol_handle->sync_stream();
    populate_host_data();
    check_device_host_coherence();
  };

  void remove_host_route(i_t route_id)
  {
    for (size_t i = route_id + 1; i < routes.size(); ++i) {
      routes[i - 1]    = routes[i];
      routes[i - 1].id = i - 1;
      NodeInfo<> start = routes[i - 1].start;
      while (!start.is_depot()) {
        nodes[start.node()].r_id = i - 1;
        nodes[start.node()].v_id = routes[i - 1].vehicle_id;
        start                    = succ[start.node()];
      }
    }
    routes.pop_back();
  }

  std::vector<i_t> get_nodes_of_routes(const std::vector<i_t>& routes_to_copy) const
  {
    raft::common::nvtx::range fun_scope("remove_route");
    std::vector<i_t> copy_nodes;
    if (routes_to_copy.empty()) { return copy_nodes; }

    copy_nodes.reserve(nodes.size());
    for (size_t i = 0; i < routes_to_copy.size(); ++i) {
      i_t id_to_remove = routes_to_copy[i];
      NodeInfo<> start = routes[id_to_remove].start;
      while (!start.is_depot()) {
        copy_nodes.push_back(start.node());
        start = succ[start.node()];
      }
    }
    return copy_nodes;
  }

  void remove_routes(std::vector<i_t> routes_to_remove)
  {
    if (routes_to_remove.empty()) { return; }
    raft::common::nvtx::range fun_scope("remove_route");
    std::sort(routes_to_remove.begin(), routes_to_remove.end());
    sol.remove_routes(routes_to_remove);
    for (size_t i = 0; i < routes_to_remove.size(); ++i) {
      i_t id_to_remove = routes_to_remove[i] - i;
      remove_host_route(id_to_remove);
    }
    RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
    populate_host_data(false, true);
    check_device_host_coherence();
  }

  std::vector<i_t> priority_remove_diff_routes(adapted_sol_t<i_t, f_t, REQUEST> const& input)
  {
    cuopt_assert(sol.get_n_routes() > input.sol.get_n_routes(), "unexpected route count!");
    size_t num_routes_to_remove = std::abs(sol.get_n_routes() - input.sol.get_n_routes());
    if (!num_routes_to_remove) { return {}; }

    std::vector<i_t> remove_route_ids;
    remove_route_ids.reserve(sol.get_n_routes());
    different_route_ids(remove_route_ids, input);

    // we need to remove num_routes_to_remove routes from this solution
    // we favor removal of routes with less number of nodes to maintain structure
    if (num_routes_to_remove <= remove_route_ids.size()) {
      // if remove_route_ids is sufficiently large then proceed to remove (a subset) of them
      std::vector<i_t> route_priority;
      route_priority.reserve(remove_route_ids.size());
      for (auto& id : remove_route_ids) {
        route_priority.push_back(routes[id].length);
      }
      std::sort(remove_route_ids.begin(), remove_route_ids.end(), [&](auto i, auto j) {
        return route_priority[i] < route_priority[j];
      });
      remove_route_ids.resize(num_routes_to_remove);
    } else {
      // if remove_route_ids is not sufficiently large then we remove them and select
      // some routes that have least number of nodes
      std::vector<i_t> route_ids(sol.get_n_routes());
      std::iota(route_ids.begin(), route_ids.end(), 0);
      std::vector<i_t> route_priority(route_ids.size(), std::numeric_limits<i_t>::max());
      for (auto& id : remove_route_ids) {
        route_priority[id] = routes[id].length;
      }
      auto diff_route_partition = std::partition(route_ids.begin(), route_ids.end(), [&](auto i) {
        return route_priority[i] < std::numeric_limits<i_t>::max();
      });
      std::sort(route_ids.begin(), diff_route_partition, [&](auto i, auto j) {
        return routes[i].length < routes[j].length;
      });
      std::sort(diff_route_partition, route_ids.end(), [&](auto i, auto j) {
        return routes[i].length < routes[j].length;
      });
      route_ids.resize(num_routes_to_remove);
      remove_route_ids = route_ids;
    }

    auto removed_nodes = get_nodes_of_routes(remove_route_ids);
    remove_routes(remove_route_ids);
    populate_host_data();
    cuopt_assert(sol.get_n_routes() == input.sol.get_n_routes(), "route count not equalized!");
    return removed_nodes;
  }

  // route_ids has a size of max n_routes, return the route size with that
  // return the vector of different route ids (A/B) not (A U B) / (A ^ B)
  void different_route_ids(std::vector<i_t>& route_ids,
                           adapted_sol_t<i_t, f_t, REQUEST> const& input) const noexcept
  {
    raft::common::nvtx::range fun_scope("different_route_ids");
    route_ids.clear();
    for (size_t i = 0; i < routes.size(); i++) {
      if (routes[i].is_empty()) { continue; }

      auto start = routes[i].start;

      if (input.pred[start.node()] != pred[start.node()]) {
        route_ids.push_back((int)i);
      } else {
        while (!start.is_depot()) {
          if (input.succ[start.node()] != succ[start.node()]) {
            route_ids.push_back((int)i);
            break;
          }
          start = succ[start.node()];
        }
      }
    }
  }

  void initialize_host_data()
  {
    nodes.resize(sol.get_num_orders());
    pred.resize(sol.get_num_orders());
    succ.resize(sol.get_num_orders());
    routes.resize(sol.n_routes);
    std::fill(nodes.begin(), nodes.end(), adapted_node_t<i_t, f_t>());
    std::fill(pred.begin(), pred.end(), NodeInfo<>());
    std::fill(succ.begin(), succ.end(), NodeInfo<>());
    std::fill(routes.begin(), routes.end(), adapted_route_t<i_t, f_t>());
  }

  void populate_unserviced_nodes()
  {
    raft::common::nvtx::range fun_scope("populate_unserviced_nodes");
    has_unserviced_nodes     = false;
    auto h_route_id_per_node = host_copy(sol.route_node_map.route_id_per_node);
    for (size_t i = 0; i < h_route_id_per_node.size(); ++i) {
      if (h_route_id_per_node[i] == -1) {
        pred[i]            = NodeInfo<>();
        succ[i]            = NodeInfo<>();
        nodes[i].r_index   = std::numeric_limits<size_t>::max();
        nodes[i].r_id      = std::numeric_limits<size_t>::max();
        nodes[i].node_info = problem->get_node_info_of_node(i);

        // Ignore depot node.
        if (i >= problem->order_info.depot_included_) { has_unserviced_nodes = true; }
      } else {
        cuopt_assert(pred[i].is_valid(), "Pred mismatch!");
        cuopt_assert(succ[i].is_valid(), "Succ mismatch!");
        cuopt_assert(nodes[i].r_id == h_route_id_per_node[i], "Route id mismatch!");
      }
    }
  }

  void clear_solution(std::vector<i_t> vehicle_ids)
  {
    raft::common::nvtx::range fun_scope("clear_solution");
    sol.clear_routes(vehicle_ids);
    initialize_host_data();
  }

  void populate_host_data(bool copy_all = false, bool skip_route_copy = false)
  {
    raft::common::nvtx::range fun_scope("populate_host_data");
    sol.compute_cost();
    cuopt_func_call(sol.check_cost_coherence(default_weights));
    sol.sol_handle->sync_stream();
    if (routes.size() != (size_t)sol.n_routes) {
      routes.resize(sol.n_routes);
      skip_route_copy = false;
    }
    std::vector<i_t> h_routes_to_copy;
    if (!copy_all) h_routes_to_copy = host_copy(sol.routes_to_copy);
    for (i_t i = 0; i < sol.n_routes && !skip_route_copy; ++i) {
      if (!copy_all && h_routes_to_copy[i] == 0) continue;
      auto& curr_route     = sol.get_route(i);
      auto node_infos_temp = host_copy(curr_route.dimensions.requests.node_info);
      i_t n_nodes          = curr_route.n_nodes.value(sol.sol_handle->get_stream());

      // Remove break nodes for diversity
      std::vector<NodeInfo<>> node_infos;
      std::copy_if(node_infos_temp.begin(),
                   node_infos_temp.begin() + n_nodes + 1,
                   std::back_inserter(node_infos),
                   [](auto& node_info) { return !node_info.is_break(); });
      n_nodes = node_infos.size() - 1;

      i_t vehicle_id = curr_route.vehicle_id.value(sol.sol_handle->get_stream());
      for (i_t j = 0; j < n_nodes; ++j) {
        if (!node_infos[j].is_service_node()) continue;
        i_t node_id              = node_infos[j].node();
        nodes[node_id].r_index   = j;
        nodes[node_id].r_id      = i;
        nodes[node_id].v_id      = vehicle_id;
        nodes[node_id].node_info = node_infos[j];
        pred[node_id]            = node_infos[j - 1];
        succ[node_id]            = node_infos[j + 1];
      }
      routes[i].initialize(
        node_infos[1], node_infos[n_nodes - 1], i, n_nodes, vehicle_id, bucket_id(vehicle_id));
    }
    // set depot's
    if (problem->order_info.depot_included_) {
      pred[0] = NodeInfo<>();
      succ[0] = NodeInfo<>();
    }

    infeasibility_cost = get_cpu_cost(sol.get_infeasibility_cost());
    sol.unset_routes_to_copy();
    populate_unserviced_nodes();
    sol.sol_handle->sync_stream();
  }
};

template struct adapted_route_t<int, float>;
template struct adapted_sol_t<int, float, request_t::PDP>;
template struct adapted_sol_t<int, float, request_t::VRP>;

}  // namespace cuopt::routing::detail
