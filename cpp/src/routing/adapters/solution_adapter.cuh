/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include "../ges_solver.cuh"
#include "../solution/solution_handle.cuh"

#include <cuopt/routing/assignment.hpp>
#include <routing/structures.hpp>

#include <thrust/sequence.h>

namespace cuopt {
namespace routing {
namespace detail {

// fills basic node id data, drop and skip return trip is not supported
template <typename i_t, typename f_t, request_t REQUEST>
void fill_routes_data(solution_t<i_t, f_t, REQUEST>& sol,
                      assignment_t<i_t>& assignment,
                      const problem_t<i_t, f_t>& problem)
{
  const auto n_routes    = assignment.get_vehicle_count();
  auto h_route           = cuopt::host_copy(assignment.get_route());
  auto h_truck_ids       = cuopt::host_copy(assignment.get_truck_id());
  auto h_route_locations = cuopt::host_copy(assignment.get_order_locations());
  auto h_node_types      = cuopt::host_copy(assignment.get_node_types());

  sol.sol_handle->sync_stream();
  assignment.get_truck_id().stream().synchronize();
  i_t route_id = -1;
  NodeInfo<i_t> curr_node;
  // depot is counter only once
  i_t counter                   = 0;
  const i_t max_nodes_per_route = next_pow2(problem.get_num_orders());
  bool route_opened             = false;
  for (size_t i = 0; i < h_route.size(); ++i) {
    assert(h_route_locations[i] == h_route[i]);
    curr_node = NodeInfo<i_t>(h_route[i], h_route_locations[i], (node_type_t)h_node_types[i]);
    if (curr_node.is_depot()) {
      if (!route_opened) {
        ++route_id;
        auto new_route = route_t<i_t, f_t, REQUEST>(
          sol.sol_handle, route_id, problem.vehicle_info, problem.dimensions_info);
        new_route.resize(max_nodes_per_route);
        sol.add_route(std::move(new_route), route_id, counter, false);
        route_opened = true;
        counter      = 0;
      } else {
        auto curr_route_size = counter;
        sol.get_route(route_id).n_nodes.set_value_async(curr_route_size,
                                                        sol.sol_handle->get_stream());
        route_opened = false;
      }
    }

    cuopt_assert((size_t)counter < sol.get_route(route_id).requests.node_info.size(),
                 "Counter is bigger than max_nodes_per_route!");
    sol.get_route(route_id).requests.node_info.set_element_async(
      counter, curr_node, sol.sol_handle->get_stream());
    counter++;
  }
  sol.n_routes = route_id + 1;
  sol.sol_handle->sync_stream();
  sol.set_route_views();
  sol.sol_handle->sync_stream();
  sol.set_nodes_data_of_solution();
  sol.sol_handle->sync_stream();
  sol.compute_initial_data();
  sol.sol_handle->sync_stream();
}

// this is a simple adapter function for quick testing
// there is no data validation or guarantee this function provide
// currently only used in unit tests
template <typename i_t, typename f_t, request_t REQUEST>
void get_solution_from_assignment(solution_t<i_t, f_t, REQUEST>& sol,
                                  assignment_t<i_t>& assignment,
                                  const problem_t<i_t, f_t>& problem)
{
  sol.sol_found = true;
  sol.clear_routes();
  fill_routes_data(sol, assignment, problem);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
