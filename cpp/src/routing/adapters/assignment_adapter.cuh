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

#include "../ges_solver.cuh"
#include "../routing_helpers.cuh"

#include <cuopt/routing/assignment.hpp>
#include <utilities/copy_helpers.hpp>

#include <thrust/sequence.h>

namespace cuopt {
namespace routing {

template <typename i_t, typename f_t, request_t REQUEST>
assignment_t<i_t> ges_solver_t<i_t, f_t, REQUEST>::get_ges_assignment(
  detail::solution_t<i_t, f_t, REQUEST>& sol, std::vector<i_t> const& accepted)
{
  // synhronize stream so that we can copy items
  sol.sol_handle->sync_stream();
  // the stream should be the initial handle stream and not the sol_handle stream as this data will
  // be exported
  auto stream = problem.handle_ptr->get_stream();
  stream.synchronize();

  const auto& problem = *sol.problem_ptr;
  i_t n_output_nodes  = sol.get_n_routes() * 2 + sol.get_num_depot_excluded_orders() +
                       sol.get_n_routes() * problem.get_max_break_dimensions();

  rmm::device_uvector<i_t> route_out(0, stream);
  rmm::device_uvector<double> arrival_out(0, stream);
  rmm::device_uvector<i_t> truck_id_out(0, stream);
  rmm::device_uvector<i_t> route_locations_out(0, stream);
  rmm::device_uvector<i_t> node_types_out(0, stream);
  auto accepted_out = cuopt::device_copy(accepted, stream);
  stream.synchronize();
  std::vector<i_t> node_types_out_h(n_output_nodes);
  std::vector<i_t> route_out_h(n_output_nodes);
  std::vector<i_t> truck_id_out_h(n_output_nodes);
  std::vector<i_t> route_locations_out_h(n_output_nodes);
  std::vector<double> arrival_out_h(n_output_nodes);

  auto sol_status = solution_status_t::EMPTY;
  std::string sol_string{solution_string_t::empty};
  sol.sol_found = (sol.get_n_routes() <= sol.problem_ptr->data_view_ptr->get_fleet_size());
  if (sol.is_feasible()) {
    sol_status = solution_status_t::SUCCESS;
    sol_string = solution_string_t::success;
  } else if (sol.sol_found) {
    sol_status = solution_status_t::INFEASIBLE;
    sol_string =
      "Feasible solutions could not be found. Closest infeasible solution is found with "
      "infeasibilities in:\n";
    auto infeasibility_cost = sol.get_infeasibility_cost();
    detail::loop_over_dimensions(problem.dimensions_info, [&](auto I) {
      if (infeasibility_cost[I] > 0.f)
        sol_string += "\t" + std::string(detail::dim_to_string<I>()) + "\n";
    });
  }

  int n_routes                = 0;
  double total_objective_cost = 0.;
  detail::objective_cost_t obj_costs;
  if (sol.sol_found) {
    i_t offset  = 0;
    i_t counter = 0;
    sol.remove_empty_routes();
    sol.compute_cost();
    sol.compute_actual_arrival_times();

    n_routes             = sol.get_n_routes();
    obj_costs            = sol.get_objective_cost();
    auto obj_weights     = sol.problem_ptr->dimensions_info.objective_weights;
    total_objective_cost = detail::objective_cost_t::dot(obj_weights, obj_costs);
    for (i_t i = 0; i < sol.get_n_routes(); ++i) {
      const auto& route         = sol.get_route(i);
      auto size_including_depot = route.n_nodes.value(stream) + 1;
      auto route_id             = route.route_id.value(stream);
      auto vehicle_id           = route.vehicle_id.value(stream);

      auto node_infos_h = cuopt::host_copy(route.dimensions.requests.node_info);
      std::vector<double> departure_forward_h(node_infos_h.size(), 0.);
      std::vector<double> actual_arrival_h(node_infos_h.size(), 0.);
      std::vector<double> earliest_arrival_backward_h(node_infos_h.size(), 0.);
      std::vector<double> latest_arrival_forward_h(node_infos_h.size(), 0.);
      if (problem.dimensions_info.has_dimension(detail::dim_t::TIME)) {
        departure_forward_h = cuopt::host_copy(route.dimensions.time_dim.departure_forward);
        actual_arrival_h    = cuopt::host_copy(route.dimensions.time_dim.actual_arrival);
        earliest_arrival_backward_h =
          cuopt::host_copy(route.dimensions.time_dim.earliest_arrival_backward);
        latest_arrival_forward_h =
          cuopt::host_copy(route.dimensions.time_dim.latest_arrival_forward);
      }

      i_t drop_return_trip = sol.problem_ptr->drop_return_trip_h[vehicle_id];
      i_t skip_first_trip  = sol.problem_ptr->skip_first_trip_h[vehicle_id];
      i_t route_end        = size_including_depot - drop_return_trip;
      i_t route_begin      = skip_first_trip;
      for (int i = route_begin; i < route_end; ++i) {
        auto node_info            = node_infos_h[i];
        auto node_id              = node_info.node();
        node_type_t node_type     = node_info.node_type();
        node_types_out_h[counter] = (int)node_type;
        // We are setting node value to be norders for depot in the solver,
        // set to zero here for backward compatibility
        if (node_info.is_depot()) {
          route_out_h[counter] = 0;
        } else {
          route_out_h[counter] = node_info.node();
        }
        route_locations_out_h[counter] = node_info.location();
        truck_id_out_h[counter]        = vehicle_id;
        arrival_out_h[counter]         = actual_arrival_h[i];
        // if feasible departure forward should be equal
        if (sol_status == solution_status_t::SUCCESS) {
          if (sol.problem_ptr->dimensions_info.time_dim.should_compute_travel_time()) {
            cuopt_assert(abs(actual_arrival_h[i] -
                             max(min(earliest_arrival_backward_h[i], latest_arrival_forward_h[i]),
                                 departure_forward_h[i])) < 0.0001f,
                         "Feasible time mismatch!");
          } else {
            cuopt_assert(abs(actual_arrival_h[i] - departure_forward_h[i]) < 0.0001f,
                         "Feasible time mismatch!");
          }
        }
        counter++;
      }
      offset += size_including_depot;
    }

    route_out_h.resize(counter);
    route_locations_out_h.resize(counter);
    truck_id_out_h.resize(counter);
    arrival_out_h.resize(counter);
    node_types_out_h.resize(counter);

    route_out           = cuopt::device_copy(route_out_h, stream);
    route_locations_out = cuopt::device_copy(route_locations_out_h, stream);
    truck_id_out        = cuopt::device_copy(truck_id_out_h, stream);
    arrival_out         = cuopt::device_copy(arrival_out_h, stream);
    node_types_out      = cuopt::device_copy(node_types_out_h, stream);
  }
  sol.sol_handle->sync_stream();

  auto unserviced_nodes_h = sol.get_unserviced_nodes();
  auto unserviced_nodes   = cuopt::device_copy(unserviced_nodes_h, stream);
  stream.synchronize();

  std::map<objective_t, double> objective_values;
  for (int i = 0; i < (int)objective_t::SIZE; ++i) {
    auto ob = (objective_t)i;
    if (problem.dimensions_info.has_objective(ob)) { objective_values[ob] = obj_costs[ob]; }
  }

  return assignment_t<i_t>(n_routes,
                           total_objective_cost,
                           objective_values,
                           route_out,
                           arrival_out,
                           truck_id_out,
                           route_locations_out,
                           node_types_out,
                           unserviced_nodes,
                           accepted_out,
                           sol_status,
                           sol_string);
}

}  // namespace routing
}  // namespace cuopt
