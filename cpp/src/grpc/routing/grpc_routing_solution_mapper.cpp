/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "grpc_routing_solution_mapper.hpp"

#include "grpc_routing_mapper_utils.hpp"

#include <cuopt/error.hpp>

#include <cstdint>
#include <string>

namespace cuopt {
namespace routing {

using grpc_map_detail::copy_repeated_to_vector;
using grpc_map_detail::copy_vector_to_repeated;

namespace {

// Mirror grpc_server_types.hpp's format_cuopt_error without pulling that heavy
// server header into this (client-linked) mapper: parse the structured
// {"error_type","msg"} payload logic_error embeds down to a clean "type: msg"
// so raw internal error detail is not sent to remote clients.
std::string sanitize_error_message(const cuopt::logic_error& e)
{
  std::string s = e.what();
  std::string msg;
  auto pos = s.find("\"msg\": \"");
  if (pos != std::string::npos) {
    pos += 8;
    auto end = s.rfind('"');
    if (end > pos) { msg = s.substr(pos, end - pos); }
  }
  if (msg.empty()) { msg = s; }
  return cuopt::error_to_string(e.get_error_type()) + ": " + msg;
}

cuopt::remote::RoutingSolutionStatus to_proto_status(cuopt::routing::solution_status_t s)
{
  using cuopt::routing::solution_status_t;
  switch (s) {
    case solution_status_t::SUCCESS: return cuopt::remote::ROUTING_SUCCESS;
    case solution_status_t::INFEASIBLE: return cuopt::remote::ROUTING_INFEASIBLE;
    case solution_status_t::TIMEOUT: return cuopt::remote::ROUTING_TIMEOUT;
    case solution_status_t::EMPTY: return cuopt::remote::ROUTING_EMPTY;
    case solution_status_t::ERROR: return cuopt::remote::ROUTING_ERROR;
  }
  return cuopt::remote::ROUTING_ERROR;
}

}  // namespace

void map_routing_solution_to_proto(const cuopt::routing::assignment_t<int>& assignment,
                                   const cuopt::routing::host_assignment_t<int>& host,
                                   cuopt::remote::RoutingSolution* pb)
{
  pb->Clear();
  copy_vector_to_repeated(host.route, pb->mutable_route());
  copy_vector_to_repeated(host.stamp, pb->mutable_arrival_stamp());
  copy_vector_to_repeated(host.truck_id, pb->mutable_truck_id());
  copy_vector_to_repeated(host.locations, pb->mutable_locations());
  copy_vector_to_repeated(host.node_types, pb->mutable_node_types());
  copy_vector_to_repeated(host.unserviced_nodes, pb->mutable_unserviced_nodes());
  copy_vector_to_repeated(host.accepted, pb->mutable_accepted());

  pb->set_vehicle_count(assignment.get_vehicle_count());
  pb->set_total_objective_value(assignment.get_total_objective());
  for (auto const& [obj, value] : assignment.get_objectives()) {
    (*pb->mutable_objective_values())[static_cast<int32_t>(obj)] = value;
  }

  pb->set_status(to_proto_status(assignment.get_status()));
  pb->set_status_message(assignment.get_status_string());
  if (assignment.get_status() == cuopt::routing::solution_status_t::ERROR) {
    try {
      pb->set_error_message(sanitize_error_message(assignment.get_error_status()));
    } catch (...) {
      pb->set_error_message("routing solve error");
    }
  }
}

void map_proto_to_routing_solution(const cuopt::remote::RoutingSolution& pb,
                                   cuopt::routing::cpu_routing_solution_t& sol)
{
  copy_repeated_to_vector(pb.route(), sol.route);
  copy_repeated_to_vector(pb.arrival_stamp(), sol.arrival_stamp);
  copy_repeated_to_vector(pb.truck_id(), sol.truck_id);
  copy_repeated_to_vector(pb.locations(), sol.locations);
  copy_repeated_to_vector(pb.node_types(), sol.node_types);
  copy_repeated_to_vector(pb.unserviced_nodes(), sol.unserviced_nodes);
  copy_repeated_to_vector(pb.accepted(), sol.accepted);

  sol.vehicle_count         = pb.vehicle_count();
  sol.total_objective_value = pb.total_objective_value();
  sol.objective_values.clear();
  for (auto const& [obj, value] : pb.objective_values()) {
    sol.objective_values[static_cast<int32_t>(obj)] = value;
  }
  sol.status         = static_cast<int32_t>(pb.status());
  sol.status_message = pb.status_message();
  sol.error_message  = pb.error_message();
}

}  // namespace routing
}  // namespace cuopt
