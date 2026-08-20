/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "grpc_routing_problem_mapper.hpp"

#include "grpc_routing_mapper_utils.hpp"

#include <cstdint>
#include <utility>

namespace cuopt {
namespace routing {

using grpc_map_detail::copy_bool_to_u8;
using grpc_map_detail::copy_repeated_to_vector;
using grpc_map_detail::copy_u32_to_u8;
using grpc_map_detail::copy_vector_to_repeated;

void map_proto_to_routing_problem(const cuopt::remote::RoutingProblem& pb,
                                  cuopt::routing::cpu_routing_problem_t& p)
{
  p               = cuopt::routing::cpu_routing_problem_t{};
  p.num_locations = pb.num_locations();
  p.fleet_size    = pb.fleet_size();
  p.num_orders    = pb.num_orders();

  for (auto const& cm : pb.cost_matrices()) {
    cuopt::routing::cpu_cost_matrix_t out;
    out.vehicle_type = static_cast<uint8_t>(cm.vehicle_type());
    copy_repeated_to_vector(cm.values(), out.matrix);
    p.cost_matrices.push_back(std::move(out));
  }
  for (auto const& tm : pb.transit_time_matrices()) {
    cuopt::routing::cpu_cost_matrix_t out;
    out.vehicle_type = static_cast<uint8_t>(tm.vehicle_type());
    copy_repeated_to_vector(tm.values(), out.matrix);
    p.transit_time_matrices.push_back(std::move(out));
  }

  copy_repeated_to_vector(pb.vehicle_start_locations(), p.vehicle_start_locations);
  copy_repeated_to_vector(pb.vehicle_return_locations(), p.vehicle_return_locations);
  copy_repeated_to_vector(pb.vehicle_tw_earliest(), p.vehicle_tw_earliest);
  copy_repeated_to_vector(pb.vehicle_tw_latest(), p.vehicle_tw_latest);
  copy_u32_to_u8(pb.vehicle_types(), p.vehicle_types);
  copy_bool_to_u8(pb.drop_return_trips(), p.drop_return_trips);
  copy_bool_to_u8(pb.skip_first_trips(), p.skip_first_trips);
  copy_repeated_to_vector(pb.vehicle_max_costs(), p.vehicle_max_costs);
  copy_repeated_to_vector(pb.vehicle_max_times(), p.vehicle_max_times);
  copy_repeated_to_vector(pb.vehicle_fixed_costs(), p.vehicle_fixed_costs);

  copy_repeated_to_vector(pb.order_locations(), p.order_locations);
  copy_repeated_to_vector(pb.order_tw_earliest(), p.order_tw_earliest);
  copy_repeated_to_vector(pb.order_tw_latest(), p.order_tw_latest);
  copy_repeated_to_vector(pb.order_prizes(), p.order_prizes);
  for (auto const& st : pb.order_service_times()) {
    std::vector<int32_t> times;
    copy_repeated_to_vector(st.service_times(), times);
    p.order_service_times[st.vehicle_id()] = std::move(times);
  }

  copy_repeated_to_vector(pb.pickup_indices(), p.pickup_indices);
  copy_repeated_to_vector(pb.delivery_indices(), p.delivery_indices);

  for (auto const& dim : pb.capacity_dimensions()) {
    cuopt::routing::cpu_capacity_dimension_t out;
    out.name = dim.name();
    copy_repeated_to_vector(dim.demand(), out.demand);
    copy_repeated_to_vector(dim.capacity(), out.capacity);
    p.capacity_dimensions.push_back(std::move(out));
  }

  copy_repeated_to_vector(pb.break_locations(), p.break_locations);
  for (auto const& ub : pb.uniform_breaks()) {
    cuopt::routing::cpu_uniform_break_t out;
    copy_repeated_to_vector(ub.earliest(), out.earliest);
    copy_repeated_to_vector(ub.latest(), out.latest);
    copy_repeated_to_vector(ub.duration(), out.duration);
    p.uniform_breaks.push_back(std::move(out));
  }
  for (auto const& pvb : pb.vehicle_breaks()) {
    std::vector<cuopt::routing::cpu_vehicle_break_t> breaks;
    for (auto const& b : pvb.breaks()) {
      cuopt::routing::cpu_vehicle_break_t out;
      out.earliest = b.earliest();
      out.latest   = b.latest();
      out.duration = b.duration();
      copy_repeated_to_vector(b.locations(), out.locations);
      breaks.push_back(std::move(out));
    }
    p.vehicle_breaks[pvb.vehicle_id()] = std::move(breaks);
  }

  for (auto const& m : pb.vehicle_order_match()) {
    std::vector<int32_t> matches;
    copy_repeated_to_vector(m.matches(), matches);
    p.vehicle_order_match[m.id()] = std::move(matches);
  }
  for (auto const& m : pb.order_vehicle_match()) {
    std::vector<int32_t> matches;
    copy_repeated_to_vector(m.matches(), matches);
    p.order_vehicle_match[m.id()] = std::move(matches);
  }
  for (auto const& prec : pb.order_precedence()) {
    std::vector<int32_t> preceding;
    copy_repeated_to_vector(prec.preceding_orders(), preceding);
    p.order_precedence[prec.order_id()] = std::move(preceding);
  }

  if (pb.has_objective()) {
    copy_repeated_to_vector(pb.objective().objectives(), p.objectives);
    copy_repeated_to_vector(pb.objective().weights(), p.objective_weights);
  }
  p.min_vehicles = pb.min_vehicles();

  if (pb.has_initial_solutions()) {
    auto const& init = pb.initial_solutions();
    copy_repeated_to_vector(init.vehicle_ids(), p.initial_solutions.vehicle_ids);
    copy_repeated_to_vector(init.routes(), p.initial_solutions.routes);
    copy_repeated_to_vector(init.types(), p.initial_solutions.types);
    copy_repeated_to_vector(init.sol_offsets(), p.initial_solutions.sol_offsets);
  }
}

void map_routing_problem_to_proto(const cuopt::routing::cpu_routing_problem_t& p,
                                  cuopt::remote::RoutingProblem* pb)
{
  pb->Clear();
  pb->set_num_locations(p.num_locations);
  pb->set_fleet_size(p.fleet_size);
  pb->set_num_orders(p.num_orders);

  for (auto const& cm : p.cost_matrices) {
    auto* out = pb->add_cost_matrices();
    out->set_vehicle_type(cm.vehicle_type);
    copy_vector_to_repeated(cm.matrix, out->mutable_values());
  }
  for (auto const& tm : p.transit_time_matrices) {
    auto* out = pb->add_transit_time_matrices();
    out->set_vehicle_type(tm.vehicle_type);
    copy_vector_to_repeated(tm.matrix, out->mutable_values());
  }

  copy_vector_to_repeated(p.vehicle_start_locations, pb->mutable_vehicle_start_locations());
  copy_vector_to_repeated(p.vehicle_return_locations, pb->mutable_vehicle_return_locations());
  copy_vector_to_repeated(p.vehicle_tw_earliest, pb->mutable_vehicle_tw_earliest());
  copy_vector_to_repeated(p.vehicle_tw_latest, pb->mutable_vehicle_tw_latest());
  for (auto v : p.vehicle_types) {
    pb->add_vehicle_types(v);
  }
  for (auto v : p.drop_return_trips) {
    pb->add_drop_return_trips(v != 0);
  }
  for (auto v : p.skip_first_trips) {
    pb->add_skip_first_trips(v != 0);
  }
  copy_vector_to_repeated(p.vehicle_max_costs, pb->mutable_vehicle_max_costs());
  copy_vector_to_repeated(p.vehicle_max_times, pb->mutable_vehicle_max_times());
  copy_vector_to_repeated(p.vehicle_fixed_costs, pb->mutable_vehicle_fixed_costs());

  copy_vector_to_repeated(p.order_locations, pb->mutable_order_locations());
  copy_vector_to_repeated(p.order_tw_earliest, pb->mutable_order_tw_earliest());
  copy_vector_to_repeated(p.order_tw_latest, pb->mutable_order_tw_latest());
  copy_vector_to_repeated(p.order_prizes, pb->mutable_order_prizes());
  for (auto const& [vehicle_id, times] : p.order_service_times) {
    auto* out = pb->add_order_service_times();
    out->set_vehicle_id(vehicle_id);
    copy_vector_to_repeated(times, out->mutable_service_times());
  }

  copy_vector_to_repeated(p.pickup_indices, pb->mutable_pickup_indices());
  copy_vector_to_repeated(p.delivery_indices, pb->mutable_delivery_indices());

  for (auto const& dim : p.capacity_dimensions) {
    auto* out = pb->add_capacity_dimensions();
    out->set_name(dim.name);
    copy_vector_to_repeated(dim.demand, out->mutable_demand());
    copy_vector_to_repeated(dim.capacity, out->mutable_capacity());
  }

  copy_vector_to_repeated(p.break_locations, pb->mutable_break_locations());
  for (auto const& ub : p.uniform_breaks) {
    auto* out = pb->add_uniform_breaks();
    copy_vector_to_repeated(ub.earliest, out->mutable_earliest());
    copy_vector_to_repeated(ub.latest, out->mutable_latest());
    copy_vector_to_repeated(ub.duration, out->mutable_duration());
  }
  for (auto const& [vehicle_id, breaks] : p.vehicle_breaks) {
    auto* out = pb->add_vehicle_breaks();
    out->set_vehicle_id(vehicle_id);
    for (auto const& b : breaks) {
      auto* brk = out->add_breaks();
      brk->set_earliest(b.earliest);
      brk->set_latest(b.latest);
      brk->set_duration(b.duration);
      copy_vector_to_repeated(b.locations, brk->mutable_locations());
    }
  }

  for (auto const& [id, matches] : p.vehicle_order_match) {
    auto* out = pb->add_vehicle_order_match();
    out->set_id(id);
    copy_vector_to_repeated(matches, out->mutable_matches());
  }
  for (auto const& [id, matches] : p.order_vehicle_match) {
    auto* out = pb->add_order_vehicle_match();
    out->set_id(id);
    copy_vector_to_repeated(matches, out->mutable_matches());
  }
  for (auto const& [order_id, preceding] : p.order_precedence) {
    auto* out = pb->add_order_precedence();
    out->set_order_id(order_id);
    copy_vector_to_repeated(preceding, out->mutable_preceding_orders());
  }

  if (!p.objectives.empty()) {
    auto* obj = pb->mutable_objective();
    copy_vector_to_repeated(p.objectives, obj->mutable_objectives());
    copy_vector_to_repeated(p.objective_weights, obj->mutable_weights());
  }
  pb->set_min_vehicles(p.min_vehicles);

  if (!p.initial_solutions.routes.empty()) {
    auto* init = pb->mutable_initial_solutions();
    copy_vector_to_repeated(p.initial_solutions.vehicle_ids, init->mutable_vehicle_ids());
    copy_vector_to_repeated(p.initial_solutions.routes, init->mutable_routes());
    copy_vector_to_repeated(p.initial_solutions.types, init->mutable_types());
    copy_vector_to_repeated(p.initial_solutions.sol_offsets, init->mutable_sol_offsets());
  }
}

}  // namespace routing
}  // namespace cuopt
