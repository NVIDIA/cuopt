/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/cpu_routing_problem.hpp>

#include <utilities/copy_helpers.hpp>

#include <rmm/device_uvector.hpp>

#include <stdexcept>
#include <vector>

namespace cuopt {
namespace routing {

struct cpu_routing_problem_t::device_data_t {
  std::vector<std::unique_ptr<rmm::device_uvector<float>>> cost_matrices;
  std::vector<std::unique_ptr<rmm::device_uvector<float>>> transit_time_matrices;

  std::unique_ptr<rmm::device_uvector<int32_t>> vehicle_start_locations;
  std::unique_ptr<rmm::device_uvector<int32_t>> vehicle_return_locations;
  std::unique_ptr<rmm::device_uvector<int32_t>> vehicle_tw_earliest;
  std::unique_ptr<rmm::device_uvector<int32_t>> vehicle_tw_latest;
  std::unique_ptr<rmm::device_uvector<uint8_t>> vehicle_types;
  std::unique_ptr<rmm::device_uvector<bool>> drop_return_trips;
  std::unique_ptr<rmm::device_uvector<bool>> skip_first_trips;
  std::unique_ptr<rmm::device_uvector<float>> vehicle_max_costs;
  std::unique_ptr<rmm::device_uvector<float>> vehicle_max_times;
  std::unique_ptr<rmm::device_uvector<float>> vehicle_fixed_costs;

  std::unique_ptr<rmm::device_uvector<int32_t>> order_locations;
  std::unique_ptr<rmm::device_uvector<int32_t>> order_tw_earliest;
  std::unique_ptr<rmm::device_uvector<int32_t>> order_tw_latest;
  std::unique_ptr<rmm::device_uvector<float>> order_prizes;

  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> service_times;
  std::vector<int32_t> service_time_vehicle_ids;

  std::unique_ptr<rmm::device_uvector<int32_t>> pickup_indices;
  std::unique_ptr<rmm::device_uvector<int32_t>> delivery_indices;

  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> capacity_demands;
  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> capacity_capacities;
  std::vector<std::string> capacity_names;

  std::unique_ptr<rmm::device_uvector<int32_t>> break_locations;
  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> uniform_break_earliest;
  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> uniform_break_latest;
  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> uniform_break_duration;

  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> vehicle_break_locations;

  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> match_buffers;
  std::vector<std::unique_ptr<rmm::device_uvector<int32_t>>> precedence_buffers;

  std::unique_ptr<rmm::device_uvector<int32_t>> objectives;
  std::unique_ptr<rmm::device_uvector<float>> objective_weights;

  std::unique_ptr<rmm::device_uvector<int32_t>> init_vehicle_ids;
  std::unique_ptr<rmm::device_uvector<int32_t>> init_routes;
  std::unique_ptr<rmm::device_uvector<node_type_t>> init_types;
  std::unique_ptr<rmm::device_uvector<int32_t>> init_sol_offsets;
};

void cpu_routing_problem_t::device_data_deleter::operator()(device_data_t* p) const { delete p; }

namespace {

template <typename T>
std::unique_ptr<rmm::device_uvector<T>> copy_vector(std::vector<T> const& host,
                                                    rmm::cuda_stream_view stream)
{
  if (host.empty()) { return nullptr; }
  return std::make_unique<rmm::device_uvector<T>>(cuopt::device_copy(host, stream));
}

std::unique_ptr<rmm::device_uvector<bool>> copy_u8_as_bool(std::vector<uint8_t> const& host,
                                                           rmm::cuda_stream_view stream)
{
  if (host.empty()) { return nullptr; }
  std::vector<bool> as_bool(host.begin(), host.end());
  auto d = std::make_unique<rmm::device_uvector<bool>>(cuopt::device_copy(as_bool, stream));
  // as_bool is a local temporary and the H2D copy above is async; drain the
  // stream before it goes out of scope so the copy does not read freed host
  // memory.
  stream.sync();
  return d;
}

}  // namespace

std::pair<data_model_view_t<int, float>, cpu_routing_problem_t::device_data_ptr>
cpu_routing_problem_t::to_device(raft::handle_t* handle) const
{
  if (handle == nullptr) {
    throw std::invalid_argument("cpu_routing_problem_t::to_device: handle must not be null");
  }
  if (num_locations <= 0 || fleet_size <= 0) {
    throw std::invalid_argument(
      "cpu_routing_problem_t::to_device: num_locations and fleet_size must be positive");
  }
  if (cost_matrices.empty()) {
    throw std::invalid_argument(
      "cpu_routing_problem_t::to_device: at least one cost matrix required");
  }

  auto stream = handle->get_stream();
  device_data_ptr data(new device_data_t());

  int32_t orders = (num_orders < 0) ? num_locations : num_orders;
  data_model_view_t<int, float> view(handle, num_locations, fleet_size, orders);

  for (auto const& cm : cost_matrices) {
    auto d = copy_vector(cm.matrix, stream);
    if (!d) { throw std::invalid_argument("cpu_routing_problem_t::to_device: empty cost matrix"); }
    view.add_cost_matrix(d->data(), cm.vehicle_type);
    data->cost_matrices.push_back(std::move(d));
  }

  for (auto const& tm : transit_time_matrices) {
    auto d = copy_vector(tm.matrix, stream);
    if (!d) {
      throw std::invalid_argument("cpu_routing_problem_t::to_device: empty transit time matrix");
    }
    view.add_transit_time_matrix(d->data(), tm.vehicle_type);
    data->transit_time_matrices.push_back(std::move(d));
  }

  if (!vehicle_start_locations.empty() && !vehicle_return_locations.empty()) {
    data->vehicle_start_locations  = copy_vector(vehicle_start_locations, stream);
    data->vehicle_return_locations = copy_vector(vehicle_return_locations, stream);
    view.set_vehicle_locations(
      data->vehicle_start_locations->data(), data->vehicle_return_locations->data(), false);
  }

  if (!vehicle_tw_earliest.empty() && !vehicle_tw_latest.empty()) {
    data->vehicle_tw_earliest = copy_vector(vehicle_tw_earliest, stream);
    data->vehicle_tw_latest   = copy_vector(vehicle_tw_latest, stream);
    view.set_vehicle_time_windows(
      data->vehicle_tw_earliest->data(), data->vehicle_tw_latest->data(), false);
  }

  if (!vehicle_types.empty()) {
    data->vehicle_types = copy_vector(vehicle_types, stream);
    view.set_vehicle_types(data->vehicle_types->data(), false);
  }

  if (!drop_return_trips.empty()) {
    data->drop_return_trips = copy_u8_as_bool(drop_return_trips, stream);
    view.set_drop_return_trips(data->drop_return_trips->data());
  }

  if (!skip_first_trips.empty()) {
    data->skip_first_trips = copy_u8_as_bool(skip_first_trips, stream);
    view.set_skip_first_trips(data->skip_first_trips->data());
  }

  if (!vehicle_max_costs.empty()) {
    data->vehicle_max_costs = copy_vector(vehicle_max_costs, stream);
    view.set_vehicle_max_costs(data->vehicle_max_costs->data());
  }

  if (!vehicle_max_times.empty()) {
    data->vehicle_max_times = copy_vector(vehicle_max_times, stream);
    view.set_vehicle_max_times(data->vehicle_max_times->data());
  }

  if (!vehicle_fixed_costs.empty()) {
    data->vehicle_fixed_costs = copy_vector(vehicle_fixed_costs, stream);
    view.set_vehicle_fixed_costs(data->vehicle_fixed_costs->data());
  }

  if (!order_locations.empty()) {
    data->order_locations = copy_vector(order_locations, stream);
    view.set_order_locations(data->order_locations->data());
  }

  if (!order_tw_earliest.empty() && !order_tw_latest.empty()) {
    data->order_tw_earliest = copy_vector(order_tw_earliest, stream);
    data->order_tw_latest   = copy_vector(order_tw_latest, stream);
    view.set_order_time_windows(
      data->order_tw_earliest->data(), data->order_tw_latest->data(), false);
  }

  if (!order_prizes.empty()) {
    data->order_prizes = copy_vector(order_prizes, stream);
    view.set_order_prizes(data->order_prizes->data(), false);
  }

  for (auto const& [vehicle_id, times] : order_service_times) {
    auto d = copy_vector(times, stream);
    if (!d) { continue; }
    view.set_order_service_times(d->data(), vehicle_id, false);
    data->service_time_vehicle_ids.push_back(vehicle_id);
    data->service_times.push_back(std::move(d));
  }

  if (!pickup_indices.empty() && !delivery_indices.empty()) {
    data->pickup_indices   = copy_vector(pickup_indices, stream);
    data->delivery_indices = copy_vector(delivery_indices, stream);
    view.set_pickup_delivery_pairs(data->pickup_indices->data(), data->delivery_indices->data());
  }

  for (auto const& dim : capacity_dimensions) {
    // Validate client-provided sizes before they reach the device view:
    // populate_demand_container reads exactly `orders` demand and `fleet_size`
    // capacity values, so an undersized request would be an out-of-bounds read.
    if (static_cast<int32_t>(dim.demand.size()) != orders ||
        static_cast<int32_t>(dim.capacity.size()) != fleet_size) {
      throw std::invalid_argument(
        "cpu_routing_problem_t::to_device: capacity dimension '" + dim.name +
        "' must have num_orders demand values and fleet_size capacity values");
    }
    auto d_demand   = copy_vector(dim.demand, stream);
    auto d_capacity = copy_vector(dim.capacity, stream);
    if (!d_demand || !d_capacity) {
      throw std::invalid_argument(
        "cpu_routing_problem_t::to_device: incomplete capacity dimension");
    }
    view.add_capacity_dimension(dim.name, d_demand->data(), d_capacity->data(), false);
    data->capacity_names.push_back(dim.name);
    data->capacity_demands.push_back(std::move(d_demand));
    data->capacity_capacities.push_back(std::move(d_capacity));
  }

  if (!break_locations.empty()) {
    data->break_locations = copy_vector(break_locations, stream);
    view.set_break_locations(
      data->break_locations->data(), static_cast<int32_t>(data->break_locations->size()), false);
  }

  for (auto const& ub : uniform_breaks) {
    auto d_e = copy_vector(ub.earliest, stream);
    auto d_l = copy_vector(ub.latest, stream);
    auto d_d = copy_vector(ub.duration, stream);
    if (!d_e || !d_l || !d_d) {
      throw std::invalid_argument("cpu_routing_problem_t::to_device: incomplete uniform break");
    }
    view.add_break_dimension(d_e->data(), d_l->data(), d_d->data(), false);
    data->uniform_break_earliest.push_back(std::move(d_e));
    data->uniform_break_latest.push_back(std::move(d_l));
    data->uniform_break_duration.push_back(std::move(d_d));
  }

  for (auto const& [vehicle_id, breaks] : vehicle_breaks) {
    for (auto const& brk : breaks) {
      auto d_locs            = copy_vector(brk.locations, stream);
      int32_t n_locs         = d_locs ? static_cast<int32_t>(d_locs->size()) : 0;
      int32_t const* loc_ptr = d_locs ? d_locs->data() : nullptr;
      view.add_vehicle_break(
        vehicle_id, brk.earliest, brk.latest, brk.duration, loc_ptr, n_locs, false);
      if (d_locs) { data->vehicle_break_locations.push_back(std::move(d_locs)); }
    }
  }

  for (auto const& [vehicle_id, orders] : vehicle_order_match) {
    auto d = copy_vector(orders, stream);
    if (!d) { continue; }
    view.add_vehicle_order_match(vehicle_id, d->data(), static_cast<int32_t>(d->size()), false);
    data->match_buffers.push_back(std::move(d));
  }

  for (auto const& [order_id, vehicles] : order_vehicle_match) {
    auto d = copy_vector(vehicles, stream);
    if (!d) { continue; }
    view.add_order_vehicle_match(order_id, d->data(), static_cast<int32_t>(d->size()), false);
    data->match_buffers.push_back(std::move(d));
  }

  for (auto const& [order_id, preceding] : order_precedence) {
    auto d = copy_vector(preceding, stream);
    if (!d) { continue; }
    view.add_order_precedence(order_id, d->data(), static_cast<int32_t>(d->size()));
    data->precedence_buffers.push_back(std::move(d));
  }

  if (!objectives.empty() && !objective_weights.empty()) {
    // objective_t is an enum class; copy as int32 then reinterpret
    data->objectives        = copy_vector(objectives, stream);
    data->objective_weights = copy_vector(objective_weights, stream);
    view.set_objective_function(reinterpret_cast<objective_t const*>(data->objectives->data()),
                                data->objective_weights->data(),
                                static_cast<int32_t>(data->objectives->size()));
  }

  if (min_vehicles > 0) { view.set_min_vehicles(min_vehicles); }

  if (!initial_solutions.routes.empty()) {
    data->init_vehicle_ids = copy_vector(initial_solutions.vehicle_ids, stream);
    data->init_routes      = copy_vector(initial_solutions.routes, stream);
    data->init_sol_offsets = copy_vector(initial_solutions.sol_offsets, stream);

    std::vector<node_type_t> types;
    types.reserve(initial_solutions.types.size());
    for (auto t : initial_solutions.types) {
      types.push_back(static_cast<node_type_t>(t));
    }
    data->init_types = copy_vector(types, stream);
    // types is a local temporary feeding an async H2D copy; drain before it
    // goes out of scope.
    stream.sync();

    int32_t n_nodes = static_cast<int32_t>(initial_solutions.routes.size());
    int32_t n_sols  = static_cast<int32_t>(initial_solutions.sol_offsets.size());
    view.add_initial_solutions(data->init_vehicle_ids->data(),
                               data->init_routes->data(),
                               data->init_types->data(),
                               data->init_sol_offsets->data(),
                               n_nodes,
                               n_sols);
  }

  handle->sync_stream();
  return {std::move(view), std::move(data)};
}

}  // namespace routing
}  // namespace cuopt
