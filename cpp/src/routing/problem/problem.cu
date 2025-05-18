/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utilities/vector_helpers.cuh>
#include "../local_search/compute_compatible.cuh"
#include "problem.cuh"

#include <utilities/vector_helpers.cuh>

#include <utilities/seed_generator.cuh>
namespace cuopt {
namespace routing {
namespace detail {

// Possibly temporary class
// Try to merge this with data_model class or create a common problem representation class that
// can be used by both solvers
template <typename i_t, typename f_t>
problem_t<i_t, f_t>::problem_t(const data_model_view_t<i_t, f_t>& data_model_view_,
                               solver_settings_t<i_t, f_t> const& solver_settings_)
  : data_view_ptr(&data_model_view_),
    solver_settings_ptr(&solver_settings_),
    handle_ptr(data_model_view_.get_handle_ptr()),
    fleet_info(data_model_view_.get_handle_ptr(), 0),
    order_info(data_model_view_.get_handle_ptr(), 0),
    viables(data_model_view_),
    start_depot_node_infos(0, handle_ptr->get_stream()),
    return_depot_node_infos(0, handle_ptr->get_stream()),
    bucket_to_vehicle_id(0, handle_ptr->get_stream()),
    special_nodes(handle_ptr)
{
  populate_fleet_info(data_model_view_, fleet_info);
  populate_order_info(data_model_view_, order_info);

  populate_special_nodes();
  populate_demand_container(data_model_view_, fleet_info, order_info);
  populate_vehicle_order_match(
    data_model_view_, fleet_info.fleet_order_constraints_, fleet_info.is_homogenous_);
  populate_vehicle_infos(data_model_view_, fleet_info);
  // populate host vectors
  populate_host_arrays();
  populate_vehicle_buckets();

  initialize_depot_info();

  // FIXME: Implement to_host for containers
  auto n_locations = data_view_ptr->get_num_locations();
  pair_indices_h.resize(order_info.v_pair_indices_.size(), 0);
  raft::copy(pair_indices_h.data(),
             order_info.v_pair_indices_.data(),
             pair_indices_h.size(),
             handle_ptr->get_stream());

  vehicle_types_h = cuopt::host_copy(fleet_info.v_types_);
  for (auto& vtype : vehicle_types_h) {
    if (!distance_matrices_h.count(vtype)) {
      auto cost_matrix = fleet_info.matrices_.get_cost_matrix(vtype);
      auto cost_matrix_h =
        cuopt::host_copy(cost_matrix, n_locations * n_locations, handle_ptr->get_stream());
      distance_matrices_h.emplace(vtype, cost_matrix_h);
    }
  }
  handle_ptr->sync_stream();

  populate_dimensions_info();
  auto& problem_ref   = *this;
  auto pickup_indices = data_view_ptr->get_pickup_delivery_pair().first;
  bool is_pdp         = pickup_indices != nullptr;
  // as we pass nullptr by default, just pass a dummy request
  if (is_pdp) {
    initialize_incompatible<i_t, f_t, request_t::PDP>(problem_ref);
  } else {
    initialize_incompatible<i_t, f_t, request_t::VRP>(problem_ref);
  }

  seed_generator::set_seed(
    order_info.get_num_requests(), order_info.get_num_orders(), order_info.get_num_orders());
}

template <typename i_t, typename f_t>
VehicleInfo<f_t, false> problem_t<i_t, f_t>::get_vehicle_info(i_t vehicle_id) const
{
  return fleet_info_h.get_vehicle_info(vehicle_id);
}

template <typename i_t, typename f_t>
i_t problem_t<i_t, f_t>::get_num_buckets() const
{
  return vehicle_buckets_h.size();
}

template <typename i_t, typename f_t>
std::vector<std::vector<i_t>> problem_t<i_t, f_t>::get_vehicle_buckets() const
{
  return vehicle_buckets_h;
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::populate_vehicle_buckets()
{
  auto fleet_size = data_view_ptr->get_fleet_size();
  vehicle_buckets_h.resize(fleet_size);
  fleet_info_h = fleet_info.to_host();

  // infer vehicle types from data model
  for (int vehicle_id = 0; vehicle_id < fleet_size; ++vehicle_id) {
    auto curr_vehicle_info = fleet_info_h.get_vehicle_info(vehicle_id);
    bool id_inserted       = false;
    // Compare curr vehicle id with each bucket and add to the right one
    for (size_t bucket = 0; bucket < vehicle_buckets_h.size() && !id_inserted; ++bucket) {
      auto const& bucket_vehicle_ids = vehicle_buckets_h[bucket];
      if (!bucket_vehicle_ids.empty()) {
        auto bucket_vehicle_id   = bucket_vehicle_ids[0];
        auto bucket_vehicle_info = fleet_info_h.get_vehicle_info(bucket_vehicle_id);
        if (curr_vehicle_info == bucket_vehicle_info) {
          vehicle_buckets_h[bucket].push_back(vehicle_id);
          id_inserted                      = true;
          fleet_info_h.buckets[vehicle_id] = bucket;
        }
      } else {
        // Found a new type
        vehicle_buckets_h[bucket].push_back(vehicle_id);
        id_inserted                      = true;
        fleet_info_h.buckets[vehicle_id] = bucket;
      }
    }
  }

  // Populate device buckets
  raft::copy(fleet_info.v_buckets_.data(),
             fleet_info_h.buckets.data(),
             fleet_info_h.buckets.size(),
             handle_ptr->get_stream());

  auto n_vehicle_types = 0;
  cuopt_func_call(int counter = 0);
  for (int i = 0; i < fleet_size; ++i) {
    if (vehicle_buckets_h[i].empty()) { break; }
    cuopt_func_call(counter += vehicle_buckets_h[i].size());
    ++n_vehicle_types;
  }
  vehicle_buckets_h.resize(n_vehicle_types);

  cuopt_assert(counter == fleet_size, "Corrupted vehicle buckets");
  if (fleet_info.is_homogenous_) {
    cuopt_assert(n_vehicle_types == 1, "Corrupted vehicle buckets");
  }

  fleet_info_h.vehicle_availability.resize(n_vehicle_types);
  bucket_to_vehicle_id_h.resize(n_vehicle_types);
  for (size_t i = 0; i < vehicle_buckets_h.size(); ++i) {
    fleet_info_h.vehicle_availability[i] = vehicle_buckets_h[i].size();
    bucket_to_vehicle_id_h[i]            = vehicle_buckets_h[i][0];
  }

  device_copy(fleet_info.v_vehicle_availability_,
              fleet_info_h.vehicle_availability,
              handle_ptr->get_stream());
  device_copy(bucket_to_vehicle_id, bucket_to_vehicle_id_h, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
std::vector<i_t> problem_t<i_t, f_t>::get_preferred_order_of_vehicles() const
{
  auto fleet_size = data_view_ptr->get_fleet_size();
  auto num_orders = data_view_ptr->get_num_orders();

  auto& capacity_vec  = fleet_info.v_capacities_;
  int n_capacity_dims = capacity_vec.size() / fleet_size;

  int n_vehicle_types = vehicle_buckets_h.size();
  // Sort vehicle ids based on either priority (if it's defined) or based on cost effectivness
  std::vector<i_t> sorted_vehicle_ids(fleet_size);
  std::iota(sorted_vehicle_ids.begin(), sorted_vehicle_ids.end(), 0);

  std::vector<double> demands(n_capacity_dims, 0.);
  const auto& order_demands_h = order_info_h.demand;
  for (int d = 0; d < n_capacity_dims; ++d) {
    for (int i = 0; i < num_orders; ++i) {
      demands[d] += order_demands_h[d * num_orders + i];
    }
  }

  std::vector<double> cost_effectiveness(n_vehicle_types, 0.);
  // For each bucket calculate cost effectiveness
  for (size_t bucket = 0; bucket < vehicle_buckets_h.size(); ++bucket) {
    i_t vehicle_id                         = vehicle_buckets_h[bucket][0];
    auto curr_vehicle_info                 = fleet_info_h.get_vehicle_info(vehicle_id);
    size_t num_vehicles_needed_from_bucket = 1;
    for (int d = 0; d < n_capacity_dims; ++d) {
      // Note that the demand would be zero for PDP use case, so the num vehicles needed is
      // exactly 1. So we purely make the decision based on vehicle cost for PDP
      size_t tmp                      = ceil(demands[d] / curr_vehicle_info.capacities[d]);
      num_vehicles_needed_from_bucket = std::max(num_vehicles_needed_from_bucket, tmp);
    }

    double cost_of_vehicle = 0.;
    if (!data_view_ptr->get_vehicle_fixed_costs().empty() && curr_vehicle_info.fixed_cost > 1e-6) {
      cost_of_vehicle = curr_vehicle_info.fixed_cost;
    } else {
      // use average cost
      cost_of_vehicle = curr_vehicle_info.get_average_cost();
    }

    cost_effectiveness[bucket] = 1.;
    if (cost_of_vehicle > 1e-10) {
      cuopt_assert(num_vehicles_needed_from_bucket > 0, "Num vehicles needed should be positive!");
      cost_effectiveness[bucket] = 1. / (num_vehicles_needed_from_bucket * cost_of_vehicle);
    }
  }

  std::sort(sorted_vehicle_ids.begin(), sorted_vehicle_ids.end(), [&](auto& a, auto& b) {
    auto bucket_a = fleet_info_h.buckets[a];
    auto bucket_b = fleet_info_h.buckets[b];
    return cost_effectiveness[bucket_a] > cost_effectiveness[bucket_b];
  });

  return sorted_vehicle_ids;
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::populate_dimensions_info()
{
  auto [obj_types_d, obj_weights_d, n_obj] = data_view_ptr->get_objective_function();
  auto obj_types_h   = cuopt::host_copy(obj_types_d, n_obj, handle_ptr->get_stream());
  auto obj_weights_h = cuopt::host_copy(obj_weights_d, n_obj, handle_ptr->get_stream());

  std::map<objective_t, double> specified_weights;
  for (size_t i = 0; i < obj_types_h.size(); ++i) {
    specified_weights.insert({obj_types_h[i], obj_weights_h[i]});
    if (obj_weights_h[i] > 0.0) {
      dimensions_info.enable_objective(obj_types_h[i], obj_weights_h[i]);
    }
  }

  // DIST dimension info
  double cost_obj_weight =
    specified_weights.count(objective_t::COST) ? specified_weights.at(objective_t::COST) : 1.0;
  dimensions_info.enable_dimension(dim_t::DIST);
  dimensions_info.enable_objective(objective_t::COST, cost_obj_weight);

  auto& cost_dim_info = dimensions_info.distance_dim;
  if (auto vehicle_max_costs = data_view_ptr->get_vehicle_max_costs(); !vehicle_max_costs.empty()) {
    cost_dim_info.has_max_constraint = true;
  }

  // TIME dimensions info
  // check vehicle max times exists
  auto vehicle_max_times        = data_view_ptr->get_vehicle_max_times();
  bool vehicle_max_times_exists = !vehicle_max_times.empty();

  // check vehicle tw exists
  bool vehicle_tw_exists = data_view_ptr->get_vehicle_time_windows().first != nullptr;

  // check if the trave time obj exists
  bool travel_time_obj_exists = dimensions_info.has_objective(objective_t::TRAVEL_TIME);

  bool order_tw_exists = std::get<0>(data_view_ptr->get_order_time_windows()) != nullptr;

  bool time_matrix_exists = data_view_ptr->get_transit_time_matrices().size() > 0;

  bool enable_time_dim = vehicle_max_times_exists || vehicle_tw_exists || travel_time_obj_exists ||
                         order_tw_exists || time_matrix_exists;

  if (enable_time_dim) {
    dimensions_info.enable_dimension(dim_t::TIME);
    auto& time_dim_info = dimensions_info.time_dim;
    if (auto vehicle_max_times = data_view_ptr->get_vehicle_max_times();
        !vehicle_max_times.empty()) {
      time_dim_info.has_max_constraint = true;
    }

    if (travel_time_obj_exists) { time_dim_info.has_travel_time_obj = true; }
  }

  // CAP dimensions info
  auto& cap_dim_info                 = dimensions_info.capacity_dim;
  auto num_vehicles                  = data_view_ptr->get_fleet_size();
  auto num_orders                    = data_view_ptr->get_num_orders();
  auto& capacity_vec                 = fleet_info.v_capacities_;
  int n_capacity_dims                = capacity_vec.size() / num_vehicles;
  cap_dim_info.n_capacity_dimensions = n_capacity_dims;
  if (data_view_ptr->get_capacity_dimensions().size() > 0) {
    dimensions_info.enable_dimension(dim_t::CAP);
  }

  // PRIZE dimension info
  if (auto prizes = data_view_ptr->get_order_prizes(); !prizes.empty()) {
    double obj_weight =
      specified_weights.count(objective_t::PRIZE) ? specified_weights.at(objective_t::PRIZE) : 1.0;
    // If the objective weight is set to zero, we disable prize collection
    if (obj_weight > 0.0) {
      dimensions_info.enable_dimension(dim_t::PRIZE);
      dimensions_info.enable_objective(objective_t::PRIZE, obj_weight);
    }
  }

  // Route size dimension info
  if (specified_weights.count(objective_t::VARIANCE_ROUTE_SIZE)) {
    dimensions_info.enable_dimension(dim_t::TASKS);
    auto& tasks_dim_info = dimensions_info.tasks_dim;
    tasks_dim_info.mean_tasks =
      ((double)order_info.get_num_depot_excluded_orders()) / (double)num_vehicles;
  }

  // service time dimension info
  if (specified_weights.count(objective_t::VARIANCE_ROUTE_SERVICE_TIME)) {
    dimensions_info.enable_dimension(dim_t::SERVICE_TIME);
    auto& service_time_dim_info = dimensions_info.service_time_dim;

    auto& service_times = fleet_info.fleet_order_constraints_.order_service_times;
    double sum          = thrust::reduce(
      rmm::exec_policy(handle_ptr->get_stream()), service_times.begin(), service_times.end());

    // estimate average service time per route
    service_time_dim_info.mean_service_time =
      (sum / service_times.size()) * ((double)num_orders / (double)num_vehicles);
  }

  // mismatch dimension info
  const auto& vehicle_order_match = data_view_ptr->get_vehicle_order_match();
  const auto& order_vehicle_match = data_view_ptr->get_order_vehicle_match();
  const bool vehicle_order_match_exists =
    !vehicle_order_match.empty() || !order_vehicle_match.empty();
  if (vehicle_order_match_exists) {
    dimensions_info.enable_dimension(dim_t::MISMATCH);
    auto& mismatch_dim_info                   = dimensions_info.mismatch_dim;
    mismatch_dim_info.has_vehicle_order_match = true;
  }

  // break dimension info
  if (data_view_ptr->has_vehicle_breaks()) {
    dimensions_info.enable_dimension(dim_t::BREAK);
    auto& break_dim_info      = dimensions_info.break_dim;
    break_dim_info.has_breaks = true;
  }

  // vehicle cost dimension info
  const auto& vehicle_fixed_cost       = data_view_ptr->get_vehicle_fixed_costs();
  const bool vehicle_fixed_cost_exists = !vehicle_fixed_cost.empty();
  if (vehicle_fixed_cost_exists) {
    dimensions_info.enable_dimension(dim_t::VEHICLE_FIXED_COST);
    if (!specified_weights.count(objective_t::VEHICLE_FIXED_COST)) {
      dimensions_info.enable_objective(objective_t::VEHICLE_FIXED_COST, 1.0);
    }
  }

  if (data_view_ptr->get_fleet_size() == 1) {
    is_tsp = true;
    loop_over_dimensions(dimensions_info, [&](auto I) {
      if constexpr (I != (size_t)dim_t::DIST) { is_tsp = false; }
    });
  }
  dimensions_info.is_tsp = is_tsp;

  if (!is_tsp) {
    is_cvrp_ = !is_pdp() && (data_view_ptr->get_cost_matrices().size() == 1);
    if (is_cvrp_) {
      loop_over_dimensions(dimensions_info, [&](auto I) {
        if (I != (int)dim_t::DIST && I != (int)dim_t::CAP) { is_cvrp_ = false; }
      });
    }
    is_cvrp_ = is_cvrp_ && n_capacity_dims == 1;
  }
}

// This is temporary
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::populate_host_arrays()
{
  auto pickup_indices = data_view_ptr->get_pickup_delivery_pair().first;
  auto stream         = data_view_ptr->get_handle_ptr()->get_stream();

  order_locations_h = cuopt::host_copy(order_info.v_order_locations_);
  // Temporarily fill is_pickup_h for diversity, should use NodeInfo instead
  bool is_pdp = pickup_indices != nullptr;
  std::vector<i_t> h_pickup_indices(get_num_requests());
  is_pickup_h.assign(get_num_orders(), false);
  if (is_pdp) {
    raft::copy(h_pickup_indices.data(), pickup_indices, get_num_requests(), stream);
    for (const auto pickup_id : h_pickup_indices) {
      is_pickup_h[pickup_id] = true;
    }
  }

  drop_return_trip_h = cuopt::host_copy(fleet_info.v_drop_return_trip_);
  skip_first_trip_h  = cuopt::host_copy(fleet_info.v_skip_first_trip_);
  order_info_h       = order_info.to_host();
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::initialize_depot_info()
{
  int nvehicles                 = fleet_info.v_start_locations_.size();
  auto vehicle_start_locations  = cuopt::host_copy(fleet_info.v_start_locations_);
  auto vehicle_return_locations = cuopt::host_copy(fleet_info.v_return_locations_);

  start_depot_node_infos_h.resize(nvehicles);
  return_depot_node_infos_h.resize(nvehicles);

  // In standard case when order locations are not specified, we assume depot is at zero
  // otherwise, depot is not included in the orders so we use norders as node id
  // so that it does not conflict with zeroth order
  const int depot_node_id = order_locations_h.empty() ? 0 : order_locations_h.size();
  bool is_single_depot    = true;
  for (size_t v = 0; v < vehicle_start_locations.size(); ++v) {
    start_depot_node_infos_h[v] =
      NodeInfo<>{depot_node_id, vehicle_start_locations[v], node_type_t::DEPOT};
    return_depot_node_infos_h[v] =
      NodeInfo<>{depot_node_id, vehicle_return_locations[v], node_type_t::DEPOT};

    if (vehicle_start_locations[v] != vehicle_return_locations[v]) { is_single_depot = false; }

    if (v > 0 && (vehicle_start_locations[v] != vehicle_start_locations[v - 1] ||
                  vehicle_return_locations[v] != vehicle_return_locations[v - 1])) {
      is_single_depot = false;
    }
  }

  if (is_single_depot) {
    single_depot_node = NodeInfo<>(depot_node_id, vehicle_start_locations[0], node_type_t::DEPOT);
  }

  start_depot_node_infos  = cuopt::device_copy(start_depot_node_infos_h, handle_ptr->get_stream());
  return_depot_node_infos = cuopt::device_copy(return_depot_node_infos_h, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::is_pickup(i_t node_id) const
{
  if (!order_info.is_pdp()) { return false; }
  return is_pickup_h[node_id];
}

// FIXME:: This is not scalable as we add more features. We should be able to use the method
// that we use in kernels
template <typename i_t, typename f_t>
double problem_t<i_t, f_t>::distance_between(const NodeInfo<>& node_1,
                                             const NodeInfo<>& node_2,
                                             const int& vehicle_id) const
{
  auto n_locations = data_view_ptr->get_num_locations();
  cuopt_assert(vehicle_id < (int)vehicle_types_h.size(), "vehicle id should be in range!");
  i_t vehicle_type = vehicle_types_h[vehicle_id];
  cuopt_assert(distance_matrices_h.count(vehicle_type), "vehicle type does not exist!");

  if (node_1.is_depot() && skip_first_trip_h[vehicle_id]) {
    return 0.;
  } else if (node_2.is_depot() && drop_return_trip_h[vehicle_id]) {
    return 0.;
  }

  return distance_matrices_h.at(vehicle_type)[node_1.location() * n_locations + node_2.location()];
}

template <typename i_t, typename f_t>
i_t problem_t<i_t, f_t>::get_num_orders() const
{
  return order_info.get_num_orders();
}

template <typename i_t, typename f_t>
i_t problem_t<i_t, f_t>::get_num_requests() const
{
  return order_info.get_num_requests();
}

template <typename i_t, typename f_t>
detail::NodeInfo<> problem_t<i_t, f_t>::get_node_info_of_node(const int node) const
{
  if (order_info.depot_included_) {  // i..e order locations is not used
    if (node == 0) {
      return NodeInfo<>(node, node, node_type_t::DEPOT);
    } else {
      bool is_pickup = is_pickup_h[node];
      return NodeInfo<>(node, node, is_pickup ? node_type_t::PICKUP : node_type_t::DELIVERY);
    }
  } else {
    assert(node < (int)order_locations_h.size());
    bool is_pickup = is_pickup_h[node];
    int location   = order_locations_h[node];
    return NodeInfo<>(node, location, is_pickup ? node_type_t::PICKUP : node_type_t::DELIVERY);
  }
}

template <typename i_t, typename f_t>
std::optional<NodeInfo<>> problem_t<i_t, f_t>::get_single_depot() const
{
  return single_depot_node;
}

template <typename i_t, typename f_t>
NodeInfo<> problem_t<i_t, f_t>::get_start_depot_node_info(const i_t vehicle_id) const
{
  return start_depot_node_infos_h[vehicle_id];
}

template <typename i_t, typename f_t>
NodeInfo<> problem_t<i_t, f_t>::get_brother_node_info(const NodeInfo<>& node) const
{
  if (!order_info.is_pdp()) { return get_node_info_of_node(node.node()); }

  int brother_id       = pair_indices_h[node.node()];
  int brother_location = order_locations_h.empty() ? brother_id : order_locations_h[brother_id];
  return NodeInfo<>(
    brother_id, brother_location, node.is_pickup() ? node_type_t::DELIVERY : node_type_t::PICKUP);
}

template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::populate_special_nodes()
{
  if (!data_view_ptr->has_vehicle_breaks()) { return; }

  int n_vehicles = get_fleet_size();

  auto vehicle_earliest_h = cuopt::host_copy(fleet_info.v_earliest_time_);
  auto vehicle_latest_h   = cuopt::host_copy(fleet_info.v_latest_time_);
  std::map<int, std::vector<int>> break_earliest_h, break_latest_h, break_duration_h;
  std::vector<int> break_offset_h(n_vehicles + 1, 0), break_nodes_offset_h;

  int n_max_break_dims = 0;
  auto& uniform_breaks = data_view_ptr->get_uniform_breaks();

  std::vector<NodeInfo<>> node_infos_h;
  std::vector<i_t> node_earliest_h, node_latest_h;
  std::vector<i_t> break_loc_to_idx_h;

  if (!uniform_breaks.empty()) {
    int n_break_dims = n_max_break_dims = uniform_breaks.size();
    for (i_t dim = 0; dim < n_break_dims; ++dim) {
      auto [break_earliest, break_latest, break_duration] = uniform_breaks[dim].get_breaks();
      // Check if break values are uniform across all vehicles
      bool uniform_earliest = all_entries_are_equal(handle_ptr, break_earliest, n_vehicles);
      bool uniform_latest   = all_entries_are_equal(handle_ptr, break_latest, n_vehicles);
      bool uniform_duration = all_entries_are_equal(handle_ptr, break_duration, n_vehicles);

      // If any of the break parameters are not uniform across vehicles, mark as non-uniform
      if (!uniform_earliest || !uniform_latest || !uniform_duration) { non_uniform_breaks_ = true; }

      auto this_dim_break_earliest =
        cuopt::host_copy(break_earliest, n_vehicles, handle_ptr->get_stream());
      auto this_dim_break_latest =
        cuopt::host_copy(break_latest, n_vehicles, handle_ptr->get_stream());
      auto this_dim_break_duration =
        cuopt::host_copy(break_duration, n_vehicles, handle_ptr->get_stream());

      for (int v = 0; v < n_vehicles; ++v) {
        break_earliest_h[v].push_back(this_dim_break_earliest[v]);
        break_latest_h[v].push_back(this_dim_break_latest[v]);
        break_duration_h[v].push_back(this_dim_break_duration[v]);

        bool expected =
          (break_earliest_h[v][dim] + break_duration_h[v][dim] <= vehicle_latest_h[v]) &&
          (vehicle_earliest_h[v] <= break_latest_h[v][dim]);
        cuopt_expects(expected,
                      error_type_t::ValidationError,
                      "break times should be within the range of vehicle time windows!");
      }

      fleet_info.is_homogenous_ = fleet_info.is_homogenous_ &&
                                  all_entries_are_equal(handle_ptr, break_earliest, n_vehicles) &&
                                  all_entries_are_equal(handle_ptr, break_latest, n_vehicles) &&
                                  all_entries_are_equal(handle_ptr, break_duration, n_vehicles);
    }

    std::vector<i_t> break_locations_h;

    // If break locations are specified use them, or use order locations if they are specified,
    // other wise use all locations
    if (auto [break_locs, n_locs] = data_view_ptr->get_break_locations(); break_locs != nullptr) {
      break_locations_h = cuopt::host_copy(break_locs, n_locs, handle_ptr->get_stream());
    } else {
      i_t num_locs = data_view_ptr->get_num_locations();
      break_locations_h.resize(num_locs);
      std::iota(break_locations_h.begin(), break_locations_h.end(), 0);
    }

    int n_break_locations = break_locations_h.size();
    int n_total_breaks    = n_vehicles * n_break_dims;
    int n_break_nodes     = n_vehicles * n_break_locations * n_break_dims;

    break_nodes_offset_h.resize(n_total_breaks + 1, 0);
    node_infos_h.resize(n_break_nodes);
    node_earliest_h.resize(n_break_nodes);
    node_latest_h.resize(n_break_nodes);

    for (i_t v = 0; v < n_vehicles; ++v) {
      break_offset_h[v + 1] = break_offset_h[v] + n_break_dims;
      for (i_t dim = 0; dim < n_break_dims; ++dim) {
        break_nodes_offset_h[v * n_break_dims + dim + 1] =
          break_nodes_offset_h[v * n_break_dims + dim] + n_break_locations;
        for (i_t l = 0; l < n_break_locations; ++l) {
          i_t special_node_id = v * n_break_dims * n_break_locations + dim * n_break_locations + l;
          // set the node id to be dimension
          i_t node_id = dim;
          node_infos_h[special_node_id] =
            NodeInfo<>{node_id, break_locations_h[l], node_type_t::BREAK};
          node_earliest_h[special_node_id] = break_earliest_h[v][dim];
          node_latest_h[special_node_id]   = break_latest_h[v][dim];
        }
      }
    }

    break_loc_to_idx_h.resize(
      *std::max_element(break_locations_h.begin(), break_locations_h.end()) + 1);
    for (i_t i = 0; i < n_break_locations; ++i) {
      auto loc                = break_locations_h[i];
      break_loc_to_idx_h[loc] = i;
    }
  } else {
    non_uniform_breaks_ = true;
    // FIXME:: This is temporary, we should just use VehicleInfo::operator ==
    // after we populate everything in fleet_info
    fleet_info.is_homogenous_ = false;
    auto& non_uniform_breaks  = data_view_ptr->get_non_uniform_breaks();
    int offset                = 0;
    // assume 2 breaks per vehicle for reserving memory
    break_nodes_offset_h.reserve(2 * n_vehicles);
    node_infos_h.reserve(2 * n_vehicles);
    node_earliest_h.reserve(2 * n_vehicles);
    node_latest_h.reserve(2 * n_vehicles);

    break_nodes_offset_h.push_back(0);

    std::vector<int> all_locations(data_view_ptr->get_num_locations());
    std::iota(all_locations.begin(), all_locations.end(), 0);
    for (i_t v = 0; v < n_vehicles; ++v) {
      if (non_uniform_breaks.count(v)) {
        auto& this_vehicle_breaks = non_uniform_breaks.at(v);
        break_offset_h[v + 1]     = break_offset_h[v] + this_vehicle_breaks.size();
        n_max_break_dims          = std::max((i_t)this_vehicle_breaks.size(), n_max_break_dims);
        // FIXME:: sort the breaks based on TW ??
        for (auto& vehicle_break : this_vehicle_breaks) {
          i_t dim = break_duration_h[v].size();
          break_duration_h[v].push_back(vehicle_break.duration_);
          break_earliest_h[v].push_back(vehicle_break.earliest_);
          break_latest_h[v].push_back(vehicle_break.latest_);

          bool expected =
            (break_earliest_h[v][dim] + break_duration_h[v][dim] <= vehicle_latest_h[v]) &&
            (vehicle_earliest_h[v] <= break_latest_h[v][dim]);
          cuopt_expects(expected,
                        error_type_t::ValidationError,
                        "break times should be within the range of vehicle time windows!");

          expected = break_latest_h[v][dim] >= break_earliest_h[v][dim];
          cuopt_expects(expected,
                        error_type_t::ValidationError,
                        "break latest should be higher than the break earliest!");
          if (dim > 0) {
            expected = break_earliest_h[v][dim] >= break_latest_h[v][dim - 1];
            cuopt_expects(
              expected, error_type_t::ValidationError, "breaks should not be overlapping!");
          }

          auto this_break_locations =
            cuopt::host_copy(vehicle_break.locations_, handle_ptr->get_stream());

          auto& use_break_locations =
            this_break_locations.empty() ? all_locations : this_break_locations;

          offset += use_break_locations.size();
          const i_t node_id = dim;
          for (auto& loc : use_break_locations) {
            node_infos_h.push_back(NodeInfo<>{node_id, loc, node_type_t::BREAK});
            node_earliest_h.push_back(break_earliest_h[v][dim]);
            node_latest_h.push_back(break_latest_h[v][dim]);
          }

          break_nodes_offset_h.push_back(offset);
        }
      } else {
        break_offset_h[v + 1] = break_offset_h[v];
      }
    }
  }

  fleet_info.v_break_offset_ = cuopt::device_copy(break_offset_h, handle_ptr->get_stream());
  fleet_info.v_break_duration_.resize(break_offset_h[n_vehicles], handle_ptr->get_stream());
  fleet_info.v_break_earliest_.resize(break_offset_h[n_vehicles], handle_ptr->get_stream());
  fleet_info.v_break_latest_.resize(break_offset_h[n_vehicles], handle_ptr->get_stream());

  for (int v = 0; v < n_vehicles; ++v) {
    int offset     = break_offset_h[v];
    int num_breaks = break_offset_h[v + 1] - offset;
    raft::copy(fleet_info.v_break_duration_.data() + offset,
               break_duration_h[v].data(),
               num_breaks,
               handle_ptr->get_stream());
    raft::copy(fleet_info.v_break_earliest_.data() + offset,
               break_earliest_h[v].data(),
               num_breaks,
               handle_ptr->get_stream());
    raft::copy(fleet_info.v_break_latest_.data() + offset,
               break_latest_h[v].data(),
               num_breaks,
               handle_ptr->get_stream());
  }

  special_nodes.num_vehicles             = n_vehicles;
  special_nodes.num_max_break_dimensions = n_max_break_dims;
  // special_nodes.nodes_per_dimension_per_vehicle = n_break_locations;
  special_nodes.num_breaks_offset = cuopt::device_copy(break_offset_h, handle_ptr->get_stream());
  special_nodes.break_nodes_offset =
    cuopt::device_copy(break_nodes_offset_h, handle_ptr->get_stream());
  special_nodes.node_infos       = cuopt::device_copy(node_infos_h, handle_ptr->get_stream());
  special_nodes.earliest_time    = cuopt::device_copy(node_earliest_h, handle_ptr->get_stream());
  special_nodes.latest_time      = cuopt::device_copy(node_latest_h, handle_ptr->get_stream());
  special_nodes.break_loc_to_idx = cuopt::device_copy(break_loc_to_idx_h, handle_ptr->get_stream());
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
i_t problem_t<i_t, f_t>::get_fleet_size() const
{
  return data_view_ptr->get_fleet_size();
}

template <typename i_t, typename f_t>
i_t problem_t<i_t, f_t>::get_max_break_dimensions() const
{
  return special_nodes.num_max_break_dimensions;
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::has_vehicle_breaks() const
{
  return data_view_ptr->has_vehicle_breaks();
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::has_non_uniform_breaks() const
{
  return non_uniform_breaks_;
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::has_prize_collection() const
{
  return dimensions_info.has_dimension(dim_t::PRIZE);
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::has_vehicle_fixed_costs() const
{
  return dimensions_info.has_dimension(dim_t::VEHICLE_FIXED_COST);
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::is_pdp() const
{
  return order_info.is_pdp();
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::is_cvrp_intra() const
{
  return !is_pdp() && !dimensions_info.has_dimension(dim_t::TIME) &&
         !dimensions_info.has_dimension(dim_t::BREAK);
}

template <typename i_t, typename f_t>
bool problem_t<i_t, f_t>::is_cvrp() const
{
  return is_cvrp_;
}

template class problem_t<int, float>;

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
