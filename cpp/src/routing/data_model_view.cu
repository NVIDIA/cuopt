/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/data_model_view.hpp>
#include <limits>
#include <ostream>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>
#include <routing/structures.hpp>
#include <routing/utilities/check_input.hpp>
#include <unordered_set>

#include <thrust/unique.h>

namespace cuopt {
namespace routing {

template <typename i_t, typename f_t>
data_model_view_t<i_t, f_t>::data_model_view_t(raft::handle_t const* handle_ptr,
                                               i_t num_locations,
                                               i_t const fleet_size,
                                               i_t num_orders)
  : handle_ptr_(handle_ptr),
    num_locations_(num_locations),
    fleet_size_(fleet_size),
    num_orders_(num_orders)
{
  cuopt_expects(
    num_locations > 0, error_type_t::ValidationError, "The data model needs at least one location");
  cuopt_expects(fleet_size > 0, error_type_t::ValidationError, "The fleet size must be positive");

  // If the number of orders is not specified, we can assume that there is one order per
  // location and one for the depot as well
  if (num_orders_ < 0) { num_orders_ = num_locations_; }
  cuopt_expects(num_orders_ < std::numeric_limits<uint16_t>::max(),
                error_type_t::ValidationError,
                "Number of nodes should be lower than 65535");
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_cost_matrix(f_t const* matrix, uint8_t vehicle_type)
{
  cuopt_expects(matrix != nullptr, error_type_t::ValidationError, "Matrix input cannot be null");
  cost_matrices_[vehicle_type] = matrix;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_transit_time_matrix(f_t const* matrix, uint8_t vehicle_type)
{
  cuopt_expects(matrix != nullptr, error_type_t::ValidationError, "Matrix input cannot be null");
  transit_time_matrices_[vehicle_type] = matrix;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_initial_solutions(i_t const* vehicle_ids,
                                                        i_t const* routes,
                                                        node_type_t const* types,
                                                        i_t const* sol_offsets,
                                                        i_t n_nodes,
                                                        i_t n_solutions)

{
  initial_vehicle_ids_ = raft::device_span<i_t const>(vehicle_ids, n_nodes);
  initial_routes_      = raft::device_span<i_t const>(routes, n_nodes);
  initial_types_       = raft::device_span<node_type_t const>(types, n_nodes);
  initial_sol_offsets_ = raft::device_span<i_t const>(sol_offsets, n_solutions);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_break_locations(i_t const* break_locations,
                                                      i_t n_break_locations,
                                                      bool validate_input)
{
  cuopt_expects(n_break_locations > 0,
                error_type_t::ValidationError,
                "Number of break locations should be positive.");
  cuopt_expects(
    break_locations != nullptr, error_type_t::ValidationError, "Break locations cannot be null");
  if (validate_input) {
    cuopt_expects(
      detail::check_min_max_values(
        break_locations, n_break_locations, 0, num_locations_ - 1, handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Break locations should be at the end of the matrix");
    rmm::device_uvector<i_t> tmp_break_nodes(n_break_locations, handle_ptr_->get_stream());
    raft::copy(
      tmp_break_nodes.begin(), break_locations, n_break_locations, handle_ptr_->get_stream());
    auto end = thrust::unique(
      handle_ptr_->get_thrust_policy(), tmp_break_nodes.begin(), tmp_break_nodes.end());
    i_t unique_items = end - tmp_break_nodes.begin();
    cuopt_expects(n_break_locations == unique_items,
                  error_type_t::ValidationError,
                  "There should be unique break locations");
  }

  break_locations_   = break_locations;
  n_break_locations_ = n_break_locations;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_order_locations(i_t const* order_locations)
{
  cuopt_expects(
    order_locations != nullptr, error_type_t::ValidationError, "Order locations cannot be null");
  order_locations_ = order_locations;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_break_dimension(i_t const* break_earliest,
                                                      i_t const* break_latest,
                                                      i_t const* break_duration,
                                                      bool validate_input)
{
  cuopt_expects(break_earliest != nullptr && break_latest != nullptr && break_duration != nullptr,
                error_type_t::ValidationError,
                "Breaks cannot be null");
  if (validate_input) {
    cuopt_expects(detail::check_min_max_values(break_earliest,
                                               get_fleet_size(),
                                               std::numeric_limits<int32_t>::min(),
                                               std::numeric_limits<int32_t>::max(),
                                               handle_ptr_->get_stream()),
                  error_type_t::ValidationError,
                  "break values must be between min int32_t and max int32_t");

    cuopt_expects(detail::check_min_max_values(break_latest,
                                               get_fleet_size(),
                                               std::numeric_limits<int32_t>::min(),
                                               std::numeric_limits<int32_t>::max(),
                                               handle_ptr_->get_stream()),
                  error_type_t::ValidationError,
                  "break values must be between min int32_t and max int32_t");

    cuopt_expects(detail::check_min_max_values(break_duration,
                                               get_fleet_size(),
                                               std::numeric_limits<int32_t>::min(),
                                               std::numeric_limits<int32_t>::max(),
                                               handle_ptr_->get_stream()),
                  error_type_t::ValidationError,
                  "break_duration values must be between min int32_t and max int32_t");
  }
  break_dimensions_.emplace_back(break_earliest, break_latest, break_duration);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_vehicle_break(i_t vehicle_id,
                                                    i_t break_earliest,
                                                    i_t break_latest,
                                                    i_t break_duration,
                                                    i_t const* break_locations,
                                                    i_t num_break_locations,
                                                    bool validate_input)
{
  vehicle_breaks_[vehicle_id].push_back(detail::vehicle_break_t<i_t>(
    break_earliest,
    break_latest,
    break_duration,
    raft::device_span<const i_t>(break_locations, num_break_locations)));

  if (validate_input && num_break_locations > 0) {
    cuopt_expects(
      detail::check_min_max_values(
        break_locations, num_break_locations, 0, num_locations_ - 1, handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Break locations should be at the end of the matrix");
    rmm::device_uvector<i_t> tmp_break_nodes(num_break_locations, handle_ptr_->get_stream());
    raft::copy(
      tmp_break_nodes.begin(), break_locations, num_break_locations, handle_ptr_->get_stream());
    auto end = thrust::unique(
      handle_ptr_->get_thrust_policy(), tmp_break_nodes.begin(), tmp_break_nodes.end());
    i_t unique_items = end - tmp_break_nodes.begin();
    cuopt_expects(num_break_locations == unique_items,
                  error_type_t::ValidationError,
                  "There should be unique break locations");
  }
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_objective_function(objective_t const* objective,
                                                         f_t const* objective_weights,
                                                         i_t n_objectives)
{
  cuopt_expects(objective != nullptr, error_type_t::ValidationError, "objective cannot be null");
  cuopt_expects(objective_weights != nullptr,
                error_type_t::ValidationError,
                "objective weights cannot be null");
  cuopt_expects(n_objectives > 0,
                error_type_t::ValidationError,
                "Number of objectives should be strictly positive");
  objective_         = objective;
  objective_weights_ = objective_weights;
  n_objectives_      = n_objectives;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_vehicle_time_windows(i_t const* earliest,
                                                           i_t const* latest,
                                                           bool validate_input)
{
  cuopt_expects(earliest != nullptr && latest != nullptr,
                error_type_t::ValidationError,
                "Vehicle time windows cannot be null");
  if (validate_input) {
    cuopt_expects(
      detail::check_min_max_values(
        earliest, fleet_size_, 0, std::numeric_limits<i_t>::max(), handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Earliest time must be between min 0 and max i_t!");
    cuopt_expects(
      detail::check_min_max_values(
        latest, fleet_size_, 0, std::numeric_limits<i_t>::max(), handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Latest time must be between min 0 and max i_t!");
  }
  vehicle_tw_ = std::move(detail::vehicle_time_window_t<i_t, f_t>(earliest, latest));
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_vehicle_locations(i_t const* start_locations,
                                                        i_t const* return_locations,
                                                        bool validate_input)
{
  cuopt_expects(start_locations != nullptr && return_locations != nullptr,
                error_type_t::ValidationError,
                "Vehicle locations cannot be null");
  if (validate_input) {
    cuopt_expects(detail::check_min_max_values(
                    start_locations, fleet_size_, 0, num_locations_ - 1, handle_ptr_->get_stream()),
                  error_type_t::ValidationError,
                  "Start location must be between min 0 and max num_locations!");
    cuopt_expects(
      detail::check_min_max_values(
        return_locations, fleet_size_, 0, num_locations_ - 1, handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Return location must be between min 0 and max num_locations!");
  }
  start_locations_  = start_locations;
  return_locations_ = return_locations;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_vehicle_types(uint8_t const* vehicle_types,
                                                    bool validate_input)
{
  cuopt_expects(
    vehicle_types != nullptr, error_type_t::ValidationError, "Vehicle types cannot be null!");
  vehicle_types_ = raft::device_span<uint8_t const>(vehicle_types, fleet_size_);
  if (validate_input) {
    cuopt_expects(
      detail::check_min_max_values(vehicle_types_.data(),
                                   fleet_size_,
                                   0,
                                   static_cast<i_t>(std::numeric_limits<uint8_t>::max()),
                                   handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Vehicle types must be between 0 and max uint8_t!");
  }
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_pickup_delivery_pairs(i_t const* pickup_indices,
                                                            i_t const* delivery_indices)
{
  cuopt_expects(pickup_indices != nullptr,
                error_type_t::ValidationError,
                "pickup_indices input cannot be null");
  cuopt_expects(delivery_indices != nullptr,
                error_type_t::ValidationError,
                "delivery_indices input cannot be null");
  pickup_indices_   = pickup_indices;
  delivery_indices_ = delivery_indices;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_drop_return_trips(bool const* drop_return_trip)
{
  cuopt_expects(
    drop_return_trip != nullptr, error_type_t::ValidationError, "drop_return_trip cannot be null");
  drop_return_trip_ = drop_return_trip;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_skip_first_trips(bool const* skip_first_trip)
{
  cuopt_expects(
    skip_first_trip != nullptr, error_type_t::ValidationError, "skip_first_trip cannot be null");
  skip_first_trip_ = skip_first_trip;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_vehicle_order_match(const i_t vehicle_id,
                                                          i_t const* orders,
                                                          const i_t norders)
{
  cuopt_expects(
    orders != nullptr, error_type_t::ValidationError, "vehicle_order_match cannot be null");
  vehicle_order_match_[vehicle_id] = raft::device_span<i_t const>(orders, norders);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_order_vehicle_match(const i_t order_id,
                                                          i_t const* vehicles,
                                                          const i_t nvehicles)
{
  cuopt_expects(
    vehicles != nullptr, error_type_t::ValidationError, "order_vehicle_match cannot be null");
  order_vehicle_match_[order_id] = raft::device_span<i_t const>(vehicles, nvehicles);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_order_service_times(i_t const* service_times,
                                                          const i_t truck_id,
                                                          bool validate_input)
{
  cuopt_expects((truck_id >= 0 && truck_id < fleet_size_) || truck_id == -1,
                error_type_t::ValidationError,
                "truck id must be between 0 and fleet size or should be defaulted");
  cuopt_expects(
    service_times != nullptr, error_type_t::ValidationError, "order service times cannot be null");
  if (validate_input) {
    cuopt_expects(
      detail::check_min_max_values(
        service_times, num_orders_, 0, std::numeric_limits<i_t>::max(), handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Service time must be between min 0 and max i_t!");
  }
  order_service_times_[truck_id] = raft::device_span<i_t const>(service_times, num_orders_);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_capacity_dimension(const std::string& name,
                                                         i_t const* demand,
                                                         i_t const* capacity,
                                                         bool validate_input)
{
  cuopt_expects(
    demand != nullptr, error_type_t::ValidationError, "Demand dimension cannot be null");
  cuopt_expects(
    capacity != nullptr, error_type_t::ValidationError, "Capacity dimension cannot be null");
  if (validate_input) {
    if constexpr (sizeof(demand_i_t) < sizeof(i_t)) {
      cuopt_expects(detail::check_min_max_values(demand,
                                                 num_orders_,
                                                 std::numeric_limits<demand_i_t>::min(),
                                                 std::numeric_limits<demand_i_t>::max(),
                                                 handle_ptr_->get_stream()),
                    error_type_t::ValidationError,
                    "Demands must be between min int16_t and max int16_t!");
    }

    if constexpr (sizeof(cap_i_t) < sizeof(i_t)) {
      cuopt_expects(detail::check_min_max_values(capacity,
                                                 fleet_size_,
                                                 std::numeric_limits<cap_i_t>::min(),
                                                 std::numeric_limits<cap_i_t>::max(),
                                                 handle_ptr_->get_stream()),
                    error_type_t::ValidationError,
                    "Capacities must be between min uint16_t and max uint16_t!");
    }
  }
  caps_.emplace_back(std::move(detail::capacity_t<i_t, f_t>(name, demand, capacity)));

  cuopt_expects(caps_.size() < 4u,
                error_type_t::ValidationError,
                "Number of capacity dimensions cannot be more than 3!");
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_order_time_windows(i_t const* earliest,
                                                         i_t const* latest,
                                                         bool validate_input)
{
  cuopt_expects(earliest != nullptr && latest != nullptr,
                error_type_t::ValidationError,
                "Time window cannot be null");
  if (validate_input) {
    cuopt_expects(
      detail::check_min_max_values(
        earliest, num_orders_, 0, std::numeric_limits<i_t>::max(), handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Earliest time must be between min 0 and max i_t!");
    cuopt_expects(
      detail::check_min_max_values(
        latest, num_orders_, 0, std::numeric_limits<i_t>::max(), handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Latest time must be between min 0 and max i_t!");
  }
  order_tw_ = std::move(detail::order_time_window_t<i_t, f_t>(earliest, latest));
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_order_prizes(f_t const* prizes, bool validate_input)
{
  order_prizes_ = raft::device_span<f_t const>(prizes, num_orders_);

  if (validate_input) {
    cuopt_expects(
      detail::check_min_max_values(
        prizes, num_orders_, (f_t)0, std::numeric_limits<f_t>::max(), handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Prizes must be between 0 and max float32!");
  }
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::add_order_precedence(i_t order_id,
                                                       i_t const* preceding_orders,
                                                       i_t n_preceding_orders)
{
  cuopt_expects(false, error_type_t::ValidationError, "Precedence constraints are not supported!");
  cuopt_expects(
    preceding_orders != nullptr, error_type_t::ValidationError, "Preceding orders cannot be null!");
  cuopt_expects(n_preceding_orders != 0,
                error_type_t::ValidationError,
                "At least 1 precedence order must be given!");
  cuopt_expects(
    detail::check_min_max_values(
      preceding_orders, n_preceding_orders, 0, num_orders_ - 1, handle_ptr_->get_stream()),
    error_type_t::ValidationError,
    "Precedence orders should be between 0 and num_orders_ - 1!");
  cuopt_expects(!detail::check_exists(
                  order_id, preceding_orders, n_preceding_orders, handle_ptr_->get_stream()),
                error_type_t::ValidationError,
                "A node cannot have precedence on itself! Order: %d",
                order_id);
  if (precedence_.find(order_id) != precedence_.end()) {
    cuopt_expects(false,
                  error_type_t::ValidationError,
                  "Precedence for this order is already given! Order: %d ",
                  order_id);
  } else {
    cuopt_expects(
      detail::check_no_circular_precedence(
        order_id, preceding_orders, n_preceding_orders, precedence_, handle_ptr_->get_stream()),
      error_type_t::ValidationError,
      "Circular precedence detected at order: %d",
      order_id);
    precedence_.emplace(order_id, std::make_pair(preceding_orders, n_preceding_orders));
  }
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_min_vehicles(i_t min_vehicles)
{
  cuopt_expects(
    min_vehicles >= 0, error_type_t::ValidationError, "min_vehicles cannot be negative!");
  cuopt_expects(min_vehicles <= fleet_size_,
                error_type_t::ValidationError,
                "min_vehicles cannot be bigger than fleet size!");
  cuopt_expects(min_vehicles <= num_orders_,
                error_type_t::ValidationError,
                "min_vehicles cannot be bigger than number of orders!");
  min_num_vehicles_ = min_vehicles;
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_vehicle_max_costs(f_t const* vehicle_max_costs)
{
  cuopt_expects(vehicle_max_costs != nullptr,
                error_type_t::ValidationError,
                "vehicle_max_costs cannot be null");
  vehicle_max_costs_ = raft::device_span<f_t const>(vehicle_max_costs, fleet_size_);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_vehicle_max_times(f_t const* vehicle_max_times)
{
  cuopt_expects(vehicle_max_times != nullptr,
                error_type_t::ValidationError,
                "vehicle_max_times cannot be null");
  vehicle_max_times_ = raft::device_span<f_t const>(vehicle_max_times, fleet_size_);
}

template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_vehicle_fixed_costs(f_t const* vehicle_fixed_costs)
{
  cuopt_expects(vehicle_fixed_costs != nullptr,
                error_type_t::ValidationError,
                "vehicle_fixed_costs cannot be null");
  vehicle_fixed_costs_ = raft::device_span<f_t const>(vehicle_fixed_costs, fleet_size_);
}

template <typename i_t, typename f_t>
f_t const* data_model_view_t<i_t, f_t>::get_cost_matrix(uint8_t vehicle_type) const noexcept
{
  if (cost_matrices_.find(vehicle_type) != cost_matrices_.end())
    return cost_matrices_.at(vehicle_type);
  return nullptr;
}

template <typename i_t, typename f_t>
f_t const* data_model_view_t<i_t, f_t>::get_transit_time_matrix(uint8_t vehicle_type) const noexcept
{
  if (transit_time_matrices_.find(vehicle_type) != transit_time_matrices_.end())
    return transit_time_matrices_.at(vehicle_type);
  return nullptr;
}

template <typename i_t, typename f_t>
std::unordered_map<uint8_t, f_t const*> data_model_view_t<i_t, f_t>::get_cost_matrices()
  const noexcept
{
  return cost_matrices_;
}

template <typename i_t, typename f_t>
std::unordered_map<uint8_t, f_t const*> data_model_view_t<i_t, f_t>::get_transit_time_matrices()
  const noexcept
{
  return transit_time_matrices_;
}

template <typename i_t, typename f_t>
std::tuple<raft::device_span<i_t const>,
           raft::device_span<i_t const>,
           raft::device_span<node_type_t const>,
           raft::device_span<i_t const>>
data_model_view_t<i_t, f_t>::get_initial_solutions() const noexcept
{
  return std::make_tuple(
    initial_vehicle_ids_, initial_routes_, initial_types_, initial_sol_offsets_);
}

template <typename i_t, typename f_t>
i_t const* data_model_view_t<i_t, f_t>::get_order_locations() const noexcept
{
  return order_locations_;
}

template <typename i_t, typename f_t>
std::tuple<i_t const*, i_t> data_model_view_t<i_t, f_t>::get_break_locations() const noexcept
{
  return std::make_tuple(break_locations_, n_break_locations_);
}

template <typename i_t, typename f_t>
std::tuple<objective_t const*, f_t const*, i_t>
data_model_view_t<i_t, f_t>::get_objective_function() const noexcept
{
  return std::make_tuple(objective_, objective_weights_, n_objectives_);
}

template <typename i_t, typename f_t>
i_t data_model_view_t<i_t, f_t>::get_num_locations() const noexcept
{
  return num_locations_;
}

template <typename i_t, typename f_t>
i_t data_model_view_t<i_t, f_t>::get_fleet_size() const noexcept
{
  return fleet_size_;
}

template <typename i_t, typename f_t>
i_t data_model_view_t<i_t, f_t>::get_num_orders() const noexcept
{
  return num_orders_;
}

template <typename i_t, typename f_t>
i_t data_model_view_t<i_t, f_t>::get_num_requests() const noexcept
{
  return n_requests_;
}

template <typename i_t, typename f_t>
i_t data_model_view_t<i_t, f_t>::get_num_vehicle_types() const noexcept
{
  return cost_matrices_.size();
}

template <typename i_t, typename f_t>
raft::device_span<uint8_t const> data_model_view_t<i_t, f_t>::get_vehicle_types() const noexcept
{
  return vehicle_types_;
}

template <typename i_t, typename f_t>
std::vector<detail::break_dimension_t<i_t, f_t>> const&
data_model_view_t<i_t, f_t>::get_uniform_breaks() const noexcept
{
  return break_dimensions_;
}

template <typename i_t, typename f_t>
std::map<i_t, std::vector<detail::vehicle_break_t<i_t>>> const&
data_model_view_t<i_t, f_t>::get_non_uniform_breaks() const noexcept
{
  return vehicle_breaks_;
}

template <typename i_t, typename f_t>
bool data_model_view_t<i_t, f_t>::has_vehicle_breaks() const noexcept
{
  return !break_dimensions_.empty() || !vehicle_breaks_.empty();
}

template <typename i_t, typename f_t>
std::pair<i_t const*, i_t const*> data_model_view_t<i_t, f_t>::get_pickup_delivery_pair()
  const noexcept
{
  return std::make_pair(pickup_indices_, delivery_indices_);
}

template <typename i_t, typename f_t>
std::unordered_map<i_t, std::pair<i_t const*, i_t>>
data_model_view_t<i_t, f_t>::get_order_precedence() const noexcept
{
  return precedence_;
}

template <typename i_t, typename f_t>
bool const* data_model_view_t<i_t, f_t>::get_drop_return_trips() const noexcept
{
  return drop_return_trip_;
}

template <typename i_t, typename f_t>
bool const* data_model_view_t<i_t, f_t>::get_skip_first_trips() const noexcept
{
  return skip_first_trip_;
}

template <typename i_t, typename f_t>
const std::unordered_map<i_t, raft::device_span<i_t const>>&
data_model_view_t<i_t, f_t>::get_vehicle_order_match() const noexcept
{
  return vehicle_order_match_;
}

template <typename i_t, typename f_t>
const std::unordered_map<i_t, raft::device_span<i_t const>>&
data_model_view_t<i_t, f_t>::get_order_vehicle_match() const noexcept
{
  return order_vehicle_match_;
}

template <typename i_t, typename f_t>
const std::unordered_map<i_t, raft::device_span<i_t const>>&
data_model_view_t<i_t, f_t>::get_order_service_times() const noexcept
{
  return order_service_times_;
}

template <typename i_t, typename f_t>
const std::vector<detail::capacity_t<i_t, f_t>>&
data_model_view_t<i_t, f_t>::get_capacity_dimensions() const noexcept
{
  return caps_;
}

template <typename i_t, typename f_t>
std::pair<i_t const*, i_t const*> data_model_view_t<i_t, f_t>::get_vehicle_time_windows()
  const noexcept
{
  return std::make_pair(vehicle_tw_.get_earliest_time(), vehicle_tw_.get_latest_time());
}

template <typename i_t, typename f_t>
std::pair<i_t const*, i_t const*> data_model_view_t<i_t, f_t>::get_vehicle_locations()
  const noexcept
{
  return std::make_pair(start_locations_, return_locations_);
}

template <typename i_t, typename f_t>
std::tuple<i_t const*, i_t const*> data_model_view_t<i_t, f_t>::get_order_time_windows()
  const noexcept
{
  return std::make_tuple(order_tw_.get_earliest_time(), order_tw_.get_latest_time());
}

template <typename i_t, typename f_t>
raft::device_span<f_t const> data_model_view_t<i_t, f_t>::get_order_prizes() const noexcept
{
  return order_prizes_;
}

template <typename i_t, typename f_t>
i_t data_model_view_t<i_t, f_t>::get_min_vehicles() const noexcept
{
  return min_num_vehicles_;
}

template <typename i_t, typename f_t>
raft::device_span<f_t const> data_model_view_t<i_t, f_t>::get_vehicle_max_costs() const noexcept
{
  return vehicle_max_costs_;
}

template <typename i_t, typename f_t>
raft::device_span<f_t const> data_model_view_t<i_t, f_t>::get_vehicle_max_times() const noexcept
{
  return vehicle_max_times_;
}

template <typename i_t, typename f_t>
raft::device_span<f_t const> data_model_view_t<i_t, f_t>::get_vehicle_fixed_costs() const noexcept
{
  return vehicle_fixed_costs_;
}

template <typename i_t, typename f_t>
raft::handle_t const* data_model_view_t<i_t, f_t>::get_handle_ptr() const noexcept
{
  return handle_ptr_;
}

template class data_model_view_t<int, float>;
}  // namespace routing
}  // namespace cuopt
