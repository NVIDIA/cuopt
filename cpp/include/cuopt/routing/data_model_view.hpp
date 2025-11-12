/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/routing/routing_structures.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <unordered_set>
namespace cuopt {
namespace routing {

/**
 * @brief A container of vehicle routing solver input
 * @tparam i_t Integer type. Needs to be int (32bit) at the moment. Please open
 * an issue if other type are needed.
 * @tparam f_t Floating point type. Needs to be float (32bit) at the moment.
 * Please open an issue if other type are needed.
 */
template <typename i_t, typename f_t>
class data_model_view_t {
 public:
  /**
   * @brief Construct a data model which holds a non-owning view of the user's
   * model. A data_model_view_t is a container for the solver input.
   */
  data_model_view_t() {}

  /**
   * @brief Construct a data model which holds a non-owning view of the user's
   * model. A data_model_view_t is a container for the solver input.
   *
   * @note A cost matrix must be set before passing this
   * object to the solver. See data_model_view_t::set_cost_matirx
   *
   * @throws cuopt::logic_error when an error occurs.
   *
   * @param[in] handle_ptr Pointer to library handle (RAFT) containing hardware
   * resources information. A default handle is valid.
   * @param num_locations number of locations to visit, including the depot.
   * @param fleet_size number of vehicles in the fleet. This is primarily used
   * to model vehicle properties in a mixed fleet context. cuOpt solution
   * contains the smallest possible number of vehicles which may be smaller or
   * equal to the fleet_size.
   */
  data_model_view_t(raft::handle_t const* handle_ptr,
                    i_t num_locations,
                    i_t const fleet_size,
                    i_t num_orders = -1);

  /**
   * @brief Set a cost (distance) matrix for all locations (depot included) at
   * once. A cost matrix is defined a square matrix containing the
   * costs, taken pairwise, between all locations. Entries are non-negative
   * real numbers. Diagonal elements
   * should be 0. Users should pre-compute costs between each pair of
   * locations with their own technique before calling this function. Entries in
   * this matrix could represent time, miles, meters or any metric that can be
   * stored as a real number and satisfy the property above.
   * The user can call add_cost_matrix multiple times. Setting the
   * vehicle type will enable heterogenous fleet. It can model traveling
   * costs for different vehicles (bicyces, bikes, trucks).
   *
   *
   * @throws cuopt::logic_error when an error occurs.
   * @param[in] matrix  device memory pointer to a floating point
   * matrix of size num_locations_ . cuOpt does not own or copy this data.
   * @param[in] vehicle_type Identifier of the vehicle.
   */
  void add_cost_matrix(f_t const* matrix, uint8_t vehicle_type = 0);

  /**
   * @brief Set a transit time matrix for all locations (depot included) at
   * once. The time matrix is used to check constraints satisfiability rather
   * than participating in cost optimization. For instance, the time matrix
   * is used to model the time between locations with time windows referring
   * to it while the solver could optimize for cost set by add_cost_matrix.
   * The transit time matrix is defined as a square matrix containing the
   * transit time taken pairwise, between all locations. Users should
   * pre-compute times between each pair of locations with their own technique
   * before calling this function. The user can call add_transit_time_matrix
   * multiple times. Setting the vehicle type will enable heterogenous fleet. It
   * can model traveling speeds for different vehicles (bicyces, bikes, trucks).
   *
   *
   * @throws cuopt::logic_error when an error occurs.
   * @param[in] matrix  device memory pointer to a floating point matrix of size
   * num_locations_ . cuOpt does not own or copy this data.
   * @param[in] vehicle_type Identifier of the vehicle.
   */
  void add_transit_time_matrix(f_t const* matrix, uint8_t vehicle_type = 0);

  void add_initial_solutions(i_t const* vehicle_ids,
                             i_t const* routes,
                             node_type_t const* types,
                             i_t const* sol_offsets,
                             i_t n_nodes,
                             i_t n_solutions);

  /**
   * @brief Set locations for all the orders at once. This is needed when there
   * is no one to one mapping from orders to locations, especially when the
   * number of orders way higher than the number of locations in the model
   */
  void set_order_locations(i_t const* order_locations);

  /**
   * @brief The vehicle is allowed to stop at specific locations during a
   * break. It can be at a customer node or another location representing for
   * instance a gas station. The solver will pick the best stop out of all break
   * nodes. The same break node can appear on several routes and satisfy
   * multiple break constraints.
   * @param[in] break_locations device memory pointer to break locations to be
   * used. These locations can be anywhere from 0 to n_locations - 1
   * @param[in] n_break_locations size of break_locations array
   * @param[in] validate_input runs expensive input checks. Defaults to true.
   */
  void set_break_locations(i_t const* break_locations,
                           i_t n_break_locations,
                           bool validate_input = true);

  /**
   * @brief  Add break time windows to model the Vehicle Routing Problem with
   * Time Windows (VRPTW): The locations have break time windows within which
   * the breaks must be taken.
   * @param[in] break_earliest device memory pointer to the earliest time a
   * vehicle can be at a break location.
   * @param[in] break_latest device memory pointer to the latest time a vehicle
   * can arrive at the break location
   * @param[in] break_duration device memory pointer to the time of the break,
   * internally equivalent to service time.
   * @param[in] validate_input runs expensive input checks. Defaults to true.
   */
  void add_break_dimension(i_t const* break_earliest,
                           i_t const* break_latest,
                           i_t const* break_duration,
                           bool validate_input = true);

  /**
   * @brief Add vehicle specific breaks
   *
   * @param vehicle_id
   * @param break_earliest
   * @param break_latest
   * @param break_duration
   * @param break_locations
   * @param num_break_locations
   * @param validate_input
   */
  void add_vehicle_break(i_t vehicle_id,
                         i_t break_earliest,
                         i_t break_latest,
                         i_t break_duration,
                         i_t const* break_locations,
                         i_t num_break_locations,
                         bool validate_input = true);

  /**
   * @brief During improvement phase the solver only optimizes for the cost.
   * This function is used to select the best solution accross all climbers
   * based on other criterias (see objective_t enum). The value for the VEHICLE
   * criteria should be set to a high value (2^32).
   * @param[in] objectives device memory pointer of objective_t
   * @param[in] objective_weights device memory pointer to the weights
   * associated with the objectives.
   * @param[in] n_objectives size of objective and objective_weights arrays
   */
  void set_objective_function(objective_t const* objective,
                              f_t const* objective_weights,
                              i_t n_objectives);

  /**
   * @brief Set starting and returning locations for vehicles in fleet
   *
   * @throws cuopt::logic_error when an error occurs.
   * @param[in] start_locations device memory pointer to the start location
   * of each vehicle. Vehicles with default start locations should be set to
   * 0 (i.e. common depot)
   * The size of this array must be equal to fleet_size
   * @param[in] return_locations device memory pointer to the return
   * location of each vehicle is available. Vehicles with default
   * return locations should be set to 0 (i.e. common depot). This value
   * will not impact results if drop_return_trip is enabled for a vehicle
   * The size of this array should be equal to fleet_size.
   * @param[in] validate_input runs expensive input checks. Defaults to true.
   */
  void set_vehicle_locations(i_t const* start_locations,
                             i_t const* return_locations,
                             bool validate_input = true);

  /**
   * @brief Set a window during which each vehicle can be used. The window could
   * refer to the primary cost matrix or to the transit time matrix.
   *
   * @throws cuopt::logic_error when an error occurs.
   * @param[in] earliest device memory pointer to the earliest time each
   * vehicle is available. Order is implicit and should be consistent with the
   * for all fleet data and capacity dimensions. cuOpt does not own or copy this
   * data. Vehicles without earliest window should be set to 0.
   * The size of this array must be equal to fleet_size
   * @param[in] latest device memory pointer to the latest time each
   * vehicle is available. Order is implicit and should be consistent with the
   * for all fleet data and capacity dimensions. cuOpt does not own or copy this
   * data. Vehicles without latest window constraint should be set to a large
   * value.
   * The size of this array should be equal to fleet_size.
   * @param[in] validate_input runs expensive input checks. Defaults to true.
   */
  void set_vehicle_time_windows(i_t const* earliest, i_t const* latest, bool validate_input = true);

  /**
   * @brief When multiple matrices are given as input the solver
   *       is enabling heterogenous cost matrix and time matrix
   *       optimization. We thus need the corresponding vehicle
   *       type id for all vehicles in the data model.
   * @throws cuopt::logic_error when an error occurs.
   * @param[in] vehicle_types Types of vehicles in the fleet given as positive
   * integer array.
   * @param[in] validate_input Runs expensive input checks. Defaults to true.
   */
  void set_vehicle_types(uint8_t const* vehicle_types, bool validate_input = true);

  /**
   * @brief Set pick-up delivery pairs given by indices to the orders.
   * Currently mixed pickup and delivery is not supported, meaning that
   * all the orders should be a included in the pick-up delivery pair indices.
   *
   * @throws cuopt::logic_error when an error occurs.
   * @param[in] pickup_indices  device memory pointer to pickup indices
   * @param[in] delivery_indices  device memory pointer to delivery indices
   */
  void set_pickup_delivery_pairs(i_t const* pickup_indices, i_t const* delivery_indices);

  /**
   * @brief Control if vehicles return to the depot after the last stop for
   * the individual vehicles in the fleet
   *
   * @param[in] drop_return_trip Boolean array containing true for dropping
   * return trip to depot, default is false
   */
  void set_drop_return_trips(bool const* drop_return_trip);

  /**
   * @brief Control if vehicles start from the first service location for
   * the individual vehicles in the fleet
   *
   * @param[in] skip_first_trip Boolean array containing true for starting from
   * a service location,
   * default is false
   */
  void set_skip_first_trips(bool const* skip_first_trip);

  /**
   * @brief Control if a specified vehicle should only serve a subset of
   * customer orders
   *
   * @param vehicle_id  vehicle id that has constraints
   * @param orders      device memory pointer to integer values corresponding to
   * list of orders
   * @param norders     number of customer orders that are served by this
   * vehicle
   */
  void add_vehicle_order_match(const i_t vehicle_id, i_t const* orders, const i_t norders);

  /**
   * @brief Control if a specified order should only serve a subset of vehicles
   *
   * @param order_id    order id that has constraints
   * @param vehicles    device memory pointer to integer values corresponding to
   * list of vehicles
   * @param nvehicles   number of vehicles that can serve this order
   */
  void add_order_vehicle_match(const i_t order_id, i_t const* vehicles, const i_t nvehicles);

  /**
   * @brief In fully heterogenous fleet mode, vehicle can take different amount
   * of times to complete a task based on their profile and the order being
   * served. Here we enable that ability to the user by setting for each vehicle
   * id the corresponding service times. They can be the same for all orders per
   * vehicle/vehicle type or unique.
   *
   * The service times are defaulted for all vehicles unless
   * vehicle id is specified. If no default service times are
   * given then the solver expects all vehicle ids up to fleet size to be
   * specified.
   *
   * @param vehicle_id  vehicle id that has constraints
   * @param service_times      device memory pointer to integer values
   * corresponding to list of service_times
   */
  void set_order_service_times(i_t const* service_times,
                               const i_t truck_id  = -1,
                               bool validate_input = true);

  /**
   * @brief Add capacity dimensions to model the Capacitated Vehicle Routing
   * Problem: CVRP. The vehicles have a limited carrying capacity of the goods
   * that must be delivered. This function can be called more than once to
   * model multiple capacity dimensions (weight, volume, number of orders).
   * After solving the problem, the demands on each route will not exceed the
   * vehicle capacities.
   *
   * @param[in] name user-specified name for the dimension
   * @param[in] demand device memory pointer to an integer demand value for
   * each locations, including the depot. Order is implicit and should be
   * consitent with the data model. cuOpt does not own or copy this data.
   * @param[in] capacity device memory pointer to an integer capacity value
   * for each vehicle in the fleet.
   * @param[in] validate_input runs expensive input checks. Defaults to true.
   */
  void add_capacity_dimension(const std::string& name,
                              i_t const* demand,
                              i_t const* capacity,
                              bool validate_input = true);

  /**
   * @brief Add time windows to model the Vehicle Routing Problem with Time
   * Windows (VRPTW): The locations have time windows within which the visits
   * must be made.
   *
   * @note calling set_order_time_windows twice with the same time_window_t is
   * an undefined behaviour
   * @param[in] earliest device memory pointer to the earliest visit time for
   * each location including the depot. Order is implicit and should be
   * consistent with the data model. cuOpt does not own or copy this data.
   * @param[in] latest device memory pointer to the latest visit time for each
   * location including the depot. Order is implicit and should be consistent
   * with the data model. cuOpt does not own or copy this data.
   * @param[in] validate_input runs expensive input checks. Defaults to true.
   */
  void set_order_time_windows(i_t const* earliest, i_t const* latest, bool validate_input = true);

  /**
   * @brief Set the order prizes for prize collection
   *
   * @throws cuopt::logic_error when an error occurs
   * @param[in] prizes Prizes of orders given as positive 32 bit float array
   * @param[in] validate_input runs expensive input checks, Defaults to true
   */
  void set_order_prizes(f_t const* prizes, bool validate_input = true);

  /**
   * @brief Add precedence constraints for a given order.
   * For each order that needs to come after one or more orders call this
   * function. Currently circular dependencies are not accepted.
   *
   * @param order_id Order id that has a precedence constraint.
   * @param preceding_orders The orders that need to be scheduled prior to
   * node_id
   * @param n_preceding_orders Number of prior orders.
   */
  void add_order_precedence(i_t order_id, i_t const* preceding_orders, i_t n_preceding_orders);

  /**
   * @brief Request a minimum number of vehicles
   *
   * @note The resulting solution may not be optimal
   *
   * @param[in] min_vehicles The minimum number of vehicle to use
   */
  void set_min_vehicles(i_t min_vehicles);

  /**
   * @brief Limits the primary matrix cost cumulated along a route.
   * @param[in] vehicle_max_costs Upper bound for route cost.
   */
  void set_vehicle_max_costs(f_t const* vehicle_max_costs);

  /**
   * @brief Lets the solver find the optimal fleet according to vehicle costs.
   * In a heterogeneous setting, not all vehicles will have the same cost.
   * Sometimes it may be optimal to use two vehicles with lower cost compared to
   * one vehicle with a huge cost.
   * @param[in] vehicle_fixed_costs Cost of each vehicle
   */
  void set_vehicle_fixed_costs(f_t const* vehicle_fixed_costs);

  /**
   * @brief Limits the time cumulated along a route. This limit accounts for
   * both travel and service time.
   * @param[in] vehicle_max_times Upper bound for route time.
   */
  void set_vehicle_max_times(f_t const* vehicle_max_times);

  /**
   * @brief Get cost matrix
   * @return Matrix pointer
   */
  f_t const* get_cost_matrix(uint8_t vehicle_type = 0) const noexcept;

  /**
   * @brief Get transit time matrix
   * @return Matrix pointer
   */
  f_t const* get_transit_time_matrix(uint8_t vehicle_type = 0) const noexcept;

  /**
   * @brief Get all cost matrices as a map
   * @return map of vehicle type to cost matrix
   */
  std::unordered_map<uint8_t, f_t const*> get_cost_matrices() const noexcept;

  /**
   * @brief Get all transit time matrices as a map
   * @return map of vehicle type to transit time matrix
   */
  std::unordered_map<uint8_t, f_t const*> get_transit_time_matrices() const noexcept;

  std::tuple<raft::device_span<i_t const>,
             raft::device_span<i_t const>,
             raft::device_span<node_type_t const>,
             raft::device_span<i_t const>>
  get_initial_solutions() const noexcept;

  /**
   * @brief Get location of orders
   *
   * @return i_t const*
   */
  i_t const* get_order_locations() const noexcept;

  /**
   * @brief Get number break location ids and size of array.
   * @return Tuple of break locations ids and number.
   */
  std::tuple<i_t const*, i_t> get_break_locations() const noexcept;

  /**
   * @brief Get objective_t,
   * weight pairs and the number of objectives.*@ return Tuple containing
   * objective_t, associated weights as arrays and the *number of objectives to
   * optimize.
   */
  std::tuple<objective_t const*, f_t const*, i_t> get_objective_function() const noexcept;

  /**
   * @brief Get number of locations in the input
   * @return Number of locations
   */
  i_t get_num_locations() const noexcept;

  /**
   * @brief Get number of vehicles in the input
   * @return Fleet size
   */
  i_t get_fleet_size() const noexcept;

  /**
   * @brief Get the number of orders in the input
   * @return i_t
   */
  i_t get_num_orders() const noexcept;

  /**
   * @brief Get the number of vehicle types set
   * @return i_t
   */
  i_t get_num_vehicle_types() const noexcept;

  /**
   * @brief Get the number of pairs for pdp use case
   * @return i_t
   */
  i_t get_num_requests() const noexcept;

  /**
   * @brief Get types for each vehicle
   * @return Integer pointer pointing to vehicle types
   */
  raft::device_span<uint8_t const> get_vehicle_types() const noexcept;

  /**
   * @brief Get break schedule for each vehicle
   * @return Pair of pointers contaning max time between breaks and duration of
   * break.
   */
  std::vector<detail::break_dimension_t<i_t, f_t>> const& get_uniform_breaks() const noexcept;

  std::map<i_t, std::vector<detail::vehicle_break_t<i_t>>> const& get_non_uniform_breaks()
    const noexcept;

  /**
   * @brief Check if there are breaks defined in the model
   * @return True if there are breaks specified
   */
  bool has_vehicle_breaks() const noexcept;

  /**
   * @brief Get dropping return trip for each vehicle
   * @return Boolean pointer pointing to array containing true for dropping
   * return trip to depot, default is false
   */
  bool const* get_drop_return_trips() const noexcept;

  /**
   * @brief Get skipping first trip for each vehicle
   * @return Boolean pointer pointing to array containing true for skipping
   * first trip from depot, default is false
   */
  bool const* get_skip_first_trips() const noexcept;

  /**
   * @brief Get the vehicle order match map
   *
   * @return A map of vehicle id and corresponding customer orders that can be
   * served
   */
  const std::unordered_map<i_t, raft::device_span<i_t const>>& get_vehicle_order_match()
    const noexcept;

  /**
   * @brief Get the vehicle service times map
   *
   * @return A map of truck id and corresponding customer orders that can be
   * served
   */
  const std::unordered_map<i_t, raft::device_span<i_t const>>& get_order_service_times()
    const noexcept;

  /**
   * @brief Get the order vehicle match map
   *
   * @return A map of order id and corresponding vehicles that can serve the
   * order
   */
  const std::unordered_map<i_t, raft::device_span<i_t const>>& get_order_vehicle_match()
    const noexcept;

  /**
   * @brief Get capacity for all dimensions
   * @return reference to vector of capacities
   */
  const std::vector<detail::capacity_t<i_t, f_t>>& get_capacity_dimensions() const noexcept;

  /**
   * @brief Get vehicle time windows
   * @return Pair of pointers containing time windows of earliest and latest
   */
  std::pair<i_t const*, i_t const*> get_vehicle_time_windows() const noexcept;

  /**
   * @brief Get vehicle start and return locations
   * @return Pair of pointers containing start and return locations
   */
  std::pair<i_t const*, i_t const*> get_vehicle_locations() const noexcept;

  /**
   * @brief Get order/location/job time windows
   * @return A tuple of 4 pointers with earliest, latest
   */
  std::tuple<i_t const*, i_t const*> get_order_time_windows() const noexcept;

  /**
   * @brief Get the order prizes
   * @return raft::device_span<f_t const>
   */
  raft::device_span<f_t const> get_order_prizes() const noexcept;

  /**
   * @brief Get pickup delivery pairs
   * @return Pair of pointers containing pick up and delivery indices
   */
  std::pair<i_t const*, i_t const*> get_pickup_delivery_pair() const noexcept;

  /**
   * @brief Get precendece vector
   * @return A set of pair of i_t* and i_t representing the precedence and its
   * size.
   */
  std::unordered_map<i_t, std::pair<i_t const*, i_t>> get_order_precedence() const noexcept;

  /**
   * @breif Return minimum vehicles set
   * @return minimum vehicles needed in solution
   */
  i_t get_min_vehicles() const noexcept;

  /**
   * @brief Return max cost allowed per vehicle
   * @return max cost per route
   */
  raft::device_span<f_t const> get_vehicle_max_costs() const noexcept;

  /**
   * @brief Return max time allowed per vehicle
   * @return max time per route
   */
  raft::device_span<f_t const> get_vehicle_max_times() const noexcept;

  /**
   * @brief Return cost per vehicle
   * @return cost per route
   */
  raft::device_span<f_t const> get_vehicle_fixed_costs() const noexcept;

  /**
   * @brief Get raft handle object containing GPU resource objects
   * @return Handle object
   */
  raft::handle_t const* get_handle_ptr() const noexcept;

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  i_t num_locations_{};
  i_t fleet_size_{};
  i_t num_orders_{};
  i_t n_requests_{};
  raft::device_span<uint8_t const> vehicle_types_;
  std::unordered_map<uint8_t, f_t const*> cost_matrices_{};
  std::unordered_map<uint8_t, f_t const*> transit_time_matrices_{};
  i_t const* order_locations_{nullptr};
  i_t const* break_locations_{nullptr};
  i_t n_break_locations_{};
  std::vector<detail::break_dimension_t<i_t, f_t>> break_dimensions_{};
  i_t const* pickup_indices_{nullptr};
  i_t const* delivery_indices_{nullptr};
  detail::vehicle_time_window_t<i_t, f_t> vehicle_tw_{};
  std::vector<detail::capacity_t<i_t, f_t>> caps_{};
  detail::order_time_window_t<i_t, f_t> order_tw_{};

  raft::device_span<f_t const> order_prizes_;

  i_t const* start_locations_{nullptr};
  i_t const* return_locations_{nullptr};
  bool const* drop_return_trip_{nullptr};
  bool const* skip_first_trip_{nullptr};
  std::unordered_map<i_t, raft::device_span<i_t const>> vehicle_order_match_;
  std::unordered_map<i_t, raft::device_span<i_t const>> order_vehicle_match_;
  std::unordered_map<i_t, raft::device_span<i_t const>> order_service_times_;
  objective_t const* objective_{};
  f_t const* objective_weights_{};
  i_t n_objectives_{};
  std::unordered_map<i_t, std::pair<i_t const*, i_t>> precedence_{};
  i_t min_num_vehicles_{0};

  raft::device_span<f_t const> vehicle_max_costs_{};
  raft::device_span<f_t const> vehicle_max_times_{};
  raft::device_span<f_t const> vehicle_fixed_costs_{};

  raft::device_span<i_t const> initial_vehicle_ids_{};
  raft::device_span<i_t const> initial_routes_{};
  raft::device_span<node_type_t const> initial_types_{};
  raft::device_span<i_t const> initial_sol_offsets_{};
  std::map<i_t, std::vector<detail::vehicle_break_t<i_t>>> vehicle_breaks_{};
};
}  // namespace routing
}  // namespace cuopt
