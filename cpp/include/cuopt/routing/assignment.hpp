/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/error.hpp>
#include <cuopt/routing/routing_structures.hpp>
#include <fstream>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <string_view>
#include <vector>

namespace cuopt {
namespace routing {

/*! Routing assignment default strings */
class solution_string_t {
 public:
  const static std::string success;
  // TODO this is for the old solver, update this to "An infeasible solution
  // found."
  const static std::string infeasible;
  const static std::string timeout;
  const static std::string empty;
  const static std::string error;
};

/*! Routing assignment status */
enum class solution_status_t {
  SUCCESS = 0, /*!< A feasible solution was found.*/
  INFEASIBLE,  /*!< No feasible solution found for the problem.*/
  TIMEOUT,     /*!< Time limit reached before finding a solution.*/
  EMPTY,       /*!< Solver did not run.*/
  ERROR        /*!< An error occured while running the cuOpt solver.*/
};

/**
 * @brief A container of  vehicle routing solver output
 * @tparam i_t Integer type. Needs to be int (32bit) at the moment. Please open
 * an issue if other type are needed.
 */
template <typename i_t>
class assignment_t {
 public:
  /**
   * @brief Constructor.
   *
   * @param status Solution status.
   * @param stream_view Non-owning stream_view object.
   */
  assignment_t(solution_status_t status, rmm::cuda_stream_view stream_view);
  /**
   * @brief Constructor.
   *
   * @param error_status Error status.
   * @param stream_view Non-owning stream_view object.
   */
  assignment_t(cuopt::logic_error error_status, rmm::cuda_stream_view stream_view);
  /**
   * @brief Constructor.
   *
   * @param vehicle_count Number of vehicles in the solution.
   * @param total_objective_value Total objective value of the solution.
   * @param objective_values Objective value of each objective
   * @param route Device vector containing the ordered node ids.
   * @param arrival_stamp Device vector containing arrival time of each node.
   * @param truck_id Device vector containing truck id of each node.
   * @param route_locations Device vector containing the location of orders
   * @param node_type Device vector containing the type of the order
   * @param unserviced_nodes Device vector containing unserviced orders
   * @param accepted Device vector containing accepted solutions
   * @param status Solution status.
   * @param solution_string Solution string explaining the status.
   */
  assignment_t(i_t vehicle_count,
               double total_objective_value,
               std::map<objective_t, double>& objective_values,
               rmm::device_uvector<i_t>& route,
               rmm::device_uvector<double>& arrival_stamp,
               rmm::device_uvector<i_t>& truck_id,
               rmm::device_uvector<i_t>& route_locations,
               rmm::device_uvector<i_t>& node_type,
               rmm::device_uvector<i_t>& unserviced_nodes,
               rmm::device_uvector<i_t>& accepted,
               solution_status_t status,
               std::string solution_string);

  /**
   * @brief Returns the objective value of the solution as a `double`. The
   * objective value is calculated based on the user provided objective function
   * and the routes found by the solver. By default the solver optimizes for
   * vehicle count first and cost second with associated weights INT_MAX and 1.
   * @return Best objective value
   */
  double get_total_objective() const;

  /**
   * @brief Returns the objective value of all the objectives as a vector
   *
   * @return rmm::device_uvector<double>
   */

  const std::map<objective_t, double>& get_objectives() const noexcept;

  /**
   * @brief Returns the number of vehicle needed for this routing assignment as
   * an `i_t`.
   * @return Number of vehicles in the solution
   */
  i_t get_vehicle_count() const;

  /**
   * @brief Returns the execution time of the solver
   * @return Number of seconds elapsed.
   */
  double get_runtime() const noexcept;

  /**
   * @brief Returns the route as a vector of locations. One entry per stop in
   * this vector.
   *
   * @return rmm::device_uvector<i_t> The device memory container for the route.
   */
  rmm::device_uvector<i_t>& get_route() noexcept;

  const rmm::device_uvector<i_t>& get_route() const noexcept;

  /**
   * @brief Returns the truck id for each stop as a vector of identifiers in [0,
   * number of vehicles in fleet).
   *
   * @return rmm::device_uvector<i_t> The device memory container for the truck
   * ids.
   */
  rmm::device_uvector<i_t>& get_truck_id() noexcept;

  const rmm::device_uvector<i_t>& get_truck_id() const noexcept;

  /**
   * @brief Returns the arrival stamp as a vector of `double` for each stop in
   * the route.
   *
   * @return rmm::device_uvector<i_t> The device memory container for the
   * arrival stamp.
   */
  rmm::device_uvector<double>& get_arrival_stamp() noexcept;

  const rmm::device_uvector<double>& get_arrival_stamp() const noexcept;

  /**
   * @brief Returns the order locations as a vector of `i_t`. One entry per stop
   * in this vector.
   *
   * @return rmm::device_uvector<i_t> The device memory container for the
   * location of orders.
   */
  rmm::device_uvector<i_t>& get_order_locations() noexcept;

  const rmm::device_uvector<i_t>& get_order_locations() const noexcept;

  /**
   * @brief Returns the node types as a vector of `i_t`. One entry per stop in
   * this vector.
   *
   * @return rmm::device_uvector<i_t> The device memory container for the type
   * of each order
   */
  rmm::device_uvector<i_t>& get_node_types() noexcept;

  const rmm::device_uvector<i_t>& get_node_types() const noexcept;

  /**
   * @brief Returns the nodes as a vector of `i_t`. One entry per unserviced
   * node
   *
   * @return rmm::device_uvector<i_t> The device memory container
   */
  rmm::device_uvector<i_t>& get_unserviced_nodes() noexcept;

  const rmm::device_uvector<i_t>& get_unserviced_nodes() const noexcept;

  /**
   * @brief Returns the nodes as a vector of `i_t`. One entry per unserviced
   * node
   *
   * @return rmm::device_uvector<i_t> The device memory container
   */
  rmm::device_uvector<i_t>& get_accepted() noexcept;

  const rmm::device_uvector<i_t>& get_accepted() const noexcept;

  /**
   * @brief Writes route to file iin csv format.
   * @param filename Name of the output file
   * @param stream_view Non-owning stream view object
   */
  void to_csv(std::string_view filename, rmm::cuda_stream_view stream_view);

  /**
   * @brief Returns the final status as a human readable string
   * @return The human readable solver status string
   */
  std::string get_status_string() const noexcept;
  /**
   * @brief Returns the error status
   * @return The error status
   */
  cuopt::logic_error get_error_status() const noexcept;
  /**
   * @brief Set vehicle count to the assignment object
   * @param vehicle_count Vehicle count
   */
  void set_vehicle_count(i_t vehicle_count);
  /**
   * @brief Set status the assignment object
   * @param status Solution status
   */
  void set_status(solution_status_t status);
  /**
   * @brief Get status
   * @return Solution status
   */
  solution_status_t get_status() const;
  /**
   * @brief Print the assignment object
   * @param os Output stream to print
   */
  void print(std::ostream& os = std::cout) const noexcept;

 private:
  double total_objective_value_;
  std::map<objective_t, double> objective_values_;
  rmm::device_uvector<i_t> route_;
  rmm::device_uvector<double> arrival_stamp_;
  rmm::device_uvector<i_t> truck_id_;
  rmm::device_uvector<i_t> route_locations_;
  rmm::device_uvector<i_t> node_types_;
  rmm::device_uvector<i_t> unserviced_nodes_;
  rmm::device_uvector<i_t> accepted_{};
  double timer{};
  i_t vehicle_count_{0};
  solution_status_t status_{solution_status_t::EMPTY};
  std::string solution_string_;
  cuopt::logic_error error_status_;
};

template <typename i_t>
struct host_assignment_t {
  host_assignment_t() = default;
  host_assignment_t(const assignment_t<i_t>& routing_solution);
  void print() const noexcept;

  std::vector<i_t> route{};
  std::vector<i_t> truck_id{};
  std::vector<double> stamp{};
  std::vector<i_t> locations{};
  std::vector<i_t> node_types{};
  std::vector<i_t> unserviced_nodes{};
  std::vector<i_t> accepted{};
};

}  // namespace routing
}  // namespace cuopt
