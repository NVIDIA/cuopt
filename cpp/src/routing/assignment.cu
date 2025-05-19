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

#include <cuopt/routing/assignment.hpp>
#include <raft/util/cudart_utils.hpp>
#include <utilities/copy_helpers.hpp>

namespace cuopt {
namespace routing {

const std::string solution_string_t::success    = "cuOpt solver success.";
const std::string solution_string_t::infeasible = "cuOpt solver infeasible solution found.";
const std::string solution_string_t::timeout =
  "cuOpt solver time limit was reached before finding a feasible solution.";
const std::string solution_string_t::empty = "cuOpt solver did not run.";
const std::string solution_string_t::error = "An error occured while running the cuOpt solver.";

template <typename i_t>
assignment_t<i_t>::assignment_t(solution_status_t status, rmm::cuda_stream_view stream_view)
  : status_(status),
    route_(0, stream_view),
    arrival_stamp_(0, stream_view),
    truck_id_(0, stream_view),
    route_locations_(0, stream_view),
    node_types_(0, stream_view),
    unserviced_nodes_(0, stream_view),
    accepted_(0, stream_view),
    solution_string_(solution_string_t::infeasible),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t>
assignment_t<i_t>::assignment_t(cuopt::logic_error error_status, rmm::cuda_stream_view stream_view)
  : status_(solution_status_t::ERROR),
    route_(0, stream_view),
    arrival_stamp_(0, stream_view),
    truck_id_(0, stream_view),
    route_locations_(0, stream_view),
    node_types_(0, stream_view),
    unserviced_nodes_(0, stream_view),
    accepted_(0, stream_view),
    solution_string_(solution_string_t::error),
    error_status_(error_status)
{
}

template <typename i_t>
assignment_t<i_t>::assignment_t(i_t vehicle_count,
                                double total_objective_value,
                                std::map<objective_t, double>& objective_values,
                                rmm::device_uvector<i_t>& route,
                                rmm::device_uvector<double>& arrival_stamp,
                                rmm::device_uvector<i_t>& truck_id,
                                rmm::device_uvector<i_t>& route_locations,
                                rmm::device_uvector<i_t>& node_types,
                                rmm::device_uvector<i_t>& unserviced_nodes,
                                rmm::device_uvector<i_t>& accepted,
                                solution_status_t status,
                                std::string solution_string)
  : vehicle_count_(vehicle_count),
    total_objective_value_(total_objective_value),
    objective_values_(std::move(objective_values)),
    route_(std::move(route)),
    arrival_stamp_(std::move(arrival_stamp)),
    truck_id_(std::move(truck_id)),
    route_locations_(std::move(route_locations)),
    node_types_(std::move(node_types)),
    unserviced_nodes_(std::move(unserviced_nodes)),
    accepted_(std::move(accepted)),
    status_(status),
    solution_string_(solution_string),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t>
double assignment_t<i_t>::get_total_objective() const
{
  return total_objective_value_;
}

template <typename i_t>
const std::map<objective_t, double>& assignment_t<i_t>::get_objectives() const noexcept
{
  return objective_values_;
}

template <typename i_t>
i_t assignment_t<i_t>::get_vehicle_count() const
{
  return vehicle_count_;
}

template <typename i_t>
double assignment_t<i_t>::get_runtime() const noexcept
{
  return timer;
}

template <typename i_t>
rmm::device_uvector<i_t>& assignment_t<i_t>::get_route() noexcept
{
  return route_;
}

template <typename i_t>
const rmm::device_uvector<i_t>& assignment_t<i_t>::get_route() const noexcept
{
  return route_;
}

template <typename i_t>
rmm::device_uvector<i_t>& assignment_t<i_t>::get_truck_id() noexcept
{
  return truck_id_;
}

template <typename i_t>
const rmm::device_uvector<i_t>& assignment_t<i_t>::get_truck_id() const noexcept
{
  return truck_id_;
}

template <typename i_t>
rmm::device_uvector<double>& assignment_t<i_t>::get_arrival_stamp() noexcept
{
  return arrival_stamp_;
}

template <typename i_t>
const rmm::device_uvector<double>& assignment_t<i_t>::get_arrival_stamp() const noexcept
{
  return arrival_stamp_;
}

template <typename i_t>
rmm::device_uvector<i_t>& assignment_t<i_t>::get_order_locations() noexcept
{
  return route_locations_;
}

template <typename i_t>
const rmm::device_uvector<i_t>& assignment_t<i_t>::get_order_locations() const noexcept
{
  return route_locations_;
}

template <typename i_t>
rmm::device_uvector<i_t>& assignment_t<i_t>::get_node_types() noexcept
{
  return node_types_;
}

template <typename i_t>
const rmm::device_uvector<i_t>& assignment_t<i_t>::get_node_types() const noexcept
{
  return node_types_;
}

template <typename i_t>
rmm::device_uvector<i_t>& assignment_t<i_t>::get_unserviced_nodes() noexcept
{
  return unserviced_nodes_;
}

template <typename i_t>
const rmm::device_uvector<i_t>& assignment_t<i_t>::get_unserviced_nodes() const noexcept
{
  return unserviced_nodes_;
}

template <typename i_t>
rmm::device_uvector<i_t>& assignment_t<i_t>::get_accepted() noexcept
{
  return accepted_;
}

template <typename i_t>
const rmm::device_uvector<i_t>& assignment_t<i_t>::get_accepted() const noexcept
{
  return accepted_;
}

template <typename i_t>
void assignment_t<i_t>::to_csv(std::string_view filename, rmm::cuda_stream_view stream_view)
{
  std::vector<i_t> route;
  std::vector<double> arrival_stamp;
  std::vector<i_t> truck_id;
  route.resize(route_.size());
  arrival_stamp.resize(arrival_stamp_.size());
  truck_id.resize(truck_id_.size());
  raft::copy(route.data(), route_.data(), route_.size(), stream_view.value());
  raft::copy(
    arrival_stamp.data(), arrival_stamp_.data(), arrival_stamp_.size(), stream_view.value());
  raft::copy(truck_id.data(), truck_id_.data(), truck_id_.size(), stream_view.value());
  std::ofstream myfile(filename.data());
  std::cout << "truck_id,\troute,\tarrival_time\n";
  for (size_t i = 0; i < route.size(); i++)
    myfile << truck_id[i] << ",\t" << route[i] << ",\t" << arrival_stamp[i] << std::endl;
}

template <typename i_t>
std::string assignment_t<i_t>::get_status_string() const noexcept
{
  return solution_string_;
}

template <typename i_t>
void assignment_t<i_t>::set_vehicle_count(i_t vehicle_count)
{
  vehicle_count_ = vehicle_count;
}

template <typename i_t>
void assignment_t<i_t>::set_status(solution_status_t status)
{
  status_ = status;
}

template <typename i_t>
solution_status_t assignment_t<i_t>::get_status() const
{
  return status_;
}

template <typename i_t>
cuopt::logic_error assignment_t<i_t>::get_error_status() const noexcept
{
  return error_status_;
}

template <typename i_t>
void assignment_t<i_t>::print(std::ostream& os) const noexcept
{
  raft::print_device_vector("truck_id", truck_id_.data(), truck_id_.size(), os);
  raft::print_device_vector("route", route_.data(), route_.size(), os);
  raft::print_device_vector(
    "route_locations", route_locations_.data(), route_locations_.size(), os);
  raft::print_device_vector("arrival", arrival_stamp_.data(), arrival_stamp_.size(), os);
}

template <typename i_t>
host_assignment_t<i_t>::host_assignment_t(const assignment_t<i_t>& routing_solution)
{
  route            = cuopt::host_copy(routing_solution.get_route());
  truck_id         = cuopt::host_copy(routing_solution.get_truck_id());
  stamp            = cuopt::host_copy(routing_solution.get_arrival_stamp());
  locations        = cuopt::host_copy(routing_solution.get_order_locations());
  node_types       = cuopt::host_copy(routing_solution.get_node_types());
  unserviced_nodes = cuopt::host_copy(routing_solution.get_unserviced_nodes());
  accepted         = cuopt::host_copy(routing_solution.get_accepted());
}

template <typename i_t>
void host_assignment_t<i_t>::print() const noexcept
{
  printf("route  truck  location  arrival_stamp node_type \n");
  for (size_t i = 0; i < route.size(); ++i) {
    printf("%d \t %d \t %d \t %.2f \t %d \n",
           route[i],
           truck_id[i],
           locations[i],
           stamp[i],
           node_types[i]);
  }
}

template class assignment_t<int>;
template class host_assignment_t<int>;

}  // namespace routing
}  // namespace cuopt
