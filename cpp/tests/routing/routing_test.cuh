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
#include <cuopt/routing/solve.hpp>
#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/test_utilities.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>

#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <routing/fleet_info.hpp>
#include <routing/generator/generator.hpp>
#include <routing/utilities/md_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/high_res_timer.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>

#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <cuda/std/functional>

#include <cuda_profiler_api.h>

#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

static const bool generate_ref = false;

typedef struct file_params_t {
  file_params_t() {}
  file_params_t(const std::string& filename = "", const float cost = 0, const int vn = 0)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
    if ((filename != "") && (filename[0] != '/')) {
      routing_file = rapidsDatasetRootDir + "/" + filename;
    } else {
      routing_file = filename;
    }
    ref_vn   = vn;
    ref_cost = cost;
  }

  std::string routing_file;
  float ref_cost{};
  int ref_vn{};
} file_params;

inline std::vector<file_params> parse_tests(const std::vector<std::string> test_files)
{
  std::vector<file_params> param_tests;
  for (std::string line : test_files) {
    auto tokens = cuopt::test::split(line, ',');
    param_tests.emplace_back(tokens[0], std::stof(tokens[1]), std::stoi(tokens[2]));
  }
  return param_tests;
}

template <typename i_t, typename f_t>
struct test_data_t {
  std::vector<f_t> x_h;
  std::vector<f_t> y_h;
  std::vector<i_t> earliest_time_h;
  std::vector<i_t> latest_time_h;
  std::vector<i_t> service_time_h;
  std::vector<i_t> demand_h;
  std::vector<i_t> pickup_delivery_demand_h;
  std::vector<i_t> pickup_earliest_time_h;
  std::vector<i_t> pickup_latest_time_h;

  std::vector<i_t> pickup_indices_h;
  std::vector<i_t> delivery_indices_h;
  std::vector<i_t> capacity_h;
  std::vector<i_t> mixed_capacity_h;
  std::vector<i_t> vehicle_earliest_h;
  std::vector<i_t> vehicle_latest_h;

  std::vector<i_t> break_earliest_h;
  std::vector<i_t> break_latest_h;
  std::vector<i_t> break_duration_h;
  std::vector<i_t> drop_return_h;
  std::vector<i_t> skip_first_h;
  std::vector<uint8_t> vehicle_types_h;
  std::vector<f_t> cost_matrix_h;

  i_t n_vehicles;
  i_t depot;
  i_t n_locations;
  std::vector<i_t> request;
  std::vector<i_t> p_scores;
  std::vector<i_t> expected_found_sol;
  std::vector<i_t> expected_route;
};

template <typename problem_t, typename i_t = int, typename f_t = float>
static test_data_t<i_t, f_t> parse_problem(problem_t&& problem)
{
  return {problem.x_h,
          problem.y_h,
          problem.earliest_time_h,
          problem.latest_time_h,
          problem.service_time_h,
          problem.demand_h,
          problem.pickup_delivery_demand_h,
          problem.pickup_earliest_time_h,
          problem.pickup_latest_time_h,
          problem.pickup_indices_h,
          problem.delivery_indices_h,
          problem.capacity_h,
          problem.mixed_capacity_h,
          problem.vehicle_earliest_h,
          problem.vehicle_latest_h,
          problem.break_earliest_h,
          problem.break_latest_h,
          problem.break_duration_h,
          problem.drop_return_h,
          problem.skip_first_h,
          problem.vehicle_types_h,
          problem.cost_matrix_h,
          problem.n_vehicles,
          problem.depot,
          problem.n_locations,
          problem.request,
          problem.p_scores,
          problem.expected_found_sol,
          problem.expected_route};
}

template <typename... Args>
std::vector<test_data_t<int, float>> parse_problems(Args&&... args)
{
  return {parse_problem(std::forward<Args>(args))...};
}

template <typename i_t, typename f_t>
Route<i_t, f_t> load_routing_file(std::string routing_file, i_t limit)
{
  std::cout << "File is: " << routing_file << "\n";
  Route<i_t, f_t> input;
  if (routing_file.find(".tsp") != std::string::npos)
    load_tsp(routing_file, input);
  else if (routing_file.find(".vrp") != std::string::npos)
    load_cvrp(routing_file, input);
  else if (routing_file.find(".dat") != std::string::npos)
    load_acvrp(routing_file, input);
  else if (routing_file.find(".pdptw") != std::string::npos)
    load_pickup(routing_file, input);
  else
    load_cvrptw(routing_file, input, limit);
  return input;
}

template <typename i_t, typename f_t>
class base_test_t {
 public:
  base_test_t(f_t cost_tol = 1E-1, i_t vn_tol = 0, i_t limit = 1001)
    : cost_tol_(cost_tol),
      vn_tol_(vn_tol),
      limit_(limit),
      dim_(1),
      stream_view_(handle_.get_stream()),
      earliest_time_h(0),
      latest_time_h(0),
      service_time_h(0),
      drop_return_trips_h(0),
      skip_first_trips_h(0),
      vehicle_types_h(0),
      vehicle_earliest_h(0),
      vehicle_latest_h(0),
      break_locations_h(0),
      break_earliest_h(0),
      break_latest_h(0),
      break_duration_h(0),
      vehicle_max_costs_h(0),
      vehicle_max_times_h(0),
      vehicle_fixed_costs_h(0),
      x_d(0, stream_view_),
      y_d(0, stream_view_),
      matrices_d(stream_view_),
      fleet_order_constraints_d(&handle_, 0, 0),
      cost_matrix_d(0, stream_view_),
      order_locations_d(0, stream_view_),
      demand_d(0, stream_view_),
      capacity_d(0, stream_view_),
      earliest_time_d(0, stream_view_),
      latest_time_d(0, stream_view_),
      service_time_d(0, stream_view_),
      drop_return_trips_d(0, stream_view_),
      skip_first_trips_d(0, stream_view_),
      bool_drop_return_trips_d(0, stream_view_),
      bool_skip_first_trips_d(0, stream_view_),
      vehicle_types_d(0, stream_view_),
      vehicle_earliest_d(0, stream_view_),
      vehicle_latest_d(0, stream_view_),
      pickup_indices_d(0, stream_view_),
      delivery_indices_d(0, stream_view_),
      break_locations_d(0, stream_view_),
      break_earliest_d(0, stream_view_),
      break_latest_d(0, stream_view_),
      break_duration_d(0, stream_view_),
      random_demand_d(0, stream_view_),
      vehicle_max_costs_d(0, stream_view_),
      vehicle_max_times_d(0, stream_view_),
      vehicle_fixed_costs_d(0, stream_view_),
      mixed_capacity_d(0, stream_view_),
      request_d(stream_view_),
      p_scores_d(0, stream_view_)
  {
  }

  void populate_device_vectors(generator::dataset_t<i_t, f_t>& dataset)
  {
    resize_device_vectors();
    auto& order_info = dataset.get_order_info();
    auto& fleet_info = dataset.get_fleet_info();

    raft::copy(matrices_d.buffer.data(),
               fleet_info.matrices_.buffer.data(),
               fleet_info.matrices_.buffer.size(),
               stream_view_);

    auto cost_matrix = matrices_d.get_cost_matrix(0);
    raft::copy(cost_matrix_d.data(), cost_matrix, n_locations * n_locations, stream_view_);

    rmm::device_uvector<i_t> demand_int(order_info.v_demand_.size(), stream_view_);
    rmm::device_uvector<i_t> caps_int(fleet_info.v_capacities_.size(), stream_view_);

    cuda::std::identity id;
    thrust::transform(handle_.get_thrust_policy(),
                      order_info.v_demand_.begin(),
                      order_info.v_demand_.end(),
                      demand_int.data(),
                      id);
    thrust::transform(handle_.get_thrust_policy(),
                      fleet_info.v_capacities_.begin(),
                      fleet_info.v_capacities_.end(),
                      caps_int.data(),
                      id);
    raft::copy(demand_d.data(), demand_int.data(), demand_int.size(), stream_view_);
    raft::copy(capacity_d.data(), caps_int.data(), caps_int.size(), stream_view_);

    // FIXME: Use structs in tests
    // Fill order info
    raft::copy(earliest_time_d.data(),
               order_info.v_earliest_time_.data(),
               order_info.v_earliest_time_.size(),
               stream_view_);
    raft::copy(latest_time_d.data(),
               order_info.v_latest_time_.data(),
               order_info.v_latest_time_.size(),
               stream_view_);

    // Fill fleet order constraints
    raft::copy(fleet_order_constraints_d.order_service_times.data(),
               fleet_info.fleet_order_constraints_.order_service_times.data(),
               n_vehicles * n_orders,
               stream_view_);
    raft::copy(service_time_d.data(),
               fleet_info.fleet_order_constraints_.order_service_times.data(),
               n_orders,
               stream_view_);
    // Fill fleet info
    raft::copy(
      vehicle_types_d.data(), fleet_info.v_types_.data(), fleet_info.v_types_.size(), stream_view_);

    raft::copy(bool_drop_return_trips_d.data(),
               fleet_info.v_drop_return_trip_.data(),
               fleet_info.v_drop_return_trip_.size(),
               stream_view_);

    thrust::transform(handle_.get_thrust_policy(),
                      bool_drop_return_trips_d.begin(),
                      bool_drop_return_trips_d.end(),
                      drop_return_trips_d.begin(),
                      id);

    raft::copy(bool_skip_first_trips_d.data(),
               fleet_info.v_skip_first_trip_.data(),
               fleet_info.v_skip_first_trip_.size(),
               stream_view_);

    raft::copy(vehicle_earliest_d.data(),
               fleet_info.v_earliest_time_.data(),
               fleet_info.v_earliest_time_.size(),
               stream_view_);
    raft::copy(vehicle_latest_d.data(),
               fleet_info.v_latest_time_.data(),
               fleet_info.v_latest_time_.size(),
               stream_view_);

    thrust::sequence(handle_.get_thrust_policy(),
                     break_locations_d.begin(),
                     break_locations_d.end(),
                     n_locations - n_break_locations_);

    auto& break_container = dataset.get_vehicle_breaks();
    for (i_t dim = 0; dim < n_break_dim_; ++dim) {
      auto& [break_earliest, break_latest, break_duration] = break_container[dim];
      raft::copy(break_earliest_d.data() + (dim * n_vehicles),
                 break_earliest.data(),
                 break_earliest.size(),
                 stream_view_);
      raft::copy(break_latest_d.data() + (dim * n_vehicles),
                 break_latest.data(),
                 break_latest.size(),
                 stream_view_);
      raft::copy(break_duration_d.data() + (dim * n_vehicles),
                 break_duration.data(),
                 break_duration.size(),
                 stream_view_);
    }
  }

  void resize_host_vectors()
  {
    // host
    x_h.resize(n_locations);
    y_h.resize(n_locations);
    matrices_h =
      detail::create_host_mdarray<f_t>(n_locations, n_vehicle_types_, 1 + use_secondary_matrix_);
    demand_h.resize(dim_ * n_orders);
    capacity_h.resize(dim_ * n_vehicles);
    earliest_time_h.resize(n_orders);
    latest_time_h.resize(n_orders);
    drop_return_trips_h.resize(n_vehicles);
    skip_first_trips_h.resize(n_vehicles);
    vehicle_types_h.resize(n_vehicles);
    vehicle_earliest_h.resize(n_vehicles);
    vehicle_latest_h.resize(n_vehicles);
    break_locations_h.resize(n_break_locations_);
    break_earliest_h.resize(n_vehicles * n_break_dim_);
    break_latest_h.resize(n_vehicles * n_break_dim_);
    break_duration_h.resize(n_vehicles * n_break_dim_);
    vehicle_max_costs_h.resize(n_vehicles);
    vehicle_max_times_h.resize(n_vehicles);
  }

  void resize_device_vectors()
  {
    // device
    x_d.resize(n_locations, stream_view_);
    y_d.resize(n_locations, stream_view_);
    matrices_d = detail::create_device_mdarray<f_t>(
      n_locations, n_vehicle_types_, 1 + use_secondary_matrix_, stream_view_);
    cost_matrix_d.resize(n_locations * n_locations, stream_view_);
    demand_d.resize(dim_ * n_orders, stream_view_);
    capacity_d.resize(dim_ * n_vehicles, stream_view_);
    earliest_time_d.resize(n_orders, stream_view_);
    latest_time_d.resize(n_orders, stream_view_);
    service_time_d.resize(n_orders, stream_view_);
    drop_return_trips_d.resize(n_vehicles, stream_view_);
    skip_first_trips_d.resize(n_vehicles, stream_view_);
    bool_drop_return_trips_d.resize(n_vehicles, stream_view_);
    bool_skip_first_trips_d.resize(n_vehicles, stream_view_);
    vehicle_types_d.resize(n_vehicles, stream_view_);
    vehicle_earliest_d.resize(n_vehicles, stream_view_);
    vehicle_latest_d.resize(n_vehicles, stream_view_);
    vehicle_max_costs_d.resize(n_vehicles, stream_view_);
    vehicle_max_times_d.resize(n_vehicles, stream_view_);
    n_pairs = (n_orders - 1) / 2;
    pickup_indices_d.resize(n_pairs, stream_view_);
    delivery_indices_d.resize(n_pairs, stream_view_);
    break_locations_d.resize(n_break_locations_, stream_view_);
    break_earliest_d.resize(n_vehicles * n_break_dim_, stream_view_);
    break_latest_d.resize(n_vehicles * n_break_dim_, stream_view_);
    break_duration_d.resize(n_vehicles * n_break_dim_, stream_view_);
    fleet_order_constraints_d.resize(n_vehicles, n_orders);
    vehicle_fixed_costs_d.resize(n_vehicles, stream_view_);
  }

  std::tuple<std::vector<f_t>, std::vector<f_t>> remove_nonunique_locations()
  {
    // Identify and remove non-unique locations and create a look up map from order to location
    std::vector<f_t> x_unique, y_unique;
    std::map<std::pair<f_t, f_t>, i_t> xy_unique;
    order_locations.resize(n_orders);
    for (i_t i = 0; i < n_orders; ++i) {
      auto xy = std::make_pair(x_h[i], y_h[i]);
      if (xy_unique.count(xy)) {
        order_locations[i] = xy_unique.at(xy);

      } else {
        order_locations[i] = x_unique.size();
        xy_unique[xy]      = x_unique.size();
        x_unique.push_back(x_h[i]);
        y_unique.push_back(y_h[i]);
      }
    }
    if (x_unique.size() == (size_t)n_orders) { order_locations.clear(); }
    return std::make_tuple(x_unique, y_unique);
  }

  void remove_zeroth_order()
  {
    n_orders--;
    order_locations.erase(order_locations.begin());

    // FIXME have to do this for every dimension
    demand_h.erase(demand_h.begin());

    earliest_time_h.erase(earliest_time_h.begin());
    latest_time_h.erase(latest_time_h.begin());
    service_time_h.erase(service_time_h.begin());
  }

  void populate_device_vectors()
  {
    resize_device_vectors();

    raft::copy(demand_d.data(), demand_h.data(), dim_ * n_orders, stream_view_);
    raft::copy(capacity_d.data(), capacity_h.data(), dim_ * n_vehicles, stream_view_);
    raft::copy(earliest_time_d.data(), earliest_time_h.data(), n_orders, stream_view_);
    raft::copy(latest_time_d.data(), latest_time_h.data(), n_orders, stream_view_);
    raft::copy(service_time_d.data(), service_time_h.data(), n_orders, stream_view_);

    for (int truck_id = 0; truck_id < n_vehicles; ++truck_id) {
      raft::copy(fleet_order_constraints_d.order_service_times.data() + truck_id * n_orders,
                 service_time_h.data(),
                 n_orders,
                 stream_view_);
    }

    if (matrices_h.buffer.empty()) {
      matrices_h =
        detail::create_host_mdarray<f_t>(n_locations, n_vehicle_types_, 1 + use_secondary_matrix_);
      auto cost_matrix_h = matrices_h.get_cost_matrix(0);
      build_dense_matrix(cost_matrix_h, x_h, y_h);
    }
    auto cost_matrix_h = matrices_h.get_cost_matrix(0);
    raft::copy(cost_matrix_d.data(), cost_matrix_h, n_locations * n_locations, stream_view_);

    if (request_h.x != -1) raft::copy(request_d.data(), &request_h, 1, stream_view_);
    if (!p_scores_h.empty()) p_scores_d = cuopt::device_copy(p_scores_h, stream_view_);
  }

  void populate_host_vectors()
  {
    resize_host_vectors();
    raft::copy(x_h.data(), x_d.data(), x_d.size(), stream_view_);
    raft::copy(y_h.data(), y_d.data(), y_d.size(), stream_view_);
    raft::copy(
      matrices_h.buffer.data(), matrices_d.buffer.data(), matrices_d.buffer.size(), stream_view_);
    raft::copy(demand_h.data(), demand_d.data(), dim_ * n_orders, stream_view_);
    raft::copy(capacity_h.data(), capacity_d.data(), dim_ * n_vehicles, stream_view_);
    raft::copy(
      earliest_time_h.data(), earliest_time_d.data(), earliest_time_d.size(), stream_view_);
    raft::copy(latest_time_h.data(), latest_time_d.data(), latest_time_d.size(), stream_view_);
    raft::copy(drop_return_trips_h.data(), drop_return_trips_d.data(), n_vehicles, stream_view_);
    raft::copy(skip_first_trips_h.data(), skip_first_trips_d.data(), n_vehicles, stream_view_);
    raft::copy(vehicle_types_h.data(), vehicle_types_d.data(), n_vehicles, stream_view_);
    raft::copy(vehicle_earliest_h.data(), vehicle_earliest_d.data(), n_vehicles, stream_view_);
    raft::copy(vehicle_latest_h.data(), vehicle_latest_d.data(), n_vehicles, stream_view_);
    raft::copy(
      break_earliest_h.data(), break_earliest_d.data(), break_earliest_d.size(), stream_view_);
    raft::copy(break_latest_h.data(), break_latest_d.data(), break_latest_d.size(), stream_view_);
    raft::copy(
      break_duration_h.data(), break_duration_d.data(), break_duration_d.size(), stream_view_);
    raft::copy(vehicle_max_costs_h.data(),
               vehicle_max_costs_d.data(),
               vehicle_max_costs_d.size(),
               stream_view_);
    raft::copy(vehicle_max_times_h.data(),
               vehicle_max_times_d.data(),
               vehicle_max_times_d.size(),
               stream_view_);
    fleet_order_constraints_h = fleet_order_constraints_d.to_host();
  }

  void check_time_windows(host_assignment_t<i_t> const& routing_solution, bool is_soft_tw = false)
  {
    auto route                = routing_solution.route;
    auto out_stamp            = routing_solution.stamp;
    auto truck_id             = routing_solution.truck_id;
    auto locations            = routing_solution.locations;
    auto node_types           = routing_solution.node_types;
    fleet_order_constraints_h = fleet_order_constraints_d.to_host();

    std::vector<i_t> temp_truck_ids(truck_id);
    auto end_it = std::unique(temp_truck_ids.begin(), temp_truck_ids.end());
    temp_truck_ids.resize(std::distance(temp_truck_ids.begin(), end_it));
    size_t i = 0;
    size_t j = 0;

    // Recompute arrival times for each route
    for (auto const& id : temp_truck_ids) {
      i_t break_dim = -1;
      std::vector<double> arrival_stamp;
      std::vector<double> latest_stamp;
      auto order     = route[i];
      auto order_loc = locations[i];
      auto node_type = (node_type_t)node_types[i];

      double vehicle_earliest = vehicle_tw_ ? vehicle_earliest_h[id] : 0.;
      double vehicle_latest =
        vehicle_tw_ ? vehicle_latest_h[id] : std::numeric_limits<int32_t>::max();
      auto vehicle_service_time_h =
        fleet_order_constraints_h.order_service_times.data() + id * n_orders;

      double depot_earliest = vehicle_earliest;
      double depot_latest   = vehicle_latest;
      bool depot_included   = this->order_locations_d.size() == 0;
      if (depot_included) {
        depot_earliest = std::max(depot_earliest, (double)earliest_time_h[0]);
        depot_latest   = std::min(depot_latest, (double)latest_time_h[0]);
      }
      auto vehicle_type     = vehicle_types_h[id];
      auto transit_matrix_h = matrices_h.get_time_matrix(vehicle_type);

      bool is_depot = node_type == node_type_t::DEPOT;

      double earliest =
        is_depot ? depot_earliest : std::max(vehicle_earliest, (double)earliest_time_h[order]);
      double latest =
        is_depot ? depot_latest : std::min(vehicle_latest, (double)latest_time_h[order]);

      // route does not have to start from the vehicle earliest time. It just has to start before
      // latest time to lower wait time
      ASSERT_LE(earliest, out_stamp[i]);
      ASSERT_LE(out_stamp[i], latest);
      double stamp = out_stamp[i];

      arrival_stamp.push_back(stamp);
      latest_stamp.push_back(latest);

      ++i;
      ++j;
      for (; i < route.size() && j < truck_id.size() && truck_id[j] == id; ++i, ++j) {
        auto new_order           = route[i];
        auto new_order_loc       = locations[i];
        auto new_node_type       = (node_type_t)node_types[i];
        auto curr_is_break_order = node_type == node_type_t::BREAK;
        auto curr_is_depot       = node_type == node_type_t::DEPOT;
        if (curr_is_break_order) ++break_dim;
        auto next_is_break_order = new_node_type == node_type_t::BREAK;
        auto next_is_depot       = new_node_type == node_type_t::DEPOT;
        double transit           = transit_matrix_h[order_loc * n_locations + new_order_loc];

        // For tests we assume break dimensions come in order
        double curr_service  = curr_is_break_order
                                 ? break_duration_h[break_dim * n_vehicles + id]
                                 : (curr_is_depot ? 0. : vehicle_service_time_h[order]);
        double order_arrival = stamp + transit + curr_service;
        double order_earliest =
          next_is_break_order
            ? break_earliest_h[(break_dim + 1) * n_vehicles + id]
            : (next_is_depot ? depot_earliest : static_cast<double>(earliest_time_h[new_order]));
        double order_latest = next_is_break_order
                                ? break_latest_h[(break_dim + 1) * n_vehicles + id]
                                : (next_is_depot ? depot_latest : latest_time_h[new_order]);
        double curr_wait    = std::max(0.0, order_earliest - order_arrival);
        order_arrival += curr_wait;

        arrival_stamp.push_back(order_arrival);
        latest_stamp.push_back(order_latest);

        order     = new_order;
        order_loc = new_order_loc;
        stamp     = order_arrival;
        node_type = new_node_type;

        ASSERT_LT(abs(out_stamp[i] - stamp), 0.1);
      }
      for (size_t k = 0; k < arrival_stamp.size() - 1; ++k)
        ASSERT_LE(arrival_stamp[k], arrival_stamp[k + 1]);
      if (!is_soft_tw) {
        for (size_t k = 0; k < latest_stamp.size(); ++k)
          ASSERT_LE(arrival_stamp[k], latest_stamp[k] + 1e-3);
      }
    }
  }

  void check_vehicle_breaks(host_assignment_t<i_t> const& h_routing_solution)
  {
    auto truck_id     = h_routing_solution.truck_id;
    auto node_types   = h_routing_solution.node_types;
    i_t curr_truck_id = -1;
    i_t break_dim     = n_break_dim_;
    for (size_t i = 0; i < truck_id.size(); i++) {
      if (truck_id[i] != curr_truck_id) {
        ASSERT_EQ(break_dim, n_break_dim_);
        curr_truck_id = truck_id[i];
        break_dim     = 0;
      }
      if (node_types[i] == (i_t)node_type_t::BREAK) ++break_dim;
    }
    ASSERT_EQ(break_dim, n_break_dim_);
  }

  void check_capacity(host_assignment_t<i_t> const& routing_solution,
                      const std::vector<i_t>& h_demand,
                      const std::vector<i_t>& h_capacity,
                      const rmm::device_uvector<i_t>& d_demand)
  {
    auto route      = routing_solution.route;
    auto truck_id   = routing_solution.truck_id;
    auto node_types = routing_solution.node_types;

    // std::set orders the truck ids, std::unordered_set keeps it at random order
    // we need to keep insertion order and unique
    std::vector<i_t> temp_truck_ids(truck_id);
    auto end_it = std::unique(temp_truck_ids.begin(), temp_truck_ids.end());
    temp_truck_ids.resize(std::distance(temp_truck_ids.begin(), end_it));
    size_t i = 0;
    size_t j = 0;

    auto global_demand = 0;
    for (auto const& id : temp_truck_ids) {
      auto curr_demand = 0;
      for (; i < route.size() && j < truck_id.size() && truck_id[j] == id; ++i, ++j) {
        auto node_type = (node_type_t)node_types[i];
        if (node_type != node_type_t::BREAK && node_type != node_type_t::DEPOT)
          curr_demand += h_demand[route[i]];
      }
      global_demand += curr_demand;
      // We do not exceed truck capacity
      ASSERT_LE(curr_demand, h_capacity[id]);
    }
    auto sum = thrust::reduce(handle_.get_thrust_policy(), d_demand.begin(), d_demand.end());
    if (!pickup_delivery_) {
      // All demand is fulfilled
      ASSERT_EQ(global_demand, sum);
    }
  }

  void check_vehicle_time_windows(host_assignment_t<int> const& h_routing_solution,
                                  const std::vector<i_t>& vehicle_earliest,
                                  const std::vector<i_t>& vehicle_latest)
  {
    auto truck_id        = h_routing_solution.truck_id;
    auto h_arrival_stamp = h_routing_solution.stamp;
    i_t curr_truck_id    = -1;
    for (size_t i = 0; i < truck_id.size(); i++) {
      if (truck_id[i] != curr_truck_id) { curr_truck_id = truck_id[i]; }
      ASSERT_LE(vehicle_earliest[curr_truck_id], h_arrival_stamp[i]);
      ASSERT_GE(vehicle_latest[curr_truck_id], h_arrival_stamp[i]);
    }
  }

  void check_cost(assignment_t<i_t>& routing_solution)
  {
    auto final_vn   = routing_solution.get_vehicle_count();
    auto final_cost = routing_solution.get_total_objective();

    // Vehicle number check
    std::cout << "Ref vehicle number: " << ref_vn << " Final vehicle number: " << final_vn << "\n";
    auto vehicle_err = abs(final_vn - ref_vn);
    EXPECT_LE(vehicle_err, vn_tol_);
    std::cout << "Vehicle mismatch: " << vehicle_err << "\n";

    // Cost check
    std::cout << "Ref cost: " << ref_cost << " Final cost: " << final_cost << "\n";
    auto cost_err = final_cost - ref_cost;
    cost_err /= ref_cost;
    std::cout << "Cost approximation error: " << cost_err * 100 << "%\n";
    EXPECT_LE(cost_err, cost_tol_);
  }

  assignment_t<i_t> solve(
    data_model_view_t<i_t, f_t> const& data_model,
    solver_settings_t<i_t, f_t> const settings = solver_settings_t<i_t, f_t>{})
  {
    handle_.sync_stream();
    hr_timer_.start("routing solver");
    auto routing_solution = cuopt::routing::solve(data_model, settings);
    handle_.sync_stream();
    if (handle_.comms_initialized()) handle_.get_comms().barrier();
    hr_timer_.stop();
    hr_timer_.display(std::cout);
    return routing_solution;
  }

  raft::handle_t handle_;
  rmm::cuda_stream_view stream_view_;
  HighResTimer hr_timer_;

  bool multi_capacity_{false};
  bool vehicle_max_costs_{false};
  bool vehicle_max_times_{false};
  bool vehicle_fixed_costs_{false};
  i_t n_vehicle_types_{1};
  i_t limit_;
  bool vehicle_tw_{false};
  bool use_secondary_matrix_{false};
  i_t dim_{0};
  i_t n_break_dim_{0};
  i_t n_orders{0};
  i_t n_pairs_{0};
  i_t n_break_locations_{0};
  bool vehicle_breaks_{false};
  bool pickup_delivery_{false};

  int vehicle_lower_bound_;
  bool drop_return_trip_;

  rmm::device_uvector<i_t> random_demand_d;
  rmm::device_uvector<i_t> mixed_capacity_d;

  f_t ref_cost;
  i_t ref_vn{-1};
  bool regression_check{false};
  i_t vn_tol_;
  f_t cost_tol_;

  i_t n_locations;
  i_t n_vehicles;
  i_t n_pairs;

  // host arrays
  std::vector<f_t> x_h;
  std::vector<f_t> y_h;
  h_mdarray_t<f_t> matrices_h;
  typename detail::fleet_order_constraints_t<i_t>::host_t fleet_order_constraints_h;
  std::vector<i_t> order_locations;
  std::vector<i_t> demand_h;
  std::vector<i_t> capacity_h;
  std::vector<i_t> earliest_time_h;
  std::vector<i_t> latest_time_h;
  std::vector<i_t> service_time_h;
  std::vector<i_t> drop_return_trips_h;
  std::vector<i_t> skip_first_trips_h;
  std::vector<uint8_t> vehicle_types_h;
  std::vector<i_t> vehicle_earliest_h;
  std::vector<i_t> vehicle_latest_h;
  std::vector<i_t> pickup_indices_h;
  std::vector<i_t> delivery_indices_h;
  std::vector<i_t> break_locations_h;
  std::vector<i_t> break_earliest_h;
  std::vector<i_t> break_latest_h;
  std::vector<i_t> break_duration_h;
  std::vector<f_t> vehicle_max_costs_h;
  std::vector<f_t> vehicle_max_times_h;
  std::vector<f_t> vehicle_fixed_costs_h;
  int2 request_h{-1, -1};
  std::vector<int> p_scores_h;
  int4 expected_found_sol;
  std::vector<i_t> expected_route_h;

  // device arrays
  rmm::device_uvector<f_t> cost_matrix_d;
  d_mdarray_t<f_t> matrices_d;
  detail::fleet_order_constraints_t<i_t> fleet_order_constraints_d;
  rmm::device_uvector<f_t> x_d;
  rmm::device_uvector<f_t> y_d;
  rmm::device_uvector<i_t> order_locations_d;
  rmm::device_uvector<i_t> capacity_d;
  rmm::device_uvector<i_t> demand_d;
  rmm::device_uvector<i_t> earliest_time_d;
  rmm::device_uvector<i_t> latest_time_d;
  rmm::device_uvector<i_t> service_time_d;
  rmm::device_uvector<i_t> drop_return_trips_d;
  rmm::device_uvector<i_t> skip_first_trips_d;
  rmm::device_uvector<bool> bool_drop_return_trips_d;
  rmm::device_uvector<bool> bool_skip_first_trips_d;
  rmm::device_uvector<uint8_t> vehicle_types_d;
  rmm::device_uvector<i_t> vehicle_earliest_d;
  rmm::device_uvector<i_t> vehicle_latest_d;
  rmm::device_uvector<i_t> pickup_indices_d;
  rmm::device_uvector<i_t> delivery_indices_d;
  rmm::device_uvector<i_t> break_locations_d;
  rmm::device_uvector<i_t> break_earliest_d;
  rmm::device_uvector<i_t> break_latest_d;
  rmm::device_uvector<i_t> break_duration_d;
  rmm::device_uvector<f_t> vehicle_max_costs_d;
  rmm::device_uvector<f_t> vehicle_max_times_d;
  rmm::device_uvector<f_t> vehicle_fixed_costs_d;
  rmm::device_scalar<int2> request_d;
  rmm::device_uvector<int> p_scores_d;
};

template <typename i_t, typename f_t>
class routing_test_t : public base_test_t<i_t, f_t> {
 public:
  routing_test_t(f_t cost_tol = 1E-2, i_t vn_tol = 0, i_t limit = 1001)
    : base_test_t<i_t, f_t>(cost_tol, vn_tol, limit)
  {
  }

  void write_ref(std::string const& out_file, assignment_t<i_t> const& routing_solution)
  {
    std::ofstream out_stream(out_file, std::ios_base::app);
    if (!out_stream.is_open()) cuopt_assert(false, "Output ref file");
    auto root = cuopt::test::get_rapids_dataset_root_dir();
    out_stream << input_file_.substr(root.size() + 1) << ","
               << routing_solution.get_total_objective() << ","
               << routing_solution.get_vehicle_count() << "\n";
  }

  void test_tsp()
  {
    // data model
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, 1, this->n_orders);
    data_model.add_cost_matrix(this->cost_matrix_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(data_model.get_num_locations() / 5);

    // solve
    auto routing_solution = this->solve(data_model, settings);

    // checks
    if (this->regression_check) this->check_cost(routing_solution);
    ASSERT_EQ(routing_solution.get_vehicle_count(), 1);
    EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    if (generate_ref) { write_ref("l1_tsp.txt", routing_solution); }
  }

  void test_vrp()
  {
    // data model
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    data_model.add_cost_matrix(this->cost_matrix_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(data_model.get_num_locations() / 5);

    // solve
    auto routing_solution = this->solve(data_model, settings);

    host_assignment_t<i_t> h_routing_solution(routing_solution);
    // checks
    check_route(data_model, h_routing_solution);
    EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  }

  void test_cvrp()
  {
    // data model
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    data_model.add_cost_matrix(this->cost_matrix_d.data());

    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(data_model.get_num_locations() / 5);

    // solve
    auto routing_solution = this->solve(data_model, settings);

    host_assignment_t<i_t> h_routing_solution(routing_solution);
    // checks
    check_route(data_model, h_routing_solution);
    this->check_capacity(h_routing_solution, this->demand_h, this->capacity_h, this->demand_d);
    EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  }

  void test_acvrp()
  {
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);
    data_model.add_cost_matrix(this->cost_matrix_d.data());
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(this->n_orders / 5);

    // solve
    auto routing_solution = this->solve(data_model, settings);

    host_assignment_t<i_t> h_routing_solution(routing_solution);

    // checks
    if (this->regression_check) this->check_cost(routing_solution);
    check_route(data_model, h_routing_solution);
    this->check_capacity(h_routing_solution, this->demand_h, this->capacity_h, this->demand_d);
    EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  }

  void test_vrptw()
  {
    // data model
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    data_model.add_cost_matrix(this->cost_matrix_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(this->n_orders / 5);

    // solve
    auto routing_solution = this->solve(data_model, settings);

    host_assignment_t<i_t> h_routing_solution(routing_solution);

    // checks
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution);
    EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  }

  void test_cvrptw()
  {
    // data model
    // for solver v2, set fleet size to ref_vn
    auto start_vehicle = this->regression_check ? this->ref_vn : this->n_vehicles;
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, start_vehicle, this->n_orders);

    data_model.add_cost_matrix(this->cost_matrix_d.data());
    if (this->order_locations_d.size()) {
      data_model.set_order_locations(this->order_locations_d.data());
    }

    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    if (this->pickup_indices_h.size()) {
      i_t n_pairs = (this->n_orders - 1) / 2;
      this->pickup_indices_d.resize(n_pairs, this->stream_view_);
      this->delivery_indices_d.resize(n_pairs, this->stream_view_);
      raft::copy(
        this->pickup_indices_d.data(), this->pickup_indices_h.data(), n_pairs, this->stream_view_);
      raft::copy(this->delivery_indices_d.data(),
                 this->delivery_indices_h.data(),
                 n_pairs,
                 this->stream_view_);
      data_model.set_pickup_delivery_pairs(this->pickup_indices_d.data(),
                                           this->delivery_indices_d.data());
    }

    // set min vehicles to ref values in the ref file
    // TODO change this later, when we have a unified API for desired_n_vehicles
    if (this->regression_check) { data_model.set_min_vehicles(this->ref_vn); }

    // if there is out of memory error with pickup and deliveries reduce climber count
    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(this->n_orders / 5);

#ifdef VERBOSE_TEST
    auto path = std::filesystem::current_path();
    path += "/test_dump.csv";
    solver.dump_best_results(path, 10);
    std::cout << this->demand_d.size() << std::endl;
    std::cout << this->earliest_time_d.size() << std::endl;
    std::cout << this->latest_time_d.size() << std::endl;
    std::cout << this->service_time_d.size() << std::endl;
    std::cout << this->n_locations << std::endl;
    raft::print_device_vector("x:", this->x_d.data(), this->x_d.size(), std::cout);
    raft::print_device_vector("y:", this->y_d.data(), this->y_d.size(), std::cout);
    raft::print_device_vector("demand:", this->demand_d.data(), this->demand_d.size(), std::cout);
    raft::print_device_vector(
      "capacity:", this->capacity_d.data(), this->capacity_d.size(), std::cout);
    raft::print_device_vector(
      "earliest:", earliest_time_d.data(), earliest_time_d.size(), std::cout);
    raft::print_device_vector(
      "latest:", this->latest_time_d.data(), this->latest_time_d.size(), std::cout);
    raft::print_device_vector(
      "service:", this->service_time_d.data(), this->service_time_d.size(), std::cout);
#endif

    // solve
    auto routing_solution = this->solve(data_model, settings);
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

    host_assignment_t<i_t> h_routing_solution(routing_solution);
    // checks
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution, false);
    this->check_capacity(h_routing_solution, this->demand_h, this->capacity_h, this->demand_d);
    if (this->regression_check) this->check_cost(routing_solution);
    std::cout << routing_solution.get_status_string() << std::endl;

    if (generate_ref) {
      auto filename = this->n_locations > 101 ? "homberger" : std::to_string(this->n_locations - 1);
      write_ref("l1_" + filename + ".txt", routing_solution);
    }
  }

 protected:
  std::string input_file_;
};

template <typename i_t, typename f_t>
class regression_routing_test_t : public routing_test_t<i_t, f_t>,
                                  public ::testing::TestWithParam<file_params> {
 public:
  typedef routing_test_t<i_t, f_t> super;
  regression_routing_test_t() : routing_test_t<i_t, f_t>() {}
  regression_routing_test_t(f_t cost_tol, i_t vn_tol, i_t limit = 1001)
    : routing_test_t<i_t, f_t>(cost_tol, vn_tol, limit)
  {
  }
  void SetUp() override
  {
    auto param             = GetParam();
    this->ref_cost         = param.ref_cost;
    this->ref_vn           = param.ref_vn;
    this->regression_check = true;
    auto input             = load_routing_file<i_t, f_t>(param.routing_file, this->limit_);

    this->n_locations        = input.n_locations;
    this->n_vehicles         = input.n_vehicles;
    this->n_orders           = this->n_locations;
    this->x_h                = input.x_h;
    this->y_h                = input.y_h;
    this->demand_h           = input.demand_h;
    this->capacity_h         = input.capacity_h;
    this->earliest_time_h    = input.earliest_time_h;
    this->latest_time_h      = input.latest_time_h;
    this->service_time_h     = input.service_time_h;
    this->pickup_indices_h   = input.pickup_indices_h;
    this->delivery_indices_h = input.delivery_indices_h;
    this->vehicle_types_h    = std::vector<uint8_t>(input.n_vehicles, 0);
    this->populate_device_vectors();
  }
};

typedef regression_routing_test_t<int, float> float_regression_test_t;

// add 1 as depot node to the limit
class regression_routing_test_tsp_t : public float_regression_test_t {
 public:
  regression_routing_test_tsp_t() : float_regression_test_t(1E-2, 0) {}
};

class regression_routing_test_cvrp_t : public float_regression_test_t {
 public:
  regression_routing_test_cvrp_t() : float_regression_test_t(1E-1, 2) {}
};

class regression_routing_test_acvrp_t : public float_regression_test_t {
 public:
  regression_routing_test_acvrp_t() : float_regression_test_t(1E-1, 2) {}
};

class regression_routing_test_25_t : public float_regression_test_t {
 public:
  regression_routing_test_25_t() : float_regression_test_t(1E-2, 2, 26) {}
};

class regression_routing_test_50_t : public float_regression_test_t {
 public:
  regression_routing_test_50_t() : float_regression_test_t(1E-2, 2, 51) {}
};

class regression_routing_test_100_t : public float_regression_test_t {
 public:
  regression_routing_test_100_t() : float_regression_test_t(1E-2, 2, 101) {}
};

class regression_routing_test_pickup_t : public float_regression_test_t {
 public:
  // for now we keep the error margin very big since there are no improvement kernels
  regression_routing_test_pickup_t() : float_regression_test_t(110E-2, 23) {}
};

class regression_routing_test_dummy : public float_regression_test_t {
 public:
  regression_routing_test_dummy() : float_regression_test_t(1E-1, 2, 201) {}
};

class float_pickup_regression_test_t : public float_regression_test_t {
 public:
  float_pickup_regression_test_t() : float_regression_test_t(200E-2, 30) {}
};

}  // namespace test
}  // namespace routing
}  // namespace cuopt
