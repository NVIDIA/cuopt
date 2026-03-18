/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/utilities/test_utilities.hpp>
#include "check_constraints.hpp"

#include <utilities/copy_helpers.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <unordered_set>

namespace cuopt {
namespace routing {
namespace test {

bool is_special_node(const node_type_t& node_type) { return node_type == node_type_t::BREAK; }

template <typename i_t, typename f_t>
void check_route(data_model_view_t<i_t, f_t> const& data_model,
                 host_assignment_t<i_t> const& h_routing_solution)
{
  auto const& route      = h_routing_solution.route;
  auto const& node_types = h_routing_solution.node_types;
  auto const& truck_id   = h_routing_solution.truck_id;
  auto const& locations  = h_routing_solution.locations;

  auto n_locations = data_model.get_num_locations();
  auto n_orders    = data_model.get_num_orders();

  auto handle_ptr = data_model.get_handle_ptr();
  auto stream     = handle_ptr->get_stream();

  std::map<uint8_t, std::vector<f_t>> cost_matrices_h;
  std::map<uint8_t, std::vector<f_t>> time_matrices_h;
  auto fleet_size = data_model.get_fleet_size();

  std::vector<uint8_t> vehicle_types_h(fleet_size, 0);
  if (auto vehicle_types = data_model.get_vehicle_types(); vehicle_types.size() > 0) {
    vehicle_types_h = cuopt::host_copy(vehicle_types, stream);
  }

  for (auto& type : vehicle_types_h) {
    int sz = std::pow(data_model.get_num_locations(), 2);
    if (cost_matrices_h.count(type) == 0) {
      auto matrix = data_model.get_cost_matrix(type);
      cost_matrices_h.emplace(type, cuopt::host_copy(matrix, sz, stream));
    }
  }

  for (auto& type : vehicle_types_h) {
    int sz = std::pow(data_model.get_num_locations(), 2);
    if (time_matrices_h.count(type) == 0) {
      auto matrix = data_model.get_transit_time_matrix(type);
      if (matrix) time_matrices_h.emplace(type, cuopt::host_copy(matrix, sz, stream));
    }
  }

  auto vehicle_max_costs = data_model.get_vehicle_max_costs();
  auto vehicle_max_costs_h =
    cuopt::host_copy(vehicle_max_costs.data(), vehicle_max_costs.size(), stream);

  auto vehicle_max_times = data_model.get_vehicle_max_times();
  auto vehicle_max_times_h =
    cuopt::host_copy(vehicle_max_times.data(), vehicle_max_times.size(), stream);

  auto drop_return_trip_h =
    cuopt::host_copy(data_model.get_drop_return_trips(), fleet_size, stream);
  auto skip_first_trip_h = cuopt::host_copy(data_model.get_skip_first_trips(), fleet_size, stream);

  auto [vehicle_start_loc, vehicle_return_loc] = data_model.get_vehicle_locations();
  std::vector<i_t> vehicle_start_locations_h(fleet_size, 0),
    vehicle_return_locations_h(fleet_size, 0);
  if (vehicle_start_loc != nullptr) {
    vehicle_start_locations_h  = cuopt::host_copy(vehicle_start_loc, fleet_size, stream);
    vehicle_return_locations_h = cuopt::host_copy(vehicle_return_loc, fleet_size, stream);
  }

  auto vehicle_order_match_d = data_model.get_vehicle_order_match();
  std::unordered_map<i_t, std::unordered_set<i_t>> vehicle_order_match_h;
  for (const auto& [vehicle_id, orders] : vehicle_order_match_d) {
    auto orders_h                     = cuopt::host_copy(orders, stream);
    vehicle_order_match_h[vehicle_id] = std::unordered_set<i_t>(orders_h.begin(), orders_h.end());
  }

  // std::set orders the truck ids, std::unordered_set keeps it at random order
  // we need to keep insertion order and unique
  std::vector<i_t> temp_truck_ids(truck_id);
  auto end_it = std::unique(temp_truck_ids.begin(), temp_truck_ids.end());
  temp_truck_ids.resize(std::distance(temp_truck_ids.begin(), end_it));
  bool is_drop_return = !drop_return_trip_h.empty();
  bool is_skip_first  = !skip_first_trip_h.empty();

  size_t i = 0;
  size_t j = 0;
  std::set<i_t> visited{};

  if (data_model.get_order_locations() == nullptr) { visited.insert(0); }

  bool has_breaks = data_model.has_vehicle_breaks();
  std::vector<std::vector<i_t>> uniform_break_earliest_h, uniform_break_latest_h;
  std::unordered_set<i_t> uniform_break_locations_set;
  if (has_breaks) {
    auto const& uniform = data_model.get_uniform_breaks();
    for (auto const& dim : uniform) {
      auto [e_ptr, l_ptr, d_ptr] = dim.get_breaks();
      uniform_break_earliest_h.push_back(
        cuopt::host_copy(e_ptr, static_cast<size_t>(fleet_size), stream));
      uniform_break_latest_h.push_back(
        cuopt::host_copy(l_ptr, static_cast<size_t>(fleet_size), stream));
    }
    auto [break_loc_ptr, n_break_loc] = data_model.get_break_locations();
    if (n_break_loc > 0) {
      auto break_locations_h =
        cuopt::host_copy(break_loc_ptr, static_cast<size_t>(n_break_loc), stream);
      uniform_break_locations_set.insert(break_locations_h.begin(), break_locations_h.end());
    }
  }

  for (auto const& id : temp_truck_ids) {
    size_t i_vehicle_start = i;
    std::vector<i_t> path, path_locations;
    f_t route_dist     = 0;
    f_t route_time     = 0.f;
    f_t max_cost_truck = !vehicle_max_costs_h.empty() ? vehicle_max_costs_h[id] : -1;
    f_t max_time_truck = !vehicle_max_times_h.empty() ? vehicle_max_times_h[id] : -1;

    const auto& cost_matrix_h = cost_matrices_h.at(vehicle_types_h[id]);
    const auto& time_matrix_h =
      time_matrices_h.empty() || time_matrices_h.find(vehicle_types_h[id]) == time_matrices_h.end()
        ? cost_matrix_h
        : time_matrices_h.at(vehicle_types_h[id]);

    const auto& possible_orders = vehicle_order_match_h.count(id) > 0 ? vehicle_order_match_h.at(id)
                                                                      : std::unordered_set<int>{};

    for (; i < route.size() && j < truck_id.size() && truck_id[j] == id; ++i, ++j) {
      ASSERT_LT(locations[i], n_locations);
      ASSERT_GE(locations[i], 0);

      // insert only order locations
      if (auto order = route[i]; !is_special_node((node_type_t)node_types[i])) {
        path.push_back(order);
        path_locations.push_back(locations[i]);
        if (!possible_orders.empty()) { EXPECT_EQ(possible_orders.count(order), 1u); }
      }
      if (j + 1 < truck_id.size() && truck_id[j + 1] == id) {
        route_dist += cost_matrix_h[locations[i] * n_locations + locations[i + 1]];
        route_time += time_matrix_h[locations[i] * n_locations + locations[i + 1]];
      }
    }

    if (has_breaks) {
      int break_dim           = 0;
      auto const& non_uniform = data_model.get_non_uniform_breaks();
      bool use_uniform        = !uniform_break_earliest_h.empty();
      bool use_non_uniform    = (non_uniform.count(id) > 0);
      for (size_t k = i_vehicle_start; k < i; ++k) {
        if (static_cast<node_type_t>(node_types[k]) == node_type_t::BREAK) {
          double arrival   = h_routing_solution.stamp[k];
          i_t break_loc_id = locations[k];
          if (use_uniform && break_dim < static_cast<int>(uniform_break_earliest_h.size())) {
            // std::cout<<"VEHID: "<<id<<" ARRIVAL_REAL: "<<arrival<<" EARLIEST:
            // "<<uniform_break_earliest_h[break_dim][id]<<" LATEST:
            // "<<uniform_break_latest_h[break_dim][id]<<"\n";
            ASSERT_GE(arrival, static_cast<double>(uniform_break_earliest_h[break_dim][id]) - 1e-6)
              << "Break " << break_dim << " vehicle " << id << " arrival " << arrival
              << " before earliest " << uniform_break_earliest_h[break_dim][id];
            ASSERT_LE(arrival, static_cast<double>(uniform_break_latest_h[break_dim][id]) + 1e-6)
              << "Break " << break_dim << " vehicle " << id << " arrival " << arrival
              << " after latest " << uniform_break_latest_h[break_dim][id];
            if (!uniform_break_locations_set.empty()) {
              ASSERT_EQ(uniform_break_locations_set.count(break_loc_id), 1u)
                << "Break " << break_dim << " vehicle " << id << " at location " << break_loc_id
                << " not in allowed break locations";
            }
          } else if (use_non_uniform) {
            auto const& breaks = non_uniform.at(id);
            if (break_dim < static_cast<int>(breaks.size())) {
              auto const& b = breaks[break_dim];
              ASSERT_GE(arrival, static_cast<double>(b.earliest_) - 1e-6)
                << "Non-uniform break " << break_dim << " vehicle " << id;
              ASSERT_LE(arrival, static_cast<double>(b.latest_) + 1e-6)
                << "Non-uniform break " << break_dim << " vehicle " << id;
              if (b.locations_.size() > 0) {
                auto allowed_locs = cuopt::host_copy(b.locations_, stream);
                bool found = std::find(allowed_locs.begin(), allowed_locs.end(), break_loc_id) !=
                             allowed_locs.end();
                ASSERT_TRUE(found)
                  << "Non-uniform break " << break_dim << " vehicle " << id << " at location "
                  << break_loc_id << " not in allowed break locations";
              }
            }
          }
          ++break_dim;
        }
      }
      if (use_uniform) {
        ASSERT_EQ(break_dim, static_cast<int>(uniform_break_earliest_h.size()))
          << "Vehicle " << id << " break count " << break_dim << " expected "
          << uniform_break_earliest_h.size();
      } else if (use_non_uniform) {
        ASSERT_EQ(break_dim, static_cast<int>(non_uniform.at(id).size()))
          << "Vehicle " << id << " non-uniform break count " << break_dim << " expected "
          << non_uniform.at(id).size();
      }
    }

    // Check for a case when user indicates that this vehicle can not carry any orders
    if (possible_orders.empty() && vehicle_order_match_h.count(id) > 0) {
      EXPECT_EQ(path.size(), 0u);
    }

    auto end   = path.end();
    auto begin = path.begin();
    if (is_drop_return) {
      if (!drop_return_trip_h[id]) {
        EXPECT_EQ(path_locations[path_locations.size() - 1], vehicle_return_locations_h[id]);
        --end;
      }
    } else {
      EXPECT_EQ(path_locations[path_locations.size() - 1], vehicle_return_locations_h[id]);
      --end;
    }

    if (is_skip_first) {
      if (!skip_first_trip_h[id]) {
        EXPECT_EQ(path_locations[0], vehicle_start_locations_h[id]);
        ++begin;
      }
    } else {
      EXPECT_EQ(path_locations[0], vehicle_start_locations_h[id]);
      ++begin;
    }

    // Duplicate check
    auto has_duplicates = std::unique(begin, end) != end;
    ASSERT_EQ(has_duplicates, false);
    if (!vehicle_max_costs_h.empty()) ASSERT_LE(route_dist, max_cost_truck + 0.001);
    if (!vehicle_max_times_h.empty()) ASSERT_LE(route_time, max_time_truck + 0.001);

    // Each truck visits its owns set of vertices
    std::vector<i_t> inter;
    std::set<i_t> set_path(begin, end);
    std::set_intersection(
      visited.begin(), visited.end(), set_path.begin(), set_path.end(), std::back_inserter(inter));

    ASSERT_EQ(inter.size(), 0);
    visited.insert(set_path.begin(), set_path.end());
  }

  if (auto prizes = data_model.get_order_prizes(); prizes.empty()) {
    // We visited every vertex
    ASSERT_EQ(visited.size(), n_orders);
  } else {
    ASSERT_LE(visited.size(), n_orders);
  }
}

template <typename i_t, typename f_t>
void check_route(data_model_view_t<i_t, f_t> const& data_model,
                 assignment_t<i_t> const& routing_solution)
{
  auto h_routing_solution = host_assignment_t<i_t>(routing_solution);
  check_route(data_model, h_routing_solution);
}

template void check_route<int, float>(data_model_view_t<int, float> const& data_model,
                                      host_assignment_t<int> const& h_routing_solution);

template void check_route<int, float>(data_model_view_t<int, float> const& data_model,
                                      assignment_t<int> const& routing_solution);

}  // namespace test
}  // namespace routing
}  // namespace cuopt
