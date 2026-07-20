/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/macros.cuh>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

struct fsmvrptwsc_instance_t {
  std::string name;
  int n_clients{};
  int n_vehicle_types{};
  int n_distance_ranges{};
  int n_vehicles{};
  std::vector<float> distance_matrix;
  std::vector<float> transit_time_matrix;
  std::vector<int> order_locations;
  std::vector<int> earliest;
  std::vector<int> latest;
  std::vector<int> service_times;
  std::vector<int> demand;
  std::vector<int> vehicle_earliest;
  std::vector<int> vehicle_latest;
  std::vector<int> capacities;
  std::vector<uint8_t> vehicle_types;
  std::vector<float> tier_thresholds;
  std::vector<float> tier_fixed_costs;
  std::vector<float> tier_costs_per_unit;
  std::vector<int> tier_offsets;
};

inline std::string normalize_token(std::string token)
{
  token.erase(std::remove(token.begin(), token.end(), ','), token.end());
  token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
  token.erase(std::remove_if(token.begin(),
                             token.end(),
                             [](unsigned char c) { return c == 0xef || c == 0xbb || c == 0xbf; }),
              token.end());
  return token;
}

template <typename value_t>
value_t read_value(std::istream& in)
{
  std::string token;
  if (!(in >> token)) { throw std::runtime_error("Unexpected end of FSMVRPTWSC file"); }
  token = normalize_token(token);
  try {
    auto const value = std::stof(token);
    if constexpr (std::is_integral_v<value_t>) {
      return static_cast<value_t>(value);
    } else {
      return static_cast<value_t>(value);
    }
  } catch (std::exception const& e) {
    throw std::runtime_error("Invalid numeric token in FSMVRPTWSC file: '" + token + "'");
  }
}

inline fsmvrptwsc_instance_t read_one_instance(std::istream& in)
{
  fsmvrptwsc_instance_t instance;
  in >> instance.name;
  instance.name              = normalize_token(instance.name);
  instance.n_clients         = read_value<int>(in);
  instance.n_vehicle_types   = read_value<int>(in);
  instance.n_distance_ranges = read_value<int>(in);

  auto const matrix_size = (instance.n_clients + 1) * (instance.n_clients + 1);
  instance.distance_matrix.resize(matrix_size);
  instance.transit_time_matrix.resize(matrix_size);
  for (auto& value : instance.distance_matrix) {
    value = read_value<float>(in);
  }
  for (auto& value : instance.transit_time_matrix) {
    value = read_value<float>(in);
  }

  for (int i = 0; i <= instance.n_clients; ++i) {
    auto const earliest = read_value<float>(in);
    auto const latest   = read_value<float>(in);
    auto const service  = read_value<float>(in);
    auto const demand   = read_value<float>(in);
    if (i > 0) {
      instance.order_locations.push_back(i);
      instance.earliest.push_back(static_cast<int>(earliest));
      instance.latest.push_back(static_cast<int>(latest));
      instance.service_times.push_back(static_cast<int>(service));
      instance.demand.push_back(static_cast<int>(demand));
    } else {
      instance.vehicle_earliest.push_back(static_cast<int>(earliest));
      instance.vehicle_latest.push_back(static_cast<int>(latest));
    }
  }

  std::vector<int> type_capacities(instance.n_vehicle_types);
  for (auto& capacity : type_capacities) {
    capacity = read_value<int>(in);
  }

  std::vector<float> range_starts(instance.n_distance_ranges);
  for (auto& range_start : range_starts) {
    range_start = read_value<float>(in);
  }

  std::vector<std::vector<float>> type_costs(instance.n_vehicle_types,
                                             std::vector<float>(instance.n_distance_ranges));
  for (auto& costs : type_costs) {
    for (auto& cost : costs) {
      cost = read_value<float>(in);
    }
  }

  // FSMVRPTWSC has an unrestricted fleet mix. Model that by making each vehicle
  // type available up to one route per client.
  instance.n_vehicles = instance.n_clients * instance.n_vehicle_types;
  instance.vehicle_earliest.resize(instance.n_vehicles, instance.vehicle_earliest.front());
  instance.vehicle_latest.resize(instance.n_vehicles, instance.vehicle_latest.front());
  instance.tier_offsets.push_back(0);
  for (int type = 0; type < instance.n_vehicle_types; ++type) {
    for (int copy = 0; copy < instance.n_clients; ++copy) {
      instance.vehicle_types.push_back(static_cast<uint8_t>(type));
      instance.capacities.push_back(type_capacities[type]);

      auto const& costs = type_costs[type];
      for (int tier = 0; tier < instance.n_distance_ranges; ++tier) {
        if (tier + 1 < instance.n_distance_ranges) {
          instance.tier_thresholds.push_back(range_starts[tier + 1]);
          auto const previous = tier == 0 ? 0.0f : costs[tier - 1];
          instance.tier_fixed_costs.push_back(costs[tier] - previous);
          instance.tier_costs_per_unit.push_back(0.0f);
        } else {
          instance.tier_thresholds.push_back(std::numeric_limits<float>::max());
          instance.tier_fixed_costs.push_back(0.0f);
          instance.tier_costs_per_unit.push_back(costs[tier]);
        }
      }
      instance.tier_offsets.push_back(static_cast<int>(instance.tier_thresholds.size()));
    }
  }

  return instance;
}

inline fsmvrptwsc_instance_t load_small_instance(std::string const& path,
                                                 std::string const& instance_name)
{
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("FSMVRPTWSC Small.txt cannot be opened: " + path);
  }

  auto const n_instances = read_value<int>(input);
  for (int i = 0; i < n_instances; ++i) {
    auto instance = read_one_instance(input);
    if (instance.name == instance_name) { return instance; }
  }

  cuopt_assert(false, "FSMVRPTWSC instance not found");
  return {};
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
