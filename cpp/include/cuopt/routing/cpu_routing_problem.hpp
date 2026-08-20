/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/routing_structures.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace raft {
class handle_t;
}

namespace cuopt {
namespace CUOPT_EXPORT routing {

/**
 * @brief Host-memory owning routing problem (gRPC / remote-execution analog of
 * data_model_view_t). Owns all arrays in std::vector and can materialize a
 * device-backed data_model_view_t via to_device().
 */
struct cpu_cost_matrix_t {
  uint8_t vehicle_type = 0;
  std::vector<float> matrix;  // num_locations x num_locations, row-major
};

struct cpu_capacity_dimension_t {
  std::string name;
  std::vector<int32_t> demand;    // per-order
  std::vector<int32_t> capacity;  // per-vehicle
};

struct cpu_vehicle_break_t {
  int32_t earliest = 0;
  int32_t latest   = 0;
  int32_t duration = 0;
  std::vector<int32_t> locations;
};

struct cpu_uniform_break_t {
  std::vector<int32_t> earliest;
  std::vector<int32_t> latest;
  std::vector<int32_t> duration;
};

struct cpu_initial_solution_t {
  std::vector<int32_t> vehicle_ids;
  std::vector<int32_t> routes;
  std::vector<int32_t> types;  // node_type_t values
  std::vector<int32_t> sol_offsets;
};

class cpu_routing_problem_t {
 public:
  int32_t num_locations = 0;
  int32_t fleet_size    = 0;
  int32_t num_orders    = -1;  // -1 => same as num_locations

  std::vector<cpu_cost_matrix_t> cost_matrices;
  std::vector<cpu_cost_matrix_t> transit_time_matrices;

  std::vector<int32_t> vehicle_start_locations;
  std::vector<int32_t> vehicle_return_locations;
  std::vector<int32_t> vehicle_tw_earliest;
  std::vector<int32_t> vehicle_tw_latest;
  std::vector<uint8_t> vehicle_types;
  std::vector<uint8_t> drop_return_trips;  // 0/1 (avoid vector<bool>)
  std::vector<uint8_t> skip_first_trips;   // 0/1
  std::vector<float> vehicle_max_costs;
  std::vector<float> vehicle_max_times;
  std::vector<float> vehicle_fixed_costs;

  std::vector<int32_t> order_locations;
  std::vector<int32_t> order_tw_earliest;
  std::vector<int32_t> order_tw_latest;
  std::vector<float> order_prizes;
  // vehicle_id -> service times; use -1 for the default (all vehicles)
  std::map<int32_t, std::vector<int32_t>> order_service_times;

  std::vector<int32_t> pickup_indices;
  std::vector<int32_t> delivery_indices;

  std::vector<cpu_capacity_dimension_t> capacity_dimensions;

  std::vector<int32_t> break_locations;
  std::vector<cpu_uniform_break_t> uniform_breaks;
  std::map<int32_t, std::vector<cpu_vehicle_break_t>> vehicle_breaks;

  std::map<int32_t, std::vector<int32_t>> vehicle_order_match;
  std::map<int32_t, std::vector<int32_t>> order_vehicle_match;
  std::map<int32_t, std::vector<int32_t>> order_precedence;

  std::vector<int32_t> objectives;  // objective_t enum values
  std::vector<float> objective_weights;
  int32_t min_vehicles = 0;

  cpu_initial_solution_t initial_solutions;

  /** Opaque owner of device buffers backing the returned data_model_view_t. */
  struct device_data_t;

  /** Deleter so unique_ptr can destroy incomplete device_data_t outside the .cu TU. */
  struct device_data_deleter {
    void operator()(device_data_t* p) const;
  };

  using device_data_ptr = std::unique_ptr<device_data_t, device_data_deleter>;

  /**
   * @brief Copy host data to the GPU and return a non-owning data_model_view_t.
   * The returned device_data_t must outlive the view.
   */
  std::pair<data_model_view_t<int, float>, device_data_ptr> to_device(raft::handle_t* handle) const;
};

/**
 * @brief Host-memory routing solution (remote-execution analog of the parsed
 * RoutingSolution proto). Populated on the client from the server's response.
 */
struct cpu_routing_solution_t {
  std::vector<int32_t> route;
  std::vector<double> arrival_stamp;
  std::vector<int32_t> truck_id;
  std::vector<int32_t> locations;
  std::vector<int32_t> node_types;
  std::vector<int32_t> unserviced_nodes;
  std::vector<int32_t> accepted;

  int32_t vehicle_count        = 0;
  double total_objective_value = 0.0;
  std::map<int32_t, double> objective_values;

  int32_t status = 0;  // cuopt.remote.RoutingSolutionStatus (0 == SUCCESS)
  std::string status_message;
  std::string error_message;
};

}  // namespace CUOPT_EXPORT routing
}  // namespace cuopt
