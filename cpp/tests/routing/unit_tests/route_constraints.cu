/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <routing/node/pdp_node.cuh>
#include <routing/utilities/check_constraints.hpp>
#include <utilities/copy_helpers.hpp>

#include <tuple>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

template <request_t REQUEST>
static auto test_vanilla_model()
{
  static_assert(REQUEST == request_t::PDP);
  int nlocations                 = 5;
  int nvehicles                  = 3;
  std::vector<float> cost_matrix = {0,  10, 10, 10, 10, 10, 0,  10, 10, 10, 10, 10, 0,
                                    10, 10, 10, 10, 10, 0,  10, 10, 10, 10, 10, 0};
  auto time_matrix               = cost_matrix;

  // cost_matrix[3 + 4 * nlocations] = 100;
  // cost_matrix[4 + 3 * nlocations] = 100;
  // std::vector<int> order_locations = {2, 4, 3, 1, 2, 3};

  // std::vector<int> order_locations = {2, 1, 3, 1, 4, 1, 1, 2};
  std::vector<int> order_locations = {0, 1, 0, 2, 0, 3, 0, 4};
  int norders                      = order_locations.size();

  std::vector<int> pickup_orders, delivery_orders;
  if constexpr (REQUEST == request_t::PDP) {
    pickup_orders   = {0, 2, 4, 6};
    delivery_orders = {1, 3, 5, 7};
  }

  std::vector<int> vehicle_start_locations  = {0, 0, 0};
  std::vector<int> vehicle_return_locations = {0, 0, 0};
  std::vector<float> vehicle_max_times      = {59, 59, 59};
  std::vector<float> vehicle_max_costs      = {70, 70, 70};

  std::vector<int> capacities = {6, 6, 6};
  std::vector<int> demands;

  if constexpr (REQUEST == request_t::PDP) {
    demands = {5, -5, 5, -5, 5, -5, 5, -5};
  } else {
    demands = {2, 2, 3, 3, 4, 4, 2, 2};
  }

  std::vector<objective_t> objective_types = {objective_t::COST, objective_t::VARIANCE_ROUTE_SIZE};
  std::vector<float> objective_weights     = {1.0, 100.};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_time_matrix     = cuopt::device_copy(time_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  rmm::device_uvector<int> v_pickup_orders(0, stream), v_delivery_orders(0, stream);

  if constexpr (REQUEST == request_t::PDP) {
    v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
    v_delivery_orders = cuopt::device_copy(delivery_orders, stream);
  }

  auto v_start_locations   = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_return_locations  = cuopt::device_copy(vehicle_return_locations, stream);
  auto v_vehicle_max_times = cuopt::device_copy(vehicle_max_times, stream);
  auto v_vehicle_max_costs = cuopt::device_copy(vehicle_max_costs, stream);

  auto v_capacities = cuopt::device_copy(capacities, stream);
  auto v_demands    = cuopt::device_copy(demands, stream);

  auto v_objective_types   = cuopt::device_copy(objective_types, stream);
  auto v_objective_weights = cuopt::device_copy(objective_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  if constexpr (REQUEST == request_t::PDP) {
    data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());
  }
  data_model.set_vehicle_locations(v_start_locations.data(), v_return_locations.data());
  data_model.set_vehicle_max_times(v_vehicle_max_times.data());
  data_model.set_vehicle_max_costs(v_vehicle_max_costs.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  data_model.set_objective_function(
    v_objective_types.data(), v_objective_weights.data(), v_objective_types.size());

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);

  // host_route.print();
}

TEST(route_constraints, vanilla_pdp) { test_vanilla_model<request_t::PDP>(); }

}  // namespace test
}  // namespace routing
}  // namespace cuopt
