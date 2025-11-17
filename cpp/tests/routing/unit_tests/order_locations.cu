/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST(order_locations, zeroth_order)
{
  std::vector<float> cost_matrix           = {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                                              1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};
  std::vector<int> order_locations         = {2, 4, 3};
  std::vector<int> vehicle_start_locations = {0, 1, 0, 1};
  std::vector<int> vehicle_end_locations   = {0, 4, 4, 0};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix             = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations         = cuopt::device_copy(order_locations, stream);
  auto v_vehicle_start_locations = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_vehicle_end_locations   = cuopt::device_copy(vehicle_end_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 2, 3);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_vehicle_start_locations.data(),
                                   v_vehicle_end_locations.data());

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);

  check_route(data_model, host_route);
}

template <request_t REQUEST>
auto test_vanilla_model()
{
  int nlocations                   = 5;
  int norders                      = 4;
  int nvehicles                    = norders;
  std::vector<float> cost_matrix   = {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                                      1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};
  std::vector<int> order_locations = {2, 4, 3, 1};

  std::vector<int> pickup_orders, delivery_orders;
  if constexpr (REQUEST == request_t::PDP) {
    pickup_orders   = {0, 2};
    delivery_orders = {1, 3};
  }

  std::vector<int> vehicle_start_locations = {0, 0, 0, 0};
  // std::vector<int> vehicle_return_locations = {1, 1, 1, 1};
  std::vector<int> vehicle_return_locations = {0, 0, 0, 0};

  std::vector<int> capacities = {10, 10, 10, 10};
  std::vector<int> demands;

  if constexpr (REQUEST == request_t::PDP) {
    demands = {5, -5, 8, -8};
  } else {
    demands = {2, 2, 3, 3};
  }

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  rmm::device_uvector<int> v_pickup_orders(0, stream), v_delivery_orders(0, stream);

  if constexpr (REQUEST == request_t::PDP) {
    v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
    v_delivery_orders = cuopt::device_copy(delivery_orders, stream);
  }

  auto v_start_locations  = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_return_locations = cuopt::device_copy(vehicle_return_locations, stream);

  auto v_capacities = cuopt::device_copy(capacities, stream);
  auto v_demands    = cuopt::device_copy(demands, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  if constexpr (REQUEST == request_t::PDP) {
    data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());
  }
  data_model.set_vehicle_locations(v_start_locations.data(), v_return_locations.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
}

TEST(order_locations, vanilla_pdp) { test_vanilla_model<request_t::PDP>(); }
TEST(order_locations, vanilla_vrp) { test_vanilla_model<request_t::VRP>(); }

}  // namespace test
}  // namespace routing
}  // namespace cuopt
