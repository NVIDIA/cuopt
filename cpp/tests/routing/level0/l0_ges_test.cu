/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/routing_test.cuh>

#include <routing/ges_solver.cuh>
#include <routing/utilities/data_model.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t, request_t REQUEST>
class routing_ges_test_t : public ::testing::TestWithParam<std::tuple<bool>>,
                           public base_test_t<i_t, f_t> {
 public:
  routing_ges_test_t() : base_test_t<i_t, f_t>(1) {}

  void SetUp() override
  {
    this->pickup_delivery_ = (bool)(REQUEST == request_t::PDP);

    this->n_locations = input_double_.n_locations;
    this->n_vehicles  = input_double_.n_vehicles;
    this->n_orders    = this->n_locations;
    this->x_h         = input_double_.x_h;
    this->y_h         = input_double_.y_h;
    this->demand_h =
      this->pickup_delivery_ ? input_double_.pickup_delivery_demand_h : input_double_.demand_h;
    this->capacity_h = input_double_.capacity_h;
    this->earliest_time_h =
      this->pickup_delivery_ ? input_double_.pickup_earliest_time_h : input_double_.earliest_time_h;
    this->latest_time_h =
      this->pickup_delivery_ ? input_double_.pickup_latest_time_h : input_double_.latest_time_h;
    this->service_time_h      = input_double_.service_time_h;
    this->drop_return_trips_h = input_double_.drop_return_h;
    this->skip_first_trips_h  = input_double_.skip_first_h;
    this->vehicle_earliest_h  = input_double_.vehicle_earliest_h;
    this->vehicle_latest_h    = input_double_.vehicle_latest_h;
    this->break_earliest_h    = input_double_.break_earliest_h;
    this->break_latest_h      = input_double_.break_latest_h;
    this->break_duration_h    = input_double_.break_duration_h;
    this->vehicle_types_h.assign(this->n_vehicles, 0);

    this->n_pairs_ = (this->n_orders - 1) / 2;
    this->pickup_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->delivery_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->populate_device_vectors();
  }

  void TearDown() override {}

  assignment_t<i_t> solve(const cuopt::routing::data_model_view_t<i_t, f_t>& data_model,
                          const cuopt::routing::solver_settings_t<i_t, f_t>& solver_settings,
                          i_t expected_route_count)
  {
    cudaDeviceSynchronize();
    ges_solver_t<i_t, f_t, REQUEST> solver{
      data_model, solver_settings, this->n_orders / 5.f, expected_route_count};
    this->hr_timer_.start("GES solver");
    auto assignment = solver.compute_ges_solution();
    cudaDeviceSynchronize();
    this->hr_timer_.stop();
    this->hr_timer_.display(std::cout);
    return assignment;
  }

  void test_cvrptw()
  {
    // data model
    // if data_model changes and there are fewer locations than orders
    // adjust the constructor accordingly
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    if constexpr (REQUEST == request_t::PDP) {
      raft::copy(this->pickup_indices_d.data(),
                 input_double_.pickup_indices_h.data(),
                 this->n_pairs_,
                 this->stream_view_);
      raft::copy(this->delivery_indices_d.data(),
                 input_double_.delivery_indices_h.data(),
                 this->n_pairs_,
                 this->stream_view_);
      data_model.set_pickup_delivery_pairs(this->pickup_indices_d.data(),
                                           this->delivery_indices_d.data());
    }

    data_model.add_cost_matrix(this->cost_matrix_d.data());
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> solver_settings;
    solver_settings.set_time_limit(120.f);

    // solve
    const i_t expected_route_count = 13;
    auto routing_solution          = this->solve(data_model, solver_settings, expected_route_count);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    i_t v_count = routing_solution.get_vehicle_count();
    f_t cost    = routing_solution.get_total_objective();
    std::cout << "Vehicle: " << v_count << " Cost: " << cost << "\n";
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    ASSERT_LE(v_count, expected_route_count);

    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution, false);
    // check weight
    this->check_capacity(
      h_routing_solution, this->demand_h, input_double_.capacity_h, this->demand_d);
  }
};

typedef routing_ges_test_t<int, float, request_t::PDP> double_test_pdp;
typedef routing_ges_test_t<int, float, request_t::VRP> double_test_vrp;

template <typename i_t, typename f_t, request_t REQUEST>
class simple_routes_ges_test_t : public ::testing::TestWithParam<test_data_t<i_t, f_t>>,
                                 public base_test_t<i_t, f_t> {
 public:
  simple_routes_ges_test_t() : base_test_t<i_t, f_t>(4) {}

  void SetUp() override
  {
    const auto& param = this->GetParam();

    this->pickup_delivery_ = (bool)(REQUEST == request_t::PDP);
    this->n_locations      = param.n_locations;
    this->n_vehicles       = param.n_vehicles;
    this->n_orders         = this->n_locations;
    this->x_h              = param.x_h;
    this->y_h              = param.y_h;
    this->demand_h   = this->pickup_delivery_ ? param.pickup_delivery_demand_h : param.demand_h;
    this->capacity_h = param.capacity_h;
    this->earliest_time_h =
      this->pickup_delivery_ ? param.pickup_earliest_time_h : param.earliest_time_h;
    this->latest_time_h = this->pickup_delivery_ ? param.pickup_latest_time_h : param.latest_time_h;
    this->service_time_h        = param.service_time_h;
    this->drop_return_trips_h   = param.drop_return_h;
    this->skip_first_trips_h    = param.skip_first_h;
    this->vehicle_earliest_h    = param.vehicle_earliest_h;
    this->vehicle_latest_h      = param.vehicle_latest_h;
    this->break_earliest_h      = param.break_earliest_h;
    this->break_latest_h        = param.break_latest_h;
    this->break_duration_h      = param.break_duration_h;
    this->pickup_indices_h      = param.pickup_indices_h;
    this->delivery_indices_h    = param.delivery_indices_h;
    this->use_secondary_matrix_ = true;
    this->expected_route_h      = param.expected_route;
    this->vehicle_types_h       = param.vehicle_types_h;

    this->n_pairs_ = (this->n_orders - 1) / 2;
    this->pickup_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->delivery_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->populate_device_vectors();
  }

  void TearDown() override {}

  assignment_t<i_t> solve(const cuopt::routing::data_model_view_t<i_t, f_t>& data_model,
                          const cuopt::routing::solver_settings_t<i_t, f_t>& solver_settings,
                          i_t expected_route_count)
  {
    this->handle_.sync_stream();
    // On purpose too small to trigger reallocation
    ges_solver_t<i_t, f_t, REQUEST> solver{
      data_model, solver_settings, this->n_orders / 5.f, expected_route_count};
    this->hr_timer_.start("GES solver");
    auto routing_solution = solver.compute_ges_solution();
    cudaDeviceSynchronize();
    this->hr_timer_.stop();
    this->hr_timer_.display(std::cout);
    return routing_solution;
  }

  void test_cvrptw()
  {
    // data model
    // if data_model changes and there are fewer locations than orders
    // adjust the constructor accordingly
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    if constexpr (REQUEST == request_t::PDP) {
      raft::copy(this->pickup_indices_d.data(),
                 this->pickup_indices_h.data(),
                 this->n_pairs_,
                 this->stream_view_);
      raft::copy(this->delivery_indices_d.data(),
                 this->delivery_indices_h.data(),
                 this->n_pairs_,
                 this->stream_view_);
      data_model.set_pickup_delivery_pairs(this->pickup_indices_d.data(),
                                           this->delivery_indices_d.data());
    }

    data_model.add_cost_matrix(this->cost_matrix_d.data());
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> solver_settings;
    solver_settings.set_time_limit(this->n_orders / 5.f);

    // solve
    const i_t expected_route_count = 3;
    data_model.set_min_vehicles(expected_route_count);
    auto routing_solution = this->solve(data_model, solver_settings, expected_route_count);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    i_t v_count = routing_solution.get_vehicle_count();
    f_t cost    = routing_solution.get_total_objective();
    std::cout << "Vehicle: " << v_count << " Cost: " << cost << "\n";
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    ASSERT_LE(v_count, expected_route_count);

    i_t returned_vehicle_count = routing_solution.get_vehicle_count();

    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution, false);
    // check weight
    this->check_capacity(
      h_routing_solution, this->demand_h, input_double_.capacity_h, this->demand_d);
  }
};

typedef simple_routes_ges_test_t<int, float, request_t::PDP> simple_routes_test_pdp;

TEST_P(double_test_pdp, GES_PDP) { test_cvrptw(); }
INSTANTIATE_TEST_SUITE_P(level0_ges, double_test_pdp, ::testing::Values(std::make_tuple(true)));
TEST_P(simple_routes_test_pdp, GES_PDP) { test_cvrptw(); }
INSTANTIATE_TEST_SUITE_P(level0_ges,
                         simple_routes_test_pdp,
                         ::testing::ValuesIn(parse_problems(simple_three_routes_)));

TEST_P(double_test_vrp, GES_VRP) { test_cvrptw(); }
INSTANTIATE_TEST_SUITE_P(level0_ges, double_test_vrp, ::testing::Values(std::make_tuple(true)));

}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
