/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/routing_test.cuh>

#include <routing/utilities/check_constraints.hpp>

#include <routing/generator/generator.hpp>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t>
class l1_retail_test_t : public base_test_t<i_t, f_t>,
                         public ::testing::TestWithParam<generator::dataset_params_t<int, float>> {
 public:
  void SetUp() override
  {
    auto params                 = GetParam();
    auto dataset                = generator::generate_dataset(this->handle_, params);
    this->use_secondary_matrix_ = params.n_matrix_types == 2;
    this->n_orders              = params.n_locations;
    this->n_vehicles            = dataset.get_fleet_info().get_num_vehicles();
    this->n_locations           = params.n_locations;
    this->n_orders              = this->n_locations;
    this->dim_                  = params.dim;
    this->n_break_dim_          = params.break_dim;
    this->n_break_locations_    = params.n_locations / 10;
    this->n_vehicle_types_      = params.n_vehicle_types;

    this->populate_device_vectors(dataset);
    this->populate_host_vectors();
  }

  auto setup_data_model()
  {
    cuopt::routing::data_model_view_t<int, float> data_model(
      &(this->handle_), this->n_locations, this->n_vehicles, this->n_orders);
    detail::fill_data_model_matrices(data_model, this->matrices_d);
    for (i_t dim = 0; dim < this->dim_; ++dim) {
      data_model.add_capacity_dimension("",
                                        this->demand_d.data() + (dim * this->n_locations),
                                        this->capacity_d.data() + (dim * this->n_vehicles));
    }
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    for (int truck_id = 0; truck_id < this->n_vehicles; ++truck_id) {
      data_model.set_order_service_times(
        this->fleet_order_constraints_d.order_service_times.data() + truck_id * this->n_orders,
        truck_id);
    }
    return data_model;
  }

  void test_vehicle_max_times()
  {
    auto time_matrix    = this->matrices_d.get_time_matrix(0);
    auto max_time_depot = thrust::reduce(this->handle_.get_thrust_policy(),
                                         time_matrix,
                                         time_matrix + this->n_locations,
                                         -1,
                                         thrust::maximum<float>());
    // New solver applies the maximum time limit to the combined drive time and
    // service time. (see PR #927)
    // The old solver applied this limit only towards the travel time.
    // Tests written for the old solver should be fixed to have service time
    // included in the maximum time limit for an apples to apples comparison.
    auto max_service       = thrust::reduce(this->handle_.get_thrust_policy(),
                                      this->service_time_d.data(),
                                      this->service_time_d.data() + this->n_orders,
                                      -1,
                                      thrust::maximum<f_t>());
    auto vehicle_max_times = max_time_depot * (2 + (i_t)this->pickup_delivery_) + 2 +
                             (1 + (i_t)this->pickup_delivery_) * max_service;
    thrust::fill(this->handle_.get_thrust_policy(),
                 this->vehicle_max_times_d.begin(),
                 this->vehicle_max_times_d.end(),
                 vehicle_max_times);
    auto data_model = this->setup_data_model();
    data_model.set_vehicle_max_times(this->vehicle_max_times_d.data());
    auto routing_solution = this->solve(data_model);
    ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution);
  }

  void test_vehicle_max_costs()
  {
    auto max_dist_depot = thrust::reduce(this->handle_.get_thrust_policy(),
                                         this->cost_matrix_d.data(),
                                         this->cost_matrix_d.data() + this->n_locations,
                                         -1,
                                         thrust::maximum<float>());
    thrust::fill(this->handle_.get_thrust_policy(),
                 this->vehicle_max_costs_d.begin(),
                 this->vehicle_max_costs_d.end(),
                 max_dist_depot * 2 + 2);
    auto data_model = this->setup_data_model();
    data_model.set_vehicle_max_costs(this->vehicle_max_costs_d.data());

    auto routing_solution = this->solve(data_model);
    ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);
  }

  void test_vehicle_breaks()
  {
    // Order ids should be between 0 and this->n_locations. The rest of the locations are optional
    // nodes (breaks) so order number is decreased in cvrptw case.
    this->n_orders  = this->n_locations - this->n_break_locations_;
    auto data_model = setup_data_model();
    data_model.set_break_locations(this->break_locations_d.data(), this->break_locations_d.size());

    for (i_t dim = 0; dim < this->n_break_dim_; ++dim) {
      data_model.add_break_dimension(this->break_earliest_d.data() + dim * this->n_vehicles,
                                     this->break_latest_d.data() + dim * this->n_vehicles,
                                     this->break_duration_d.data() + dim * this->n_vehicles);
    }
    auto routing_solution = this->solve(data_model);
    ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS);

    // checks
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution);
    this->check_vehicle_breaks(h_routing_solution);
  }

  void test_drop_return_trips()
  {
    auto data_model = setup_data_model();
    data_model.set_drop_return_trips(this->bool_drop_return_trips_d.data());
    auto routing_solution = this->solve(data_model);
    // checks
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution);
  }

  void test_vehicle_lower_bound()
  {
    auto data_model = setup_data_model();
    auto init_sol   = this->solve(data_model);
    auto init_vn    = init_sol.get_vehicle_count();

    auto min_vehicles = init_vn + 2;

    data_model.set_min_vehicles(min_vehicles);
    auto routing_solution = this->solve(data_model);
    // checks
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution);
    ASSERT_GE(routing_solution.get_vehicle_count(), min_vehicles);
  }

  void test_vehicle_time_windows()
  {
    this->vehicle_tw_ = true;
    auto data_model   = setup_data_model();
    data_model.set_vehicle_time_windows(this->vehicle_earliest_d.data(),
                                        this->vehicle_latest_d.data());
    auto routing_solution = this->solve(data_model);
    ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS);

    // checks
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution);
    this->check_vehicle_time_windows(
      h_routing_solution, this->vehicle_earliest_h, this->vehicle_latest_h);
  }
};

static const int dim                    = 2;
static const demand_i_t min_demand[dim] = {0, 1};  // demand, order_count
static const demand_i_t max_demand[dim] = {20, 1};
static const cap_i_t capacities[dim]    = {200, 150};

// Variating params
static const int sizes[1]                               = {201};
static const generator::dataset_distribution_t dists[3] = {
  generator::dataset_distribution_t::RANDOM,
  generator::dataset_distribution_t::CLUSTERED,
  generator::dataset_distribution_t::RANDOM_CLUSTERED};

auto get_params()
{
  std::vector<generator::dataset_params_t<int, float>> params;
  for (auto const& dist : dists) {
    for (auto const& size : sizes) {
      generator::dataset_params_t<int, float> p{.n_locations       = size,
                                                .asymmetric        = false,
                                                .dim               = dim,
                                                .min_demand        = min_demand,
                                                .max_demand        = max_demand,
                                                .min_capacities    = capacities,
                                                .max_capacities    = capacities,
                                                .min_service_time  = 0,
                                                .max_service_time  = 10,
                                                .tw_tightness      = 0.5,
                                                .drop_return_trips = 0.5,
                                                .n_shifts          = 2,
                                                .n_vehicle_types   = 1,
                                                .n_matrix_types    = 2,
                                                .break_dim         = 1,
                                                .distrib           = dist};
      params.push_back(p);
    }
  }
  return params;
}

typedef l1_retail_test_t<int, float> retail_float_test_t;

TEST_P(retail_float_test_t, vehicle_max_times) { test_vehicle_max_times(); }
TEST_P(retail_float_test_t, vehicle_max_costs) { test_vehicle_max_costs(); }
TEST_P(retail_float_test_t, vehicle_breaks) { test_vehicle_breaks(); }
TEST_P(retail_float_test_t, vehicle_lower_bound) { test_vehicle_lower_bound(); }
TEST_P(retail_float_test_t, drop_return_trips) { test_drop_return_trips(); }
TEST_P(retail_float_test_t, vehicle_time_windows) { test_vehicle_time_windows(); }
///@todo: add precedence tests

INSTANTIATE_TEST_SUITE_P(level1_retail, retail_float_test_t, ::testing::ValuesIn(get_params()));

}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
