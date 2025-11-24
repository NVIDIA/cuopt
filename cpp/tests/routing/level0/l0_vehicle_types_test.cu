/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/routing_test.cuh>

#include <routing/utilities/data_model.hpp>

#include <routing/generator/generator.hpp>

#include <raft/random/rng.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t>
class vehicle_types_test_t : public base_test_t<i_t, f_t>, public ::testing::Test {
 public:
  vehicle_types_test_t() : base_test_t<i_t, f_t>(512, 5E-2, 0) {}
  void SetUp() override
  {
    this->n_locations     = input_.n_locations;
    this->n_vehicles      = input_.n_vehicles;
    this->n_orders        = this->n_locations;
    this->x_h             = input_.x_h;
    this->y_h             = input_.y_h;
    this->capacity_h      = input_.capacity_h;
    this->demand_h        = input_.demand_h;
    this->earliest_time_h = input_.earliest_time_h;
    this->latest_time_h   = input_.latest_time_h;
    this->service_time_h  = input_.service_time_h;
    this->vehicle_types_h = input_.vehicle_types_h;
    this->n_vehicle_types_ =
      std::unordered_set<uint8_t>(this->vehicle_types_h.begin(), this->vehicle_types_h.end())
        .size();
    this->populate_device_vectors();
  }

  void test_vehicle_types()
  {
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    generator::dataset_params_t<i_t, f_t> params;
    params.n_locations     = this->n_locations;
    params.n_vehicle_types = this->n_vehicle_types_;
    auto coordinates       = generator::generate_coordinates<i_t, f_t>(this->handle_, params);
    auto matrices = generator::generate_matrices<i_t, f_t>(this->handle_, params, coordinates);

    detail::fill_data_model_matrices(data_model, matrices);

    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());
    data_model.set_vehicle_types(this->vehicle_types_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> settings;

    auto routing_solution = this->solve(data_model, settings);
    int vehicles          = routing_solution.get_vehicle_count();
    double cost           = routing_solution.get_total_objective();
    std::cout << "Vehicle: " << vehicles << " Cost: " << cost << "\n";
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  }
};

typedef vehicle_types_test_t<int, float> vehicle_types_float_test_t;

TEST_F(vehicle_types_float_test_t, VEHICLE_TYPES) { test_vehicle_types(); }

}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
