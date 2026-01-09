/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/routing_test.cuh>

#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/data_model.hpp>
#include <utilities/copy_helpers.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <cuda/std/functional>

#include <random>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t>
class vehicle_order_test_t : public base_test_t<i_t, f_t>, public ::testing::TestWithParam<float> {
 public:
  vehicle_order_test_t() : base_test_t<i_t, f_t>(512, 0, 0) {}
  void SetUp() override
  {
    this->not_matching_constraints_fraction = GetParam();
    this->n_locations                       = input_.n_locations;
    this->n_vehicles                        = input_.n_vehicles;
    this->n_orders                          = this->n_locations;
    this->x_h                               = input_.x_h;
    this->y_h                               = input_.y_h;
    this->capacity_h                        = input_.capacity_h;
    this->demand_h                          = input_.demand_h;
    this->earliest_time_h                   = input_.earliest_time_h;
    this->latest_time_h                     = input_.latest_time_h;
    this->service_time_h                    = input_.service_time_h;
    this->populate_device_vectors();
  }

  void test_vehicle_order_match()
  {
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles);
    data_model.add_cost_matrix(this->cost_matrix_d.data());
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    rmm::device_uvector<bool> d_drop_return_trip(input_.n_vehicles, this->stream_view_);
    {
      auto d_int_vec = cuopt::device_copy(input_.drop_return_h, this->stream_view_);
      thrust::transform(rmm::exec_policy(this->stream_view_),
                        d_int_vec.begin(),
                        d_int_vec.end(),
                        d_drop_return_trip.begin(),
                        cuda::std::identity{});
      RAFT_CUDA_TRY(cudaStreamSynchronize(this->stream_view_.value()));
    }
    data_model.set_drop_return_trips(d_drop_return_trip.data());

    int num_constraints =
      int(not_matching_constraints_fraction * this->n_vehicles * (this->n_locations - 1));

    // use a seed so that the test is deterministic
    srand(124);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
      0, this->n_vehicles * this->n_locations - 1);

    // first assume that all vehicles can pickup all orders
    std::vector<int> order_vec(this->n_locations);
    std::iota(order_vec.begin(), order_vec.end(), 0);
    std::unordered_set<int> order_set(order_vec.begin(), order_vec.end());
    std::unordered_map<int, std::unordered_set<int>> vehicle_order_match;
    for (int i = 0; i < this->n_vehicles; ++i) {
      vehicle_order_match.emplace(i, order_set);
    }

    // remove some
    int cnt = 0;
    while (cnt < num_constraints) {
      int id      = dist(rng);
      int order   = id % this->n_locations;
      int vehicle = id / this->n_vehicles;
      if (order > 0) {
        auto& order_set = vehicle_order_match[vehicle];
        if (order_set.count(order)) {
          order_set.erase(order);
          cnt++;
        }
      }
    }

    std::unordered_map<int, rmm::device_uvector<int>> vehicle_order_match_d;
    for (const auto& [id, orders] : vehicle_order_match) {
      if (!orders.empty()) {
        auto order_vec = std::vector<int>(orders.begin(), orders.end());
        vehicle_order_match_d.emplace(id, cuopt::device_copy(order_vec, this->stream_view_));
      }
    }

    for (const auto& [id, orders] : vehicle_order_match_d) {
      ASSERT_GE(orders.size(), 0u);
      data_model.add_vehicle_order_match(id, orders.data(), orders.size());
    }

    cuopt::routing::solver_settings_t<i_t, f_t> settings;

    // Call solve.
    auto routing_solution = this->solve(data_model, settings);
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    // Checks
    check_route(data_model, h_routing_solution);
  }

  float not_matching_constraints_fraction = 0.f;
};

typedef vehicle_order_test_t<int, float> vehicle_order_float_test_t;

TEST_P(vehicle_order_float_test_t, VEHICLE_ORDER_MATCH_CONSTRAINTS) { test_vehicle_order_match(); }
INSTANTIATE_TEST_SUITE_P(level0_vehicle_order_match,
                         vehicle_order_float_test_t,
                         ::testing::Values(0.01, 0.05, 0.1, 0.2));

}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
