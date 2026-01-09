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

#include <random>
#include <tuple>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t>
class objective_function_test_t : public base_test_t<i_t, f_t>,
                                  public ::testing::TestWithParam<std::pair<objective_t, double>> {
 public:
  objective_function_test_t() : base_test_t<i_t, f_t>(512, 0, 0) {}
  void SetUp() override
  {
    auto p                = GetParam();
    this->objective_type  = std::get<0>(p);
    this->min_weight      = std::get<1>(p);
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
    this->populate_device_vectors();
  }

  /**
   * @brief This function tests one objective type at a time. The expectation is that the objective
   * (excluding the weight) should decrease as we increase the weight.  We vary the weight from 1 to
   * 10000 factors and check if the objective is decreased or not
   */
  void test_objective_function()
  {
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles);
    data_model.add_cost_matrix(this->cost_matrix_d.data());
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    cuopt::routing::solver_settings_t<i_t, f_t> settings;

    // assumes that only one objective function is set along with primary cost with weight 1
    // a better way would be to implement storing all costs separately in assignment_t class
    auto get_objective = [](const auto& routing_solution, const double weight) {
      return (routing_solution.get_total_objective() - routing_solution.get_total_objective()) /
             weight;
    };

    int ntests = 4;

    std::vector<float> objective_values(ntests);
    for (int i = 0; i < ntests; ++i) {
      float objective_weight         = min_weight * pow(10, i);
      std::vector<objective_t> types = {objective_t::COST, this->objective_type};
      std::vector<float> values      = {1, objective_weight};

      auto objective_types_d   = cuopt::device_copy(types, this->stream_view_);
      auto objective_weights_d = cuopt::device_copy(values, this->stream_view_);
      data_model.set_objective_function(
        objective_types_d.data(), objective_weights_d.data(), objective_types_d.size());
      // Call solve.
      auto routing_solution = this->solve(data_model, settings);
      ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
      host_assignment_t<i_t> h_routing_solution(routing_solution);
      // Checks
      check_route(data_model, h_routing_solution);
      objective_values[i] = get_objective(routing_solution, objective_weight);
    }

    for (int i = 1; i < ntests; ++i) {
      EXPECT_LE(objective_values[i], objective_values[i - 1] + 1e-10);
    }
  }

  objective_t objective_type;
  double min_weight;
};

typedef objective_function_test_t<int, float> objective_function_float_test_t;

TEST_P(objective_function_float_test_t, OBJECTIVE_FUNCTION) { test_objective_function(); }

// note that we use 1.0/90./90. for service time case. This is to normalize with respect to service
// time of 90
INSTANTIATE_TEST_SUITE_P(level0_objective_function,
                         objective_function_float_test_t,
                         ::testing::Values(std::pair(objective_t::VARIANCE_ROUTE_SIZE, 1.0),
                                           std::pair(objective_t::VARIANCE_ROUTE_SERVICE_TIME,
                                                     1.0 / 90. / 90.)));
}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
