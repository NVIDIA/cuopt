/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <routing/routing_test.cuh>

#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/data_model.hpp>
#include <routing/utilities/retail_params.hpp>

#include <raft/random/rng.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <cuda/std/functional>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t>
class l0_routing_test_t : public routing_test_t<i_t, f_t>,
                          public ::testing::TestWithParam<file_params> {
 public:
  typedef routing_test_t<i_t, f_t> super;
  l0_routing_test_t() : super() {}

  void SetUp() override
  {
    auto param     = GetParam();
    this->ref_cost = param.ref_cost;
    this->ref_vn   = param.ref_vn;

    this->n_locations     = input_.n_locations;
    this->n_vehicles      = input_.n_vehicles;
    this->n_orders        = this->n_locations;
    this->x_h             = input_.x_h;
    this->y_h             = input_.y_h;
    this->demand_h        = input_.demand_h;
    this->capacity_h      = input_.capacity_h;
    this->earliest_time_h = input_.earliest_time_h;
    this->latest_time_h   = input_.latest_time_h;
    this->service_time_h  = input_.service_time_h;
    this->vehicle_types_h = std::vector<uint8_t>(input_.n_vehicles, 0);
    this->populate_device_vectors();
  }
};

template <typename i_t, typename f_t>
class lut_test_t : public routing_test_t<i_t, f_t>, public ::testing::TestWithParam<bool> {
 public:
  typedef routing_test_t<i_t, f_t> super;
  lut_test_t() : super() {}

  void SetUp() override
  {
    this->ref_cost        = 854.943;
    this->ref_vn          = 11;
    this->vehicle_breaks_ = GetParam();

    this->regression_check = true;

    this->n_locations      = input_.n_locations;
    this->n_vehicles       = input_.n_vehicles;
    this->n_orders         = this->n_locations;
    this->x_h              = input_.x_h;
    this->y_h              = input_.y_h;
    this->demand_h         = input_.demand_h;
    this->capacity_h       = input_.capacity_h;
    this->earliest_time_h  = input_.earliest_time_h;
    this->latest_time_h    = input_.latest_time_h;
    this->service_time_h   = input_.service_time_h;
    this->break_earliest_h = input_.break_earliest_h;
    this->break_latest_h   = input_.break_latest_h;
    this->break_duration_h = input_.break_duration_h;
    this->vehicle_types_h  = std::vector<uint8_t>(input_.n_vehicles, 0);
    this->n_break_dim_     = 1;

    // Duplicate order locations so that we have redundant locations
    // Copy 8 to 59
    this->x_h[59] = this->x_h[8];
    this->y_h[59] = this->y_h[8];

    // Copy 20 to  21, 22, 23, 24
    this->x_h[21] = this->x_h[20];
    this->x_h[22] = this->x_h[20];
    this->x_h[23] = this->x_h[20];
    this->x_h[24] = this->x_h[20];

    this->y_h[21] = this->y_h[20];
    this->y_h[22] = this->y_h[20];
    this->y_h[23] = this->y_h[20];
    this->y_h[24] = this->y_h[20];

    // Copy 62 to 63, 64, 65, 66, 67
    this->x_h[63] = this->x_h[62];
    this->x_h[64] = this->x_h[62];
    this->x_h[65] = this->x_h[62];
    this->x_h[66] = this->x_h[62];
    this->x_h[67] = this->x_h[62];

    this->y_h[63] = this->y_h[62];
    this->y_h[64] = this->y_h[62];
    this->y_h[65] = this->y_h[62];
    this->y_h[66] = this->y_h[62];
    this->y_h[67] = this->y_h[62];
  }

  void setup_lut()
  {
    auto [x_unique, y_unique] = this->remove_nonunique_locations();
    this->n_locations         = x_unique.size() + this->n_break_locations_;
    for (i_t i = this->n_orders; i < this->n_orders + this->n_break_locations_; ++i) {
      x_unique.push_back(this->x_h[i]);
      y_unique.push_back(this->y_h[i]);
    }

    this->matrices_h   = detail::create_host_mdarray<f_t>(this->n_locations,
                                                        this->n_vehicle_types_,
                                                        1 + this->use_secondary_matrix_,
                                                        this->stream_view_);
    auto cost_matrix_h = this->matrices_h.get_cost_matrix(0);
    build_dense_matrix(cost_matrix_h, x_unique, y_unique);

    this->remove_zeroth_order();
    if (!this->order_locations.empty()) {
      this->order_locations_d.resize(this->n_orders, this->stream_view_);
      raft::copy(this->order_locations_d.data(),
                 this->order_locations.data(),
                 this->n_orders,
                 this->stream_view_);
    }

    this->populate_device_vectors();
  }

  void test_cvrptw()
  {
    setup_lut();
    super::test_cvrptw();
  }

  auto run_solver()
  {
    this->break_duration_d.resize(this->n_break_locations_, this->stream_view_);
    thrust::sequence(this->handle_.get_thrust_policy(),
                     this->break_locations_d.begin(),
                     this->break_locations_d.end(),
                     this->n_locations - this->n_break_locations_);

    // data model
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    data_model.add_cost_matrix(this->cost_matrix_d.data());
    if (this->order_locations_d.size()) {
      data_model.set_order_locations(this->order_locations_d.data());
    }

    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    int shift_break =
      (this->break_latest_h[0] - this->break_earliest_h[0] + this->break_duration_h[0]) * 1.1;
    for (i_t dim = 0; dim < this->n_break_dim_; ++dim) {
      raft::copy(this->break_earliest_d.data() + (dim * this->n_vehicles),
                 this->break_earliest_h.data(),
                 this->break_earliest_h.size(),
                 this->stream_view_);
      raft::copy(this->break_latest_d.data() + (dim * this->n_vehicles),
                 this->break_latest_h.data(),
                 this->break_latest_h.size(),
                 this->stream_view_);
      raft::copy(this->break_duration_d.data() + (dim * this->n_vehicles),
                 this->break_duration_h.data(),
                 this->break_duration_h.size(),
                 this->stream_view_);
      thrust::transform(this->handle_.get_thrust_policy(),
                        this->break_earliest_d.begin(),
                        this->break_earliest_d.end(),
                        thrust::make_constant_iterator(dim * shift_break),
                        this->break_earliest_d.begin(),
                        thrust::plus<i_t>());
      thrust::transform(this->handle_.get_thrust_policy(),
                        this->break_latest_d.begin(),
                        this->break_latest_d.end(),
                        thrust::make_constant_iterator(dim * shift_break),
                        this->break_latest_d.begin(),
                        thrust::plus<i_t>());
    }
    data_model.set_break_locations(this->break_locations_d.data(), this->break_locations_d.size());
    for (i_t dim = 0; dim < this->n_break_dim_; ++dim) {
      data_model.add_break_dimension(this->break_earliest_d.data() + dim * this->n_vehicles,
                                     this->break_latest_d.data() + dim * this->n_vehicles,
                                     this->break_duration_d.data() + dim * this->n_vehicles);
    }

    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(this->n_orders / 5);
    // solve
    auto routing_solution = this->solve(data_model, settings);

    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);

    return routing_solution;
  }

  void test_vehicle_breaks()
  {
    this->n_break_locations_ = 5;
    for (i_t i = 0; i < this->n_break_locations_; ++i) {
      this->x_h.push_back(50 * i);
      this->y_h.push_back(50 * i);
    }
    this->n_locations = this->n_orders + this->n_break_locations_;

    this->populate_device_vectors();
    auto routing_solution = run_solver();
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    this->ref_vn   = routing_solution.get_vehicle_count();
    this->ref_cost = routing_solution.get_total_objective();

    // Second call with lut
    setup_lut();
    routing_solution = run_solver();
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    // copy route to host

    host_assignment_t<i_t> h_routing_solution(routing_solution);

    this->break_earliest_h.resize(this->n_break_dim_ * this->n_vehicles);
    this->break_latest_h.resize(this->n_break_dim_ * this->n_vehicles);
    this->break_duration_h.resize(this->n_break_dim_ * this->n_vehicles);
    for (auto dim = 0; dim < this->n_break_dim_; ++dim) {
      raft::update_host(this->break_earliest_h.data() + dim * this->n_vehicles,
                        this->break_earliest_d.data() + dim * this->n_vehicles,
                        this->n_vehicles,
                        this->stream_view_);
      raft::update_host(this->break_latest_h.data() + dim * this->n_vehicles,
                        this->break_latest_d.data() + dim * this->n_vehicles,
                        this->n_vehicles,
                        this->stream_view_);
      raft::update_host(this->break_duration_h.data() + dim * this->n_vehicles,
                        this->break_duration_d.data() + dim * this->n_vehicles,
                        this->n_vehicles,
                        this->stream_view_);
    }

    // checks
    this->check_time_windows(h_routing_solution);
    this->check_capacity(h_routing_solution, this->demand_h, this->capacity_h, this->demand_d);
    this->check_vehicle_breaks(h_routing_solution);
    this->check_cost(routing_solution);
  }
};

template <typename i_t, typename f_t>
class routing_retail_test_t : public base_test_t<i_t, f_t>,
                              public ::testing::TestWithParam<retail_params_t> {
 public:
  routing_retail_test_t() : base_test_t<i_t, f_t>(1024) {}

  void SetUp() override
  {
    auto param                 = GetParam();
    this->drop_return_trip_    = param.drop_return_trip;
    this->multi_capacity_      = param.multi_capacity;
    this->vehicle_tw_          = param.vehicle_tw;
    this->vehicle_lower_bound_ = param.vehicle_lower_bound;
    this->pickup_delivery_     = param.pickup;
    this->vehicle_breaks_      = param.vehicle_breaks;
    this->vehicle_max_costs_   = param.vehicle_max_costs;
    this->vehicle_max_times_   = param.vehicle_max_times;
    this->vehicle_fixed_costs_ = param.vehicle_fixed_costs;

    this->n_locations = input_.n_locations;
    this->n_vehicles  = input_.n_vehicles;
    this->n_orders    = this->n_locations;
    this->x_h         = input_.x_h;
    this->y_h         = input_.y_h;
    this->demand_h    = this->pickup_delivery_ ? input_.pickup_delivery_demand_h : input_.demand_h;
    this->capacity_h  = input_.capacity_h;
    this->earliest_time_h =
      this->pickup_delivery_ ? input_.pickup_earliest_time_h : input_.earliest_time_h;
    this->latest_time_h =
      this->pickup_delivery_ ? input_.pickup_latest_time_h : input_.latest_time_h;
    this->service_time_h        = input_.service_time_h;
    this->drop_return_trips_h   = input_.drop_return_h;
    this->skip_first_trips_h    = input_.skip_first_h;
    this->vehicle_earliest_h    = input_.vehicle_earliest_h;
    this->vehicle_latest_h      = input_.vehicle_latest_h;
    this->break_earliest_h      = input_.break_earliest_h;
    this->break_latest_h        = input_.break_latest_h;
    this->break_duration_h      = input_.break_duration_h;
    this->vehicle_types_h       = std::vector<uint8_t>(input_.n_vehicles, 0);
    this->vehicle_fixed_costs_h = input_.vehicle_fixed_costs_h;

    if (this->vehicle_breaks_) {
      this->n_break_locations_ = 5;
      this->n_break_dim_       = 1;
      this->n_locations += this->n_break_locations_;
      for (i_t i = 0; i < this->n_break_locations_; ++i) {
        this->x_h.push_back(50 * i);
        this->y_h.push_back(50 * i);
      }
    }

    this->random_demand_d.resize(this->n_locations, this->stream_view_);
    this->mixed_capacity_d.resize(this->n_vehicles, this->stream_view_);
    this->vehicle_earliest_d.resize(this->n_vehicles, this->stream_view_);
    this->vehicle_latest_d.resize(this->n_vehicles, this->stream_view_);
    this->n_pairs_ = (this->n_orders - 1) / 2;
    this->pickup_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->delivery_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->populate_device_vectors();
  }

  void TearDown() {}

  void test_cvrptw()
  {
    // data model
    // if data_model changes and there are fewer locations than orders
    // adjust the constructor accordingly
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    if (this->pickup_delivery_) {
      raft::copy(this->pickup_indices_d.data(),
                 input_.pickup_indices_h.data(),
                 this->n_pairs_,
                 this->stream_view_);
      raft::copy(this->delivery_indices_d.data(),
                 input_.delivery_indices_h.data(),
                 this->n_pairs_,
                 this->stream_view_);
      data_model.set_pickup_delivery_pairs(this->pickup_indices_d.data(),
                                           this->delivery_indices_d.data());
    }
    if (this->n_break_locations_) {
      thrust::sequence(this->handle_.get_thrust_policy(),
                       this->break_locations_d.begin(),
                       this->break_locations_d.end(),
                       this->n_locations - this->n_break_locations_);
      for (i_t dim = 0; dim < this->n_break_dim_; ++dim) {
        raft::copy(this->break_earliest_d.data() + (dim * this->n_vehicles),
                   this->break_earliest_h.data(),
                   this->break_earliest_h.size(),
                   this->stream_view_);
        raft::copy(this->break_latest_d.data() + (dim * this->n_vehicles),
                   this->break_latest_h.data(),
                   this->break_latest_h.size(),
                   this->stream_view_);
        raft::copy(this->break_duration_d.data() + (dim * this->n_vehicles),
                   this->break_duration_h.data(),
                   this->break_duration_h.size(),
                   this->stream_view_);
      }
      data_model.set_break_locations(this->break_locations_d.data(),
                                     this->break_locations_d.size());
      for (i_t dim = 0; dim < this->n_break_dim_; ++dim) {
        data_model.add_break_dimension(this->break_earliest_d.data() + dim * this->n_vehicles,
                                       this->break_latest_d.data() + dim * this->n_vehicles,
                                       this->break_duration_d.data() + dim * this->n_vehicles);
      }
    }

    if (this->vehicle_tw_) {
      raft::copy(this->vehicle_earliest_d.data(),
                 this->vehicle_earliest_h.data(),
                 input_.n_vehicles,
                 this->stream_view_.value());
      raft::copy(this->vehicle_latest_d.data(),
                 this->vehicle_latest_h.data(),
                 input_.n_vehicles,
                 this->stream_view_.value());
      data_model.set_vehicle_time_windows(this->vehicle_earliest_d.data(),
                                          this->vehicle_latest_d.data());
    }

    data_model.add_cost_matrix(this->cost_matrix_d.data());

    rmm::device_uvector<bool> d_drop_return_trip(input_.n_vehicles, this->stream_view_);
    rmm::device_uvector<bool> d_skip_first_trip(input_.n_vehicles, this->stream_view_);
    if (this->drop_return_trip_) {
      cuda::std::identity id;
      rmm::device_uvector<i_t> d_int_drop_return_trip(input_.n_vehicles, this->stream_view_);
      rmm::device_uvector<i_t> d_int_skip_first_trip(input_.n_vehicles, this->stream_view_);
      raft::copy(d_int_drop_return_trip.data(),
                 this->drop_return_trips_h.data(),
                 input_.n_vehicles,
                 this->stream_view_.value());
      thrust::transform(this->handle_.get_thrust_policy(),
                        d_int_drop_return_trip.begin(),
                        d_int_drop_return_trip.end(),
                        d_drop_return_trip.begin(),
                        id);

      raft::copy(d_int_skip_first_trip.data(),
                 this->skip_first_trips_h.data(),
                 input_.n_vehicles,
                 this->stream_view_.value());
      thrust::transform(this->handle_.get_thrust_policy(),
                        d_int_skip_first_trip.begin(),
                        d_int_skip_first_trip.end(),
                        d_skip_first_trip.begin(),
                        id);
      RAFT_CUDA_TRY(cudaStreamSynchronize(this->stream_view_.value()));
      data_model.set_drop_return_trips(d_drop_return_trip.data());
      data_model.set_skip_first_trips(d_skip_first_trip.data());
    }

    std::vector<i_t> shuffled_vec(this->demand_h);
    if (this->multi_capacity_) {
      // generate shuffled demands
      if (!this->pickup_delivery_) {
        std::mt19937 mersenne_engine{12345u};  // Generates random integer
        std::shuffle(shuffled_vec.begin() + 1, shuffled_vec.end(), mersenne_engine);
      }
      raft::copy(this->random_demand_d.data(),
                 shuffled_vec.data(),
                 this->n_orders,
                 this->stream_view_.value());
      raft::copy(this->mixed_capacity_d.data(),
                 input_.mixed_capacity_h.data(),
                 this->n_vehicles,
                 this->stream_view_.value());
      data_model.add_capacity_dimension(
        "random", this->random_demand_d.data(), this->mixed_capacity_d.data());
    }
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());

    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    routing::solver_settings_t<i_t, f_t> settings;
    if (this->vehicle_lower_bound_) data_model.set_min_vehicles(this->vehicle_lower_bound_);

    if (this->vehicle_max_costs_) {
      auto max_dist_depot    = thrust::reduce(this->handle_.get_thrust_policy(),
                                           this->cost_matrix_d.data(),
                                           this->cost_matrix_d.data() + this->n_locations,
                                           -1,
                                           thrust::maximum<f_t>());
      auto vehicle_max_costs = max_dist_depot * (2 + (i_t)this->pickup_delivery_) + 2;
      thrust::fill(this->handle_.get_thrust_policy(),
                   this->vehicle_max_costs_d.begin(),
                   this->vehicle_max_costs_d.end(),
                   vehicle_max_costs);
      data_model.set_vehicle_max_costs(this->vehicle_max_costs_d.data());
    }
    if (this->vehicle_max_times_) {
      data_model.add_transit_time_matrix(this->cost_matrix_d.data());
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
      auto max_dist_depot    = thrust::reduce(this->handle_.get_thrust_policy(),
                                           this->cost_matrix_d.data(),
                                           this->cost_matrix_d.data() + this->n_locations,
                                           -1,
                                           thrust::maximum<f_t>());
      auto vehicle_max_times = max_dist_depot * (2 + (i_t)this->pickup_delivery_) + 2 +
                               (1 + (i_t)this->pickup_delivery_) * max_service;
      thrust::fill(this->handle_.get_thrust_policy(),
                   this->vehicle_max_times_d.begin(),
                   this->vehicle_max_times_d.end(),
                   vehicle_max_times);
      // add max service time to vehicle max time
      data_model.set_vehicle_max_times(this->vehicle_max_times_d.data());
    }

    if (this->vehicle_fixed_costs_) {
      raft::copy(this->vehicle_fixed_costs_d.data(),
                 this->vehicle_fixed_costs_h.data(),
                 this->vehicle_fixed_costs_h.size(),
                 this->stream_view_);
      data_model.set_vehicle_fixed_costs(this->vehicle_fixed_costs_d.data());
    }

    settings.set_time_limit(this->n_orders / 5);

    // solve
    // FIXME:: Do not construct solver object here,
    // doing it just for now so that we can extract is_vanilla_pdp
    routing::solver_t<i_t, f_t> solver(data_model, settings);
    auto routing_solution = solver.solve();
    int v_count           = routing_solution.get_vehicle_count();
    float cost            = routing_solution.get_total_objective();
    std::cout << "Vehicle: " << v_count << " Cost: " << cost << "\n";
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    int returned_vehicle_count = routing_solution.get_vehicle_count();
    ASSERT_LE(this->vehicle_lower_bound_, returned_vehicle_count);
    // checks
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution);
    // check weight
    this->check_capacity(h_routing_solution, this->demand_h, input_.capacity_h, this->demand_d);
    // check random
    if (this->multi_capacity_) {
      this->check_capacity(
        h_routing_solution, shuffled_vec, input_.mixed_capacity_h, this->random_demand_d);
    }

    if (this->vehicle_tw_) {
      this->check_vehicle_time_windows(
        h_routing_solution, this->vehicle_earliest_h, this->vehicle_latest_h);
    }
    if (this->n_break_locations_) this->check_vehicle_breaks(h_routing_solution);
  }
};

typedef l0_routing_test_t<int, float> l0_float_test_t;
typedef routing_retail_test_t<int, float> retail_float_test_t;

TEST_P(l0_float_test_t, TSP) { test_tsp(); }
TEST_P(l0_float_test_t, VRP) { test_vrp(); }
TEST_P(l0_float_test_t, CVRP) { test_cvrp(); }
TEST_P(l0_float_test_t, VRPTW) { test_vrptw(); }
TEST_P(l0_float_test_t, CVRPTW) { test_cvrptw(); }
INSTANTIATE_TEST_SUITE_P(level0_base, l0_float_test_t, ::testing::Values(""));

TEST_P(retail_float_test_t, CVRPTW_Retail) { test_cvrptw(); }
INSTANTIATE_TEST_SUITE_P(
  level0_retail,
  retail_float_test_t,
  ::testing::Values(
    retail_params_t{},                       // 0
                                             // multi_capacity test
    retail_params_t{}.set_multi_capacity(),  // 1
    // vehicle_tw test
    retail_params_t{}.set_vehicle_tw(),  // 2
    // test with drop return and skip first trip
    retail_params_t{}.set_drop_return_trip(),  // 3
    // test with multiple capacity dimensions and vehicle tw
    retail_params_t{}.set_multi_capacity().set_vehicle_tw(),  // 4
    // vehicle_count test
    retail_params_t{}.set_vehicle_lower_bound(15),  // 5
    // pickup_delivery with vehicle count
    retail_params_t{}.set_vehicle_lower_bound(20).set_pickup(),  // 6
    // pickup_delivery test
    retail_params_t{}.set_pickup(),  // 7
    // pickup_delivery with multi cap
    retail_params_t{}.set_pickup().set_multi_capacity(),  // 8
    // break locations
    retail_params_t{}.set_vehicle_breaks(),  // 9
    // break location with pickup_delivery
    retail_params_t{}.set_vehicle_breaks().set_pickup(),  // 10
    // vehicle_max_costs
    retail_params_t{}.set_vehicle_max_costs(),  // 11
    // vehicle_max_costs and drop frirst/return trips
    retail_params_t{}.set_vehicle_max_costs().set_drop_return_trip(),  // 12
    // max_distance and pickup
    retail_params_t{}.set_vehicle_max_costs().set_pickup(),  // 13
    // vehicle_max_times
    retail_params_t{}.set_vehicle_max_times(),  // 14
    // vehicle max times and pickup
    retail_params_t{}.set_vehicle_max_times().set_pickup(),  // 15
    // vehicle_max_costs and vehicle_max_times
    retail_params_t{}.set_vehicle_max_costs().set_vehicle_max_times(),  // 16
    // vehicle fixed costs
    retail_params_t{}.set_vehicle_fixed_costs(),  // 17
    // vehicle fixed costs with multi-capacity and vehicle tw
    retail_params_t{}.set_vehicle_fixed_costs().set_multi_capacity().set_vehicle_tw()  // 18
    ));
}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
