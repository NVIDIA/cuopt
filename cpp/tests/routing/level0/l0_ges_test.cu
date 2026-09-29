/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/routing_test.cuh>

#include <routing/ges/lexicographic_search/node_stack.cuh>
#include <routing/ges_solver.cuh>
#include <routing/problem/problem.cuh>
#include <routing/solution/solution_handle.cuh>
#include <routing/utilities/data_model.hpp>
#include <utilities/copy_helpers.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

namespace cuopt {
namespace routing {
namespace test {

namespace {

__global__ void copy_ges_distance_forward_kernel(detail::enabled_dimensions_t dimensions,
                                                 double* copied_distances)
{
  using node_t       = detail::node_t<int, float, request_t::VRP>;
  using node_stack_t = detail::node_stack_t<int, float, request_t::VRP>;

  node_t source(dimensions);
  node_t node_destination(dimensions);
  node_t second_node_destination(dimensions);
  typename node_stack_t::item_t item{};
  typename node_stack_t::item_t second_item{};

  source.request =
    detail::request_info_t<int, request_t::VRP>(detail::NodeInfo<int>{0, 0, node_type_t::DEPOT});
  source.cost_dim.distance_forward = 37.0;

  item.intra_idx        = 0;
  item.from_idx         = 0;
  second_item.intra_idx = 0;
  second_item.from_idx  = 0;

  item                = source;
  copied_distances[0] = item.distance_forward;
  second_item         = item;
  copied_distances[1] = second_item.distance_forward;
  detail::copy_forward_data(node_destination, second_item);
  copied_distances[2] = node_destination.cost_dim.distance_forward;
  detail::copy_forward_data(second_node_destination, source);
  copied_distances[3] = second_node_destination.cost_dim.distance_forward;
}

__global__ void get_ges_direct_distance_kernel(float const* matrices, double* distances)
{
  if (threadIdx.x == 0) {
    mdarray_view_t<float> matrix_view;
    matrix_view.buffer_ptr            = matrices;
    matrix_view.extent[0]             = 1;
    matrix_view.extent[1]             = 2;
    matrix_view.extent[2]             = 4;
    matrix_view.extent[3]             = 4;
    matrix_view.cost_matrix_index     = 0;
    matrix_view.distance_matrix_index = 1;

    detail::VehicleInfo<float> vehicle_info;
    vehicle_info.matrices     = matrix_view;
    vehicle_info.max_distance = 10000.f;

    const detail::NodeInfo<int> from{0, 0, node_type_t::DEPOT};
    const detail::NodeInfo<int> via{1, 1, node_type_t::PICKUP};
    const detail::NodeInfo<int> to{2, 2, node_type_t::PICKUP};
    distances[0] = detail::node_stack_t<int, float, request_t::VRP>::get_travel_distance_between(
      from, to, vehicle_info);
    distances[1] = detail::get_travel_distance(from, to, vehicle_info);
    distances[2] = detail::get_travel_distance(from, via, vehicle_info) +
                   detail::get_travel_distance(via, to, vehicle_info);
  }
}

TEST(ges_node_stack, copies_distance_forward_in_all_directions)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();
  detail::enabled_dimensions_t dimensions;
  dimensions.enable_dimension(detail::dim_t::COST);
  rmm::device_uvector<double> copied_distances(4, stream);

  copy_ges_distance_forward_kernel<<<1, 1, 0, stream.get()>>>(dimensions, copied_distances.data());
  RAFT_CHECK_CUDA(stream.get());
  auto host_distances = cuopt::host_copy(copied_distances, stream);

  EXPECT_EQ(host_distances, (std::vector<double>{37.0, 37.0, 37.0, 37.0}));
}

TEST(ges_node_stack, uses_direct_arc_from_separate_distance_matrix)
{
  constexpr int n_locations = 4;
  constexpr int n_threads   = 32;

  std::vector<float> cost_matrix(n_locations * n_locations, 1.f);
  std::vector<float> distance_matrix(n_locations * n_locations);
  for (int from = 0; from < n_locations; ++from) {
    for (int to = 0; to < n_locations; ++to) {
      const auto index       = from * n_locations + to;
      cost_matrix[index]     = from == to ? 0.f : 1.f;
      distance_matrix[index] = from == to ? 0.f : 10.f * from + to + 1.f;
    }
  }

  std::vector<float> matrices = cost_matrix;
  matrices.insert(matrices.end(), distance_matrix.begin(), distance_matrix.end());

  raft::handle_t handle;
  auto stream     = handle.get_stream();
  auto d_matrices = cuopt::device_copy(matrices, stream);
  rmm::device_uvector<double> distances(3, stream);

  get_ges_direct_distance_kernel<<<1, n_threads, 0, stream.get()>>>(d_matrices.data(),
                                                                    distances.data());
  RAFT_CHECK_CUDA(stream.get());
  auto host_distances = cuopt::host_copy(distances, stream);

  EXPECT_DOUBLE_EQ(host_distances[0], host_distances[1]);
  EXPECT_NE(host_distances[1], host_distances[2]);
}

}  // namespace

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
