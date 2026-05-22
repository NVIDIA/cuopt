/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/routing_test.cuh>

#include <routing/adapters/solution_adapter.cuh>
#include <routing/ges_solver.cuh>
#include <routing/local_search/compute_insertions.cuh>
#include <routing/utilities/data_model.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t, request_t REQUEST>
class simple_scross_test_t : public ::testing::TestWithParam<test_data_t<i_t, f_t>>,
                             public base_test_t<i_t, f_t> {
 public:
  simple_scross_test_t() : base_test_t<i_t, f_t>(4) {}

  void SetUp() override
  {
    const auto& param = this->GetParam();

    this->pickup_delivery_ = true;
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

    this->matrices_h   = detail::create_host_mdarray<f_t>(this->n_locations,
                                                        this->n_vehicle_types_,
                                                        1 + this->use_secondary_matrix_,
                                                        this->stream_view_);
    f_t* cost_matrix_h = this->matrices_h.get_cost_matrix(0);
    for (int i = 0; i < (this->n_locations * this->n_locations); ++i)
      cost_matrix_h[i] = param.cost_matrix_h[i];
    cost_matrix_h = this->matrices_h.get_cost_matrix(1);
    for (int i = 0; i < (this->n_locations * this->n_locations); ++i)
      cost_matrix_h[i] = param.cost_matrix_h[i];
    this->populate_device_vectors();
  }

  void TearDown() {}

  // creates a simple assignment object that allows scross
  assignment_t<i_t> get_simple_assignment(const std::vector<i_t>& route)
  {
    std::vector<i_t> truck_id(route.size());
    int route_id = -1;
    for (size_t i = 0; i < route.size() - 1; ++i) {
      if (route[i] == 0) ++route_id;
      truck_id[i] = route_id;
      if (route[i + 1] == 0) {
        truck_id[i + 1] = route_id;
        ++i;
      }
    }

    std::vector<f_t> stamp(route.size());
    for (size_t i = 0; i < route.size(); ++i)
      stamp[i] = 0;

    rmm::device_uvector<i_t> route_out(route.size(), this->stream_view_);
    rmm::device_uvector<f_t> arrival_out(route.size(), this->stream_view_);
    rmm::device_uvector<i_t> truck_id_out(route.size(), this->stream_view_);
    rmm::device_uvector<i_t> route_locations_out(route.size(), this->stream_view_);
    rmm::device_uvector<i_t> node_types_out(route.size(), this->stream_view_);

    raft::copy(route_out.data(), route.data(), route.size(), this->stream_view_);
    raft::copy(arrival_out.data(), stamp.data(), route.size(), this->stream_view_);
    raft::copy(truck_id_out.data(), truck_id.data(), route.size(), this->stream_view_);
    raft::copy(route_locations_out.data(), route.data(), route.size(), this->stream_view_);

    // FIXME: wire the node types
    thrust::fill(
      this->handle_.get_thrust_policy(), node_types_out.begin(), node_types_out.end(), 0);

    return assignment_t<i_t>(route_id,
                             0,
                             0,
                             route_out,
                             arrival_out,
                             truck_id_out,
                             route_locations_out,
                             node_types_out,
                             solution_status_t::SUCCESS);
  }

  void solve(const cuopt::routing::data_model_view_t<i_t, f_t>& data_model,
             i_t expected_route_count)
  {
    cudaDeviceSynchronize();

    detail::problem_t<i_t, f_t> problem(data_model);
    std::unique_ptr<detail::solution_handle_t<i_t, f_t>> sol_handle =
      std::make_unique<detail::solution_handle_t<i_t, f_t>>(problem.handle_ptr->get_stream());
    std::vector<detail::solution_t<i_t, f_t, REQUEST>> sol_pool_vec = {
      detail::solution_t<i_t, f_t, REQUEST>(problem, 0, sol_handle.get(), expected_route_count, 6)};
    // clang-format off
    std::vector<i_t> route = {0, 1, 2, 7, 8, 0,
                            0, 3, 4, 0,
                            0, 5, 6, 0};
    // clang-format on
    auto simple_assignment = get_simple_assignment(route);
    detail::get_solution_from_assignment(sol_pool_vec[0], simple_assignment, problem);
    ges_solver_t<i_t, f_t, REQUEST> ges_solver(data_model, 1, this->n_orders / 5);
    ges_solver.pool_allocator.solution_pool = std::move(sol_pool_vec);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    this->hr_timer_.start("PDP scross");
    detail::local_search_t<i_t, f_t, REQUEST>::start_timer(1.f);
    auto& sol                           = ges_solver.pool_allocator.solution_pool[0];
    auto [resource, index]              = ges_solver.pool_allocator.resource_pool->acquire();
    constexpr double default_weights[5] = {1., 1., 1., 1., 1.};
    resource.ls.set_active_weights(default_weights, 0.0f);
    resource.ls.calculate_route_compatibility(sol);
    const i_t n_ejections = 1;
    auto& move_candidates = resource.ls.move_candidates;

    detail::search_type_t search_type = detail::search_type_t::CROSS;
    detail::find_insertions<i_t, f_t>(sol, move_candidates, n_ejections, search_type);
    resource.ls.populate_cross_moves(sol, move_candidates);
    assert(move_candidates.move_path.n_insertions.value(sol.sol_handle->get_stream()) > 0);
    resource.ls.perform_moves(sol, move_candidates);

    // Check route
    ges_solver.pool_allocator.sync_all_streams();
    this->handle_.sync_stream();
    this->hr_timer_.stop();
    auto routing_solution = ges_solver.get_ges_assignment(sol);
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);

    ges_solver.pool_allocator.resource_pool->release(index);
    ges_solver.pool_allocator.sync_all_streams();
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    this->hr_timer_.stop();
    this->hr_timer_.display(std::cout);
  }

  void test_scross()
  {
    // data model
    // if data_model changes and there are fewer locations than orders
    // adjust the constructor accordingly
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    if (this->pickup_delivery_) {
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

    // solve
    const i_t expected_route_count = 3;
    this->solve(data_model, expected_route_count);
  }
};

typedef simple_scross_test_t<int, float, request_t::PDP> scross_three_routes_test;

TEST_P(scross_three_routes_test, SCROSS_GES) { test_scross(); }
INSTANTIATE_TEST_SUITE_P(simple_scross_test,
                         scross_three_routes_test,
                         ::testing::ValuesIn(parse_problems(scross_three_routes_)));

}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
