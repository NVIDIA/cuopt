/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/routing_test.cuh>

#include <routing/adapters/solution_adapter.cuh>
#include <routing/ges_solver.cuh>
#include <routing/routing_helpers.cuh>

#include <routing/utilities/data_model.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

namespace cuopt {
namespace routing {
namespace test {

enum class test_t { ONLY_CYCLE = 0, LOCAL_SEARCH_UNTIL_END, ACYCLIC, RANDOM_CYCLE, INFEASIBLE };

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void introduce_infeasibility(typename detail::solution_t<i_t, f_t, REQUEST>::view_t sol)
{
  // insert the items of the first route to other routes in round robin fashion
  // leave only 1 request not the have route reduction
  auto& first_route      = sol.routes[0];
  i_t n_ejected          = 0;
  i_t route_id_to_insert = 1;
  for (i_t i = 1; i < first_route.get_num_nodes() - n_ejected * 2 - 6; ++i) {
    auto pickup_node = first_route.get_node(i);
    if (!pickup_node.request.is_pickup()) continue;
    i_t pair_index     = sol.intra_route_idx_per_node[pickup_node.request.brother_id()];
    auto delivery_node = first_route.get_node(pair_index);
    first_route.eject_node(i, sol.route_id_per_node, sol.intra_route_idx_per_node);
    first_route.eject_node(pair_index - 1, sol.route_id_per_node, sol.intra_route_idx_per_node);
    route_id_to_insert = (n_ejected % (sol.n_routes - 1)) + 1;
    assert(route_id_to_insert > 0);
    auto request_node = detail::request_node_t<i_t, f_t, REQUEST>(pickup_node, delivery_node);
    // insert it in the beginning
    detail::request_id_t<REQUEST> request_id(0, 0);
    sol.routes[route_id_to_insert].insert_request(
      request_id, request_node, sol.route_id_per_node, sol.intra_route_idx_per_node);
    ++n_ejected;
    --i;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
class routing_ges_test_t : public ::testing::TestWithParam<std::tuple<bool, test_t>>,
                           public base_test_t<i_t, f_t> {
 public:
  routing_ges_test_t() : base_test_t<i_t, f_t>(4) {}

  void SetUp() override
  {
    this->pickup_delivery_ = std::get<0>(GetParam());
    this->test_type        = std::get<1>(GetParam());

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
    this->pickup_indices_h    = input_double_.pickup_indices_h;
    this->delivery_indices_h  = input_double_.delivery_indices_h;
    this->vehicle_types_h.assign(this->n_vehicles, 0);

    this->n_pairs_ = (this->n_orders - 1) / 2;
    this->pickup_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->delivery_indices_d.resize(this->n_pairs_, this->stream_view_);
    this->populate_device_vectors();
  }

  void TearDown() {}

  // IN ORDER TO EXCLUDE THE CPU SOLVER DEPENDENCY WE COMMENT THIS OUT
  // I THINK WE SHOULD KEEP IT FOR SOME TIME UNTIL WE ARE SURE IT WORKS CORRECTLY

  // minisolver::Problem get_problem()
  // {
  //   minisolver::Problem P(this->n_locations);
  //   auto n_pd_request = (this->n_locations - 1) / 2;
  //   for (auto i = 0; i < n_pd_request; ++i) {
  //     auto pickup_id   = this->pickup_indices_h[i];
  //     auto delivery_id = this->delivery_indices_h[i];
  //     auto pickup_earliest =
  //       this->earliest_time_h[pickup_id] + this->service_time_h[pickup_id];
  //     auto pickup_latest =
  //       this->latest_time_h[pickup_id] + this->service_time_h[pickup_id];
  //     auto delivery_earliest = this->earliest_time_h[delivery_id] +
  //                              this->service_time_h[delivery_id];
  //     auto delivery_latest =
  //       this->latest_time_h[delivery_id] + this->service_time_h[delivery_id];
  //     auto pickup_demand   = this->demand_h[pickup_id];
  //     auto delivery_demand = this->demand_h[delivery_id];
  //     // Adjust time window to include service time
  //     P.set_time_window(pickup_id, pickup_earliest, pickup_latest);
  //     P.set_time_window(delivery_id, delivery_earliest, delivery_latest);
  //     P.set_node_demand(pickup_id, pickup_demand);
  //     P.set_node_demand(delivery_id, delivery_demand);
  //     P.set_brother_id(pickup_id, delivery_id);
  //     P.set_brother_id(delivery_id, pickup_id);

  //     for (auto j = 0; j < this->n_locations; ++j) {
  //       double dist = this->cost_matrix_h[pickup_id * this->n_locations + j];
  //       P.set_distance_between(pickup_id, j, dist);
  //       // Add service time to travel time
  //       P.set_time_between(pickup_id, j, dist + this->service_time_h[j]);
  //       dist = this->cost_matrix_h[delivery_id * this->n_locations + j];
  //       P.set_distance_between(delivery_id, j, dist);
  //       // Add service time to travel time
  //       P.set_time_between(delivery_id, j, dist + this->service_time_h[j]);
  //     }
  //   }
  //   auto depot_earliest = this->earliest_time_h[0] + this->service_time_h[0];
  //   auto depot_latest   = this->latest_time_h[0] + this->service_time_h[0];
  //   P.set_time_window(0, depot_earliest, depot_latest);
  //   P.set_node_demand(0, 0);
  //   P.set_brother_id(0, 0);

  //   for (auto j = 0; j < this->n_locations; ++j) {
  //     double dist = this->cost_matrix_h[0 * this->n_locations + j];
  //     P.set_distance_between(0, j, dist);
  //     // Add service time to travel time
  //     P.set_time_between(0, j, dist + this->service_time_h[j]);
  //   }
  //   P.set_vehicle_capacity(this->capacity_h[0]);
  //   return P;
  // }

  // minisolver::Solution get_minisolver_solution(minisolver::Problem* P,
  //                                              const std::vector<i_t>& route)
  // {
  //   minisolver::Solution S(P);
  //   std::vector<std::vector<int>> inst_data;
  //   bool route_opened = false;
  //   i_t route_id      = -1;
  //   i_t curr_node     = -1;
  //   for (auto i = 0; i < route.size(); ++i) {
  //     curr_node = route[i];
  //     if (curr_node == 0) {
  //       if (!route_opened) {
  //         ++route_id;
  //         inst_data.push_back(std::vector<i_t>());
  //         route_opened = true;
  //       } else {
  //         route_opened = false;
  //       }
  //       // depots are not included
  //       continue;
  //     }
  //     inst_data[route_id].push_back(curr_node);
  //   }
  //   for (auto i = 0; i < inst_data.size(); ++i) {
  //     S.add_new_route(inst_data[i]);
  //   }
  //   return S;
  // }

  // void compare_to_cpu(const std::vector<i_t>& route, f_t gpu_cost)
  // {
  //   minisolver::Problem P  = get_problem();
  //   minisolver::Solution S = get_minisolver_solution(&P, route);
  //   EXPECT_FLOAT_EQ(S.actual_cost[0], gpu_cost);
  //   minisolver::CyclesGraph graph;
  //   costs weights({1.0, 0.0, 0.0, 0.0, 0.0});
  //   graph.initialize_pdp_graph_2_2(P, S);
  //   graph.find_relocate_2_2 (P, S, weights);
  //   graph.find_rem1_add_12(P, S, weights, false);

  //   CycleFinder::ExactCycleFinder<128> E(graph.Graph.size());
  //   E.reset_graph(graph.Graph,graph.ids);
  //   auto ret                   = E.find_best_cycles(graph.pseudo_nodes_number);

  //   auto cpu_negative_cycles   = std::get<0>(ret);
  //   auto cpu_best_cycles       = std::get<1>(ret);
  //   auto gpu_negative_cycles   = std::get<0>(this->cycles);
  //   auto gpu_best_cycles       = std::get<1>(this->cycles);
  //   auto n_negative_cycles_cpu = cpu_negative_cycles.size();
  //   auto n_best_cycles_cpu     = cpu_best_cycles.size();
  //   auto n_negative_cycles_gpu = gpu_negative_cycles.size();
  //   auto n_best_cycles_gpu     = gpu_best_cycles.size();
  //   ASSERT_EQ(n_negative_cycles_cpu, n_negative_cycles_gpu);
  //   ASSERT_EQ(n_best_cycles_cpu, n_best_cycles_gpu);
  //   // check whether the cycle is the same, check the weight
  //   // check whether the order of the cycles in the vector are the same
  //   for (auto i = 0; i < n_negative_cycles_cpu; ++i) {
  //     EXPECT_FLOAT_EQ(std::get<0>(cpu_negative_cycles[i]), std::get<0>(gpu_negative_cycles[i]));
  //     ASSERT_EQ(std::get<2>(cpu_negative_cycles[i]), std::get<2>(gpu_negative_cycles[i]));
  //   }
  //   // best cycles are currently wrong since CPU finds more paths
  //   // for(auto i = 0 ; i < n_best_cycles_cpu ; ++i){
  //   //   EXPECT_FLOAT_EQ(std::get<0>(cpu_best_cycles[i]) , std::get<0>(gpu_best_cycles[i]));
  //   //   ASSERT_EQ(std::get<2>(cpu_best_cycles[i]) , std::get<2>(gpu_best_cycles[i]));
  //   // }
  // }

  // creates a simple assignment object that allows acyclic moves
  assignment_t<i_t> get_simple_assignment(const std::vector<i_t>& route,
                                          const std::vector<i_t>& route_locations,
                                          const std::vector<i_t>& node_types)
  {
    std::vector<i_t> truck_id = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
    std::vector<f_t> stamp    = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto route_out           = cuopt::device_copy(route, this->stream_view_);
    auto arrival_out         = cuopt::device_copy(stamp, this->stream_view_);
    auto truck_id_out        = cuopt::device_copy(truck_id, this->stream_view_);
    auto route_locations_out = cuopt::device_copy(route_locations, this->stream_view_);
    auto node_types_out      = cuopt::device_copy(node_types, this->stream_view_);

    return assignment_t<i_t>(3,
                             0,
                             0,
                             route_out,
                             arrival_out,
                             truck_id_out,
                             route_locations_out,
                             node_types_out,
                             solution_status_t::SUCCESS);
  }

  void check_acyclic()
  {
    this->n_locations = 17;
    this->n_vehicles  = 10;
    this->n_orders    = 17;
    this->n_pairs_    = 8;
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    this->x_h = {
      0, -500, -550, -600, -650, -500, -550, 500, 500, -500, -550, -600, -650, 500, 500, 500, 500};
    this->y_h = {
      0, 1000, 1000, 1000, 1000, -500, -550, 500, 500, -500, -550, -600, -650, 500, 500, 500, 500};
    this->demand_h   = {0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
    this->capacity_h = {10, 10, 10};
    this->earliest_time_h.assign(this->n_orders, 0);
    this->service_time_h.assign(this->n_orders, 0);
    this->latest_time_h.assign(this->n_orders, std::numeric_limits<int>::max());
    this->populate_device_vectors();

    this->pickup_indices_h   = {1, 3, 5, 7, 9, 11, 13, 15};
    this->delivery_indices_h = {2, 4, 6, 8, 10, 12, 14, 16};
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
    data_model.add_transit_time_matrix(this->cost_matrix_d.data());
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    detail::problem_t<i_t, f_t> problem(data_model);
    std::unique_ptr<detail::solution_handle_t<i_t, f_t>> sol_handle =
      std::make_unique<detail::solution_handle_t<i_t, f_t>>(problem.handle_ptr->get_stream());
    std::vector<detail::solution_t<i_t, f_t, REQUEST>> sol_pool_vec = {
      detail::solution_t<i_t, f_t, REQUEST>(problem, 0, sol_handle.get())};
    std::vector<i_t> route           = {0, 1,  2,  3,  4, 5, 6,  0,  0,  7,  8,
                                        9, 10, 11, 12, 0, 0, 13, 14, 15, 16, 0};
    std::vector<i_t> route_locations = route;  // assume one to one map

    constexpr i_t DEP  = (i_t)node_type_t::DEPOT;
    constexpr i_t PICK = (i_t)node_type_t::PICKUP;
    constexpr i_t DEL  = (i_t)node_type_t::DELIVERY;

    std::vector<i_t> node_types = {DEP,  PICK, DEL,  PICK, DEL, PICK, DEL,  DEP, DEP,  PICK, DEL,
                                   PICK, DEL,  PICK, DEL,  DEP, DEP,  PICK, DEL, PICK, DEL,  DEP};
    auto simple_assignment      = get_simple_assignment(route, route_locations, node_types);
    detail::get_solution_from_assignment(sol_pool_vec[0], simple_assignment, problem);
    ges_solver_t<i_t, f_t, REQUEST> ges_solver(data_model, 1, this->n_orders / 5);
    ges_solver.pool_allocator.solution_pool = std::move(sol_pool_vec);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    const i_t max_iterations = 1;

    this->hr_timer_.start("GES solver");
    detail::local_search_t<i_t, f_t, REQUEST>::start_timer(1.f);
    auto& sol                  = ges_solver.pool_allocator.solution_pool[0];
    auto [resource, index]     = ges_solver.pool_allocator.resource_pool->acquire();
    resource.ls.max_iterations = max_iterations;
    resource.ls.run_best_local_search(sol, false, false);
    ges_solver.pool_allocator.resource_pool->release(index);
    ges_solver.pool_allocator.sync_all_streams();
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    this->hr_timer_.stop();
    this->hr_timer_.display(std::cout);
  }

  assignment_t<i_t> solve(const cuopt::routing::data_model_view_t<i_t, f_t>& data_model_const)
  {
    constexpr i_t n_sol  = 1;
    constexpr i_t n_secs = 1;

    // Copy
    auto data_model = data_model_const;
    cuopt::routing::solver_settings_t<i_t, f_t> settings;
    settings.set_time_limit(n_secs);

    // solve
    // cuopt::routing::solver_t solver(data_model, settings);
    auto routing_solution = base_test_t<i_t, f_t>::solve(data_model, settings);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    detail::problem_t<i_t, f_t> problem(data_model_const);

    ges_solver_t<i_t, f_t, REQUEST> ges_solver(data_model_const, 1, this->n_orders / 5);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    printf("solver created\n");
    detail::get_solution_from_assignment(
      ges_solver.pool_allocator.solution_pool[0], routing_solution, problem);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    printf("assignment received\n");
    ges_solver.pool_allocator.solution_pool[0].global_runtime_checks(
      false, false, "local_search_test_1");
    const i_t max_iterations =
      this->test_type == test_t::ONLY_CYCLE ? 1 : std::numeric_limits<i_t>::max();
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    auto& sol = ges_solver.pool_allocator.solution_pool[0];
    sol.global_runtime_checks(false, false, "local_search_test_2");
    printf("test start\n");
    this->hr_timer_.start("GES solver");
    if (this->test_type == test_t::RANDOM_CYCLE) {
      // this call might change later
      auto [resource, index]     = ges_solver.pool_allocator.resource_pool->acquire();
      resource.ls.max_iterations = max_iterations;
      resource.ls.run_random_local_search(sol, true);
      ges_solver.pool_allocator.resource_pool->release(index);
    } else if (this->test_type == test_t::INFEASIBLE) {
      double w[] = {100., 10000., 100., 100., 100.};
      introduce_infeasibility<i_t, f_t, REQUEST>
        <<<1, 1, 0, sol.sol_handle->get_stream()>>>(sol.view());
      sol.set_nodes_data_of_solution();
      sol.compute_initial_data();
      f_t old_cost = sol.get_total_cost(w);
      detail::local_search_t<i_t, f_t, REQUEST>::start_timer(20.f);
      auto [resource, index]     = ges_solver.pool_allocator.resource_pool->acquire();
      resource.ls.max_iterations = max_iterations;
      resource.ls.set_active_weights(w, 10000000.f);
      resource.ls.run_best_local_search(sol, false, true);
      ges_solver.pool_allocator.resource_pool->release(index);
      // if we can't make it feasible at the end, we can meaure the infeasibility diff
      sol.compute_cost();
      f_t improved_cost = sol.get_total_cost(w);
      bool is_feasible  = sol.is_feasible();
      EXPECT_TRUE(old_cost >= improved_cost);
    } else {
      bool compute_initial_solution = false;
      ges_solver.compute_ges_solution(compute_initial_solution);
    }
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    auto assignment = ges_solver.get_ges_assignment(ges_solver.pool_allocator.solution_pool[0]);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    // this->cycles = ges_solver.local_search.cycles;
    // compare generated graph (single iteration single node) to cpu solution
    // this might move to some other test case
    // if (this->test_type == test_t::ONLY_CYCLE) {
    //   compare_to_cpu(h_routing_solution.route, routing_solution.get_total_objective());
    // }
    this->hr_timer_.stop();
    this->hr_timer_.display(std::cout);
    return assignment;
  }

  void test_cvrptw()
  {
    if (this->test_type == test_t::ACYCLIC) {
      check_acyclic();
      return;
    }

    // data model
    // if data_model changes and there are fewer locations than orders
    // adjust the constructor accordingly
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);

    if (this->pickup_delivery_) {
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
    data_model.add_transit_time_matrix(this->cost_matrix_d.data());
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    // solve
    auto routing_solution = this->solve(data_model);
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    i_t v_count = routing_solution.get_vehicle_count();
    f_t cost    = routing_solution.get_total_objective();
    std::cout << "Vehicle: " << v_count << " Cost: " << cost << "\n";
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution, false);
    // check weight
    this->check_capacity(
      h_routing_solution, this->demand_h, input_double_.capacity_h, this->demand_d);
  }

  std::pair<std::vector<std::tuple<double, double, std::deque<int>>>,
            std::vector<std::tuple<double, double, std::deque<int>>>>
    cycles;
  test_t test_type;
};

typedef routing_ges_test_t<int, float, request_t::PDP> double_test;

TEST_P(double_test, CVRPTW_GES) { test_cvrptw(); }
// INSTANTIATE_TEST_SUITE_P(candidate_unit_test,
//                          double_test,
//                          ::testing::Values(std::make_tuple(true, test_t::ONLY_CYCLE),
//                                            std::make_tuple(true, test_t::LOCAL_SEARCH_UNTIL_END),
//                                            //  best cycle finder is integrated
//                                            std::make_tuple(true, test_t::RANDOM_CYCLE)));

INSTANTIATE_TEST_SUITE_P(acyclic_simple_test,
                         double_test,
                         ::testing::Values(std::make_tuple(true, test_t::ACYCLIC)));

// INSTANTIATE_TEST_SUITE_P(infeasibility_test,
//                          double_test,
//                          ::testing::Values(std::make_tuple(true, test_t::INFEASIBLE)));

}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
