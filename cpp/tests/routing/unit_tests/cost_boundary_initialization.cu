/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <routing/problem/problem.cuh>
#include <routing/solution/solution.cuh>
#include <routing/solution/solution_handle.cuh>
#include <routing/util_kernels/set_nodes_data.cuh>
#include <utilities/copy_helpers.hpp>

#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt {
namespace routing {
namespace test {

template <request_t REQUEST>
__global__ void set_route_data_kernel(typename detail::solution_t<int, float, REQUEST>::view_t sol,
                                      const typename detail::problem_t<int, float>::view_t problem)
{
  detail::set_route_data<int, float, REQUEST>(problem, sol.routes[0]);
}

template <request_t REQUEST>
void test_cost_boundaries_reset()
{
  constexpr bool is_pdp       = REQUEST == request_t::PDP;
  const int n_locations       = is_pdp ? 3 : 2;
  const int n_orders          = is_pdp ? 2 : 1;
  const int expected_n_nodes  = is_pdp ? 3 : 2;
  const double forward_dirty  = 123.;
  const double backward_dirty = 456.;

  std::vector<float> cost_matrix(n_locations * n_locations, 0.f);
  for (int from = 0; from < n_locations; ++from) {
    for (int to = 0; to < n_locations; ++to) {
      if (from != to) {
        cost_matrix[from * n_locations + to] = from * n_locations + to + 1.f;
      }
    }
  }

  std::vector<int> order_locations = is_pdp ? std::vector<int>{1, 2} : std::vector<int>{1};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto d_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto d_order_locations = cuopt::device_copy(order_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, n_locations, 1, n_orders);
  data_model.add_cost_matrix(d_cost_matrix.data());
  data_model.set_order_locations(d_order_locations.data());

  rmm::device_uvector<int> d_pickups(is_pdp ? 1 : 0, stream);
  rmm::device_uvector<int> d_deliveries(is_pdp ? 1 : 0, stream);
  if constexpr (is_pdp) {
    const int pickup   = 0;
    const int delivery = 1;
    raft::copy(d_pickups.data(), &pickup, 1, stream);
    raft::copy(d_deliveries.data(), &delivery, 1, stream);
    data_model.set_pickup_delivery_pairs(d_pickups.data(), d_deliveries.data());
  }

  cuopt::routing::solver_settings_t<int, float> settings;
  detail::problem_t<int, float> problem(data_model, settings);
  detail::solution_handle_t<int, float> sol_handle(problem.handle_ptr->get_stream());
  detail::solution_t<int, float, REQUEST> solution(
    problem, 0, &sol_handle, std::vector<int>{0}, expected_n_nodes + 1);

  auto& route = solution.get_route(0);
  ASSERT_EQ(route.n_nodes.value(stream), expected_n_nodes);

  auto& cost_route = route.template get_dim<detail::dim_t::COST>();
  raft::copy(cost_route.cost_forward.data(), &forward_dirty, 1, stream);
  raft::copy(cost_route.cost_backward.data() + expected_n_nodes, &backward_dirty, 1, stream);
  handle.sync_stream();

  {
    const auto cost_forward  = cuopt::host_copy(cost_route.cost_forward, stream);
    const auto cost_backward = cuopt::host_copy(cost_route.cost_backward, stream);
    ASSERT_DOUBLE_EQ(cost_forward[0], forward_dirty);
    ASSERT_DOUBLE_EQ(cost_backward[expected_n_nodes], backward_dirty);
  }

  set_route_data_kernel<REQUEST><<<1, 32, 0, stream>>>(solution.view(), problem.view());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  handle.sync_stream();

  const auto cost_forward  = cuopt::host_copy(cost_route.cost_forward, stream);
  const auto cost_backward = cuopt::host_copy(cost_route.cost_backward, stream);

  EXPECT_DOUBLE_EQ(cost_forward[0], 0.);
  EXPECT_DOUBLE_EQ(cost_backward[expected_n_nodes], 0.);
}

TEST(cost_boundary_initialization, vrp_set_route_data_resets_cost_boundaries)
{
  test_cost_boundaries_reset<request_t::VRP>();
}

TEST(cost_boundary_initialization, pdp_set_route_data_resets_cost_boundaries)
{
  test_cost_boundaries_reset<request_t::PDP>();
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
