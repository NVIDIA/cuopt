// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// libcuopt-routing (+ client) only: solve a tiny CVRP.
#include <cuopt/routing/solve.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <vector>

template <typename T>
static rmm::device_uvector<T> to_device(std::vector<T> const& h, rmm::cuda_stream_view stream)
{
  rmm::device_uvector<T> d(h.size(), stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d.data(), h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();
  return d;
}

TEST(IsolatedRouting, SolvesSmallCvrp)
{
  // depot (0) + 4 delivery points, 3 vehicles
  std::vector<float> cost = {0,  10, 15, 20, 25, 10, 0,  35, 25, 30, 15, 35, 0,
                             30, 20, 20, 25, 30, 0,  15, 25, 30, 20, 15, 0};
  std::vector<int> orders = {1, 2, 3, 4};
  std::vector<int> starts = {0, 0, 0};

  raft::handle_t handle;
  auto stream = handle.get_stream();
  auto d_cost = to_device(cost, stream);
  auto d_ord  = to_device(orders, stream);
  auto d_st   = to_device(starts, stream);

  cuopt::routing::data_model_view_t<int, float> model(&handle, 5, 3, 4);
  model.add_cost_matrix(d_cost.data());
  model.set_order_locations(d_ord.data());
  model.set_vehicle_locations(d_st.data(), d_st.data());

  auto solution = cuopt::routing::solve(model);
  handle.sync_stream();
  EXPECT_EQ(solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  EXPECT_GT(solution.get_vehicle_count(), 0);
  EXPECT_GT(solution.get_total_objective(), 0.0);
}
