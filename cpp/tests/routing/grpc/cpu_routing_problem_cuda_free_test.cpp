/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Regression guard for the header leak fixed alongside this test: cpu_routing_problem.hpp
// is cuopt_client's host-only routing representation and must stay free of raft/rmm. This
// binary links cuopt_client alone (see CMakeLists.txt) -- if the header (or anything it
// pulls in) starts requiring a GPU-stack symbol again, this fails to *link*, not just run.
//
// If this test starts failing to link: do not "fix" it by adding rmm/raft to its
// target_link_libraries. That hides the regression instead of fixing it. Find what new
// include or usage reintroduced the GPU-stack dependency and remove it instead.

#include "routing/grpc_routing_problem_mapper.hpp"

#include <cuopt/routing/cpu_routing_problem.hpp>

#include <gtest/gtest.h>

TEST(CpuRoutingProblemCudaFree, ConstructAndMapWithoutGpuStack)
{
  cuopt::routing::cpu_routing_problem_t problem;
  problem.num_locations = 3;
  problem.fleet_size     = 1;
  problem.num_orders     = 3;

  cuopt::remote::RoutingProblem pb;
  cuopt::routing::map_routing_problem_to_proto(problem, &pb);

  cuopt::routing::cpu_routing_problem_t round_tripped;
  cuopt::routing::map_proto_to_routing_problem(pb, round_tripped);

  EXPECT_EQ(round_tripped.num_locations, problem.num_locations);
  EXPECT_EQ(round_tripped.fleet_size, problem.fleet_size);
}
