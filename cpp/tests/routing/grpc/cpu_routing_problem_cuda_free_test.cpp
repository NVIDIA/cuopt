/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Links cuopt_client alone with no rmm/raft: a regression here must fail to *link*,
// not be papered over by adding rmm/raft back to this target.

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
