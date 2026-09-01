/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "routing/grpc_routing_problem_mapper.hpp"

#include <cuopt/routing/cpu_routing_problem.hpp>
#include <cuopt_routing.pb.h>

#include <gtest/gtest.h>

#include <vector>

namespace {

cuopt::routing::cpu_routing_problem_t make_base_problem()
{
  cuopt::routing::cpu_routing_problem_t p;
  p.num_locations = 5;
  p.fleet_size    = 2;
  p.num_orders    = 5;
  return p;
}

}  // namespace

TEST(RoutingProblemMapper, VehicleBreaksRoundTrip)
{
  auto p = make_base_problem();

  cuopt::routing::cpu_vehicle_break_t b0;
  b0.earliest  = 10;
  b0.latest    = 20;
  b0.duration  = 5;
  b0.locations = {3, 6};
  p.vehicle_breaks[0].push_back(b0);

  cuopt::routing::cpu_vehicle_break_t b1a;
  b1a.earliest = 30;
  b1a.latest   = 40;
  b1a.duration = 5;
  p.vehicle_breaks[1].push_back(b1a);

  cuopt::routing::cpu_vehicle_break_t b1b;
  b1b.earliest  = 60;
  b1b.latest    = 70;
  b1b.duration  = 5;
  b1b.locations = {1, 4};
  p.vehicle_breaks[1].push_back(b1b);

  cuopt::remote::RoutingProblem pb;
  cuopt::routing::map_routing_problem_to_proto(p, &pb);
  ASSERT_EQ(pb.vehicle_breaks_size(), 2);

  cuopt::routing::cpu_routing_problem_t back;
  cuopt::routing::map_proto_to_routing_problem(pb, back);
  ASSERT_EQ(back.vehicle_breaks.size(), 2u);
  ASSERT_EQ(back.vehicle_distance_breaks.size(), 0u);

  ASSERT_EQ(back.vehicle_breaks[0].size(), 1u);
  EXPECT_EQ(back.vehicle_breaks[0][0].earliest, 10);
  EXPECT_EQ(back.vehicle_breaks[0][0].latest, 20);
  EXPECT_EQ(back.vehicle_breaks[0][0].duration, 5);
  EXPECT_EQ(back.vehicle_breaks[0][0].locations, (std::vector<int32_t>{3, 6}));

  ASSERT_EQ(back.vehicle_breaks[1].size(), 2u);
  EXPECT_EQ(back.vehicle_breaks[1][0].earliest, 30);
  EXPECT_EQ(back.vehicle_breaks[1][0].latest, 40);
  EXPECT_TRUE(back.vehicle_breaks[1][0].locations.empty());
  EXPECT_EQ(back.vehicle_breaks[1][1].earliest, 60);
  EXPECT_EQ(back.vehicle_breaks[1][1].latest, 70);
  EXPECT_EQ(back.vehicle_breaks[1][1].locations, (std::vector<int32_t>{1, 4}));
}

TEST(RoutingProblemMapper, VehicleDistanceBreaksRoundTrip)
{
  auto p = make_base_problem();

  cuopt::routing::cpu_vehicle_distance_break_t b0;
  b0.distance_min = 120.f;
  b0.distance_max = 150.f;
  b0.duration     = 10;
  b0.locations    = {3, 6};
  p.vehicle_distance_breaks[0].push_back(b0);

  cuopt::routing::cpu_vehicle_distance_break_t b1a;
  b1a.distance_min = 0.f;
  b1a.distance_max = 200.f;
  b1a.duration     = 10;
  p.vehicle_distance_breaks[1].push_back(b1a);

  cuopt::routing::cpu_vehicle_distance_break_t b1b;
  b1b.distance_min = 270.f;
  b1b.distance_max = 300.f;
  b1b.duration     = 10;
  b1b.locations    = {1, 4};
  p.vehicle_distance_breaks[1].push_back(b1b);

  cuopt::remote::RoutingProblem pb;
  cuopt::routing::map_routing_problem_to_proto(p, &pb);
  ASSERT_EQ(pb.vehicle_distance_breaks_size(), 2);
  ASSERT_EQ(pb.vehicle_breaks_size(), 0);

  auto const& proto_v0 = pb.vehicle_distance_breaks(0).vehicle_id() == 0
                           ? pb.vehicle_distance_breaks(0)
                           : pb.vehicle_distance_breaks(1);
  ASSERT_EQ(proto_v0.breaks_size(), 1);
  EXPECT_FLOAT_EQ(proto_v0.breaks(0).distance_min(), 120.f);
  EXPECT_FLOAT_EQ(proto_v0.breaks(0).distance_max(), 150.f);
  EXPECT_EQ(proto_v0.breaks(0).duration(), 10);
  ASSERT_EQ(proto_v0.breaks(0).locations_size(), 2);

  cuopt::routing::cpu_routing_problem_t back;
  cuopt::routing::map_proto_to_routing_problem(pb, back);
  ASSERT_EQ(back.vehicle_distance_breaks.size(), 2u);
  ASSERT_EQ(back.vehicle_breaks.size(), 0u);

  ASSERT_EQ(back.vehicle_distance_breaks[0].size(), 1u);
  EXPECT_FLOAT_EQ(back.vehicle_distance_breaks[0][0].distance_min, 120.f);
  EXPECT_FLOAT_EQ(back.vehicle_distance_breaks[0][0].distance_max, 150.f);
  EXPECT_EQ(back.vehicle_distance_breaks[0][0].duration, 10);
  EXPECT_EQ(back.vehicle_distance_breaks[0][0].locations, (std::vector<int32_t>{3, 6}));

  ASSERT_EQ(back.vehicle_distance_breaks[1].size(), 2u);
  EXPECT_FLOAT_EQ(back.vehicle_distance_breaks[1][0].distance_min, 0.f);
  EXPECT_FLOAT_EQ(back.vehicle_distance_breaks[1][0].distance_max, 200.f);
  EXPECT_TRUE(back.vehicle_distance_breaks[1][0].locations.empty());
  EXPECT_FLOAT_EQ(back.vehicle_distance_breaks[1][1].distance_min, 270.f);
  EXPECT_FLOAT_EQ(back.vehicle_distance_breaks[1][1].distance_max, 300.f);
  EXPECT_EQ(back.vehicle_distance_breaks[1][1].locations, (std::vector<int32_t>{1, 4}));
}
