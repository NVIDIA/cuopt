/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "routing/grpc_routing_problem_mapper.hpp"

#include <cuopt_routing.pb.h>
#include <cuopt/routing/cpu_routing_problem.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
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

TEST(RoutingProblemMapper, VehicleDistanceTiersRoundTrip)
{
  auto p                     = make_base_problem();
  p.distance_matrices        = {{1, {0.f, 2.f, 2.f, 0.f}}};
  p.vehicle_max_distances    = {10.f, 20.f};
  p.distance_tier_thresholds = {
    5.f, std::numeric_limits<float>::max(), 7.f, std::numeric_limits<float>::max()};
  p.distance_tier_fixed_costs    = {3.f, 0.f, 4.f, 0.f};
  p.distance_tier_costs_per_unit = {0.f, 2.f, 0.f, 3.f};
  p.distance_tier_offsets        = {0, 2, 4};

  cuopt::remote::RoutingProblem pb;
  cuopt::routing::map_routing_problem_to_proto(p, &pb);

  ASSERT_EQ(pb.distance_matrices_size(), 1);
  ASSERT_TRUE(pb.has_vehicle_distance_tiers());
  ASSERT_EQ(pb.vehicle_distance_tiers().thresholds_size(), 4);

  cuopt::routing::cpu_routing_problem_t back;
  cuopt::routing::map_proto_to_routing_problem(pb, back);
  ASSERT_EQ(back.distance_matrices.size(), 1u);
  EXPECT_EQ(back.distance_matrices[0].vehicle_type, 1);
  EXPECT_EQ(back.distance_matrices[0].matrix, (std::vector<float>{0.f, 2.f, 2.f, 0.f}));
  EXPECT_EQ(back.vehicle_max_distances, (std::vector<float>{10.f, 20.f}));
  EXPECT_EQ(back.distance_tier_thresholds,
            (std::vector<float>{
              5.f, std::numeric_limits<float>::max(), 7.f, std::numeric_limits<float>::max()}));
  EXPECT_EQ(back.distance_tier_fixed_costs, (std::vector<float>{3.f, 0.f, 4.f, 0.f}));
  EXPECT_EQ(back.distance_tier_costs_per_unit, (std::vector<float>{0.f, 2.f, 0.f, 3.f}));
  EXPECT_EQ(back.distance_tier_offsets, (std::vector<int32_t>{0, 2, 4}));
}

TEST(RoutingProblemMapper, PreservesPartialDistanceTiersForValidation)
{
  auto p                         = make_base_problem();
  p.distance_tier_fixed_costs    = {3.f};
  p.distance_tier_costs_per_unit = {2.f};
  p.distance_tier_offsets        = {0, 1, 1};

  cuopt::remote::RoutingProblem pb;
  cuopt::routing::map_routing_problem_to_proto(p, &pb);

  ASSERT_TRUE(pb.has_vehicle_distance_tiers());
  EXPECT_EQ(pb.vehicle_distance_tiers().thresholds_size(), 0);
  EXPECT_EQ(pb.vehicle_distance_tiers().fixed_costs_size(), 1);

  cuopt::routing::cpu_routing_problem_t back;
  cuopt::routing::map_proto_to_routing_problem(pb, back);
  EXPECT_TRUE(back.distance_tier_thresholds.empty());
  EXPECT_EQ(back.distance_tier_fixed_costs, (std::vector<float>{3.f}));
  EXPECT_EQ(back.distance_tier_costs_per_unit, (std::vector<float>{2.f}));
  EXPECT_EQ(back.distance_tier_offsets, (std::vector<int32_t>{0, 1, 1}));
}

TEST(RoutingProblemMapper, RejectsOutOfRangeVehicleType)
{
  cuopt::remote::RoutingProblem pb;
  pb.add_vehicle_types(256);

  cuopt::routing::cpu_routing_problem_t problem;
  EXPECT_THROW(cuopt::routing::map_proto_to_routing_problem(pb, problem), std::invalid_argument);
}
