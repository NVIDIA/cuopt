/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Round-trip coverage for the routing settings mapper. This is the mapper that reads
// routing::solver_settings_t, so it exercises the host-only accessors that build into
// cuopt_client rather than the routing engine.

#include "routing/grpc_routing_settings_mapper.hpp"

#include <cuopt_routing.pb.h>
#include <cuopt/routing/solver_settings.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace {

using settings_t = cuopt::routing::solver_settings_t<int, float>;

}  // namespace

TEST(RoutingSettingsMapper, RoundTripPreservesEveryField)
{
  settings_t original;
  original.set_time_limit(12.5f);
  original.set_verbose_mode(true);
  original.set_error_logging_mode(true);

  cuopt::remote::RoutingSolverSettings pb;
  cuopt::routing::map_routing_settings_to_proto(original, &pb);

  settings_t restored;
  cuopt::routing::map_proto_to_routing_settings(pb, restored);

  EXPECT_FLOAT_EQ(restored.get_time_limit(), 12.5f);
  EXPECT_TRUE(restored.get_verbose_mode());
  EXPECT_TRUE(restored.get_error_logging_mode());
}

// The client may still serialize dump_best_results_path. The inbound mapper
// leaves dump disabled so the routing solver does not open that path.
TEST(RoutingSettingsMapper, DumpBestResultsPathIsNotApplied)
{
  settings_t original;
  original.dump_best_results("/tmp/cuopt-client-path", 7);

  cuopt::remote::RoutingSolverSettings pb;
  cuopt::routing::map_routing_settings_to_proto(original, &pb);
  EXPECT_EQ(pb.dump_best_results_path(), "/tmp/cuopt-client-path");
  EXPECT_EQ(pb.dump_best_results_interval(), 7);

  settings_t restored;
  cuopt::routing::map_proto_to_routing_settings(pb, restored);
  auto [interval, dump, path] = restored.get_dump_best_results();
  EXPECT_FALSE(dump);
  EXPECT_TRUE(path.empty());
  EXPECT_EQ(interval, std::numeric_limits<int>::max());
}

// An unset time_limit must not be serialized: the solver derives its own default
// (num_orders / 5) from absence, so emitting the sentinel would silently override it.
TEST(RoutingSettingsMapper, UnsetTimeLimitIsNotSerialized)
{
  settings_t defaults;
  ASSERT_EQ(defaults.get_time_limit(), std::numeric_limits<float>::max());

  cuopt::remote::RoutingSolverSettings pb;
  cuopt::routing::map_routing_settings_to_proto(defaults, &pb);

  EXPECT_FALSE(pb.has_time_limit());

  settings_t restored;
  cuopt::routing::map_proto_to_routing_settings(pb, restored);
  EXPECT_EQ(restored.get_time_limit(), std::numeric_limits<float>::max());
}

// Zero is a legitimate time limit and is distinct from "unset", so it has to survive
// the round trip rather than being folded into absence.
TEST(RoutingSettingsMapper, ZeroTimeLimitSurvivesRoundTrip)
{
  settings_t original;
  original.set_time_limit(0.0f);

  cuopt::remote::RoutingSolverSettings pb;
  cuopt::routing::map_routing_settings_to_proto(original, &pb);
  EXPECT_TRUE(pb.has_time_limit());

  settings_t restored;
  cuopt::routing::map_proto_to_routing_settings(pb, restored);
  EXPECT_FLOAT_EQ(restored.get_time_limit(), 0.0f);
}

// dump_best_results is only emitted when enabled, so a default-constructed settings
// object must not carry a path across the wire.
TEST(RoutingSettingsMapper, DumpBestResultsOmittedWhenDisabled)
{
  settings_t defaults;
  cuopt::remote::RoutingSolverSettings pb;
  cuopt::routing::map_routing_settings_to_proto(defaults, &pb);

  EXPECT_TRUE(pb.dump_best_results_path().empty());

  settings_t restored;
  cuopt::routing::map_proto_to_routing_settings(pb, restored);
  auto [interval, dump, path] = restored.get_dump_best_results();
  EXPECT_FALSE(dump);
  EXPECT_TRUE(path.empty());
}

TEST(RoutingSettingsMapper, BooleanFlagsRoundTripIndependently)
{
  for (bool verbose : {false, true}) {
    for (bool error_logging : {false, true}) {
      settings_t original;
      original.set_verbose_mode(verbose);
      original.set_error_logging_mode(error_logging);

      cuopt::remote::RoutingSolverSettings pb;
      cuopt::routing::map_routing_settings_to_proto(original, &pb);

      settings_t restored;
      cuopt::routing::map_proto_to_routing_settings(pb, restored);

      EXPECT_EQ(restored.get_verbose_mode(), verbose);
      EXPECT_EQ(restored.get_error_logging_mode(), error_logging);
    }
  }
}
