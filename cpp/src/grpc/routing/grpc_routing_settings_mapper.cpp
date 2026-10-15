/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "grpc_routing_settings_mapper.hpp"

#include <limits>

namespace cuopt {
namespace routing {

void map_proto_to_routing_settings(const cuopt::remote::RoutingSolverSettings& pb,
                                   cuopt::routing::solver_settings_t<int, float>& settings)
{
  // Honor an explicit time_limit (including 0); absence means "use solver default".
  if (pb.has_time_limit()) { settings.set_time_limit(pb.time_limit()); }
  settings.set_verbose_mode(pb.verbose());
  settings.set_error_logging_mode(pb.error_logging());
  if (!pb.dump_best_results_path().empty()) {
    settings.dump_best_results(pb.dump_best_results_path(), pb.dump_best_results_interval());
  }
}

void map_routing_settings_to_proto(const cuopt::routing::solver_settings_t<int, float>& settings,
                                   cuopt::remote::RoutingSolverSettings* pb)
{
  pb->Clear();
  // Only emit time_limit when the caller set it; the unset sentinel (f_t max)
  // stays absent so the server derives its default instead of receiving a huge
  // value. An explicit 0 is a real value and is forwarded.
  if (settings.get_time_limit() != std::numeric_limits<float>::max()) {
    pb->set_time_limit(settings.get_time_limit());
  }
  pb->set_verbose(settings.get_verbose_mode());
  pb->set_error_logging(settings.get_error_logging_mode());
  auto [interval, dump, path] = settings.get_dump_best_results();
  if (dump) {
    pb->set_dump_best_results_path(path);
    pb->set_dump_best_results_interval(interval);
  }
}

}  // namespace routing
}  // namespace cuopt
