/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
#include <cuopt/routing/solver_settings.hpp>

#include <cuopt_routing.pb.h>

namespace cuopt {
namespace CUOPT_EXPORT routing {

void map_proto_to_routing_settings(const cuopt::remote::RoutingSolverSettings& pb,
                                   cuopt::routing::solver_settings_t<int, float>& settings);

void map_routing_settings_to_proto(const cuopt::routing::solver_settings_t<int, float>& settings,
                                   cuopt::remote::RoutingSolverSettings* pb);

}  // namespace CUOPT_EXPORT routing
}  // namespace cuopt
