/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
#include <cuopt/routing/cpu_routing_problem.hpp>

#include <cuopt_routing.pb.h>

namespace cuopt {
namespace CUOPT_EXPORT routing {

void map_proto_to_routing_problem(const cuopt::remote::RoutingProblem& pb,
                                  cuopt::routing::cpu_routing_problem_t& problem);

void map_routing_problem_to_proto(const cuopt::routing::cpu_routing_problem_t& problem,
                                  cuopt::remote::RoutingProblem* pb);

}  // namespace CUOPT_EXPORT routing
}  // namespace cuopt
