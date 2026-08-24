/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/cpu_routing_problem.hpp>

#include <cuopt_routing_solution.pb.h>  // RoutingSolution / RoutingSolutionStatus

namespace cuopt {
namespace CUOPT_EXPORT routing {

// Server direction: serialize a solved assignment into a RoutingSolution proto.
void map_routing_solution_to_proto(const cuopt::routing::assignment_t<int>& assignment,
                                   const cuopt::routing::host_assignment_t<int>& host,
                                   cuopt::remote::RoutingSolution* pb);

// Client direction: parse a RoutingSolution proto into a host solution struct.
void map_proto_to_routing_solution(const cuopt::remote::RoutingSolution& pb,
                                   cuopt::routing::cpu_routing_solution_t& sol);

}  // namespace CUOPT_EXPORT routing
}  // namespace cuopt
