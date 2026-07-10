/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/cpu_routing_problem.hpp>
#include <cuopt/routing/solver_settings.hpp>

#include <cuopt_routing.pb.h>

namespace cuopt {
namespace mathematical_optimization {

void map_proto_to_routing_problem(const cuopt::remote::RoutingProblem& pb,
                                  cuopt::routing::cpu_routing_problem_t& problem);

void map_routing_problem_to_proto(const cuopt::routing::cpu_routing_problem_t& problem,
                                  cuopt::remote::RoutingProblem* pb);

void map_proto_to_routing_settings(const cuopt::remote::RoutingSolverSettings& pb,
                                   cuopt::routing::solver_settings_t<int, float>& settings);

void map_routing_settings_to_proto(const cuopt::routing::solver_settings_t<int, float>& settings,
                                   cuopt::remote::RoutingSolverSettings* pb);

void map_routing_solution_to_proto(const cuopt::routing::assignment_t<int>& assignment,
                                   const cuopt::routing::host_assignment_t<int>& host,
                                   cuopt::remote::RoutingSolution* pb);

}  // namespace mathematical_optimization
}  // namespace cuopt
