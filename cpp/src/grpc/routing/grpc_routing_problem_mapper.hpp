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

#include "grpc_chunk_key.hpp"

#include <cstdint>
#include <map>
#include <vector>

namespace cuopt {
namespace CUOPT_EXPORT routing {

void map_proto_to_routing_problem(
  const cuopt::remote::RoutingProblem& pb,
  cuopt::routing::cpu_routing_problem_t& problem,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<cuopt::mathematical_optimization::container_array_key_t, std::vector<uint8_t>>&
    container_arrays);

inline void map_proto_to_routing_problem(const cuopt::remote::RoutingProblem& pb,
                                         cuopt::routing::cpu_routing_problem_t& problem)
{
  map_proto_to_routing_problem(pb, problem, {}, {});
}

// include_matrix_values=false omits matrix payloads. include_other_arrays=false omits every
// other array payload. Record keys and scalars stay so chunk indexes still line up.
void map_routing_problem_to_proto(const cuopt::routing::cpu_routing_problem_t& problem,
                                  cuopt::remote::RoutingProblem* pb,
                                  bool include_matrix_values = true,
                                  bool include_other_arrays  = true);

}  // namespace CUOPT_EXPORT routing
}  // namespace cuopt
