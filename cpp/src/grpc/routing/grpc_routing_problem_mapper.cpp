/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "grpc_routing_problem_mapper.hpp"

#include "grpc_routing_mapper_utils.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuopt {
namespace routing {

using grpc_map_detail::copy_bool_to_u8;
using grpc_map_detail::copy_repeated_to_vector;
using grpc_map_detail::copy_u32_to_u8;
using grpc_map_detail::copy_vector_to_repeated;

void map_proto_to_routing_problem(
  const cuopt::remote::RoutingProblem& pb,
  cuopt::routing::cpu_routing_problem_t& p,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<cuopt::mathematical_optimization::container_array_key_t, std::vector<uint8_t>>&
    container_arrays)
{
#include "generated_routing_problem_from_proto.inc"
}

void map_routing_problem_to_proto(const cuopt::routing::cpu_routing_problem_t& p,
                                  cuopt::remote::RoutingProblem* pb,
                                  bool include_matrix_values,
                                  bool include_other_arrays)
{
#include "generated_routing_problem_to_proto.inc"
}

}  // namespace routing
}  // namespace cuopt
