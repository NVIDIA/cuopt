/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuopt::routing {

// ArrayChunk addressing for the VRP matrices: the field numbers of RoutingProblem.cost_matrices /
// transit_time_matrices; container_index is the matrix index and field_id 0 its `values`.
constexpr int32_t kCostMatricesContainer    = 10;
constexpr int32_t kTransitMatricesContainer = 11;
constexpr int32_t kMatrixValuesFieldId      = 0;

}  // namespace cuopt::routing
