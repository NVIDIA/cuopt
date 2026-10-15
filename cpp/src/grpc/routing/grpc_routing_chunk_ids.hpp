/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuopt::routing {

// ArrayChunk addressing for the VRP matrices. container_field_num is the field
// number of RoutingProblem.cost_matrices / transit_time_matrices.
// container_index is the matrix index. field_id is CostMatrix.ArrayId.VALUES.
constexpr int32_t kCostMatricesContainer    = 10;
constexpr int32_t kTransitMatricesContainer = 11;
constexpr int32_t kMatrixValuesFieldId      = 0;  // CostMatrix::VALUES

}  // namespace cuopt::routing
