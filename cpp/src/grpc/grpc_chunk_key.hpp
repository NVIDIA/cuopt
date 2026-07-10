/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuopt::mathematical_optimization {

// Composite key for chunks targeting arrays inside repeated nested messages
// (e.g., a single QuadraticConstraint's linear_values).  Top-level arrays
// continue to use std::map<int32_t, ...> on the wire — this key is only used
// for container-aware paths.
//
// Lives in its own header so both the server-side pipe serialization layer
// and the higher-level problem-mapper public API can reference it without
// pulling in each other's transitive dependencies.
struct container_array_key_t {
  int32_t container_field_num;  // parent message's field_num for the repeated_messages entry
  int32_t container_index;      // 0-based index within the repeated field
  int32_t field_id;             // container-relative id

  bool operator<(const container_array_key_t& other) const
  {
    if (container_field_num != other.container_field_num)
      return container_field_num < other.container_field_num;
    if (container_index != other.container_index) return container_index < other.container_index;
    return field_id < other.field_id;
  }

  bool operator==(const container_array_key_t& other) const
  {
    return container_field_num == other.container_field_num &&
           container_index == other.container_index && field_id == other.field_id;
  }
};

}  // namespace cuopt::mathematical_optimization
