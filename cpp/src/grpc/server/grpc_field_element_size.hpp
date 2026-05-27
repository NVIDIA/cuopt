/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include <cstdint>
#include "cuopt_remote.pb.h"

// Element byte size for a chunk's payload, dispatched on the chunk's
// (container_field_num, field_id) pair.  For top-level chunks the caller
// passes -1 for container_field_num (no container) and the chunk's
// field_id (an ArrayFieldId value).  The int32 typing on the front-end
// lets callers feed proto-decoded field_id values directly — proto3
// silently converts enums to int32 on the wire, so a chunk's field_id
// may carry either an ArrayFieldId value (top-level) or a
// container-relative index.
inline int64_t array_field_element_size(int32_t field_id)
{
#include "generated_array_field_element_size.inc"
}

#endif  // CUOPT_ENABLE_GRPC
