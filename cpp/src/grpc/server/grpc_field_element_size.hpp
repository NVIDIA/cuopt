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
// field_id (an ArrayFieldId value).  For chunks targeting an array inside
// a repeated nested message (e.g. a QuadraticConstraint), the caller
// passes the container's parent field_num and the container-relative
// field_id (small dense int starting at 0; see field_registry.yaml).
//
// Returns the element width in bytes on success, or -1 if the (container,
// field_id) pair is not registered.  Callers MUST treat -1 as an invalid
// chunk and reject it; otherwise downstream arithmetic on elem_size will
// produce garbage.
inline int64_t array_field_element_size(int32_t container_field_num, int32_t field_id)
{
#include "generated_array_field_element_size.inc"
}

#endif  // CUOPT_ENABLE_GRPC
