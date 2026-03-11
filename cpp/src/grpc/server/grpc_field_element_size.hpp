/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include <cstdint>
#include "cuopt_remote.pb.h"

inline int64_t array_field_element_size(cuopt::remote::ArrayFieldId field_id)
{
#include "generated_array_field_element_size.inc"
}

#endif  // CUOPT_ENABLE_GRPC
