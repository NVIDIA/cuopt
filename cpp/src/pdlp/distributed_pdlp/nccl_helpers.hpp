/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/error.hpp>

#include <nccl.h>

namespace cuopt::linear_programming::detail {

// Wraps a NCCL call and throws cuopt::logic_error (RuntimeError) on any non-
// ncclSuccess return code.
#define CUOPT_NCCL_TRY(call)                                                            \
  do {                                                                                  \
    ::ncclResult_t const _cuopt_nccl_status = (call);                                   \
    ::cuopt::cuopt_expects(_cuopt_nccl_status == ncclSuccess,                           \
                           ::cuopt::error_type_t::RuntimeError,                         \
                           "NCCL error at %s:%d: %s",                                   \
                           __FILE__,                                                    \
                           __LINE__,                                                    \
                           ::ncclGetErrorString(_cuopt_nccl_status));                   \
  } while (0)

}  // namespace cuopt::linear_programming::detail
