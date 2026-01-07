/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "cuda.h"

namespace cuopt {

namespace detail {

inline auto get_driver_entry_point(const char* name)
{
  void* func = nullptr;
  cudaDriverEntryPointQueryResult driver_status;
  
  // Use runtime version instead of compile-time CUDART_VERSION to ensure
  // compatibility when code built with newer CUDA runs on older CUDA runtime
  int runtime_version = 0;
  cudaError_t version_result = cudaRuntimeGetVersion(&runtime_version);
  
  // Fall back to compile-time version if runtime query fails
  int version_to_use = (version_result == cudaSuccess) ? runtime_version : CUDART_VERSION;
  
  cudaGetDriverEntryPointByVersion(name, &func, version_to_use, cudaEnableDefault, &driver_status);
  if (driver_status != cudaDriverEntryPointSuccess) {
    fprintf(stderr, "Failed to fetch symbol for %s (version %d)\n", name, version_to_use);
    return static_cast<void*>(nullptr);
  }
  return func;
}

}  // namespace detail
}  // namespace cuopt
