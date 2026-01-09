/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  // Request CUDA 13.0 (13000) version of symbols for Green Context API
  // Green contexts are guarded by CUDART_VERSION >= 13000, so we know they're only
  // used when compiled with CUDA 13.0+. Requesting v13000 ensures compatibility
  // across CUDA 13.x versions (e.g., built with 13.1, run on 13.0).
  cudaGetDriverEntryPointByVersion(name, &func, 13000, cudaEnableDefault, &driver_status);
  if (driver_status != cudaDriverEntryPointSuccess) {
    fprintf(stderr, "Failed to fetch symbol for %s\n", name);
    return static_cast<void*>(nullptr);
  }
  return func;
}

}  // namespace detail
}  // namespace cuopt
