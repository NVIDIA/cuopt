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
  void* func;
  cudaDriverEntryPointQueryResult driver_status;
  cudaGetDriverEntryPointByVersion(name, &func, CUDART_VERSION, cudaEnableDefault, &driver_status);
  if (driver_status != cudaDriverEntryPointSuccess) {
    fprintf(stderr, "Failed to fetch symbol for %s\n", name);
  }
  return func;
}

}  // namespace detail
}  // namespace cuopt
