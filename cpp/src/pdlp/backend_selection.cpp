/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/backend_selection.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace cuopt::linear_programming {

bool is_remote_execution_enabled()
{
  const char* remote_host = std::getenv("CUOPT_REMOTE_HOST");
  const char* remote_port = std::getenv("CUOPT_REMOTE_PORT");
  return (remote_host != nullptr && remote_port != nullptr);
}

execution_mode_t get_execution_mode()
{
  return is_remote_execution_enabled() ? execution_mode_t::REMOTE : execution_mode_t::LOCAL;
}

bool use_gpu_memory_for_remote()
{
  const char* use_gpu_mem = std::getenv("CUOPT_USE_GPU_MEM_FOR_REMOTE");
  if (use_gpu_mem != nullptr) {
    std::string value(use_gpu_mem);
    // Convert to lowercase for case-insensitive comparison
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return (value == "true" || value == "1");
  }
  return false;
}

bool use_cpu_memory_for_local()
{
  const char* use_cpu_mem = std::getenv("CUOPT_USE_CPU_MEM_FOR_LOCAL");
  if (use_cpu_mem != nullptr) {
    std::string value(use_cpu_mem);
    // Convert to lowercase for case-insensitive comparison
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return (value == "true" || value == "1");
  }
  return false;
}

memory_backend_t get_memory_backend_type()
{
  if (get_execution_mode() == execution_mode_t::LOCAL) {
    // Local execution: GPU memory by default, CPU if test mode enabled
    return use_cpu_memory_for_local() ? memory_backend_t::CPU : memory_backend_t::GPU;
  }
  // Remote execution: CPU memory by default, GPU if explicitly requested
  return use_gpu_memory_for_remote() ? memory_backend_t::GPU : memory_backend_t::CPU;
}

}  // namespace cuopt::linear_programming
