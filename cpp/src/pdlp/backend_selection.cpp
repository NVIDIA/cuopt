/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/backend_selection.hpp>
#include <utilities/logger.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>

namespace cuopt::mathematical_optimization {

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

bool cpu_memory_backend_solve_allowed()
{
  return is_remote_execution_enabled() || use_cpu_memory_for_local();
}

memory_backend_t get_memory_backend_type()
{
  // Remote execution and the undocumented local test mode force CPU memory
  // regardless of hardware, so decide without probing CUDA.
  if (is_remote_execution_enabled() || use_cpu_memory_for_local()) { return memory_backend_t::CPU; }

  int cuda_count        = 0;
  const cudaError_t err = cudaGetDeviceCount(&cuda_count);
  if (err == cudaSuccess && cuda_count > 0) { return memory_backend_t::GPU; }

  // Local run with no usable device: fall back to CPU memory, which requires remote
  // execution at solve time. Log at INFO (the shipped level) so this surprising
  // fallback is visible without a recompile -- the CUDA error code distinguishes
  // "no device" from a driver/OS fault, and CUDA_VISIBLE_DEVICES reveals an
  // intentionally hidden GPU. This query is invoked at both problem creation and
  // solve time; the decision is fixed for the life of the process, so log only once
  // to avoid duplicate lines.
  static std::atomic<bool> already_logged{false};
  if (!already_logged.exchange(true)) {
    CUOPT_LOG_INFO(
      "cuOpt selected CPU memory backend: no usable CUDA device "
      "(cudaGetDeviceCount err=%d (%s) count=%d, CUDA_VISIBLE_DEVICES=%s)",
      static_cast<int>(err),
      cudaGetErrorString(err),
      cuda_count,
      std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "(unset)");
  }
  return memory_backend_t::CPU;
}

}  // namespace cuopt::mathematical_optimization
