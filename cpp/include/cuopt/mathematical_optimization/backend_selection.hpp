/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::mathematical_optimization {

/**
 * @brief Enum for execution mode (local vs remote solve)
 */
enum class execution_mode_t {
  LOCAL,  ///< Solve locally on this machine
  REMOTE  ///< Solve remotely via gRPC
};

/**
 * @brief Enum for memory backend type (GPU vs CPU memory)
 */
enum class memory_backend_t {
  GPU,  ///< Use GPU memory (device memory via RMM)
  CPU   ///< Use CPU memory (host memory)
};

/**
 * @brief Check if remote execution is enabled via environment variables
 * @return true if both CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT are set
 */
bool is_remote_execution_enabled();

/**
 * @brief Determine execution mode based on environment variables
 *
 * @return execution_mode_t::REMOTE if CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT are set,
 *         execution_mode_t::LOCAL otherwise
 */
execution_mode_t get_execution_mode();

/**
 * @brief Check if CPU memory should be used for local execution (test mode)
 *
 * Undocumented. Intended for testing CPU problem/solution structures without remote execution.
 * When enabled, local solve will convert CPU problems to GPU, solve, and convert back.
 *
 * @return true if CUOPT_USE_CPU_MEM_FOR_LOCAL is set to "true" or "1" (case-insensitive)
 */
bool use_cpu_memory_for_local();

/**
 * @brief Whether a CPU-memory-backed problem has a configured solve path without re-probing CUDA.
 *
 * True when remote execution is enabled or undocumented test mode CUOPT_USE_CPU_MEM_FOR_LOCAL is
 * set (local CPU-to-GPU bridge on a GPU host).
 */
bool cpu_memory_backend_solve_allowed();

/**
 * @brief Determine which memory backend to use based on execution mode and hardware
 *
 * Logic:
 *   - REMOTE execution -> always CPU memory
 *   - LOCAL + CUOPT_USE_CPU_MEM_FOR_LOCAL -> CPU memory (undocumented test mode)
 *   - LOCAL + no visible CUDA devices -> CPU memory (auto-detect CPU-only host)
 *   - LOCAL otherwise -> GPU memory
 *
 * @return memory_backend_t::GPU or memory_backend_t::CPU
 */
memory_backend_t get_memory_backend_type();

}  // namespace cuopt::mathematical_optimization
