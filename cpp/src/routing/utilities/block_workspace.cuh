/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>

#include <raft/util/cuda_dev_essentials.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

/**
 * @brief Per-block workspace with shmem-first, global-memory fallback.
 *
 * Policy
 * ------
 *   - Global memory is ALWAYS pre-allocated (n_blocks × workspace_size bytes) so that
 *     a global fallback is available regardless of the shmem outcome.
 *   - shmem_size is capped at: default_limit + opt_in_fraction*(optin_limit - default_limit)
 *       • opt_in_fraction = 0.0 → default ~48 KB, no attribute setting, L1 preserved.
 *       • opt_in_fraction = 1.0 → full opt-in max (e.g. ~228 KB on GH200); calls
 *         cudaFuncSetAttribute for the primary constructor.
 *       • Secondary (bool false) constructor: capped at the default limit regardless
 *         of opt_in_fraction (attribute cannot be set for these kernels).
 *   - Call block_workspace_t::set_opt_in_fraction(f) once at startup to tune.
 *   - The kernel receives shmem_size bytes of dynamic shared memory.
 *   - Per-allocation placement is decided by workspace_bump_t (see below).
 *
 * Single-allocation kernels (workspace == one route)
 * ---------------------------------------------------
 *   Use view_t::get_workspace(shmem_buf): returns shmem if all of workspace fits,
 *   otherwise returns the per-block global slice.
 *
 * Multi-allocation kernels
 * ------------------------
 *   Construct a workspace_bump_t from the view and the shmem pointer.  Call
 *   bump.alloc<T>(count) for each piece; each piece is placed entirely in shmem
 *   when it fits, otherwise entirely in global.  No piece is ever split.
 *
 * Usage at kernel launch site:
 *
 *   block_workspace_t ws(my_kernel<i_t, f_t, REQUEST>, sh_size, n_blocks, stream);
 *   my_kernel<<<n_blocks, n_threads, ws.shmem_size(), stream>>>(
 *       ..., ws.view());
 *
 * Inside the kernel (single-allocation):
 *
 *   extern __shared__ char shmem[];
 *   char* mem = block_workspace.get_workspace(shmem);
 *
 * Inside the kernel (multi-allocation):
 *
 *   extern __shared__ char shmem[];
 *   workspace_bump_t bump(block_workspace, shmem);
 *   auto* arr1 = bump.alloc<T1>(n1);   // shmem if fits, else global
 *   auto* arr2 = bump.alloc<T2>(n2);   // shmem if remaining fits, else global
 */
struct block_workspace_t {
  // -------------------------------------------------------------------------
  // Device-side view — passed to kernels as a plain struct.
  // -------------------------------------------------------------------------
  struct view_t {
    // Per-cluster global backup; always non-null.
    char* global_ptr{nullptr};
    // Aligned bytes reserved per block in global memory.
    size_t workspace_size{0};
    // Bytes of dynamic shared memory in the kernel launch (≤ workspace_size).
    size_t shmem_size{0};

    /**
     * @brief Per-block slice of global memory.
     */
    DI char* global_workspace() const
    {
      return global_ptr + static_cast<size_t>(blockIdx.x) * workspace_size;
    }

    /**
     * @brief Single-allocation helper: returns shmem if the entire workspace
     * fits there (shmem_size >= workspace_size), otherwise returns the
     * per-block global slice.
     *
     * @param shmem  Pointer from `extern __shared__ char shmem[]` in the caller.
     */
    DI char* get_workspace(void* shmem) const
    {
      return (shmem_size >= workspace_size) ? static_cast<char*>(shmem) : global_workspace();
    }
  };

  // -------------------------------------------------------------------------
  // Tunable: fraction of the opt-in range to use above the default limit.
  //
  //   0.0 → cap at the hardware default (~48 KB); no L1 cache traded away.
  //   1.0 → cap at the full opt-in maximum (e.g. ~228 KB on GH200);
  //          cudaFuncSetAttribute is called to unlock extra shmem.
  //   Values in [0, 1] interpolate linearly between the two limits.
  //
  // Formula:  shmem_cap = default_limit + fraction * (optin_limit - default_limit)
  //
  // Set this once before launching kernels (not thread-safe for concurrent sets).
  // -------------------------------------------------------------------------
  static void set_opt_in_fraction(double fraction) { s_opt_in_fraction_ = fraction; }
  static double get_opt_in_fraction() { return s_opt_in_fraction_; }

  // -------------------------------------------------------------------------
  // Host-side constructors — decide shmem vs global at construction time.
  // -------------------------------------------------------------------------

  // Primary overload: deduces kernel type and sets the shmem attribute.
  // Works for kernels with only type template parameters and no __launch_bounds__.
  template <typename Function>
  block_workspace_t(Function* kernel,
                    size_t workspace_size,
                    int n_blocks,
                    rmm::cuda_stream_view stream)
    : workspace_size_(raft::alignTo(workspace_size, kAlignment)),
      shmem_size_(std::min(raft::alignTo(workspace_size, kAlignment), shmem_cap())),
      global_buffer_(static_cast<size_t>(n_blocks) * raft::alignTo(workspace_size, kAlignment),
                     stream)
  {
    // Only need cudaFuncSetAttribute when exceeding the no-opt-in default.
    if (shmem_size_ > device_shmem_default_limit()) {
      if (!set_shmem_of_kernel(kernel, shmem_size_)) {
        // Attribute failed; fall back to the default limit.
        shmem_size_ = std::min(workspace_size_, device_shmem_default_limit());
      }
    }
  }

  // Secondary overload: for kernels where NVCC cannot deduce Function* in the primary overload.
  // This happens for __global__ kernels annotated with __launch_bounds__ (with or without NTTPs).
  // NVCC cannot resolve the kernel to a unique function pointer in a dependent template context.
  //
  //   shmem_fits = true  → caller guarantees the workspace fits; use shmem for all of it.
  //   shmem_fits = false → cannot set attribute; use shmem up to shmem_cap(),
  //                        global memory for any overflow.
  //
  // Note: when opt_in_fraction > 0 and shmem_fits = false, shmem is still capped
  // at the default limit because the attribute cannot be set for these kernels.
  block_workspace_t(bool shmem_fits,
                    size_t workspace_size,
                    int n_blocks,
                    rmm::cuda_stream_view stream)
    : workspace_size_(raft::alignTo(workspace_size, kAlignment)),
      shmem_size_(shmem_fits
                    ? raft::alignTo(workspace_size, kAlignment)
                    : std::min(raft::alignTo(workspace_size, kAlignment),
                               device_shmem_default_limit())),
      global_buffer_(static_cast<size_t>(n_blocks) * raft::alignTo(workspace_size, kAlignment),
                     stream)
  {
  }

  /** @brief Dynamic shared memory size to pass to the kernel launch chevrons. */
  size_t shmem_size() const noexcept { return shmem_size_; }

  /** @brief True when global memory is needed for some or all of the workspace. */
  bool uses_global_memory() const noexcept { return shmem_size_ < workspace_size_; }

  /** @brief Device-side view to pass to the kernel as a parameter. */
  view_t view() const noexcept
  {
    return view_t{const_cast<char*>(global_buffer_.data()), workspace_size_, shmem_size_};
  }

 private:
  // Align each per-block slice to 256 bytes so the first field of any type
  // placed at the start of the slice satisfies device memory alignment.
  static constexpr size_t kAlignment = 256;

  // Fraction in [0, 1] controlling how far into the opt-in range to go.
  // 0 = default limit only, 1 = full opt-in max.
  static inline double s_opt_in_fraction_{1.0};

  // Effective shmem cap for this process, given the current opt_in_fraction.
  static size_t shmem_cap()
  {
    size_t def = device_shmem_default_limit();
    size_t opt = device_shmem_optin_limit();
    return def + static_cast<size_t>(s_opt_in_fraction_ * static_cast<double>(opt - def));
  }

  // Maximum dynamic shmem per block WITH cudaFuncSetAttribute (opt-in).
  // On GH200 (Hopper) this is ~228 KB; on A100 ~164 KB; on V100 ~96 KB.
  static size_t device_shmem_optin_limit()
  {
    static size_t limit = [] {
      int device = 0, val = 0;
      cudaGetDevice(&device);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      return static_cast<size_t>(val);
    }();
    return limit;
  }

  // Maximum dynamic shmem per block without opting in (hardware default).
  // On Volta/Ampere/Hopper this is typically 49152 bytes (48 KB).
  static size_t device_shmem_default_limit()
  {
    static size_t limit = [] {
      int device = 0, val = 0;
      cudaGetDevice(&device);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlock, device);
      return static_cast<size_t>(val);
    }();
    return limit;
  }

  size_t workspace_size_;
  size_t shmem_size_;
  rmm::device_uvector<char> global_buffer_;
};

// ---------------------------------------------------------------------------
// Device-side mixed allocator for multi-allocation kernels.
//
// Allocates each piece entirely in shmem if it fits there, otherwise entirely
// in the per-block global slice.  No single piece is ever split between the two
// memory spaces.
//
// Typical use (inside a __global__ kernel):
//
//   extern __shared__ char shmem_buf[];
//   workspace_bump_t bump(block_workspace, shmem_buf);
//   auto* nodes = bump.alloc<node_t>(n);        // shmem if fits, else global
//   auto* route = bump.alloc_for<i_t>(bytes);   // shmem if fits, else global
//   auto s_route = route_t::view_t::create_shared_route(route, ...);
// ---------------------------------------------------------------------------
struct workspace_bump_t {
  char*  sh_;        // shmem base pointer
  char*  gl_;        // per-block global base pointer
  size_t sh_avail_;  // bytes of shmem available for this block
  size_t sh_used_{0};
  size_t gl_used_{0};

  DI workspace_bump_t(const block_workspace_t::view_t& v, void* shmem)
    : sh_(static_cast<char*>(shmem)), gl_(v.global_workspace()), sh_avail_(v.shmem_size)
  {
  }

  /**
   * @brief Allocate `count` contiguous elements of type T (8-byte aligned).
   * Each allocation is placed entirely in shmem or entirely in global.
   */
  template <typename T>
  DI T* alloc(size_t count)
  {
    return reinterpret_cast<T*>(bump_(count * sizeof(T)));
  }

  /**
   * @brief Allocate `bytes` raw bytes and return as T* — for use with create_shared_route
   * and other APIs that consume a typed pointer and internally bump it.
   * The byte count is already known by the caller (e.g. workspace_size - node_bytes).
   */
  template <typename T>
  DI T* alloc_for(size_t bytes)
  {
    return reinterpret_cast<T*>(bump_(bytes));
  }

 private:
  // Core allocator: 8-byte-align `bytes`, place in shmem if it fits, else global.
  DI char* bump_(size_t bytes)
  {
    bytes = (bytes + 7u) & ~7u;
    if (sh_used_ + bytes <= sh_avail_) {
      char* ptr = sh_ + sh_used_;
      sh_used_ += bytes;
      return ptr;
    }
    char* ptr = gl_ + gl_used_;
    gl_used_ += bytes;
    return ptr;
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
