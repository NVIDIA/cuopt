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
 * @brief Per-block workspace with shmem-first, mixed shmem+global fallback.
 *
 * Policy
 * ------
 *   - Global memory is ALWAYS pre-allocated (n_blocks × workspace_size bytes).
 *   - shmem_cap = default_limit + opt_in_fraction*(optin_limit - default_limit)
 *
 *   Case A — workspace ≤ shmem_cap:
 *     Entire workspace fits in shmem.  cudaFuncSetAttribute is called if needed
 *     (workspace > default_limit).  shmem_size() == workspace_size.
 *
 *   Case B — workspace > shmem_cap:
 *     No cudaFuncSetAttribute call.  shmem_size() == default_limit (~48 KB).
 *     The kernel is launched with default_limit bytes of dynamic shmem.
 *     - Single-allocation kernels (get_workspace): use global for the full workspace.
 *     - Multi-allocation kernels (workspace_bump_t): shmem receives allocations up
 *       to default_limit, then overflow goes to global automatically.
 *
 *   opt_in_fraction = 0.0 → shmem_cap == default_limit; Case B applies for
 *                            everything above ~48 KB.
 *   opt_in_fraction = 1.0 → shmem_cap == optin_limit (~228 KB on GH200); larger
 *                            workspaces mix shmem (48 KB) + global.
 *
 * Usage at kernel launch site:
 *
 *   block_workspace_t ws(my_kernel<i_t, f_t, REQUEST>, sh_size, n_blocks, stream);
 *   my_kernel<<<n_blocks, n_threads, ws.shmem_size(), stream>>>(
 *       ..., ws.view());
 *
 * Inside the kernel (single-allocation, Case A only):
 *
 *   extern __shared__ char shmem[];
 *   char* mem = block_workspace.get_workspace(shmem);  // shmem or global
 *
 * Inside the kernel (multi-allocation, Cases A and B):
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
  /** @brief Effective shmem cap (same value used internally); for call-site shmem_fits checks. */
  static size_t shmem_cap_static() { return shmem_cap(); }

  // -------------------------------------------------------------------------
  // Host-side constructors — decide shmem vs global at construction time.
  // -------------------------------------------------------------------------

  // Primary overload: deduces kernel type and sets the shmem attribute when needed.
  // Works for kernels with only type template parameters and no __launch_bounds__.
  //
  // cudaFuncSetAttribute(MaxDynamicSharedMemorySize) is called ONLY for Case A when
  // workspace > default_limit.  It is never called for Case B (workspace > shmem_cap)
  // so the kernel's occupancy footprint is never permanently inflated for large workspaces.
  template <typename Function>
  block_workspace_t(Function* kernel,
                    size_t workspace_size,
                    int n_blocks,
                    rmm::cuda_stream_view stream)
    : workspace_size_(raft::alignTo(workspace_size, kAlignment)),
      shmem_size_(device_shmem_default_limit()),
      global_buffer_(static_cast<size_t>(n_blocks) * raft::alignTo(workspace_size, kAlignment),
                     stream)
  {
    size_t cap = shmem_cap();
    size_t def = device_shmem_default_limit();
    if (workspace_size_ <= cap) {
      // Case A: workspace fits in (possibly opt-in) shmem.
      shmem_size_ = workspace_size_;
      if (shmem_size_ > def) {
        if (!set_shmem_of_kernel(kernel, shmem_size_)) {
          // Opt-in attribute failed: fall back to mixed mode (def shmem + global).
          shmem_size_ = (workspace_size_ <= def) ? workspace_size_ : def;
        }
      }
    }
    // Case B (workspace > cap): shmem_size_ stays at default_limit.
    // Kernel launched with default_limit shmem; overflow handled via global memory.
    // No cudaFuncSetAttribute call — occupancy footprint is not inflated.
  }

  // Secondary overload: for kernels where NVCC cannot deduce Function* in the primary overload.
  // This happens for __global__ kernels annotated with __launch_bounds__ (with or without NTTPs).
  //
  //   shmem_fits = true  → caller guarantees workspace ≤ shmem_cap and the attribute
  //                         has been set externally; all of workspace goes to shmem.
  //   shmem_fits = false → Case B: shmem = default_limit, overflow to global.
  //                         No cudaFuncSetAttribute is called.
  block_workspace_t(bool shmem_fits,
                    size_t workspace_size,
                    int n_blocks,
                    rmm::cuda_stream_view stream)
    : workspace_size_(raft::alignTo(workspace_size, kAlignment)),
      shmem_size_(shmem_fits ? raft::alignTo(workspace_size, kAlignment)
                             : std::min(raft::alignTo(workspace_size, kAlignment),
                                        device_shmem_default_limit())),
      global_buffer_(static_cast<size_t>(n_blocks) * raft::alignTo(workspace_size, kAlignment),
                     stream)
  {
  }

  /**
   * @brief Dynamic shared memory size to pass to the kernel launch chevrons.
   *
   * Case A (workspace ≤ shmem_cap): returns workspace_size — full shmem.
   * Case B (workspace > shmem_cap): returns default_limit (~48 KB) — partial shmem.
   *   The kernel's get_workspace() falls through to global (shmem < workspace),
   *   while workspace_bump_t uses the 48 KB for its early allocations.
   */
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
  char* sh_;         // shmem base pointer
  char* gl_;         // per-block global base pointer
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
