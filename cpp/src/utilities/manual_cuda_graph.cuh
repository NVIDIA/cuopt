/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <utility>

namespace cuopt {

// Wrapper around a CUDA graph captured from a callable. CUB / Thrust / RAFT /
// cuSPARSE calls inside the captured region are preserved.
//
// Invalidation recovery:
//   If cudaStreamEndCapture returns cudaErrorStreamCaptureInvalidated
//   (typically because another thread issued a synchronous CUDA call --
//   cudaDeviceSynchronize, cudaMalloc, cudaFree, or a library first-use that
//   internally syncs the device -- concurrently with this capture window),
//   the captured work has NOT been issued to the device. The wrapper drains
//   the sticky error, re-executes `work` eagerly so the current iteration
//   still produces correct results, and leaves itself uninitialized so the
//   next `run` call retries capture.
//   IMPORTANT: because `work` is invoked a second time on recovery, any
//   host-side mutations inside the callable will run twice -- keep `work`
//   host-idempotent or move host bookkeeping (counters, flags, hash updates,
//   etc.) outside the callable.
//
// Not thread-safe per instance: a single manual_cuda_graph_t must be driven
// from one thread at a time. Multiple instances on per-thread streams,
// captured concurrently across threads, is the supported multi-threaded
// pattern.
class manual_cuda_graph_t {
 public:
  manual_cuda_graph_t() = default;

  manual_cuda_graph_t(const manual_cuda_graph_t&)            = delete;
  manual_cuda_graph_t& operator=(const manual_cuda_graph_t&) = delete;

  manual_cuda_graph_t(manual_cuda_graph_t&& other) noexcept { swap(other); }
  manual_cuda_graph_t& operator=(manual_cuda_graph_t&& other) noexcept
  {
    if (this != &other) {
      destroy();
      swap(other);
    }
    return *this;
  }

  ~manual_cuda_graph_t() { destroy(); }

  template <typename F>
  void run(rmm::cuda_stream_view stream, F&& work)
  {
    if (instance_ != nullptr) {
      RAFT_CUDA_TRY(cudaGraphLaunch(instance_, stream.value()));
      return;
    }

    // RAII: if user code throws mid-capture, end capture so the stream isn't
    // left in capture state. Errors are swallowed -- we're already unwinding.
    capture_guard_t guard{stream.value()};

    RAFT_CUDA_TRY(cudaStreamBeginCapture(stream.value(), cudaStreamCaptureModeThreadLocal));
    guard.capture_active = true;

    work();

    cudaGraph_t captured = nullptr;
    cudaError_t end_err  = cudaStreamEndCapture(stream.value(), &captured);
    guard.capture_active = false;

    if (end_err == cudaErrorStreamCaptureInvalidated) {
      // The recorded work has NOT been issued; drain the sticky error and re-run
      // eagerly so this iteration still produces correct results. Stay uninitialized
      // so the next call retries capture. EndCapture sets `captured` to nullptr on
      // invalidation, so there is no graph to free here.
      cudaGetLastError();
      work();
      return;
    }
    RAFT_CUDA_TRY(end_err);

    // Destroy the source graph regardless of whether instantiation succeeded:
    // on failure cudaGraphInstantiate leaves instance_ at nullptr per the API
    // contract, and the source graph is unconditionally not needed any more.
    cudaError_t inst_err = cudaGraphInstantiate(&instance_, captured);
    RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(captured));
    RAFT_CUDA_TRY(inst_err);

    RAFT_CUDA_TRY(cudaGraphLaunch(instance_, stream.value()));
  }

  bool is_initialized() const noexcept { return instance_ != nullptr; }

  // Drop the instantiated graph so the next run() re-captures from scratch.
  void reset() { destroy(); }

 private:
  // RAII helper: cleans up a partial capture if the user-supplied callable
  // throws between start- and end-capture.
  struct capture_guard_t {
    cudaStream_t stream{};
    bool capture_active{false};

    ~capture_guard_t() noexcept
    {
      if (capture_active) {
        cudaGraph_t dummy = nullptr;
        // best-effort; we're already unwinding
        cudaStreamEndCapture(stream, &dummy);
        if (dummy != nullptr) { cudaGraphDestroy(dummy); }
      }
    }
  };

  void destroy() noexcept
  {
    if (instance_ != nullptr) {
      RAFT_CUDA_TRY_NO_THROW(cudaGraphExecDestroy(instance_));
      instance_ = nullptr;
    }
  }

  void swap(manual_cuda_graph_t& other) noexcept { std::swap(instance_, other.instance_); }

  cudaGraphExec_t instance_{nullptr};
};

}  // namespace cuopt
