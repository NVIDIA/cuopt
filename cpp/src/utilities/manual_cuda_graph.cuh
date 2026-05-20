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

// Wrapper around a CUDA graph that is built via *manual* parent-graph
// construction. The cudaGraph_t object is explicitly created with
// cudaGraphCreate; user work is then captured directly into that
// manually-owned parent via cudaStreamBeginCaptureToGraph. CUB / Thrust /
// RAFT / cuSPARSE calls inside the captured region are preserved.
//
// The single public entry point is `run(stream, callable)`. It either
// launches the previously-instantiated graph or, on first call, captures the
// callable into a fresh graph, instantiates it, and launches.
//
// Invalidation recovery:
//   If cudaStreamEndCapture returns cudaErrorStreamCaptureInvalidated
//   (typically because another thread issued a synchronous CUDA call --
//   cudaDeviceSynchronize, cudaMalloc, cudaFree, or a library first-use that
//   internally syncs the device -- concurrently with this capture window),
//   the captured work has been recorded but NOT issued to the device. The
//   wrapper discards the partial graph, re-executes `work` eagerly so the
//   current iteration still produces correct results, and leaves itself
//   uninitialized so the next `run` call retries capture. The cost of an
//   invalidation is therefore one extra eager pass, not a crash.
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

    cudaGraph_t parent = nullptr;
    RAFT_CUDA_TRY(cudaGraphCreate(&parent, 0));

    // RAII: if user code throws mid-capture, end capture so the stream isn't
    // left in capture state. Errors here are intentionally swallowed -- we're
    // already unwinding for another reason and the parent graph is being
    // destroyed below.
    capture_guard_t guard{stream.value(), parent};

    RAFT_CUDA_TRY(cudaStreamBeginCaptureToGraph(
      stream.value(), parent, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    guard.capture_active = true;

    work();

    cudaGraph_t captured = nullptr;
    cudaError_t end_err  = cudaStreamEndCapture(stream.value(), &captured);
    guard.capture_active = false;

    if (end_err == cudaErrorStreamCaptureInvalidated) {
      CUOPT_LOG_INFO("Capture got invalidated by a concurrent synchronous CUDA call");
      // Capture got invalidated by a concurrent synchronous CUDA call on
      // another thread (cudaMalloc / cudaDeviceSynchronize / library
      // first-use). The recorded work has NOT been issued to the device.
      // Clear the error, drop the partial graph, and re-run eagerly so the
      // current iteration still produces correct results. The wrapper stays
      // uninitialized so the next call retries capture.
      guard.parent = nullptr;  // we're about to destroy it ourselves
      if (captured != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(captured)); }
      RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(parent));
      // Drain the sticky error so the next CUDA call doesn't see it.
      cudaGetLastError();
      work();
      return;
    }
    RAFT_CUDA_TRY(end_err);

    // cudaStreamBeginCaptureToGraph guarantees the returned graph IS the one
    // we passed in; the captured handle is just an alias.
    cuopt_assert(captured == parent, "cudaStreamEndCapture returned an unexpected graph handle");

    RAFT_CUDA_TRY(cudaGraphInstantiate(&instance_, parent));
    guard.parent = nullptr;  // ownership transferred; we destroy explicitly below
    RAFT_CUDA_TRY(cudaGraphDestroy(parent));

    RAFT_CUDA_TRY(cudaGraphLaunch(instance_, stream.value()));
  }

  bool is_initialized() const noexcept { return instance_ != nullptr; }

  // Drop the instantiated graph so the next run() re-captures from scratch.
  void reset() { destroy(); }

 private:
  // RAII helper: cleans up a partial capture and the manually-created parent
  // graph if the user-supplied callable throws between start- and end-capture.
  struct capture_guard_t {
    cudaStream_t stream{};
    cudaGraph_t parent{nullptr};
    bool capture_active{false};

    ~capture_guard_t() noexcept
    {
      if (capture_active) {
        cudaGraph_t dummy = nullptr;
        // best-effort; we're already unwinding
        cudaStreamEndCapture(stream, &dummy);
        if (dummy != nullptr) { cudaGraphDestroy(dummy); }
      }
      if (parent != nullptr) { cudaGraphDestroy(parent); }
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
