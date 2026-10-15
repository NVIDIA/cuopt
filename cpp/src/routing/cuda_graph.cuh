/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <utilities/scope_guard.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cuda/stream>

#pragma once

namespace cuopt {
namespace routing {
namespace detail {

// This is not a thread-safe class, be careful on multi-threading
struct cuda_graph_t {
  void start_capture(cuda::stream_ref stream)
  {
    // Use ThreadLocal mode to allow multi-threaded batch execution
    // Global mode blocks other streams from performing operations during capture
    RAFT_CUDA_TRY(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    capture_started = true;
  }

  void end_capture(cuda::stream_ref stream)
  {
    cuopt_expects(
      capture_started, error_type_t::RuntimeError, "CUDA graph capture has not started!");
    auto end_err    = cudaStreamEndCapture(stream.get(), &graph);
    capture_started = false;
    RAFT_CUDA_TRY(end_err);
    scope_guard destroy_graph([&] { RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(graph)); });
    if (graph_created) {
      // If the graph fails to update, errorNode will be set to the
      // node causing the failure and updateResult will be set to a
      // reason code.
      auto update_err = cudaGraphExecUpdate(instance, graph, &errorNode, &updateResult);
      if (update_err == cudaErrorGraphExecUpdateFailure) {
        // Expected update failures must not poison later CUDA error checks.
        cudaGetLastError();
      } else {
        RAFT_CUDA_TRY(update_err);
      }
    }
    // Instantiate during the first iteration or whenever the update
    // cannot reuse the existing executable graph.
    if (!graph_created || updateResult != cudaGraphExecUpdateSuccess) {
      // If a previous update failed, destroy the cudaGraphExec_t
      // before re-instantiating it
      if (graph_created) {
        graph_created = false;
        RAFT_CUDA_TRY(cudaGraphExecDestroy(instance));
      }
      // Instantiate graphExec from graph. The error node and
      // error message parameters are unused here.
      RAFT_CUDA_TRY(cudaGraphInstantiate(&instance, graph));
      graph_created = true;
    }
  }

  void launch_graph(cuda::stream_ref stream)
  {
    cuopt_expects(graph_created, error_type_t::RuntimeError, "CUDA graph is not instantiated!");
    RAFT_CUDA_TRY(cudaGraphLaunch(instance, stream.get()));
  }

  bool graph_created   = false;
  bool capture_started = false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaGraphExecUpdateResult updateResult;
  cudaGraphNode_t errorNode;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
