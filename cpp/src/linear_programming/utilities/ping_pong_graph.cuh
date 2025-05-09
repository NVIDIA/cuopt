/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_runtime.h>

namespace cuopt::linear_programming::detail {

// Helper class to capture and launch CUDA graph
// No additional checks for safe usage (calling launch() before initializing the graph) use with
// caution Binary part is because in pdlp we swap pointers instead of copying vectors to accept a
// valid pdhg step So every odd pdlp step it's one graph, every even step it's another graph
template <typename i_t>
class ping_pong_graph_t {
 public:
  ping_pong_graph_t(rmm::cuda_stream_view stream_view) : stream_view_(stream_view) {}

  ~ping_pong_graph_t()
  {
    if (even_initialized) { RAFT_CUDA_TRY_NO_THROW(cudaGraphExecDestroy(even_instance)); }
    if (odd_initialized) { RAFT_CUDA_TRY_NO_THROW(cudaGraphExecDestroy(odd_instance)); }
  }

  void start_capture(i_t total_pdlp_iterations)
  {
    if (total_pdlp_iterations % 2 == 0 && !even_initialized) {
      RAFT_CUDA_TRY(cudaStreamBeginCapture(stream_view_.value(), cudaStreamCaptureModeThreadLocal));
    } else if (total_pdlp_iterations % 2 == 1 && !odd_initialized) {
      RAFT_CUDA_TRY(cudaStreamBeginCapture(stream_view_.value(), cudaStreamCaptureModeThreadLocal));
    }
  }

  void end_capture(i_t total_pdlp_iterations)
  {
    if (total_pdlp_iterations % 2 == 0 && !even_initialized) {
      RAFT_CUDA_TRY(cudaStreamEndCapture(stream_view_.value(), &even_graph));
      // Extra NULL NULL 0 mandatory for cuda 11.8
      RAFT_CUDA_TRY(cudaGraphInstantiate(&even_instance, even_graph, nullptr, nullptr, 0));
      even_initialized = true;
      RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(even_graph));
    } else if (total_pdlp_iterations % 2 == 1 && !odd_initialized) {
      RAFT_CUDA_TRY(cudaStreamEndCapture(stream_view_.value(), &odd_graph));
      // Extra NULL NULL 0 mandatory for cuda 11.8
      RAFT_CUDA_TRY(cudaGraphInstantiate(&odd_instance, odd_graph, nullptr, nullptr, 0));
      odd_initialized = true;
      RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(odd_graph));
    }
  }

  void launch(i_t total_pdlp_iterations)
  {
    if (total_pdlp_iterations % 2 == 0 && even_initialized) {
      RAFT_CUDA_TRY(cudaGraphLaunch(even_instance, stream_view_.value()));
    } else if (total_pdlp_iterations % 2 == 1 && odd_initialized) {
      RAFT_CUDA_TRY(cudaGraphLaunch(odd_instance, stream_view_.value()));
    }
  }

  bool is_initialized(i_t total_pdlp_iterations)
  {
    return (total_pdlp_iterations % 2 == 0 && even_initialized) ||
           (total_pdlp_iterations % 2 == 1 && odd_initialized);
  }

 private:
  cudaGraph_t even_graph;
  cudaGraph_t odd_graph;
  cudaGraphExec_t even_instance;
  cudaGraphExec_t odd_instance;
  rmm::cuda_stream_view stream_view_;
  bool even_initialized{false};
  bool odd_initialized{false};
};
}  // namespace cuopt::linear_programming::detail