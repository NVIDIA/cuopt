/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
  ping_pong_graph_t(rmm::cuda_stream_view stream_view, bool is_batch_mode = false)
    : stream_view_(stream_view), is_batch_mode_(is_batch_mode)
  {
  }

  ~ping_pong_graph_t()
  {
    if (!is_batch_mode_) {
      if (even_initialized) { RAFT_CUDA_TRY_NO_THROW(cudaGraphExecDestroy(even_instance)); }
      if (odd_initialized) { RAFT_CUDA_TRY_NO_THROW(cudaGraphExecDestroy(odd_instance)); }
    }
  }

  void start_capture(i_t total_pdlp_iterations)
  {
    if (!is_batch_mode_) {
      if (total_pdlp_iterations % 2 == 0 && !even_initialized) {
        RAFT_CUDA_TRY(
          cudaStreamBeginCapture(stream_view_.value(), cudaStreamCaptureModeThreadLocal));
      } else if (total_pdlp_iterations % 2 == 1 && !odd_initialized) {
        RAFT_CUDA_TRY(
          cudaStreamBeginCapture(stream_view_.value(), cudaStreamCaptureModeThreadLocal));
      }
    }
  }

  void end_capture(i_t total_pdlp_iterations)
  {
    if (!is_batch_mode_) {
      if (total_pdlp_iterations % 2 == 0 && !even_initialized) {
        RAFT_CUDA_TRY(cudaStreamEndCapture(stream_view_.value(), &even_graph));
        RAFT_CUDA_TRY(cudaGraphInstantiate(&even_instance, even_graph));
        even_initialized = true;
        RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(even_graph));
      } else if (total_pdlp_iterations % 2 == 1 && !odd_initialized) {
        RAFT_CUDA_TRY(cudaStreamEndCapture(stream_view_.value(), &odd_graph));
        RAFT_CUDA_TRY(cudaGraphInstantiate(&odd_instance, odd_graph));
        odd_initialized = true;
        RAFT_CUDA_TRY_NO_THROW(cudaGraphDestroy(odd_graph));
      }
    }
  }

  void launch(i_t total_pdlp_iterations)
  {
    if (!is_batch_mode_) {
      if (total_pdlp_iterations % 2 == 0 && even_initialized) {
        RAFT_CUDA_TRY(cudaGraphLaunch(even_instance, stream_view_.value()));
      } else if (total_pdlp_iterations % 2 == 1 && odd_initialized) {
        RAFT_CUDA_TRY(cudaGraphLaunch(odd_instance, stream_view_.value()));
      }
    }
  }

  bool is_initialized(i_t total_pdlp_iterations)
  {
    if (!is_batch_mode_) {
      return (total_pdlp_iterations % 2 == 0 && even_initialized) ||
             (total_pdlp_iterations % 2 == 1 && odd_initialized);
    }
    return false;
  }

 private:
  cudaGraph_t even_graph;
  cudaGraph_t odd_graph;
  cudaGraphExec_t even_instance;
  cudaGraphExec_t odd_instance;
  rmm::cuda_stream_view stream_view_;
  bool even_initialized{false};
  bool odd_initialized{false};
  // Temporary fix to disable cuda graph in batch mode
  bool is_batch_mode_{false};
};
}  // namespace cuopt::linear_programming::detail
