/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/utilities/ping_pong_graph.cuh>

#include <raft/core/error.hpp>

#include <utilities/logger.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t>
ping_pong_graph_t<i_t>::ping_pong_graph_t(rmm::cuda_stream_view stream_view,
                                          bool is_legacy_batch_mode)
  : stream_view_(stream_view), is_legacy_batch_mode_(is_legacy_batch_mode)
{
}

template <typename i_t>
ping_pong_graph_t<i_t>::~ping_pong_graph_t()
{
#ifndef CUPDLP_DEBUG_MODE
  if (!is_legacy_batch_mode_) {
    if (even_initialized) { RAFT_CUDA_TRY_NO_THROW(cudaGraphExecDestroy(even_instance)); }
    if (odd_initialized) { RAFT_CUDA_TRY_NO_THROW(cudaGraphExecDestroy(odd_instance)); }
  }
#endif
}

template <typename i_t>
void ping_pong_graph_t<i_t>::start_capture(i_t total_pdlp_iterations)
{
#ifndef CUPDLP_DEBUG_MODE
  if (!is_legacy_batch_mode_) {
    if (total_pdlp_iterations % 2 == 0 && !even_initialized) {
      RAFT_CUDA_TRY(cudaStreamBeginCapture(stream_view_.value(), cudaStreamCaptureModeThreadLocal));
      capture_even_active_ = true;
    } else if (total_pdlp_iterations % 2 == 1 && !odd_initialized) {
      RAFT_CUDA_TRY(cudaStreamBeginCapture(stream_view_.value(), cudaStreamCaptureModeThreadLocal));
      capture_odd_active_ = true;
    }
  }
#endif
}

template <typename i_t>
void ping_pong_graph_t<i_t>::end_capture(i_t total_pdlp_iterations)
{
#ifndef CUPDLP_DEBUG_MODE
  if (!is_legacy_batch_mode_) {
    if (total_pdlp_iterations % 2 == 0 && !even_initialized) {
      RAFT_CUDA_TRY(cudaStreamEndCapture(stream_view_.value(), &even_graph));
      capture_even_active_ = false;
      RAFT_CUDA_TRY(cudaGraphInstantiate(&even_instance, even_graph));
      even_initialized = true;
      RAFT_CUDA_TRY(cudaGraphDestroy(even_graph));
    } else if (total_pdlp_iterations % 2 == 1 && !odd_initialized) {
      RAFT_CUDA_TRY(cudaStreamEndCapture(stream_view_.value(), &odd_graph));
      capture_odd_active_ = false;
      RAFT_CUDA_TRY(cudaGraphInstantiate(&odd_instance, odd_graph));
      odd_initialized = true;
      RAFT_CUDA_TRY(cudaGraphDestroy(odd_graph));
    }
  }
#endif
}

template <typename i_t>
void ping_pong_graph_t<i_t>::launch(i_t total_pdlp_iterations)
{
#ifndef CUPDLP_DEBUG_MODE
  if (!is_legacy_batch_mode_) {
    if (total_pdlp_iterations % 2 == 0 && even_initialized) {
      RAFT_CUDA_TRY(cudaGraphLaunch(even_instance, stream_view_.value()));
    } else if (total_pdlp_iterations % 2 == 1 && odd_initialized) {
      RAFT_CUDA_TRY(cudaGraphLaunch(odd_instance, stream_view_.value()));
    }
  }
#endif
}

template <typename i_t>
bool ping_pong_graph_t<i_t>::is_initialized(i_t total_pdlp_iterations)
{
#ifndef CUPDLP_DEBUG_MODE
  if (!is_legacy_batch_mode_) {
    return (total_pdlp_iterations % 2 == 0 && even_initialized) ||
           (total_pdlp_iterations % 2 == 1 && odd_initialized);
  }
#endif
  return false;
}

template class ping_pong_graph_t<int>;

}  // namespace cuopt::linear_programming::detail
