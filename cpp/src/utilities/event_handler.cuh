/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cuda_runtime.h>
#include <cuda/stream>
#include <raft/util/cuda_utils.cuh>

namespace cuopt {

class event_handler_t {
 public:
  event_handler_t() { RAFT_CUDA_TRY(cudaEventCreate(&event_)); }
  event_handler_t(unsigned int flags) { RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event_, flags)); }
  ~event_handler_t() { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event_)); }

  event_handler_t(const event_handler_t&)            = delete;
  event_handler_t& operator=(const event_handler_t&) = delete;

  void record(cuda::stream_ref stream_view)
  {
    RAFT_CUDA_TRY(cudaEventRecord(event_, stream_view.get()));
  }

  void record_with_flags(cuda::stream_ref stream_view, int flags)
  {
    RAFT_CUDA_TRY(cudaEventRecordWithFlags(event_, stream_view.get(), flags));
  }

  void stream_wait(cuda::stream_ref stream_view)
  {
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream_view.get(), event_));
  }

  float elapsed_time_since_ms(const event_handler_t& start)
  {
    float ms;
    // TODO: use cudaEventElapsedTime_v2 with CUDA 12.8?
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
  }

  void synchronize() { RAFT_CUDA_TRY(cudaEventSynchronize(event_)); }

 private:
  cudaEvent_t event_;
};
}  // namespace cuopt
