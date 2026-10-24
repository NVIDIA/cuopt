/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include <sys/types.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

// Shared-memory job slot used by the gRPC parent and workers.
// Slot index is ownership only; dispatch order uses enqueue_seq.
struct JobQueueEntry {
  char job_id[64];
  uint32_t problem_category;      // cuopt::remote::ProblemCategory (LP, MIP, ...)
  uint64_t data_size;             // Size of problem data (uint64 for large problems)
  std::atomic<bool> ready;        // Job is ready to be processed
  std::atomic<bool> claimed;      // Worker has claimed this job
  std::atomic<pid_t> worker_pid;  // PID of worker that claimed this job (0 if none)
  std::atomic<bool> cancelled;    // Job has been cancelled (worker should skip)
  std::atomic<int> worker_index;  // Index of worker that claimed this job (-1 if none)
  std::atomic<bool> data_sent;    // Server has sent data to worker's pipe
  std::atomic<bool> is_chunked;   // True when data is in pending_chunked_data (streamed to pipe)
  std::atomic<bool>
    unified_header;  // Chunked pipe header is SubmitJobRequest, not ChunkedProblemHeader
  // Monotonic submission order. 0 means the slot is free or has been reset;
  // workers never dispatch a ready entry with enqueue_seq == 0.
  std::atomic<uint64_t> enqueue_seq;
};

// Claim the ready, unclaimed job with the lowest enqueue_seq.
// Retries when a concurrent claim or slot recycle invalidates the selection.
// Returns the slot index, or -1 when no dispatchable job is present.
inline int claim_ready_job_slot(JobQueueEntry* queue,
                                size_t n_jobs,
                                int worker_id,
                                pid_t worker_pid)
{
  while (true) {
    int best_slot     = -1;
    uint64_t best_seq = std::numeric_limits<uint64_t>::max();

    for (size_t i = 0; i < n_jobs; ++i) {
      // Acquire on ready publishes the enqueue_seq written before ready=true.
      if (!queue[i].ready.load(std::memory_order_acquire)) { continue; }
      if (queue[i].claimed.load(std::memory_order_acquire)) { continue; }
      const uint64_t seq = queue[i].enqueue_seq.load(std::memory_order_relaxed);
      if (seq == 0) { continue; }
      if (best_slot < 0 || seq < best_seq) {
        best_seq  = seq;
        best_slot = static_cast<int>(i);
      }
    }

    if (best_slot < 0) { return -1; }

    bool expected = false;
    if (!queue[best_slot].claimed.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
      continue;
    }

    if (!queue[best_slot].ready.load(std::memory_order_acquire) ||
        queue[best_slot].enqueue_seq.load(std::memory_order_relaxed) != best_seq) {
      queue[best_slot].claimed.store(false, std::memory_order_release);
      continue;
    }

    queue[best_slot].worker_pid.store(worker_pid, std::memory_order_relaxed);
    queue[best_slot].worker_index.store(worker_id, std::memory_order_relaxed);
    return best_slot;
  }
}

#endif  // CUOPT_ENABLE_GRPC
