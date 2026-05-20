/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_pipe_serialization.hpp"
#include "grpc_server_types.hpp"

void cleanup_shared_memory()
{
  if (job_queue) {
    munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
    shm_unlink(SHM_JOB_QUEUE.c_str());
  }
  if (result_queue) {
    munmap(result_queue, sizeof(ResultQueueEntry) * MAX_RESULTS);
    shm_unlink(SHM_RESULT_QUEUE.c_str());
  }
  if (shm_ctrl) {
    munmap(shm_ctrl, sizeof(SharedMemoryControl));
    shm_unlink(SHM_CONTROL.c_str());
  }
}

static void close_and_reset(int& fd)
{
  if (fd >= 0) {
    close(fd);
    fd = -1;
  }
}

static void close_all_worker_pipes(WorkerPipes& wp)
{
  close_and_reset(wp.worker_read_fd);
  close_and_reset(wp.to_worker_fd);
  close_and_reset(wp.from_worker_fd);
  close_and_reset(wp.worker_write_fd);
  close_and_reset(wp.incumbent_from_worker_fd);
  close_and_reset(wp.worker_incumbent_write_fd);
}

bool create_worker_pipes(int worker_id)
{
  while (static_cast<int>(worker_pipes.size()) <= worker_id) {
    worker_pipes.push_back({-1, -1, -1, -1, -1, -1});
  }

  WorkerPipes& wp = worker_pipes[worker_id];

  int fds[2];

  if (pipe(fds) < 0) {
    SERVER_LOG_ERROR("[Server] Failed to create input pipe for worker %d", worker_id);
    return false;
  }
  wp.worker_read_fd = fds[0];
  wp.to_worker_fd   = fds[1];
  fcntl(wp.to_worker_fd, F_SETPIPE_SZ, kPipeBufferSize);

  if (pipe(fds) < 0) {
    SERVER_LOG_ERROR("[Server] Failed to create output pipe for worker %d", worker_id);
    close_all_worker_pipes(wp);
    return false;
  }
  wp.from_worker_fd  = fds[0];
  wp.worker_write_fd = fds[1];
  fcntl(wp.worker_write_fd, F_SETPIPE_SZ, kPipeBufferSize);

  if (pipe(fds) < 0) {
    SERVER_LOG_ERROR("[Server] Failed to create incumbent pipe for worker %d", worker_id);
    close_all_worker_pipes(wp);
    return false;
  }
  wp.incumbent_from_worker_fd  = fds[0];
  wp.worker_incumbent_write_fd = fds[1];

  return true;
}

void close_worker_pipes_server(int worker_id)
{
  if (worker_id < 0 || worker_id >= static_cast<int>(worker_pipes.size())) return;

  WorkerPipes& wp = worker_pipes[worker_id];
  close_and_reset(wp.to_worker_fd);
  close_and_reset(wp.from_worker_fd);
  close_and_reset(wp.incumbent_from_worker_fd);
}

void close_worker_pipes_child_ends(int worker_id)
{
  if (worker_id < 0 || worker_id >= static_cast<int>(worker_pipes.size())) return;

  WorkerPipes& wp = worker_pipes[worker_id];
  close_and_reset(wp.worker_read_fd);
  close_and_reset(wp.worker_write_fd);
  close_and_reset(wp.worker_incumbent_write_fd);
}

pid_t spawn_worker(int worker_id, bool is_replacement)
{
  std::lock_guard<std::mutex> lock(worker_pipes_mutex);

  if (is_replacement) { close_worker_pipes_server(worker_id); }

  if (!create_worker_pipes(worker_id)) {
    SERVER_LOG_ERROR("[Server] Failed to create pipes for %s%d",
                     is_replacement ? "replacement worker " : "worker ",
                     worker_id);
    return -1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    SERVER_LOG_ERROR("[Server] Failed to fork %s%d",
                     is_replacement ? "replacement worker " : "worker ",
                     worker_id);
    close_all_worker_pipes(worker_pipes[worker_id]);
    return -1;
  } else if (pid == 0) {
    // Child: close all fds belonging to other workers.
    for (int j = 0; j < static_cast<int>(worker_pipes.size()); ++j) {
      if (j != worker_id) { close_all_worker_pipes(worker_pipes[j]); }
    }
    // Close the server-side ends of this worker's pipes (child uses the other ends).
    close_and_reset(worker_pipes[worker_id].to_worker_fd);
    close_and_reset(worker_pipes[worker_id].from_worker_fd);
    close_and_reset(worker_pipes[worker_id].incumbent_from_worker_fd);
    worker_process(worker_id);
    _exit(0);
  }

  close_worker_pipes_child_ends(worker_id);
  return pid;
}

void spawn_workers()
{
  for (int i = 0; i < config.num_workers; ++i) {
    pid_t pid = spawn_worker(i, false);
    if (pid < 0) { continue; }
    worker_pids.push_back(pid);
  }
}

void wait_for_workers()
{
  for (pid_t pid : worker_pids) {
    if (pid <= 0) continue;
    int status;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  }
  worker_pids.clear();
}

pid_t spawn_single_worker(int worker_id) { return spawn_worker(worker_id, true); }

// Called by the worker-monitor thread when waitpid() detects a dead worker.
// Scans the shared-memory job queue for any job that was assigned to the dead
// worker and transitions it to FAILED (or CANCELLED if it was a user-initiated
// cancel that killed the worker). Three data structures must be updated:
//   1. pending_job_data  — discard the serialized request bytes
//   2. result_queue      — post a synthetic error result so the client unblocks
//   3. job_queue + job_tracker — mark the slot free and record final status
void mark_worker_jobs_failed(pid_t dead_worker_pid)
{
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready && job_queue[i].claimed && job_queue[i].worker_pid == dead_worker_pid) {
      std::string job_id(job_queue[i].job_id);
      bool was_cancelled = job_queue[i].cancelled;

      if (was_cancelled) {
        SERVER_LOG_WARN(
          "[Server] Worker %d killed for cancelled job: %s", dead_worker_pid, job_id.c_str());
      } else {
        SERVER_LOG_ERROR(
          "[Server] Worker %d died while processing job: %s", dead_worker_pid, job_id.c_str());
      }

      // 1. Drop the buffered request data (no longer needed).
      {
        std::lock_guard<std::mutex> lock(pending_data_mutex);
        pending_job_data.erase(job_id);
      }

      // 2. Post a synthetic error result into the first free result_queue slot
      //    so that any client polling for results gets a clear failure message.
      //    Uses the same CAS protocol as store_simple_result (see comment there).
      for (size_t j = 0; j < MAX_RESULTS; ++j) {
        if (result_queue[j].ready.load(std::memory_order_acquire)) continue;
        bool exp = false;
        if (!result_queue[j].claimed.compare_exchange_strong(
              exp, true, std::memory_order_acq_rel)) {
          continue;
        }
        if (result_queue[j].ready.load(std::memory_order_acquire)) {
          result_queue[j].claimed.store(false, std::memory_order_release);
          continue;
        }
        copy_cstr(result_queue[j].job_id, job_id);
        result_queue[j].status    = was_cancelled ? RESULT_CANCELLED : RESULT_ERROR;
        result_queue[j].data_size = 0;
        result_queue[j].worker_index.store(-1, std::memory_order_relaxed);
        copy_cstr(result_queue[j].error_message,
                  was_cancelled ? "Job was cancelled" : "Worker process died unexpectedly");
        result_queue[j].retrieved.store(false, std::memory_order_relaxed);
        result_queue[j].ready.store(true, std::memory_order_release);
        result_queue[j].claimed.store(false, std::memory_order_release);
        break;
      }

      // 3. Release the job queue slot and update the in-process job tracker.
      job_queue[i].worker_pid   = 0;
      job_queue[i].worker_index = -1;
      job_queue[i].data_sent    = false;
      job_queue[i].is_chunked   = false;
      job_queue[i].ready        = false;
      job_queue[i].claimed      = false;
      job_queue[i].cancelled    = false;

      {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        auto it = job_tracker.find(job_id);
        if (it != job_tracker.end()) {
          if (was_cancelled) {
            it->second.status        = JobStatus::CANCELLED;
            it->second.error_message = "Job was cancelled";
          } else {
            it->second.status        = JobStatus::FAILED;
            it->second.error_message = "Worker process died unexpectedly";
          }
        }
      }
    }
  }
}

#endif  // CUOPT_ENABLE_GRPC
