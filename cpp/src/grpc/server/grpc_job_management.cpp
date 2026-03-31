/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_pipe_serialization.hpp"
#include "grpc_server_types.hpp"

// write_to_pipe / read_from_pipe are defined in grpc_pipe_io.cpp

bool send_job_data_pipe(int worker_idx, const std::vector<uint8_t>& data)
{
  int fd;
  {
    std::lock_guard<std::mutex> lock(worker_pipes_mutex);
    if (worker_idx < 0 || worker_idx >= static_cast<int>(worker_pipes.size())) { return false; }
    fd = worker_pipes[worker_idx].to_worker_fd;
  }
  if (fd < 0) return false;

  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

bool recv_job_data_pipe(int fd, uint64_t expected_size, std::vector<uint8_t>& data)
{
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size != expected_size) {
    SERVER_LOG_ERROR("[Worker] Size mismatch: expected %lu, got %lu", expected_size, size);
    return false;
  }
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

bool send_incumbent_pipe(int fd, const std::vector<uint8_t>& data)
{
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

bool recv_incumbent_pipe(int fd, std::vector<uint8_t>& data)
{
  static constexpr uint64_t kMaxIncumbentBytes = 256ULL * 1024 * 1024;
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size > kMaxIncumbentBytes) return false;
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

// =============================================================================
// Job management
// =============================================================================

// Reserve a shared-memory job queue slot, store the serialized request data,
// and register the job in the tracker. Returns {true, job_id} on success.
// Uses CAS on `claimed` for lock-free slot reservation and release semantics
// on `ready` to publish all writes to the dispatch thread.
std::pair<bool, std::string> submit_job_async(std::vector<uint8_t>&& request_data,
                                              uint32_t problem_category)
{
  std::string job_id = generate_job_id();

  // Atomically reserve a free slot.
  int slot = -1;
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready.load()) continue;
    bool expected = false;
    if (job_queue[i].claimed.compare_exchange_strong(expected, true)) {
      slot = static_cast<int>(i);
      break;
    }
  }
  if (slot < 0) { return {false, "Job queue full"}; }

  // Populate the slot while we hold the `claimed` flag.
  copy_cstr(job_queue[slot].job_id, job_id);
  job_queue[slot].problem_category = problem_category;
  job_queue[slot].data_size        = request_data.size();
  job_queue[slot].cancelled.store(false);
  job_queue[slot].worker_index.store(-1);
  job_queue[slot].data_sent.store(false);
  job_queue[slot].is_chunked = false;
  job_queue[slot].worker_pid = 0;

  {
    std::lock_guard<std::mutex> lock(pending_data_mutex);
    pending_job_data[job_id] = std::move(request_data);
  }

  {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    JobInfo info;
    info.job_id           = job_id;
    info.status           = JobStatus::QUEUED;
    info.submit_time      = std::chrono::steady_clock::now();
    info.problem_category = problem_category;
    info.is_blocking      = false;
    job_tracker[job_id]   = std::move(info);
  }

  // Publish: release makes all writes above visible to the dispatch thread.
  job_queue[slot].ready.store(true, std::memory_order_release);
  job_queue[slot].claimed.store(false, std::memory_order_release);

  if (config.verbose) { SERVER_LOG_DEBUG("[Server] Job submitted (async): %s", job_id.c_str()); }

  return {true, job_id};
}

// Same as submit_job_async but for the chunked upload path. Stores the
// header + chunks in pending_chunked_data and marks the slot as is_chunked
// so the dispatch thread calls write_chunked_request_to_pipe().
std::pair<bool, std::string> submit_chunked_job_async(PendingChunkedUpload&& chunked_data,
                                                      uint32_t problem_category)
{
  std::string job_id = generate_job_id();

  int slot = -1;
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready.load()) continue;
    bool expected = false;
    if (job_queue[i].claimed.compare_exchange_strong(expected, true)) {
      slot = static_cast<int>(i);
      break;
    }
  }
  if (slot < 0) { return {false, "Job queue full"}; }

  copy_cstr(job_queue[slot].job_id, job_id);
  job_queue[slot].problem_category = problem_category;
  job_queue[slot].data_size        = 0;
  job_queue[slot].cancelled.store(false);
  job_queue[slot].worker_index.store(-1);
  job_queue[slot].data_sent.store(false);
  job_queue[slot].is_chunked = true;
  job_queue[slot].worker_pid = 0;

  {
    std::lock_guard<std::mutex> lock(pending_data_mutex);
    pending_chunked_data[job_id] = std::move(chunked_data);
  }

  {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    JobInfo info;
    info.job_id           = job_id;
    info.status           = JobStatus::QUEUED;
    info.submit_time      = std::chrono::steady_clock::now();
    info.problem_category = problem_category;
    info.is_blocking      = false;
    job_tracker[job_id]   = std::move(info);
  }

  job_queue[slot].ready.store(true, std::memory_order_release);
  job_queue[slot].claimed.store(false, std::memory_order_release);

  if (config.verbose) {
    SERVER_LOG_DEBUG("[Server] Chunked job submitted (async): %s", job_id.c_str());
  }

  return {true, job_id};
}

JobStatus check_job_status(const std::string& job_id, std::string& message)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) {
    message = "Job ID not found";
    return JobStatus::NOT_FOUND;
  }

  if (it->second.status == JobStatus::QUEUED) {
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready && job_queue[i].claimed &&
          std::string(job_queue[i].job_id) == job_id) {
        it->second.status = JobStatus::PROCESSING;
        break;
      }
    }
  }

  switch (it->second.status) {
    case JobStatus::QUEUED: message = "Job is queued"; break;
    case JobStatus::PROCESSING: message = "Job is being processed"; break;
    case JobStatus::COMPLETED: message = "Job completed"; break;
    case JobStatus::FAILED: message = "Job failed: " + it->second.error_message; break;
    case JobStatus::CANCELLED: message = "Job was cancelled"; break;
    default: message = "Unknown status";
  }

  return it->second.status;
}

void ensure_log_dir_exists()
{
  struct stat st;
  if (stat(LOG_DIR.c_str(), &st) != 0) { mkdir(LOG_DIR.c_str(), 0755); }
}

void delete_log_file(const std::string& job_id)
{
  std::string log_file = get_log_file_path(job_id);
  unlink(log_file.c_str());
}

int cancel_job(const std::string& job_id, JobStatus& job_status_out, std::string& message)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) {
    message        = "Job ID not found";
    job_status_out = JobStatus::NOT_FOUND;
    return 1;
  }

  JobStatus current_status = it->second.status;

  if (current_status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  if (current_status == JobStatus::CANCELLED) {
    message        = "Job already cancelled";
    job_status_out = JobStatus::CANCELLED;
    return 3;
  }

  if (current_status == JobStatus::FAILED) {
    message        = "Cannot cancel failed job";
    job_status_out = JobStatus::FAILED;
    return 2;
  }

  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (!job_queue[i].ready.load(std::memory_order_acquire)) continue;
    if (strcmp(job_queue[i].job_id, job_id.c_str()) != 0) continue;

    // Re-validate the slot: the job_id could have changed between the
    // initial check and now if the slot was recycled.  Load ready with
    // acquire so we see all writes that published it.
    if (!job_queue[i].ready.load(std::memory_order_acquire) ||
        strcmp(job_queue[i].job_id, job_id.c_str()) != 0) {
      continue;
    }

    pid_t worker_pid = job_queue[i].worker_pid.load(std::memory_order_relaxed);

    if (worker_pid > 0 && job_queue[i].claimed.load(std::memory_order_relaxed)) {
      if (config.verbose) {
        SERVER_LOG_DEBUG(
          "[Server] Cancelling running job %s (killing worker %d)", job_id.c_str(), worker_pid);
      }
      job_queue[i].cancelled.store(true, std::memory_order_release);
      kill(worker_pid, SIGKILL);
    } else {
      if (config.verbose) { SERVER_LOG_DEBUG("[Server] Cancelling queued job %s", job_id.c_str()); }
      job_queue[i].cancelled.store(true, std::memory_order_release);
    }

    it->second.status        = JobStatus::CANCELLED;
    it->second.error_message = "Job cancelled by user";
    job_status_out           = JobStatus::CANCELLED;
    message                  = "Job cancelled successfully";

    delete_log_file(job_id);

    {
      std::lock_guard<std::mutex> wlock(waiters_mutex);
      auto wit = waiting_threads.find(job_id);
      if (wit != waiting_threads.end()) {
        auto waiter = wit->second;
        {
          std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
          waiter->error_message = "Job cancelled by user";
          waiter->success       = false;
          waiter->ready         = true;
        }
        waiter->cv.notify_all();
        waiting_threads.erase(wit);
      }
    }

    return 0;
  }

  if (it->second.status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  it->second.status        = JobStatus::CANCELLED;
  it->second.error_message = "Job cancelled by user";
  job_status_out           = JobStatus::CANCELLED;
  message                  = "Job cancelled";

  {
    std::lock_guard<std::mutex> wlock(waiters_mutex);
    auto wit = waiting_threads.find(job_id);
    if (wit != waiting_threads.end()) {
      auto waiter = wit->second;
      {
        std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
        waiter->error_message = "Job cancelled by user";
        waiter->success       = false;
        waiter->ready         = true;
      }
      waiter->cv.notify_all();
      waiting_threads.erase(wit);
    }
  }

  return 0;
}

std::string generate_job_id()
{
  uuid_t uuid;
  uuid_generate_random(uuid);
  char buf[37];
  uuid_unparse_lower(uuid, buf);
  return std::string(buf);
}

#endif  // CUOPT_ENABLE_GRPC
