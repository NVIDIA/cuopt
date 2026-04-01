/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_field_element_size.hpp"
#include "grpc_pipe_serialization.hpp"
#include "grpc_server_types.hpp"

class CuOptRemoteServiceImpl final : public cuopt::remote::CuOptRemoteService::Service {
 public:
  // Unary submit: the entire problem fits in a single gRPC message.
  // Serializes the request and delegates slot reservation + tracking to
  // submit_job_async (shared with the chunked path's submit_chunked_job_async).
  Status SubmitJob(ServerContext* context,
                   const cuopt::remote::SubmitJobRequest* request,
                   cuopt::remote::SubmitJobResponse* response) override
  {
    uint32_t problem_category;
    if (request->has_lp_request()) {
      problem_category = cuopt::remote::LP;
    } else if (request->has_mip_request()) {
      problem_category = cuopt::remote::MIP;
    } else {
      return Status(StatusCode::INVALID_ARGUMENT, "No problem data provided");
    }

    if (config.verbose && problem_category == cuopt::remote::LP) {
      const auto& lp_req = request->lp_request();
      SERVER_LOG_DEBUG(
        "[gRPC] SubmitJob LP fields: bytes=%zu objective_scaling_factor=%f objective_offset=%f "
        "iteration_limit=%d method=%d",
        lp_req.ByteSizeLong(),
        lp_req.problem().objective_scaling_factor(),
        lp_req.problem().objective_offset(),
        lp_req.settings().iteration_limit(),
        lp_req.settings().method());
    }

    auto job_data = serialize_submit_request_to_pipe(*request);
    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] SubmitJob: UNARY %s, pipe payload=%zu bytes",
                       problem_category == cuopt::remote::LP ? "LP" : "MIP",
                       job_data.size());
    }

    auto [ok, job_id] = submit_job_async(std::move(job_data), problem_category);
    if (!ok) { return Status(StatusCode::RESOURCE_EXHAUSTED, job_id); }

    response->set_job_id(job_id);
    response->set_message("Job submitted successfully");

    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] Job submitted: %s (type=%s)",
                       job_id.c_str(),
                       problem_category == cuopt::remote::LP ? "LP" : "MIP");
    }

    return Status::OK;
  }

  // =========================================================================
  // Chunked Array Upload
  // =========================================================================

  Status StartChunkedUpload(ServerContext* context,
                            const cuopt::remote::StartChunkedUploadRequest* request,
                            cuopt::remote::StartChunkedUploadResponse* response) override
  {
    (void)context;

    std::string upload_id     = generate_job_id();
    const auto& header        = request->problem_header();
    uint32_t problem_category = header.header().problem_category();

    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] StartChunkedUpload upload_id=%s problem_category=%u",
                       upload_id.c_str(),
                       problem_category);
    }

    {
      std::lock_guard<std::mutex> lock(chunked_uploads_mutex);
      if (chunked_uploads.size() >= kMaxChunkedSessions) {
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "Too many concurrent chunked upload sessions (limit " +
                        std::to_string(kMaxChunkedSessions) + ")");
      }
      auto& state            = chunked_uploads[upload_id];
      state.problem_category = problem_category;
      state.header           = header;
      state.total_chunks     = 0;
      state.last_activity    = std::chrono::steady_clock::now();
    }

    response->set_upload_id(upload_id);
    response->set_max_message_bytes(server_max_message_bytes());

    return Status::OK;
  }

  // Receive one chunk of array data for a chunked upload session.
  // Chunks are accumulated in memory until FinishChunkedUpload, which hands
  // them to the dispatch thread for pipe serialization to the worker.
  Status SendArrayChunk(ServerContext* context,
                        const cuopt::remote::SendArrayChunkRequest* request,
                        cuopt::remote::SendArrayChunkResponse* response) override
  {
    (void)context;

    const std::string& upload_id = request->upload_id();
    const auto& ac               = request->chunk();

    std::lock_guard<std::mutex> lock(chunked_uploads_mutex);
    auto it = chunked_uploads.find(upload_id);
    if (it == chunked_uploads.end()) {
      return Status(StatusCode::NOT_FOUND, "Unknown upload_id: " + upload_id);
    }

    auto& state         = it->second;
    state.last_activity = std::chrono::steady_clock::now();

    int32_t field_id    = static_cast<int32_t>(ac.field_id());
    int64_t elem_offset = ac.element_offset();
    int64_t total_elems = ac.total_elements();
    const auto& raw     = ac.data();

    if (!cuopt::remote::ArrayFieldId_IsValid(field_id)) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Unknown array field_id: " + std::to_string(field_id));
    }
    if (elem_offset < 0) {
      return Status(StatusCode::INVALID_ARGUMENT, "element_offset must be non-negative");
    }
    if (total_elems < 0) {
      return Status(StatusCode::INVALID_ARGUMENT, "total_elements must be non-negative");
    }

    // On the first chunk for a field, record its total size and element width.
    // Subsequent chunks for the same field reuse these values.
    auto& meta = state.field_meta[field_id];
    if (meta.total_elements == 0 && total_elems > 0) {
      int64_t elem_size = array_field_element_size(ac.field_id());
      if (total_elems > kMaxChunkedArrayBytes / elem_size) {
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "Array too large (" + std::to_string(total_elems) + " x " +
                        std::to_string(elem_size) + " bytes exceeds " +
                        std::to_string(kMaxChunkedArrayBytes) + " byte limit)");
      }
      meta.total_elements = total_elems;
      meta.element_size   = elem_size;
    }

    // Validate that the chunk's byte range falls within the declared array bounds.
    int64_t elem_size = meta.element_size > 0 ? meta.element_size : 1;

    if (elem_size > 1 && (raw.size() % static_cast<size_t>(elem_size)) != 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Chunk data size (" + std::to_string(raw.size()) +
                      ") not aligned to element size (" + std::to_string(elem_size) + ")");
    }

    int64_t array_bytes = meta.total_elements * elem_size;
    if (elem_offset > meta.total_elements) {
      return Status(StatusCode::INVALID_ARGUMENT, "ArrayChunk offset exceeds array size");
    }
    int64_t byte_offset = elem_offset * elem_size;
    if (byte_offset + static_cast<int64_t>(raw.size()) > array_bytes) {
      return Status(StatusCode::INVALID_ARGUMENT, "ArrayChunk out of bounds");
    }

    // Accumulate: the raw ArrayChunk protobuf is stored as-is and will be
    // assembled into contiguous arrays during pipe serialization.
    meta.received_bytes += static_cast<int64_t>(raw.size());
    state.total_bytes += static_cast<int64_t>(raw.size());
    state.chunks.push_back(ac);
    ++state.total_chunks;

    response->set_upload_id(upload_id);
    response->set_chunks_received(state.total_chunks);
    return Status::OK;
  }

  // Finalize a chunked upload: move the accumulated header + chunks into a
  // PendingChunkedUpload and enqueue it as a job. The dispatch thread will
  // call write_chunked_request_to_pipe() to send it to a worker.
  Status FinishChunkedUpload(ServerContext* context,
                             const cuopt::remote::FinishChunkedUploadRequest* request,
                             cuopt::remote::SubmitJobResponse* response) override
  {
    (void)context;

    const std::string& upload_id = request->upload_id();

    // Take ownership of the upload session and remove it from the active map.
    ChunkedUploadState state;
    {
      std::lock_guard<std::mutex> lock(chunked_uploads_mutex);
      auto it = chunked_uploads.find(upload_id);
      if (it == chunked_uploads.end()) {
        return Status(StatusCode::NOT_FOUND, "Unknown upload_id: " + upload_id);
      }
      state = std::move(it->second);
      chunked_uploads.erase(it);
    }

    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] FinishChunkedUpload upload_id=%s chunks=%d fields=%zu",
                       upload_id.c_str(),
                       state.total_chunks,
                       state.field_meta.size());
    }

    // Package the header and chunks for the dispatch thread. Field metadata
    // was only needed for validation during SendArrayChunk and can be dropped.
    PendingChunkedUpload pending;
    pending.header = std::move(state.header);
    pending.chunks = std::move(state.chunks);
    state.field_meta.clear();

    if (config.verbose) {
      SERVER_LOG_DEBUG(
        "[gRPC] FinishChunkedUpload: CHUNKED path, %d chunks, %ld bytes, upload_id=%s",
        state.total_chunks,
        state.total_bytes,
        upload_id.c_str());
    }

    auto [ok, job_id] = submit_chunked_job_async(std::move(pending), state.problem_category);
    if (!ok) { return Status(StatusCode::RESOURCE_EXHAUSTED, job_id); }

    response->set_job_id(job_id);
    response->set_message("Job submitted via chunked arrays");

    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] FinishChunkedUpload enqueued job: %s (type=%s)",
                       job_id.c_str(),
                       state.problem_category == cuopt::remote::MIP ? "MIP" : "LP");
    }

    return Status::OK;
  }

  // =========================================================================
  // Job Status and Result RPCs
  // =========================================================================

  Status CheckStatus(ServerContext* context,
                     const cuopt::remote::StatusRequest* request,
                     cuopt::remote::StatusResponse* response) override
  {
    (void)context;
    std::string job_id = request->job_id();

    std::string message;
    JobStatus status = check_job_status(job_id, message);

    switch (status) {
      case JobStatus::QUEUED: response->set_job_status(cuopt::remote::QUEUED); break;
      case JobStatus::PROCESSING: response->set_job_status(cuopt::remote::PROCESSING); break;
      case JobStatus::COMPLETED: response->set_job_status(cuopt::remote::COMPLETED); break;
      case JobStatus::FAILED: response->set_job_status(cuopt::remote::FAILED); break;
      case JobStatus::CANCELLED: response->set_job_status(cuopt::remote::CANCELLED); break;
      default: response->set_job_status(cuopt::remote::NOT_FOUND); break;
    }
    response->set_message(message);

    response->set_max_message_bytes(server_max_message_bytes());

    int64_t result_size_bytes = 0;
    if (status == JobStatus::COMPLETED) {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      auto it = job_tracker.find(job_id);
      if (it != job_tracker.end()) { result_size_bytes = it->second.result_size_bytes; }
    }
    response->set_result_size_bytes(result_size_bytes);

    return Status::OK;
  }

  // Return the full result in a single gRPC response (unary path).
  // If the result exceeds the server's max message size, the client must
  // fall back to the chunked download RPCs instead.
  Status GetResult(ServerContext* context,
                   const cuopt::remote::GetResultRequest* request,
                   cuopt::remote::ResultResponse* response) override
  {
    (void)context;
    std::string job_id = request->job_id();

    std::lock_guard<std::mutex> lock(tracker_mutex);
    auto it = job_tracker.find(job_id);

    if (it == job_tracker.end()) { return Status(StatusCode::NOT_FOUND, "Job not found"); }

    if (it->second.status != JobStatus::COMPLETED && it->second.status != JobStatus::FAILED) {
      return Status(StatusCode::UNAVAILABLE, "Result not ready");
    }

    if (it->second.status == JobStatus::FAILED) {
      response->set_status(cuopt::remote::ERROR_SOLVE_FAILED);
      response->set_error_message(it->second.error_message);
      return Status::OK;
    }

    // Guard against results that would exceed gRPC/protobuf message limits.
    // The client detects RESOURCE_EXHAUSTED and switches to chunked download.
    int64_t total_result_bytes = it->second.result_size_bytes;
    const int64_t max_bytes    = server_max_message_bytes();
    if (max_bytes > 0 && total_result_bytes > max_bytes) {
      std::string msg = "Result size (~" + std::to_string(total_result_bytes) +
                        " bytes) exceeds max message size (" + std::to_string(max_bytes) +
                        " bytes). Use StartChunkedDownload/GetResultChunk RPCs instead.";
      if (config.verbose) {
        SERVER_LOG_DEBUG("[gRPC] GetResult rejected for job %s: %s", job_id.c_str(), msg.c_str());
      }
      return Status(StatusCode::RESOURCE_EXHAUSTED, msg);
    }

    // Build the full protobuf solution from the raw arrays that were read
    // back from the worker pipe by the result retrieval thread.
    if (it->second.problem_category == cuopt::remote::MIP) {
      cuopt::remote::MIPSolution mip_solution;
      build_mip_solution_proto<int, double>(
        it->second.result_header, it->second.result_arrays, &mip_solution);
      response->mutable_mip_solution()->Swap(&mip_solution);
    } else {
      cuopt::remote::LPSolution lp_solution;
      build_lp_solution_proto<int, double>(
        it->second.result_header, it->second.result_arrays, &lp_solution);
      response->mutable_lp_solution()->Swap(&lp_solution);
    }

    response->set_status(cuopt::remote::SUCCESS);
    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] GetResult: UNARY response for job %s (%ld bytes, %zu arrays)",
                       job_id.c_str(),
                       total_result_bytes,
                       it->second.result_arrays.size());
    }

    return Status::OK;
  }

  // =========================================================================
  // Chunked Result Download RPCs
  // =========================================================================

  // Begin a chunked result download: snapshot the result arrays into a
  // download session. The client calls GetResultChunk to fetch slices and
  // FinishChunkedDownload when done (which frees the session).
  Status StartChunkedDownload(ServerContext* context,
                              const cuopt::remote::StartChunkedDownloadRequest* request,
                              cuopt::remote::StartChunkedDownloadResponse* response) override
  {
    std::string job_id = request->job_id();

    // Copy the result data into a download session. This snapshot lets the
    // client fetch chunks at its own pace without holding the tracker lock.
    ChunkedDownloadState state;
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      auto it = job_tracker.find(job_id);
      if (it == job_tracker.end()) {
        return Status(StatusCode::NOT_FOUND, "Job not found: " + job_id);
      }
      if (it->second.status != JobStatus::COMPLETED) {
        return Status(StatusCode::FAILED_PRECONDITION, "Result not ready for job: " + job_id);
      }
      state.problem_category = it->second.problem_category;
      state.created          = std::chrono::steady_clock::now();
      state.result_header    = it->second.result_header;
      state.raw_arrays       = it->second.result_arrays;
    }

    response->mutable_header()->CopyFrom(state.result_header);

    std::string download_id = generate_job_id();
    response->set_download_id(download_id);
    response->set_max_message_bytes(server_max_message_bytes());

    {
      std::lock_guard<std::mutex> lock(chunked_downloads_mutex);
      if (chunked_downloads.size() >= kMaxChunkedSessions) {
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "Too many concurrent chunked download sessions (limit " +
                        std::to_string(kMaxChunkedSessions) + ")");
      }
      chunked_downloads[download_id] = std::move(state);
    }

    if (config.verbose) {
      SERVER_LOG_DEBUG(
        "[gRPC] StartChunkedDownload: CHUNKED response for job %s, download_id=%s, arrays=%d, "
        "problem_category=%u",
        job_id.c_str(),
        download_id.c_str(),
        response->header().arrays_size(),
        state.problem_category);
    }

    return Status::OK;
  }

  Status GetResultChunk(ServerContext* context,
                        const cuopt::remote::GetResultChunkRequest* request,
                        cuopt::remote::GetResultChunkResponse* response) override
  {
    std::string download_id = request->download_id();
    auto field_id           = request->field_id();
    int64_t elem_offset     = request->element_offset();
    int64_t max_elements    = request->max_elements();

    std::lock_guard<std::mutex> lock(chunked_downloads_mutex);
    auto it = chunked_downloads.find(download_id);
    if (it == chunked_downloads.end()) {
      return Status(StatusCode::NOT_FOUND, "Unknown download_id: " + download_id);
    }

    const auto& state = it->second;

    const uint8_t* raw_bytes = nullptr;
    int64_t total_bytes      = 0;
    auto array_it            = state.raw_arrays.find(static_cast<int32_t>(field_id));
    if (array_it != state.raw_arrays.end() && !array_it->second.empty()) {
      raw_bytes   = array_it->second.data();
      total_bytes = static_cast<int64_t>(array_it->second.size());
    }

    if (raw_bytes == nullptr || total_bytes == 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Unknown or empty result field: " + std::to_string(field_id));
    }

    const int64_t elem_size  = sizeof(double);
    const int64_t array_size = total_bytes / elem_size;

    if (elem_offset < 0 || elem_offset >= array_size) {
      return Status(StatusCode::OUT_OF_RANGE,
                    "element_offset " + std::to_string(elem_offset) + " out of range [0, " +
                      std::to_string(array_size) + ")");
    }

    int64_t elems_available = array_size - elem_offset;
    int64_t elems_to_send =
      (max_elements > 0) ? std::min(max_elements, elems_available) : elems_available;

    response->set_download_id(download_id);
    response->set_field_id(field_id);
    response->set_element_offset(elem_offset);
    response->set_elements_in_chunk(elems_to_send);
    response->set_data(reinterpret_cast<const char*>(raw_bytes + elem_offset * elem_size),
                       static_cast<size_t>(elems_to_send) * elem_size);

    return Status::OK;
  }

  Status FinishChunkedDownload(ServerContext* context,
                               const cuopt::remote::FinishChunkedDownloadRequest* request,
                               cuopt::remote::FinishChunkedDownloadResponse* response) override
  {
    std::string download_id = request->download_id();
    response->set_download_id(download_id);

    std::lock_guard<std::mutex> lock(chunked_downloads_mutex);
    auto it = chunked_downloads.find(download_id);
    if (it == chunked_downloads.end()) {
      return Status(StatusCode::NOT_FOUND, "Unknown download_id: " + download_id);
    }

    if (config.verbose) {
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - it->second.created)
                          .count();
      SERVER_LOG_DEBUG("[gRPC] FinishChunkedDownload: download_id=%s elapsed_ms=%ld",
                       download_id.c_str(),
                       elapsed_ms);
    }

    chunked_downloads.erase(it);
    return Status::OK;
  }

  // =========================================================================
  // Delete, Cancel, Wait, StreamLogs, GetIncumbents
  // =========================================================================

  Status DeleteResult(ServerContext* context,
                      const cuopt::remote::DeleteRequest* request,
                      cuopt::remote::DeleteResponse* response) override
  {
    std::string job_id = request->job_id();

    size_t erased = 0;
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      erased = job_tracker.erase(job_id);
    }

    if (erased == 0) {
      response->set_status(cuopt::remote::ERROR_NOT_FOUND);
      response->set_message("Job not found: " + job_id);
      if (config.verbose) {
        SERVER_LOG_DEBUG("[gRPC] DeleteResult job not found: %s", job_id.c_str());
      }
      return Status::OK;
    }

    delete_log_file(job_id);

    response->set_status(cuopt::remote::SUCCESS);
    response->set_message("Result deleted");

    if (config.verbose) { SERVER_LOG_DEBUG("[gRPC] Result deleted for job: %s", job_id.c_str()); }

    return Status::OK;
  }

  Status CancelJob(ServerContext* context,
                   const cuopt::remote::CancelRequest* request,
                   cuopt::remote::CancelResponse* response) override
  {
    (void)context;
    std::string job_id = request->job_id();

    JobStatus internal_status = JobStatus::NOT_FOUND;
    std::string message;
    int rc = cancel_job(job_id, internal_status, message);

    cuopt::remote::JobStatus pb_status = cuopt::remote::NOT_FOUND;
    switch (internal_status) {
      case JobStatus::QUEUED: pb_status = cuopt::remote::QUEUED; break;
      case JobStatus::PROCESSING: pb_status = cuopt::remote::PROCESSING; break;
      case JobStatus::COMPLETED: pb_status = cuopt::remote::COMPLETED; break;
      case JobStatus::FAILED: pb_status = cuopt::remote::FAILED; break;
      case JobStatus::CANCELLED: pb_status = cuopt::remote::CANCELLED; break;
      case JobStatus::NOT_FOUND: pb_status = cuopt::remote::NOT_FOUND; break;
    }

    response->set_job_status(pb_status);
    response->set_message(message);

    if (rc == 0 || rc == 3) {
      response->set_status(cuopt::remote::SUCCESS);
    } else if (rc == 1) {
      response->set_status(cuopt::remote::ERROR_NOT_FOUND);
    } else {
      response->set_status(cuopt::remote::ERROR_INVALID_REQUEST);
    }

    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] CancelJob job_id=%s rc=%d status=%d msg=%s",
                       job_id.c_str(),
                       rc,
                       static_cast<int>(pb_status),
                       message.c_str());
    }

    return Status::OK;
  }

  // Block until a job reaches a terminal state (COMPLETED / FAILED / CANCELLED).
  // Uses a shared JobWaiter with a condition variable that the result retrieval
  // thread signals when it processes the job's result. Falls back to polling
  // every 200ms in case the signal is missed (e.g., worker crash recovery).
  Status WaitForCompletion(ServerContext* context,
                           const cuopt::remote::WaitRequest* request,
                           cuopt::remote::WaitResponse* response) override
  {
    const std::string job_id = request->job_id();

    // Fast path: if the job is already in a terminal state, return immediately.
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      auto it = job_tracker.find(job_id);
      if (it == job_tracker.end()) {
        response->set_job_status(cuopt::remote::NOT_FOUND);
        response->set_message("Job not found");
        response->set_result_size_bytes(0);
        return Status::OK;
      }
      if (it->second.status == JobStatus::COMPLETED) {
        response->set_job_status(cuopt::remote::COMPLETED);
        response->set_message("");
        response->set_result_size_bytes(it->second.result_size_bytes);
        return Status::OK;
      }
      if (it->second.status == JobStatus::FAILED) {
        response->set_job_status(cuopt::remote::FAILED);
        response->set_message(it->second.error_message);
        response->set_result_size_bytes(0);
        return Status::OK;
      }
      if (it->second.status == JobStatus::CANCELLED) {
        response->set_job_status(cuopt::remote::CANCELLED);
        response->set_message("Job was cancelled");
        response->set_result_size_bytes(0);
        return Status::OK;
      }
    }

    // Slow path: register a waiter. Multiple concurrent WaitForCompletion
    // RPCs for the same job share a single JobWaiter instance.
    std::shared_ptr<JobWaiter> waiter;
    {
      std::lock_guard<std::mutex> lock(waiters_mutex);
      auto it = waiting_threads.find(job_id);
      if (it != waiting_threads.end()) {
        waiter = it->second;
      } else {
        waiter                  = std::make_shared<JobWaiter>();
        waiting_threads[job_id] = waiter;
      }
    }
    waiter->waiters.fetch_add(1, std::memory_order_relaxed);

    // Wait loop: cv is signaled by the result retrieval thread; we also
    // poll check_job_status as a safety net and check for client disconnect.
    // All exit paths (ready, terminal status, cancellation) break out of the
    // loop so that cleanup (waiters decrement) happens in one place below.
    bool client_cancelled = false;
    {
      std::unique_lock<std::mutex> lock(waiter->mutex);
      while (!waiter->ready) {
        if (context->IsCancelled()) {
          client_cancelled = true;
          break;
        }
        lock.unlock();
        std::string msg;
        JobStatus current = check_job_status(job_id, msg);
        lock.lock();
        if (current == JobStatus::COMPLETED || current == JobStatus::FAILED ||
            current == JobStatus::CANCELLED) {
          break;
        }
        waiter->cv.wait_for(lock, std::chrono::milliseconds(200));
      }
    }

    waiter->waiters.fetch_sub(1, std::memory_order_relaxed);

    if (client_cancelled) {
      if (config.verbose) {
        SERVER_LOG_DEBUG("[gRPC] WaitForCompletion cancelled by client, job_id=%s", job_id.c_str());
      }
      return Status(StatusCode::CANCELLED, "Client cancelled WaitForCompletion");
    }

    // Build the response from the final job state.
    // The waiter's `success` flag is set by the result retrieval thread when it
    // processes a successful result. It is true only for normal completion.
    if (waiter->success) {
      response->set_job_status(cuopt::remote::COMPLETED);
      response->set_message("");
      {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        auto job_it = job_tracker.find(job_id);
        response->set_result_size_bytes(
          (job_it != job_tracker.end()) ? job_it->second.result_size_bytes : 0);
      }
    } else {
      // The waiter was not signaled with success. This happens when:
      //   - The job failed (solver error, worker crash)
      //   - The job was cancelled by the user
      //   - The wait loop exited via the polling safety net (check_job_status
      //     detected a terminal state before the cv was signaled)
      // Re-check the authoritative job status to determine what happened.
      std::string msg;
      JobStatus status = check_job_status(job_id, msg);
      switch (status) {
        case JobStatus::COMPLETED: {
          response->set_job_status(cuopt::remote::COMPLETED);
          response->set_message("");
          std::lock_guard<std::mutex> lock(tracker_mutex);
          auto job_it = job_tracker.find(job_id);
          response->set_result_size_bytes(
            (job_it != job_tracker.end()) ? job_it->second.result_size_bytes : 0);
          break;
        }
        case JobStatus::FAILED: response->set_job_status(cuopt::remote::FAILED); break;
        case JobStatus::CANCELLED: response->set_job_status(cuopt::remote::CANCELLED); break;
        case JobStatus::NOT_FOUND: response->set_job_status(cuopt::remote::NOT_FOUND); break;
        default: response->set_job_status(cuopt::remote::FAILED); break;
      }
      if (status != JobStatus::COMPLETED) {
        response->set_message(msg);
        response->set_result_size_bytes(0);
      }
    }

    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] WaitForCompletion finished job_id=%s", job_id.c_str());
    }

    return Status::OK;
  }

  // Server-streaming RPC: tails the solver log file for a job, sending one
  // LogMessage per line as new output appears (like `tail -f` over gRPC).
  // The client supplies a byte offset so it can resume after reconnection.
  // The stream ends with a sentinel message (job_complete=true) once the
  // job reaches a terminal state and all remaining log content is flushed.
  Status StreamLogs(ServerContext* context,
                    const cuopt::remote::StreamLogsRequest* request,
                    ServerWriter<cuopt::remote::LogMessage>* writer) override
  {
    const std::string job_id   = request->job_id();
    int64_t from_byte          = request->from_byte();
    const std::string log_path = get_log_file_path(job_id);

    // Phase 1: Wait for the log file to appear on disk.
    // The worker may not have created it yet, so poll with a short sleep.
    // Every 2 s, verify the job still exists to avoid waiting forever on
    // a deleted/unknown job.
    int waited_ms = 0;
    while (!context->IsCancelled()) {
      struct stat st;
      if (stat(log_path.c_str(), &st) == 0) { break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      waited_ms += 50;
      if (waited_ms >= 2000) {
        std::string msg;
        JobStatus s = check_job_status(job_id, msg);
        if (s == JobStatus::NOT_FOUND) {
          if (config.verbose) {
            SERVER_LOG_DEBUG("[gRPC] StreamLogs job not found: %s", job_id.c_str());
          }
          return Status(grpc::StatusCode::NOT_FOUND, "Job not found: " + job_id);
        }
        if (s == JobStatus::COMPLETED || s == JobStatus::FAILED || s == JobStatus::CANCELLED) {
          cuopt::remote::LogMessage done;
          done.set_line("");
          done.set_byte_offset(from_byte);
          done.set_job_complete(true);
          writer->Write(done);
          return Status::OK;
        }
        waited_ms = 0;
      }
    }

    // Phase 2: Open the file and seek to the caller's resume point.
    std::ifstream in(log_path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
      cuopt::remote::LogMessage m;
      m.set_line("Failed to open log file");
      m.set_byte_offset(from_byte);
      m.set_job_complete(true);
      writer->Write(m);
      return Status::OK;
    }

    if (from_byte > 0) { in.seekg(from_byte, std::ios::beg); }

    int64_t current_offset = from_byte;
    std::string line;

    // Phase 3: Tail loop — read available lines, stream each one, then
    // poll for more.  Each LogMessage carries the byte offset of the *next*
    // unread byte so the client can resume from that point.
    while (!context->IsCancelled()) {
      std::streampos before = in.tellg();
      if (before >= 0) { current_offset = static_cast<int64_t>(before); }

      if (std::getline(in, line)) {
        std::streampos after = in.tellg();
        int64_t next_offset  = current_offset;
        if (after >= 0) {
          next_offset = static_cast<int64_t>(after);
        } else {
          // tellg() can return -1 after the last line when there is no
          // trailing newline; fall back to estimating from line length.
          next_offset = current_offset + static_cast<int64_t>(line.size());
        }

        cuopt::remote::LogMessage m;
        m.set_line(line);
        m.set_byte_offset(next_offset);
        m.set_job_complete(false);
        if (!writer->Write(m)) { break; }
        continue;
      }

      // Caught up to the current end of file — clear the EOF/fail bit
      // so the next getline attempt can see newly appended data.
      if (in.eof()) {
        in.clear();
      } else if (in.fail()) {
        in.clear();
      }

      // Check whether the job has finished.  If so, drain any final
      // bytes the solver may have flushed after our last read, then
      // send the job_complete sentinel and close the stream.
      std::string msg;
      JobStatus s = check_job_status(job_id, msg);
      if (s == JobStatus::COMPLETED || s == JobStatus::FAILED || s == JobStatus::CANCELLED) {
        std::streampos before2 = in.tellg();
        if (before2 >= 0) { current_offset = static_cast<int64_t>(before2); }
        if (std::getline(in, line)) {
          std::streampos after2 = in.tellg();
          int64_t next_offset2  = current_offset + static_cast<int64_t>(line.size());
          if (after2 >= 0) { next_offset2 = static_cast<int64_t>(after2); }
          cuopt::remote::LogMessage m;
          m.set_line(line);
          m.set_byte_offset(next_offset2);
          m.set_job_complete(false);
          writer->Write(m);
        }

        cuopt::remote::LogMessage done;
        done.set_line("");
        done.set_byte_offset(current_offset);
        done.set_job_complete(true);
        writer->Write(done);
        return Status::OK;
      }

      // Job still running but no new data yet — back off briefly before
      // retrying so we don't spin-wait on the file.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return Status::OK;
  }

  Status GetIncumbents(ServerContext* context,
                       const cuopt::remote::IncumbentRequest* request,
                       cuopt::remote::IncumbentResponse* response) override
  {
    (void)context;
    const std::string job_id = request->job_id();
    int64_t from_index       = request->from_index();
    int32_t max_count        = request->max_count();

    if (from_index < 0) { from_index = 0; }

    std::lock_guard<std::mutex> lock(tracker_mutex);
    auto it = job_tracker.find(job_id);
    if (it == job_tracker.end()) { return Status(StatusCode::NOT_FOUND, "Job not found"); }

    const auto& incumbents = it->second.incumbents;
    int64_t available      = static_cast<int64_t>(incumbents.size());
    if (from_index > available) { from_index = available; }

    int64_t count = available - from_index;
    if (max_count > 0 && count > max_count) { count = max_count; }

    for (int64_t i = 0; i < count; ++i) {
      const auto& inc = incumbents[static_cast<size_t>(from_index + i)];
      auto* out       = response->add_incumbents();
      out->set_index(from_index + i);
      out->set_objective(inc.objective);
      for (double v : inc.assignment) {
        out->add_assignment(v);
      }
      out->set_job_id(job_id);
    }

    // next_index is the resume cursor: the client passes it back as from_index
    // on the next call.  Must be from_index + count (the first unsent entry),
    // NOT available (total size), or the client skips entries when max_count
    // limits the batch.
    response->set_next_index(from_index + count);
    bool done =
      (it->second.status == JobStatus::COMPLETED || it->second.status == JobStatus::FAILED ||
       it->second.status == JobStatus::CANCELLED);
    response->set_job_complete(done);
    if (config.verbose) {
      SERVER_LOG_DEBUG("[gRPC] GetIncumbents job_id=%s from=%ld returned=%d next=%ld done=%d",
                       job_id.c_str(),
                       from_index,
                       response->incumbents_size(),
                       from_index + count,
                       done ? 1 : 0);
    }
    return Status::OK;
  }
};

// Provide access to the service implementation type from grpc_server_main.cpp.
// This avoids exposing the class definition in a header (it's only needed once in main).
std::unique_ptr<grpc::Service> create_cuopt_grpc_service()
{
  return std::make_unique<CuOptRemoteServiceImpl>();
}

#endif  // CUOPT_ENABLE_GRPC
