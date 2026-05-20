/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_incumbent_proto.hpp"
#include "grpc_pipe_serialization.hpp"
#include "grpc_server_types.hpp"

// ---------------------------------------------------------------------------
// Data-transfer structs used to pass results between decomposed functions.
// ---------------------------------------------------------------------------

struct DeserializedJob {
  cpu_optimization_problem_t<int, double> problem;
  pdlp_solver_settings_t<int, double> lp_settings;
  mip_solver_settings_t<int, double> mip_settings;
  bool enable_incumbents = true;
  bool success           = false;
};

struct SolveResult {
  cuopt::remote::ChunkedResultHeader header;
  std::map<int32_t, std::vector<uint8_t>> arrays;
  std::string error_message;
  bool success = false;
};

// ---------------------------------------------------------------------------
// Solver callback that forwards each new MIP incumbent to the server thread
// via a pipe.  A fresh instance is created per solve (as a unique_ptr scoped
// to run_mip_solve) and registered with mip_settings.set_mip_callback().
// The solver calls get_solution() every time it finds a better integer-feasible
// solution; we serialize the objective + variable assignment into a protobuf
// and push it down the incumbent pipe FD.  The server thread reads the other
// end to serve GetIncumbents RPCs.
// ---------------------------------------------------------------------------

class IncumbentPipeCallback : public cuopt::internals::get_solution_callback_t {
 public:
  IncumbentPipeCallback(std::string job_id, int fd, size_t num_vars, bool is_float)
    : job_id_(std::move(job_id)), fd_(fd)
  {
    n_variables = num_vars;
    isFloat     = is_float;
  }

  // Called by the MIP solver each time a new incumbent is found.
  // data/objective_value arrive as raw void* whose actual type depends on
  // isFloat; we normalize everything to double before serializing.
  void get_solution(void* data,
                    void* objective_value,
                    void* /*solution_bound*/,
                    void* /*user_data*/) override
  {
    if (fd_ < 0 || n_variables == 0) { return; }

    double objective = 0.0;
    std::vector<double> assignment;
    assignment.resize(n_variables);

    if (isFloat) {
      const float* float_data = static_cast<const float*>(data);
      for (size_t i = 0; i < n_variables; ++i) {
        assignment[i] = static_cast<double>(float_data[i]);
      }
      objective = static_cast<double>(*static_cast<const float*>(objective_value));
    } else {
      const double* double_data = static_cast<const double*>(data);
      std::copy(double_data, double_data + n_variables, assignment.begin());
      objective = *static_cast<const double*>(objective_value);
    }

    auto buffer = build_incumbent_proto(job_id_, objective, assignment);
    if (!send_incumbent_pipe(fd_, buffer)) {
      SERVER_LOG_ERROR("[Worker] Incumbent pipe write failed for job %s, disabling further sends",
                       job_id_.c_str());
      fd_ = -1;
      return;
    }
  }

 private:
  std::string job_id_;
  int fd_;
};

// ---------------------------------------------------------------------------
// Small utility helpers
// ---------------------------------------------------------------------------

// Reset every field in a job slot so it can be reused by the next submission.
static void reset_job_slot(JobQueueEntry& job)
{
  job.worker_pid   = 0;
  job.worker_index = -1;
  job.data_sent    = false;
  job.is_chunked   = false;
  job.ready        = false;
  job.claimed      = false;
  job.cancelled    = false;
}

// Log pipe throughput when config.verbose is enabled.
static void log_pipe_throughput(const char* phase,
                                int64_t total_bytes,
                                std::chrono::steady_clock::time_point t0)
{
  auto pipe_us =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
      .count();
  double pipe_sec = pipe_us / 1e6;
  double pipe_mb  = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  double pipe_mbs = (pipe_sec > 0.0) ? (pipe_mb / pipe_sec) : 0.0;
  SERVER_LOG_INFO("[THROUGHPUT] phase=%s bytes=%ld elapsed_ms=%.1f throughput_mb_s=%.1f",
                  phase,
                  total_bytes,
                  pipe_us / 1000.0,
                  pipe_mbs);
}

// Copy a device vector of T to a newly allocated host std::vector<T>.
template <typename T>
static std::vector<T> device_to_host(const auto& device_vec)
{
  std::vector<T> host(device_vec.size());
  cudaError_t err = cudaMemcpy(
    host.data(), device_vec.data(), device_vec.size() * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy device-to-host failed: ") +
                             cudaGetErrorString(err));
  }
  return host;
}

// Write a result entry with no payload (error, cancellation, etc.) into the
// first free slot in the shared-memory result_queue.
//
// Lock-free protocol for cross-process writes (workers are forked):
//   1. Skip slots where ready==true (still being consumed by the reader).
//   2. CAS claimed false→true to get exclusive write access.  Another
//      writer (different worker process) that races on the same slot will
//      see the CAS fail and move to the next slot.
//   3. Re-check ready after claiming, in case the reader set ready=true
//      between step 1 and step 2.
//   4. Write all non-atomic fields, then publish with ready=true (release)
//      so the reader sees a consistent entry.
//   5. Clear claimed so the slot can be recycled after the reader is done.
//
// The same protocol is used by publish_result() and the crash-recovery
// path in grpc_worker_infra.cpp.
static void store_simple_result(const std::string& job_id,
                                int worker_id,
                                ResultStatus status,
                                const char* error_message)
{
  for (size_t i = 0; i < MAX_RESULTS; ++i) {
    if (result_queue[i].ready.load(std::memory_order_acquire)) continue;
    bool expected = false;
    if (!result_queue[i].claimed.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
      continue;
    }
    if (result_queue[i].ready.load(std::memory_order_acquire)) {
      result_queue[i].claimed.store(false, std::memory_order_release);
      continue;
    }
    copy_cstr(result_queue[i].job_id, job_id);
    result_queue[i].status    = status;
    result_queue[i].data_size = 0;
    result_queue[i].worker_index.store(worker_id, std::memory_order_relaxed);
    copy_cstr(result_queue[i].error_message, error_message);
    result_queue[i].error_message[sizeof(result_queue[i].error_message) - 1] = '\0';
    result_queue[i].retrieved.store(false, std::memory_order_relaxed);
    result_queue[i].ready.store(true, std::memory_order_release);
    result_queue[i].claimed.store(false, std::memory_order_release);
    break;
  }
}

// ---------------------------------------------------------------------------
// Stage functions called from the worker_process main loop
// ---------------------------------------------------------------------------

// Atomically claim the first ready-but-unclaimed job slot, stamping it with
// this worker's PID and index.  Returns the slot index, or -1 if none found.
static int claim_job_slot(int worker_id)
{
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready && !job_queue[i].claimed) {
      bool expected = false;
      if (job_queue[i].claimed.compare_exchange_strong(expected, true)) {
        job_queue[i].worker_pid   = getpid();
        job_queue[i].worker_index = worker_id;
        return static_cast<int>(i);
      }
    }
  }
  return -1;
}

// Deserialize the problem from the worker's pipe.  Handles both chunked and
// unary IPC formats.  Returns a DeserializedJob with success=false on error.
static DeserializedJob read_problem_from_pipe(int worker_id, const JobQueueEntry& job)
{
  DeserializedJob dj;

  int read_fd         = worker_pipes[worker_id].worker_read_fd;
  bool is_chunked_job = job.is_chunked.load();

  auto pipe_recv_t0 = std::chrono::steady_clock::now();

  if (is_chunked_job) {
    // Chunked path: the server wrote a ChunkedProblemHeader followed by
    // a set of raw typed arrays (constraint matrix, bounds, etc.).
    // This avoids a single giant protobuf allocation for large problems.
    cuopt::remote::ChunkedProblemHeader chunked_header;
    std::map<int32_t, std::vector<uint8_t>> arrays;
    if (!read_chunked_request_from_pipe(read_fd, chunked_header, arrays)) { return dj; }

    if (config.verbose) {
      int64_t total_bytes = 0;
      for (const auto& [fid, data] : arrays) {
        total_bytes += data.size();
      }
      log_pipe_throughput("pipe_job_recv", total_bytes, pipe_recv_t0);
      SERVER_LOG_INFO(
        "[Worker] IPC path: CHUNKED (%zu arrays, %ld bytes)", arrays.size(), total_bytes);
    }
    if (chunked_header.has_lp_settings()) {
      map_proto_to_pdlp_settings(chunked_header.lp_settings(), dj.lp_settings);
    }
    if (chunked_header.has_mip_settings()) {
      map_proto_to_mip_settings(chunked_header.mip_settings(), dj.mip_settings);
    }
    dj.enable_incumbents = chunked_header.enable_incumbents();
    map_chunked_arrays_to_problem(chunked_header, arrays, dj.problem);
  } else {
    // Unary path: the entire SubmitJobRequest was serialized as a single
    // protobuf blob.  Simpler but copies more memory for large problems.
    std::vector<uint8_t> request_data;
    if (!recv_job_data_pipe(read_fd, job.data_size, request_data)) { return dj; }

    if (config.verbose) {
      log_pipe_throughput("pipe_job_recv", static_cast<int64_t>(request_data.size()), pipe_recv_t0);
    }
    cuopt::remote::SubmitJobRequest submit_request;
    if (!submit_request.ParseFromArray(request_data.data(),
                                       static_cast<int>(request_data.size())) ||
        (!submit_request.has_lp_request() && !submit_request.has_mip_request())) {
      return dj;
    }
    if (submit_request.has_lp_request()) {
      const auto& req = submit_request.lp_request();
      SERVER_LOG_INFO("[Worker] IPC path: UNARY LP (%zu bytes)", request_data.size());
      map_proto_to_problem(req.problem(), dj.problem);
      map_proto_to_pdlp_settings(req.settings(), dj.lp_settings);
    } else {
      const auto& req = submit_request.mip_request();
      SERVER_LOG_INFO("[Worker] IPC path: UNARY MIP (%zu bytes)", request_data.size());
      map_proto_to_problem(req.problem(), dj.problem);
      map_proto_to_mip_settings(req.settings(), dj.mip_settings);
      dj.enable_incumbents = req.has_enable_incumbents() ? req.enable_incumbents() : true;
    }
  }

  dj.success = true;
  return dj;
}

// Run the MIP solver on the GPU and serialize the solution into chunked format.
// The incumbent callback is created and scoped here so it lives exactly as
// long as the solve.  Exceptions are caught and returned as error messages.
static SolveResult run_mip_solve(DeserializedJob& dj,
                                 raft::handle_t& handle,
                                 const std::string& log_file,
                                 const std::string& job_id,
                                 int worker_id)
{
  SolveResult sr;
  try {
    dj.mip_settings.log_file       = log_file;
    dj.mip_settings.log_to_console = config.log_to_console;

    // Create a per-solve incumbent callback wired to this worker's
    // incumbent pipe.  Destroyed automatically when sr is returned.
    std::unique_ptr<IncumbentPipeCallback> incumbent_cb;
    if (dj.enable_incumbents) {
      incumbent_cb =
        std::make_unique<IncumbentPipeCallback>(job_id,
                                                worker_pipes[worker_id].worker_incumbent_write_fd,
                                                dj.problem.get_n_variables(),
                                                false);
      dj.mip_settings.set_mip_callback(incumbent_cb.get());
      SERVER_LOG_INFO("[Worker] Registered incumbent callback for job_id=%s n_vars=%d",
                      job_id.c_str(),
                      dj.problem.get_n_variables());
    }

    SERVER_LOG_INFO("[Worker] Converting CPU problem to GPU problem...");
    auto gpu_problem = dj.problem.to_optimization_problem(&handle);

    SERVER_LOG_INFO("[Worker] Calling solve_mip...");
    auto gpu_solution = solve_mip(*gpu_problem, dj.mip_settings);
    SERVER_LOG_INFO("[Worker] solve_mip done");

    SERVER_LOG_INFO("[Worker] Converting solution to CPU format...");

    auto host_solution = device_to_host<double>(gpu_solution.get_solution());

    cpu_mip_solution_t<int, double> cpu_solution(std::move(host_solution),
                                                 gpu_solution.get_termination_status(),
                                                 gpu_solution.get_objective_value(),
                                                 gpu_solution.get_mip_gap(),
                                                 gpu_solution.get_solution_bound(),
                                                 gpu_solution.get_total_solve_time(),
                                                 gpu_solution.get_presolve_time(),
                                                 gpu_solution.get_max_constraint_violation(),
                                                 gpu_solution.get_max_int_violation(),
                                                 gpu_solution.get_max_variable_bound_violation(),
                                                 gpu_solution.get_num_nodes(),
                                                 gpu_solution.get_num_simplex_iterations());

    populate_chunked_result_header_mip(cpu_solution, &sr.header);
    sr.arrays = collect_mip_solution_arrays(cpu_solution);
    SERVER_LOG_INFO("[Worker] Result path: MIP solution -> %zu array(s)", sr.arrays.size());
    sr.success = true;
  } catch (const cuopt::logic_error& e) {
    sr.error_message = format_cuopt_error(e);
  } catch (const std::exception& e) {
    sr.error_message = std::string("RuntimeError: ") + e.what();
  }
  return sr;
}

// Run the LP solver on the GPU and serialize the solution into chunked format.
// No incumbent callback (LP solvers don't produce intermediate solutions).
// Exceptions are caught and returned as error messages.
static SolveResult run_lp_solve(DeserializedJob& dj,
                                raft::handle_t& handle,
                                const std::string& log_file)
{
  SolveResult sr;
  try {
    dj.lp_settings.log_file       = log_file;
    dj.lp_settings.log_to_console = config.log_to_console;

    SERVER_LOG_INFO("[Worker] Converting CPU problem to GPU problem...");
    auto gpu_problem = dj.problem.to_optimization_problem(&handle);

    SERVER_LOG_INFO("[Worker] Calling solve_lp...");
    auto gpu_solution = solve_lp(*gpu_problem, dj.lp_settings);
    SERVER_LOG_INFO("[Worker] solve_lp done");

    SERVER_LOG_INFO("[Worker] Converting solution to CPU format...");

    auto host_primal       = device_to_host<double>(gpu_solution.get_primal_solution());
    auto host_dual         = device_to_host<double>(gpu_solution.get_dual_solution());
    auto host_reduced_cost = device_to_host<double>(gpu_solution.get_reduced_cost());

    auto term_info = gpu_solution.get_additional_termination_information();

    // Warm-start data lets clients resume an interrupted LP solve from
    // where it left off without starting over.
    auto cpu_ws =
      convert_to_cpu_warmstart(gpu_solution.get_pdlp_warm_start_data(), handle.get_stream());

    cpu_lp_solution_t<int, double> cpu_solution(std::move(host_primal),
                                                std::move(host_dual),
                                                std::move(host_reduced_cost),
                                                gpu_solution.get_termination_status(),
                                                gpu_solution.get_objective_value(),
                                                gpu_solution.get_dual_objective_value(),
                                                term_info.solve_time,
                                                term_info.l2_primal_residual,
                                                term_info.l2_dual_residual,
                                                term_info.gap,
                                                term_info.number_of_steps_taken,
                                                term_info.solved_by,
                                                std::move(cpu_ws));

    populate_chunked_result_header_lp(cpu_solution, &sr.header);
    sr.arrays = collect_lp_solution_arrays(cpu_solution);
    SERVER_LOG_INFO("[Worker] Result path: LP solution -> %zu array(s)", sr.arrays.size());
    sr.success = true;
  } catch (const cuopt::logic_error& e) {
    sr.error_message = format_cuopt_error(e);
  } catch (const std::exception& e) {
    sr.error_message = std::string("RuntimeError: ") + e.what();
  }
  return sr;
}

// Publish a solve result: claim a slot in the shared-memory result_queue
// (metadata) and, for successful solves, stream the full solution payload
// through the worker's result pipe for the server thread to read.
static void publish_result(const SolveResult& sr, const std::string& job_id, int worker_id)
{
  int64_t result_total_bytes = 0;
  if (sr.success) {
    for (const auto& [fid, data] : sr.arrays) {
      result_total_bytes += data.size();
    }
  }

  // Same CAS protocol as store_simple_result (see comment there).
  int result_slot = -1;
  for (size_t i = 0; i < MAX_RESULTS; ++i) {
    if (result_queue[i].ready.load(std::memory_order_acquire)) continue;
    bool expected = false;
    if (!result_queue[i].claimed.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
      continue;
    }
    if (result_queue[i].ready.load(std::memory_order_acquire)) {
      result_queue[i].claimed.store(false, std::memory_order_release);
      continue;
    }
    result_slot              = static_cast<int>(i);
    ResultQueueEntry& result = result_queue[i];
    copy_cstr(result.job_id, job_id);
    result.status    = sr.success ? RESULT_SUCCESS : RESULT_ERROR;
    result.data_size = sr.success ? std::max<uint64_t>(result_total_bytes, 1) : 0;
    result.worker_index.store(worker_id, std::memory_order_relaxed);
    if (!sr.success) { copy_cstr(result.error_message, sr.error_message); }
    result.retrieved.store(false, std::memory_order_relaxed);
    result.ready.store(true, std::memory_order_release);
    result.claimed.store(false, std::memory_order_release);
    if (config.verbose) {
      SERVER_LOG_DEBUG(
        "[Worker %d] Enqueued result metadata for job %s in result_slot=%d status=%d data_size=%lu",
        worker_id,
        job_id.c_str(),
        result_slot,
        static_cast<int>(result.status),
        result.data_size);
    }
    break;
  }

  // Stream the full solution payload through the worker's result pipe.
  // The server thread reads the other end when the client calls
  // GetResult / DownloadChunk.
  if (sr.success && result_slot >= 0) {
    int write_fd = worker_pipes[worker_id].worker_write_fd;
    if (config.verbose) {
      SERVER_LOG_DEBUG("[Worker %d] Streaming result (%zu arrays, %ld bytes) to pipe for job %s",
                       worker_id,
                       sr.arrays.size(),
                       result_total_bytes,
                       job_id.c_str());
    }
    auto pipe_result_t0 = std::chrono::steady_clock::now();
    bool write_success  = write_result_to_pipe(write_fd, sr.header, sr.arrays);
    if (write_success && config.verbose) {
      log_pipe_throughput("pipe_result_send", result_total_bytes, pipe_result_t0);
    }
    if (!write_success) {
      SERVER_LOG_ERROR("[Worker %d] Failed to write result to pipe", worker_id);
      result_queue[result_slot].status = RESULT_ERROR;
      copy_cstr(result_queue[result_slot].error_message, "Failed to write result to pipe");
    } else if (config.verbose) {
      SERVER_LOG_DEBUG(
        "[Worker %d] Finished writing result payload for job %s", worker_id, job_id.c_str());
    }
  } else if (config.verbose) {
    SERVER_LOG_DEBUG(
      "[Worker %d] No result payload write needed for job %s (success=%d, result_slot=%d, "
      "payload_bytes=%ld)",
      worker_id,
      job_id.c_str(),
      static_cast<int>(sr.success),
      result_slot,
      result_total_bytes);
  }
}

// ---------------------------------------------------------------------------
// Main worker loop — pure policy.  All implementation detail is in the
// stage functions above.
// ---------------------------------------------------------------------------

void worker_process(int worker_id)
{
  SERVER_LOG_INFO("[Worker %d] Started (PID: %d)", worker_id, getpid());

  shm_ctrl->active_workers++;

  while (!shm_ctrl->shutdown_requested) {
    int job_slot = claim_job_slot(worker_id);
    if (job_slot < 0) {
      usleep(10000);
      continue;
    }

    JobQueueEntry& job = job_queue[job_slot];
    std::string job_id(job.job_id);
    uint32_t problem_category = job.problem_category;

    if (job.cancelled) {
      SERVER_LOG_INFO("[Worker %d] Job cancelled before processing: %s", worker_id, job_id.c_str());
      store_simple_result(job_id, worker_id, RESULT_CANCELLED, "Job was cancelled");
      reset_job_slot(job);
      continue;
    }

    SERVER_LOG_INFO("[Worker %d] Processing job: %s (type: %s)",
                    worker_id,
                    job_id.c_str(),
                    problem_category == cuopt::remote::MIP ? "MIP" : "LP");

    auto deserialized = read_problem_from_pipe(worker_id, job);
    if (!deserialized.success) {
      SERVER_LOG_ERROR("[Worker %d] Failed to read job data from pipe", worker_id);
      store_simple_result(job_id, worker_id, RESULT_ERROR, "Failed to read job data");
      reset_job_slot(job);
      continue;
    }

    SERVER_LOG_INFO("[Worker] Problem reconstructed: %d constraints, %d variables, %d nonzeros",
                    deserialized.problem.get_n_constraints(),
                    deserialized.problem.get_n_variables(),
                    deserialized.problem.get_nnz());

    std::string log_file = get_log_file_path(job_id);
    raft::handle_t handle;

    SolveResult result = (problem_category == cuopt::remote::MIP)
                           ? run_mip_solve(deserialized, handle, log_file, job_id, worker_id)
                           : run_lp_solve(deserialized, handle, log_file);

    publish_result(result, job_id, worker_id);
    reset_job_slot(job);

    SERVER_LOG_INFO("[Worker %d] Completed job: %s (success: %d)",
                    worker_id,
                    job_id.c_str(),
                    static_cast<int>(result.success));
  }

  shm_ctrl->active_workers--;
  SERVER_LOG_INFO("[Worker %d] Stopped", worker_id);
  // _exit() instead of exit() to avoid running atexit handlers or flushing
  // parent-inherited stdio buffers a second time in the forked child.
  _exit(0);
}

#endif  // CUOPT_ENABLE_GRPC
