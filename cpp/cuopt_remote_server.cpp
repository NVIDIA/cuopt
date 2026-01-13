/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuopt_remote_server.cpp
 * @brief Remote solve server with sync and async support using pluggable serialization
 *
 * Features:
 * - Sync mode: Submit job with blocking=true, wait for result, return immediately
 * - Async mode: Submit job, get job_id, poll for status, retrieve result, delete
 * - Uses pluggable serialization (default: Protocol Buffers)
 * - Worker processes with shared memory job queues
 * - Real-time log streaming to client (sync mode only)
 *
 * Async workflow:
 *   1. Client sends SUBMIT_JOB request → Server returns job_id
 *   2. Client sends CHECK_STATUS request → Server returns job status
 *   3. Client sends GET_RESULT request → Server returns solution
 *   4. Client sends DELETE_RESULT request → Server cleans up job
 *
 * Sync workflow:
 *   1. Client sends SUBMIT_JOB with blocking=true → Server solves and returns result directly
 */

#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace cuopt::linear_programming;

// ============================================================================
// Shared Memory Structures (must match between main process and workers)
// ============================================================================

constexpr size_t MAX_JOBS    = 100;
constexpr size_t MAX_RESULTS = 100;

// Job queue entry - small fixed size, data stored in separate per-job shared memory or sent via
// pipe
struct JobQueueEntry {
  char job_id[64];
  uint32_t problem_type;          // 0 = LP, 1 = MIP
  uint64_t data_size;             // Size of problem data (uint64 for large problems)
  char shm_data_name[128];        // Name of per-job shared memory segment (shm mode only)
  std::atomic<bool> ready;        // Job is ready to be processed
  std::atomic<bool> claimed;      // Worker has claimed this job
  std::atomic<pid_t> worker_pid;  // PID of worker that claimed this job (0 if none)
  std::atomic<bool> cancelled;    // Job has been cancelled (worker should skip)
  // Pipe mode fields
  std::atomic<int> worker_index;  // Index of worker that claimed this job (-1 if none)
  std::atomic<bool> data_sent;    // Server has sent data to worker's pipe (pipe mode)
};

// Result queue entry - small fixed size, data stored in separate per-result shared memory or pipe
struct ResultQueueEntry {
  char job_id[64];
  uint32_t status;          // 0 = success, 1 = error, 2 = cancelled
  uint64_t data_size;       // Size of result data (uint64 for large results)
  char shm_data_name[128];  // Name of per-result shared memory segment (shm mode only)
  char error_message[1024];
  std::atomic<bool> ready;        // Result is ready
  std::atomic<bool> retrieved;    // Result has been retrieved
  std::atomic<int> worker_index;  // Index of worker that produced this result (pipe mode)
};

// Shared memory control block
struct SharedMemoryControl {
  std::atomic<bool> shutdown_requested;
  std::atomic<int> active_workers;
};

// ============================================================================
// Message types for streaming protocol
// ============================================================================

enum class MessageType : uint8_t {
  LOG_MESSAGE = 0,  // Log output from server
  SOLUTION    = 1,  // Final solution data
};

// Helper to send a framed message with type
static bool send_typed_message(int sockfd, MessageType type, const void* data, size_t size)
{
  uint8_t msg_type      = static_cast<uint8_t>(type);
  uint64_t payload_size = static_cast<uint64_t>(size);

  if (::write(sockfd, &msg_type, 1) != 1) return false;
  if (::write(sockfd, &payload_size, sizeof(payload_size)) != sizeof(payload_size)) return false;
  if (size > 0) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    size_t remaining   = size;
    while (remaining > 0) {
      ssize_t written = ::write(sockfd, ptr, remaining);
      if (written <= 0) return false;
      ptr += written;
      remaining -= written;
    }
  }
  return true;
}

// ============================================================================
// RAII stdout streamer for log streaming to client
// ============================================================================

class stdout_streamer_t {
 public:
  stdout_streamer_t(int client_fd, bool enabled)
    : client_fd_(client_fd), enabled_(enabled), running_(false), original_stdout_(-1)
  {
    if (!enabled_) return;

    fflush(stdout);

    if (pipe(pipe_fds_) < 0) {
      std::cerr << "[Server] Failed to create pipe for stdout streaming\n";
      enabled_ = false;
      return;
    }

    original_stdout_ = dup(STDOUT_FILENO);
    if (original_stdout_ < 0) {
      close(pipe_fds_[0]);
      close(pipe_fds_[1]);
      enabled_ = false;
      return;
    }

    if (dup2(pipe_fds_[1], STDOUT_FILENO) < 0) {
      close(original_stdout_);
      close(pipe_fds_[0]);
      close(pipe_fds_[1]);
      enabled_ = false;
      return;
    }

    close(pipe_fds_[1]);

    running_       = true;
    reader_thread_ = std::thread(&stdout_streamer_t::reader_loop, this);
  }

  ~stdout_streamer_t()
  {
    if (!enabled_) return;

    fflush(stdout);
    dup2(original_stdout_, STDOUT_FILENO);
    close(original_stdout_);

    running_ = false;
    close(pipe_fds_[0]);

    if (reader_thread_.joinable()) { reader_thread_.join(); }
  }

 private:
  void reader_loop()
  {
    char buffer[4096];
    while (running_) {
      ssize_t n = read(pipe_fds_[0], buffer, sizeof(buffer) - 1);
      if (n <= 0) break;

      buffer[n] = '\0';

      if (original_stdout_ >= 0) { write(original_stdout_, buffer, n); }
      send_typed_message(client_fd_, MessageType::LOG_MESSAGE, buffer, n);
    }
  }

  int client_fd_;
  bool enabled_;
  std::atomic<bool> running_;
  int original_stdout_;
  int pipe_fds_[2];
  std::thread reader_thread_;
};

// ============================================================================
// Job status tracking (main process only)
// ============================================================================

enum class JobStatus { QUEUED, PROCESSING, COMPLETED, FAILED, NOT_FOUND, CANCELLED };

struct JobInfo {
  std::string job_id;
  JobStatus status;
  std::chrono::steady_clock::time_point submit_time;
  std::vector<uint8_t> result_data;
  bool is_mip;
  std::string error_message;
  bool is_blocking;  // True if a client is waiting synchronously
};

// Per-job condition variable for synchronous waiting
struct JobWaiter {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<uint8_t> result_data;
  std::string error_message;
  bool success;
  bool ready;

  JobWaiter() : success(false), ready(false) {}
};

// ============================================================================
// Global state
// ============================================================================

std::atomic<bool> keep_running{true};
std::map<std::string, JobInfo> job_tracker;
std::mutex tracker_mutex;
std::condition_variable result_cv;  // Notified when results arrive

std::map<std::string, std::shared_ptr<JobWaiter>> waiting_threads;
std::mutex waiters_mutex;

// Shared memory
JobQueueEntry* job_queue       = nullptr;
ResultQueueEntry* result_queue = nullptr;
SharedMemoryControl* shm_ctrl  = nullptr;

// Worker PIDs
std::vector<pid_t> worker_pids;

// Server configuration
struct ServerConfig {
  int port         = 9090;
  int num_workers  = 1;
  bool verbose     = true;
  bool stream_logs = true;
  bool use_pipes   = true;  // Default to pipes (container-friendly), --use-shm to disable
};

ServerConfig config;

// Worker state for pipe-based IPC
struct WorkerPipes {
  int to_worker_fd;     // Server writes job data to this (pipe write end)
  int from_worker_fd;   // Server reads results from this (pipe read end)
  int worker_read_fd;   // Worker reads job data from this (inherited, closed in parent)
  int worker_write_fd;  // Worker writes results to this (inherited, closed in parent)
};

std::vector<WorkerPipes> worker_pipes;

// Pending job data for pipe mode (job_id -> serialized data)
std::mutex pending_data_mutex;
std::map<std::string, std::vector<uint8_t>> pending_job_data;

// Shared memory names
const char* SHM_JOB_QUEUE    = "/cuopt_job_queue";
const char* SHM_RESULT_QUEUE = "/cuopt_result_queue";
const char* SHM_CONTROL      = "/cuopt_control";

// ============================================================================
// Signal handling
// ============================================================================

void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\n[Server] Received shutdown signal\n";
    keep_running = false;
    if (shm_ctrl) { shm_ctrl->shutdown_requested = true; }
    result_cv.notify_all();
  }
}

// ============================================================================
// Utilities
// ============================================================================

std::string generate_job_id()
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uint64_t> dis;

  uint64_t id = dis(gen);
  char buf[32];
  snprintf(buf, sizeof(buf), "job_%016lx", id);
  return std::string(buf);
}

static bool write_all(int sockfd, const void* data, size_t size)
{
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t remaining   = size;
  while (remaining > 0) {
    ssize_t written = ::write(sockfd, ptr, remaining);
    if (written <= 0) return false;
    ptr += written;
    remaining -= written;
  }
  return true;
}

static bool read_all(int sockfd, void* data, size_t size)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    ssize_t nread = ::read(sockfd, ptr, remaining);
    if (nread <= 0) return false;
    ptr += nread;
    remaining -= nread;
  }
  return true;
}

static bool send_solution_message(int sockfd, const std::vector<uint8_t>& data)
{
  return send_typed_message(sockfd, MessageType::SOLUTION, data.data(), data.size());
}

static bool receive_request(int sockfd, std::vector<uint8_t>& data)
{
  uint64_t size;
  if (!read_all(sockfd, &size, sizeof(size))) return false;

  // Sanity check - reject requests larger than 16GB
  if (size > 16ULL * 1024 * 1024 * 1024) {
    std::cerr << "[Server] Request too large: " << size << " bytes\n";
    return false;
  }

  data.resize(size);
  if (!read_all(sockfd, data.data(), size)) return false;
  return true;
}

// ============================================================================
// Per-job Shared Memory Helpers (forward declarations)
// ============================================================================

static std::string create_job_shm(const std::string& job_id,
                                  const std::vector<uint8_t>& data,
                                  const char* prefix);
static bool read_job_shm(const char* shm_name, size_t data_size, std::vector<uint8_t>& data);
static std::string write_result_shm(const std::string& job_id, const std::vector<uint8_t>& data);
static void cleanup_job_shm(const char* shm_name);

// ============================================================================
// Shared Memory Management
// ============================================================================

bool init_shared_memory()
{
  // Create job queue shared memory
  int fd_jobs = shm_open(SHM_JOB_QUEUE, O_CREAT | O_RDWR, 0666);
  if (fd_jobs < 0) {
    std::cerr << "[Server] Failed to create job queue shared memory\n";
    return false;
  }
  size_t job_queue_size = sizeof(JobQueueEntry) * MAX_JOBS;
  if (ftruncate(fd_jobs, job_queue_size) < 0) {
    std::cerr << "[Server] Failed to size job queue shared memory\n";
    close(fd_jobs);
    return false;
  }
  job_queue = static_cast<JobQueueEntry*>(
    mmap(nullptr, job_queue_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_jobs, 0));
  close(fd_jobs);
  if (job_queue == MAP_FAILED) {
    std::cerr << "[Server] Failed to map job queue\n";
    return false;
  }

  // Initialize job queue entries
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    job_queue[i].ready        = false;
    job_queue[i].claimed      = false;
    job_queue[i].worker_pid   = 0;
    job_queue[i].cancelled    = false;
    job_queue[i].worker_index = -1;
    job_queue[i].data_sent    = false;
  }

  // Create result queue shared memory
  int fd_results = shm_open(SHM_RESULT_QUEUE, O_CREAT | O_RDWR, 0666);
  if (fd_results < 0) {
    std::cerr << "[Server] Failed to create result queue shared memory\n";
    return false;
  }
  size_t result_queue_size = sizeof(ResultQueueEntry) * MAX_RESULTS;
  if (ftruncate(fd_results, result_queue_size) < 0) {
    std::cerr << "[Server] Failed to size result queue shared memory\n";
    close(fd_results);
    return false;
  }
  result_queue = static_cast<ResultQueueEntry*>(
    mmap(nullptr, result_queue_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_results, 0));
  close(fd_results);
  if (result_queue == MAP_FAILED) {
    std::cerr << "[Server] Failed to map result queue\n";
    return false;
  }

  // Initialize result queue entries
  for (size_t i = 0; i < MAX_RESULTS; ++i) {
    result_queue[i].ready        = false;
    result_queue[i].retrieved    = false;
    result_queue[i].worker_index = -1;
  }

  // Create control shared memory
  int fd_ctrl = shm_open(SHM_CONTROL, O_CREAT | O_RDWR, 0666);
  if (fd_ctrl < 0) {
    std::cerr << "[Server] Failed to create control shared memory\n";
    return false;
  }
  if (ftruncate(fd_ctrl, sizeof(SharedMemoryControl)) < 0) {
    std::cerr << "[Server] Failed to size control shared memory\n";
    close(fd_ctrl);
    return false;
  }
  shm_ctrl = static_cast<SharedMemoryControl*>(
    mmap(nullptr, sizeof(SharedMemoryControl), PROT_READ | PROT_WRITE, MAP_SHARED, fd_ctrl, 0));
  close(fd_ctrl);
  if (shm_ctrl == MAP_FAILED) {
    std::cerr << "[Server] Failed to map control\n";
    return false;
  }

  shm_ctrl->shutdown_requested = false;
  shm_ctrl->active_workers     = 0;

  return true;
}

void cleanup_shared_memory()
{
  if (job_queue) {
    munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
    shm_unlink(SHM_JOB_QUEUE);
  }
  if (result_queue) {
    munmap(result_queue, sizeof(ResultQueueEntry) * MAX_RESULTS);
    shm_unlink(SHM_RESULT_QUEUE);
  }
  if (shm_ctrl) {
    munmap(shm_ctrl, sizeof(SharedMemoryControl));
    shm_unlink(SHM_CONTROL);
  }
}

// ============================================================================
// Forward declarations for log file management
// ============================================================================
std::string get_log_file_path(const std::string& job_id);
void ensure_log_dir_exists();
void delete_log_file(const std::string& job_id);

// ============================================================================
// Forward declarations for pipe I/O helpers
// ============================================================================
static bool write_to_pipe(int fd, const void* data, size_t size);
static bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms = 120000);
static bool send_job_data_pipe(int worker_idx, const std::vector<uint8_t>& data);
static bool recv_job_data_pipe(int fd, uint64_t expected_size, std::vector<uint8_t>& data);
static bool send_result_pipe(int fd, const std::vector<uint8_t>& data);
static bool recv_result_pipe(int worker_idx, uint64_t expected_size, std::vector<uint8_t>& data);

// ============================================================================
// Worker Process
// ============================================================================

void worker_process(int worker_id)
{
  std::cout << "[Worker " << worker_id << "] Started (PID: " << getpid() << ")\n";

  // Increment active worker count
  shm_ctrl->active_workers++;

  // NOTE: We create raft::handle_t AFTER stdout redirect (per-job) so that
  // CUDA logging uses the redirected output streams.

  // Get serializer
  auto serializer = get_serializer<int, double>();

  while (!shm_ctrl->shutdown_requested) {
    // Find a job to process
    int job_slot = -1;
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready && !job_queue[i].claimed) {
        // Try to claim this job atomically
        bool expected = false;
        if (job_queue[i].claimed.compare_exchange_strong(expected, true)) {
          job_queue[i].worker_pid   = getpid();   // Record our PID
          job_queue[i].worker_index = worker_id;  // Record worker index for pipe mode
          job_slot                  = i;
          break;
        }
      }
    }

    if (job_slot < 0) {
      // No job available, sleep briefly
      usleep(10000);  // 10ms
      continue;
    }

    // Process the job
    JobQueueEntry& job = job_queue[job_slot];
    std::string job_id(job.job_id);
    bool is_mip = (job.problem_type == 1);

    // Check if job was cancelled before we start processing
    if (job.cancelled) {
      std::cout << "[Worker " << worker_id << "] Job cancelled before processing: " << job_id
                << "\n";
      std::cout.flush();

      // Cleanup job input shm (shm mode only)
      if (!config.use_pipes) { cleanup_job_shm(job.shm_data_name); }

      // Store cancelled result in result queue
      for (size_t i = 0; i < MAX_RESULTS; ++i) {
        if (!result_queue[i].ready) {
          strncpy(result_queue[i].job_id, job_id.c_str(), sizeof(result_queue[i].job_id) - 1);
          result_queue[i].status           = 2;  // Cancelled status
          result_queue[i].data_size        = 0;
          result_queue[i].shm_data_name[0] = '\0';
          result_queue[i].worker_index     = worker_id;  // For pipe mode
          strncpy(result_queue[i].error_message,
                  "Job was cancelled",
                  sizeof(result_queue[i].error_message) - 1);
          result_queue[i].retrieved = false;
          result_queue[i].ready     = true;
          break;
        }
      }

      // Clear job slot (don't exit/restart worker)
      job.worker_pid   = 0;
      job.worker_index = -1;
      job.data_sent    = false;
      job.ready        = false;
      job.claimed      = false;
      job.cancelled    = false;
      continue;  // Go back to waiting for next job
    }

    std::cout << "[Worker " << worker_id << "] Processing job: " << job_id
              << " (type: " << (is_mip ? "MIP" : "LP") << ")\n";
    std::cout.flush();

    // Redirect stdout AND stderr to per-job log file for client log retrieval
    // (Solver may use either stream for output)
    std::string log_file = get_log_file_path(job_id);
    int saved_stdout     = dup(STDOUT_FILENO);
    int saved_stderr     = dup(STDERR_FILENO);
    int log_fd           = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log_fd >= 0) {
      // Flush C++ streams before changing fd
      std::cout.flush();
      std::cerr.flush();
      fflush(stdout);
      fflush(stderr);

      dup2(log_fd, STDOUT_FILENO);
      dup2(log_fd, STDERR_FILENO);
      close(log_fd);

      // Use unbuffered output for real-time log streaming
      setvbuf(stdout, NULL, _IONBF, 0);
      setvbuf(stderr, NULL, _IONBF, 0);

      // Test that redirection works
      printf("[Worker %d] Log file initialized: %s\n", worker_id, log_file.c_str());
      fflush(stdout);
    }

    // Create RAFT handle AFTER stdout redirect so CUDA sees the new streams
    const char* msg0 = "[Worker] Creating raft::handle_t...\n";
    write(STDOUT_FILENO, msg0, 36);
    fsync(STDOUT_FILENO);

    raft::handle_t handle;

    const char* msg01 = "[Worker] Handle created, starting solve...\n";
    write(STDOUT_FILENO, msg01, 44);
    fsync(STDOUT_FILENO);

    // Read problem data (pipe mode or shm mode)
    std::vector<uint8_t> request_data;
    bool read_success = false;

    if (config.use_pipes) {
      // Pipe mode: read from pipe (blocks until server writes data)
      // No need to wait for data_sent flag - pipe read naturally blocks
      int read_fd  = worker_pipes[worker_id].worker_read_fd;
      read_success = recv_job_data_pipe(read_fd, job.data_size, request_data);
      if (!read_success) {
        std::cerr << "[Worker " << worker_id << "] Failed to read job data from pipe\n";
      }
    } else {
      // SHM mode: read from shared memory
      read_success = read_job_shm(job.shm_data_name, job.data_size, request_data);
      if (!read_success) {
        std::cerr << "[Worker " << worker_id << "] Failed to read job data from shm\n";
      }
      // Cleanup job input shm now that we've read it
      cleanup_job_shm(job.shm_data_name);
    }

    if (!read_success) {
      // Store error result
      for (size_t i = 0; i < MAX_RESULTS; ++i) {
        if (!result_queue[i].ready) {
          strncpy(result_queue[i].job_id, job_id.c_str(), sizeof(result_queue[i].job_id) - 1);
          result_queue[i].status           = 1;  // Error status
          result_queue[i].data_size        = 0;
          result_queue[i].shm_data_name[0] = '\0';
          result_queue[i].worker_index     = worker_id;
          strncpy(result_queue[i].error_message,
                  "Failed to read job data",
                  sizeof(result_queue[i].error_message) - 1);
          result_queue[i].retrieved = false;
          result_queue[i].ready     = true;
          break;
        }
      }
      // Clear job slot
      job.worker_pid   = 0;
      job.worker_index = -1;
      job.data_sent    = false;
      job.ready        = false;
      job.claimed      = false;
      continue;
    }

    std::vector<uint8_t> result_data;
    std::string error_message;
    bool success = false;

    try {
      if (is_mip) {
        cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
        mip_solver_settings_t<int, double> settings;

        if (serializer->deserialize_mip_request(request_data, mps_data, settings)) {
          auto solution = solve_mip(&handle, mps_data, settings);
          solution.to_host(handle.get_stream());
          result_data = serializer->serialize_mip_solution(solution);
          success     = true;
        } else {
          error_message = "Failed to deserialize MIP request";
        }
      } else {
        cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
        pdlp_solver_settings_t<int, double> settings;

        if (serializer->deserialize_lp_request(request_data, mps_data, settings)) {
          const char* msg1 = "[Worker] Calling solve_lp via write()...\n";
          write(STDOUT_FILENO, msg1, strlen(msg1));
          fsync(STDOUT_FILENO);
          auto solution    = solve_lp(&handle, mps_data, settings);
          const char* msg2 = "[Worker] solve_lp done via write()\n";
          write(STDOUT_FILENO, msg2, strlen(msg2));
          fsync(STDOUT_FILENO);
          solution.to_host(handle.get_stream());
          result_data = serializer->serialize_lp_solution(solution);
          success     = true;
        } else {
          error_message = "Failed to deserialize LP request";
        }
      }
    } catch (const std::exception& e) {
      error_message = std::string("Exception: ") + e.what();
    }

    // Restore stdout and stderr to console
    fflush(stdout);
    fflush(stderr);
    dup2(saved_stdout, STDOUT_FILENO);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stdout);
    close(saved_stderr);


    // Store result (pipe mode: write to pipe, shm mode: write to shared memory)
    if (config.use_pipes) {
      // PIPE MODE: Set result_queue metadata FIRST, THEN write to pipe.
      // This avoids deadlock: the main thread's result_retrieval_thread
      // needs to see ready=true before it will read from the pipe,
      // but if we write to pipe first with a large result, we'll block
      // waiting for the reader that will never come.

      // Find a free result slot and populate metadata
      int result_slot = -1;
      for (size_t i = 0; i < MAX_RESULTS; ++i) {
        if (!result_queue[i].ready) {
          result_slot = i;
          ResultQueueEntry& result = result_queue[i];
          strncpy(result.job_id, job_id.c_str(), sizeof(result.job_id) - 1);
          result.status           = success ? 0 : 1;
          result.data_size        = success ? result_data.size() : 0;
          result.shm_data_name[0] = '\0';  // Not used in pipe mode
          result.worker_index     = worker_id;
          if (!success) {
            strncpy(result.error_message, error_message.c_str(), sizeof(result.error_message) - 1);
          }
          result.retrieved = false;
          // Set ready=true BEFORE writing to pipe so reader thread starts reading
          // This prevents deadlock with large results that exceed pipe buffer size
          result.ready = true;
          break;
        }
      }

      // Now write result data to pipe (reader thread should be ready to receive)
      if (success && !result_data.empty() && result_slot >= 0) {
        int write_fd       = worker_pipes[worker_id].worker_write_fd;
        bool write_success = send_result_pipe(write_fd, result_data);
        if (!write_success) {
          std::cerr << "[Worker " << worker_id << "] Failed to write result to pipe\n";
          // Mark as failed in result queue
          result_queue[result_slot].status = 1;
          strncpy(result_queue[result_slot].error_message,
                  "Failed to write result to pipe",
                  sizeof(result_queue[result_slot].error_message) - 1);
        }
      }
    } else {
      // SHM mode: store result in shared memory
      for (size_t i = 0; i < MAX_RESULTS; ++i) {
        if (!result_queue[i].ready) {
          ResultQueueEntry& result = result_queue[i];
          strncpy(result.job_id, job_id.c_str(), sizeof(result.job_id) - 1);
          result.status       = success ? 0 : 1;
          result.worker_index = worker_id;
          if (success && !result_data.empty()) {
            // Create per-result shared memory
            std::string shm_name = write_result_shm(job_id, result_data);
            if (shm_name.empty()) {
              // Failed to create shm - report error
              result.status           = 1;
              result.data_size        = 0;
              result.shm_data_name[0] = '\0';
              strncpy(result.error_message,
                      "Failed to create shared memory for result",
                      sizeof(result.error_message) - 1);
            } else {
              result.data_size = result_data.size();
              strncpy(result.shm_data_name, shm_name.c_str(), sizeof(result.shm_data_name) - 1);
            }
          } else if (!success) {
            strncpy(result.error_message, error_message.c_str(), sizeof(result.error_message) - 1);
            result.data_size        = 0;
            result.shm_data_name[0] = '\0';
          } else {
            result.data_size        = 0;
            result.shm_data_name[0] = '\0';
          }
          result.retrieved = false;
          result.ready     = true;  // Mark as ready last
          break;
        }
      }
    }

    // Clear job slot
    job.worker_pid   = 0;
    job.worker_index = -1;
    job.data_sent    = false;
    job.ready        = false;
    job.claimed      = false;
    job.cancelled    = false;

    std::cout << "[Worker " << worker_id << "] Completed job: " << job_id
              << " (success: " << success << ")\n";
  }

  shm_ctrl->active_workers--;
  std::cout << "[Worker " << worker_id << "] Stopped\n";
  _exit(0);
}

// Create pipes for a worker (pipe mode only)
bool create_worker_pipes(int worker_id)
{
  if (!config.use_pipes) return true;

  // Ensure worker_pipes has enough slots
  while (static_cast<int>(worker_pipes.size()) <= worker_id) {
    worker_pipes.push_back({-1, -1, -1, -1});
  }

  WorkerPipes& wp = worker_pipes[worker_id];

  // Create pipe for server -> worker data
  int input_pipe[2];
  if (pipe(input_pipe) < 0) {
    std::cerr << "[Server] Failed to create input pipe for worker " << worker_id << "\n";
    return false;
  }
  wp.worker_read_fd = input_pipe[0];  // Worker reads from this
  wp.to_worker_fd   = input_pipe[1];  // Server writes to this

  // Create pipe for worker -> server results
  int output_pipe[2];
  if (pipe(output_pipe) < 0) {
    std::cerr << "[Server] Failed to create output pipe for worker " << worker_id << "\n";
    close(input_pipe[0]);
    close(input_pipe[1]);
    return false;
  }
  wp.from_worker_fd  = output_pipe[0];  // Server reads from this
  wp.worker_write_fd = output_pipe[1];  // Worker writes to this

  return true;
}

// Close server-side pipe ends for a worker (called when restarting)
void close_worker_pipes_server(int worker_id)
{
  if (!config.use_pipes) return;
  if (worker_id < 0 || worker_id >= static_cast<int>(worker_pipes.size())) return;

  WorkerPipes& wp = worker_pipes[worker_id];
  if (wp.to_worker_fd >= 0) {
    close(wp.to_worker_fd);
    wp.to_worker_fd = -1;
  }
  if (wp.from_worker_fd >= 0) {
    close(wp.from_worker_fd);
    wp.from_worker_fd = -1;
  }
}

// Close worker-side pipe ends in parent after fork
void close_worker_pipes_child_ends(int worker_id)
{
  if (!config.use_pipes) return;
  if (worker_id < 0 || worker_id >= static_cast<int>(worker_pipes.size())) return;

  WorkerPipes& wp = worker_pipes[worker_id];
  if (wp.worker_read_fd >= 0) {
    close(wp.worker_read_fd);
    wp.worker_read_fd = -1;
  }
  if (wp.worker_write_fd >= 0) {
    close(wp.worker_write_fd);
    wp.worker_write_fd = -1;
  }
}

void spawn_workers()
{
  for (int i = 0; i < config.num_workers; ++i) {
    // Create pipes before forking (pipe mode)
    if (config.use_pipes && !create_worker_pipes(i)) {
      std::cerr << "[Server] Failed to create pipes for worker " << i << "\n";
      continue;
    }

    pid_t pid = fork();
    if (pid < 0) {
      std::cerr << "[Server] Failed to fork worker " << i << "\n";
      close_worker_pipes_server(i);
    } else if (pid == 0) {
      // Child process
      if (config.use_pipes) {
        // Close all other workers' pipe fds
        for (int j = 0; j < static_cast<int>(worker_pipes.size()); ++j) {
          if (j != i) {
            if (worker_pipes[j].worker_read_fd >= 0) close(worker_pipes[j].worker_read_fd);
            if (worker_pipes[j].worker_write_fd >= 0) close(worker_pipes[j].worker_write_fd);
            if (worker_pipes[j].to_worker_fd >= 0) close(worker_pipes[j].to_worker_fd);
            if (worker_pipes[j].from_worker_fd >= 0) close(worker_pipes[j].from_worker_fd);
          }
        }
        // Close server ends of our pipes
        close(worker_pipes[i].to_worker_fd);
        close(worker_pipes[i].from_worker_fd);
      }
      worker_process(i);
      _exit(0);  // Should not reach here
    } else {
      // Parent process
      worker_pids.push_back(pid);
      // Close worker ends of pipes (parent doesn't need them)
      close_worker_pipes_child_ends(i);
    }
  }
}

void wait_for_workers()
{
  for (pid_t pid : worker_pids) {
    int status;
    waitpid(pid, &status, 0);
  }
  worker_pids.clear();
}

// Spawn a single replacement worker and return its PID
pid_t spawn_single_worker(int worker_id)
{
  // Create new pipes for the replacement worker (pipe mode)
  if (config.use_pipes) {
    // Close old pipes first
    close_worker_pipes_server(worker_id);
    if (!create_worker_pipes(worker_id)) {
      std::cerr << "[Server] Failed to create pipes for replacement worker " << worker_id << "\n";
      return -1;
    }
  }

  pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "[Server] Failed to fork replacement worker " << worker_id << "\n";
    close_worker_pipes_server(worker_id);
    return -1;
  } else if (pid == 0) {
    // Child process
    if (config.use_pipes) {
      // Close all other workers' pipe fds
      for (int j = 0; j < static_cast<int>(worker_pipes.size()); ++j) {
        if (j != worker_id) {
          if (worker_pipes[j].worker_read_fd >= 0) close(worker_pipes[j].worker_read_fd);
          if (worker_pipes[j].worker_write_fd >= 0) close(worker_pipes[j].worker_write_fd);
          if (worker_pipes[j].to_worker_fd >= 0) close(worker_pipes[j].to_worker_fd);
          if (worker_pipes[j].from_worker_fd >= 0) close(worker_pipes[j].from_worker_fd);
        }
      }
      // Close server ends of our pipes
      close(worker_pipes[worker_id].to_worker_fd);
      close(worker_pipes[worker_id].from_worker_fd);
    }
    worker_process(worker_id);
    _exit(0);  // Should not reach here
  }

  // Parent: close worker ends of new pipes
  close_worker_pipes_child_ends(worker_id);
  return pid;
}

// Mark jobs being processed by a dead worker as failed (or cancelled if it was cancelled)
void mark_worker_jobs_failed(pid_t dead_worker_pid)
{
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready && job_queue[i].claimed && job_queue[i].worker_pid == dead_worker_pid) {
      std::string job_id(job_queue[i].job_id);
      bool was_cancelled = job_queue[i].cancelled;

      if (was_cancelled) {
        std::cerr << "[Server] Worker " << dead_worker_pid
                  << " killed for cancelled job: " << job_id << "\n";
      } else {
        std::cerr << "[Server] Worker " << dead_worker_pid
                  << " died while processing job: " << job_id << "\n";
      }

      // Cleanup job data
      if (config.use_pipes) {
        // Pipe mode: remove from pending data if not yet sent
        std::lock_guard<std::mutex> lock(pending_data_mutex);
        pending_job_data.erase(job_id);
      } else {
        // SHM mode: cleanup job input shm (worker may not have done it)
        cleanup_job_shm(job_queue[i].shm_data_name);
      }

      // Store result in result queue (cancelled or failed)
      for (size_t j = 0; j < MAX_RESULTS; ++j) {
        if (!result_queue[j].ready) {
          strncpy(result_queue[j].job_id, job_id.c_str(), sizeof(result_queue[j].job_id) - 1);
          result_queue[j].status           = was_cancelled ? 2 : 1;  // 2=cancelled, 1=error
          result_queue[j].data_size        = 0;
          result_queue[j].shm_data_name[0] = '\0';
          result_queue[j].worker_index     = -1;
          strncpy(result_queue[j].error_message,
                  was_cancelled ? "Job was cancelled" : "Worker process died unexpectedly",
                  sizeof(result_queue[j].error_message) - 1);
          result_queue[j].retrieved = false;
          result_queue[j].ready     = true;
          break;
        }
      }

      // Clear the job slot
      job_queue[i].worker_pid   = 0;
      job_queue[i].worker_index = -1;
      job_queue[i].data_sent    = false;
      job_queue[i].ready        = false;
      job_queue[i].claimed      = false;
      job_queue[i].cancelled    = false;

      // Update job tracker
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

// Worker monitor thread - detects dead workers and restarts them
void worker_monitor_thread()
{
  std::cout << "[Server] Worker monitor thread started\n";
  std::cout.flush();

  while (keep_running) {
    // Check all worker PIDs for dead workers
    for (size_t i = 0; i < worker_pids.size(); ++i) {
      pid_t pid = worker_pids[i];
      if (pid <= 0) continue;

      int status;
      pid_t result = waitpid(pid, &status, WNOHANG);

      if (result == pid) {
        // Worker has exited
        int exit_code  = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        bool signaled  = WIFSIGNALED(status);
        int signal_num = signaled ? WTERMSIG(status) : 0;

        if (signaled) {
          std::cerr << "[Server] Worker " << pid << " killed by signal " << signal_num << "\n";
          std::cerr.flush();
        } else if (exit_code != 0) {
          std::cerr << "[Server] Worker " << pid << " exited with code " << exit_code << "\n";
          std::cerr.flush();
        } else {
          // Clean exit during shutdown - don't restart
          if (shm_ctrl && shm_ctrl->shutdown_requested) {
            worker_pids[i] = 0;
            continue;
          }
          std::cerr << "[Server] Worker " << pid << " exited unexpectedly\n";
          std::cerr.flush();
        }

        // Mark any jobs this worker was processing as failed
        mark_worker_jobs_failed(pid);

        // Spawn replacement worker (unless shutting down)
        if (keep_running && shm_ctrl && !shm_ctrl->shutdown_requested) {
          pid_t new_pid = spawn_single_worker(static_cast<int>(i));
          if (new_pid > 0) {
            worker_pids[i] = new_pid;
            std::cout << "[Server] Restarted worker " << i << " with PID " << new_pid << "\n";
            std::cout.flush();
          } else {
            worker_pids[i] = 0;  // Failed to restart
          }
        } else {
          worker_pids[i] = 0;
        }
      }
    }

    // Check every 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "[Server] Worker monitor thread stopped\n";
  std::cout.flush();
}

// ============================================================================
// Result Retrieval Thread (main process)
// Also handles sending job data to workers in pipe mode
// ============================================================================

void result_retrieval_thread()
{
  std::cout << "[Server] Result retrieval thread started\n";

  while (keep_running) {
    bool found = false;

    // PIPE MODE: Check for jobs that need data sent to workers
    if (config.use_pipes) {
      for (size_t i = 0; i < MAX_JOBS; ++i) {
        if (job_queue[i].ready && job_queue[i].claimed && !job_queue[i].data_sent &&
            !job_queue[i].cancelled) {
          std::string job_id(job_queue[i].job_id);
          int worker_idx = job_queue[i].worker_index;

          if (worker_idx >= 0) {
            // Get pending job data
            std::vector<uint8_t> job_data;
            {
              std::lock_guard<std::mutex> lock(pending_data_mutex);
              auto it = pending_job_data.find(job_id);
              if (it != pending_job_data.end()) {
                job_data = std::move(it->second);
                pending_job_data.erase(it);
              }
            }

            if (!job_data.empty()) {
              // Send data to worker's pipe
              if (send_job_data_pipe(worker_idx, job_data)) {
                job_queue[i].data_sent = true;
                if (config.verbose) {
                  std::cout << "[Server] Sent " << job_data.size() << " bytes to worker "
                            << worker_idx << " for job " << job_id << "\n";
                }
              } else {
                std::cerr << "[Server] Failed to send job data to worker " << worker_idx << "\n";
                // Mark job as failed
                job_queue[i].cancelled = true;
              }
              found = true;
            }
          }
        }
      }
    }

    // Check for completed results
    for (size_t i = 0; i < MAX_RESULTS; ++i) {
      if (result_queue[i].ready && !result_queue[i].retrieved) {
        std::string job_id(result_queue[i].job_id);
        uint32_t result_status = result_queue[i].status;
        bool success           = (result_status == 0);
        bool cancelled         = (result_status == 2);
        int worker_idx         = result_queue[i].worker_index;

        std::vector<uint8_t> result_data;
        std::string error_message;

        if (success && result_queue[i].data_size > 0) {
          if (config.use_pipes) {
            // Pipe mode: read result from worker's output pipe
            if (!recv_result_pipe(worker_idx, result_queue[i].data_size, result_data)) {
              error_message = "Failed to read result data from pipe";
              success       = false;
            }
          } else {
            // SHM mode: read from shared memory
            if (!read_job_shm(
                  result_queue[i].shm_data_name, result_queue[i].data_size, result_data)) {
              error_message = "Failed to read result data from shared memory";
              success       = false;
            }
            // Cleanup result shm after reading
            cleanup_job_shm(result_queue[i].shm_data_name);
          }
        } else if (!success) {
          error_message = result_queue[i].error_message;
        }

        // Check if there's a blocking waiter
        {
          std::lock_guard<std::mutex> lock(waiters_mutex);
          auto wit = waiting_threads.find(job_id);
          if (wit != waiting_threads.end()) {
            // Wake up the waiting thread
            auto waiter           = wit->second;
            waiter->result_data   = std::move(result_data);
            waiter->error_message = error_message;
            waiter->success       = success;
            waiter->ready         = true;
            waiter->cv.notify_one();
          }
        }

        // Update job tracker
        {
          std::lock_guard<std::mutex> lock(tracker_mutex);
          auto it = job_tracker.find(job_id);
          if (it != job_tracker.end()) {
            if (success) {
              it->second.status      = JobStatus::COMPLETED;
              it->second.result_data = result_data;
            } else if (cancelled) {
              it->second.status        = JobStatus::CANCELLED;
              it->second.error_message = error_message;
            } else {
              it->second.status        = JobStatus::FAILED;
              it->second.error_message = error_message;
            }
          }
        }

        result_queue[i].retrieved    = true;
        result_queue[i].worker_index = -1;
        result_queue[i].ready        = false;  // Free slot
        found                        = true;
      }
    }

    if (!found) {
      usleep(10000);  // 10ms
    }

    result_cv.notify_all();
  }

  std::cout << "[Server] Result retrieval thread stopped\n";
}

// ============================================================================
// Async Request Handlers
// ============================================================================

// Create per-job shared memory segment and copy data into it
// Returns the shm name on success, empty string on failure
static std::string create_job_shm(const std::string& job_id,
                                  const std::vector<uint8_t>& data,
                                  const char* prefix)
{
  std::string shm_name = std::string("/cuopt_") + prefix + "_" + job_id;

  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd < 0) {
    std::cerr << "[Server] Failed to create shm " << shm_name << ": " << strerror(errno) << "\n";
    return "";
  }

  if (ftruncate(fd, data.size()) < 0) {
    std::cerr << "[Server] Failed to size shm " << shm_name << ": " << strerror(errno) << "\n";
    close(fd);
    shm_unlink(shm_name.c_str());
    return "";
  }

  void* ptr = mmap(nullptr, data.size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);

  if (ptr == MAP_FAILED) {
    std::cerr << "[Server] Failed to map shm " << shm_name << ": " << strerror(errno) << "\n";
    shm_unlink(shm_name.c_str());
    return "";
  }

  memcpy(ptr, data.data(), data.size());
  munmap(ptr, data.size());

  return shm_name;
}

// Read data from per-job shared memory segment
static bool read_job_shm(const char* shm_name, size_t data_size, std::vector<uint8_t>& data)
{
  int fd = shm_open(shm_name, O_RDONLY, 0666);
  if (fd < 0) {
    std::cerr << "[Worker] Failed to open shm " << shm_name << ": " << strerror(errno) << "\n";
    return false;
  }

  void* ptr = mmap(nullptr, data_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);

  if (ptr == MAP_FAILED) {
    std::cerr << "[Worker] Failed to map shm " << shm_name << ": " << strerror(errno) << "\n";
    return false;
  }

  data.resize(data_size);
  memcpy(data.data(), ptr, data_size);
  munmap(ptr, data_size);

  return true;
}

// Write data to per-result shared memory segment
static std::string write_result_shm(const std::string& job_id, const std::vector<uint8_t>& data)
{
  return create_job_shm(job_id, data, "result");
}

// Cleanup per-job shared memory segment
static void cleanup_job_shm(const char* shm_name)
{
  if (shm_name[0] != '\0') { shm_unlink(shm_name); }
}

// ============================================================================
// Pipe I/O Helpers
// ============================================================================

// Write all data to a pipe (handles partial writes)
static bool write_to_pipe(int fd, const void* data, size_t size)
{
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t remaining   = size;
  while (remaining > 0) {
    ssize_t written = ::write(fd, ptr, remaining);
    if (written <= 0) {
      if (errno == EINTR) continue;
      return false;
    }
    ptr += written;
    remaining -= written;
  }
  return true;
}

// Read all data from a pipe (handles partial reads) with timeout
// timeout_ms: milliseconds to wait for data (default 120000 = 2 minutes)
static bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    // Use poll() to wait for data with timeout
    struct pollfd pfd;
    pfd.fd     = fd;
    pfd.events = POLLIN;

    int poll_result = poll(&pfd, 1, timeout_ms);
    if (poll_result < 0) {
      if (errno == EINTR) continue;
      std::cerr << "[Server] poll() failed on pipe: " << strerror(errno) << "\n";
      return false;
    }
    if (poll_result == 0) {
      std::cerr << "[Server] Timeout waiting for pipe data (waited " << timeout_ms << "ms)\n";
      return false;
    }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      std::cerr << "[Server] Pipe error/hangup detected\n";
      return false;
    }

    ssize_t nread = ::read(fd, ptr, remaining);
    if (nread <= 0) {
      if (errno == EINTR) continue;
      if (nread == 0) {
        std::cerr << "[Server] Pipe EOF (writer closed)\n";
      }
      return false;
    }
    ptr += nread;
    remaining -= nread;
  }
  return true;
}

// Send job data to worker via pipe (length-prefixed)
static bool send_job_data_pipe(int worker_idx, const std::vector<uint8_t>& data)
{
  if (worker_idx < 0 || worker_idx >= static_cast<int>(worker_pipes.size())) { return false; }
  int fd = worker_pipes[worker_idx].to_worker_fd;
  if (fd < 0) return false;

  // Send size first
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  // Send data
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

// Receive job data from pipe (length-prefixed) - called by worker
static bool recv_job_data_pipe(int fd, uint64_t expected_size, std::vector<uint8_t>& data)
{
  // Read size
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size != expected_size) {
    std::cerr << "[Worker] Size mismatch: expected " << expected_size << ", got " << size << "\n";
    return false;
  }
  // Read data
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

// Send result data to server via pipe (length-prefixed) - called by worker
static bool send_result_pipe(int fd, const std::vector<uint8_t>& data)
{
  // Send size first
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  // Send data
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

// Receive result data from worker via pipe (length-prefixed)
static bool recv_result_pipe(int worker_idx, uint64_t expected_size, std::vector<uint8_t>& data)
{
  if (worker_idx < 0 || worker_idx >= static_cast<int>(worker_pipes.size())) { return false; }
  int fd = worker_pipes[worker_idx].from_worker_fd;
  if (fd < 0) return false;

  // Read size
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size != expected_size) {
    std::cerr << "[Server] Result size mismatch: expected " << expected_size << ", got " << size
              << "\n";
    return false;
  }
  // Read data
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

// Submit a job asynchronously (returns job_id)
std::pair<bool, std::string> submit_job_async(const std::vector<uint8_t>& request_data, bool is_mip)
{
  std::string job_id = generate_job_id();

  std::string shm_name;
  if (config.use_pipes) {
    // Pipe mode: store data in pending map (will be sent when worker claims job)
    {
      std::lock_guard<std::mutex> lock(pending_data_mutex);
      pending_job_data[job_id] = request_data;
    }
  } else {
    // SHM mode: create per-job shared memory for problem data
    shm_name = create_job_shm(job_id, request_data, "job");
    if (shm_name.empty()) { return {false, "Failed to create shared memory for job data"}; }
  }

  // Find free job slot
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (!job_queue[i].ready && !job_queue[i].claimed) {
      strncpy(job_queue[i].job_id, job_id.c_str(), sizeof(job_queue[i].job_id) - 1);
      job_queue[i].problem_type = is_mip ? 1 : 0;
      job_queue[i].data_size    = request_data.size();
      if (!config.use_pipes) {
        strncpy(
          job_queue[i].shm_data_name, shm_name.c_str(), sizeof(job_queue[i].shm_data_name) - 1);
      } else {
        job_queue[i].shm_data_name[0] = '\0';
      }
      job_queue[i].worker_pid   = 0;
      job_queue[i].worker_index = -1;
      job_queue[i].data_sent    = false;
      job_queue[i].claimed      = false;
      job_queue[i].cancelled    = false;
      job_queue[i].ready        = true;  // Mark as ready last

      // Track job
      {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        JobInfo info;
        info.job_id         = job_id;
        info.status         = JobStatus::QUEUED;
        info.submit_time    = std::chrono::steady_clock::now();
        info.is_mip         = is_mip;
        info.is_blocking    = false;
        job_tracker[job_id] = info;
      }

      if (config.verbose) { std::cout << "[Server] Job submitted (async): " << job_id << "\n"; }

      return {true, job_id};
    }
  }

  // No free slot - cleanup
  if (config.use_pipes) {
    std::lock_guard<std::mutex> lock(pending_data_mutex);
    pending_job_data.erase(job_id);
  } else {
    shm_unlink(shm_name.c_str());
  }
  return {false, "Job queue full"};
}

// Check job status
JobStatus check_job_status(const std::string& job_id, std::string& message)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) {
    message = "Job ID not found";
    return JobStatus::NOT_FOUND;
  }

  // If status is QUEUED, check if the job has been claimed by a worker
  // (which means it's now PROCESSING)
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

// Check if a job is MIP (vs LP)
bool get_job_is_mip(const std::string& job_id)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);
  if (it == job_tracker.end()) {
    return false;  // Default to LP if not found
  }
  return it->second.is_mip;
}

// Get job result
bool get_job_result(const std::string& job_id,
                    std::vector<uint8_t>& result_data,
                    std::string& error_message)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) {
    error_message = "Job ID not found";
    return false;
  }

  if (it->second.status == JobStatus::COMPLETED) {
    result_data = it->second.result_data;
    return true;
  } else if (it->second.status == JobStatus::FAILED) {
    error_message = it->second.error_message;
    return false;
  } else {
    error_message = "Job not completed yet";
    return false;
  }
}

// Wait for job to complete (blocking)
// This uses condition variables - the thread will sleep until the job is done
bool wait_for_result(const std::string& job_id,
                     std::vector<uint8_t>& result_data,
                     std::string& error_message)
{
  // First check if job already completed
  {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    auto it = job_tracker.find(job_id);

    if (it == job_tracker.end()) {
      error_message = "Job ID not found";
      return false;
    }

    // If already in terminal state, return immediately
    if (it->second.status == JobStatus::COMPLETED) {
      result_data = it->second.result_data;
      return true;
    } else if (it->second.status == JobStatus::FAILED) {
      error_message = it->second.error_message;
      return false;
    } else if (it->second.status == JobStatus::CANCELLED) {
      error_message = "Job was cancelled";
      return false;
    }
  }

  // Job is still running - create a waiter and wait on condition variable
  auto waiter = std::make_shared<JobWaiter>();

  {
    std::lock_guard<std::mutex> lock(waiters_mutex);
    waiting_threads[job_id] = waiter;
  }

  if (config.verbose) {
    std::cout << "[Server] WAIT_FOR_RESULT: waiting for job " << job_id << "\n";
  }

  // Wait on the condition variable - this thread will sleep until signaled
  {
    std::unique_lock<std::mutex> lock(waiter->mutex);
    waiter->cv.wait(lock, [&waiter] { return waiter->ready; });
  }

  // Remove from waiting_threads
  {
    std::lock_guard<std::mutex> lock(waiters_mutex);
    waiting_threads.erase(job_id);
  }

  if (config.verbose) {
    std::cout << "[Server] WAIT_FOR_RESULT: job " << job_id
              << " completed, success=" << waiter->success << "\n";
  }

  if (waiter->success) {
    result_data = std::move(waiter->result_data);
    return true;
  } else {
    error_message = waiter->error_message;
    return false;
  }
}

// ============================================================================
// Log File Management
// ============================================================================

// Directory for per-job log files
const std::string LOG_DIR = "/tmp/cuopt_logs";

// Get the log file path for a given job_id
std::string get_log_file_path(const std::string& job_id) { return LOG_DIR + "/log_" + job_id; }

// Ensure log directory exists
void ensure_log_dir_exists()
{
  struct stat st;
  if (stat(LOG_DIR.c_str(), &st) != 0) { mkdir(LOG_DIR.c_str(), 0755); }
}

// Delete log file for a job
void delete_log_file(const std::string& job_id)
{
  std::string log_file = get_log_file_path(job_id);
  unlink(log_file.c_str());  // Ignore errors if file doesn't exist
}

// Delete job
bool delete_job(const std::string& job_id)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) { return false; }

  job_tracker.erase(it);

  // Also delete the log file
  delete_log_file(job_id);

  if (config.verbose) { std::cout << "[Server] Job deleted: " << job_id << "\n"; }

  return true;
}

// Cancel job - returns: 0=success, 1=job_not_found, 2=already_completed, 3=already_cancelled
// Also returns the job's status after cancel attempt via job_status_out
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

  // Can't cancel completed jobs
  if (current_status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  // Already cancelled
  if (current_status == JobStatus::CANCELLED) {
    message        = "Job already cancelled";
    job_status_out = JobStatus::CANCELLED;
    return 3;
  }

  // Can't cancel failed jobs
  if (current_status == JobStatus::FAILED) {
    message        = "Cannot cancel failed job";
    job_status_out = JobStatus::FAILED;
    return 2;
  }

  // Find the job in the shared memory queue
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready && strcmp(job_queue[i].job_id, job_id.c_str()) == 0) {
      // Check if job is being processed by a worker
      pid_t worker_pid = job_queue[i].worker_pid;

      if (worker_pid > 0 && job_queue[i].claimed) {
        // Job is being processed - kill the worker
        if (config.verbose) {
          std::cout << "[Server] Cancelling running job " << job_id << " (killing worker "
                    << worker_pid << ")\n";
        }
        kill(worker_pid, SIGKILL);
        // The worker monitor thread will detect the dead worker, restart it,
        // and mark_worker_jobs_failed will be called. But we want CANCELLED not FAILED.
        // So we mark it as cancelled here first.
        job_queue[i].cancelled = true;
      } else {
        // Job is queued but not yet claimed - mark as cancelled
        if (config.verbose) { std::cout << "[Server] Cancelling queued job " << job_id << "\n"; }
        job_queue[i].cancelled = true;
      }

      // Update job tracker
      it->second.status        = JobStatus::CANCELLED;
      it->second.error_message = "Job cancelled by user";
      job_status_out           = JobStatus::CANCELLED;
      message                  = "Job cancelled successfully";

      // Delete the log file for this job
      delete_log_file(job_id);

      // Wake up any threads waiting for this job
      {
        std::lock_guard<std::mutex> wlock(waiters_mutex);
        auto wit = waiting_threads.find(job_id);
        if (wit != waiting_threads.end()) {
          auto waiter           = wit->second;
          waiter->error_message = "Job cancelled by user";
          waiter->success       = false;
          waiter->ready         = true;
          waiter->cv.notify_one();
        }
      }

      return 0;
    }
  }

  // Job not found in queue (might have already finished processing)
  // Re-check status since we hold the lock
  if (it->second.status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  // Job must be in flight or in an edge case - mark as cancelled anyway
  it->second.status        = JobStatus::CANCELLED;
  it->second.error_message = "Job cancelled by user";
  job_status_out           = JobStatus::CANCELLED;
  message                  = "Job cancelled";

  // Wake up any threads waiting for this job
  {
    std::lock_guard<std::mutex> wlock(waiters_mutex);
    auto wit = waiting_threads.find(job_id);
    if (wit != waiting_threads.end()) {
      auto waiter           = wit->second;
      waiter->error_message = "Job cancelled by user";
      waiter->success       = false;
      waiter->ready         = true;
      waiter->cv.notify_one();
    }
  }

  return 0;
}

// ============================================================================
// Sync Mode Handler (with log streaming)
// ============================================================================

/**
 * @brief Handle synchronous (blocking) solve requests directly.
 *
 * For sync mode, we solve directly in the main thread instead of using worker
 * processes. This allows stdout log streaming to work correctly since the
 * stdout_streamer_t captures output from the same process.
 */
void handle_sync_solve(int client_fd,
                       const std::vector<uint8_t>& request_data,
                       bool is_mip,
                       bool stream_logs)
{
  std::string job_id = generate_job_id();

  if (config.verbose) {
    std::cout << "[Server] Sync solve request, job_id: " << job_id
              << " (streaming: " << (stream_logs ? "yes" : "no") << ")\n";
  }

  auto serializer = get_serializer<int, double>();

  // Create RAFT handle for GPU operations
  raft::handle_t handle;

  std::vector<uint8_t> result_data;
  std::string error_message;
  bool success = false;

  // Use RAII stdout streamer - captures stdout and streams to client while
  // also echoing to server console. Destructor restores original stdout.
  {
    stdout_streamer_t streamer(client_fd, stream_logs);

    try {
      if (is_mip) {
        cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
        mip_solver_settings_t<int, double> settings;

        if (serializer->deserialize_mip_request(request_data, mps_data, settings)) {
          auto solution = solve_mip(&handle, mps_data, settings);
          solution.to_host(handle.get_stream());
          result_data = serializer->serialize_mip_solution(solution);
          success     = true;
        } else {
          error_message = "Failed to deserialize MIP request";
        }
      } else {
        cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
        pdlp_solver_settings_t<int, double> settings;

        if (serializer->deserialize_lp_request(request_data, mps_data, settings)) {
          auto solution = solve_lp(&handle, mps_data, settings);
          solution.to_host(handle.get_stream());
          result_data = serializer->serialize_lp_solution(solution);
          success     = true;
        } else {
          error_message = "Failed to deserialize LP request";
        }
      }
    } catch (const std::exception& e) {
      error_message = std::string("Exception: ") + e.what();
    }
  }  // streamer destructor restores stdout

  if (config.verbose) {
    std::cout << "[Server] Sync solve completed: " << job_id << " (success: " << success << ")\n";
  }

  // Send result to client
  if (success) {
    std::cout << "[Server] Sending solution message, size = " << result_data.size() << " bytes\n";
    send_solution_message(client_fd, result_data);
  } else {
    std::cerr << "[Server] Sync solve failed: " << error_message << "\n";
    // Send empty solution to indicate failure
    std::vector<uint8_t> empty;
    send_solution_message(client_fd, empty);
  }

  close(client_fd);
}

// ============================================================================
// Client Connection Handler
// ============================================================================

void handle_client(int client_fd, bool stream_logs)
{
  auto serializer = get_serializer<int, double>();

  // Receive request
  std::vector<uint8_t> request_data;
  if (!receive_request(client_fd, request_data)) {
    std::cerr << "[Server] Failed to receive request\n";
    close(client_fd);
    return;
  }

  if (config.verbose) {
    std::cout << "[Server] Received request, size: " << request_data.size() << " bytes\n";
  }

  // Determine if this is an async protocol request
  bool is_async_request = serializer->is_async_request(request_data);

  if (is_async_request) {
    // Parse async request type and handle accordingly
    auto request_type = serializer->get_async_request_type(request_data);

    if (request_type == 0) {  // SUBMIT_JOB
      bool blocking = serializer->is_blocking_request(request_data);
      bool is_mip   = serializer->is_mip_request(request_data);

      // Extract the actual problem data from the async request
      std::vector<uint8_t> problem_data = serializer->extract_problem_data(request_data);

      // UNIFIED ARCHITECTURE: All jobs go through the queue
      // Submit job to queue (same for both sync and async)
      auto [submit_ok, job_id_or_error] = submit_job_async(problem_data, is_mip);

      if (!submit_ok) {
        // Submission failed
        auto response = serializer->serialize_submit_response(false, job_id_or_error);
        uint64_t size = response.size();
        write_all(client_fd, &size, sizeof(size));
        write_all(client_fd, response.data(), response.size());
      } else if (blocking) {
        // BLOCKING MODE: Wait for result using condition variable (no polling)
        // This unifies sync/async - job goes through queue but we wait here
        std::string job_id = job_id_or_error;

        if (config.verbose) {
          std::cout << "[Server] Blocking request, job_id: " << job_id
                    << " (waiting for completion)\n";
        }

        std::vector<uint8_t> result_data;
        std::string error_message;

        // Block on condition variable until job completes
        bool success = wait_for_result(job_id, result_data, error_message);

        // NOTE: We do NOT auto-delete here. The client should call DELETE_RESULT
        // after consuming all logs. This allows the pattern:
        //   1. Submit job (blocking=true or async + WAIT_FOR_RESULT)
        //   2. Retrieve logs (GET_LOGS) - can be done in parallel thread
        //   3. Delete job (DELETE_RESULT) when done with logs

        // Return result response (same format as GET_RESULT)
        bool job_is_mip = is_mip;  // Use the is_mip from the submit request
        auto response =
          serializer->serialize_result_response(success, result_data, error_message, job_is_mip);
        uint64_t size = response.size();
        write_all(client_fd, &size, sizeof(size));
        write_all(client_fd, response.data(), response.size());

        if (config.verbose) {
          std::cout << "[Server] Blocking request completed: " << job_id << ", success=" << success
                    << "\n";
        }
      } else {
        // ASYNC MODE: Return job_id immediately
        auto response = serializer->serialize_submit_response(true, job_id_or_error);
        uint64_t size = response.size();
        write_all(client_fd, &size, sizeof(size));
        write_all(client_fd, response.data(), response.size());
      }
    } else if (request_type == 1) {  // CHECK_STATUS
      std::string job_id = serializer->get_job_id(request_data);
      std::string message;
      JobStatus status = check_job_status(job_id, message);

      int status_code = 0;
      switch (status) {
        case JobStatus::QUEUED: status_code = 0; break;
        case JobStatus::PROCESSING: status_code = 1; break;
        case JobStatus::COMPLETED: status_code = 2; break;
        case JobStatus::FAILED: status_code = 3; break;
        case JobStatus::NOT_FOUND: status_code = 4; break;
        case JobStatus::CANCELLED: status_code = 5; break;
      }

      auto response = serializer->serialize_status_response(status_code, message);

      uint64_t size = response.size();
      write_all(client_fd, &size, sizeof(size));
      write_all(client_fd, response.data(), response.size());
    } else if (request_type == 2) {  // GET_RESULT
      std::string job_id = serializer->get_job_id(request_data);
      std::vector<uint8_t> result_data;
      std::string error_message;

      bool success    = get_job_result(job_id, result_data, error_message);
      bool job_is_mip = get_job_is_mip(job_id);
      auto response =
        serializer->serialize_result_response(success, result_data, error_message, job_is_mip);

      uint64_t size = response.size();
      write_all(client_fd, &size, sizeof(size));
      write_all(client_fd, response.data(), response.size());
    } else if (request_type == 3) {  // DELETE_RESULT
      std::string job_id = serializer->get_job_id(request_data);
      bool success       = delete_job(job_id);

      auto response = serializer->serialize_delete_response(success);

      uint64_t size = response.size();
      write_all(client_fd, &size, sizeof(size));
      write_all(client_fd, response.data(), response.size());
    } else if (request_type == 4) {  // GET_LOGS
      std::string job_id = serializer->get_job_id(request_data);
      int64_t frombyte   = serializer->get_frombyte(request_data);

      std::vector<std::string> log_lines;
      int64_t nbytes  = 0;
      bool job_exists = false;

      // Read logs from file
      std::string log_file = get_log_file_path(job_id);
      std::ifstream ifs(log_file);
      if (ifs.is_open()) {
        job_exists = true;
        ifs.seekg(frombyte);
        std::string line;
        while (std::getline(ifs, line)) {
          log_lines.push_back(line);
        }
        nbytes = ifs.tellg();
        if (nbytes < 0) {
          // tellg returns -1 at EOF, get actual file size
          ifs.clear();
          ifs.seekg(0, std::ios::end);
          nbytes = ifs.tellg();
        }
        ifs.close();
      } else {
        // Check if job exists but log file doesn't (not started yet)
        std::lock_guard<std::mutex> lock(tracker_mutex);
        job_exists = (job_tracker.find(job_id) != job_tracker.end());
      }

      auto response = serializer->serialize_logs_response(job_id, log_lines, nbytes, job_exists);

      uint64_t size = response.size();
      write_all(client_fd, &size, sizeof(size));
      write_all(client_fd, response.data(), response.size());

      if (config.verbose) {
        std::cout << "[Server] GET_LOGS: job=" << job_id << ", frombyte=" << frombyte
                  << ", lines=" << log_lines.size() << ", nbytes=" << nbytes << "\n";
      }
    } else if (request_type == 5) {  // CANCEL_JOB
      std::string job_id = serializer->get_job_id(request_data);

      JobStatus job_status_out;
      std::string message;
      int result = cancel_job(job_id, job_status_out, message);

      // Convert JobStatus to status code
      int status_code = 0;
      switch (job_status_out) {
        case JobStatus::QUEUED: status_code = 0; break;
        case JobStatus::PROCESSING: status_code = 1; break;
        case JobStatus::COMPLETED: status_code = 2; break;
        case JobStatus::FAILED: status_code = 3; break;
        case JobStatus::NOT_FOUND: status_code = 4; break;
        case JobStatus::CANCELLED: status_code = 5; break;
      }

      bool success  = (result == 0);
      auto response = serializer->serialize_cancel_response(success, message, status_code);

      uint64_t size = response.size();
      write_all(client_fd, &size, sizeof(size));
      write_all(client_fd, response.data(), response.size());

      if (config.verbose) {
        std::cout << "[Server] CANCEL_JOB: job=" << job_id << ", success=" << success
                  << ", message=" << message << "\n";
      }
    } else if (request_type == 6) {  // WAIT_FOR_RESULT
      std::string job_id = serializer->get_job_id(request_data);

      if (config.verbose) {
        std::cout << "[Server] WAIT_FOR_RESULT: job=" << job_id << " (blocking until complete)\n";
      }

      std::vector<uint8_t> result_data;
      std::string error_message;

      // This will block until the job completes (uses condition variable, no polling)
      bool success = wait_for_result(job_id, result_data, error_message);

      // Send result response (same format as GET_RESULT)
      bool job_is_mip = get_job_is_mip(job_id);
      auto response =
        serializer->serialize_result_response(success, result_data, error_message, job_is_mip);

      uint64_t size = response.size();
      write_all(client_fd, &size, sizeof(size));
      write_all(client_fd, response.data(), response.size());

      if (config.verbose) {
        std::cout << "[Server] WAIT_FOR_RESULT: job=" << job_id << " completed, success=" << success
                  << "\n";
      }
    }

    close(client_fd);
  } else {
    // Legacy/simple request format - treat as sync LP/MIP request
    bool is_mip = serializer->is_mip_request(request_data);
    handle_sync_solve(client_fd, request_data, is_mip, stream_logs);
  }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog)
{
  std::cout << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  -p PORT    Port to listen on (default: 9090)\n"
            << "  -w NUM     Number of worker processes (default: 1)\n"
            << "  -q         Quiet mode (less verbose output)\n"
            << "  --no-stream  Disable real-time log streaming to clients\n"
            << "  --use-shm    Use POSIX shared memory for IPC (default: pipes)\n"
            << "               Pipes are container-friendly; shm may be faster but\n"
            << "               requires /dev/shm with sufficient size\n"
            << "  -h         Show this help\n"
            << "\n"
            << "Environment Variables (client-side):\n"
            << "  CUOPT_REMOTE_USE_SYNC=1  Force sync mode (default is async)\n";
}

int main(int argc, char** argv)
{
  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
      config.port = std::stoi(argv[++i]);
    } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
      config.num_workers = std::stoi(argv[++i]);
    } else if (strcmp(argv[i], "-q") == 0) {
      config.verbose = false;
    } else if (strcmp(argv[i], "--no-stream") == 0) {
      config.stream_logs = false;
    } else if (strcmp(argv[i], "--use-shm") == 0) {
      config.use_pipes = false;  // Use shared memory instead of pipes
    } else if (strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  // Set up signal handlers
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGPIPE, SIG_IGN);  // Ignore SIGPIPE (broken pipe) - happens when writing to closed pipes

  // IMPORTANT: Clear remote solve environment variables to prevent infinite recursion
  unsetenv("CUOPT_REMOTE_HOST");
  unsetenv("CUOPT_REMOTE_PORT");

  // Ensure log directory exists for per-job log files
  ensure_log_dir_exists();

  std::cout << "=== cuOpt Remote Solve Server (Async) ===\n";
  std::cout << "Port: " << config.port << "\n";
  std::cout << "Workers: " << config.num_workers << " (processes)\n";
  std::cout << "Log streaming: " << (config.stream_logs ? "enabled" : "disabled") << "\n";
  std::cout << "IPC mode: " << (config.use_pipes ? "pipes (container-friendly)" : "shared memory")
            << "\n";
  std::cout << "\n";
  std::cout << "Async API:\n";
  std::cout << "  SUBMIT_JOB      - Submit a job, get job_id\n";
  std::cout << "  CHECK_STATUS    - Check job status\n";
  std::cout << "  GET_RESULT      - Retrieve completed result\n";
  std::cout << "  DELETE_RESULT   - Delete job from server\n";
  std::cout << "  GET_LOGS        - Retrieve log output\n";
  std::cout << "  CANCEL_JOB      - Cancel a queued or running job\n";
  std::cout << "  WAIT_FOR_RESULT - Block until job completes (no polling)\n";
  std::cout << "\n";

  // Initialize shared memory
  if (!init_shared_memory()) {
    std::cerr << "[Server] Failed to initialize shared memory\n";
    return 1;
  }

  // Spawn worker processes
  spawn_workers();

  // Start result retrieval thread
  std::thread result_thread(result_retrieval_thread);

  // Start worker monitor thread (detects dead workers and restarts them)
  std::thread monitor_thread(worker_monitor_thread);

  // Create server socket
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    std::cerr << "[Server] Failed to create socket\n";
    cleanup_shared_memory();
    return 1;
  }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port        = htons(config.port);

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "[Server] Failed to bind to port " << config.port << "\n";
    close(server_fd);
    cleanup_shared_memory();
    return 1;
  }

  if (listen(server_fd, 10) < 0) {
    std::cerr << "[Server] Failed to listen\n";
    close(server_fd);
    cleanup_shared_memory();
    return 1;
  }

  std::cout << "[Server] Listening on port " << config.port << "\n";

  // Flush stdout before accept loop
  std::cout.flush();

  // Accept connections
  while (keep_running) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(server_fd, &read_fds);

    struct timeval tv;
    tv.tv_sec  = 1;
    tv.tv_usec = 0;

    int ready = select(server_fd + 1, &read_fds, nullptr, nullptr, &tv);
    if (ready < 0) {
      if (errno == EINTR) continue;
      std::cerr << "[Server] Select error\n";
      break;
    }
    if (ready == 0) continue;

    int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      std::cerr << "[Server] Accept error\n";
      continue;
    }

    if (config.verbose) {
      char client_ip[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
      std::cout << "[Server] Connection from " << client_ip << "\n";
    }

    // Handle client in separate thread
    std::thread([client_fd]() { handle_client(client_fd, config.stream_logs); }).detach();
  }

  // Shutdown
  std::cout << "[Server] Shutting down...\n";
  close(server_fd);

  // Signal workers to stop
  if (shm_ctrl) { shm_ctrl->shutdown_requested = true; }

  // Wait for result retrieval thread
  result_cv.notify_all();
  if (result_thread.joinable()) { result_thread.join(); }

  // Wait for worker monitor thread
  if (monitor_thread.joinable()) { monitor_thread.join(); }

  // Wait for workers
  wait_for_workers();

  // Cleanup
  cleanup_shared_memory();

  std::cout << "[Server] Stopped\n";
  return 0;
}
