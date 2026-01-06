/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuopt_remote_server.cpp
 * @brief Remote solve server with sync and async support using pluggable serialization
 *
 * Features:
 * - Sync mode: Submit job, wait for result, return immediately
 * - Async mode: Submit job, get job_id, poll for status, retrieve result
 * - Uses pluggable serialization (default: Protocol Buffers)
 * - Threaded request handling
 * - Real-time log streaming to client
 */

#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace cuopt::linear_programming;

// Message types for streaming protocol
enum class MessageType : uint8_t {
  LOG_MESSAGE = 0,  // Log output from server
  SOLUTION    = 1,  // Final solution data
};

// Helper to send a framed message with type
static bool send_typed_message(int sockfd, MessageType type, const void* data, size_t size)
{
  // Message format: [type:1][size:4][payload:size]
  uint8_t msg_type      = static_cast<uint8_t>(type);
  uint32_t payload_size = static_cast<uint32_t>(size);

  // Write type
  if (::write(sockfd, &msg_type, 1) != 1) return false;
  // Write size
  if (::write(sockfd, &payload_size, 4) != 4) return false;
  // Write payload
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

/**
 * @brief RAII class to redirect stdout to a pipe and stream output to client
 *
 * This captures all stdout output from the solver and sends it to the client
 * in real-time while also echoing to the original stdout (server console).
 */
class stdout_streamer_t {
 public:
  stdout_streamer_t(int client_fd, bool enabled)
    : client_fd_(client_fd), enabled_(enabled), running_(false), original_stdout_(-1)
  {
    if (!enabled_) return;

    // Flush any buffered stdout to prevent old content from being captured
    fflush(stdout);

    // Create pipe
    if (pipe(pipe_fds_) < 0) {
      std::cerr << "[Server] Failed to create pipe for stdout streaming\n";
      enabled_ = false;
      return;
    }

    // Save original stdout
    original_stdout_ = dup(STDOUT_FILENO);
    if (original_stdout_ < 0) {
      close(pipe_fds_[0]);
      close(pipe_fds_[1]);
      enabled_ = false;
      return;
    }

    // Redirect stdout to pipe
    if (dup2(pipe_fds_[1], STDOUT_FILENO) < 0) {
      close(original_stdout_);
      close(pipe_fds_[0]);
      close(pipe_fds_[1]);
      enabled_ = false;
      return;
    }

    // Close write end of pipe (stdout now writes to it)
    close(pipe_fds_[1]);

    // Start reader thread
    running_       = true;
    reader_thread_ = std::thread(&stdout_streamer_t::reader_loop, this);
  }

  ~stdout_streamer_t()
  {
    if (!enabled_) return;

    // Flush stdout to ensure all output is in the pipe
    fflush(stdout);

    // Restore original stdout
    dup2(original_stdout_, STDOUT_FILENO);
    close(original_stdout_);

    // Signal reader to stop and close pipe read end
    running_ = false;
    close(pipe_fds_[0]);

    // Wait for reader thread
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

      // Echo to original stdout (server console)
      if (original_stdout_ >= 0) { write(original_stdout_, buffer, n); }

      // Send to client
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

// Job status
enum class JobStatus { QUEUED, PROCESSING, COMPLETED, FAILED, NOT_FOUND };

// Job info for tracking
struct JobInfo {
  std::string job_id;
  JobStatus status;
  std::chrono::steady_clock::time_point submit_time;
  std::vector<uint8_t> request_data;  // Stored request
  std::vector<uint8_t> result_data;   // Stored result
  bool is_mip;                        // true for MIP, false for LP
  std::string error_message;
};

// Global state
std::atomic<bool> keep_running{true};
std::map<std::string, JobInfo> job_tracker;
std::mutex tracker_mutex;
std::condition_variable job_cv;

// Server configuration
struct ServerConfig {
  int port         = 9090;
  int num_workers  = 1;
  bool verbose     = true;
  bool stream_logs = true;  // Enable real-time log streaming to clients
};

ServerConfig config;

void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\n[Server] Received shutdown signal\n";
    keep_running = false;
    job_cv.notify_all();
  }
}

// Generate unique job ID
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

// Socket helpers
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

static bool send_response(int sockfd, const std::vector<uint8_t>& data)
{
  // Legacy response format (for non-streaming clients)
  uint32_t size = static_cast<uint32_t>(data.size());
  if (!write_all(sockfd, &size, sizeof(size))) return false;
  if (!write_all(sockfd, data.data(), data.size())) return false;
  return true;
}

// Send solution using streaming protocol
static bool send_solution_message(int sockfd, const std::vector<uint8_t>& data)
{
  return send_typed_message(sockfd, MessageType::SOLUTION, data.data(), data.size());
}

static bool receive_request(int sockfd, std::vector<uint8_t>& data)
{
  uint32_t size;
  if (!read_all(sockfd, &size, sizeof(size))) return false;

  // Sanity check
  if (size > 100 * 1024 * 1024) {  // Max 100MB
    std::cerr << "[Server] Request too large: " << size << " bytes\n";
    return false;
  }

  data.resize(size);
  if (!read_all(sockfd, data.data(), size)) return false;
  return true;
}

// Worker thread - processes jobs from the queue
void worker_thread(int worker_id)
{
  std::cout << "[Worker " << worker_id << "] Started\n";

  // Create RAFT handle for GPU operations
  raft::handle_t handle;

  // Get serializer
  auto serializer = get_serializer<int, double>();

  while (keep_running) {
    std::string job_id;
    std::vector<uint8_t> request_data;
    bool is_mip = false;

    // Find a queued job
    {
      std::unique_lock<std::mutex> lock(tracker_mutex);
      job_cv.wait(lock, []() {
        if (!keep_running) return true;
        for (const auto& [id, info] : job_tracker) {
          if (info.status == JobStatus::QUEUED) return true;
        }
        return false;
      });

      if (!keep_running) break;

      // Find and claim a job
      for (auto& [id, info] : job_tracker) {
        if (info.status == JobStatus::QUEUED) {
          info.status  = JobStatus::PROCESSING;
          job_id       = id;
          request_data = info.request_data;
          is_mip       = info.is_mip;
          break;
        }
      }
    }

    if (job_id.empty()) continue;

    if (config.verbose) {
      std::cout << "[Worker " << worker_id << "] Processing job: " << job_id
                << " (type: " << (is_mip ? "MIP" : "LP") << ")\n";
    }

    std::vector<uint8_t> result_data;
    std::string error_message;
    bool success = false;

    try {
      if (is_mip) {
        // Deserialize MIP request
        cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
        mip_solver_settings_t<int, double> settings;

        if (serializer->deserialize_mip_request(request_data, mps_data, settings)) {
          // Solve using the data model directly
          auto solution = solve_mip(&handle, mps_data, settings);

          // Serialize result
          result_data = serializer->serialize_mip_solution(solution);
          success     = true;
        } else {
          error_message = "Failed to deserialize MIP request";
        }
      } else {
        // Deserialize LP request
        cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
        pdlp_solver_settings_t<int, double> settings;

        if (serializer->deserialize_lp_request(request_data, mps_data, settings)) {
          // Debug: print deserialized data
          std::cout << "[Server DEBUG] Deserialized LP problem:\n";
          std::cout << "  Maximize: " << mps_data.get_sense() << "\n";
          std::cout << "  Objective coeffs: [";
          for (size_t i = 0; i < mps_data.get_objective_coefficients().size(); ++i) {
            std::cout << mps_data.get_objective_coefficients()[i];
            if (i + 1 < mps_data.get_objective_coefficients().size()) std::cout << ", ";
          }
          std::cout << "]\n";
          std::cout << "  Constraint lower bounds: [";
          for (size_t i = 0; i < mps_data.get_constraint_lower_bounds().size(); ++i) {
            std::cout << mps_data.get_constraint_lower_bounds()[i];
            if (i + 1 < mps_data.get_constraint_lower_bounds().size()) std::cout << ", ";
          }
          std::cout << "]\n";
          std::cout << "  Constraint upper bounds: [";
          for (size_t i = 0; i < mps_data.get_constraint_upper_bounds().size(); ++i) {
            std::cout << mps_data.get_constraint_upper_bounds()[i];
            if (i + 1 < mps_data.get_constraint_upper_bounds().size()) std::cout << ", ";
          }
          std::cout << "]\n";
          std::cout.flush();

          // Solve using the data model directly
          auto solution = solve_lp(&handle, mps_data, settings);

          // Serialize result
          result_data = serializer->serialize_lp_solution(solution);
          success     = true;
        } else {
          error_message = "Failed to deserialize LP request";
        }
      }
    } catch (const std::exception& e) {
      error_message = std::string("Exception: ") + e.what();
    }

    // Update job status
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      auto it = job_tracker.find(job_id);
      if (it != job_tracker.end()) {
        if (success) {
          it->second.status      = JobStatus::COMPLETED;
          it->second.result_data = std::move(result_data);
        } else {
          it->second.status        = JobStatus::FAILED;
          it->second.error_message = error_message;
        }
      }
    }
    job_cv.notify_all();

    if (config.verbose) {
      std::cout << "[Worker " << worker_id << "] Completed job: " << job_id
                << " (success: " << success << ")\n";
    }
  }

  std::cout << "[Worker " << worker_id << "] Stopped\n";
}

// Handle client connection with streaming log support
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

  // Determine request type
  bool is_mip = serializer->is_mip_request(request_data);

  // For sync mode with streaming, process directly in this thread
  // to enable log streaming to the client
  std::string job_id = generate_job_id();

  if (config.verbose) {
    std::cout << "[Server] Processing job: " << job_id << " (type: " << (is_mip ? "MIP" : "LP")
              << ", streaming: " << (stream_logs ? "yes" : "no") << ")\n";
  }

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
          result_data   = serializer->serialize_mip_solution(solution);
          success       = true;
        } else {
          error_message = "Failed to deserialize MIP request";
        }
      } else {
        cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
        pdlp_solver_settings_t<int, double> settings;

        if (serializer->deserialize_lp_request(request_data, mps_data, settings)) {
          auto solution = solve_lp(&handle, mps_data, settings);
          result_data   = serializer->serialize_lp_solution(solution);
          success       = true;
        } else {
          error_message = "Failed to deserialize LP request";
        }
      }
    } catch (const std::exception& e) {
      error_message = std::string("Exception: ") + e.what();
    }
  }  // streamer destructor restores stdout

  if (config.verbose) {
    std::cout << "[Server] Completed job: " << job_id << " (success: " << success << ")\n";
  }

  // Send response - always use streaming protocol for consistency
  // (stream_logs only controls whether LOG_MESSAGE are sent during solve)
  if (!send_solution_message(client_fd, result_data)) {
    std::cerr << "[Server] Failed to send solution message\n";
  }

  close(client_fd);
}

// Legacy handle_client without streaming (backward compatible)
void handle_client(int client_fd) { handle_client(client_fd, false); }

void print_usage(const char* prog)
{
  std::cout << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  -p PORT    Port to listen on (default: 9090)\n"
            << "  -w NUM     Number of worker threads (default: 1)\n"
            << "  -q         Quiet mode (less verbose output)\n"
            << "  --no-stream  Disable real-time log streaming to clients\n"
            << "  -h         Show this help\n";
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
    } else if (strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  // Set up signal handlers
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  // IMPORTANT: Clear remote solve environment variables to prevent infinite recursion
  // The server should always do local solves, never try to connect to itself
  unsetenv("CUOPT_REMOTE_HOST");
  unsetenv("CUOPT_REMOTE_PORT");

  std::cout << "=== cuOpt Remote Solve Server ===\n";
  std::cout << "Port: " << config.port << "\n";
  std::cout << "Workers: " << config.num_workers << "\n";
  std::cout << "Log streaming: " << (config.stream_logs ? "enabled" : "disabled") << "\n";

  // Start worker threads
  std::vector<std::thread> workers;
  for (int i = 0; i < config.num_workers; ++i) {
    workers.emplace_back(worker_thread, i);
  }

  // Create server socket
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    std::cerr << "[Server] Failed to create socket\n";
    return 1;
  }

  // Allow address reuse
  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // Bind
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port        = htons(config.port);

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "[Server] Failed to bind to port " << config.port << "\n";
    close(server_fd);
    return 1;
  }

  // Listen
  if (listen(server_fd, 10) < 0) {
    std::cerr << "[Server] Failed to listen\n";
    close(server_fd);
    return 1;
  }

  std::cout << "[Server] Listening on port " << config.port << "\n";

  // Accept connections
  while (keep_running) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    // Use select for timeout so we can check keep_running
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
    if (ready == 0) continue;  // Timeout

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

    // Handle in separate thread with streaming based on config
    std::thread([client_fd]() { handle_client(client_fd, config.stream_logs); }).detach();
  }

  // Shutdown
  std::cout << "[Server] Shutting down...\n";
  close(server_fd);

  // Wait for workers
  job_cv.notify_all();
  for (auto& w : workers) {
    if (w.joinable()) w.join();
  }

  std::cout << "[Server] Stopped\n";
  return 0;
}
