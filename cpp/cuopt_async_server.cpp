/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

/**
 * @file cuopt_async_server.cpp
 * @brief Async remote solve server with job queue and worker processes
 *
 * This server:
 * - Accepts async requests (submit, check status, get result, delete)
 * - Uses shared memory queues for job distribution
 * - Spawns solver worker processes
 * - Tracks job status and stores results
 * - Threaded result retrieval
 */

#include <cuopt_remote.pb.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

// Shared memory structures (must match worker)
struct JobQueueEntry {
  char job_id[64];
  uint32_t problem_type;
  uint32_t data_size;
  uint8_t data[1024 * 1024];  // 1MB
  bool ready;
  bool processed;
};

struct ResultQueueEntry {
  char job_id[64];
  uint32_t status;
  uint32_t data_size;
  uint8_t data[2 * 1024 * 1024];  // 2MB
  bool ready;
  bool retrieved;
};

const size_t MAX_JOBS    = 100;
const size_t MAX_RESULTS = 100;

// Job tracking
struct JobInfo {
  std::string job_id;
  cuopt::remote::JobStatus status;
  std::chrono::steady_clock::time_point submit_time;
  std::vector<uint8_t> result_data;  // Stored result
  uint32_t result_status;            // 0 = success, 1 = error
  bool is_blocking;                  // True if a thread is waiting synchronously
};

// Per-job condition variable for synchronous waiting
struct JobWaiter {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<uint8_t> result_data;
  uint32_t result_status;
  bool ready;

  JobWaiter() : ready(false), result_status(0) {}
};

// Global state
volatile sig_atomic_t keep_running = 1;
std::map<std::string, JobInfo> job_tracker;
std::mutex tracker_mutex;

std::map<std::string, std::shared_ptr<JobWaiter>> waiting_threads;
std::mutex waiters_mutex;

JobQueueEntry* job_queue       = nullptr;
ResultQueueEntry* result_queue = nullptr;
pid_t worker_pid               = -1;

void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\n[Server] Received shutdown signal\n";
    keep_running = 0;
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
static void write_all(int sockfd, const void* data, size_t size)
{
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t remaining   = size;
  while (remaining > 0) {
    ssize_t written = ::write(sockfd, ptr, remaining);
    if (written <= 0) throw std::runtime_error("Socket write failed");
    ptr += written;
    remaining -= written;
  }
}

static void read_all(int sockfd, void* data, size_t size)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    ssize_t nread = ::read(sockfd, ptr, remaining);
    if (nread <= 0) throw std::runtime_error("Socket read failed");
    ptr += nread;
    remaining -= nread;
  }
}

// Result retrieval thread
void result_retrieval_thread()
{
  std::cout << "[Server] Result retrieval thread started\n";

  while (keep_running) {
    bool found = false;

    // Scan result queue
    for (size_t i = 0; i < MAX_RESULTS; ++i) {
      if (result_queue[i].ready && !result_queue[i].retrieved) {
        found = true;
        std::string job_id(result_queue[i].job_id);

        std::cout << "[Server] Result retrieved for job: " << job_id << "\n";

        // Check if this is a blocking request (thread waiting)
        bool is_blocking = false;
        {
          std::lock_guard<std::mutex> lock(tracker_mutex);
          auto it = job_tracker.find(job_id);
          if (it != job_tracker.end()) { is_blocking = it->second.is_blocking; }
        }

        if (is_blocking) {
          // Synchronous mode - notify specific waiting thread
          std::lock_guard<std::mutex> lock(waiters_mutex);
          auto waiter_it = waiting_threads.find(job_id);

          if (waiter_it != waiting_threads.end()) {
            auto waiter = waiter_it->second;

            // Store result and signal THIS specific waiter
            {
              std::lock_guard<std::mutex> result_lock(waiter->mutex);
              waiter->result_data.assign(result_queue[i].data,
                                         result_queue[i].data + result_queue[i].data_size);
              waiter->result_status = result_queue[i].status;
              waiter->ready         = true;
              waiter->cv.notify_one();  // Wake ONLY this thread
            }

            std::cout << "[Server] Notified blocking thread for job: " << job_id << "\n";
          }
        } else {
          // Asynchronous mode - store in job tracker
          std::lock_guard<std::mutex> lock(tracker_mutex);
          auto it = job_tracker.find(job_id);
          if (it != job_tracker.end()) {
            it->second.status        = cuopt::remote::COMPLETED;
            it->second.result_status = result_queue[i].status;
            it->second.result_data.assign(result_queue[i].data,
                                          result_queue[i].data + result_queue[i].data_size);
          }
        }

        result_queue[i].retrieved = true;
        result_queue[i].ready     = false;  // Free slot
      }
    }

    if (!found) {
      usleep(50000);  // Sleep 50ms
    }
  }

  std::cout << "[Server] Result retrieval thread stopped\n";
}

// Handle job submission (async mode)
cuopt::remote::AsyncResponse handle_submit(const cuopt::remote::AsyncRequest& request)
{
  cuopt::remote::AsyncResponse response;
  response.set_request_type(cuopt::remote::SUBMIT_JOB);

  auto* submit_resp = response.mutable_submit_response();

  try {
    std::string job_id = generate_job_id();

    // Serialize the job data
    std::string job_data;
    if (request.has_lp_request()) {
      job_data = request.lp_request().SerializeAsString();
    } else if (request.has_mip_request()) {
      job_data = request.mip_request().SerializeAsString();
    } else {
      submit_resp->set_status(cuopt::remote::ERROR_INVALID_REQUEST);
      submit_resp->set_message("No job data provided");
      return response;
    }

    if (job_data.size() > sizeof(job_queue[0].data)) {
      submit_resp->set_status(cuopt::remote::ERROR_INVALID_REQUEST);
      submit_resp->set_message("Problem data too large");
      return response;
    }

    // Find free job slot
    bool queued = false;
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (!job_queue[i].ready) {
        strncpy(job_queue[i].job_id, job_id.c_str(), sizeof(job_queue[i].job_id) - 1);
        job_queue[i].problem_type = request.has_lp_request() ? 0 : 1;
        job_queue[i].data_size    = job_data.size();
        std::memcpy(job_queue[i].data, job_data.data(), job_data.size());
        job_queue[i].processed = false;
        job_queue[i].ready     = true;
        queued                 = true;
        break;
      }
    }

    if (!queued) {
      submit_resp->set_status(cuopt::remote::ERROR_INTERNAL);
      submit_resp->set_message("Job queue full");
      return response;
    }

    // Track job (async mode)
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      JobInfo info;
      info.job_id         = job_id;
      info.status         = cuopt::remote::QUEUED;
      info.submit_time    = std::chrono::steady_clock::now();
      info.is_blocking    = false;  // Async mode
      job_tracker[job_id] = info;
    }

    submit_resp->set_status(cuopt::remote::SUCCESS);
    submit_resp->set_job_id(job_id);
    submit_resp->set_message("Job queued successfully");

    std::cout << "[Server] Job submitted (async): " << job_id << "\n";

  } catch (const std::exception& e) {
    submit_resp->set_status(cuopt::remote::ERROR_INTERNAL);
    submit_resp->set_message(std::string("Error: ") + e.what());
  }

  return response;
}

// Handle synchronous (blocking) solve request
cuopt::remote::AsyncResponse handle_sync_solve(const cuopt::remote::AsyncRequest& request)
{
  cuopt::remote::AsyncResponse response;
  response.set_request_type(cuopt::remote::SUBMIT_JOB);  // Reuse submit type

  try {
    std::string job_id = generate_job_id();

    std::cout << "[Server] Sync solve request, job_id: " << job_id << "\n";

    // Serialize the job data
    std::string job_data;
    bool is_lp = false;
    if (request.has_lp_request()) {
      job_data = request.lp_request().SerializeAsString();
      is_lp    = true;
    } else if (request.has_mip_request()) {
      job_data = request.mip_request().SerializeAsString();
      is_lp    = false;
    } else {
      auto* error_resp = response.mutable_result_response();
      error_resp->set_status(cuopt::remote::ERROR_INVALID_REQUEST);
      error_resp->set_error_message("No job data provided");
      return response;
    }

    if (job_data.size() > sizeof(job_queue[0].data)) {
      auto* error_resp = response.mutable_result_response();
      error_resp->set_status(cuopt::remote::ERROR_INVALID_REQUEST);
      error_resp->set_error_message("Problem data too large");
      return response;
    }

    // Create waiter BEFORE submitting job
    auto waiter = std::make_shared<JobWaiter>();
    {
      std::lock_guard<std::mutex> lock(waiters_mutex);
      waiting_threads[job_id] = waiter;
    }

    // Submit to job queue
    bool queued = false;
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (!job_queue[i].ready) {
        strncpy(job_queue[i].job_id, job_id.c_str(), sizeof(job_queue[i].job_id) - 1);
        job_queue[i].problem_type = is_lp ? 0 : 1;
        job_queue[i].data_size    = job_data.size();
        std::memcpy(job_queue[i].data, job_data.data(), job_data.size());
        job_queue[i].processed = false;
        job_queue[i].ready     = true;
        queued                 = true;
        break;
      }
    }

    if (!queued) {
      std::lock_guard<std::mutex> lock(waiters_mutex);
      waiting_threads.erase(job_id);

      auto* error_resp = response.mutable_result_response();
      error_resp->set_status(cuopt::remote::ERROR_INTERNAL);
      error_resp->set_error_message("Job queue full");
      return response;
    }

    // Track job (blocking mode)
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      JobInfo info;
      info.job_id         = job_id;
      info.status         = cuopt::remote::QUEUED;
      info.submit_time    = std::chrono::steady_clock::now();
      info.is_blocking    = true;  // Blocking mode!
      job_tracker[job_id] = info;
    }

    std::cout << "[Server] Job queued (blocking), waiting for result...\n";

    // WAIT for result using per-job condition variable
    {
      std::unique_lock<std::mutex> lock(waiter->mutex);
      waiter->cv.wait(lock, [&waiter] { return waiter->ready; });
    }

    std::cout << "[Server] Job completed, returning result\n";

    // Result is ready, return it
    auto* result_resp = response.mutable_result_response();

    if (waiter->result_status == 0) {
      // Parse and return solution
      cuopt::remote::ResultResponse stored_result;
      if (stored_result.ParseFromArray(waiter->result_data.data(), waiter->result_data.size())) {
        result_resp->set_status(stored_result.status());
        if (stored_result.has_lp_solution()) {
          result_resp->mutable_lp_solution()->CopyFrom(stored_result.lp_solution());
        }
        if (stored_result.has_mip_solution()) {
          result_resp->mutable_mip_solution()->CopyFrom(stored_result.mip_solution());
        }
      } else {
        result_resp->set_status(cuopt::remote::ERROR_INTERNAL);
        result_resp->set_error_message("Failed to parse result");
      }
    } else {
      result_resp->set_status(cuopt::remote::ERROR_SOLVE_FAILED);
      result_resp->set_error_message("Solve failed");
    }

    // Cleanup waiter
    {
      std::lock_guard<std::mutex> lock(waiters_mutex);
      waiting_threads.erase(job_id);
    }

    // Cleanup job tracker
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      job_tracker.erase(job_id);
    }

  } catch (const std::exception& e) {
    auto* error_resp = response.mutable_result_response();
    error_resp->set_status(cuopt::remote::ERROR_INTERNAL);
    error_resp->set_error_message(std::string("Error: ") + e.what());
  }

  return response;
}

// Handle status check
cuopt::remote::AsyncResponse handle_status(const cuopt::remote::AsyncRequest& request)
{
  cuopt::remote::AsyncResponse response;
  response.set_request_type(cuopt::remote::CHECK_STATUS);

  auto* status_resp = response.mutable_status_response();

  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(request.job_id());

  if (it == job_tracker.end()) {
    status_resp->set_job_status(cuopt::remote::NOT_FOUND);
    status_resp->set_message("Job ID not found");
  } else {
    status_resp->set_job_status(it->second.status);

    switch (it->second.status) {
      case cuopt::remote::QUEUED: status_resp->set_message("Job is queued"); break;
      case cuopt::remote::PROCESSING: status_resp->set_message("Job is being processed"); break;
      case cuopt::remote::COMPLETED: status_resp->set_message("Job completed"); break;
      case cuopt::remote::FAILED: status_resp->set_message("Job failed"); break;
      default: status_resp->set_message("Unknown status");
    }
  }

  return response;
}

// Handle result retrieval
cuopt::remote::AsyncResponse handle_get_result(const cuopt::remote::AsyncRequest& request)
{
  cuopt::remote::AsyncResponse response;
  response.set_request_type(cuopt::remote::GET_RESULT);

  auto* result_resp = response.mutable_result_response();

  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(request.job_id());

  if (it == job_tracker.end()) {
    result_resp->set_status(cuopt::remote::ERROR_NOT_FOUND);
    result_resp->set_error_message("Job ID not found");
    return response;
  }

  if (it->second.status != cuopt::remote::COMPLETED) {
    result_resp->set_status(cuopt::remote::ERROR_INVALID_REQUEST);
    result_resp->set_error_message("Job not completed yet");
    return response;
  }

  if (it->second.result_status != 0) {
    result_resp->set_status(cuopt::remote::ERROR_SOLVE_FAILED);
    result_resp->set_error_message("Solve failed");
    return response;
  }

  // Parse stored result
  cuopt::remote::ResultResponse stored_result;
  if (!stored_result.ParseFromArray(it->second.result_data.data(), it->second.result_data.size())) {
    result_resp->set_status(cuopt::remote::ERROR_INTERNAL);
    result_resp->set_error_message("Failed to parse result");
    return response;
  }

  // Copy result
  result_resp->set_status(stored_result.status());
  if (stored_result.has_lp_solution()) {
    result_resp->mutable_lp_solution()->CopyFrom(stored_result.lp_solution());
  }
  if (stored_result.has_mip_solution()) {
    result_resp->mutable_mip_solution()->CopyFrom(stored_result.mip_solution());
  }

  std::cout << "[Server] Result retrieved for job: " << request.job_id() << "\n";

  return response;
}

// Handle delete
cuopt::remote::AsyncResponse handle_delete(const cuopt::remote::AsyncRequest& request)
{
  cuopt::remote::AsyncResponse response;
  response.set_request_type(cuopt::remote::DELETE_RESULT);

  auto* delete_resp = response.mutable_delete_response();

  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(request.job_id());

  if (it == job_tracker.end()) {
    delete_resp->set_status(cuopt::remote::ERROR_NOT_FOUND);
    delete_resp->set_message("Job ID not found");
  } else {
    job_tracker.erase(it);
    delete_resp->set_status(cuopt::remote::SUCCESS);
    delete_resp->set_message("Job deleted");

    std::cout << "[Server] Job deleted: " << request.job_id() << "\n";
  }

  return response;
}

// Handle client connection
void handle_client(int client_socket)
{
  try {
    // Read request size
    uint32_t request_size;
    read_all(client_socket, &request_size, sizeof(request_size));

    // Read request data
    std::vector<uint8_t> request_data(request_size);
    read_all(client_socket, request_data.data(), request_size);

    // Parse request
    cuopt::remote::AsyncRequest request;
    if (!request.ParseFromArray(request_data.data(), request_size)) {
      throw std::runtime_error("Failed to parse request");
    }

    // Handle based on request type and blocking flag
    cuopt::remote::AsyncResponse response;

    // Check for synchronous (blocking) mode
    if (request.request_type() == cuopt::remote::SUBMIT_JOB && request.blocking()) {
      // SYNCHRONOUS MODE - handler will block until result ready
      std::cout << "[Server] Handling blocking request\n";
      response = handle_sync_solve(request);
    } else {
      // ASYNCHRONOUS MODE - normal async workflow
      switch (request.request_type()) {
        case cuopt::remote::SUBMIT_JOB: response = handle_submit(request); break;
        case cuopt::remote::CHECK_STATUS: response = handle_status(request); break;
        case cuopt::remote::GET_RESULT: response = handle_get_result(request); break;
        case cuopt::remote::DELETE_RESULT: response = handle_delete(request); break;
        default: throw std::runtime_error("Unknown request type");
      }
    }

    // Send response (socket kept open during wait for blocking requests)
    std::string response_data = response.SerializeAsString();
    uint32_t response_size    = static_cast<uint32_t>(response_data.size());

    write_all(client_socket, &response_size, sizeof(response_size));
    write_all(client_socket, response_data.data(), response_data.size());

  } catch (const std::exception& e) {
    std::cerr << "[Server] Error handling client: " << e.what() << "\n";
  }

  close(client_socket);
}

int main(int argc, char* argv[])
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  int port = 9999;
  if (argc > 1) {
    port = std::atoi(argv[1]);
    if (port <= 0 || port > 65535) {
      std::cerr << "Error: Invalid port\n";
      return 1;
    }
  }

  std::cout << "==========================================================\n";
  std::cout << "cuOpt Async Remote Solve Server\n";
  std::cout << "==========================================================\n";
  std::cout << "Port: " << port << "\n";
  std::cout << "Press Ctrl+C to stop\n";
  std::cout << "==========================================================\n\n";

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  // Create shared memory for job queue
  shm_unlink("/cuopt_job_queue");
  int job_shm_fd = shm_open("/cuopt_job_queue", O_CREAT | O_RDWR, 0666);
  if (job_shm_fd == -1) {
    std::cerr << "Error: Failed to create job queue\n";
    return 1;
  }
  ftruncate(job_shm_fd, sizeof(JobQueueEntry) * MAX_JOBS);

  job_queue = static_cast<JobQueueEntry*>(mmap(
    nullptr, sizeof(JobQueueEntry) * MAX_JOBS, PROT_READ | PROT_WRITE, MAP_SHARED, job_shm_fd, 0));

  if (job_queue == MAP_FAILED) {
    std::cerr << "Error: Failed to map job queue\n";
    return 1;
  }

  // Initialize job queue
  std::memset(job_queue, 0, sizeof(JobQueueEntry) * MAX_JOBS);

  // Create shared memory for result queue
  shm_unlink("/cuopt_result_queue");
  int result_shm_fd = shm_open("/cuopt_result_queue", O_CREAT | O_RDWR, 0666);
  if (result_shm_fd == -1) {
    std::cerr << "Error: Failed to create result queue\n";
    return 1;
  }
  ftruncate(result_shm_fd, sizeof(ResultQueueEntry) * MAX_RESULTS);

  result_queue = static_cast<ResultQueueEntry*>(mmap(nullptr,
                                                     sizeof(ResultQueueEntry) * MAX_RESULTS,
                                                     PROT_READ | PROT_WRITE,
                                                     MAP_SHARED,
                                                     result_shm_fd,
                                                     0));

  if (result_queue == MAP_FAILED) {
    std::cerr << "Error: Failed to map result queue\n";
    return 1;
  }

  // Initialize result queue
  std::memset(result_queue, 0, sizeof(ResultQueueEntry) * MAX_RESULTS);

  std::cout << "[Server] Shared memory queues created\n";

  // Spawn solver worker
  worker_pid = fork();
  if (worker_pid == 0) {
    // Child process - become worker

    // First, try to find worker in same directory as this executable
    char self_path[1024];
    ssize_t len = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
    if (len != -1) {
      self_path[len] = '\0';
      // Find last '/' to get directory
      char* last_slash = strrchr(self_path, '/');
      if (last_slash) {
        *(last_slash + 1) = '\0';  // Truncate to directory
        strcat(self_path, "cuopt_solver_worker");
        execl(self_path, "cuopt_solver_worker", nullptr);
      }
    }

    // Try PATH search
    execlp("cuopt_solver_worker", "cuopt_solver_worker", nullptr);

    // Try other common locations
    const char* worker_paths[] = {"/usr/local/bin/cuopt_solver_worker",
                                  "/usr/bin/cuopt_solver_worker",
                                  "./cuopt_solver_worker",
                                  nullptr};

    for (int i = 0; worker_paths[i] != nullptr; ++i) {
      execl(worker_paths[i], "cuopt_solver_worker", nullptr);
    }

    std::cerr << "Error: Failed to exec worker\n";
    std::cerr << "Searched: /proc/self/exe directory, PATH, and standard locations\n";
    exit(1);
  } else if (worker_pid < 0) {
    std::cerr << "Error: Failed to fork worker\n";
    return 1;
  }

  std::cout << "[Server] Solver worker started (PID: " << worker_pid << ")\n";

  // Start result retrieval thread
  std::thread result_thread(result_retrieval_thread);

  // Create socket
  int server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket < 0) {
    std::cerr << "Error: Failed to create socket\n";
    return 1;
  }

  int opt = 1;
  setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in server_addr;
  std::memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family      = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port        = htons(port);

  if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    std::cerr << "Error: Failed to bind\n";
    return 1;
  }

  if (listen(server_socket, 5) < 0) {
    std::cerr << "Error: Failed to listen\n";
    return 1;
  }

  std::cout << "[Server] Listening on port " << port << "...\n\n";

  // Accept loop
  while (keep_running) {
    struct timeval tv;
    tv.tv_sec  = 1;
    tv.tv_usec = 0;
    setsockopt(server_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);

    if (client_socket < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      if (keep_running) std::cerr << "[Server] Warning: Failed to accept\n";
      continue;
    }

    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
    std::cout << "[Server] Connection from " << client_ip << "\n";

    handle_client(client_socket);
  }

  // Cleanup
  std::cout << "[Server] Shutting down...\n";

  close(server_socket);
  result_thread.join();

  if (worker_pid > 0) {
    kill(worker_pid, SIGTERM);
    waitpid(worker_pid, nullptr, 0);
    std::cout << "[Server] Worker process stopped\n";
  }

  munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
  munmap(result_queue, sizeof(ResultQueueEntry) * MAX_RESULTS);
  close(job_shm_fd);
  close(result_shm_fd);
  shm_unlink("/cuopt_job_queue");
  shm_unlink("/cuopt_result_queue");

  google::protobuf::ShutdownProtobufLibrary();

  std::cout << "[Server] Stopped\n";
  return 0;
}
