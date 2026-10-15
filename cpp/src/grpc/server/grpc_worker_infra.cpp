/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_pipe_serialization.hpp"
#include "grpc_server_types.hpp"

#include <spawn.h>

#include <dirent.h>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>

extern char** environ;

namespace {

// GPU discovery for startup logging only. The parent must not call CUDA;
// each worker initializes its own context after exec.

// Descriptors the exec'd worker keeps. File actions dup2 the pipe ends onto
// these, then close every other inherited descriptor at or above this bound.
constexpr int kSpawnedJobReadFd        = 3;
constexpr int kSpawnedResultWriteFd    = 4;
constexpr int kSpawnedIncumbentWriteFd = 5;
constexpr int kSpawnedKeepBelowFd      = 6;

int dup_above_kept_fds(int fd)
{
  if (fd < 0) return -1;
  return fcntl(fd, F_DUPFD_CLOEXEC, kSpawnedKeepBelowFd);
}

void close_if_open(int fd)
{
  if (fd >= 0) close(fd);
}

// Exec this path rather than the readlink() string. After the binary is
// unlinked or replaced, readlink appends " (deleted)" and exec of that string
// fails with ENOENT. /proc/self/exe still names the running inode.
constexpr const char* kSpawnExecutable = "/proc/self/exe";

void* map_existing_shared_memory(const char* name, size_t size)
{
  const int fd = shm_open(name, O_RDWR, 0600);
  if (fd < 0) return MAP_FAILED;
  void* ptr       = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  const int saved = errno;
  close(fd);
  errno = saved;
  return ptr;
}

const char* next_arg(int argc, char** argv, int& i)
{
  if (i + 1 >= argc) return nullptr;
  return argv[++i];
}

// Snapshot of fds the child must not keep. Each open fd gets its own addclose.
// The directory fd is omitted because it is closed before spawn. Returns false
// when /proc/self/fd cannot be read.
bool inherited_fds_to_close(std::vector<int>& fds)
{
  fds.clear();
  DIR* dir = opendir("/proc/self/fd");
  if (dir == nullptr) return false;
  const int dir_fd = dirfd(dir);
  while (dirent* ent = readdir(dir)) {
    char* end         = nullptr;
    errno             = 0;
    const long parsed = std::strtol(ent->d_name, &end, 10);
    if (errno != 0 || end == ent->d_name || *end != '\0') continue;
    if (parsed < kSpawnedKeepBelowFd || parsed > INT_MAX) continue;
    const int fd = static_cast<int>(parsed);
    if (fd == dir_fd) continue;
    fds.push_back(fd);
  }
  closedir(dir);
  return true;
}

int count_cuda_visible_devices()
{
  const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
  if (visible == nullptr || visible[0] == '\0') { return 0; }

  int count     = 0;
  bool in_token = false;
  for (const char* p = visible; *p != '\0'; ++p) {
    if (*p == ',') {
      in_token = false;
    } else if (!std::isspace(static_cast<unsigned char>(*p))) {
      if (!in_token) {
        ++count;
        in_token = true;
      }
    }
  }
  return count;
}

int discover_gpu_count_via_nvidia_smi()
{
  FILE* fp = popen("nvidia-smi -L 2>/dev/null", "r");
  if (fp == nullptr) { return 0; }

  int count = 0;
  char line[512];
  while (fgets(line, sizeof(line), fp) != nullptr) {
    int gpu_id = -1;
    if (std::sscanf(line, "GPU %d:", &gpu_id) == 1) { ++count; }
  }
  pclose(fp);
  return count;
}

}  // namespace

void log_worker_gpu_layout()
{
  visible_gpu_count = count_cuda_visible_devices();
  if (visible_gpu_count <= 0) { visible_gpu_count = discover_gpu_count_via_nvidia_smi(); }

  if (visible_gpu_count <= 0) {
    SERVER_LOG_WARN(
      "[Server] Could not determine GPU count; workers will select devices at startup");
    return;
  }

  SERVER_LOG_INFO("[Server] %d GPU(s) available for worker assignment", visible_gpu_count);
  if (config.num_workers > visible_gpu_count) {
    SERVER_LOG_WARN("[Server] %d workers on %d GPU(s); multiple workers will share a GPU",
                    config.num_workers,
                    visible_gpu_count);
  }
}

void cleanup_shared_memory()
{
  if (job_queue) {
    munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
    shm_unlink(SHM_JOB_QUEUE.c_str());
    job_queue = nullptr;
  }
  if (result_queue) {
    munmap(result_queue, sizeof(ResultQueueEntry) * MAX_RESULTS);
    shm_unlink(SHM_RESULT_QUEUE.c_str());
    result_queue = nullptr;
  }
  if (worker_ready_flags) {
    munmap(worker_ready_flags, sizeof(std::atomic<bool>) * static_cast<size_t>(config.num_workers));
    shm_unlink(SHM_WORKER_READY.c_str());
    worker_ready_flags = nullptr;
  }
  if (shm_ctrl) {
    munmap(shm_ctrl, sizeof(SharedMemoryControl));
    shm_unlink(SHM_CONTROL.c_str());
    shm_ctrl = nullptr;
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
  // Nonblocking write end: write_to_pipe polls + retries so shutdown can abort.
  fcntl(wp.to_worker_fd, F_SETFL, fcntl(wp.to_worker_fd, F_GETFL) | O_NONBLOCK);

  if (pipe(fds) < 0) {
    SERVER_LOG_ERROR("[Server] Failed to create output pipe for worker %d", worker_id);
    close_all_worker_pipes(wp);
    return false;
  }
  wp.from_worker_fd  = fds[0];
  wp.worker_write_fd = fds[1];
  fcntl(wp.worker_write_fd, F_SETPIPE_SZ, kPipeBufferSize);
  fcntl(wp.worker_write_fd, F_SETFL, fcntl(wp.worker_write_fd, F_GETFL) | O_NONBLOCK);
  // Parent read end is also nonblocking; read_from_pipe polls before read.
  fcntl(wp.from_worker_fd, F_SETFL, fcntl(wp.from_worker_fd, F_GETFL) | O_NONBLOCK);

  if (pipe(fds) < 0) {
    SERVER_LOG_ERROR("[Server] Failed to create incumbent pipe for worker %d", worker_id);
    close_all_worker_pipes(wp);
    return false;
  }
  wp.incumbent_from_worker_fd  = fds[0];
  wp.worker_incumbent_write_fd = fds[1];
  fcntl(wp.worker_incumbent_write_fd,
        F_SETFL,
        fcntl(wp.worker_incumbent_write_fd, F_GETFL) | O_NONBLOCK);
  fcntl(
    wp.incumbent_from_worker_fd, F_SETFL, fcntl(wp.incumbent_from_worker_fd, F_GETFL) | O_NONBLOCK);

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

  const char* which = is_replacement ? "replacement worker " : "worker ";

  if (is_replacement) { close_worker_pipes_server(worker_id); }

  if (!create_worker_pipes(worker_id)) {
    SERVER_LOG_ERROR("[Server] Failed to create pipes for %s%d", which, worker_id);
    return -1;
  }

  WorkerPipes& wp = worker_pipes[worker_id];

  // posix_spawn + exec, not fork. The server is multithreaded. fork() would
  // copy it with the other threads' mutexes still locked, and the worker's
  // first log call deadlocks before CUDA init. exec starts one thread; the
  // solve creates its OpenMP team later.
  const int hold_read  = dup_above_kept_fds(wp.worker_read_fd);
  const int hold_write = dup_above_kept_fds(wp.worker_write_fd);
  const int hold_inc   = dup_above_kept_fds(wp.worker_incumbent_write_fd);
  if (hold_read < 0 || hold_write < 0 || hold_inc < 0) {
    SERVER_LOG_ERROR(
      "[Server] Failed to dup pipe ends for %s%d: %s", which, worker_id, strerror(errno));
    close_if_open(hold_read);
    close_if_open(hold_write);
    close_if_open(hold_inc);
    close_all_worker_pipes(wp);
    return -1;
  }

  // Exec drops the parent's address space. The worker only receives the fields
  // listed below: num_workers, verbose, log_to_console, and server_log_file.
  // Port, message size, chunk timeout, and TLS stay in the parent process.
  // Worker code must not read those ServerConfig fields; after exec they are
  // the struct defaults.
  const std::string id_str      = std::to_string(worker_id);
  const std::string workers_str = std::to_string(config.num_workers);
  std::vector<std::string> arg_storage;
  arg_storage.push_back(kSpawnExecutable);
  arg_storage.push_back("--worker");
  arg_storage.push_back("--worker-id");
  arg_storage.push_back(id_str);
  arg_storage.push_back("--workers");
  arg_storage.push_back(workers_str);
  arg_storage.push_back("--shm-job");
  arg_storage.push_back(SHM_JOB_QUEUE);
  arg_storage.push_back("--shm-result");
  arg_storage.push_back(SHM_RESULT_QUEUE);
  arg_storage.push_back("--shm-control");
  arg_storage.push_back(SHM_CONTROL);
  arg_storage.push_back("--shm-ready");
  arg_storage.push_back(SHM_WORKER_READY);
  arg_storage.push_back(config.verbose ? "--verbose" : "--quiet");
  if (is_replacement) { arg_storage.push_back("--replacement"); }
  if (config.log_to_console) { arg_storage.push_back("--log-to-console"); }
  if (!config.server_log_file.empty()) {
    arg_storage.push_back("--server-log");
    arg_storage.push_back(config.server_log_file);
  }

  std::vector<char*> argv;
  argv.reserve(arg_storage.size() + 1);
  for (std::string& arg : arg_storage) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  // The snapshot is taken inside the loop. Create any descriptor this function
  // needs before that, as the pipes above are. A descriptor opened afterward
  // without O_CLOEXEC, including from another thread, is inherited by the
  // worker. Do not add a plain open, pipe, or socket on a thread that runs
  // during respawn. gRPC sockets are already close-on-exec.
  //
  // Another thread can also close an fd between the snapshot and posix_spawn,
  // and addclose of a missing fd fails the spawn. Retry that race only.
  constexpr int kSpawnAttempts = 5;
  pid_t pid                    = -1;
  int spawn_rc                 = EAGAIN;
  for (int attempt = 0; attempt < kSpawnAttempts; ++attempt) {
    std::vector<int> close_fds;
    if (!inherited_fds_to_close(close_fds)) {
      spawn_rc = errno != 0 ? errno : EIO;
      break;
    }
    posix_spawn_file_actions_t actions;
    // Returns the error number and does not set errno. A zero errno here would
    // look like success, skip the pipe cleanup, and hand back pid -1.
    const int init_rc = posix_spawn_file_actions_init(&actions);
    if (init_rc != 0) {
      spawn_rc = init_rc;
      break;
    }

    int action_rc = posix_spawn_file_actions_adddup2(&actions, hold_read, kSpawnedJobReadFd);
    if (action_rc == 0) {
      action_rc = posix_spawn_file_actions_adddup2(&actions, hold_write, kSpawnedResultWriteFd);
    }
    if (action_rc == 0) {
      action_rc = posix_spawn_file_actions_adddup2(&actions, hold_inc, kSpawnedIncumbentWriteFd);
    }
    for (int fd : close_fds) {
      if (action_rc != 0) break;
      action_rc = posix_spawn_file_actions_addclose(&actions, fd);
    }

    spawn_rc = action_rc;
    if (action_rc == 0) {
      spawn_rc = posix_spawn(&pid, kSpawnExecutable, &actions, nullptr, argv.data(), environ);
    }
    posix_spawn_file_actions_destroy(&actions);
    if (spawn_rc == 0) break;
    if (spawn_rc != EBADF) break;
  }

  close_if_open(hold_read);
  close_if_open(hold_write);
  close_if_open(hold_inc);

  if (spawn_rc != 0) {
    SERVER_LOG_ERROR("[Server] Failed to spawn %s%d: %s", which, worker_id, strerror(spawn_rc));
    close_all_worker_pipes(wp);
    return -1;
  }

  close_worker_pipes_child_ends(worker_id);
  return pid;
}

int run_spawned_worker(int argc, char** argv)
{
  int worker_id           = -1;
  int num_workers         = -1;
  const char* shm_job     = nullptr;
  const char* shm_result  = nullptr;
  const char* shm_control = nullptr;
  const char* shm_ready   = nullptr;
  const char* server_log  = nullptr;
  bool verbose            = true;
  bool log_to_console     = false;
  bool is_replacement     = false;

  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--worker") == 0) {
      continue;
    } else if (std::strcmp(arg, "--verbose") == 0) {
      verbose = true;
    } else if (std::strcmp(arg, "--quiet") == 0) {
      verbose = false;
    } else if (std::strcmp(arg, "--log-to-console") == 0) {
      log_to_console = true;
    } else if (std::strcmp(arg, "--replacement") == 0) {
      is_replacement = true;
    } else if (std::strcmp(arg, "--worker-id") == 0 || std::strcmp(arg, "--workers") == 0) {
      const char* value = next_arg(argc, argv, i);
      if (value == nullptr) {
        std::cerr << "cuopt_grpc_server --worker: missing value for " << arg << "\n";
        return kWorkerAttachFailedExitCode;
      }
      char* end         = nullptr;
      errno             = 0;
      const long parsed = std::strtol(value, &end, 10);
      if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        std::cerr << "cuopt_grpc_server --worker: invalid " << arg << "\n";
        return kWorkerAttachFailedExitCode;
      }
      if (std::strcmp(arg, "--worker-id") == 0) {
        worker_id = static_cast<int>(parsed);
      } else {
        num_workers = static_cast<int>(parsed);
      }
    } else if (std::strcmp(arg, "--shm-job") == 0 || std::strcmp(arg, "--shm-result") == 0 ||
               std::strcmp(arg, "--shm-control") == 0 || std::strcmp(arg, "--shm-ready") == 0 ||
               std::strcmp(arg, "--server-log") == 0) {
      const char* value = next_arg(argc, argv, i);
      if (value == nullptr || value[0] == '\0') {
        std::cerr << "cuopt_grpc_server --worker: missing value for " << arg << "\n";
        return kWorkerAttachFailedExitCode;
      }
      if (std::strcmp(arg, "--shm-job") == 0) {
        shm_job = value;
      } else if (std::strcmp(arg, "--shm-result") == 0) {
        shm_result = value;
      } else if (std::strcmp(arg, "--shm-control") == 0) {
        shm_control = value;
      } else if (std::strcmp(arg, "--shm-ready") == 0) {
        shm_ready = value;
      } else {
        server_log = value;
      }
    } else {
      std::cerr << "cuopt_grpc_server --worker: unknown argument " << arg << "\n";
      return kWorkerAttachFailedExitCode;
    }
  }

  if (worker_id < 0 || num_workers < 1 || worker_id >= num_workers || shm_job == nullptr ||
      shm_result == nullptr || shm_control == nullptr || shm_ready == nullptr) {
    std::cerr << "cuopt_grpc_server --worker: incomplete arguments\n";
    return kWorkerAttachFailedExitCode;
  }

  config.num_workers    = num_workers;
  config.verbose        = verbose;
  config.log_to_console = log_to_console;
  if (server_log != nullptr) { config.server_log_file = server_log; }
  // A throw here leaves the process. The monitor treats a signal death as a
  // crash and respawns. Opening the log file cannot succeed on retry, so this
  // is the same fatal attach path as a missing shared-memory segment.
  try {
    init_server_logger(config.server_log_file, /*to_console=*/true, config.verbose);
  } catch (const std::exception& e) {
    std::cerr << "cuopt_grpc_server --worker: failed to initialize server logger: " << e.what()
              << "\n";
    return kWorkerAttachFailedExitCode;
  }

  // Map the parent's segments. Do not construct the entries: the parent
  // placement-new'd them, and a job may already be published.
  void* job_map = map_existing_shared_memory(shm_job, sizeof(JobQueueEntry) * MAX_JOBS);
  if (job_map == MAP_FAILED) {
    SERVER_LOG_ERROR("[Worker] Failed to map job queue: %s", strerror(errno));
    return kWorkerAttachFailedExitCode;
  }
  void* result_map = map_existing_shared_memory(shm_result, sizeof(ResultQueueEntry) * MAX_RESULTS);
  if (result_map == MAP_FAILED) {
    SERVER_LOG_ERROR("[Worker] Failed to map result queue: %s", strerror(errno));
    return kWorkerAttachFailedExitCode;
  }
  void* ctrl_map = map_existing_shared_memory(shm_control, sizeof(SharedMemoryControl));
  if (ctrl_map == MAP_FAILED) {
    SERVER_LOG_ERROR("[Worker] Failed to map control block: %s", strerror(errno));
    return kWorkerAttachFailedExitCode;
  }
  void* ready_map = map_existing_shared_memory(
    shm_ready, sizeof(std::atomic<bool>) * static_cast<size_t>(num_workers));
  if (ready_map == MAP_FAILED) {
    SERVER_LOG_ERROR("[Worker] Failed to map worker-ready flags: %s", strerror(errno));
    return kWorkerAttachFailedExitCode;
  }
  job_queue          = static_cast<JobQueueEntry*>(job_map);
  result_queue       = static_cast<ResultQueueEntry*>(result_map);
  shm_ctrl           = static_cast<SharedMemoryControl*>(ctrl_map);
  worker_ready_flags = static_cast<std::atomic<bool>*>(ready_map);

  while (static_cast<int>(worker_pipes.size()) <= worker_id) {
    worker_pipes.push_back({-1, -1, -1, -1, -1, -1});
  }
  WorkerPipes& pipes              = worker_pipes[worker_id];
  pipes.worker_read_fd            = kSpawnedJobReadFd;
  pipes.worker_write_fd           = kSpawnedResultWriteFd;
  pipes.worker_incumbent_write_fd = kSpawnedIncumbentWriteFd;

  worker_process(worker_id, is_replacement);
  _exit(0);
}

void spawn_workers()
{
  std::lock_guard<std::mutex> lock(worker_pids_mutex);
  // Index i is worker_id: keep failed startups as 0 so monitor/respawn and
  // pipe tables stay aligned even when some initial spawns fail.
  worker_pids.assign(static_cast<size_t>(config.num_workers), 0);
  for (int i = 0; i < config.num_workers; ++i) {
    pid_t pid = spawn_worker(i, false);
    if (pid > 0) { worker_pids[static_cast<size_t>(i)] = pid; }
  }
}

void kill_all_workers()
{
  std::lock_guard<std::mutex> lock(worker_pids_mutex);
  for (pid_t pid : worker_pids) {
    if (pid > 0) { kill(pid, SIGKILL); }
  }
}

void close_all_server_worker_pipes()
{
  std::lock_guard<std::mutex> lock(worker_pipes_mutex);
  for (auto& wp : worker_pipes) {
    close_all_worker_pipes(wp);
  }
}

void cancel_all_active_jobs_for_shutdown()
{
  if (job_queue != nullptr) {
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready.load(std::memory_order_acquire)) {
        job_queue[i].cancelled.store(true, std::memory_order_release);
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    for (auto& [job_id, info] : job_tracker) {
      (void)job_id;
      if (info.status == JobStatus::QUEUED || info.status == JobStatus::PROCESSING) {
        info.status        = JobStatus::CANCELLED;
        info.error_message = "Server shutting down";
      }
    }
  }

  {
    std::lock_guard<std::mutex> wlock(waiters_mutex);
    for (auto& [job_id, waiter] : waiting_threads) {
      (void)job_id;
      {
        std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
        waiter->error_message = "Server shutting down";
        waiter->success       = false;
        waiter->ready         = true;
      }
      waiter->cv.notify_all();
    }
    waiting_threads.clear();
  }

  result_cv.notify_all();
}

void wait_for_workers()
{
  // Mid-solve workers only check shutdown_requested between jobs, so force-kill
  // before waitpid or Ctrl-C / SIGTERM can hang until the current solve ends.
  // A worker killed mid-CUDA can sit in uninterruptible D-state while the GPU
  // driver tears down; a blocking waitpid would then hang the whole server
  // (and ignore SIGTERM because we retry on EINTR). Bound the wait and abandon
  // stragglers — the test harness / init reaps them.
  kill_all_workers();

  constexpr auto kShutdownWait = std::chrono::seconds(2);
  auto deadline                = std::chrono::steady_clock::now() + kShutdownWait;
  while (std::chrono::steady_clock::now() < deadline) {
    bool any_alive = false;
    {
      std::lock_guard<std::mutex> lock(worker_pids_mutex);
      for (pid_t& pid : worker_pids) {
        if (pid <= 0) continue;
        int status   = 0;
        pid_t reaped = waitpid(pid, &status, WNOHANG);
        if (reaped == pid || (reaped < 0 && errno == ECHILD)) {
          pid = 0;
        } else if (reaped < 0 && errno == EINTR) {
          any_alive = true;
        } else {
          any_alive = true;
        }
      }
      if (!any_alive) {
        worker_pids.clear();
        return;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  {
    std::lock_guard<std::mutex> lock(worker_pids_mutex);
    for (pid_t pid : worker_pids) {
      if (pid > 0) {
        kill(pid, SIGKILL);
        SERVER_LOG_WARN(
          "[Server] Worker pid %d did not exit within shutdown grace period; abandoning",
          static_cast<int>(pid));
      }
    }
    worker_pids.clear();
  }
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
