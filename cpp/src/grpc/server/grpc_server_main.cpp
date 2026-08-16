/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_server_main.cpp
 * @brief gRPC-based remote solve server entry point
 *
 * This server uses gRPC for client communication with fork-based worker
 * process infrastructure:
 * - Worker processes with shared memory job queues
 * - Pipe-based IPC for problem/result data
 * - Result tracking and retrieval threads
 * - Log streaming
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_server_types.hpp"

#include <argparse/argparse.hpp>
#include <cuopt/version_config.hpp>

#include <pthread.h>

// Defined in grpc_service_impl.cpp
std::unique_ptr<grpc::Service> create_cuopt_grpc_service();

// Open, size, and mmap a POSIX shared memory segment.  Throws on failure.
static void* create_shared_memory(const char* name, size_t size)
{
  using cuopt::cuopt_expects;
  using cuopt::error_type_t;

  int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
  cuopt_expects(fd >= 0,
                error_type_t::RuntimeError,
                "Failed to create shared memory '%s': %s",
                name,
                strerror(errno));
  if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
    int saved = errno;
    close(fd);
    cuopt_expects(false,
                  error_type_t::RuntimeError,
                  "Failed to size shared memory '%s': %s",
                  name,
                  strerror(saved));
  }
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  int saved = errno;
  close(fd);
  cuopt_expects(ptr != MAP_FAILED,
                error_type_t::RuntimeError,
                "Failed to mmap shared memory '%s': %s",
                name,
                strerror(saved));
  return ptr;
}

int main(int argc, char** argv)
{
  const std::string version_string =
    std::string("cuOpt gRPC Server ") + std::to_string(CUOPT_VERSION_MAJOR) + "." +
    std::to_string(CUOPT_VERSION_MINOR) + "." + std::to_string(CUOPT_VERSION_PATCH);

  argparse::ArgumentParser program("cuopt_grpc_server", version_string);

  program.add_argument("-p", "--port")
    .help("Listen port")
    .default_value(cuopt_default_grpc_port)
    .scan<'i', int>();

  program.add_argument("-w", "--workers")
    .help("Number of worker processes")
    .default_value(1)
    .scan<'i', int>();

  program.add_argument("--max-message-mb")
    .help("gRPC max send/recv message size in MiB")
    .default_value(256)
    .scan<'i', int>();

  program.add_argument("--max-message-bytes")
    .help("Set max message size in exact bytes (min 4096, for testing)")
    .scan<'i', int64_t>();

  program.add_argument("--chunk-timeout")
    .help("Per-chunk timeout in seconds for streaming (0=disabled)")
    .default_value(60)
    .scan<'i', int>();

  program.add_argument("--tls")
    .help("Enable TLS (requires --tls-cert and --tls-key)")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--tls-cert").help("Path to PEM-encoded server certificate");

  program.add_argument("--tls-key").help("Path to PEM-encoded server private key");

  program.add_argument("--tls-root").help("Path to PEM root certs for client verification");

  program.add_argument("--require-client-cert")
    .help("Require and verify client certs (mTLS)")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--log-to-console")
    .help("Enable solver log output to console")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("-v", "--verbose")
    .help("Increase verbosity (default: on)")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("-q", "--quiet")
    .help("Reduce verbosity")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--server-log")
    .help("Path to server operational log file (in addition to console)");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    std::cerr << program;
    return 1;
  }

  config.port        = program.get<int>("--port");
  config.num_workers = program.get<int>("--workers");

  if (program.is_used("--max-message-bytes")) {
    config.max_message_bytes =
      std::max(static_cast<int64_t>(4096), program.get<int64_t>("--max-message-bytes"));
  } else {
    config.max_message_bytes = static_cast<int64_t>(program.get<int>("--max-message-mb")) * kMiB;
  }

  config.chunk_timeout_seconds = program.get<int>("--chunk-timeout");
  config.enable_tls            = program.get<bool>("--tls");
  config.require_client        = program.get<bool>("--require-client-cert");
  config.log_to_console        = program.get<bool>("--log-to-console");

  if (auto val = program.present("--tls-cert")) config.tls_cert_path = *val;
  if (auto val = program.present("--tls-key")) config.tls_key_path = *val;
  if (auto val = program.present("--tls-root")) config.tls_root_path = *val;

  config.verbose = !program.get<bool>("--quiet");

  if (auto val = program.present("--server-log")) config.server_log_file = *val;

  init_server_logger(config.server_log_file, /*to_console=*/true, config.verbose);

  // ---------------------------------------------------------------------------
  // Startup validation and resource allocation.
  // cuopt_expects throws cuopt::logic_error on failure; the catch block
  // prints a clean message and tears down any shared memory created so far.
  // ---------------------------------------------------------------------------
  using cuopt::cuopt_expects;
  using cuopt::error_type_t;

  std::string server_address;
  std::shared_ptr<grpc::ServerCredentials> creds;

  try {
    cuopt_expects(config.port >= 1 && config.port <= 65535,
                  error_type_t::ValidationError,
                  "--port must be in range 1-65535");
    cuopt_expects(config.num_workers >= 1, error_type_t::ValidationError, "--workers must be >= 1");
    cuopt_expects(config.chunk_timeout_seconds >= 0,
                  error_type_t::ValidationError,
                  "--chunk-timeout must be >= 0");

    config.max_message_bytes =
      std::clamp(config.max_message_bytes, kServerMinMessageBytes, kServerMaxMessageBytes);

    SERVER_LOG_INFO("cuOpt gRPC Remote Solve Server");
    SERVER_LOG_INFO("==============================");
    SERVER_LOG_INFO("Port: %d", config.port);
    SERVER_LOG_INFO("Workers: %d", config.num_workers);

    // Block shutdown signals in this thread before spawning workers / gRPC /
    // background threads so they inherit the mask. A dedicated sigwait thread
    // (started later) is then the only place SIGINT/SIGTERM are consumed.
    // Async signal() handlers are unreliable once libraries mask signals.
    sigset_t shutdown_sigset;
    sigemptyset(&shutdown_sigset);
    sigaddset(&shutdown_sigset, SIGINT);
    sigaddset(&shutdown_sigset, SIGTERM);
    int mask_rc = pthread_sigmask(SIG_BLOCK, &shutdown_sigset, nullptr);
    if (mask_rc != 0) {
      SERVER_LOG_ERROR("[Server] Failed to block shutdown signals: %s", strerror(mask_rc));
      return 1;
    }

    ensure_log_dir_exists();

    shm_unlink(SHM_JOB_QUEUE.c_str());
    shm_unlink(SHM_RESULT_QUEUE.c_str());
    shm_unlink(SHM_CONTROL.c_str());

    job_queue = static_cast<JobQueueEntry*>(
      create_shared_memory(SHM_JOB_QUEUE.c_str(), sizeof(JobQueueEntry) * MAX_JOBS));
    result_queue = static_cast<ResultQueueEntry*>(
      create_shared_memory(SHM_RESULT_QUEUE.c_str(), sizeof(ResultQueueEntry) * MAX_RESULTS));
    shm_ctrl = static_cast<SharedMemoryControl*>(
      create_shared_memory(SHM_CONTROL.c_str(), sizeof(SharedMemoryControl)));
    new (shm_ctrl) SharedMemoryControl{};

    for (size_t i = 0; i < MAX_JOBS; ++i) {
      new (&job_queue[i]) JobQueueEntry{};
      job_queue[i].ready.store(false);
      job_queue[i].claimed.store(false);
      job_queue[i].cancelled.store(false);
      job_queue[i].worker_index.store(-1);
    }

    for (size_t i = 0; i < MAX_RESULTS; ++i) {
      new (&result_queue[i]) ResultQueueEntry{};
      result_queue[i].claimed.store(false);
      result_queue[i].ready.store(false);
      result_queue[i].retrieved.store(false);
    }

    shm_ctrl->shutdown_requested.store(false);
    shm_ctrl->active_workers.store(0);

    // Build credentials before spawning workers so TLS validation failures
    // don't leak worker processes or background threads.
    server_address = "0.0.0.0:" + std::to_string(config.port);
    if (config.enable_tls) {
      cuopt_expects(!config.tls_cert_path.empty() && !config.tls_key_path.empty(),
                    error_type_t::ValidationError,
                    "TLS enabled but --tls-cert/--tls-key not provided");

      grpc::SslServerCredentialsOptions ssl_opts;
      grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert;
      key_cert.cert_chain  = read_file_to_string(config.tls_cert_path);
      key_cert.private_key = read_file_to_string(config.tls_key_path);
      cuopt_expects(!key_cert.cert_chain.empty() && !key_cert.private_key.empty(),
                    error_type_t::RuntimeError,
                    "Failed to read TLS cert/key files");
      ssl_opts.pem_key_cert_pairs.push_back(key_cert);

      if (!config.tls_root_path.empty()) {
        ssl_opts.pem_root_certs = read_file_to_string(config.tls_root_path);
        cuopt_expects(!ssl_opts.pem_root_certs.empty(),
                      error_type_t::RuntimeError,
                      "Failed to read TLS root cert file");
      }

      cuopt_expects(!config.require_client || !ssl_opts.pem_root_certs.empty(),
                    error_type_t::ValidationError,
                    "--require-client-cert requires --tls-root");

      if (config.require_client) {
        ssl_opts.client_certificate_request =
          GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
      } else if (!ssl_opts.pem_root_certs.empty()) {
        ssl_opts.client_certificate_request = GRPC_SSL_REQUEST_CLIENT_CERTIFICATE_AND_VERIFY;
      }

      creds = grpc::SslServerCredentials(ssl_opts);
    } else {
      creds = grpc::InsecureServerCredentials();
    }

    signal(SIGPIPE, SIG_IGN);
    log_worker_gpu_layout();
    spawn_workers();

    {
      std::lock_guard<std::mutex> lock(worker_pids_mutex);
      bool any_worker = false;
      for (pid_t pid : worker_pids) {
        if (pid > 0) {
          any_worker = true;
          break;
        }
      }
      cuopt_expects(any_worker, error_type_t::RuntimeError, "No workers started");
    }

  } catch (const cuopt::logic_error& e) {
    SERVER_LOG_ERROR("[Server] %s", format_cuopt_error(e));
    cleanup_shared_memory();
    return 1;
  } catch (const std::exception& e) {
    SERVER_LOG_ERROR("[Server] %s", e.what());
    cleanup_shared_memory();
    return 1;
  }

  // ---------------------------------------------------------------------------
  // Server is initialized.  Start background threads and the gRPC listener.
  // From here, shutdown requires joining threads and killing workers, so
  // errors use explicit checks with shutdown_all().
  // ---------------------------------------------------------------------------
  std::thread result_thread(result_retrieval_thread);
  std::thread incumbent_thread(incumbent_retrieval_thread);
  std::thread monitor_thread(worker_monitor_thread);
  std::thread reaper_thread(session_reaper_thread);

  auto shutdown_all = [&]() {
    keep_running                 = false;
    shm_ctrl->shutdown_requested = true;
    cancel_all_active_jobs_for_shutdown();
    kill_all_workers();
    result_cv.notify_all();

    // Join writers/readers before closing FDs. Closing a pipe FD from another
    // thread does not unblock an in-flight write on Linux and can race with
    // FD-number reuse. Pipe I/O aborts via shutdown_requested + O_NONBLOCK.
    if (result_thread.joinable()) result_thread.join();
    if (incumbent_thread.joinable()) incumbent_thread.join();
    if (monitor_thread.joinable()) monitor_thread.join();
    if (reaper_thread.joinable()) reaper_thread.join();

    close_all_server_worker_pipes();
    wait_for_workers();
    cleanup_shared_memory();
  };

  auto service = create_cuopt_grpc_service();

  ServerBuilder builder;
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(service.get());
  const int64_t max_bytes = server_max_message_bytes();
  const int channel_limit =
    static_cast<int>(std::min<int64_t>(max_bytes, std::numeric_limits<int>::max()));
  builder.SetMaxReceiveMessageSize(channel_limit);
  builder.SetMaxSendMessageSize(channel_limit);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  if (!server) {
    SERVER_LOG_ERROR("[Server] Failed to bind to %s", server_address);
    shutdown_all();
    return 1;
  }

  SERVER_LOG_INFO("[gRPC Server] Listening on %s", server_address);
  SERVER_LOG_INFO("[gRPC Server] Workers: %d", config.num_workers);
  SERVER_LOG_INFO("[gRPC Server] Max message size: %ld bytes (%ld MiB)",
                  server_max_message_bytes(),
                  server_max_message_bytes() / kMiB);
  SERVER_LOG_INFO("[gRPC Server] Press Ctrl+C to shutdown");

  // Dedicated signal thread: sigwait is reliable in multithreaded processes
  // where gRPC/CUDA may mask signals on their threads. On SIGINT/SIGTERM we
  // cancel jobs, kill workers, close pipes, and shut down gRPC. A cancelable
  // watchdog forces process exit only if cleanup hangs (e.g. GPU D-state).
  sigset_t shutdown_sigset;
  sigemptyset(&shutdown_sigset);
  sigaddset(&shutdown_sigset, SIGINT);
  sigaddset(&shutdown_sigset, SIGTERM);

  // Shared so the detached watchdog can observe cancellation after a healthy
  // shutdown_all() completes. Margin is well above the bounded pieces of the
  // clean path (gRPC Shutdown deadline 1s + wait_for_workers grace 2s + joins).
  // static constexpr: nested lambdas can use it without capturing.
  static constexpr auto kShutdownWatchdog = std::chrono::seconds(10);
  auto shutdown_watchdog_cancelled        = std::make_shared<std::atomic<bool>>(false);

  std::thread shutdown_thread([&server, shutdown_sigset, shutdown_watchdog_cancelled]() mutable {
    int sig = 0;
    int rc  = sigwait(&shutdown_sigset, &sig);
    if (rc != 0) {
      SERVER_LOG_ERROR("[Server] sigwait failed: %s", strerror(rc));
      sig = SIGTERM;
    }

    SERVER_LOG_INFO("[Server] Shutdown signal %d received; cancelling jobs and killing workers",
                    sig);
    keep_running = false;
    if (shm_ctrl) { shm_ctrl->shutdown_requested = true; }

    std::thread([shutdown_watchdog_cancelled]() {
      auto deadline = std::chrono::steady_clock::now() + kShutdownWatchdog;
      while (!shutdown_watchdog_cancelled->load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
          SERVER_LOG_ERROR("[Server] Shutdown watchdog expired after %llds; forcing unclean exit",
                           static_cast<long long>(kShutdownWatchdog.count()));
          _exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }).detach();

    cancel_all_active_jobs_for_shutdown();
    kill_all_workers();
    // Do not close server-side pipes here: result/incumbent threads may still
    // be in write_to_pipe/read_from_pipe. They abort via shutdown_requested;
    // shutdown_all() closes FDs after those threads join.
    if (server) {
      auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
      server->Shutdown(deadline);
    }
  });

  server->Wait();
  if (shutdown_thread.joinable()) shutdown_thread.join();

  SERVER_LOG_INFO("[Server] Shutting down...");
  shutdown_all();

  // Disarm the watchdog so a healthy cleanup cannot race into _exit(1).
  shutdown_watchdog_cancelled->store(true, std::memory_order_release);

  SERVER_LOG_INFO("[Server] Shutdown complete");
  return 0;
}

#else  // !CUOPT_ENABLE_GRPC

#include <iostream>

int main()
{
  std::cerr << "Error: cuopt_grpc_server requires gRPC support.\n"
            << "Rebuild with gRPC enabled (CUOPT_ENABLE_GRPC=ON)" << std::endl;
  return 1;
}

#endif  // CUOPT_ENABLE_GRPC
