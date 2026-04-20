# NVIDIA cuOpt gRPC server architecture

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

> **Audience:** cuOpt contributors and advanced integrators debugging the server.
>
> End users should start with the cuOpt documentation **gRPC remote execution** section — Quick start, **Advanced configuration** (flags, TLS, Docker, client env vars), and the short **gRPC server behavior** overview (`docs/cuopt/source/cuopt-grpc/grpc-server-architecture.md` in this repository). Those pages intentionally omit the C++-level detail below.

The NVIDIA cuOpt gRPC server (`cuopt_grpc_server`) is a multi-process architecture designed for:
- **Isolation**: Each solve runs in a separate worker process for fault tolerance
- **Parallelism**: Multiple workers can process jobs concurrently
- **Large Payloads**: Handles multi-GB problems and solutions
- **Real-Time Feedback**: Log streaming and incumbent callbacks during solve

Server source files live under `cpp/src/grpc/server/`.

## Process Model

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Main Server Process                          │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │  gRPC       │  │  Job         │  │  Background Threads         │ │
│  │  Service    │  │  Tracker     │  │  - Result retrieval         │ │
│  │  Handler    │  │  (job status,│  │  - Incumbent retrieval      │ │
│  │             │  │   results)   │  │  - Worker monitor           │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────────┘ │
│         │                                        ▲                  │
│         │ shared memory                          │ pipes            │
│         ▼                                        │                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Shared Memory Queues                         ││
│  │   ┌─────────────────┐        ┌─────────────────────┐            ││
│  │   │  Job Queue      │        │  Result Queue       │            ││
│  │   │  (MAX_JOBS=100) │        │  (MAX_RESULTS=100)  │            ││
│  │   └─────────────────┘        └─────────────────────┘            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
         │                                        ▲
         │ fork()                                 │
         ▼                                        │
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Worker 0       │  │  Worker 1       │  │  Worker N       │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ GPU Solve │  │  │  │ GPU Solve │  │  │  │ GPU Solve │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
│  (separate proc)│  │  (separate proc)│  │  (separate proc)│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Inter-Process Communication

### Shared Memory Segments

| Segment | Purpose |
|---------|---------|
| `/cuopt_job_queue` | Job metadata (ID, type, size, status) |
| `/cuopt_result_queue` | Result metadata (ID, status, size, error) |
| `/cuopt_control` | Server control flags (shutdown, worker count) |

### Pipe Communication

Each worker has dedicated pipes for data transfer:

```cpp
struct WorkerPipes {
  int to_worker_fd;               // Main   -> Worker: server writes job data
  int from_worker_fd;             // Worker -> Main: server reads result data
  int worker_read_fd;             // Main   -> Worker: worker reads job data
  int worker_write_fd;            // Worker -> Main: worker writes result data
  int incumbent_from_worker_fd;   // Worker -> Main: server reads incumbent solutions
  int worker_incumbent_write_fd;  // Worker -> Main: worker writes incumbent solutions
};
```

**Why pipes instead of shared memory for data?**
- Pipes handle backpressure naturally (blocking writes)
- No need to manage large shared memory segments
- Simpler lifecycle: data is consumed by the worker read and requires no explicit cleanup

### Source File Roles

All paths below are under `cpp/src/grpc/server/`.

| File | Role |
|------|------|
| `grpc_server_main.cpp` | `main()`, argument parsing (via argparse), shared-memory init, gRPC server run/stop. |
| `grpc_service_impl.cpp` | `CuOptRemoteServiceImpl`: all 14 RPC handlers (SubmitJob, CheckStatus, GetResult, StartChunkedUpload, SendArrayChunk, FinishChunkedUpload, StartChunkedDownload, GetResultChunk, FinishChunkedDownload, StreamLogs, GetIncumbents, CancelJob, DeleteResult, WaitForCompletion). Uses mappers and job_management to enqueue jobs and trigger pipe I/O. |
| `grpc_server_types.hpp` | Shared structs (e.g. `JobQueueEntry`, `ResultQueueEntry`, `ServerConfig`, `JobInfo`), enums, globals (atomics, mutexes, condition variables), and forward declarations used across server .cpp files. |
| `grpc_server_logger.hpp` | Server operational logger declaration (`server_logger()`, `init_server_logger()`) and `SERVER_LOG_*` convenience macros built on `rapids_logger`. Separate from the solver logger. |
| `grpc_server_logger.cpp` | Server logger implementation: constructs a `rapids_logger::logger` with configurable console/file sinks and verbose/quiet levels. Created before `fork()` so both main and worker processes share the same output. |
| `grpc_field_element_size.hpp` | Maps `cuopt::remote::ArrayFieldId` to element byte size; used by pipe deserialization and chunked logic. |
| `grpc_pipe_io.cpp` | Low-level pipe I/O primitives: `write_to_pipe()` (blocking retry loop) and `read_from_pipe()` (poll-based timeout + blocking read). Used by all higher-level pipe functions. |
| `grpc_pipe_serialization.hpp` | Protobuf-level pipe serialization: write/read length-prefixed protobuf messages and raw arrays to/from pipe fds. Also serializes `SubmitJobRequest` for unary pipe transfer. Defines `kPipeBufferSize` and `kMaxProtobufMessageBytes`. |
| `grpc_incumbent_proto.hpp` | Build `Incumbent` proto from (job_id, objective, assignment) and parse it back; used by worker when pushing incumbents and by main when reading from the incumbent pipe. |
| `grpc_worker.cpp` | `worker_process(worker_index)`: loop over job queue, receive job data via pipe (unary or chunked), call solver, send result (and optionally incumbents) back. Contains `IncumbentPipeCallback`, `store_simple_result`, and `publish_result`. |
| `grpc_worker_infra.cpp` | Pipe creation/teardown, `spawn_worker` / `spawn_workers` / `spawn_single_worker`, `wait_for_workers`, `mark_worker_jobs_failed`, `cleanup_shared_memory`. |
| `grpc_server_threads.cpp` | `worker_monitor_thread`, `result_retrieval_thread` (also dispatches job data to workers), `incumbent_retrieval_thread`, `session_reaper_thread`. |
| `grpc_job_management.cpp` | Pipe-level send/recv (`send_job_data_pipe`, `recv_job_data_pipe`, `send_incumbent_pipe`, `recv_incumbent_pipe`), `submit_job_async`, `submit_chunked_job_async`, `check_job_status`, `cancel_job`, `generate_job_id`, log-dir helpers. |

### Large Payload Handling

For large problems uploaded via chunked gRPC RPCs:

1. Server holds chunked upload state in memory (`ChunkedUploadState`: header + array chunks per `upload_id`).
2. When `FinishChunkedUpload` is called, the header and chunks are stored in `pending_chunked_data`. The result retrieval thread (which also handles job dispatch) streams them directly to the worker pipe as individual length-prefixed protobuf messages — no intermediate blob is created.
3. Worker reads the streamed messages from the pipe, reassembles arrays, runs the solver, and writes the result (and optionally incumbents) back via pipes using the same streaming format.
4. Main process result-retrieval thread reads the streamed result messages from the pipe and stores the result for `GetResult` or chunked download.

This streaming approach avoids creating a single large buffer, eliminating the 2 GiB protobuf serialization limit for pipe transfers and reducing peak memory usage. Each individual protobuf message (max 64 MiB) is serialized with standard `SerializeToArray`/`ParseFromArray`.

No disk spooling: chunked data is kept in memory in the main process until forwarded to the worker.

## Job Lifecycle

### 1. Submission

```text
Client                     Server                      Worker
   │                          │                           │
   │─── SubmitJob ──────────► │                           │
   │                          │ Create job entry          │
   │                          │ Store problem data        │
   │                          │ job_queue[slot].ready=true│
   │◄── job_id ────────────── │                           │
```

### 2. Processing

```text
Client                     Server                      Worker
   │                          │                           │
   │                          │                           │ Poll job_queue
   │                          │                           │ Claim job (CAS)
   │                          │ ── job data via pipe ───> │
   │                          │                           │
   │                          │                           │ Convert CPU→GPU
   │                          │                           │ solve_lp/solve_mip
   │                          │                           │ Convert GPU→CPU
   │                          │                           │
   │                          │ result_queue[slot].ready  │ (worker sets flag)
   │                          │ <── result data via pipe ─│
```

### 3. Result Retrieval

```text
Client                     Server                      Worker
   │                          │                           │
   │─── CheckStatus ────────►│                           │
   │◄── COMPLETED ───────────│                           │
   │                          │                           │
   │─── GetResult ──────────►│                           │
   │                          │ Look up job_tracker      │
   │◄── solution ────────────│                           │
```

## Data Type Conversions

Workers perform CPU↔GPU conversions to minimize client complexity:

```text
Client                     Worker
   │                          │
   │  cpu_optimization_       │
   │  problem_t        ──────►│ map_proto_to_problem()
   │                          │      ↓
   │                          │ to_optimization_problem()
   │                          │      ↓ (GPU)
   │                          │ solve_lp() / solve_mip()
   │                          │      ↓ (GPU)
   │                          │ cudaMemcpy() to host
   │                          │      ↓
   │  cpu_lp_solution_t/      │ map_lp_solution_to_proto() /
   │  cpu_mip_solution_t ◄────│ map_mip_solution_to_proto()
```

## Background Threads

### Result Retrieval Thread

This thread handles both job dispatch and result retrieval:

**Job dispatch** (first scan):
- Scans `job_queue` for claimed jobs with `data_sent == false`
- Sends job data to the worker's pipe (unary or chunked)
- Marks `data_sent = true` on success

**Result retrieval** (second scan):
- Monitors `result_queue` for completed jobs
- Reads streamed result data from worker pipes
- Updates `job_tracker` with results
- Notifies waiting clients (via condition variable)

### Incumbent Retrieval Thread

- Monitors incumbent pipes from all workers
- Parses `Incumbent` protobuf messages
- Stores in `job_tracker[job_id].incumbents`
- Enables `GetIncumbents` RPC to return data

### Worker Monitor Thread

- Detects crashed workers (via `waitpid`)
- Marks affected jobs as FAILED
- Can respawn workers (optional)

### Session Reaper Thread

- Runs every 60 seconds
- Removes stale chunked upload and download sessions after 300 seconds of inactivity
- Prevents memory leaks from abandoned upload/download sessions

## Log Streaming

Workers write logs to per-job files:

```text
/tmp/cuopt_logs/job_<job_id>.log
```

The `StreamLogs` RPC:
1. Opens the log file
2. Reads and sends new content periodically
3. Closes when job completes

## Job States

```text
┌─────────┐  submit   ┌───────────┐  claim   ┌────────────┐
│ QUEUED  │──────────►│ PROCESSING│─────────►│ COMPLETED  │
└─────────┘           └───────────┘          └────────────┘
     │                      │
     │ cancel               │ error
     ▼                      ▼
┌───────────┐          ┌─────────┐
│ CANCELLED │          │ FAILED  │
└───────────┘          └─────────┘
```

## Configuration Options

```bash
cuopt_grpc_server [options]

  -p, --port PORT              gRPC listen port (default: 5001)
  -w, --workers NUM            Number of worker processes (default: 1)
      --max-message-mb N       Max gRPC message size in MiB (default: 256; clamped to [4 KiB, ~2 GiB])
      --max-message-bytes N    Max gRPC message size in bytes (exact; min 4096)
      --chunk-timeout N        Per-chunk timeout in seconds for streaming (0=disabled, default: 60)
      --log-to-console         Echo solver logs to server console
  -v, --verbose                Increase verbosity (default: on)
  -q, --quiet                  Reduce verbosity (verbose is the default)
      --server-log PATH        Path to server operational log file (in addition to console)

TLS Options:
      --tls                    Enable TLS encryption
      --tls-cert PATH          Server certificate (PEM)
      --tls-key PATH           Server private key (PEM)
      --tls-root PATH          Root CA certificate (for client verification)
      --require-client-cert    Require client certificate (mTLS)
```

### NVIDIA cuOpt container image

When you use the official NVIDIA cuOpt container **without** an explicit command, the entrypoint chooses between the Python REST server and `cuopt_grpc_server`. User-facing Docker and client configuration is documented in `docs/cuopt/source/cuopt-grpc/advanced.rst` in this repository (the published **Advanced configuration** page).

When **`CUOPT_SERVER_TYPE=grpc`**, the entrypoint maps:

| Variable | Role |
|----------|------|
| `CUOPT_SERVER_PORT` | Passed as `--port` (default `5001`). |
| `CUOPT_GPU_COUNT` | When set, passed as `--workers`. When unset, `--workers` is omitted and the server uses its default worker count. |
| `CUOPT_GRPC_ARGS` | Optional whitespace-separated **extra** `cuopt_grpc_server` flags (TLS, message limits, logging, and so on). Each token becomes one argv word; embedded spaces inside a single flag value are not supported through this variable—invoke `cuopt_grpc_server` directly if you need complex quoting. |

Any flag listed in *Configuration options* above can be supplied on the host CLI or inside `CUOPT_GRPC_ARGS`.

## Fault Tolerance

### Worker Crashes

If a worker process crashes:
1. Monitor thread detects via `waitpid(WNOHANG)`
2. Any jobs the worker was processing are marked as FAILED
3. A replacement worker is automatically spawned (unless shutting down)
4. Other workers continue operating unaffected

### Graceful Shutdown

On SIGINT/SIGTERM:
1. Set `shm_ctrl->shutdown_requested = true`
2. Workers finish current job and exit
3. Main process waits for workers
4. Cleanup shared memory segments

### Job Cancellation

When `CancelJob` is called:
1. Set `job_queue[slot].cancelled = true`
2. If the job is **queued** (no worker yet): the worker checks the flag before starting and skips to the next job
3. If the job is **running** (worker has claimed it): the worker process is killed with `SIGKILL`, the worker-monitor thread detects the exit and posts a `RESULT_CANCELLED` status, and a replacement worker is spawned automatically

## Memory Management

| Resource | Location | Cleanup |
|----------|----------|---------|
| Job queue entries | Shared memory | Reused after completion |
| Result queue entries | Shared memory | Reused after retrieval |
| Problem data | Pipe (transient) | Consumed by worker |
| Chunked upload state | Main process memory | After `FinishChunkedUpload` (forwarded to worker) |
| Result data | `job_tracker` map | `DeleteResult` RPC |
| Log files | `/tmp/cuopt_logs/` | `DeleteResult` RPC |

## Performance Considerations

### Worker Count

- Each worker needs a GPU (or shares with others)
- Too many workers: GPU memory contention
- Too few workers: Underutilized when jobs queue
- Recommendation: 1 worker per GPU. Higher values are possible depending on the problems being solved but there is no specific guidance at this time

### Pipe Buffering

- Pipe buffer size is set to 1 MiB via `fcntl(F_SETPIPE_SZ)` (Linux default is 64 KiB)
- Large results block worker until main process reads
- Result retrieval thread should read promptly
- Deadlock prevention: Set `result.ready = true` BEFORE writing pipe

### Shared Memory Limits

- `MAX_JOBS = 100`: Maximum concurrent queued jobs
- `MAX_RESULTS = 100`: Maximum stored results
- Increase if needed for high-throughput scenarios

## File Locations

### POSIX Shared Memory

These names are passed to `shm_open()` and live under `/dev/shm/` (a kernel tmpfs), not on the regular filesystem. Writable on virtually all Linux systems and standard container runtimes.

| Name | Purpose |
|------|---------|
| `/cuopt_job_queue` | Job metadata (slots, flags, job IDs) |
| `/cuopt_result_queue` | Result metadata (status, error messages) |
| `/cuopt_control` | Server control (shutdown flag, worker count) |

### Filesystem

| Path | Purpose |
|------|---------|
| `/tmp/cuopt_logs/` | Per-job solver log files |

The log directory is hardcoded. `ensure_log_dir_exists()` calls `mkdir()` but does not check the return value — if the process lacks write permission on `/tmp`, log file creation will silently fail.

Chunked upload state is held in memory in the main process (no upload directory).
