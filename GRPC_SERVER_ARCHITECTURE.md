# Server Architecture

## Overview

The cuOpt gRPC server (`cuopt_grpc_server`) is a multi-process architecture designed for:
- **Isolation**: Each solve runs in a separate worker process for fault tolerance
- **Parallelism**: Multiple workers can process jobs concurrently
- **Large Payloads**: Handles multi-GB problems and solutions
- **Real-Time Feedback**: Log streaming and incumbent callbacks during solve

For gRPC protocol and client API, see `GRPC_INTERFACE.md`. Server source files live under `cpp/src/grpc/server/`.

## Process Model

```text
┌────────────────────────────────────────────────────────────────────┐
│                        Main Server Process                          │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │  gRPC       │  │  Job         │  │  Background Threads         │ │
│  │  Service    │  │  Tracker     │  │  - Result retrieval         │ │
│  │  Handler    │  │  (job status,│  │  - Incumbent retrieval      │ │
│  │             │  │   results)   │  │  - Worker monitor           │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────────┘ │
│         │                                        ▲                   │
│         │ shared memory                         │ pipes              │
│         ▼                                        │                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Shared Memory Queues                          ││
│  │   ┌─────────────────┐        ┌─────────────────────┐            ││
│  │   │  Job Queue      │        │  Result Queue       │            ││
│  │   │  (MAX_JOBS=100) │        │  (MAX_RESULTS=100)  │            ││
│  │   └─────────────────┘        └─────────────────────┘            ││
│  └─────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
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
  int to_worker_fd;               // Main → Worker: job data (server writes)
  int from_worker_fd;             // Worker → Main: result data (server reads)
  int worker_read_fd;             // Worker end of input pipe (worker reads)
  int worker_write_fd;            // Worker end of output pipe (worker writes)
  int incumbent_from_worker_fd;   // Worker → Main: incumbent solutions (server reads)
  int worker_incumbent_write_fd;  // Worker end of incumbent pipe (worker writes)
};
```

**Why pipes instead of shared memory for data?**
- Pipes handle backpressure naturally (blocking writes)
- No need to manage large shared memory segments
- Works well with streaming uploads (data flows through)

### Source File Roles

All paths below are under `cpp/src/grpc/server/`.

| File | Role |
|------|------|
| `grpc_server_main.cpp` | `main()`, `print_usage()`, argument parsing, shared-memory init, gRPC server run/stop. |
| `grpc_service_impl.cpp` | `CuOptRemoteServiceImpl`: all 14 RPC handlers (SubmitJob, CheckStatus, GetResult, chunked upload/download, StreamLogs, GetIncumbents, CancelJob, DeleteResult, WaitForCompletion, Status probe). Uses mappers and job_management to enqueue jobs and trigger pipe I/O. |
| `grpc_server_types.hpp` | Shared structs (e.g. `JobQueueEntry`, `ResultQueueEntry`, `ServerConfig`, `JobInfo`), enums, globals (atomics, mutexes, condition variables), and forward declarations used across server .cpp files. |
| `grpc_field_element_size.hpp` | Maps `cuopt::remote::ArrayFieldId` to element byte size; used by pipe deserialization and chunked logic. |
| `grpc_pipe_serialization.hpp` | Streaming pipe I/O: write/read individual length-prefixed protobuf messages (ChunkedProblemHeader, ChunkedResultHeader, ArrayChunk) directly to/from pipe fds. Avoids large intermediate buffers. Also serializes SubmitJobRequest for unary pipe transfer. |
| `grpc_incumbent_proto.hpp` | Build `Incumbent` proto from (job_id, objective, assignment) and parse it back; used by worker when pushing incumbents and by main when reading from the incumbent pipe. |
| `grpc_worker.cpp` | `worker_process(worker_index)`: loop over job queue, receive job data via pipe (unary or chunked), call solver, send result (and optionally incumbents) back. Contains `IncumbentPipeCallback` and `store_simple_result`. |
| `grpc_worker_infra.cpp` | Pipe creation/teardown, `spawn_worker` / `spawn_workers`, `wait_for_workers`, `mark_worker_jobs_failed`, `cleanup_shared_memory`. |
| `grpc_server_threads.cpp` | `worker_monitor_thread`, `result_retrieval_thread`, `incumbent_retrieval_thread`, `session_reaper_thread`. |
| `grpc_job_management.cpp` | Low-level pipe read/write, `send_job_data_pipe` / `recv_job_data_pipe`, `submit_job_async`, `check_job_status`, `cancel_job`, `generate_job_id`, log-dir helpers. |

### Large Payload Handling

For large problems uploaded via chunked gRPC RPCs:

1. Server holds chunked upload state in memory (`ChunkedUploadState`: header + array chunks per `upload_id`).
2. When `FinishChunkedUpload` is called, the header and chunks are stored in `pending_chunked_data`. The data dispatch thread streams them directly to the worker pipe as individual length-prefixed protobuf messages — no intermediate blob is created.
3. Worker reads the streamed messages from the pipe, reassembles arrays, runs the solver, and writes the result (and optionally incumbents) back via pipes using the same streaming format.
4. Main process result-retrieval thread reads the streamed result messages from the pipe and stores the result for `GetResult` or chunked download.

This streaming approach avoids creating a single large buffer, eliminating the 2 GiB protobuf serialization limit for pipe transfers and reducing peak memory usage. Each individual protobuf message (max 64 MiB) is serialized with standard `SerializeToArray`/`ParseFromArray`.

No disk spooling: chunked data is kept in memory in the main process until forwarded to the worker.

## Job Lifecycle

### 1. Submission

```text
Client                     Server                      Worker
   │                          │                           │
   │─── SubmitJob ──────────►│                           │
   │                          │ Create job entry          │
   │                          │ Store problem data        │
   │                          │ job_queue[slot].ready=true│
   │◄── job_id ──────────────│                           │
```

### 2. Processing

```text
Client                     Server                      Worker
   │                          │                           │
   │                          │                           │ Poll job_queue
   │                          │                           │ Claim job (CAS)
   │                          │◄─────────────────────────│ Read problem via pipe
   │                          │                           │
   │                          │                           │ Convert CPU→GPU
   │                          │                           │ solve_lp/solve_mip
   │                          │                           │ Convert GPU→CPU
   │                          │                           │
   │                          │ result_queue[slot].ready │◄──────────────────
   │                          │◄── result data via pipe ─│
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

- Monitors `result_queue` for completed jobs
- Reads result data from worker pipes
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

  -p, --port PORT              gRPC listen port (default: 8765)
  -w, --workers NUM            Number of worker processes (default: 1)
      --max-message-mb N       Max gRPC message size in MiB (default: 256; clamped to [4 KiB, ~2 GiB])
      --max-message-bytes N    Max gRPC message size in bytes (exact; min 4096)
      --enable-transfer-hash   Log data hashes for streaming transfers (for testing)
      --log-to-console         Echo solver logs to server console
  -q, --quiet                  Reduce verbosity (verbose is the default)

TLS Options:
      --tls                    Enable TLS encryption
      --tls-cert PATH          Server certificate (PEM)
      --tls-key PATH           Server private key (PEM)
      --tls-root PATH          Root CA certificate (for client verification)
      --require-client-cert    Require client certificate (mTLS)
```

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
2. Worker checks the flag before starting the solve
3. If cancelled, worker stores CANCELLED result and skips to the next job
4. If the solve has already started, it runs to completion (no mid-solve cancellation)

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
- Recommendation: 1-2 workers per GPU

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

| Path | Purpose |
|------|---------|
| `/tmp/cuopt_logs/` | Per-job solver log files |
| `/cuopt_job_queue` | Shared memory (job metadata) |
| `/cuopt_result_queue` | Shared memory (result metadata) |
| `/cuopt_control` | Shared memory (server control) |

Chunked upload state is held in memory in the main process (no upload directory).
