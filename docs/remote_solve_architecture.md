# cuOpt Remote Solve Architecture

## Document Purpose

This document describes the client-server architecture for cuOpt's remote solve capability. It is intended for security review and covers communication protocols, process architecture, data flow, and trust boundaries.

---

## 1. System Overview

The remote solve feature allows clients to submit optimization problems (LP/MIP) to a server for execution on GPU-accelerated hardware. The architecture supports both synchronous (blocking) and asynchronous (job-based) operation modes.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT PROCESS                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────────┐   │
│  │  cuOpt Library  │───▶│ Remote Serializer│───▶│   TCP Socket Client   │   │
│  │  (User Code)    │    │ (Protobuf/Custom)│    │                       │   │
│  └─────────────────┘    └──────────────────┘    └───────────┬───────────┘   │
└──────────────────────────────────────────────────────────────┼──────────────┘
                                                               │
                                              TCP Connection   │ Port 9090
                                              (Binary Protocol)│
                                                               ▼
┌──────────────────────────────────────────────────────────────┼──────────────┐
│                            SERVER PROCESS                    │              │
│  ┌───────────────────────────────────────────────────────────┴───────────┐  │
│  │                        Main Server Thread                             │  │
│  │  - Accept connections (thread-per-connection)                         │  │
│  │  - Parse requests via pluggable serializer                            │  │
│  │  - Route to sync handler or async job queue                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │ Result Retrieval  │  │  Worker Monitor   │  │  Connection       │        │
│  │     Thread        │  │     Thread        │  │  Handler Threads  │        │
│  └─────────┬─────────┘  └─────────┬─────────┘  └───────────────────┘        │
│            │                      │                                         │
│            │     POSIX Shared Memory (Job Queue, Result Queue)              │
│            │                      │                                         │
│  ┌─────────┴──────────────────────┴─────────────────────────────────────┐   │
│  │                        Shared Memory Region                          │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │   │
│  │  │   Job Queue     │  │  Result Queue    │  │  Control Block      │  │   │
│  │  │  (MAX_JOBS=64)  │  │ (MAX_RESULTS=64) │  │ (shutdown flag)     │  │   │
│  │  └─────────────────┘  └──────────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │   Worker Process  │  │   Worker Process  │  │   Worker Process  │        │
│  │      (fork)       │  │      (fork)       │  │      (fork)       │        │
│  │  - GPU Solver     │  │  - GPU Solver     │  │  - GPU Solver     │        │
│  │  - Isolated       │  │  - Isolated       │  │  - Isolated       │        │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Components

### 2.1 Client Components

| Component | Description |
|-----------|-------------|
| **cuOpt Library** | User-facing API (`solve_lp_remote`, `solve_mip_remote`) |
| **Remote Serializer** | Pluggable serialization (default: Protocol Buffers) |
| **TCP Client** | Socket connection to server, length-prefixed messages |

### 2.2 Server Components

| Component | Description |
|-----------|-------------|
| **Main Thread** | Accepts TCP connections, spawns handler threads |
| **Connection Handlers** | Per-connection threads that parse and route requests |
| **Result Retrieval Thread** | Polls shared memory for completed results |
| **Worker Monitor Thread** | Monitors worker processes via `waitpid()`, restarts dead workers |
| **Worker Processes** | Forked processes that execute GPU solves |
| **Shared Memory** | POSIX shared memory for IPC between main process and workers |

---

## 3. Communication Protocol

### 3.1 Transport Layer

- **Protocol**: TCP/IP
- **Default Port**: 9090 (configurable via `-p` flag)
- **Encryption**: None (plaintext) - **Security Note: TLS not implemented**
- **Authentication**: None - **Security Note: No auth mechanism**

### 3.2 Message Format

All messages use a simple length-prefixed binary format:

```
┌────────────────┬─────────────────────────────────┐
│  Length (4B)   │         Payload (N bytes)       │
│  Little-endian │    Serialized Protobuf/Custom   │
└────────────────┴─────────────────────────────────┘
```

### 3.3 Request Types

| Code | Type | Description |
|------|------|-------------|
| 0 | `SUBMIT_JOB` | Submit optimization problem, receive job_id |
| 1 | `CHECK_STATUS` | Query job status by job_id |
| 2 | `GET_RESULT` | Retrieve solution for completed job |
| 3 | `DELETE_RESULT` | Delete job and free resources |
| 4 | `GET_LOGS` | Retrieve solver output logs |
| 5 | `CANCEL_JOB` | Cancel queued or running job |
| 6 | `WAIT_FOR_RESULT` | Block until job completes (no polling) |

### 3.4 Job Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 0 | `QUEUED` | Job submitted, waiting for worker |
| 1 | `PROCESSING` | Worker is solving the problem |
| 2 | `COMPLETED` | Solution available |
| 3 | `FAILED` | Solve failed with error |
| 4 | `NOT_FOUND` | Job ID does not exist |
| 5 | `CANCELLED` | Job was cancelled by user |

---

## 4. Sequence Diagrams

### 4.1 Asynchronous Job Flow (Normal Case)

```
Client                    Server (Main)              Worker Process
  │                           │                           │
  │  SUBMIT_JOB (problem)     │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │                           │ Generate job_id           │
  │                           │ Write to Job Queue (shm)  │
  │                           │                           │
  │  Response (job_id)        │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │                           │      Poll Job Queue       │
  │                           │◀──────────────────────────│
  │                           │                           │
  │                           │      Claim Job            │
  │                           │       (set claimed=true,  │
  │                           │        worker_pid)        │
  │                           │──────────────────────────▶│
  │                           │                           │
  │  CHECK_STATUS (job_id)    │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │  Response (PROCESSING)    │      Execute GPU Solve    │
  │◀──────────────────────────│                           │
  │                           │                           │
  │         ...               │         ...               │
  │                           │                           │
  │                           │      Write Result (shm)   │
  │                           │◀──────────────────────────│
  │                           │                           │
  │                           │ Result Retrieval Thread   │
  │                           │ updates job_tracker       │
  │                           │                           │
  │  CHECK_STATUS (job_id)    │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │  Response (COMPLETED)     │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  GET_RESULT (job_id)      │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │  Response (solution)      │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  DELETE_RESULT (job_id)   │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │  Response (OK)            │                           │
  │◀──────────────────────────│                           │
```

### 4.2 WAIT_FOR_RESULT Flow (Blocking Wait)

```
Client                    Server (Main)              Worker Process
  │                           │                           │
  │  SUBMIT_JOB (problem)     │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │  Response (job_id)        │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  WAIT_FOR_RESULT (job_id) │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │                           │ Handler thread creates    │
  │                           │ JobWaiter with CV         │
  │                           │                           │
  │    (connection held       │ Thread blocks on          │
  │     open, no response     │ condition_variable.wait() │
  │     yet)                  │                           │
  │                           │                           │
  │                           │      Execute GPU Solve    │
  │                           │◀─────────────────────────▶│
  │                           │                           │
  │                           │      Write Result (shm)   │
  │                           │◀──────────────────────────│
  │                           │                           │
  │                           │ Result thread signals     │
  │                           │ condition_variable        │
  │                           │                           │
  │                           │ Handler thread wakes      │
  │                           │                           │
  │  Response (solution)      │                           │
  │◀──────────────────────────│                           │
```

### 4.3 Job Cancellation Flow

```
Client A                  Server (Main)              Worker Process
  │                           │                           │
  │  SUBMIT_JOB               │                           │
  │──────────────────────────▶│                           │
  │  Response (job_id)        │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │                           │      Worker claims job    │
  │                           │◀──────────────────────────│
  │                           │                           │
Client B                      │      Solving...           │
  │                           │                           │
  │  CANCEL_JOB (job_id)      │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │                           │ kill(worker_pid, SIGKILL) │
  │                           │──────────────────────────▶│
  │                           │                           │
  │                           │      Worker dies          │
  │                           │                           ✗
  │                           │                           │
  │  Response (CANCELLED)     │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │                           │ Monitor thread detects    │
  │                           │ dead worker via waitpid() │
  │                           │                           │
  │                           │ Restart worker (fork)     │
  │                           │                           │
  │                           │      New Worker           │
  │                           │◀──────────────────────────│
```

### 4.4 Worker Crash Recovery

```
Server (Main)                         Worker Process
     │                                      │
     │           Worker processing job      │
     │◀────────────────────────────────────▶│
     │                                      │
     │              CRASH/SEGFAULT          │
     │                                      ✗
     │                                      │
     │  Worker Monitor Thread               │
     │  waitpid() returns                   │
     │                                      │
     │  Mark job as FAILED                  │
     │  Signal any waiting threads          │
     │                                      │
     │  fork() new worker                   │
     │──────────────────────────────────────▶
     │                                      │
     │           New Worker Ready           │
     │◀─────────────────────────────────────│
```

---

## 5. Shared Memory Architecture

### 5.1 Memory Regions

Three POSIX shared memory segments are created:

| Name | Size | Purpose |
|------|------|---------|
| `/cuopt_job_queue` | ~64MB | Pending job entries |
| `/cuopt_result_queue` | ~64MB | Completed job results |
| `/cuopt_control` | ~64B | Shutdown flag, control signals |

### 5.2 Job Queue Entry Structure

```cpp
struct JobQueueEntry {
    char job_id[32];              // Unique job identifier
    uint32_t problem_type;        // 0=LP, 1=MIP
    uint32_t data_size;           // Size of serialized problem
    uint8_t data[MAX_JOB_DATA];   // Serialized problem data (~1MB)
    pid_t worker_pid;             // PID of worker processing this job
    std::atomic<bool> ready;      // Job ready for processing
    std::atomic<bool> claimed;    // Job claimed by a worker
    std::atomic<bool> cancelled;  // Job cancelled by user
};
```

### 5.3 Result Queue Entry Structure

```cpp
struct ResultQueueEntry {
    char job_id[32];              // Job identifier
    uint32_t status;              // 0=success, 1=error, 2=cancelled
    uint32_t data_size;           // Size of result data
    uint8_t data[MAX_RESULT_DATA]; // Serialized solution (~1MB)
    char error_message[256];      // Error message if failed
    std::atomic<bool> ready;      // Result ready for retrieval
    std::atomic<bool> retrieved;  // Result has been retrieved
};
```

---

## 6. Process Model

### 6.1 Process Hierarchy

```
cuopt_remote_server (main process)
├── Result Retrieval Thread
├── Worker Monitor Thread
├── Connection Handler Thread (per client)
├── Connection Handler Thread (per client)
│   ...
├── Worker Process 0 (forked)
├── Worker Process 1 (forked)
│   ...
└── Worker Process N (forked)
```

### 6.2 Worker Isolation

- Each worker is a separate process created via `fork()`
- Workers have independent memory spaces (except shared memory regions)
- GPU context is isolated per worker process
- Worker crash does not affect main server or other workers
- Workers are automatically restarted by monitor thread

### 6.3 Resource Limits

| Resource | Current Limit | Notes |
|----------|---------------|-------|
| Max concurrent jobs | 64 (`MAX_JOBS`) | Compile-time constant |
| Max job data size | ~1MB (`MAX_JOB_DATA_SIZE`) | Per job |
| Max result data size | ~1MB (`MAX_RESULT_DATA_SIZE`) | Per job |
| Max workers | Configurable (`-w` flag) | Default: 1 |
| Connection timeout | None | **Security Note** |

---

## 7. Security Considerations

### 7.1 Current Security Posture

| Aspect | Status | Risk Level |
|--------|--------|------------|
| Transport encryption (TLS) | Not implemented | **HIGH** |
| Client authentication | Not implemented | **HIGH** |
| Authorization | Not implemented | **MEDIUM** |
| Input validation | Basic size checks | **MEDIUM** |
| Rate limiting | Not implemented | **MEDIUM** |
| Connection limits | Not implemented | **MEDIUM** |
| Resource quotas | Not implemented | **LOW** |

### 7.2 Trust Boundaries

```
┌──────────────────────────────────────────────────────────────────────┐
│                         UNTRUSTED ZONE                               │
│                                                                      │
│   ┌─────────────┐                                                    │
│   │   Client    │  Network boundary (no encryption, no auth)         │
│   └──────┬──────┘                                                    │
│          │                                                           │
└──────────┼───────────────────────────────────────────────────────────┘
           │ TCP Port 9090
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         TRUSTED ZONE                                 │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                    Server Process                            │   │
│   │  - All clients treated equally                               │   │
│   │  - No per-user isolation                                     │   │
│   │  - Shared job queue visible to all workers                   │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                    Worker Processes                          │   │
│   │  - Process-level isolation from each other                   │   │
│   │  - No sandboxing (full system access)                        │   │
│   │  - GPU access                                                │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.3 Attack Surface

| Attack Vector | Description | Mitigation |
|---------------|-------------|------------|
| Network eavesdropping | Plaintext TCP traffic | Requires TLS implementation |
| Unauthorized access | No authentication | Requires auth mechanism |
| Denial of service | No rate limiting, connection limits | Requires implementation |
| Malformed input | Invalid protobuf/msgpack | Protobuf parsing with error handling |
| Resource exhaustion | Large jobs, many connections | Size limits, but no connection limits |
| Job ID guessing | Sequential-ish IDs | Uses random hex (128-bit) |
| Worker escape | Malicious solver code | Workers are forked, not sandboxed |

### 7.4 Recommended Security Enhancements

1. **TLS/SSL**: Add transport encryption for all client-server communication
2. **Authentication**: Implement API key or certificate-based authentication
3. **Authorization**: Per-user job isolation, prevent access to other users' jobs
4. **Rate Limiting**: Limit requests per client per time window
5. **Connection Limits**: Maximum concurrent connections per IP
6. **Input Validation**: Deep validation of problem data before solving
7. **Audit Logging**: Log all operations with client identity
8. **Sandboxing**: Consider containerization or seccomp for workers

---

## 8. Data Flow Summary

### 8.1 Problem Data Flow

```
Client Problem Data
        │
        ▼
┌───────────────────┐
│   Serialization   │  Client-side: optimization_problem_t → protobuf bytes
└─────────┬─────────┘
          │
          ▼ (TCP)
┌───────────────────┐
│   Job Queue       │  Stored in shared memory (~1MB max)
└─────────┬─────────┘
          │
          ▼ (shm read)
┌───────────────────┐
│ Worker Process    │  Deserialize, create GPU data structures
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   GPU Solver      │  PDLP or MIP solver execution
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Solution         │  GPU → Host copy, serialize to bytes
└─────────┬─────────┘
          │
          ▼ (shm write)
┌───────────────────┐
│   Result Queue    │  Stored in shared memory (~1MB max)
└─────────┬─────────┘
          │
          ▼ (TCP)
┌───────────────────┐
│   Client          │  Deserialize: protobuf bytes → solution_t
└───────────────────┘
```

---

## 9. Configuration

### 9.1 Server Command-Line Options

```
Usage: cuopt_remote_server [options]
Options:
  -p PORT       Port to listen on (default: 9090)
  -w WORKERS    Number of worker processes (default: 1)
  -v            Verbose logging
  -q            Quiet mode (minimal output)
  --no-logs     Disable log streaming feature
```

### 9.2 Environment Variables

| Variable | Description |
|----------|-------------|
| `CUOPT_REMOTE_USE_SYNC` | Force synchronous mode (bypass job queue) |

---

## 10. File Locations

| Path | Description |
|------|-------------|
| `/tmp/cuopt_logs/log_{job_id}` | Per-job solver log files |
| `/dev/shm/cuopt_job_queue` | Job queue shared memory |
| `/dev/shm/cuopt_result_queue` | Result queue shared memory |
| `/dev/shm/cuopt_control` | Control block shared memory |

---

## Appendix A: Protocol Buffer Schema

See `cpp/src/linear_programming/utilities/cuopt_remote.proto` for the complete schema definition.

Key messages:
- `AsyncRequest`: Wrapper for all request types
- `AsyncResponse`: Wrapper for all response types
- `OptimizationProblem`: LP/MIP problem definition
- `LPSolution` / `MIPSolution`: Solution data
- `PDLPSolverSettings` / `MIPSolverSettings`: Solver configuration

---

## Appendix B: Pluggable Serialization

The server supports custom serialization formats via a plugin interface:

```cpp
template <typename i_t, typename f_t>
class remote_serializer_t {
    // Serialize optimization problem
    virtual std::vector<uint8_t> serialize_lp_request(...) = 0;

    // Deserialize solution
    virtual lp_solution_t<i_t, f_t> deserialize_lp_solution(...) = 0;

    // ... additional methods for async protocol
};
```

Built-in serializers:
- **Protobuf** (default): High performance, schema-based
- **MsgPack**: Lightweight, schema-less alternative

---

*Document Version: 1.0*
*Last Updated: January 2026*
