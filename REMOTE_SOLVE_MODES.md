# cuOpt Remote Solve Modes and Protocols

This document describes the different operating modes, log retrieval methods, and APIs available for cuOpt remote solving.

---

## Table of Contents

1. [Operating Modes](#1-operating-modes)
2. [Supported Interfaces](#2-supported-interfaces)
3. [Log Retrieval Methods](#3-log-retrieval-methods)
4. [API Endpoints](#4-api-endpoints)
5. [WAIT_FOR_RESULT API](#5-wait_for_result-api)
6. [Detection and Configuration](#6-detection-and-configuration)
7. [Job Status States](#7-job-status-states)
8. [Python APIs](#8-python-apis)
9. [Workflow Comparisons](#9-workflow-comparisons)
10. [Best Practices](#10-best-practices)

---

## 1. Operating Modes

### 1.1 Sync Mode (`CUOPT_REMOTE_USE_SYNC=1`)

**Behavior**: Client sends request and **blocks** until completion

**Architecture**:
- Job still goes through server queue and worker process
- Server uses condition variable to block connection until job completes
- Returns result directly when complete

**Log Streaming**: Real-time log streaming during solve

**Use Case**: Interactive/development use where you want immediate feedback

**Workflow**:
```
Client → SUBMIT_JOB (blocking=true) → Server blocks → Worker solves → Server returns result
```

**Example**:
```bash
CUOPT_REMOTE_HOST=localhost \
CUOPT_REMOTE_PORT=8765 \
CUOPT_REMOTE_USE_SYNC=1 \
  cuopt_cli --log-to-console 1 problem.mps
```

---

### 1.2 Async Mode (default)

**Behavior**: Client submits job, receives `job_id` immediately, non-blocking

**Architecture**:
- Client polls for status (QUEUED → PROCESSING → COMPLETED/FAILED)
- Client retrieves logs incrementally using byte offset
- Client retrieves result when ready
- Client deletes job to free server memory

**Log Retrieval**: Incremental polling via `GET_LOGS` API

**Use Case**: Batch processing, long-running jobs, multiple concurrent jobs

**Workflow**:
```
Client → SUBMIT_JOB → job_id
     ↓ (poll)
     → CHECK_STATUS (returns QUEUED/PROCESSING/COMPLETED/FAILED)
     → GET_LOGS (frombyte offset, returns new log lines)
     → GET_RESULT (when COMPLETED)
     → DELETE_RESULT (cleanup)
```

**Example**:
```bash
CUOPT_REMOTE_HOST=localhost \
CUOPT_REMOTE_PORT=8765 \
  cuopt_cli problem.mps
```

---

### 1.3 Hybrid Mode (Async Submit + WAIT_FOR_RESULT)

**Behavior**: Submit async to get `job_id`, then block on wait

**Architecture**:
- Non-blocking submission returns `job_id` immediately
- `WAIT_FOR_RESULT` API blocks until completion
- Can stream logs in parallel thread while waiting

**Log Retrieval**: Parallel thread polls `GET_LOGS` while main thread waits

**Use Case**: Interactive use with job management (cancellation, log streaming)

**Workflow**:
```
Client → SUBMIT_JOB (blocking=false) → job_id
     ↓
[Thread 1] WAIT_FOR_RESULT(job_id)  [BLOCKS until complete]
[Thread 2] while running: GET_LOGS(job_id, frombyte)
     ↓
result returned automatically
DELETE_RESULT(job_id)
```

**Example**: See [Section 5.3](#53-python-usage-example) for Python code

---

## 2. Supported Interfaces

| Interface | Sync Mode | Async Mode | WAIT_FOR_RESULT |
|-----------|-----------|------------|-----------------|
| **C++ API** (`solve_lp`, `solve_mip`) | ✓ | ✓ | ✓ (internal) |
| **Python API** (`Solve()`, `Problem.solve()`) | ✓ | ✓ | ❌ (not wrapped) |
| **cuopt_cli** | ✓ | ✓ | ❌ (not exposed) |
| **C API** | ✓ | ✓ | ❌ (not exposed) |
| **Python `cancel_job()`** | - | ✓ | ✓ |
| **Low-level Protobuf** | ✓ | ✓ | ✓ |

**Note**: All high-level interfaces are **transparent** - they automatically detect remote solve via environment variables and handle the full async polling loop internally.

---

## 3. Log Retrieval Methods

### 3.1 Sync Mode Logging

**Method**: Real-time streaming

**How it works**:
- Server captures stdout from worker process
- Streams log data to client in real-time over TCP connection
- Logs printed to console as solver runs

**Control**: Set `log_to_console=1` in solver settings

**Example**:
```bash
CUOPT_REMOTE_HOST=localhost \
CUOPT_REMOTE_PORT=8765 \
CUOPT_REMOTE_USE_SYNC=1 \
  cuopt_cli --log-to-console 1 problem.mps
```

**Output**: Logs appear immediately as solver runs

---

### 3.2 Async Mode Logging

**Method**: Incremental polling via `GET_LOGS` API

**How it works**:
- Logs written to `/tmp/cuopt_logs/log_{job_id}` on server
- Client calls `GET_LOGS` with byte offset (`frombyte`)
- Server returns new log content from offset to current position
- Client updates offset for next poll

**Implementation** (C++ client):
```cpp
int64_t log_frombyte = 0;
while (true) {
  auto [job_exists, new_frombyte] = get_logs(host, port, job_id, log_frombyte);
  if (job_exists) {
    log_frombyte = new_frombyte;
  }
  // Check status...
  sleep(0.1);
}
```

**Python example** (low-level):
```python
req = pb.AsyncRequest()
req.request_type = pb.GET_LOGS
req.job_id = job_id
req.frombyte = frombyte

response = send_recv(req)
logs = response.logs_response
for line in logs.log_lines:
    print(line)
frombyte = logs.nbytes  # Update for next poll
```

---

### 3.3 WAIT_FOR_RESULT Logging

**Method**: Parallel thread polling `GET_LOGS` while main thread waits

**How it works**:
- Main thread blocks on `WAIT_FOR_RESULT`
- Separate thread continuously polls `GET_LOGS`
- Both threads access same job using `job_id`
- Log thread stops when main thread returns result

**Advantages**:
- Real-time log visibility
- No need to manually poll status
- Clean blocking semantics with parallel logging

See [Section 5.3](#53-python-usage-example) for complete code example.

---

## 4. API Endpoints

### 4.1 Protobuf Protocol

All communication uses length-prefixed Protocol Buffer messages:

```
[8-byte size (uint64_t)][serialized protobuf data]
```

### 4.2 Request Types

| Endpoint | Sync | Async | Hybrid | Returns |
|----------|------|-------|--------|---------|
| `SUBMIT_JOB` (blocking=false) | - | ✓ | ✓ | `job_id` |
| `SUBMIT_JOB` (blocking=true) | ✓ | - | - | Full result (blocks) |
| `CHECK_STATUS` | - | ✓ | ✓ | `JobStatus` enum |
| `GET_LOGS` | - | ✓ | ✓ | Log content + new offset |
| `GET_RESULT` | - | ✓ | ✓ | Serialized solution |
| `DELETE_RESULT` | - | ✓ | ✓ | Success status |
| `CANCEL_JOB` | - | ✓ | ✓ | Cancel status |
| `WAIT_FOR_RESULT` | - | - | ✓ | Serialized solution (blocks) |

### 4.3 Protobuf Enum Definition

File: `cpp/src/linear_programming/utilities/cuopt_remote.proto`

```protobuf
enum AsyncRequestType {
  SUBMIT_JOB = 0;        // Submit a new job
  CHECK_STATUS = 1;      // Check job status
  GET_RESULT = 2;        // Retrieve completed result
  DELETE_RESULT = 3;     // Delete result from server
  GET_LOGS = 4;          // Retrieve buffered log entries
  CANCEL_JOB = 5;        // Cancel a queued or running job
  WAIT_FOR_RESULT = 6;   // Block until job completes, returns result
}
```

---

## 5. WAIT_FOR_RESULT API

### 5.1 Overview

`WAIT_FOR_RESULT` is a **hybrid async/blocking mode** that combines the benefits of both sync and async modes:

- **Submit async** → Get `job_id` back immediately (non-blocking submission)
- **Wait on result** → Block until job completes (no polling loop needed)
- **Stream logs in parallel** → Another thread can poll `GET_LOGS` while waiting

**This is the best of both worlds for interactive use!**

### 5.2 Server Implementation

File: `cpp/cuopt_remote_server.cpp`

#### Core Function (lines 1539-1602)

```cpp
bool wait_for_result(const std::string& job_id,
                     std::vector<uint8_t>& result_data,
                     std::string& error_message)
{
  // First check if job already completed
  {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    auto it = job_tracker.find(job_id);

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

  // Wait on the condition variable - this thread will sleep until signaled
  {
    std::unique_lock<std::mutex> lock(waiter->mutex);
    waiter->cv.wait(lock, [&waiter] { return waiter->ready; });
  }

  // Wakes up when result_retrieval_thread signals the CV
  if (waiter->success) {
    result_data = std::move(waiter->result_data);
    return true;
  } else {
    error_message = waiter->error_message;
    return false;
  }
}
```

#### Synchronization Mechanism

**JobWaiter struct**:
```cpp
struct JobWaiter {
  std::mutex mutex;
  std::condition_variable cv;
  bool ready = false;
  bool success = false;
  std::vector<uint8_t> result_data;
  std::string error_message;
};
```

**Signaling in result_retrieval_thread** (line 1186-1198):
```cpp
// Check if there's a blocking waiter
{
  std::lock_guard<std::mutex> lock(waiters_mutex);
  auto wit = waiting_threads.find(job_id);
  if (wit != waiting_threads.end()) {
    // Wake up the waiting thread
    auto waiter = wit->second;
    waiter->result_data = std::move(result_data);
    waiter->error_message = error_message;
    waiter->success = success;
    waiter->ready = true;
    waiter->cv.notify_one();  // <-- WAKE UP!
  }
}
```

### 5.3 Python Usage Example

File: `test_wait_with_logs.py`

```python
import threading
import time
import cuopt_remote_pb2 as pb

# 1. Submit job (async) to get job_id
req = pb.AsyncRequest()
req.request_type = pb.SUBMIT_JOB
req.blocking = False  # Get job_id back immediately

# Set up problem...
lp = req.lp_request
lp.problem.c.extend([1.0] * n_vars)
# ... more problem setup ...

response = send_recv(req)
job_id = response.submit_response.job_id
print(f"Job ID: {job_id}")

# 2. Start log streaming thread
def log_streaming_thread(job_id, stop_event):
    frombyte = 0
    while not stop_event.is_set():
        # Poll GET_LOGS
        req = pb.AsyncRequest()
        req.request_type = pb.GET_LOGS
        req.job_id = job_id
        req.frombyte = frombyte

        response = send_recv(req)
        logs_resp = response.logs_response

        if logs_resp.log_lines:
            for line in logs_resp.log_lines:
                print(f"[LOG] {line}")
            frombyte = logs_resp.nbytes

        time.sleep(0.05)  # Small delay to avoid hammering server

stop_event = threading.Event()
log_thread = threading.Thread(target=log_streaming_thread, args=(job_id, stop_event))
log_thread.start()

# 3. Main thread: Wait for result (BLOCKS)
print("Calling WAIT_FOR_RESULT (blocking)...")
req = pb.AsyncRequest()
req.request_type = pb.WAIT_FOR_RESULT
req.job_id = job_id

start = time.time()
response = send_recv(req, timeout=120)
elapsed = time.time() - start

# 4. Stop log thread
stop_event.set()
log_thread.join()

# 5. Use result
result = response.result_response
if result.HasField('lp_solution'):
    sol = result.lp_solution
    print(f"Completed in {elapsed:.2f}s")
    print(f"Objective: {sol.primal_objective:.6f}")
else:
    print(f"Error: {result.error_message}")

# 6. Cleanup
del_req = pb.AsyncRequest()
del_req.request_type = pb.DELETE_RESULT
del_req.job_id = job_id
send_recv(del_req)
```

### 5.4 Key Benefits

| Feature | Sync Mode | Async Polling | **WAIT_FOR_RESULT** |
|---------|-----------|---------------|---------------------|
| Non-blocking submit | ❌ | ✓ | ✓ |
| Get job_id back | ❌ | ✓ | ✓ |
| Parallel log streaming | ❌ | ✓ | ✓ |
| No polling loop needed | ✓ | ❌ | ✓ |
| Connection efficiency | Poor | Good | Good |
| Can cancel job | ✓ | ✓ | ✓ |

**WAIT_FOR_RESULT combines all the best features!**

### 5.5 Important Notes

1. **No auto-delete**: `WAIT_FOR_RESULT` does NOT automatically delete the job after returning the result. This allows you to:
   - Retrieve remaining logs with `GET_LOGS` after completion
   - Call `DELETE_RESULT` when you're done with logs

2. **Connection held open**: The TCP connection remains open while waiting (can be many seconds/minutes for large problems)

3. **Same result format**: The response is identical to `GET_RESULT` (serialized solution)

4. **Thread-safe**: Multiple clients can wait on different jobs simultaneously

5. **Not yet wrapped**: `WAIT_FOR_RESULT` is currently only available via low-level Protobuf protocol. High-level Python/C++ wrappers would need to be added (similar to existing `cancel_job()` wrapper).

---

## 6. Detection and Configuration

### 6.1 Transparent Remote Solve Detection

The same code works for both local and remote:

```python
# Works locally if no env vars set, remotely if set
solution = solve_lp(data_model, settings)
```

**Detection logic** (in all interfaces):
```cpp
bool is_remote = (getenv("CUOPT_REMOTE_HOST") && getenv("CUOPT_REMOTE_PORT"));
bool sync_mode = (getenv("CUOPT_REMOTE_USE_SYNC") == "1");
```

### 6.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUOPT_REMOTE_HOST` | Server hostname/IP | (none - local solve) |
| `CUOPT_REMOTE_PORT` | Server port | (none - local solve) |
| `CUOPT_REMOTE_USE_SYNC` | Use sync mode if "1" | "0" (async) |
| `CUOPT_SERIALIZER_LIB` | Path to custom serializer | (uses protobuf) |

**Example**:
```bash
export CUOPT_REMOTE_HOST=gpu-server.example.com
export CUOPT_REMOTE_PORT=8765
export CUOPT_REMOTE_USE_SYNC=1  # Optional: force sync mode
```

---

## 7. Job Status States

```
QUEUED      (0) → Job waiting for available worker
PROCESSING  (1) → Worker is solving the problem
COMPLETED   (2) → Solve finished successfully
FAILED      (3) → Solve failed with error
NOT_FOUND   (4) → Job ID doesn't exist on server
CANCELLED   (5) → Job was cancelled by user
```

**State Transitions**:
```
QUEUED → PROCESSING → COMPLETED
                   → FAILED
                   → CANCELLED (via CANCEL_JOB)
```

---

## 8. Python APIs

### 8.1 Cancel Job (Available Now)

File: `python/cuopt/cuopt/linear_programming/remote.py`

```python
from cuopt.linear_programming import cancel_job, JobStatus

result = cancel_job("job_abc123", host="localhost", port=8765)
print(f"Success: {result.success}")
print(f"Status: {result.job_status}")  # JobStatus enum
print(f"Message: {result.message}")
```

**Returns**: `CancelResult` dataclass
- `success: bool` - Whether cancellation succeeded
- `message: str` - Status message
- `job_status: JobStatus` - Final job status

### 8.2 Future API (Not Yet Implemented)

To match the `cancel_job()` pattern, these could be added:

```python
# Submit async and get job_id
def submit_job_async(data_model, settings, host=None, port=None) -> str:
    """Submit job and return job_id immediately."""
    pass

# Block until complete
def wait_for_result(job_id: str, host=None, port=None) -> Solution:
    """Wait for job completion and return result (blocks)."""
    pass

# Poll for status
def get_job_status(job_id: str, host=None, port=None) -> JobStatus:
    """Check job status without blocking."""
    pass

# Get logs
def get_job_logs(job_id: str, frombyte: int = 0, host=None, port=None) -> tuple[list[str], int]:
    """Get logs from byte offset, returns (log_lines, new_offset)."""
    pass

# Cleanup
def delete_job(job_id: str, host=None, port=None) -> bool:
    """Delete job from server."""
    pass
```

---

## 9. Workflow Comparisons

### 9.1 Sync Mode Workflow

```
Client:
  connect()
  send(SUBMIT_JOB, blocking=true, problem_data)
  [BLOCKS - connection stays open]
  receive(solution)
  [Logs streamed in real-time during blocking]
  disconnect()
```

**Pros**:
- Simple - one request/response
- Real-time logs
- Blocking semantics

**Cons**:
- Connection held open entire time
- Can't cancel easily
- No job_id for tracking

---

### 9.2 Async Mode Workflow (Traditional Polling)

```
Client:
  # Submit
  job_id = submit_job(problem)

  # Poll until complete
  while True:
    status = check_status(job_id)
    if status == COMPLETED:
      break
    if status == FAILED:
      error()

    # Optionally get logs
    if verbose:
      logs = get_logs(job_id, frombyte)
      print(logs)
      frombyte = update_offset(logs)

    sleep(0.1)

  # Retrieve result
  solution = get_result(job_id)

  # Cleanup
  delete_job(job_id)
```

**Pros**:
- Non-blocking
- Can cancel anytime
- Job_id for tracking
- Multiple clients can monitor same job

**Cons**:
- Polling loop complexity
- Delayed log visibility (polling interval)
- More network requests

---

### 9.3 WAIT_FOR_RESULT Workflow (Best of Both)

```
Client:
  # Submit async
  job_id = submit_job(problem, blocking=false)

  # Start log thread
  Thread 1:
    while not done:
      logs = get_logs(job_id, frombyte)
      print(logs)
      frombyte = update_offset(logs)
      sleep(0.05)

  # Main thread: wait for result
  Thread 2 (main):
    solution = wait_for_result(job_id)  # BLOCKS

  # Stop log thread
  stop_log_thread()

  # Cleanup
  delete_job(job_id)
```

**Pros**:
- Non-blocking submit (get job_id)
- Real-time logs via parallel thread
- Blocking wait (no polling loop)
- Can cancel from another client
- Clean separation of concerns

**Cons**:
- Requires threading
- Slightly more complex than sync mode

---

## 10. Best Practices

### 10.1 Choosing a Mode

**Use Sync Mode when**:
- Interactive development/debugging
- Single-shot solves
- Simplicity is priority
- Don't need job_id

**Use Async Mode when**:
- Batch processing
- Long-running jobs
- Need to manage multiple jobs
- Want to cancel jobs
- Production systems

**Use WAIT_FOR_RESULT when**:
- Interactive use
- Want real-time logs
- Need job_id for tracking/cancellation
- Don't want polling complexity

### 10.2 Log Retrieval Best Practices

**For Sync Mode**:
```bash
# Always enable log_to_console for sync mode
CUOPT_REMOTE_USE_SYNC=1 cuopt_cli --log-to-console 1 problem.mps
```

**For Async Mode**:
```python
# Poll logs frequently for near-real-time visibility
frombyte = 0
while job_running:
    logs = get_logs(job_id, frombyte)
    frombyte = logs.nbytes
    time.sleep(0.05)  # 50ms polling interval
```

**For WAIT_FOR_RESULT**:
```python
# Use dedicated thread for logs
log_thread = threading.Thread(target=log_poller, args=(job_id,))
log_thread.daemon = True  # Exit when main thread exits
log_thread.start()

result = wait_for_result(job_id)
```

### 10.3 Error Handling

**Always check job status**:
```python
result = wait_for_result(job_id)
if result.status == pb.FAILED:
    print(f"Error: {result.error_message}")
    # Get final logs for debugging
    logs = get_logs(job_id, 0)
    print(logs)
```

**Handle worker crashes**:
```python
# Server automatically marks jobs as FAILED if worker dies
# Check status periodically
status = check_status(job_id)
if status == FAILED:
    # Retrieve error message
    result = get_result(job_id)  # Contains error_message
```

### 10.4 Resource Cleanup

**Always delete completed jobs**:
```python
try:
    result = wait_for_result(job_id)
    # ... use result ...
finally:
    delete_job(job_id)  # Free server memory
```

**For long-running servers**:
- Implement periodic cleanup of old completed jobs
- Monitor `/tmp/cuopt_logs` directory size
- Consider auto-deletion after N hours

---

## Appendix: Architecture Diagrams

### A.1 Sync Mode Flow

```
Client                    Server (Main)              Worker Process
  │                           │                           │
  │  SUBMIT_JOB (blocking=1)  │                           │
  │──────────────────────────▶│                           │
  │                           │  Add to job queue         │
  │                           │──────────────────────────▶│
  │                           │                           │
  │    (connection held       │                           │
  │     open, blocking)       │      Execute GPU Solve    │
  │                           │◀─────────────────────────▶│
  │                           │                           │
  │                           │  Logs streamed to client  │
  │◀─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│                           │
  │                           │                           │
  │                           │      Write Result         │
  │                           │◀──────────────────────────│
  │                           │                           │
  │  Response (solution)      │                           │
  │◀──────────────────────────│                           │
```

### A.2 Async Mode Flow

```
Client                    Server (Main)              Worker Process
  │                           │                           │
  │  SUBMIT_JOB (blocking=0)  │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │  Response (job_id)        │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  CHECK_STATUS (job_id)    │                           │
  │──────────────────────────▶│                           │
  │  Response (QUEUED)        │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │                           │      Execute GPU Solve    │
  │                           │◀─────────────────────────▶│
  │                           │                           │
  │  GET_LOGS (job_id, 0)     │                           │
  │──────────────────────────▶│                           │
  │  Response (logs 0-1000)   │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  CHECK_STATUS (job_id)    │                           │
  │──────────────────────────▶│                           │
  │  Response (PROCESSING)    │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │                           │      Write Result         │
  │                           │◀──────────────────────────│
  │                           │                           │
  │  CHECK_STATUS (job_id)    │                           │
  │──────────────────────────▶│                           │
  │  Response (COMPLETED)     │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  GET_RESULT (job_id)      │                           │
  │──────────────────────────▶│                           │
  │  Response (solution)      │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  DELETE_RESULT (job_id)   │                           │
  │──────────────────────────▶│                           │
  │  Response (SUCCESS)       │                           │
  │◀──────────────────────────│                           │
```

### A.3 WAIT_FOR_RESULT Flow

```
Client                    Server (Main)              Worker Process
  │                           │                           │
  │  SUBMIT_JOB (blocking=0)  │                           │
  │──────────────────────────▶│                           │
  │                           │                           │
  │  Response (job_id)        │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  ├─[Thread 1]────────────────┤                           │
  │  WAIT_FOR_RESULT (job_id) │                           │
  │──────────────────────────▶│                           │
  │                           │  Handler creates JobWaiter│
  │    (connection held       │  Thread blocks on CV      │
  │     open, no response)    │                           │
  │                           │                           │
  ├─[Thread 2]────────────────┤                           │
  │  GET_LOGS (job_id, 0)     │                           │
  │──────────────────────────▶│                           │
  │  Response (logs)          │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │  GET_LOGS (job_id, 1000)  │      Execute GPU Solve    │
  │──────────────────────────▶│◀─────────────────────────▶│
  │  Response (logs)          │                           │
  │◀──────────────────────────│                           │
  │                           │                           │
  │                           │      Write Result         │
  │                           │◀──────────────────────────│
  │                           │                           │
  │                           │  Result thread signals CV │
  │                           │  Handler wakes up         │
  │                           │                           │
  │  Response (solution)      │                           │
  │◀──────────────────────────│                           │
  └─[Thread 1 returns]        │                           │
  │                           │                           │
  └─[Thread 2 stops polling]  │                           │
```

---

## References

- **Implementation**: `cpp/cuopt_remote_server.cpp`
- **Client Logic**: `cpp/src/linear_programming/utilities/remote_solve.cu`
- **Protocol Definition**: `cpp/src/linear_programming/utilities/cuopt_remote.proto`
- **Python Wrappers**: `python/cuopt/cuopt/linear_programming/remote.py`
- **Architecture Doc**: `docs/remote_solve_architecture.md`
- **Developer Guide**: `docs/developer/REMOTE_SOLVE_GUIDE.md`
- **Test Example**: `test_wait_with_logs.py`
