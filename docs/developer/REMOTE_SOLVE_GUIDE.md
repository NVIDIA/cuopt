# cuOpt Remote Solve Guide

This guide covers the remote solve feature for cuOpt, enabling GPU-accelerated optimization from CPU-only client machines.

## Overview

The remote solve feature allows:
- **CPU-only clients** to solve LP/MIP problems using a GPU-equipped server
- **Async job management** for non-blocking operations
- **Pluggable serialization** (default: Protocol Buffers, also supports MsgPack)
- **Real-time log streaming** from solver to client
- **Worker process isolation** with automatic restart on failure

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT                                │
│  - Submits job → receives job ID                            │
│  - Polls for status (QUEUED/PROCESSING/COMPLETED/FAILED)    │
│  - Retrieves logs incrementally via GET_LOGS                │
│  - Retrieves result → gets solution                         │
│  - Deletes result → frees server memory                     │
└────────────────────┬────────────────────────────────────────┘
                     │ TCP/IP (Protobuf or custom serializer)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   ASYNC SERVER (Main Process)                │
│  - Accepts connections (default port 8765)                  │
│  - Handles: SUBMIT_JOB, CHECK_STATUS, GET_RESULT,           │
│             DELETE_RESULT, GET_LOGS                         │
│  - Shared memory job queue                                  │
│  - Worker monitor thread (restarts dead workers)            │
│  - Result retrieval thread                                  │
└──────┬──────────────────────────────────┬──────────────────┘
       │ Shared Memory                    │
       ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                WORKER PROCESS(es)                            │
│  - Reads jobs from shared memory queue                      │
│  - Logs to per-job file (/tmp/cuopt_logs/log_{job_id})     │
│  - Solves using GPU (cuOpt library)                        │
│  - Writes results to shared memory result queue            │
│  - Isolated - crash doesn't affect server                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Building

```bash
# Build with remote server support
./build.sh libcuopt cuopt_remote_server
```

### Running the Server

```bash
# Start server with 4 worker processes on port 8765
./cpp/build/cuopt_remote_server -p 8765 -w 4
```

Server options:
- `-p PORT` - Port number (default: 8765)
- `-w NUM` - Number of worker processes (default: 4)

### Client Configuration

Set environment variables to enable remote solve:

```bash
export CUOPT_REMOTE_HOST=server.example.com
export CUOPT_REMOTE_PORT=8765

# Optional: Use sync mode (blocking with log streaming)
export CUOPT_REMOTE_USE_SYNC=1
```

### Using cuopt_cli

```bash
# Remote solve (async mode by default)
CUOPT_REMOTE_HOST=localhost CUOPT_REMOTE_PORT=8765 \
  ./cpp/build/cuopt_cli problem.mps

# Remote solve with log streaming (sync mode)
CUOPT_REMOTE_HOST=localhost CUOPT_REMOTE_PORT=8765 \
CUOPT_REMOTE_USE_SYNC=1 \
  ./cpp/build/cuopt_cli --log-to-console 1 problem.mps
```

### Python Usage

```python
import os
os.environ['CUOPT_REMOTE_HOST'] = 'localhost'
os.environ['CUOPT_REMOTE_PORT'] = '8765'

from cuopt.linear_programming import DataModel, SolverSettings, solve_lp

# Create problem
dm = DataModel()
dm.set_csr_constraint_matrix(...)
# ... set up problem ...

# Solve remotely (transparent to user)
solution = solve_lp(dm, SolverSettings())
print(f"Objective: {solution.get_objective_value()}")
```

## Operating Modes

### Sync Mode (`CUOPT_REMOTE_USE_SYNC=1`)
- Client sends request and waits for result
- Server streams solver logs in real-time
- Best for interactive use

### Async Mode (default)
- Client submits job, receives job_id immediately
- Client polls for status and logs incrementally
- Client retrieves result when complete
- Best for batch processing, long-running jobs

## Job Lifecycle (Async Mode)

1. **SUBMIT_JOB** → Returns `job_id`
2. **CHECK_STATUS** → Returns QUEUED | PROCESSING | COMPLETED | FAILED
3. **GET_LOGS** → Returns log lines from `frombyte` offset
4. **GET_RESULT** → Returns serialized solution
5. **DELETE_RESULT** → Removes job from server

## Custom Serialization

The default serializer uses Protocol Buffers. You can provide a custom serializer:

```bash
# Set custom serializer library
export CUOPT_SERIALIZER_LIB=/path/to/libcustom_serializer.so

# Run server and client with same serializer
CUOPT_SERIALIZER_LIB=... ./cpp/build/cuopt_remote_server -p 8765 -w 2
CUOPT_SERIALIZER_LIB=... ./cpp/build/cuopt_cli problem.mps
```

See `docs/developer/SERIALIZATION_PLUGIN_GUIDE.md` for implementation details.

## Worker Monitoring

The server automatically monitors worker processes:
- Detects worker death via `waitpid`
- Automatically restarts dead workers
- Marks in-progress jobs as FAILED if worker dies
- Logs worker lifecycle events

## Files and Components

| Component | Location |
|-----------|----------|
| Server executable | `cpp/cuopt_remote_server.cpp` |
| Client logic | `cpp/src/linear_programming/utilities/remote_solve.cu` |
| Serialization interface | `cpp/include/cuopt/linear_programming/utilities/remote_serialization.hpp` |
| Protobuf serializer | `cpp/src/linear_programming/utilities/protobuf_serializer.cu` |
| Protobuf schema | `cpp/src/linear_programming/utilities/cuopt_remote.proto` |
| MsgPack serializer (example) | `cpp/src/linear_programming/utilities/serializers/msgpack_serializer.cpp` |

## Troubleshooting

### Library Loading Issues (Development)

If you're developing and the wrong `libcuopt.so` is being loaded:

```bash
# Use LD_PRELOAD to force loading local build
LD_PRELOAD=cpp/build/libcuopt.so ./cpp/build/cuopt_cli problem.mps
```

For production, use `build.sh` which sets up proper RPATH.

### Server Not Responding

1. Check server is running: `pgrep -af cuopt_remote_server`
2. Check port is listening: `ss -tlnp | grep 8765`
3. Check firewall allows connections

### Job Stuck in PROCESSING

- Worker may have crashed - check server logs
- Server will mark job as FAILED after detecting worker death

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `CUOPT_REMOTE_HOST` | Server hostname/IP | (none - local solve) |
| `CUOPT_REMOTE_PORT` | Server port | (none - local solve) |
| `CUOPT_REMOTE_USE_SYNC` | Use sync mode if "1" | "0" (async) |
| `CUOPT_SERIALIZER_LIB` | Path to custom serializer | (uses protobuf) |
