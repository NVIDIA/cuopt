# gRPC Chunked Transfer Protocol

## Overview

The cuOpt remote execution system uses gRPC for client-server communication. The interface
supports arbitrarily large optimization problems (multi-GB) through a chunked array transfer
protocol that uses only unary (request-response) RPCs — no bidirectional streaming.

## Chunked Array Transfer Protocol

### Why Chunking?

gRPC has per-message size limits (configurable, default set to 256 MiB in cuOpt), and
protobuf has a hard 2 GB serialization limit. Optimization problems and their solutions
can exceed several gigabytes, so a chunked transfer mechanism is needed.

The protocol uses only **unary RPCs** (no bidirectional streaming), which simplifies
error handling, load balancing, and proxy compatibility.

### Upload Protocol (Large Problems)

When the estimated serialized problem size exceeds 75% of `max_message_bytes`, the client
splits large arrays into chunks and sends them via multiple unary RPCs:

```text
Client                                          Server
  |                                               |
  |-- StartChunkedUpload(header, settings) -----> |
  |<-- upload_id, max_message_bytes -------------- |
  |                                               |
  |-- SendArrayChunk(upload_id, field, data) ----> |
  |<-- ok ---------------------------------------- |
  |                                               |
  |-- SendArrayChunk(upload_id, field, data) ----> |
  |<-- ok ---------------------------------------- |
  |           ...                                 |
  |                                               |
  |-- FinishChunkedUpload(upload_id) ------------> |
  |<-- job_id ------------------------------------ |
```

**Key features:**
- `StartChunkedUpload` sends a `ChunkedProblemHeader` with all scalar fields
  and solver settings (no per-array metadata in the header)
- Each `SendArrayChunk` carries one chunk of one array via an `ArrayChunk`
  message, which includes the `ArrayFieldId`, `element_offset`, and
  `total_elements` (for server-side pre-allocation)
- The server reports `max_message_bytes` so the client can adapt chunk sizing
- `FinishChunkedUpload` triggers server-side reassembly and job submission

### Download Protocol (Large Results)

When the result exceeds the gRPC max message size, the client fetches it via
chunked unary RPCs (mirrors the upload pattern):

```text
Client                                           Server
  |                                                |
  |-- StartChunkedDownload(job_id) --------------> |
  |<-- download_id, ChunkedResultHeader ---------- |
  |                                                |
  |-- GetResultChunk(download_id, field, off) ----> |
  |<-- data bytes --------------------------------- |
  |                                                |
  |-- GetResultChunk(download_id, field, off) ----> |
  |<-- data bytes --------------------------------- |
  |           ...                                  |
  |                                                |
  |-- FinishChunkedDownload(download_id) ---------> |
  |<-- ok ----------------------------------------- |
```

**Key features:**
- `ChunkedResultHeader` carries all scalar fields (termination status, objectives,
  residuals, solve time, warm start scalars) plus `ResultArrayDescriptor` entries
  for each array (solution vectors, warm start arrays)
- Each `GetResultChunk` fetches a slice of one array, identified by `ResultFieldId`
  and `element_offset`
- `FinishChunkedDownload` releases the server-side download session state
- LP results include PDLP warm start data (9 arrays + 8 scalars) for subsequent
  warm-started solves

### Automatic Routing

The client handles size-based routing transparently:

1. **Upload**: Estimate serialized problem size
   - Below 75% of `max_message_bytes` → unary `SubmitJob`
   - Above threshold → `StartChunkedUpload` + `SendArrayChunk` + `FinishChunkedUpload`
2. **Download**: Check `result_size_bytes` from `CheckStatus`
   - Below `max_message_bytes` → unary `GetResult`
   - Above limit (or `RESOURCE_EXHAUSTED`) → chunked download RPCs

## Message Size Limits

| Configuration | Default | Notes |
|---------------|---------|-------|
| Server `--max-message-mb` | 256 MiB | Per-message limit (also `--max-message-bytes` for exact byte values) |
| Server clamping | [4 KiB, ~2 GiB] | Enforced at startup to stay within protobuf's serialization limit |
| Client `max_message_bytes` | 256 MiB | Clamped to [4 MiB, ~2 GiB] at construction |
| Chunk size | 16 MiB | Payload per `SendArrayChunk`/`GetResultChunk` |
| Chunked threshold | 75% of max_message_bytes | Problems above this use chunked upload (e.g. 192 MiB when max is 256 MiB) |

Chunked transfer allows unlimited total payload size; only individual
chunks must fit within the per-message limit. Neither client nor server
allows "unlimited" message size — both clamp to the protobuf 2 GiB ceiling.

## Error Handling

### gRPC Status Codes

| Code | Meaning | Client Action |
|------|---------|---------------|
| `OK` | Success | Process result |
| `NOT_FOUND` | Job ID not found | Check job ID |
| `RESOURCE_EXHAUSTED` | Message too large | Use chunked transfer |
| `CANCELLED` | Job was cancelled | Handle gracefully |
| `DEADLINE_EXCEEDED` | Timeout | Retry or increase timeout |
| `UNAVAILABLE` | Server not reachable | Retry with backoff |
| `INTERNAL` | Server error | Report to user |
| `INVALID_ARGUMENT` | Bad request | Fix request |

### Connection Handling

- Client detects `context->IsCancelled()` for graceful disconnect
- Server cleans up job state on client disconnect during upload
- Automatic reconnection is NOT built-in (caller should retry)

## Related Documentation

- `GRPC_SERVER_ARCHITECTURE.md` — Server process model, IPC, threads, job lifecycle.
- `GRPC_QUICK_START.md` — Starting the server and solving remotely from Python, CLI, or C.
- `GRPC_CODE_GENERATION.md` — Registry format, generated file inventory, and walkthrough examples.
