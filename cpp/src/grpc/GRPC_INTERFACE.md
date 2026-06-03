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
  message, which includes the `field_id`, `element_offset`, and
  `total_elements` (for server-side pre-allocation). For arrays inside a
  repeated nested message (e.g. a `QuadraticConstraint`), the chunk
  additionally carries `container_field_num` and `container_index` to
  identify the parent message and entry; both fields must be set or both
  unset (see `cuopt_remote_service.proto::ArrayChunk` for the full contract)
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
- LP results include PDLP warm start data for subsequent warm-started solves
  (the exact set of warm-start scalars and arrays is defined by the
  `warm_start:` section of `field_registry.yaml`)

### Automatic Routing

The cuOpt client handles size-based routing transparently:

1. **Upload**: Estimate serialized problem size
   - Below 75% of `max_message_bytes` → unary `SubmitJob`
   - Above threshold → `StartChunkedUpload` + `SendArrayChunk` + `FinishChunkedUpload`
2. **Download**: Check `result_size_bytes` from `CheckStatus`
   - Below `max_message_bytes` → unary `GetResult`
   - Above limit (or `RESOURCE_EXHAUSTED`) → chunked download RPCs

The 75% threshold is a cuOpt-client safety margin, not a wire requirement;
a custom client may pick any threshold (or simply react to a
`RESOURCE_EXHAUSTED` reply by retrying via the chunked path). What the
server requires is just that any single submission above its configured
`max_message_bytes` use the chunked upload — see
[Message Size Limits](#message-size-limits).

### Custom Clients

The protos and this document together fully specify the wire contract;
any gRPC-capable language can implement a client against
`cuopt_remote_service.proto` + `cuopt_remote_data.proto`. The cuOpt
project does not consider this protocol private.

A custom client has two reasonable strategies:

1. **Implement the chunked path** (described in the next section). This
   is the same path the cuOpt-provided client uses for large problems
   and is also the *faster* path on big payloads: 16 MiB chunks fit in
   CPU L3 cache and consecutive `SendArrayChunk` RPCs pipeline
   naturally, where a single multi-hundred-MB unary message must
   traverse DRAM bandwidth and serializes the request/response cycle.
   In an example test on a generic host with a 706 MB LP problem,
   the default chunked upload measures ~580 MB/s versus ~280 MB/s for a
   forced unary upload of the same problem — about a 2× difference.

2. **Stick to unary `SubmitJob`/`GetResult` and raise
   `max_message_bytes` on both sides.** Simpler to implement and fine
   for most problem sizes, at a small throughput cost on the largest
   problems. The server's `--max-message-mb` (or
   `--max-message-bytes`) and the client's `max_message_bytes` both
   accept any value in [4 KiB, ~2 GiB], so a custom client can send
   any problem up to ~2 GiB or so as a single round-trip and skip the
   chunked machinery entirely. Only problems that genuinely exceed the
   gRPC/protobuf ~2 GiB ceiling strictly require chunking.

The cuOpt-provided client and server both default `max_message_bytes`
to **256 MiB**, which causes the cuOpt client to switch to chunked
uploads at ~192 MiB estimated payload (75% of 256 MiB) and to send
**16 MiB** per `SendArrayChunk` thereafter. These are tunable
defaults, not forced compromises.

### Chunked Array Wire Format

The chunked path exists for one reason: protobuf's serialization layer
caps a single message at ~2 GiB, and a typical real-world LP/QP can
exceed that on the constraint-matrix values alone (hundreds of millions
of nonzeros × 8 bytes/double). To send such problems we split the large
arrays into independent unary RPCs (`SendArrayChunk`) carrying raw byte
slices, and the server reassembles them into the per-array buffers it
hands to the solver. The format below is what makes that reassembly
unambiguous.

External clients reassembling chunked uploads (or producing them) need
two pieces of information about each array: which array a chunk targets,
and how its bytes are laid out.

- **Top-level arrays** (e.g. `A_values`, `c`, `variable_lower_bounds`):
  set `ArrayChunk.field_id` to an `ArrayFieldId` enum value
  (`FIELD_A_VALUES`, `FIELD_C`, …) and leave `container_field_num` /
  `container_index` unset. Each `ArrayFieldId` value in
  `cuopt_remote_data.proto` carries a comment giving the on-the-wire
  element layout for `data`.

- **Arrays inside a repeated nested message** (e.g. a single
  `QuadraticConstraint`'s `linear_values`): set `container_field_num` to
  the parent's repeated-field tag (e.g. `25` for
  `OptimizationProblem.quadratic_constraints`), `container_index` to the
  0-based entry index, and `field_id` to a value of the nested
  `<MessageType>.ArrayId` enum (e.g. `QuadraticConstraint.LINEAR_VALUES`).
  Each nested `ArrayId` value carries the same byte-layout comment.
  `container_field_num` and `container_index` are an indivisible pair —
  setting one without the other is rejected with `INVALID_ARGUMENT`.

`ArrayChunk.data` always carries **raw native-endian element bytes**
(memcpy'd directly from the producer's contiguous buffer), not a
protobuf-encoded value. `total_elements` counts logical elements, so:

```
data.size() == elements_in_this_chunk * element_size_bytes
```

All officially supported cuOpt platforms (x86-64, aarch64) are
little-endian, so a from-scratch client can write IEEE-754 doubles and
two's-complement int32s in LE form. A client on a big-endian host must
byte-swap before populating `data`.

For string-list arrays (`variable_names`, `row_names`), the wire form is
NUL-terminated UTF-8 strings concatenated end-to-end; `total_elements` is
the total byte length of the blob (not the string count), and the server
splits on `\0`.

## Message Size Limits

| Configuration | Default | Notes |
|---------------|---------|-------|
| Server `--max-message-mb` | 256 MiB | Per-message limit (also `--max-message-bytes` for exact byte values) |
| Server clamping | [4 KiB, ~2 GiB] | Enforced at startup to stay within protobuf's serialization limit |
| Client `max_message_bytes` | 256 MiB | Clamped to [4 MiB, ~2 GiB] at construction |
| Chunk size | 16 MiB | Payload per `SendArrayChunk`/`GetResultChunk` |
| Chunked threshold (cuOpt-client heuristic) | 75% of `max_message_bytes` | The cuOpt client switches to chunked upload at this estimated size to avoid a `RESOURCE_EXHAUSTED` round-trip. **Not a wire requirement** — a from-scratch client may pick any threshold or react to `RESOURCE_EXHAUSTED` lazily. |

**Chunking is mandatory above hard limits, not just a client preference.**
Two ceilings apply to every gRPC client of this server:

1. **Protobuf serialization ceiling (~2 GiB).** Protobuf's wire-format
   length fields are int32, so any single message — by anyone, on any
   client — physically cannot serialize beyond ~2 GiB. A real-world LP
   with hundreds of millions of nonzeros at 8 bytes per double already
   exceeds this on the values array alone, so it **must** chunk.
2. **The server's configured `max_message_bytes`** (default 256 MiB,
   operator-controlled via `--max-message-mb`). Any unary submission
   above this is rejected with `RESOURCE_EXHAUSTED` regardless of what
   the client does. The server hard-clamps this to [4 KiB, ~2 GiB] at
   startup so it can never accept a value the protobuf layer cannot
   serialize.

Chunked transfer escapes only the per-message limit — the *total* problem
payload can be arbitrarily large, since each `SendArrayChunk` is its own
unary RPC. Neither client nor server allows "unlimited" individual
messages.

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
- `GRPC_CODE_GENERATION.md` — Codegen architecture: what gets generated and how it's wired into the build.
- `codegen/FIELD_REGISTRY_REFERENCE.md` — Field-level reference for `field_registry.yaml` and walkthroughs for adding new fields.
