# cuopt_mcp — MCP server for NVIDIA cuOpt

Exposes cuOpt LP and MILP solving to MCP clients (Claude Code, Cursor, Codex)
over the cuOpt gRPC backend.

```text
MCP client ──stdio (JSON-RPC)──> cuopt-mcp ──gRPC──> cuopt_grpc_server (GPU)
```

The MCP server runs as a stdio subprocess on the user's machine and needs no
GPU: the solve happens wherever `cuopt_grpc_server` runs. No HTTP application
endpoint is exposed — the client speaks MCP over stdio, and this process
speaks gRPC to the backend.

## Install

```bash
pip install cuopt_mcp
```

## Configure

Start the solver backend on a GPU host (5001 is `cuopt_grpc_server`'s own
default; shown explicitly here since the client points at it by name):

```bash
cuopt_grpc_server --port 5001
```

Then register the MCP server with your client:

```json
{
  "mcpServers": {
    "cuopt": {
      "command": "cuopt-mcp",
      "env": { "CUOPT_REMOTE_HOST": "gpu-host", "CUOPT_REMOTE_PORT": "5001" }
    }
  }
}
```

Configuration reuses the environment the cuOpt gRPC client already honours —
`CUOPT_REMOTE_HOST`, `CUOPT_REMOTE_PORT`, and `CUOPT_TLS_*`. The connection is
plain TCP unless `CUOPT_TLS_ENABLED=true` is set; set it (with
`CUOPT_TLS_ROOT_CERT`/`CUOPT_TLS_CLIENT_CERT`/`CUOPT_TLS_CLIENT_KEY` as
needed) whenever `gpu-host` isn't a trusted local network.

## Tools

| Tool | Purpose |
|------|---------|
| `cuopt_solve_lp` | Submit an LP; returns a `job_id` immediately |
| `cuopt_solve_milp` | Submit a MILP; returns a `job_id` immediately |
| `cuopt_status` | Poll job state |
| `cuopt_result` | Fetch the solution, shaped to stay readable |
| `cuopt_incumbents` | Watch a MILP's objective improve (needs `track_incumbents=true` at submit) |
| `cuopt_logs` | Solver log lines for a finished job (no live tail yet) |
| `cuopt_cancel` | Stop a running job |
| `cuopt_delete` | Release a job's server-side state once its result is no longer needed |
| `cuopt_list_settings` | Discover solver parameters |

Solves are asynchronous by design. A blocking call would exceed the MCP
client timeout on any realistic MILP and would make cancellation impossible.

## Design notes

**No per-job state.** Column names needed to label a solution are supplied
per call via `names_from`, so any process can retrieve a named result for a
job it did not submit. The only state this process holds is the gRPC channel.

**Result shaping.** Problems can have millions of variables; the binding
limit on a tool result is the model's context window, not the transport. So
`cuopt_result` returns a summary plus narrow accessors (`variables`,
`nonzero_only`), writing the full vector to a file past `limit`.

**Settings catalogue is generated.** `_generated/cuopt_mcp_schema.json` is
emitted from `cpp/src/grpc/codegen/field_registry.yaml` by
`./build.sh codegen`, the same source of truth that drives the proto and the
C++ conversion code. A new solver parameter reaches this server with no
MCP-specific work.
