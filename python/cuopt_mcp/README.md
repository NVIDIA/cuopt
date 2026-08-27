# cuopt_mcp — MCP server for NVIDIA cuOpt

Exposes cuOpt LP and MILP solving to MCP clients (Claude Code, Cursor, Codex)
over the cuOpt gRPC backend.

```
MCP client ──stdio (JSON-RPC)──> cuopt-mcp ──gRPC──> cuopt_grpc_server (GPU)
```

The MCP server runs as a stdio subprocess on the user's machine and needs no
GPU: the solve happens wherever `cuopt_grpc_server` runs. No HTTP is involved.

## Install

```bash
pip install cuopt_mcp
```

Not published yet — until then, build from source (see [Testing](#build-and-install-from-source)).

## Configure

Start the solver backend on a GPU host:

```bash
cuopt_grpc_server --port 50051
```

Then register the MCP server with your client:

```json
{
  "mcpServers": {
    "cuopt": {
      "command": "cuopt-mcp",
      "env": { "CUOPT_REMOTE_HOST": "gpu-host", "CUOPT_REMOTE_PORT": "50051" }
    }
  }
}
```

Configuration reuses the environment the cuOpt gRPC client already honours —
`CUOPT_REMOTE_HOST`, `CUOPT_REMOTE_PORT`, and `CUOPT_TLS_*`.

## Tools

| Tool | Purpose |
|------|---------|
| `cuopt_health` | Report the configured gRPC target and whether it answers |
| `cuopt_solve_lp` | Submit an LP; returns a `job_id` immediately |
| `cuopt_solve_milp` | Submit a MILP; returns a `job_id` immediately |
| `cuopt_status` | Poll job state |
| `cuopt_result` | Fetch the solution, shaped to stay readable |
| `cuopt_incumbents` | Watch a MILP's objective improve |
| `cuopt_logs` | Recent solver log lines |
| `cuopt_cancel` | Stop a running job |
| `cuopt_list_settings` | Discover solver parameters |

Solves are asynchronous by design. A blocking call would exceed the MCP
client timeout on any realistic MILP and would make cancellation impossible.

## Testing

### Build and install from source

```bash
conda activate ./.cuopt_env          # the repo-local env, see CONTRIBUTING.md
./build.sh cuopt_mcp                 # installs into the active env
```

This installs as `cuopt_mcp-cu13`. The CUDA suffix is inherited from the
`cuopt` dependency, not from anything this pure-Python package compiles.

### Smoke test

```bash
cuopt_grpc_server --port 50051 &
python -c "from cuopt_mcp import tools; print(tools.health())"
```

`reachable: true` means the MCP server can see the backend. If it is false the
message names the endpoint and what to check — a wrong `CUOPT_REMOTE_PORT` and
a server that is not running look identical from the client side, so it does
not assume either.

### Test suite

```bash
# Unit tests: no GPU, no server, stubbed gRPC client
pytest python/cuopt_mcp/tests -q

# Plus end-to-end against a live server, over real MCP stdio
CUOPT_TEST_GRPC_PORT=50051 pytest python/cuopt_mcp/tests -q
```

Without `CUOPT_TEST_GRPC_PORT` the end-to-end tests skip rather than fail.

**The end-to-end fixture launches `cuopt-mcp` from `PATH`**, not from the
interpreter running pytest. If another environment shadows the one you built,
the suite silently exercises that install instead — which surfaces as
unrelated-looking failures such as `undefined symbol: _ZTIN3rmm...bad_allocE`
from an ABI mismatch. Check with `which cuopt-mcp` before believing a failure.

### Driving it from an MCP client

Point the client at the built entry point and confirm with `cuopt_health`
before submitting a model — every other tool reports a connection problem only
after a model has been built.

```json
{
  "mcpServers": {
    "cuopt": {
      "command": "/path/to/.cuopt_env/bin/cuopt-mcp",
      "env": { "CUOPT_REMOTE_HOST": "localhost", "CUOPT_REMOTE_PORT": "50051" }
    }
  }
}
```

A minimal end-to-end exercise: `cuopt_health`, then `cuopt_solve_lp` with a
small JSON model, then `cuopt_status` until terminal, then `cuopt_result`.

### If the backend looks unreachable

`cuopt-mcp` never starts or stops `cuopt_grpc_server`. Before starting one,
check whether one is already running — a second server can share the listen
port, after which a job submitted to one process can be polled from the other:

```bash
pgrep -af cuopt_grpc_server
```

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
