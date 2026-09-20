# cuopt_mcp — MCP server for NVIDIA cuOpt

Exposes cuOpt LP, MILP, and vehicle routing (VRP/PDP) solving to MCP clients
(Claude Code, Cursor, Codex) over the cuOpt gRPC backend.

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

Not published yet — until then, build from source (see [Testing](#build-and-install-from-source)).

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
| `cuopt_health` | Report the configured gRPC target and whether it answers |
| `cuopt_solve_lp` | Submit an LP; returns a `job_id` immediately |
| `cuopt_solve_milp` | Submit a MILP; returns a `job_id` immediately |
| `cuopt_solve_vrp` | Submit a vehicle routing problem; returns a `job_id` immediately |
| `cuopt_status` | Poll job state (LP, MILP, or VRP) |
| `cuopt_result` | Fetch an LP/MILP solution, shaped to stay readable |
| `cuopt_vrp_result` | Fetch a VRP solution (route stops), shaped to stay readable |
| `cuopt_incumbents` | Watch a MILP's objective improve (needs `track_incumbents=true` at submit) |
| `cuopt_logs` | Solver log lines for a finished job (no live tail yet) |
| `cuopt_cancel` | Stop a running job |
| `cuopt_delete` | Release a job's server-side state once its result is no longer needed |
| `cuopt_list_settings` | Discover LP/MILP solver parameters |

Solves are asynchronous by design. A blocking call would exceed the MCP
client timeout on any realistic MILP and would make cancellation impossible.

## Testing

### Build and install from source

```bash
conda activate ./.cuopt_env          # the repo-local env, see CONTRIBUTING.md
./build.sh cuopt_mcp                 # installs into the active env
```

This installs as plain `cuopt_mcp` (unsuffixed), matching its `cuopt`
dependency -- `./build.sh` installs every Python package against the
unsuffixed, CPU-only `cuopt`.

### Smoke test

```bash
cuopt_grpc_server --port 5001 &
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
CUOPT_TEST_GRPC_PORT=5001 pytest python/cuopt_mcp/tests -q
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
      "env": { "CUOPT_REMOTE_HOST": "localhost", "CUOPT_REMOTE_PORT": "5001" }
    }
  }
}
```

A minimal end-to-end exercise: `cuopt_health`, then `cuopt_solve_lp` with a
small JSON model, then `cuopt_status` until terminal, then `cuopt_result`.

### If the backend looks unreachable

`cuopt-mcp` never starts or stops `cuopt_grpc_server`. Before starting one,
check whether one is already running on the configured port:

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

**Settings catalogue is generated (LP/MILP only).** `_generated/
cuopt_mcp_schema.json` is emitted from `cpp/src/grpc/codegen/
field_registry.yaml` by `./build.sh codegen`, the same source of truth
that drives the proto and the C++ conversion code. A new LP/MILP solver
parameter reaches this server with no MCP-specific work. VRP settings
aren't in this registry (only `time_limit`/`verbose_mode`/`error_logging`
reach the server; see `cuopt_solve_vrp`'s docstring) so there's no
equivalent `cuopt_list_settings` coverage for VRP.

**VRP submission has no host-CUDA dependency at record time.**
`cuopt.routing.DataModel` records setter calls (numpy arrays) and never
builds a device model on this host -- it only serializes the recorded
calls onto the wire, the same way `cuopt_solve_lp`/`cuopt_solve_milp`'s
JSON path never runs a solve locally.
