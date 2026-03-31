# cuOpt gRPC Remote Execution Quick Start

This guide shows how to start the cuOpt gRPC server and solve
optimization problems remotely from Python, `cuopt_cli`, or the C API.

All three interfaces use the same environment variables for remote
configuration. Once the env vars are set, your code works exactly the
same as a local solve &mdash; no API changes required.

## Prerequisites

- A host with an NVIDIA GPU and cuOpt installed (server side).
- cuOpt client libraries installed on the client host (can be CPU-only).
- `cuopt_grpc_server` binary available (ships with the cuOpt package).

## 1. Start the Server

### Basic (no TLS)

```bash
cuopt_grpc_server --port 8765 --workers 1
```

### TLS (server authentication)

```bash
cuopt_grpc_server --port 8765 \
  --tls \
  --tls-cert server.crt \
  --tls-key server.key
```

### mTLS (mutual authentication)

```bash
cuopt_grpc_server --port 8765 \
  --tls \
  --tls-cert server.crt \
  --tls-key server.key \
  --tls-root ca.crt \
  --require-client-cert
```

See `GRPC_SERVER_ARCHITECTURE.md` for the full set of server flags.

### How mTLS Works

With mTLS the server verifies every client, and the client verifies the
server. The trust model is based on Certificate Authorities (CAs), not
individual certificates:

- **`--tls-root ca.crt`** tells the server which CA to trust. Any client
  presenting a certificate signed by this CA is accepted. The server
  never sees or stores individual client certificates.
- **`--require-client-cert`** makes client verification mandatory. Without
  it the server requests a client cert but still allows unauthenticated
  connections.
- On the client side, `CUOPT_TLS_ROOT_CERT` is the CA that signed the
  *server* certificate, so the client can verify the server's identity.

### Restricting Access with a Custom CA

To limit which clients can reach your server, create a private CA and
only issue client certificates to authorized users. Anyone without a
certificate signed by your CA is rejected at the TLS handshake before
any solver traffic is exchanged.

**1. Create a private CA (one-time setup):**

```bash
# Generate CA private key
openssl genrsa -out ca.key 4096

# Generate self-signed CA certificate (valid 10 years)
openssl req -new -x509 -key ca.key -sha256 -days 3650 \
  -subj "/CN=cuopt-internal-ca" -out ca.crt
```

**2. Issue a client certificate:**

```bash
# Generate client key
openssl genrsa -out client.key 2048

# Create a certificate signing request
openssl req -new -key client.key \
  -subj "/CN=team-member-alice" -out client.csr

# Sign with your CA
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -days 365 -sha256 -out client.crt
```

Repeat step 2 for each authorized client. Keep `ca.key` private;
distribute only `ca.crt` (to the server) and the per-client
`client.crt` + `client.key` pairs.

**3. Issue a server certificate (signed by the same CA):**

```bash
# Generate server key
openssl genrsa -out server.key 2048

# Create CSR with subjectAltName matching the hostname clients will use
openssl req -new -key server.key \
  -subj "/CN=server.example.com" -out server.csr

# Write a SAN extension file (DNS and/or IP must match client's target)
cat > server.ext <<EOF
subjectAltName=DNS:server.example.com,DNS:localhost,IP:127.0.0.1
EOF

# Sign with your CA
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -days 365 -sha256 -extfile server.ext -out server.crt
```

> **Note:** `server.crt` must be signed by the same CA distributed to
> clients, and its `subjectAltName` must match the hostname or IP that
> clients connect to. gRPC (BoringSSL) requires SAN — `CN` alone is
> not sufficient for hostname verification.

**4. Start the server with your CA:**

```bash
cuopt_grpc_server --port 8765 \
  --tls \
  --tls-cert server.crt \
  --tls-key server.key \
  --tls-root ca.crt \
  --require-client-cert
```

**5. Configure an authorized client:**

```bash
export CUOPT_REMOTE_HOST=server.example.com
export CUOPT_REMOTE_PORT=8765
export CUOPT_TLS_ENABLED=1
export CUOPT_TLS_ROOT_CERT=ca.crt          # verifies the server
export CUOPT_TLS_CLIENT_CERT=client.crt    # proves client identity
export CUOPT_TLS_CLIENT_KEY=client.key
```

**Revoking access:** gRPC's built-in TLS does not support Certificate
Revocation Lists (CRL) or OCSP. To revoke a client, either stop issuing
new certs from the compromised CA and rotate to a new one, or deploy a
reverse proxy (e.g., Envoy) in front of the server that supports CRL
checking.

## 2. Configure the Client (All Interfaces)

Set these environment variables before running any cuOpt client.
They apply identically to the Python API, `cuopt_cli`, and the C API.

### Required

```bash
export CUOPT_REMOTE_HOST=<server-hostname>
export CUOPT_REMOTE_PORT=8765
```

When both `CUOPT_REMOTE_HOST` and `CUOPT_REMOTE_PORT` are set, every
call to `solve_lp` / `solve_mip` is transparently forwarded to the
remote server. No code changes are needed.

### TLS (optional)

```bash
export CUOPT_TLS_ENABLED=1
export CUOPT_TLS_ROOT_CERT=ca.crt               # verify server certificate
```

For mTLS, also provide the client identity:

```bash
export CUOPT_TLS_CLIENT_CERT=client.crt
export CUOPT_TLS_CLIENT_KEY=client.key
```

### Tuning (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `CUOPT_CHUNK_SIZE` | 16 MiB | Bytes per chunk for large problem transfer |
| `CUOPT_MAX_MESSAGE_BYTES` | 256 MiB | Client-side gRPC max message size |
| `CUOPT_GRPC_DEBUG` | `0` | Enable debug / throughput logging (`1` to enable) |

## 3. Usage Examples

Once the env vars are set, write your solver code exactly as you would
for a local solve. The remote transport is handled automatically.

### Python

```python
import cuopt_mps_parser
from cuopt import linear_programming

# Parse an MPS file
dm = cuopt_mps_parser.ParseMps("model.mps")

# Solve (routed to remote server via env vars)
solution = linear_programming.Solve(dm, linear_programming.SolverSettings())

print("Objective:", solution.get_primal_objective())
print("Primal:   ", solution.get_primal_solution()[:5], "...")
```

### cuopt_cli

```bash
cuopt_cli model.mps
```

With solver options:

```bash
cuopt_cli model.mps --time-limit 30 --relaxation
```

### C++ API

```cpp
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>

// Build problem using cpu_optimization_problem_t ...
auto solution = cuopt::linear_programming::solve_lp(cpu_problem, settings);
```

The same `solve_lp` / `solve_mip` functions automatically detect the
`CUOPT_REMOTE_HOST` / `CUOPT_REMOTE_PORT` env vars and forward to the
gRPC server when they are set.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Connection refused | Verify the server is running and the host/port are correct. |
| TLS handshake failure | Ensure `CUOPT_TLS_ENABLED=1` is set and certificate paths are correct. |
| `Cannot open TLS file: ...` | The path in the TLS env var does not exist or is not readable. |
| Timeout on large problems | Increase the solver `time_limit` or the client `timeout_seconds`. |

## Further Reading

- `GRPC_INTERFACE.md` &mdash; Protocol details, chunked transfer, client config, message sizes.
- `GRPC_SERVER_ARCHITECTURE.md` &mdash; Server process model, IPC, threads, job lifecycle.
