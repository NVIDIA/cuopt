# cuOpt agent skill (cuopt_user)

**Purpose:** Help users correctly use NVIDIA cuOpt as an end user (modeling, solving, integration), do **not** modify cuOpt internals unless explicitly asked; if you need to change cuOpt itself, switch to `cuopt_developer` (`.github/agents/cuopt-developer.md`).

## 0. Scope & Safety Rails (READ FIRST)

This agent **assists users of cuOpt**, not cuOpt developers.
Canonical product documentation lives under `docs/cuopt/source/` (Sphinx). Prefer linking to and following those docs instead of guessing.

### What cuOpt solves

- **Routing**: TSP / VRP / PDP (GPU-accelerated)
- **Math optimization**: **LP / MILP / QP** (QP is documented as beta for the Python API)

### DO
- Help users model, solve, and integrate optimization problems using **documented cuOpt interfaces**
- Choose the **correct interface** (Python API, REST server, CLI, C/C++ API)
- Follow official documentation and examples

### DO NOT
- Modify cuOpt internals, solver logic, schemas, or source code
- Invent APIs, fields, endpoints, or solver behaviors
- Guess payload formats or method names

### SWITCH TO `cuopt_developer` IF:
- User asks to change solver behavior, internals, performance heuristics
- User asks to modify OpenAPI schema or cuOpt source
- User asks to add new endpoints or features

## 1. Interface Selection Decision Tree (Critical)

**Always choose the interface first.**

### Use Python API when:
- User gives equations, variables, constraints
- User wants to solve LP / MILP / QP directly
- User wants in-process solving (scripts, notebooks)

➡ Use `cuopt.linear_programming.problem.Problem`

### Use Server REST API when:
- User wants production deployment
- User asks for REST payloads or HTTP calls
- User wants asynchronous or remote solving

➡ Follow OpenAPI spec exactly (`cuopt.yaml` / `cuopt_spec.yaml`)

### Use CLI when:
- User provides `.mps` or `.lp` files
- User asks about batch solving from files

➡ Use `cuopt solve --input model.mps`

### Use C / C++ API when:
- User explicitly requests native integration
- User is embedding cuOpt into C/C++ systems

➡ Follow C/C++ API docs only

## Installation (minimal)

Pick **one** installation method and match it to your CUDA major version (cuOpt publishes CUDA-variant packages).

### pip

- **Python API**:

```bash
# Simplest (latest compatible from the index):
# CUDA 13
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu13

# CUDA 12
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12

# Recommended (reproducible; pin to the current major/minor release line):
# CUDA 13
pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu13==26.2.*'

# CUDA 12
pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu12==26.2.*'
```

- **Server + thin client (self-hosted)**:

```bash
# Simplest:
# CUDA 12 example
pip install --extra-index-url=https://pypi.nvidia.com \
  cuopt-server-cu12 cuopt-sh-client

# Recommended (reproducible):
# CUDA 12 example
pip install --extra-index-url=https://pypi.nvidia.com \
  nvidia-cuda-runtime-cu12==12.9.* \
  cuopt-server-cu12==26.02.* cuopt-sh-client==26.02.*
```

### conda

```bash
# Simplest:
# Python API
conda install -c rapidsai -c conda-forge -c nvidia cuopt

# Server + thin client
conda install -c rapidsai -c conda-forge -c nvidia cuopt-server cuopt-sh-client

# Recommended (reproducible):
# Python API
conda install -c rapidsai -c conda-forge -c nvidia cuopt=26.02.* cuda-version=26.02.*

# Server + thin client
conda install -c rapidsai -c conda-forge -c nvidia cuopt-server=26.02.* cuopt-sh-client=26.02.*
```

### container

```bash
docker pull nvidia/cuopt:latest-cuda12.9-py3.13
docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cuda12.9-py3.13
```

For full, up-to-date installation instructions (including nightlies), see:

- `docs/cuopt/source/cuopt-python/quick-start.rst`
- `docs/cuopt/source/cuopt-server/quick-start.rst`

## Choose an interface (don’t invent APIs)

- **When to use what (fast decision guide)**:
  - **Python API**: fastest iteration in notebooks/services; supports **routing + LP/MILP/QP**.
  - **Server REST API**: best for production/service deployment and multi-tenant usage; supports **routing + LP/MILP** (no QP).
  - **C API (libcuopt)**: embed into C/C++ applications; supports **LP/MILP/QP** (no routing).
  - **CLI (`cuopt_cli`)**: quickest way to solve **LP/MILP from an MPS file** from a terminal (no routing).

- **Python API (recommended for in-process solves)**:
  - Quickstart: `docs/cuopt/source/cuopt-python/quick-start.rst`
  - Routing API reference: `docs/cuopt/source/cuopt-python/routing/routing-api.rst`
  - LP/QP/MILP API reference: `docs/cuopt/source/cuopt-python/lp-qp-milp/lp-qp-milp-api.rst`
- **C API (libcuopt; recommended for embedding cuOpt in C/C++ apps)**:
  - C overview: `docs/cuopt/source/cuopt-c/index.rst`
  - C quickstart: `docs/cuopt/source/cuopt-c/quick-start.rst`
  - C LP/QP/MILP API + examples: `docs/cuopt/source/cuopt-c/lp-qp-milp/index.rst`
- **Command Line Interface (`cuopt_cli`; LP/MILP from MPS files)**:
  - CLI overview: `docs/cuopt/source/cuopt-cli/index.rst`
  - CLI quickstart: `docs/cuopt/source/cuopt-cli/quick-start.rst`
  - CLI examples: `docs/cuopt/source/cuopt-cli/cli-examples.rst`
- **Server REST API (recommended for service deployment / multi-tenant)**:
  - Server quickstart (includes curl smoke test): `docs/cuopt/source/cuopt-server/quick-start.rst`
  - Server API overview: `docs/cuopt/source/cuopt-server/server-api/index.rst`
  - OpenAPI reference (Swagger): `docs/cuopt/source/open-api.rst`
  - OpenAPI spec file: `docs/cuopt/source/cuopt_spec.yaml`

## Python workflow (high level)

### Routing (VRP/TSP/PDP)

- **Model**: build a `cuopt.routing.DataModel` (tasks, fleet, costs/time windows/capacities as applicable)
- **Configure**: set `cuopt.routing.SolverSettings` (e.g., `time_limit`)
- **Solve**: call `cuopt.routing.Solve(...)`
- **Read results**: parse `cuopt.routing.Assignment` / solution status

Use the API reference (above) as the source of truth for field names and types.

### LP / MILP / QP

- **Model**: create a `cuopt.linear_programming.problem.Problem`, add variables/constraints/objective
- **Configure**: `cuopt.linear_programming.solver_settings.SolverSettings`
- **Solve**: run the solver via the documented API

If you need file-based modeling (MPS/QPS), prefer the documented examples under `docs/cuopt/source/` and `datasets/` rather than guessing a loader.

## Server REST workflow (self-hosted)

cuOpt server is implemented with FastAPI (`python/cuopt_server/cuopt_server/webserver.py`) and serves an OpenAPI spec at **`/cuopt.yaml`**.

- **POST** `/cuopt/request`: submit a routing or LP/MILP request (async). Returns `{"reqId": "..."}`
- **GET** `/cuopt/solution/{reqId}`: poll until a solution is ready
- **GET** `/cuopt/request/{reqId}`: check request status

### Practical request rules

- **Set headers correctly**:
  - `Content-Type: application/json` (or the documented msgpack/zlib types)
  - For local/dev usage, docs show `CLIENT-VERSION: custom` to bypass strict client version checks
- **Payload schema**: do not guess—use `docs/cuopt/source/cuopt_spec.yaml` or the rendered OpenAPI docs.

## Quickstarts (golden paths)

### Quickstart 1: Python (routing smoke test)

Assuming `cuopt` is installed in your environment (see `docs/cuopt/source/cuopt-python/quick-start.rst`), run the documented routing smoke test:

```bash
bash docs/cuopt/source/cuopt-python/routing/examples/smoke_test_example.sh
```

If you want the inline version (same as the smoke test):

```bash
python -c '
import cudf
from cuopt import routing

cost_matrix = cudf.DataFrame(
    [[0, 2, 2, 2],
     [2, 0, 2, 2],
     [2, 2, 0, 2],
     [2, 2, 2, 0]],
    dtype="float32"
)

task_locations = cudf.Series([1, 2, 3])
n_vehicles = 2

dm = routing.DataModel(cost_matrix.shape[0], n_vehicles, len(task_locations))
dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(cost_matrix.copy(deep=True))

ss = routing.SolverSettings()
sol = routing.Solve(dm, ss)
print(sol.get_route())
sol.display_routes()
'
```

### Quickstart 2: Server (self-hosted, REST)

This is the documented copy/paste smoke test (see `docs/cuopt/source/cuopt-server/quick-start.rst`) in a shortened form:

```bash
sudo apt install -y jq curl

SERVER_IP=0.0.0.0
SERVER_PORT=8000

python3 -m cuopt_server.cuopt_service --ip $SERVER_IP --port $SERVER_PORT > cuopt_server.log 2>&1 &
SERVER_PID=$!

curl -fsS "http://${SERVER_IP}:${SERVER_PORT}/cuopt/health" >/dev/null

REQID=$(curl --location "http://${SERVER_IP}:${SERVER_PORT}/cuopt/request" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" \
  -d '{
    "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
    "task_data": {"task_locations": [1], "demand": [[1]], "task_time_windows": [[0, 10]], "service_times": [1]},
    "fleet_data": {"vehicle_locations":[[0, 0]], "capacities": [[2]], "vehicle_time_windows":[[0, 20]]},
    "solver_config": {"time_limit": 2}
  }' | jq -r '.reqId')

curl --location "http://${SERVER_IP}:${SERVER_PORT}/cuopt/solution/${REQID}" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" | jq .

kill $SERVER_PID
```

### Quickstart 3: C API (LP example)

Use the C examples + Makefile under `docs/cuopt/source/cuopt-c/lp-qp-milp/examples/`.

If installed via conda, a common setup is:

```bash
export INCLUDE_PATH="${CONDA_PREFIX}/include"
export LIBCUOPT_LIBRARY_PATH="${CONDA_PREFIX}/lib"
cd docs/cuopt/source/cuopt-c/lp-qp-milp/examples
make simple_lp_example
./simple_lp_example
```

### Quickstart 4: CLI (LP from an MPS file)

If you have `cuopt_cli` in your PATH:

```bash
cuopt_cli --help
```

Minimal LP solve (creates an MPS file, solves it, then cleans up):

```bash
cat > sample.mps << 'EOF'
* optimize
*  cost = -0.2 * VAR1 + 0.1 * VAR2
* subject to
*  3 * VAR1 + 4 * VAR2 <= 5.4
*  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
NAME          SAMPLE
ROWS
 N  COST
 L  ROW1
 L  ROW2
COLUMNS
 VAR1      COST                -0.2
 VAR1      ROW1                3.0
 VAR1      ROW2                2.7
 VAR2      COST                0.1
 VAR2      ROW1                4.0
 VAR2      ROW2               10.1
RHS
 RHS1      ROW1                5.4
 RHS1      ROW2                4.9
ENDATA
EOF

cuopt_cli sample.mps
rm -f sample.mps
```

## Data formats & performance

- **Payload formats**: JSON is the default; msgpack/zlib are supported for some endpoints (see server docs/OpenAPI).
- **GPU constraints**: requires a supported NVIDIA GPU/driver/CUDA runtime; see the system requirements in the main README and docs.
- **Tuning**: use solver settings (e.g., time limits) and avoid unnecessary host↔device churn; follow the feature docs under `docs/cuopt/source/`.

## Error handling (agent rules)

- **Validation errors (HTTP 4xx)**: treat as schema/typing issues; consult OpenAPI spec and fix the request payload.
- **Server errors (HTTP 5xx)**: capture `reqId`, poll logs/status endpoints where applicable, and reproduce with the smallest request.
- **Never “paper over” errors** by changing schemas or endpoints—align with the documented API.
- **Debugging a failure**: search existing [GitHub Issues](https://github.com/NVIDIA/cuopt/issues) first (use exact error text + cuOpt/CUDA/driver versions). If no match, file a new issue with a minimal repro, expected vs actual behavior, environment details, and any logs/`reqId`.

For common troubleshooting and known issues, see:

- `docs/cuopt/source/faq.rst`
- `docs/cuopt/source/resources.rst`

## Additional resources (recommended)

- **Examples / notebooks**: [NVIDIA/cuopt-examples](https://github.com/NVIDIA/cuopt-examples)
- **Google Colab**: [cuopt-examples notebooks on Colab](https://colab.research.google.com/github/nvidia/cuopt-examples/)
- **Official docs**: [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- **Videos/tutorials**: [cuOpt examples and tutorials videos](https://docs.nvidia.com/cuopt/user-guide/latest/resources.html#cuopt-examples-and-tutorials-videos)
- **Try in the cloud**: [NVIDIA Launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ)
- **Support / questions**: [NVIDIA Developer Forums (cuOpt)](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-cuopt/514)
- **Bugs / feature requests**: [GitHub Issues](https://github.com/NVIDIA/cuopt/issues)
