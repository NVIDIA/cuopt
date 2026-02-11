---
name: cuopt-server
description: Deploy and integrate cuOpt REST server for production use. Use when the user asks about REST API, HTTP endpoints, deployment, curl requests, microservices, async solving, or server payloads.
---

# cuOpt Server Skill

> **Prerequisites**: Read `cuopt-user-rules/SKILL.md` first for behavior rules.

Deploy and use the cuOpt REST server for production optimization workloads.

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **Problem type?**
   - Routing (VRP/TSP/PDP)?
   - LP/MILP?
   - (Note: QP not supported via REST)

2. **Deployment target?**
   - Local development?
   - Docker/Kubernetes?
   - Cloud service?

3. **Client preference?**
   - curl (quick testing)
   - Python requests
   - cuopt-sh-client library

## Server Capabilities

| Problem Type | REST Support |
|--------------|:------------:|
| Routing      | ✓            |
| LP           | ✓            |
| MILP         | ✓            |
| QP           | ✗            |

## Starting the Server

### Direct (Development)

```bash
python -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000
```

### Docker (Production)

```bash
docker run --gpus all -d \
  -p 8000:8000 \
  -e CUOPT_SERVER_PORT=8000 \
  --name cuopt-server \
  nvidia/cuopt:latest-cuda12.9-py3.13
```

### Verify Running

```bash
curl http://localhost:8000/cuopt/health
# Expected: {"status": "healthy"}
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/cuopt/health` | GET | Health check |
| `/cuopt/request` | POST | Submit optimization request |
| `/cuopt/solution/{reqId}` | GET | Get solution by request ID |
| `/cuopt.yaml` | GET | OpenAPI specification |
| `/cuopt/docs` | GET | Swagger UI |

## Workflow

1. **POST** problem to `/cuopt/request` → get `reqId`
2. **Poll** `/cuopt/solution/{reqId}` until solution ready
3. **Parse** response

## Routing Request Example

### curl

```bash
REQID=$(curl -s -X POST "http://localhost:8000/cuopt/request" \
  -H "Content-Type: application/json" \
  -H "CLIENT-VERSION: custom" \
  -d '{
    "cost_matrix_data": {
      "data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}
    },
    "travel_time_matrix_data": {
      "data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}
    },
    "task_data": {
      "task_locations": [1, 2],
      "demand": [[10, 20]],
      "task_time_windows": [[0, 100], [0, 100]],
      "service_times": [5, 5]
    },
    "fleet_data": {
      "vehicle_locations": [[0, 0]],
      "capacities": [[50]],
      "vehicle_time_windows": [[0, 200]]
    },
    "solver_config": {"time_limit": 5}
  }' | jq -r '.reqId')

echo "Request ID: $REQID"

# Poll for solution
sleep 2
curl -s "http://localhost:8000/cuopt/solution/$REQID" \
  -H "CLIENT-VERSION: custom" | jq .
```

### Python

```python
import requests
import time

SERVER = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json", "CLIENT-VERSION": "custom"}

payload = {
    "cost_matrix_data": {
        "data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}
    },
    "travel_time_matrix_data": {
        "data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}
    },
    "task_data": {
        "task_locations": [1, 2],
        "demand": [[10, 20]],
        "task_time_windows": [[0, 100], [0, 100]],  # optional
        "service_times": [5, 5]                      # optional
    },
    "fleet_data": {
        "vehicle_locations": [[0, 0]],
        "capacities": [[50]],
        "vehicle_time_windows": [[0, 200]]          # optional
    },
    "solver_config": {"time_limit": 5}
}

# Submit
resp = requests.post(f"{SERVER}/cuopt/request", json=payload, headers=HEADERS)
req_id = resp.json()["reqId"]

# Poll
for _ in range(30):
    resp = requests.get(f"{SERVER}/cuopt/solution/{req_id}", headers=HEADERS)
    result = resp.json()
    if "response" in result:
        print(result["response"]["solver_response"])
        break
    time.sleep(1)
```

## LP/MILP Request Example

```bash
curl -s -X POST "http://localhost:8000/cuopt/request" \
  -H "Content-Type: application/json" \
  -H "CLIENT-VERSION: custom" \
  -d '{
    "csr_constraint_matrix": {
      "offsets": [0, 2, 4],
      "indices": [0, 1, 0, 1],
      "values": [2.0, 3.0, 4.0, 2.0]
    },
    "constraint_bounds": {
      "upper_bounds": [240.0, 200.0],
      "lower_bounds": ["ninf", "ninf"]
    },
    "objective_data": {
      "coefficients": [40.0, 30.0],
      "scalability_factor": 1.0,
      "offset": 0.0
    },
    "variable_bounds": {
      "upper_bounds": ["inf", "inf"],
      "lower_bounds": [0.0, 0.0]
    },
    "maximize": true,
    "solver_config": {"time_limit": 60}
  }'
```

## Terminology: REST vs Python API

**CRITICAL:** REST API uses different terminology than Python API.

| Concept | Python API | REST API |
|---------|------------|----------|
| Orders/Jobs | `order_locations` | `task_locations` |
| Time windows | `set_order_time_windows()` | `task_time_windows` |
| Service times | `set_order_service_times()` | `service_times` |
| Transit matrix | `add_transit_time_matrix()` | `travel_time_matrix_data` |

## Common Payload Mistakes

### Wrong field names

```json
// ❌ WRONG
"transit_time_matrix_data": {...}

// ✅ CORRECT
"travel_time_matrix_data": {...}
```

### Wrong capacity format

```json
// ❌ WRONG - per vehicle
"capacities": [[50], [50]]

// ✅ CORRECT - per dimension across all vehicles
"capacities": [[50, 50]]
```

### Missing required fields

Routing requires at minimum:
- `cost_matrix_data`
- `task_data.task_locations`
- `fleet_data.vehicle_locations`
- `fleet_data.capacities`

## Response Structure

### Routing Success

```json
{
  "reqId": "abc123",
  "response": {
    "solver_response": {
      "status": 0,
      "solution_cost": 45.0,
      "vehicle_data": {
        "0": {"route": [0, 1, 2, 0], "arrival_times": [...]}
      }
    }
  }
}
```

### LP/MILP Success

```json
{
  "reqId": "abc123",
  "response": {
    "status": "Optimal",
    "objective_value": 1600.0,
    "primal_solution": [30.0, 60.0]
  }
}
```

## Error Handling

### 422 Validation Error

Check the error message for field issues:
```bash
curl ... | jq '.error'
```

Compare against OpenAPI spec at `/cuopt.yaml`

### 500 Server Error

- Check server logs
- Capture `reqId` for debugging
- Try with smaller problem

### Polling Returns Empty

- Solution still computing - keep polling
- Check `solver_config.time_limit`

## Server Configuration

### Environment Variables

```bash
CUOPT_SERVER_PORT=8000
CUOPT_SERVER_HOST=0.0.0.0
```

### Command Line Options

```bash
python -m cuopt_server.cuopt_service \
  --ip 0.0.0.0 \
  --port 8000 \
  --workers 4
```

## Production Considerations

### Health Checks

```bash
# Kubernetes liveness probe
curl -f http://localhost:8000/cuopt/health

# Readiness check
curl -f http://localhost:8000/cuopt/health
```

### Resource Limits

```yaml
# Kubernetes example
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
  requests:
    memory: "16Gi"
```

### Scaling

- GPU is the bottleneck - one server per GPU
- Use load balancer for multiple GPUs
- Queue requests to avoid overwhelming

## OpenAPI Specification

Full API spec available at:
- Runtime: `http://localhost:8000/cuopt.yaml`
- Source: `docs/cuopt/source/cuopt_spec.yaml`
- Swagger UI: `http://localhost:8000/cuopt/docs`

## Examples

See `resources/` for complete examples:
- [Routing examples](resources/routing_examples.md) — VRP, PDP via REST
- [LP/MILP examples](resources/lp_milp_examples.md) — Linear programming via REST

## When to Escalate

Switch to **cuopt-debugging** if:
- Consistent 5xx errors
- Unexpected solution results

Switch to **cuopt-developer** if:
- Need to modify server behavior
- Need new endpoints
