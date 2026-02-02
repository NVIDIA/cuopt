# Server: Routing Examples

## Start Server

```bash
python -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000 &
sleep 5
curl http://localhost:8000/cuopt/health
```

> **Note:** Using `--ip 0.0.0.0` binds to all interfaces for development convenience; use `--ip 127.0.0.1` or a specific interface in production or untrusted networks.

## Basic VRP (curl)

```bash
REQID=$(curl -s -X POST "http://localhost:8000/cuopt/request" \
  -H "Content-Type: application/json" \
  -H "CLIENT-VERSION: custom" \
  -d '{
    "cost_matrix_data": {
      "data": {"0": [[0,10,15,20],[10,0,12,18],[15,12,0,10],[20,18,10,0]]}
    },
    "travel_time_matrix_data": {
      "data": {"0": [[0,10,15,20],[10,0,12,18],[15,12,0,10],[20,18,10,0]]}
    },
    "task_data": {
      "task_locations": [1, 2, 3],
      "demand": [[10, 15, 20]],
      "service_times": [5, 5, 5]
    },
    "fleet_data": {
      "vehicle_locations": [[0, 0], [0, 0]],
      "capacities": [[50, 50]]
    },
    "solver_config": {"time_limit": 5}
  }' | jq -r '.reqId')

curl -s "http://localhost:8000/cuopt/solution/$REQID" -H "CLIENT-VERSION: custom" | jq .
```

## VRP with Time Windows (Python)

```python
import requests
import time

SERVER = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json", "CLIENT-VERSION": "custom"}

payload = {
    "cost_matrix_data": {
        "data": {"0": [[0,10,15,20,25],[10,0,12,18,22],[15,12,0,10,15],[20,18,10,0,8],[25,22,15,8,0]]}
    },
    "travel_time_matrix_data": {
        "data": {"0": [[0,10,15,20,25],[10,0,12,18,22],[15,12,0,10,15],[20,18,10,0,8],[25,22,15,8,0]]}
    },
    "task_data": {
        "task_locations": [1, 2, 3, 4],
        "demand": [[20, 30, 25, 15]],
        "task_time_windows": [[0, 50], [10, 60], [20, 70], [0, 80]],
        "service_times": [5, 5, 5, 5]
    },
    "fleet_data": {
        "vehicle_locations": [[0, 0], [0, 0]],
        "capacities": [[100, 100]],
        "vehicle_time_windows": [[0, 200], [0, 200]]
    },
    "solver_config": {
        "time_limit": 10
    }
}

# Submit
response = requests.post(f"{SERVER}/cuopt/request", json=payload, headers=HEADERS)
req_id = response.json()["reqId"]
print(f"Submitted: {req_id}")

# Poll for solution
for attempt in range(30):
    response = requests.get(f"{SERVER}/cuopt/solution/{req_id}", headers=HEADERS)
    result = response.json()

    if "response" in result:
        solver_response = result["response"].get("solver_response", {})
        print(f"Status: {solver_response.get('status')}")
        print(f"Cost: {solver_response.get('solution_cost')}")
        if "vehicle_data" in solver_response:
            for vid, vdata in solver_response["vehicle_data"].items():
                print(f"Vehicle {vid}: {vdata.get('route', [])}")
        break
    time.sleep(1)
```

## Pickup and Delivery (curl)

```bash
curl -s -X POST "http://localhost:8000/cuopt/request" \
  -H "Content-Type: application/json" \
  -H "CLIENT-VERSION: custom" \
  -d '{
    "cost_matrix_data": {
      "data": {"0": [[0,10,20,30,40],[10,0,15,25,35],[20,15,0,10,20],[30,25,10,0,15],[40,35,20,15,0]]}
    },
    "travel_time_matrix_data": {
      "data": {"0": [[0,10,20,30,40],[10,0,15,25,35],[20,15,0,10,20],[30,25,10,0,15],[40,35,20,15,0]]}
    },
    "task_data": {
      "task_locations": [1, 2, 3, 4],
      "demand": [[10, -10, 15, -15]],
      "pickup_and_delivery_pairs": [[0, 1], [2, 3]]
    },
    "fleet_data": {
      "vehicle_locations": [[0, 0]],
      "capacities": [[50]]
    },
    "solver_config": {"time_limit": 10}
  }' | jq .
```

## Terminology: Python vs REST

| Python API | REST Server |
|------------|-------------|
| `order_locations` | `task_locations` |
| `set_order_time_windows()` | `task_time_windows` |
| `set_order_service_times()` | `service_times` |
| `add_transit_time_matrix()` | `travel_time_matrix_data` |
| `set_pickup_delivery_pairs()` | `pickup_and_delivery_pairs` |

## Common Mistakes

```json
// ❌ WRONG field name
"transit_time_matrix_data": {...}

// ✅ CORRECT
"travel_time_matrix_data": {...}
```

```json
// ❌ WRONG capacity format (per vehicle)
"capacities": [[50], [50]]

// ✅ CORRECT (per dimension across vehicles)
"capacities": [[50, 50]]
```

---

## Additional References (tested in CI)

For more complete examples, read these files:

| Example | File |
|---------|------|
| Basic Routing (Python) | `docs/cuopt/source/cuopt-server/examples/routing/examples/basic_routing_example.py` |
| Basic Routing (curl) | `docs/cuopt/source/cuopt-server/examples/routing/examples/basic_routing_example.sh` |
| Initial Solution | `docs/cuopt/source/cuopt-server/examples/routing/examples/initial_solution_example.py` |

These examples are tested by CI (`ci/test_doc_examples.sh`).
