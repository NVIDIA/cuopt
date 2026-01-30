# Server: Routing Examples

## Start Server

```bash
python -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000 &
sleep 5
curl http://localhost:8000/cuopt/health
```

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
    "solver_config": {"time_limit": 10}
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

## Response Format

```json
{
  "reqId": "abc123",
  "response": {
    "solver_response": {
      "status": 0,
      "solution_cost": 45.0,
      "vehicle_data": {
        "0": {
          "route": [0, 1, 2, 0],
          "arrival_times": [0, 10, 22, 32]
        }
      }
    }
  }
}
```

## Terminology Mapping

| Python API | REST Server |
|------------|-------------|
| `order_locations` | `task_locations` |
| `set_order_time_windows()` | `task_time_windows` |
| `set_order_service_times()` | `service_times` |
| `add_transit_time_matrix()` | `travel_time_matrix_data` |
