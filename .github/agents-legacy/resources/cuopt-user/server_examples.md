# Server REST API examples (cuOpt)

## Server: Start the Server

```bash
# Start server in background
python3 -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
sleep 5
curl -fsS "http://localhost:8000/cuopt/health"
```

## Server: Routing Request (curl)

```bash
# Submit a VRP request
REQID=$(curl -s --location "http://localhost:8000/cuopt/request" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" \
  -d '{
    "cost_matrix_data": {
      "data": {
        "0": [
          [0, 10, 15, 20],
          [10, 0, 12, 18],
          [15, 12, 0, 10],
          [20, 18, 10, 0]
        ]
      }
    },
    "travel_time_matrix_data": {
      "data": {
        "0": [
          [0, 10, 15, 20],
          [10, 0, 12, 18],
          [15, 12, 0, 10],
          [20, 18, 10, 0]
        ]
      }
    },
    "task_data": {
      "task_locations": [1, 2, 3],
      "demand": [[10, 15, 20]],
      "task_time_windows": [[0, 100], [10, 80], [20, 90]],
      "service_times": [5, 5, 5]
    },
    "fleet_data": {
      "vehicle_locations": [[0, 0], [0, 0]],
      "capacities": [[50, 50]],
      "vehicle_time_windows": [[0, 200], [0, 200]]
    },
    "solver_config": {
      "time_limit": 5
    }
  }' | jq -r '.reqId')

echo "Request ID: $REQID"

# Poll for solution
sleep 2
curl -s --location "http://localhost:8000/cuopt/solution/${REQID}" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" | jq .
```

## Server: Routing Request (Python with requests)

```python
import requests
import time

SERVER_URL = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/json",
    "CLIENT-VERSION": "custom"
}

# VRP problem data
payload = {
    "cost_matrix_data": {
        "data": {
            "0": [
                [0, 10, 15, 20, 25],
                [10, 0, 12, 18, 22],
                [15, 12, 0, 10, 15],
                [20, 18, 10, 0, 8],
                [25, 22, 15, 8, 0]
            ]
        }
    },
    "travel_time_matrix_data": {
        "data": {
            "0": [
                [0, 10, 15, 20, 25],
                [10, 0, 12, 18, 22],
                [15, 12, 0, 10, 15],
                [20, 18, 10, 0, 8],
                [25, 22, 15, 8, 0]
            ]
        }
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

# Submit request
response = requests.post(
    f"{SERVER_URL}/cuopt/request",
    json=payload,
    headers=HEADERS
)
response.raise_for_status()
req_id = response.json()["reqId"]
print(f"Request submitted: {req_id}")

# Poll for solution
max_attempts = 30
for attempt in range(max_attempts):
    response = requests.get(
        f"{SERVER_URL}/cuopt/solution/{req_id}",
        headers=HEADERS
    )
    result = response.json()

    if "response" in result:
        solver_response = result["response"].get("solver_response", {})
        print(f"\nSolution found!")
        print(f"Status: {solver_response.get('status', 'N/A')}")
        print(f"Cost: {solver_response.get('solution_cost', 'N/A')}")

        if "vehicle_data" in solver_response:
            for vid, vdata in solver_response["vehicle_data"].items():
                route = vdata.get("route", [])
                print(f"Vehicle {vid}: {' -> '.join(map(str, route))}")
        break
    else:
        print(f"Waiting... (attempt {attempt + 1})")
        time.sleep(1)
```

## Server: LP/MILP Request

```bash
# Submit LP problem via REST
# Production Planning: maximize 40*chairs + 30*tables
#   subject to: 2*chairs + 3*tables <= 240 (wood)
#               4*chairs + 2*tables <= 200 (labor)
#               chairs, tables >= 0
REQID=$(curl -s --location "http://localhost:8000/cuopt/request" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" \
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
    "solver_config": {
      "tolerances": {"optimality": 0.0001},
      "time_limit": 60
    }
  }' | jq -r '.reqId')

echo "Request ID: $REQID"

# Get solution
sleep 2
curl -s --location "http://localhost:8000/cuopt/solution/${REQID}" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" | jq .
```
