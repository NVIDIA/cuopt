# Server: LP/MILP Examples

## LP Request (curl)

```bash
# maximize 40*x + 30*y
# s.t. 2x + 3y <= 240
#      4x + 2y <= 200

REQID=$(curl -s -X POST "http://localhost:8000/cuopt/request" \
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
    "solver_config": {
      "time_limit": 60
    }
  }' | jq -r '.reqId')

sleep 2
curl -s "http://localhost:8000/cuopt/solution/$REQID" -H "CLIENT-VERSION: custom" | jq .
```

## MILP Request (curl)

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
      "coefficients": [40.0, 30.0]
    },
    "variable_bounds": {
      "upper_bounds": ["inf", "inf"],
      "lower_bounds": [0.0, 0.0]
    },
    "variable_types": ["integer", "continuous"],
    "maximize": true,
    "solver_config": {
      "time_limit": 120,
      "tolerances": {
        "mip_relative_gap": 0.01
      }
    }
  }' | jq .
```

## LP Request (Python)

```python
import requests
import time

SERVER = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json", "CLIENT-VERSION": "custom"}

payload = {
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
        "coefficients": [40.0, 30.0]
    },
    "variable_bounds": {
        "upper_bounds": ["inf", "inf"],
        "lower_bounds": [0.0, 0.0]
    },
    "maximize": True,
    "solver_config": {"time_limit": 60}
}

resp = requests.post(f"{SERVER}/cuopt/request", json=payload, headers=HEADERS)
req_id = resp.json()["reqId"]

for _ in range(30):
    resp = requests.get(f"{SERVER}/cuopt/solution/{req_id}", headers=HEADERS)
    result = resp.json()
    if "response" in result:
        print(f"Status: {result['response'].get('status')}")
        print(f"Objective: {result['response'].get('objective_value')}")
        print(f"Solution: {result['response'].get('primal_solution')}")
        break
    time.sleep(1)
```

## CSR Matrix Format

```
Matrix:  [2, 3]    row 0
         [4, 2]    row 1

offsets: [0, 2, 4]       # row pointers
indices: [0, 1, 0, 1]    # column indices
values:  [2, 3, 4, 2]    # values
```

## Special Values

```json
"lower_bounds": ["ninf", "ninf"]   // negative infinity
"upper_bounds": ["inf", 100.0]     // positive infinity
```

## Variable Types

```json
"variable_types": ["continuous", "integer", "binary"]
```

## Response Format

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
