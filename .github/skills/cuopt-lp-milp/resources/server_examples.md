# LP/MILP: REST Server Examples

## LP Request (curl)

```bash
# Production Planning LP via REST
# maximize 40*chairs + 30*tables
# s.t. 2*chairs + 3*tables <= 240
#      4*chairs + 2*tables <= 200

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
      "tolerances": {"optimality": 0.0001},
      "time_limit": 60
    }
  }' | jq -r '.reqId')

echo "Request ID: $REQID"

# Get solution
sleep 2
curl -s "http://localhost:8000/cuopt/solution/$REQID" \
  -H "CLIENT-VERSION: custom" | jq .
```

## MILP Request (curl)

```bash
# Add integer variable types
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
      "mip_relative_gap": 0.01
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
        "coefficients": [40.0, 30.0],
        "scalability_factor": 1.0,
        "offset": 0.0
    },
    "variable_bounds": {
        "upper_bounds": ["inf", "inf"],
        "lower_bounds": [0.0, 0.0]
    },
    "maximize": True,
    "solver_config": {
        "time_limit": 60
    }
}

# Submit
response = requests.post(f"{SERVER}/cuopt/request", json=payload, headers=HEADERS)
req_id = response.json()["reqId"]
print(f"Submitted: {req_id}")

# Poll for solution
for _ in range(30):
    response = requests.get(f"{SERVER}/cuopt/solution/{req_id}", headers=HEADERS)
    result = response.json()
    
    if "response" in result:
        print(f"Status: {result['response'].get('status')}")
        print(f"Objective: {result['response'].get('objective_value')}")
        print(f"Solution: {result['response'].get('primal_solution')}")
        break
    time.sleep(1)
```

## CSR Matrix Format

The constraint matrix uses Compressed Sparse Row (CSR) format:

```
Matrix:  [2, 3]    (row 0: 2*x0 + 3*x1)
         [4, 2]    (row 1: 4*x0 + 2*x1)

CSR format:
  offsets: [0, 2, 4]           # Row pointers
  indices: [0, 1, 0, 1]        # Column indices
  values:  [2.0, 3.0, 4.0, 2.0] # Non-zero values
```

## Special Values

```json
{
  "constraint_bounds": {
    "lower_bounds": ["ninf", "ninf"],  // -infinity
    "upper_bounds": [100.0, "inf"]      // +infinity
  }
}
```

## Variable Types

```json
{
  "variable_types": ["continuous", "integer", "binary"]
}
```

- `"continuous"` - real-valued
- `"integer"` - integer-valued  
- `"binary"` - 0 or 1 only
