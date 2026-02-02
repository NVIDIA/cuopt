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
# Submit MILP request and capture reqId
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
  }' | jq -r '.reqId')
# Note: objective_data also supports optional "scalability_factor" and "offset" fields

# Poll for solution (MILP may take longer than LP)
sleep 3
curl -s "http://localhost:8000/cuopt/solution/$REQID" -H "CLIENT-VERSION: custom" | jq .
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

```
Matrix:  [2, 3]    (row 0: 2*x0 + 3*x1)
         [4, 2]    (row 1: 4*x0 + 2*x1)

CSR format:
  offsets: [0, 2, 4]           # Row pointers (n_rows + 1)
  indices: [0, 1, 0, 1]        # Column indices
  values:  [2.0, 3.0, 4.0, 2.0] # Non-zero values
```

## Special Values

```json
{
  "constraint_bounds": {
    "lower_bounds": ["ninf", "ninf"],
    "upper_bounds": [100.0, "inf"]
  }
}
```

## Variable Types

- `"continuous"` - real-valued
- `"integer"` - integer-valued
- `"binary"` - 0 or 1 only

---

## Additional References (tested in CI)

For more complete examples, read these files:

| Example | File |
|---------|------|
| Basic LP (Python) | `docs/cuopt/source/cuopt-server/examples/lp/examples/basic_lp_example.py` |
| Basic LP (curl) | `docs/cuopt/source/cuopt-server/examples/lp/examples/basic_lp_example.sh` |
| MPS File Input | `docs/cuopt/source/cuopt-server/examples/lp/examples/mps_file_example.py` |
| Warmstart | `docs/cuopt/source/cuopt-server/examples/lp/examples/warmstart_example.py` |
| Basic MILP | `docs/cuopt/source/cuopt-server/examples/milp/examples/basic_milp_example.py` |
| Incumbent Callback | `docs/cuopt/source/cuopt-server/examples/milp/examples/incumbent_callback_example.py` |

These examples are tested by CI (`ci/test_doc_examples.sh`).
