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

echo "Request ID: $REQID"

# Poll for solution (MILP may take longer than LP)
while true; do
  RESULT=$(curl -s "http://localhost:8000/cuopt/solution/$REQID" \
    -H "CLIENT-VERSION: custom")
  STATUS=$(echo "$RESULT" | jq -r '.response.status // empty')
  if [ -n "$STATUS" ]; then
    echo "$RESULT" | jq .
    break
  fi
  sleep 2
done
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
    "lower_bounds": ["ninf", "ninf"],
    "upper_bounds": [100.0, "inf"]
  }
}
```

- `"ninf"` — negative infinity (-∞)
- `"inf"` — positive infinity (+∞)

## Variable Types

```json
{
  "variable_types": ["continuous", "integer", "binary"]
}
```

- `"continuous"` - real-valued
- `"integer"` - integer-valued
- `"binary"` - 0 or 1 only

---

## Additional References (tested in CI)

For more complete examples, read these files:

| Example | File | Description |
|---------|------|-------------|
| Basic LP (Python) | `docs/cuopt/source/cuopt-server/examples/lp/examples/basic_lp_example.py` | LP via REST |
| Basic LP (curl) | `docs/cuopt/source/cuopt-server/examples/lp/examples/basic_lp_example.sh` | LP shell script |
| MPS Input | `docs/cuopt/source/cuopt-server/examples/lp/examples/mps_file_example.py` | MPS file format |
| MPS DataModel | `docs/cuopt/source/cuopt-server/examples/lp/examples/mps_datamodel_example.py` | MPS in payload |
| Warmstart | `docs/cuopt/source/cuopt-server/examples/lp/examples/warmstart_example.py` | Warm starting |
| Basic MILP (Python) | `docs/cuopt/source/cuopt-server/examples/milp/examples/basic_milp_example.py` | MILP via REST |
| Basic MILP (curl) | `docs/cuopt/source/cuopt-server/examples/milp/examples/basic_milp_example.sh` | MILP shell script |
| Incumbent Callback | `docs/cuopt/source/cuopt-server/examples/milp/examples/incumbent_callback_example.py` | MIP progress tracking |
| Abort Job | `docs/cuopt/source/cuopt-server/examples/milp/examples/abort_job_example.py` | Canceling requests |
| Batch Mode | `docs/cuopt/source/cuopt-server/examples/lp/examples/batch_mode_example.sh` | Multiple problems |

These examples are tested by CI (`ci/test_doc_examples.sh`) and represent canonical usage.
