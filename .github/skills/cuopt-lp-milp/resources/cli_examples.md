# LP/MILP: CLI Examples

## LP from MPS File

```bash
# Create sample LP in MPS format
cat > production.mps << 'EOF'
* Production Planning: maximize 40*chairs + 30*tables
* s.t. 2*chairs + 3*tables <= 240 (wood)
*      4*chairs + 2*tables <= 200 (labor)
NAME          PRODUCTION
ROWS
 N  PROFIT
 L  WOOD
 L  LABOR
COLUMNS
    CHAIRS    PROFIT           -40.0
    CHAIRS    WOOD               2.0
    CHAIRS    LABOR              4.0
    TABLES    PROFIT           -30.0
    TABLES    WOOD               3.0
    TABLES    LABOR              2.0
RHS
    RHS1      WOOD             240.0
    RHS1      LABOR            200.0
ENDATA
EOF

# Solve
cuopt_cli production.mps

# With time limit
cuopt_cli production.mps --time-limit 30
```

## MILP from MPS File

```bash
# Create MILP with integer variables
cat > facility.mps << 'EOF'
* Facility location with binary variables
NAME          FACILITY
ROWS
 N  COST
 G  DEMAND1
 L  CAP1
 L  CAP2
COLUMNS
    MARKER    'MARKER'         'INTORG'
    OPEN1     COST             100.0
    OPEN1     CAP1              50.0
    OPEN2     COST             150.0
    OPEN2     CAP2              70.0
    MARKER    'MARKER'         'INTEND'
    SHIP11    COST               5.0
    SHIP11    DEMAND1            1.0
    SHIP11    CAP1              -1.0
    SHIP21    COST               7.0
    SHIP21    DEMAND1            1.0
    SHIP21    CAP2              -1.0
RHS
    RHS1      DEMAND1           30.0
BOUNDS
 BV BND1      OPEN1
 BV BND1      OPEN2
 LO BND1      SHIP11             0.0
 LO BND1      SHIP21             0.0
ENDATA
EOF

# Solve MILP
cuopt_cli facility.mps --time-limit 60 --mip-relative-tolerance 0.01
```

## Common CLI Options

```bash
# Show all options
cuopt_cli --help

# Time limit (seconds)
cuopt_cli problem.mps --time-limit 120

# MIP gap tolerance (stop when within X% of optimal)
cuopt_cli problem.mps --mip-relative-tolerance 0.001

# MIP absolute tolerance
cuopt_cli problem.mps --mip-absolute-tolerance 0.0001

# Enable presolve
cuopt_cli problem.mps --presolve

# Iteration limit
cuopt_cli problem.mps --iteration-limit 10000

# Solver method (0=auto, 1=pdlp, 2=dual_simplex, 3=barrier)
cuopt_cli problem.mps --method 1
```

## MPS Format Reference

### Required Sections (in order)

```
NAME          problem_name
ROWS
 N  objective_row    (N = free/objective)
 L  constraint1      (L = <=)
 G  constraint2      (G = >=)
 E  constraint3      (E = ==)
COLUMNS
    var1    row1    coefficient
    var1    row2    coefficient
RHS
    rhs1    row1    value
ENDATA
```

### Optional: BOUNDS Section

```
BOUNDS
 LO bnd1    var1    0.0       (lower bound)
 UP bnd1    var1    100.0     (upper bound)
 FX bnd1    var2    50.0      (fixed value)
 FR bnd1    var3              (free variable)
 BV bnd1    var4              (binary 0/1)
 LI bnd1    var5    0         (integer, lower bound)
 UI bnd1    var5    10        (integer, upper bound)
```

### Integer Markers

```
COLUMNS
    MARKER    'MARKER'         'INTORG'
    int_var1  OBJ              1.0
    int_var2  OBJ              2.0
    MARKER    'MARKER'         'INTEND'
    cont_var  OBJ              3.0
```

## Troubleshooting

**"Failed to parse MPS file"**
- Check for missing ENDATA
- Verify section order: NAME, ROWS, COLUMNS, RHS, [BOUNDS], ENDATA
- Check integer markers format

**"Problem is infeasible"**
- Check constraint directions (L/G/E)
- Verify RHS values are consistent

---

## Additional References (tested in CI)

For more complete examples, read these files:

| Example | File | Description |
|---------|------|-------------|
| Basic LP | `docs/cuopt/source/cuopt-cli/examples/lp/examples/basic_lp_example.sh` | Simple LP via CLI |
| Basic MILP | `docs/cuopt/source/cuopt-cli/examples/milp/examples/basic_milp_example.sh` | MILP with integers |
| Solver Parameters | `docs/cuopt/source/cuopt-cli/examples/lp/examples/solver_parameters_example.sh` | CLI options |

These examples are tested by CI and represent canonical usage.
