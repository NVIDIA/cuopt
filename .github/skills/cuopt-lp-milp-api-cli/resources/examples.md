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
cuopt_cli --help

# Time limit (seconds)
cuopt_cli problem.mps --time-limit 120

# MIP gap tolerance
cuopt_cli problem.mps --mip-relative-tolerance 0.001

# MIP absolute tolerance
cuopt_cli problem.mps --mip-absolute-tolerance 0.0001

# Presolve, iteration limit, method (0=auto, 1=pdlp, 2=dual_simplex, 3=barrier)
cuopt_cli problem.mps --presolve --iteration-limit 10000 --method 1
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
RHS
    rhs1    row1    value
ENDATA
```

### BOUNDS (optional)

```
BOUNDS
 LO bnd1    var1    0.0       (lower bound)
 UP bnd1    var1    100.0     (upper bound)
 FX bnd1    var2    50.0      (fixed)
 BV bnd1    var4              (binary 0/1)
 LI bnd1    var5    0         (integer lower)
 UI bnd1    var5    10        (integer upper)
```

### Integer markers

```
COLUMNS
    MARKER    'MARKER'         'INTORG'
    int_var1  OBJ              1.0
    MARKER    'MARKER'         'INTEND'
```

## Troubleshooting

- **Failed to parse MPS** — Check ENDATA, section order (NAME, ROWS, COLUMNS, RHS, [BOUNDS], ENDATA), integer markers.
- **Infeasible** — Check constraint directions (L/G/E) and RHS values.

## Canonical examples (in repo)

- `docs/cuopt/source/cuopt-cli/examples/lp/examples/basic_lp_example.sh`
- `docs/cuopt/source/cuopt-cli/examples/milp/examples/basic_milp_example.sh`
- `docs/cuopt/source/cuopt-cli/examples/lp/examples/solver_parameters_example.sh`
