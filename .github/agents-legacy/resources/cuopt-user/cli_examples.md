# CLI examples (cuOpt)

## CLI: LP from MPS File

```bash
# Create sample LP problem in MPS format
cat > production.mps << 'EOF'
* Production Planning Problem
* maximize 40*chairs + 30*tables
* s.t.    2*chairs + 3*tables <= 240 (wood)
*         4*chairs + 2*tables <= 200 (labor)
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

# Solve with cuopt_cli
cuopt_cli production.mps

# Solve with options
cuopt_cli production.mps --time-limit 30

# Cleanup
rm -f production.mps
```

## CLI: MILP from MPS File

```bash
# Create MILP problem (with integer variables)
cat > facility.mps << 'EOF'
* Facility location - simplified
* Binary variables for opening facilities
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

# Cleanup
rm -f facility.mps
```

## CLI: Common Options

```bash
# Show all options
cuopt_cli --help

# Set time limit (seconds)
cuopt_cli problem.mps --time-limit 120

# Set MIP relative gap tolerance (for MILP, e.g., 0.1% = 0.001)
cuopt_cli problem.mps --mip-relative-tolerance 0.001

# Set MIP absolute tolerance (for MILP)
cuopt_cli problem.mps --mip-absolute-tolerance 0.0001

# Enable presolve
cuopt_cli problem.mps --presolve

# Set iteration limit
cuopt_cli problem.mps --iteration-limit 10000

# Specify solver method (0=auto, 1=pdlp, 2=dual_simplex, 3=barrier, etc.)
cuopt_cli problem.mps --method 1
```
