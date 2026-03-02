# Assets — reference C examples

LP/MILP C API reference implementations. Use as reference when building new applications; do not edit in place. Build requires cuOpt installed (include and lib paths set).

| Example | Type | Description |
|---------|------|-------------|
| [lp_basic](lp_basic/) | LP | Simple LP: create problem, solve, get solution |
| [lp_duals](lp_duals/) | LP | Dual values and reduced costs |
| [lp_warmstart](lp_warmstart/) | LP | PDLP warmstart (see README) |
| [milp_basic](milp_basic/) | MILP | Simple MILP with integer variable |
| [milp_production_planning](milp_production_planning/) | MILP | Production planning with resource constraints |
| [mps_solver](mps_solver/) | LP/MILP | Solve from MPS file via `cuOptReadProblem` |

Build (after setting `INCLUDE_PATH` and `LIB_PATH` to cuOpt):

```bash
gcc -I${INCLUDE_PATH} -L${LIB_PATH} -o lp_basic/lp_simple lp_basic/lp_simple.c -lcuopt
LD_LIBRARY_PATH=${LIB_PATH}:$LD_LIBRARY_PATH ./lp_basic/lp_simple
```

Each subdirectory has its own README with build and run commands.
