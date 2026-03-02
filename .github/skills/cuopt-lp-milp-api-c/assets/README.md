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

## Build and run

Set include and library paths, then build and run.

**Using conda:** Activate your cuOpt env first (`conda activate cuopt`), then:

```bash
# Paths from active conda env (CONDA_PREFIX is set when env is activated)
export INCLUDE_PATH="${CONDA_PREFIX}/include"
export LIB_PATH="${CONDA_PREFIX}/lib"
export LD_LIBRARY_PATH="${LIB_PATH}:${LD_LIBRARY_PATH}"

# Build (from this assets/ directory)
gcc -I"${INCLUDE_PATH}" -L"${LIB_PATH}" -o lp_basic/lp_simple       lp_basic/lp_simple.c -lcuopt
gcc -I"${INCLUDE_PATH}" -L"${LIB_PATH}" -o lp_duals/lp_duals         lp_duals/lp_duals.c -lcuopt
gcc -I"${INCLUDE_PATH}" -L"${LIB_PATH}" -o milp_basic/milp_simple   milp_basic/milp_simple.c -lcuopt
gcc -I"${INCLUDE_PATH}" -L"${LIB_PATH}" -o milp_production_planning/milp_production milp_production_planning/milp_production.c -lcuopt
gcc -I"${INCLUDE_PATH}" -L"${LIB_PATH}" -o mps_solver/mps_solver     mps_solver/mps_solver.c -lcuopt

# Run
./lp_basic/lp_simple
./lp_duals/lp_duals
./milp_basic/milp_simple
./milp_production_planning/milp_production
./mps_solver/mps_solver mps_solver/data/sample.mps
```

Without conda, set `INCLUDE_PATH` and `LIB_PATH` to your cuOpt include and lib directories, then use the same `gcc` and `LD_LIBRARY_PATH` as above. Each subdirectory README has a one-line build/run for that example.
