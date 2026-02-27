# Assets — reference C examples

LP/MILP C API reference implementations. Use as reference when building new applications; do not edit in place. Build requires cuOpt installed (see `cuopt-installation-api-c`).

| Example | Type | Description |
|---------|------|-------------|
| [lp_basic](lp_basic/) | LP | Simple LP: create problem, solve, get solution |
| [milp_basic](milp_basic/) | MILP | Simple MILP with integer variable |

Build (after setting `INCLUDE_PATH` and `LIB_PATH` to cuOpt):

```bash
gcc -I${INCLUDE_PATH} -L${LIB_PATH} -o lp_basic/lp_simple lp_basic/lp_simple.c -lcuopt
LD_LIBRARY_PATH=${LIB_PATH}:$LD_LIBRARY_PATH ./lp_basic/lp_simple
```
