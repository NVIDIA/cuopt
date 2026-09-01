# Build and run `cuopt_cache_reuse_FSI`

Solver caching for barrier QP re-solves ([PR #1821](https://github.com/NVIDIA/cuopt/pull/1821)).

This is **not** in pip or conda. You must build this branch from source. Follow [cuOpt contributing: setting up your build environment](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md#setting-up-your-build-environment), with the changes below so you are **not** on `main`.

## Prerequisites

- Linux (x86_64 or aarch64)
- NVIDIA GPU, Volta or newer (compute capability ≥ 7.0)
- Driver that supports CUDA 12 or 13 (`nvidia-smi` → top-right “CUDA Version”)
- Python 3.11–3.14
- [Miniforge](https://conda-forge.org/download/) (or Miniconda). Prefer `mamba`. No `sudo`.
- Use `channel_priority: flexible`, not `strict`

If you have extra packages under `~/.local/lib/python*/site-packages`, they can shadow the build:

```bash
export PYTHONNOUSERSITE=1
```

## 1. Clone this branch (not `main`)

```bash
git clone https://github.com/NVIDIA/cuopt.git
cd cuopt
git fetch origin pull/1821/head:cuopt_cache_reuse_FSI
git checkout cuopt_cache_reuse_FSI
```

Or from the fork:

```bash
git clone https://github.com/Iroy30/NvidiaCuopt.git
cd NvidiaCuopt
git checkout cuopt_cache_reuse_FSI
```

Stay on this checkout for **both** the conda env file and the source.

## 2. Create the conda env from this tree

Match CUDA to your driver:

- CUDA 12.x → `conda/environments/all_cuda-129_arch-$(uname -m).yaml`
- CUDA 13.x → `conda/environments/all_cuda-133_arch-$(uname -m).yaml`

```bash
mamba env create -p ./.cuopt_env --file conda/environments/all_cuda-129_arch-$(uname -m).yaml
# or all_cuda-133_... for CUDA 13

mamba activate ./.cuopt_env
which nvcc    # must be the env's nvcc
```

CUDA compiles use a lot of RAM (~4–8 GB per job). Cap if needed:

```bash
export PARALLEL_LEVEL=8
```

## 3. Build and install C++ + Python

You need **both** `libcuopt` and the Python package (`update_linear_objective` / `sequence_solve` are in Python, not only `libcuopt.so`).

```bash
export PYTHONNOUSERSITE=1
./build.sh libcuopt cuopt --install
```

QP-only, faster:

```bash
./build.sh libcuopt cuopt --install --skip-routing-build --skip-tests-build
```

`--install` puts `libcuopt` into the active conda env (`$CONDA_PREFIX`).

## 4. Confirm this build is what Python loads

Same env you built in:

```bash
python -c "from cuopt.linear_programming import DataModel, SolverSettings, Solve; print('update_linear_objective', hasattr(DataModel(), 'update_linear_objective')); print('sequence_solve', hasattr(SolverSettings(), 'sequence_solve'))"
```

Expect `True` / `True`. If `update_linear_objective` is missing, Python is using a released wheel, not this build.

## 5. Run cache reuse

Same `DataModel` instance. Barrier + `sequence_solve`. `update_linear_objective` changes **only** the linear objective (`c`). Leave `A`, `Q`, and bounds unchanged. QP only (quadratic objective, no quadratic constraints / SOC). First solve must be Optimal.

```python
from cuopt.linear_programming import DataModel, SolverSettings, SolverMethod, Solve
from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD

dm = DataModel()
# set A, bounds, Q, and the first c as usual

settings = SolverSettings()
settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
settings.sequence_solve = True   # required

sol0 = Solve(dm, settings)       # cold: full convert/presolve/scaling; cache on Optimal

dm.update_linear_objective(new_c)               # only c; same dm
sol1 = Solve(dm, settings)       # reuse: skips convert/presolve/scaling
```

Warm solves still Mehrotra-start (they do not reuse the last iterate). Solver log line on reuse:

```text
Barrier: reusing cache (skip convert/presolve/scaling)
```

## 6. Optional: ADAT vs augmented bench

See **[README_ADAT_VS_AUG.md](README_ADAT_VS_AUG.md)**.

## Notes

- Do not `pip install` or conda-install a released `cuopt` and expect this API.
- Do not mix **current `main`’s** env yaml with this older branch; use the yaml from this checkout.
- If compile fails on `rmm::device_scalar` (deleted rvalue constructor), the tree you checked out is missing the RMM [#1795](https://github.com/NVIDIA/cuopt/pull/1795) fix. Use a branch that includes that commit, or rebuild against the env yaml shipped with this branch.
