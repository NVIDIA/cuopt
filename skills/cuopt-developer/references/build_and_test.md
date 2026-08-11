# Build & Test

Read this for component-level build commands, run-test commands, and `PARALLEL_LEVEL` detail. **Pre-flight checks** (CUDA driver compatibility, conda env activation, dataset setup) live in [SKILL.md → Build & Test → Pre-flight Checks](../SKILL.md#pre-flight-checks-required-before-first-build-or-test) — always run those first.

## PARALLEL_LEVEL

`PARALLEL_LEVEL` controls the number of parallel compile jobs. It defaults to `$(nproc)` (all cores), which can cause OOM on machines with limited RAM — CUDA compilation needs roughly 4–8 GB per job. Set it based on available RAM:

```bash
export PARALLEL_LEVEL=8   # adjust based on available RAM
```

**Inside a container (Docker / Kubernetes), `nproc` and `free` report the *host's* cores and RAM, not the container's limits.** Sizing `PARALLEL_LEVEL` from them over-subscribes a memory-capped pod and gets the build OOM-killed (exit 137), while a CPU quota silently throttles the surplus jobs. First detect whether you are containerized, then read the real caps from the cgroup and budget from whichever is tighter — ~4–8 GB per CUDA job against `memory.max`, and no more jobs than `cpu.max` grants:

```bash
# 1. Detect containerization — any hit means nproc/free report the host, not your limits.
grep -qaE 'kubepods|docker|containerd|libpod|lxc' /proc/1/cgroup 2>/dev/null && echo containerized
[ -f /.dockerenv ] && echo docker                     # Docker
[ -n "$KUBERNETES_SERVICE_HOST" ] && echo kubernetes  # Kubernetes (also: kubepods in the grep above)

# 2. Read the real caps from the cgroup (v2). If memory.max is absent at the mount
#    root you are seeing the host root cgroup, so resolve your leaf via /proc/self/cgroup.
CG=/sys/fs/cgroup
[ -f "$CG/memory.max" ] || CG=/sys/fs/cgroup$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)
cat "$CG/memory.max"   # bytes, or "max" = no limit
cat "$CG/cpu.max"      # "<quota> <period>"; effective cores = quota / period
```

## Build Everything

```bash
./build.sh
```

By default, `build.sh` builds without installing `libcuopt` into the conda environment. This prevents stale installed libraries from shadowing freshly compiled code. Pass `--install` to explicitly install into the active conda environment:

```bash
./build.sh --install      # build and install libcuopt into conda env
```

## Build Specific Components

```bash
./build.sh --help                                       # Lists build options
./build.sh libcuopt                                     # C++ library (no conda install)
./build.sh libcuopt --skip-routing-build --skip-tests-build --skip-c-python-adapters --cache-tool=ccache  # native LP/MIP-focused build without routing/tests/adapters
./build.sh cuopt                                        # Python package
./build.sh cuopt_server                                 # Server
./build.sh docs                                         # Documentation
```

## Run Tests

> Activate the conda env used to build first (`conda activate <env-name>`) and ensure datasets are fetched — see [Pre-flight Checks](../SKILL.md#pre-flight-checks-required-before-first-build-or-test) in SKILL.md.

```bash
# C++ tests
ctest --test-dir cpp/build

# Python tests
pytest -v python/cuopt/cuopt/tests

# Server tests
pytest -v python/cuopt_server/tests
```
