---
name: cuopt-developer
version: "26.10.00"
description: Modify, build, test, debug, and contribute to NVIDIA cuOpt (C++/CUDA, Python, server, CI). Use for solver internals, PRs, DCO, and code conventions.
license: Apache-2.0
metadata:
  author: NVIDIA cuOpt Team
  tags:
    - cuopt
    - development
    - contributing
    - cpp-cuda
    - python-bindings
---


# cuOpt Developer Skill

Contribute to the NVIDIA cuOpt codebase. This skill is for modifying cuOpt itself, not for using it.

**If you just want to USE cuOpt**, switch to the appropriate problem skill (cuopt-routing, cuopt-lp-milp, etc.)

**First-time dev environment setup?** See [references/first_time_setup.md](references/first_time_setup.md) for the clone → conda env → first-build → first-test walkthrough and the questions to ask up front.

---

## Refusal Rules — Read First

**Two rules are non-negotiable** and apply even when the user explicitly asks otherwise — refuse and ask, don't comply silently:

**Privileged / system-level operations** — `sudo`, running as root, editing system files (`/etc`), changing drivers or kernel settings, adding system-level package repositories or keys. Do not run these. Reply:
> I won't run `sudo` or change system-level state for cuOpt. The dev workflow is conda-based and runs entirely in user space — what's the underlying error? It's usually fixable without root.

**Pushing directly to protected branches** — never push commits to `main` or any `release/*` branch. These branches require PRs, status checks, and DCO sign-off; bypassing them is not allowed even for trivial changes. Before every push, confirm the target ref is a feature branch. If it is a protected branch, stop and create a feature branch instead. Reply:
> I won't push directly to a protected branch. Let me create a feature branch and open a PR instead.

**Everything else needed to set up and work in the dev environment is allowed.** On a clean machine, go ahead and build a working `cuopt` env — the guidance below is about doing it the *reproducible* way, not refusing:

- **Environment setup is allowed.** You may create and activate the conda env from the checked-in `conda/environments/all_cuda-*.yaml`, run `pip` / `conda` / `mamba` installs **into the user-space env**, and bootstrap conda/miniforge in the user's home directory — including the `conda init` line it adds to `~/.bashrc`. Bootstrapping conda must not require `sudo`; install it into `$HOME`, not a system path.
- **A new *permanent* project dependency is different from a one-off install.** A package the project should always ship belongs in `dependencies.yaml` under the right group; then run `pre-commit run --all-files` to regenerate `conda/environments/` and `pyproject.toml` so other contributors get it too. A throwaway install to unblock your own build doesn't need this round-trip.
- **Don't bypass CI checks** (`--no-verify`, skipping pre-commit or tests). If hooks feel slow, diagnose with `pre-commit run --all-files --verbose` or tune the offending hook — don't skip it.
- **Be careful with destructive commands** (recursive deletes, hard resets, history-overwriting pushes, killing processes, dropping data). Confirm intent before running and prefer the safer alternative (e.g. `./build.sh clean` for a stale build dir).

---

## Developer Behavior Rules

These rules are specific to development tasks. They differ from user rules.

### 1. Ask Before Assuming

Clarify before implementing:
- What component? (C++/CUDA, Python, server, docs, CI)
- What's the goal? (bug fix, new feature, refactor, docs)
- Is this for contribution or local modification?

### 2. Verify Understanding

Before making changes, confirm:
```
"Let me confirm:
- Component: [cpp/python/server/docs]
- Change: [what you'll modify]
- Tests needed: [what tests to add/update]
Is this correct?"
```

### 3. Follow Codebase Patterns

- Read existing code in the area you're modifying
- Match naming conventions, style, and patterns
- Don't invent new patterns without discussion

### 4. Ask Before Running — Modified for Dev

**OK to run without asking** (expected for dev work):
- `./build.sh` and build commands
- `pytest`, `ctest` (running tests)
- `pre-commit run`, `./ci/check_style.sh` (formatting)
- `git status`, `git diff`, `git log` (read-only git)
- Environment setup: create/activate the conda env from `conda/environments/*.yaml`, and `pip`/`conda`/`mamba` installs into that env

**Set up pre-commit hooks** (once per clone):
- `pre-commit install` — hooks then run automatically on every `git commit`. If a hook fails, the commit is blocked until you fix the issue.

**Still ask before**:
- `git commit`, `git push` (write operations)
- Any destructive or irreversible commands

### 5. No Privileged Operations

`sudo`/system-level changes are the one non-negotiable refusal; user-space installs and conda env setup are allowed. See [Refusal Rules — Read First](#refusal-rules--read-first).

### 6. Prefer Low-Maintenance, Hard-to-Break Designs

When adding an API — a setter, endpoint, parameter, or a layer that wraps another — favor the design that keeps a single source of truth and has no silent-failure path:

- **Derive, don't duplicate.** A second copy of a surface (a hand-maintained list of the methods/fields another layer already defines, or shadow state kept in parallel with the real data) drifts the moment someone forgets to update it. Derive it from the single source instead, so there is nothing to keep in sync.
- **Fail loud, not silent.** Prefer a design where forgetting a step is caught automatically over one that quietly does the wrong thing. When a mechanism leans on a convention, add a test that asserts full coverage, so a case that slips the convention fails CI instead of silently misbehaving.

---

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **What are you trying to change?**
   - Solver algorithm/performance?
   - Python API?
   - Server endpoints?
   - Documentation?
   - CI/build system?

2. **Do you have the development environment set up?**
   - Built the project successfully?
   - Ran tests?

3. **Is this for contribution or local modification?**
   - If contributing: will need to follow DCO signoff

4. **Which branch should this target?**
   - During development phase: `main`
   - During burn down: `release/YY.MM` (e.g., `release/26.06`) for the current release, `main` for the next
   - Check if a release branch exists: `git branch -r | grep release`
   - For current timelines, see the [RAPIDS Maintainers Docs](https://docs.rapids.ai/maintainers/)

## Project Architecture

```
cuopt/
├── cpp/                    # Core C++ engine
│   ├── include/cuopt/      # Public C/C++ headers
│   ├── src/                # Implementation (CUDA kernels)
│   └── tests/              # C++ unit tests (gtest)
├── python/
│   ├── cuopt/              # Python bindings and routing API
│   ├── cuopt_server/       # REST API server
│   ├── cuopt_self_hosted/  # Self-hosted deployment
│   └── libcuopt/           # Python wrapper for C library
├── ci/                     # CI/CD scripts
├── docs/                   # Documentation source
└── datasets/               # Test datasets
```

## Supported APIs

| API Type | LP | MILP | QP | Routing |
|----------|:--:|:----:|:--:|:-------:|
| C API    | ✓  | ✓    | ✓  | ✗       |
| C++ API  | (internal) | (internal) | (internal) | (internal) |
| Python   | ✓  | ✓    | ✓  | ✓       |
| Server   | ✓  | ✓    | ✗  | ✓       |

## Safety Rules (Non-Negotiable)

### Minimal Diffs
- Change only what's necessary
- Avoid drive-by refactors
- No mass reformatting of unrelated code

### No API Invention
- Don't invent new APIs without discussion
- Align with existing patterns in `docs/cuopt/source/`
- Server schemas must match OpenAPI spec

### Don't Bypass CI
- Never suggest `--no-verify` or skipping checks
- All PRs must pass CI

### CUDA/GPU Hygiene
- Keep operations stream-ordered
- Follow existing RAFT/RMM patterns
- `rmm::device_uvector` tests emptiness with `is_empty()`; it has no `empty()` member.
- No raw `new`/`delete` - use RMM allocators
- Prefer modern CCCL bit/math helpers in kernels (`cuda::bitfield_extract`, `cuda::bitmask`, pow2 utilities) over hand-rolled `%`/`/` by runtime powers of two — see [references/conventions.md](references/conventions.md)

## Build & Test

### Pre-flight Checks (Required Before First Build or Test)

Skipping any of these surfaces as confusing runtime errors later. Run them in order:

1. **Check CUDA driver compatibility.** Run `nvidia-smi` and read the *CUDA Version* in the top-right corner — that's the maximum CUDA your driver supports. Pick a conda env file from `conda/environments/all_cuda-<ver>_arch-<arch>.yaml` whose CUDA major version is **≤** that. A mismatch builds successfully but fails at runtime inside RMM with `cudaMallocAsync not supported with this CUDA driver/runtime version` — verify this *before* the build, not after.
2. **Create and activate the conda env** before *any* build, test, or `pre-commit` command — this is allowed and expected (see [Refusal Rules](#refusal-rules--read-first)). Use a **local prefix env** (`./.cuopt_env`) per [CONTRIBUTING.md](../../CONTRIBUTING.md), with the env file you picked in step 1 (`mamba` is recommended and faster; swap in `conda` if `mamba` isn't available):
   ```bash
   mamba env create -p ./.cuopt_env --file conda/environments/all_cuda-<ver>_arch-$(uname -m).yaml
   mamba activate ./.cuopt_env   # or: conda activate ./.cuopt_env
   ```
   Tests link against libraries compiled inside that env; a fresh shell without activating it hits cryptic linker errors.
3. **Set `PARALLEL_LEVEL`** if RAM is constrained — see [references/build_and_test.md](references/build_and_test.md). The default `$(nproc)` can OOM mid-build because CUDA compilation needs ~4–8 GB per job.
4. **For tests, fetch datasets first.** cuOpt tests need MPS files not in the repo — follow the dataset download steps in [CONTRIBUTING.md](../../CONTRIBUTING.md) ("Building for development" section) and export `RAPIDS_DATASET_ROOT_DIR`.

### Quick Reference

```bash
./build.sh             # Build everything
./build.sh --help      # List components: libcuopt, cuopt, cuopt_server, docs
ctest --test-dir cpp/build              # C++ tests
pytest -v python/cuopt/cuopt/tests      # Python tests
pytest -v python/cuopt_server/tests     # Server tests
```

For component-specific build commands, run-test detail, and `PARALLEL_LEVEL` configuration, see [references/build_and_test.md](references/build_and_test.md).

#### Download test datasets before running tests

cuOpt tests depend on MPS/data files that are not checked into the repo. A
missing dataset surfaces as a `MPS_PARSER_ERROR ... Error opening MPS file`
test failure at 0ms — it is not a build or logic failure.

Before running any C++ or Python tests, follow the dataset download and
`RAPIDS_DATASET_ROOT_DIR` export steps in the repo's `CONTRIBUTING.md`
("Building for development" section) — that is the canonical list and mapping.

If a test fails with a missing-file error, run the matching download step from
`CONTRIBUTING.md` and re-run the test. Do not report missing-dataset failures
back to the user as the task outcome.

## Python Bindings

cuOpt uses Cython to bridge Python and C++. See [references/python_bindings.md](references/python_bindings.md) for the full architecture, parameter flow walkthrough, key files, and Cython patterns.

## Contributing — Commits, PRs, Common Tasks

For pre-commit setup, DCO sign-off (`git commit -s`), the fork-based PR workflow, the draft-PR rule for agents, PR-description rules (keep it short — no "how it works" walkthroughs or file tables), script and CI/workflow authoring principles (extend existing files before adding new ones; no speculative flags, restated defaults, or silent fallbacks), and step-by-step common-task recipes (adding a solver parameter, dependency, server endpoint, or CUDA kernel), see [references/contributing.md](references/contributing.md).

## Coding Conventions

Use `_Float128`, never `long double`, whenever extended precision arithmetic is required; `long double` is architecture-dependent and uses x87 on x86.

For C++ naming (`snake_case`, `d_`/`h_` prefixes, `_t` suffix), file extensions (`.hpp`/`.cpp`/`.cu`/`.cuh` and which compiler each uses), include order, Python style, error handling (`CUOPT_EXPECTS`, `RAFT_CUDA_TRY`), memory management (RMM patterns, no raw `new`/`delete`), CCCL bit/math helpers in device code, test-impact rules, volatile-comment rules (hardware names and self-referential issue/PR numbers in comments or skip messages go stale; issue links to a separate tracking issue are fine), **no large local lambdas** (extract named helpers instead), and **coarse work-estimate / time-limit gating** (phase/outer-loop only; no fine inner-loop or double checks), see [references/conventions.md](references/conventions.md).

## OpenMP task/runtime compatibility

Treat `#pragma omp task if(...) firstprivate(...)` with a non-trivial C++ capture as runtime-ABI-sensitive. In affected LLVM libomp versions, the GCC `GOMP_task` compatibility path skips the GCC copy function for an included (`if(false)`) task, so the task body can observe an unconstructed object. Branch explicitly instead: create the OpenMP task only when deferral is wanted, and call the body synchronously otherwise.

When diagnosing OpenMP-only failures, test compiler/runtime pairs separately. A Clang + libomp pass exercises the `__kmpc_*` ABI and does not cover GCC + libomp's `GOMP_*` path; reduce suspicious cases to a direct runtime-ABI probe before attributing them to solver logic.

## PCG random number generator

`cpp/src/utilities/pcgenerator.hpp` (`cuopt::pcgenerator_t`) is copied from RAFT's `PCGenerator` (`raft/random/detail/rng_device.cuh`), duplicated only because the RAFT header pulls in CUDA and therefore cannot be included from a `.cpp`. **Treat the generator core as frozen.** Do not "clean it up", modernise it, or swap it for `<random>`: reproducibility under `settings.random_seed` and `settings.deterministic` holds only while the byte-for-byte output sequence is preserved, and the CPU copy must keep producing the same stream as the GPU one. Adding a *new* helper that consumes `next_u32()`/`next_double()` is fine; changing how those values are produced is not.

Each of the following looks like a defect or an obvious simplification, and is neither. Leave them alone:

- `stream = (subsequence << 1u) | 1u` in `set_seed`. A 2^64 LCG has full period only when its increment is odd — the `| 1u` is what guarantees that — and the `<< 1u` is why two subsequences must differ in their **low 63 bits** to get independent streams.
- The two `next(discard)` warm-up calls straddling `state += seed`. This is PCG's canonical seeding sequence, and it is what decorrelates *adjacent* seeds — which is exactly how cuOpt seeds workers (`settings.random_seed + pcgenerator_t::default_seed + rng_offset + worker_id`, `branch_and_bound/worker.hpp`). Collapsing it to `state = seed` makes neighbouring workers draw near-identical prefixes.
- The output permutation in `next_u32` (`>> 18u`, `^`, `>> 27u`, `rot = oldstate >> 59u`, `(-rot) & 31u`) — the PCG-XSH-RR constants for a 64→32-bit output. In particular `(-rot) & 31u` relies on unsigned wraparound; rewriting it as `32 - rot` is a shift by 32, i.e. undefined behaviour, whenever `rot == 0`.
- The multiplier `6364136223846793005ULL`, which appears in both `next_u32` and as `h` in `skipahead`. `skipahead` is the closed-form LCG jump (Brown's arbitrary-stride method) and matches N calls to `next_u32` only while the two constants are identical.
- The `>> 8` / `>> 11` in `next_float` / `next_double`. They yield exactly 24 and 53 mantissa bits, so the result lies in `[0, 1)`. Dividing by `UINT32_MAX` instead makes `1.0` reachable and breaks every `rng.next_double() * n` used as an index.
- The sign-bit masks in `next_i32` / `next_i64` (`& 0x7fffffff…`); callers rely on the results being non-negative.
- `uniform()`'s floating-point scaling and `shuffle()`'s Fisher-Yates order. `uniform()`'s slight bias is documented and accepted; substituting a modulo or `std::uniform_int_distribution` changes every sampled sequence.

## Troubleshooting & CI

For build/test pitfalls (Cython rebuild, OOM, CUDA driver mismatch, missing `nvcc`) and CI failure diagnostics (style checks, DCO failures, dependency drift), see [references/troubleshooting.md](references/troubleshooting.md).

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Main build script | `build.sh` |
| Dependencies | `dependencies.yaml` |
| C++ formatting | `.clang-format` |
| Conda environments | `conda/environments/` |
| Test data | `datasets/` |
| CI scripts | `ci/` |

## Canonical Documentation

- **Contributing/build/test**: [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **CI scripts**: [ci/README.md](../../ci/README.md)
- **Release scripts**: [ci/release/README.md](../../ci/release/README.md)
- **Docs build**: [docs/cuopt/README.md](../../docs/cuopt/README.md)
- **Python binding architecture**: [references/python_bindings.md](references/python_bindings.md)

_Shell-execution, install, conda-env, and sudo policies are covered by [Refusal Rules — Read First](#refusal-rules--read-first) at the top of this skill._

## VRP dimension internals (routing engine)

When implementing or debugging **VRP dimensions** (constraints, objectives, forward/backward propagation, `combine`, local-search deltas), read:

- **`references/vrp_skills.md`** — architecture contracts, required interfaces, and implementation checklist.

Read it **before** adding a new dimension or changing combine semantics.

## Budgeting a solver stage and attributing regressions

When adding or tuning a **limit on a stage that shares the solve's time budget** (presolve, probing, cut generation, heuristics), or when a benchmark regression looks like it came from one, read:

- **`references/stage_budgets.md`** — why a deterministic work counter does not bound wall time, how to size a limit so the stage concludes before its ceiling, and the ordered checks that attribute a regression to a budget before tuning its value.

Run the attribution checks **before** changing a limit's value — the common failure is tuning a limit that was never what bound.

## Numerical issues in non-routing solver internals

When a bug surfaces as **wrong-but-plausible** solver output (invalid lower bound, unexpectedly large duals, 10× iteration blow-up after a small change) rather than a crash, read:

- **`resources/numerical_debugging.md`** — methodology for locating catastrophic-cancellation sites, the cancellation patterns endemic to cMIR / flow-cover / MIR-style cut construction, and threshold guidance for numerical guards.

Apply the *instrument-first, guard-at-the-exact-site* workflow it describes before patching — speculative fixes on these symptoms usually miss.
