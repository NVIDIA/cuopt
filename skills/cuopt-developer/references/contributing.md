# Contributing — Commits, PRs, and Common Tasks

Read this for anything related to committing, pushing, opening PRs, or making structural changes to cuOpt (adding a solver parameter, dependency, server endpoint, or CUDA kernel).

## Before You Commit

### 1. Install Pre-commit Hooks

Run once per clone to have style checks run automatically on every `git commit`:

```bash
pre-commit install
```

If a hook fails, the commit is blocked — fix the issues and commit again. To check all files manually (e.g., before pushing), run `pre-commit run --all-files --show-diff-on-failure`.

### 2. Make Meaningful Commits

Group related changes into logical commits rather than committing all files at once. Each commit should represent one coherent change (e.g., separate the C++ change from the Python binding update from the test addition). This makes `git log` and `git bisect` useful for debugging later.

### 3. Sign Your Commits (DCO Required)

```bash
git commit -s -m "Your message"
```

To fix a prior commit missing the sign-off, use `git commit --amend -s` (or an interactive rebase for older commits). Do **not** use `--no-verify` to bypass the DCO check.

### 4. Use Forks for Pull Requests

Never push branches directly to the main cuOpt repository. Use the fork workflow:

```bash
# 1. Clone the main repo
git clone https://github.com/NVIDIA/cuopt.git
cd cuopt

# 2. Add your fork as a remote
git remote add fork https://github.com/<your-username>/cuopt.git

# 3. Create a branch from the appropriate base
git checkout -b my-feature-branch

# 4. Make changes, commit, then push to your fork
git push fork my-feature-branch

# 5. Create PR from your fork → upstream base branch
```

This applies to both human contributors and AI agents. Agents must never push to the upstream repo directly — provide the push command for the user to review and execute from their fork.

### Pull Requests Created by Agents

When an AI agent creates a pull request, it **must be a draft PR** (`gh pr create --draft`). This gives the developer time to review and iterate on the changes before any reviewers get pinged. The developer marks it as ready for review when satisfied.

### PR Descriptions

Keep summaries short — a paragraph or 3–5 bullets stating *what* and *why*. Skim recent merges on the target branch to calibrate.

Skip how-it-works walkthroughs, file-by-file tables, exhaustive test-plan checklists, prose restatements of the diff, and screenshots of output the reviewer can reproduce locally. Reviewers read the code; long structured summaries signal LLM-generated and erode trust.

For extra context (a design decision, unusual constraint, follow-up), one or two sentences with a link to an issue or doc beats expanding the body.

### Writing scripts and CI workflows

Follow YAGNI strictly here — flags, fallbacks, env-var overrides, and config knobs without a concrete failure mode they prevent should be dropped. This applies to scripts and CI workflows specifically, not the codebase as a whole.

A few non-YAGNI points worth keeping in mind:

- Prefer extending an existing script over adding a new one.
- Validate inputs at the top, before any expensive work.
- One shell command per line over chained `&&`; no comments that restate the next line.
- Keep informational CI jobs (reporting, dashboards, comment posting) out of any required-checks list.

When in doubt, mirror how the surrounding cuOpt code handles the same concern.

## Resolving Merge Conflicts

Don't resolve a conflict by mechanically picking the side that looks like a superset. A small, local conflict (a few changed lines in one function) often sits on top of a larger architectural divergence — one branch refactored a mechanism the other left alone — and the conflict markers only show the tip of it. Picking "the bigger hunk" then strands the rest of that mechanism.

Before choosing a side, reconstruct what each branch actually did:

- Diff the conflicting symbols across **both branches and the merge base**, not just the two conflict hunks: `git show <branch>:<path>` and `git merge-base A B`. Watch for changes to a member's *type*, an ownership/lifetime model, or a synchronization/threading model (e.g. `std::future` → OpenMP task, `std::atomic` → `omp_atomic_t`). Those changes ripple beyond the conflict region.
- Check how the **already-merged, non-conflicted files** use the symbol. If a caller (constructor call, factory, task spawn) was auto-merged to one branch's signature, the conflicted file must conform to that branch — keeping the other branch's member or wait logic leaves it dead.
- When one branch *removed* a mechanism and the other *built on top of it*, the correct resolution is usually to adopt the removal (the newer baseline) and re-port the feature onto the new mechanism — not to keep both, which yields a member that is never set and a guard that never fires.

A wrong merge resolution frequently **compiles cleanly and fails silently**: a dead pointer stays `nullptr`, the guard that depended on it never triggers, and a whole feature quietly disables itself with no error. Compilation is not evidence of a correct merge — trace the runtime wiring (who sets this field? who waits on it? is that path still reachable?) before declaring the conflict resolved.

## Common Tasks

### Adding a Solver Parameter

Internal settings struct fields (e.g. a new MIP cut toggle) are often **already
read by the solver** but not yet exposed to the CLI/string interface, Python, or
the server. To fully expose one, wire it through every layer below — missing any
one silently drops the parameter from that interface. Use a sibling parameter
(e.g. `clique_cuts`) as the template and grep for it across the repo to find
every spot.

1. **Settings struct** — add the field in `cpp/include/cuopt/.../solver_settings.hpp` (and the internal `simplex_solver_settings.hpp` if the solver reads it there). Default to `-1` (automatic) for cut toggles.
2. **C constant** — add a `#define CUOPT_MIP_<NAME> "mip_<name>"` in `cpp/include/cuopt/linear_programming/constants.h`, next to the related parameters.
3. **String-parameter registry** — add a tuple to the `int_parameters` (or `float_parameters`) table in `cpp/src/math_optimization/solver_settings.cu`: `{CUOPT_MIP_<NAME>, &mip_settings.<field>, <min>, <max>, <default>}`. This single registry is what drives CLI parsing, `set_parameter`/`get_parameter`, and Python auto-discovery — no `.pyx` change needed.
4. **gRPC/server path** — add the field under the right message in `cpp/src/grpc/codegen/field_registry.yaml` with the next free `field_num` for that message, then regenerate with `python cpp/src/grpc/codegen/generate_conversions.py` (needs `pyyaml`). The generated `cuopt_remote_data.proto` and `generated_*_to_*.inc` files are **auto-generated — never hand-edit** unless you cannot run the generator, in which case mirror the sibling parameter exactly (proto fields are ordered by `field_num`; the `.inc` files follow registry declaration order).
5. **Docs** — add a `.. doxygendefine:: CUOPT_MIP_<NAME>` to `docs/cuopt/source/cuopt-c/mip/mip-c-api.rst` and a settings section to `docs/cuopt/source/mip-settings.rst`. Add to the server schema (`docs/cuopt/source/cuopt_spec.yaml`) if applicable.
6. **Tests** — add C++ and Python coverage.
7. **Rebuild**: `./build.sh libcuopt && ./build.sh cuopt`

### Adding a Dependency

All dependencies are managed through `dependencies.yaml` — never edit `conda/environments/*.yaml` or `pyproject.toml` files directly. The file uses [RAPIDS dependency-file-generator](https://github.com/rapidsai/dependency-file-generator) format:

1. Find the appropriate group in `dependencies.yaml` (e.g., `build_cpp`, `run_common`, `test_python_common`)
2. Add the package under the correct `output_types` (`conda`, `requirements`, `pyproject`, or a combination)
3. Run `pre-commit run --all-files` — the RAPIDS dependency file generator hook regenerates downstream files automatically
4. Verify: check that `conda/environments/` and relevant `pyproject.toml` files were updated

### Adding a Server Endpoint

1. Add route in `python/cuopt_server/cuopt_server/webserver.py`
2. Update OpenAPI spec `docs/cuopt/source/cuopt_spec.yaml`
3. Add tests in `python/cuopt_server/tests/`
4. Update documentation

### Modifying CUDA Kernels

1. Edit kernel in `cpp/src/`
2. Follow stream-ordering patterns
3. Run C++ tests: `ctest --test-dir cpp/build`
4. Run benchmarks to check performance

## Third-Party Code

**Always ask before including external code.** When copying or adapting external code, you must attribute it properly, verify license compatibility, and flag it in the PR. See the [Third-Party Code section in CONTRIBUTING.md](../../../CONTRIBUTING.md#third-party-code) for the full process.
