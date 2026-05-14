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
git clone git@github.com:NVIDIA/cuopt.git
cd cuopt

# 2. Add your fork as a remote
git remote add fork git@github.com:<your-username>/cuopt.git

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

Keep PR summaries **short and informative** — typically a few bullets stating *what* changed and *why*. Most merged cuOpt PRs run a single short paragraph or 3–5 bullets; calibrate by skimming a few recent merges on the target branch before writing.

**Don't include:**
- "How it works" walkthroughs of the implementation — reviewers read the code.
- File-by-file tables listing every changed path — the diff already shows this.
- Exhaustive test-plan checklists enumerating every assertion. A short high-level test note is fine; a 10-item checklist is not.
- Restating the diff in prose.
- Embedded screenshots of CI dashboards or generated output unless they show something the reviewer can't reproduce locally.

**Why:** Reviewers skim PR descriptions to get oriented, then read the code. A long, structured summary signals "LLM-generated" and erodes trust in the change. The shorter the summary, the more carefully each line gets read.

If the change genuinely needs more context (a design decision, an unusual constraint, a follow-up plan), state it in one or two sentences and link to an issue or design doc rather than expanding the PR body.

### Editing CI scripts and workflows

CI scripts (`ci/`) and GitHub Actions workflows (`.github/workflows/`) attract LLM-generated cruft more than any other area of the repo because the conventions are unfamiliar and "safe defaults" look helpful. Reviewers push back hard on it. Apply these rules before writing or extending CI:

- **Prefer extending an existing script or workflow over adding a new one.** New files in `ci/` or `.github/workflows/` need a justification that can't be met by extending what's already there. If you're tempted to add a new file, first identify the closest existing one and explain why it doesn't fit.
- **Every flag, option, and env-var override must trace to a real problem.** If you can't point to the failure mode it prevents, drop it. Reviewers will (and do) ask "is this something you added for a real problem, or LLM-generated?" — assume that question on every line.
- **Don't restate defaults.** GitHub Actions already runs steps with `shell: bash -e {0}`; don't add it explicitly. Same for any framework default — restating it implies the writer thought the default was wrong, which confuses readers.
- **Make interfaces strict; no fallback defaults for required inputs.** If an env var, CLI flag, or workflow input is required, fail loudly when it's missing rather than silently defaulting. The risk of "the job has been silently failing for months" outweighs the convenience of a fallback.
- **Hard-code GitHub-specific URLs.** Use `https://github.com/${GITHUB_REPOSITORY}/...` directly. Don't introduce `${GITHUB_SERVER_URL}` overrides unless cuOpt actually runs on GHES.
- **Validate inputs at the top of the script, before any expensive work.** Argument and env-var checks belong before downloads, S3 calls, or aggregation — surface the misconfiguration fast.
- **Split chained bash commands onto their own lines.** `apt-get update && apt-get install -y curl` reads worse than the two-line form and obscures which command failed when one does.
- **No comments that restate the code.** If a comment would tell a reader something the next line already says, delete it. Reserve comments for the non-obvious *why*.
- **Keep PR-scoped CI additions informational and non-blocking.** A new reporting/aggregation job should not be added to `pr-builder`'s `needs:` list — comment posting and dashboards must not gate merging.

When in doubt, look at how the surrounding cuOpt scripts handle the same concern and match that style rather than introducing a new convention.

## Common Tasks

### Adding a Solver Parameter

1. Add to settings struct in `cpp/include/cuopt/` and wire into `set_parameter_from_string()` in `cpp/src/`
2. Expose in Python — if using the string-based interface, the parameter is auto-discovered (no `.pyx` change needed). Add a convenience method in `SolverSettings` if warranted. See [python_bindings.md](python_bindings.md) for the full checklist.
3. Add to server schema (`docs/cuopt/source/cuopt_spec.yaml`) if applicable
4. Add tests at C++ and Python levels
5. Rebuild: `./build.sh libcuopt && ./build.sh cuopt`
6. Update documentation

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
