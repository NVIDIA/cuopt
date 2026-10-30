# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide which PR CI jobs to run from the files a pull request changes.

Each changed file is classified by the first matching entry in _RULES, which
names the jobs that file can affect (none for files no PR job reads). The
prerequisites of those jobs are added afterwards (_NEEDS). A file that
matches no rule runs every job, so a new path costs CI time instead of
silently skipping tests.

Prints {"<job>": true|false, ...} as the `jobs` output of the workflow step.
"""

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# Job -> jobs that must also run for it (they produce its inputs).
_NEEDS = {
    "conda_cpp_build": (),
    "conda_cpp_tests": ("conda_cpp_build",),
    "multi_gpu_cpp_tests": ("conda_cpp_build",),
    "conda_python_build": ("conda_cpp_build",),
    "conda_python_tests": ("conda_python_build",),
    "docs_build": ("conda_python_build",),
    "java_build": ("conda_cpp_build",),
    "java_static": (),
    "wheel_build": (),
    "wheel_tests": ("wheel_build",),
    "self_hosted": ("wheel_build",),
}
_ALL = tuple(_NEEDS)

_CPP_TESTS = ("conda_cpp_tests", "multi_gpu_cpp_tests")
# Everything that consumes the libcuopt conda package.
_CONDA = _CPP_TESTS + (
    "conda_cpp_build",
    "conda_python_build",
    "conda_python_tests",
    "docs_build",
    "java_build",
)
_PYTHON = (
    "conda_python_build",
    "conda_python_tests",
    "docs_build",
    "wheel_build",
    "wheel_tests",
    "self_hosted",
)
_WHEELS = ("wheel_build", "wheel_tests", "self_hosted")
# Test data is read by every job that runs tests.
_DATA = _CPP_TESTS + ("conda_python_tests", "wheel_tests", "self_hosted")

# (globs, jobs): the first rule with a matching glob wins, so put specific
# rules first. `**` crosses directories, `*` does not.
_RULES = (
    ((".github/workflows/pr.yaml",), _ALL),
    ((".github/workflows/multi_gpu_cpp_test.yaml",), ("multi_gpu_cpp_tests",)),
    (
        (".github/workflows/self_hosted_service_test.yaml",),
        ("self_hosted",),
    ),
    # Nothing in a PR runs these: nightly/release workflows, repo and agent
    # metadata, nightly tooling, and the notebooks that only nightly tests.
    (
        (
            ".github/**",
            ".ai/**",
            ".agents/**",
            ".claude/**",
            ".claude-plugin/**",
            ".clinerules/**",
            ".cursor-plugin/**",
            ".opencode/**",
            "agents/**",
            "ci/dashboard/**",
            "ci/docker/**",
            "ci/release/**",
            "ci/split_package_smoke/**",
            "ci/thirdparty-testing/**",
            "ci/vale/**",
            "ci/build_summary.sh",
            "ci/check_style.sh",
            "ci/nightly_summary.sh",
            "ci/run-vale.sh",
            "ci/test_conda_split_packages.sh",
            "ci/test_notebooks.sh",
            "ci/test_wheel_split_packages.sh",
            "ci/utils/aggregate_nightly.py",
            "ci/utils/generate_slack_payloads.py",
            "ci/utils/generate_step_summary.py",
            "ci/utils/pr_test_summary.py",
            "ci/utils/redeploy_nvcf_function.py",
            "ci/utils/send_consolidated_summary.sh",
            "ci/utils/sync_skills_version.sh",
            "ci/utils/update_doc_versions.py",
            "ci/utils/validate_skills.sh",
            "container-builder/**",
            "helmchart/**",
            "img/**",
            "ngc/**",
            "notebooks/**",
            "regression/**",
            "sonarqube/**",
            "ucf/**",
            ".clang-format",
            ".coderabbit.yaml",
            ".gitattributes",
            ".gitignore",
            ".pre-commit-config.yaml",
            ".vale.ini",
            ".yamllint.yaml",
            "codecov.yml",
            "CONVENTIONS.md",
            "gemini-extension.json",
            "JULES.md",
            "LICENSE",
            "sonar-project.properties",
        ),
        (),
    ),
    # The wheel tests run the examples in the docs.
    (("docs/**", "ci/test_doc_examples.sh"), ("docs_build", "wheel_tests")),
    (("ci/build_docs.sh",), ("docs_build",)),
    # Only the skills assets are executed (by the conda Python tests).
    (
        (
            "skills/**/*.md",
            "skills/**/skill.oms.sig",
            "skills/**/evals/**",
            "skills/**/benchmark/**",
            "skills/**/references/**",
        ),
        (),
    ),
    (("skills/**", "ci/test_skills_assets.sh"), ("conda_python_tests",)),
    (("ci/test_python.sh",), ("conda_python_tests",)),
    (
        ("ci/run_cuopt_pytests.sh", "ci/run_cuopt_server_pytests.sh"),
        ("conda_python_tests", "wheel_tests"),
    ),
    (
        ("conda/recipes/cuopt*/**", "ci/build_python.sh"),
        ("conda_python_build", "conda_python_tests", "docs_build"),
    ),
    # python/ holds the wheel packaging too, and its READMEs are wheel
    # metadata, so it is matched before the Markdown rule below.
    (("python/**",), _PYTHON),
    (("**/*.md",), ()),
    (("datasets/**",), _DATA),
    (
        (
            "conda/**",
            "ci/build_cpp.sh",
            "ci/check_symbols.sh",
            "ci/verify_grpc_codegen.sh",
        ),
        _CONDA,
    ),
    (
        (
            "cpp/tests/**",
            "ci/run_ctests.sh",
            "ci/test_cpp.sh",
            "ci/test_cpp_memcheck.sh",
            "ci/test_cpp_multi_gpu.sh",
        ),
        _CPP_TESTS,
    ),
    (("cpp/**",), _ALL),
    (
        (
            "java/**",
            "ci/build_java_static.sh",
            "ci/test_java_static.sh",
        ),
        ("java_static", "java_build"),
    ),
    (("ci/build_java.sh", "ci/test_java.sh"), ("java_build",)),
    (
        ("ci/build_wheel*.sh", "ci/validate_wheel.sh"),
        _WHEELS,
    ),
    (("ci/test_wheel_cuopt*.sh",), ("wheel_tests",)),
    (("ci/test_self_hosted_service.sh",), ("self_hosted",)),
)

_HTTP_TIMEOUT_SEC = 30
_HTTP_ATTEMPTS = 4
_NETWORK_ERRORS = (OSError, http.client.HTTPException)
# GitHub stops listing a PR's files at 3000.
_MAX_FILES = 3000


def _glob_regex(pattern):
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


_COMPILED = tuple(
    (tuple(_glob_regex(p) for p in globs), jobs) for globs, jobs in _RULES
)


def jobs_for_file(path):
    """Return the jobs `path` can affect, or None if no rule matches."""
    for regexes, jobs in _COMPILED:
        if any(r.match(path) for r in regexes):
            return jobs
    return None


def select_jobs(paths):
    """Return {job: bool} for the changed paths, and the unmatched paths."""
    selected = set()
    unmatched = []
    for path in paths:
        jobs = jobs_for_file(path)
        if jobs is None:
            unmatched.append(path)
            selected.update(_ALL)
        else:
            selected.update(jobs)
    if not paths:
        selected.update(_ALL)
    pending = list(selected)
    while pending:
        for needed in _NEEDS[pending.pop()]:
            if needed not in selected:
                selected.add(needed)
                pending.append(needed)
    return {job: job in selected for job in _NEEDS}, unmatched


def _get_json(url, token):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    for attempt in range(1, _HTTP_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                req, timeout=_HTTP_TIMEOUT_SEC
            ) as resp:
                return json.loads(resp.read())
        except _NETWORK_ERRORS as e:
            transient = not isinstance(e, urllib.error.HTTPError) or (
                e.code == 429 or 500 <= e.code < 600
            )
            if not transient or attempt == _HTTP_ATTEMPTS:
                raise
            time.sleep(2**attempt)


def changed_paths(repo, pr_number, token):
    paths = []
    page = 1
    while True:
        files = _get_json(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
            f"?per_page=100&page={page}",
            token,
        )
        for f in files:
            paths.append(f["filename"])
            if f.get("previous_filename"):
                paths.append(f["previous_filename"])
        if len(files) < 100:
            return paths
        if page * 100 >= _MAX_FILES:
            raise RuntimeError(f"PR lists {_MAX_FILES} files or more")
        page += 1


def _write(path, text):
    if path:
        with open(path, "a") as f:
            f.write(text)


def main():
    branch = os.environ.get("GITHUB_REF", "").removeprefix("refs/heads/")
    try:
        if not branch.startswith("pull-request/"):
            raise RuntimeError(f"not a PR branch: {branch!r}")
        paths = changed_paths(
            os.environ["GITHUB_REPOSITORY"],
            int(branch.removeprefix("pull-request/")),
            os.environ["GH_TOKEN"],
        )
        selected, unmatched = select_jobs(paths)
    except Exception as e:
        # Running everything is always safe; skipping tests on a guess is not.
        print(f"::warning::Running all jobs: {e}", file=sys.stderr)
        selected, unmatched, paths = select_jobs([]), [], []

    _write(os.environ.get("GITHUB_OUTPUT"), f"jobs={json.dumps(selected)}\n")
    lines = [f"Changed files: {len(paths)}\n", "| Job | Runs |", "|---|---|"]
    lines += [
        f"| {job} | {'yes' if run else 'no'} |"
        for job, run in selected.items()
    ]
    if unmatched:
        lines += ["", "No rule matches (all jobs run):"]
        lines += [f"- `{p}`" for p in unmatched[:20]]
    _write(os.environ.get("GITHUB_STEP_SUMMARY"), "\n".join(lines) + "\n")
    print(json.dumps(selected))


if __name__ == "__main__":
    main()
