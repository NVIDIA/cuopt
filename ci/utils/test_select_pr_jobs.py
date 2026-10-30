# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for select_pr_jobs.py and for its fit with pr.yaml.

Run from the repository root: python3 -m unittest ci/utils/test_select_pr_jobs.py
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import select_pr_jobs as sel  # noqa: E402

_PR_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    ".github",
    "workflows",
    "pr.yaml",
)

_ALL = set(sel._NEEDS)
_CPP_TESTS = {"conda_cpp_build", "conda_cpp_tests", "multi_gpu_cpp_tests"}
_PYTHON = {
    "conda_cpp_build",
    "conda_python_build",
    "conda_python_tests",
    "docs_build",
    "wheel_build",
    "wheel_tests",
    "self_hosted",
}
_DOCS = {
    "conda_cpp_build",
    "conda_python_build",
    "docs_build",
    "wheel_build",
    "wheel_tests",
}

# Changed files -> the jobs that must run.
_CASES = {
    "cpp test only": (
        ["cpp/tests/linear_programming/pdlp_test.cu"],
        _CPP_TESTS,
    ),
    "cpp source": (["cpp/src/mip_heuristics/solve.cu"], _ALL),
    "cpp cmake": (["cpp/CMakeLists.txt"], _ALL),
    "python source": (
        ["python/cuopt/cuopt/linear_programming/problem.py"],
        _PYTHON,
    ),
    "python readme is wheel metadata": (
        ["python/libcuopt/README.md"],
        _PYTHON,
    ),
    "docs run in the wheel tests": (["docs/cuopt/source/advanced.rst"], _DOCS),
    "java": (
        ["java/cuopt/pom.xml"],
        {"java_build", "java_static", "conda_cpp_build"},
    ),
    "skills markdown": (["skills/cuopt-developer/SKILL.md"], set()),
    "skills asset": (
        ["skills/cuopt-numerical-optimization-api/assets/lp/model.py"],
        {"conda_cpp_build", "conda_python_build", "conda_python_tests"},
    ),
    "markdown only": (["README.md", "cpp/README.md"], set()),
    "nightly workflow": ([".github/workflows/build.yaml"], set()),
    "pr workflow": ([".github/workflows/pr.yaml"], _ALL),
    "multi-gpu workflow": (
        [".github/workflows/multi_gpu_cpp_test.yaml"],
        {"conda_cpp_build", "multi_gpu_cpp_tests"},
    ),
    "summary script": (["ci/utils/pr_test_summary.py"], set()),
    "unclassified ci helper": (["ci/utils/crash_helpers.sh"], _ALL),
    "unclassified top-level path": (["newdir/file.py"], _ALL),
    "libcuopt recipe": (
        ["conda/recipes/libcuopt/recipe.yaml"],
        {
            "conda_cpp_build",
            "conda_cpp_tests",
            "multi_gpu_cpp_tests",
            "conda_python_build",
            "conda_python_tests",
            "docs_build",
            "java_build",
        },
    ),
    "python recipe": (
        ["conda/recipes/cuopt/recipe.yaml"],
        {
            "conda_cpp_build",
            "conda_python_build",
            "conda_python_tests",
            "docs_build",
        },
    ),
    "test data": (
        ["datasets/linear_programming/download_pdlp_test_dataset.sh"],
        {
            "conda_cpp_build",
            "conda_cpp_tests",
            "multi_gpu_cpp_tests",
            "conda_python_build",
            "conda_python_tests",
            "wheel_build",
            "wheel_tests",
            "self_hosted",
        },
    ),
    "wheel script": (
        ["ci/build_wheel_cuopt.sh"],
        {"wheel_build", "wheel_tests", "self_hosted"},
    ),
    "mixed": (
        ["cpp/tests/qp/two_variable_test.cu", "docs/cuopt/source/index.rst"],
        _CPP_TESTS | _DOCS,
    ),
    "no files": ([], _ALL),
}


def _selected(paths):
    jobs, _ = sel.select_jobs(paths)
    return {job for job, run in jobs.items() if run}


def _workflow_jobs():
    """Return {job id: block text} for the jobs in pr.yaml."""
    with open(_PR_YAML) as f:
        text = f.read()
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\s*$", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


class SelectJobsTest(unittest.TestCase):
    def test_cases(self):
        for name, (paths, expected) in _CASES.items():
            with self.subTest(name):
                self.assertEqual(_selected(paths), expected)

    def test_unmatched_paths_are_reported(self):
        _, unmatched = sel.select_jobs(["cpp/src/a.cu", "newdir/file.py"])
        self.assertEqual(unmatched, ["newdir/file.py"])

    def test_rules_name_known_jobs(self):
        for _, jobs in sel._RULES:
            self.assertLessEqual(set(jobs), _ALL)

    def test_needs_name_known_jobs(self):
        for needs in sel._NEEDS.values():
            self.assertLessEqual(set(needs), _ALL)


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.jobs = _workflow_jobs()
        self.flags = {}
        for name, block in self.jobs.items():
            m = re.search(
                r"^    if: fromJSON\(.*\.outputs\.jobs\)\.(\w+)", block, re.M
            )
            if m:
                self.flags[name] = m.group(1)

    def test_every_flag_is_a_known_job(self):
        self.assertLessEqual(set(self.flags.values()), _ALL)

    def test_every_job_is_used(self):
        self.assertEqual(set(self.flags.values()), _ALL)

    def test_needs_match_the_workflow(self):
        # A job that needs another gated job must pull its flag in.
        for name, flag in self.flags.items():
            m = re.search(
                r"^    needs:\s*(\[.*?\]|(?:\n      - .*)+)",
                self.jobs[name],
                re.M,
            )
            needed = re.findall(r"[A-Za-z0-9_-]+", m.group(1)) if m else []
            closure = {flag}
            pending = [flag]
            while pending:
                for n in sel._NEEDS[pending.pop()]:
                    if n not in closure:
                        closure.add(n)
                        pending.append(n)
            for dep in needed:
                if dep in self.flags:
                    self.assertIn(
                        self.flags[dep],
                        closure,
                        f"{name} needs {dep}, but {flag} does not pull in {self.flags[dep]}",
                    )

    def test_files_read_by_jobs_are_not_ignored(self):
        # Scripts and workflows the jobs run must select at least one job.
        for name, block in self.jobs.items():
            if name == "pr-test-summary":
                continue
            refs = re.findall(r'^\s+script: "?(ci/[^"\s]+)', block, re.M)
            refs += re.findall(r"uses: \./(\.github/workflows/\S+)", block)
            for path in refs:
                with self.subTest(f"{name}: {path}"):
                    self.assertTrue(
                        sel.jobs_for_file(path),
                        f"{path} is read by {name} but selects no job",
                    )


if __name__ == "__main__":
    unittest.main()
