# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post or update a sticky PR comment summarizing CI test job failures.

Reads GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_REF, and GH_TOKEN from the
environment.  Finds every test job in the current workflow run, then posts (or
updates) a single comment on the pull request listing any failed jobs with
direct log links and a collapsible list of individual failed test names.

Usage (called from a GitHub Actions step):
    python3 ci/utils/pr_test_summary.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

# Job name prefixes that are considered test jobs.
_TEST_PREFIXES = (
    "conda-cpp-tests",
    "conda-python-tests",
    "wheel-tests-cuopt",
    "wheel-tests-cuopt-server",
    "test-self-hosted-server",
)

_MARKER = "<!-- pr-test-summary -->"
_HTTP_TIMEOUT_SEC = 30
# Maximum failed test names shown per job dropdown.
_MAX_TESTS = 50

# Ordered by specificity; first match wins.
_CRASH_PATTERNS = [
    ("Segmentation fault", "SIGSEGV (segfault)"),
    ("SIGSEGV", "SIGSEGV (segfault)"),
    ("signal 11", "SIGSEGV (signal 11)"),
    ("Aborted (core dumped)", "SIGABRT"),
    ("SIGABRT", "SIGABRT"),
    ("signal 6", "SIGABRT (signal 6)"),
    ("SIGKILL", "SIGKILL"),
    ("signal 9", "SIGKILL (signal 9)"),
    ("Out of memory", "OOM"),
    ("oom-kill", "OOM"),
    ("core dumped", "core dumped"),
]


def _headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _paginate(path, token):
    """Yield all items from a paginated GitHub REST API GET endpoint."""
    url = f"https://api.github.com{path}?per_page=100"
    while url:
        req = urllib.request.Request(url, headers=_headers(token))
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read())
            # Jobs endpoint wraps items in {"jobs": [...]}; comments is a bare list.
            yield from (data["jobs"] if isinstance(data, dict) else data)
            link = resp.headers.get("Link", "")
            url = next(
                (
                    p.split(";")[0].strip().strip("<>")
                    for p in link.split(",")
                    if 'rel="next"' in p
                ),
                None,
            )


def _api(path, token, method, payload):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=json.dumps(payload).encode(),
        method=method,
        headers={**_headers(token), "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _is_test_job(name):
    # Matrix jobs use " / " or " (" as separator depending on GHA version.
    return any(name == p or name.startswith(p + " ") for p in _TEST_PREFIXES)


def _analyze_job_log(job_id, repo, token):
    """Return (failed_test_ids, crash_description_or_None) from a job's log."""
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs",
        headers=_headers(token),
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            # Stream the log, retaining only the last 512 KB so the pytest
            # summary section at the end of the output is always captured.
            chunks = []
            total = 0
            while chunk := resp.read(65536):
                chunks.append(chunk)
                total += len(chunk)
                if total > 512 * 1024:
                    chunks = chunks[-8:]
                    total = sum(len(c) for c in chunks)
    except (urllib.error.HTTPError, urllib.error.URLError):
        return [], None

    text = b"".join(chunks).decode("utf-8", errors="replace")

    crash = next(
        (desc for pattern, desc in _CRASH_PATTERNS if pattern in text), None
    )

    failed = []
    in_summary = False
    for raw in text.splitlines():
        # Strip GHA timestamp prefix: "2024-01-15T10:30:45.1234567Z content"
        parts = raw.split("Z ", 1)
        line = parts[1] if len(parts) > 1 and len(parts[0]) < 35 else raw

        if "short test summary info" in line:
            in_summary = True
        elif in_summary:
            if line.startswith(("FAILED ", "ERROR ")):
                test_id = line.split(" ", 1)[1].split(" - ")[0].strip()
                if test_id:
                    failed.append(test_id)
            elif line.startswith("=") and failed:
                break

    return failed[:_MAX_TESTS], crash


def _build_body(failed, passed, skipped, job_analysis):
    lines = [_MARKER, "## CI Test Summary", ""]
    if not failed:
        lines.append(f"✅ All {len(passed)} test job(s) passed.")
    else:
        lines.append(
            f"**{len(failed)} failed** · {len(passed)} passed · {len(skipped)} skipped"
        )
        lines += ["", "| Job | Logs |", "|-----|------|"]
        for job in failed:
            lines.append(
                f"| ❌ `{job['name']}` | [View logs]({job['html_url']}) |"
            )

        for job in failed:
            tests, crash = job_analysis.get(job["id"], ([], None))
            if not tests and not crash:
                continue
            if crash and not tests:
                summary = f"💥 crashed ({crash})"
                detail = "Process was terminated before pytest completed."
            else:
                n = len(tests)
                noun = "test" if n == 1 else "tests"
                summary = f"{n} failed {noun}" + (
                    f" · 💥 {crash}" if crash else ""
                )
                detail = "\n".join(f"- `{t}`" for t in tests)
            lines += [
                "",
                "<details>",
                f"<summary><code>{job['name']}</code> — {summary}</summary>",
                "",
                detail,
                "",
                "</details>",
            ]

    return "\n".join(lines)


def main():
    token = os.environ["GH_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    ref = os.environ["GITHUB_REF"]  # refs/heads/pull-request/NNN

    branch = ref.removeprefix("refs/heads/")
    if not branch.startswith("pull-request/"):
        print(f"Not a PR branch ({branch}), skipping.", file=sys.stderr)
        return
    pr_number = int(branch.removeprefix("pull-request/"))

    jobs = list(_paginate(f"/repos/{repo}/actions/runs/{run_id}/jobs", token))
    test_jobs = [j for j in jobs if _is_test_job(j["name"])]
    if not test_jobs:
        print("No test jobs found in this run, skipping.", file=sys.stderr)
        return

    passed = [j for j in test_jobs if j["conclusion"] == "success"]
    skipped = [j for j in test_jobs if j["conclusion"] == "skipped"]
    failed = [j for j in test_jobs if j not in passed and j not in skipped]

    job_analysis = {
        job["id"]: _analyze_job_log(job["id"], repo, token) for job in failed
    }

    body = _build_body(failed, passed, skipped, job_analysis)

    comments = list(
        _paginate(f"/repos/{repo}/issues/{pr_number}/comments", token)
    )
    existing = next(
        (c for c in comments if c.get("body", "").startswith(_MARKER)), None
    )

    if existing:
        _api(
            f"/repos/{repo}/issues/comments/{existing['id']}",
            token,
            "PATCH",
            {"body": body},
        )
        print(f"Updated comment {existing['id']} on PR #{pr_number}.")
    else:
        result = _api(
            f"/repos/{repo}/issues/{pr_number}/comments",
            token,
            "POST",
            {"body": body},
        )
        print(f"Posted comment {result['id']} on PR #{pr_number}.")


if __name__ == "__main__":
    main()
