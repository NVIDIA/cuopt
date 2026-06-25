# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post or update a sticky PR comment summarizing CI test job failures."""

import io
import json
import os
import re
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
_MAX_TESTS = 50

# gtest prints "[  FAILED  ] Suite.Test (12 ms)" per failing test and again
# without the timing suffix; the dedup set collapses the duplicate.
_GTEST_FAILED = re.compile(r"\[  FAILED  \] (\S+\.\S+?)(?: \(\d+ ms\))?$")


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    # GitHub's job-log endpoint redirects to a presigned S3 URL. Forwarding
    # the Authorization header causes S3 to return 400, so strip it first.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            for key in list(new_req.headers):
                if key.lower() in ("authorization", "x-github-api-version"):
                    del new_req.headers[key]
        return new_req


def _headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _paginate(path, token):
    url = f"https://api.github.com{path}?per_page=100"
    while url:
        req = urllib.request.Request(url, headers=_headers(token))
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read())
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
    return any(name.startswith(p) for p in _TEST_PREFIXES)


def _analyze_job_log(job_id, repo, token):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs",
        headers=_headers(token),
    )
    failed = []
    seen = set()
    in_pytest_summary = False

    def _add(test_id):
        if test_id and test_id not in seen:
            seen.add(test_id)
            failed.append(test_id)

    try:
        opener = urllib.request.build_opener(_DropAuthOnRedirect)
        with opener.open(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            for raw in io.TextIOWrapper(
                resp, encoding="utf-8", errors="replace"
            ):
                # Strip GHA timestamp prefix: "2024-01-15T10:30:45.1234567Z text"
                parts = raw.rstrip("\n").split("Z ", 1)
                line = (
                    parts[1]
                    if len(parts) > 1 and len(parts[0]) < 35
                    else raw.rstrip("\n")
                )

                m = _GTEST_FAILED.match(line)
                if m:
                    _add(m.group(1))
                    continue

                if "short test summary info" in line:
                    in_pytest_summary = True
                elif in_pytest_summary:
                    if line.startswith(("FAILED ", "ERROR ")):
                        _add(line.split(" ", 1)[1].split(" - ")[0].strip())
                    elif line.startswith("="):
                        in_pytest_summary = False
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(
            f"Warning: could not fetch logs for job {job_id}: {exc}",
            file=sys.stderr,
        )

    return failed[:_MAX_TESTS]


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
            tests = job_analysis[job["id"]]
            if not tests:
                continue
            n = len(tests)
            lines += [
                "",
                "<details>",
                f"<summary><code>{job['name']}</code> — {n} failed {'test' if n == 1 else 'tests'}</summary>",
                "",
                "\n".join(f"- `{t}`" for t in tests),
                "",
                "</details>",
            ]

    return "\n".join(lines)


def main():
    token = os.environ["GH_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    ref = os.environ["GITHUB_REF"]

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
