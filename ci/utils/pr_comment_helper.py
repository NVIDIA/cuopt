#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GitHub PR helpers for the PR test-summary workflow.

Two subcommands:

    base-ref     Print the PR's target branch (e.g., ``main``).
    post         Post (or update) a single sticky comment identified by a
                 hidden HTML-comment marker.

Stdlib only (urllib + json) so this runs in slim CI containers without
extra installs.  Both ``ci/pr_summary.sh`` and ``pr_test_summary.yaml``
dispatch into this module rather than embedding inline Python.

The hidden marker is defined here as the single source of truth and
re-used by ``aggregate_pr.py`` when it builds the comment body.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib import error, request

GITHUB_API = "https://api.github.com"

# Imported by aggregate_pr.py so the body it writes and the marker the
# poster searches for stay in sync.
COMMENT_MARKER = "<!-- pr-test-classification -->"


def _gh_request(method, url, token, payload=None, timeout=30):
    """Issue a GitHub API request and return parsed JSON (or ``None``)."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
    except error.HTTPError as exc:
        detail = exc.read().decode()[:300]
        raise RuntimeError(
            f"GitHub API {method} {url} failed: {exc.code} {detail}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"GitHub API {method} {url} failed: {exc}") from exc

    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def resolve_base_ref(repo: str, pr_number: int, token: str) -> str:
    """Return the PR's target branch (e.g. ``main``).

    Args:
        repo: GitHub ``owner/name`` slug.
        pr_number: Pull-request number.
        token: GitHub token with at least ``pull-requests: read``.

    Returns:
        The PR's base ref, or ``"main"`` if the API response lacks one.

    Raises:
        RuntimeError: If the underlying GitHub API call fails.
    """
    data = _gh_request(
        "GET", f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}", token
    )
    return ((data or {}).get("base") or {}).get("ref", "main")


def find_existing_comment_id(
    repo: str, pr_number: int, token: str, marker: str = COMMENT_MARKER
) -> int | None:
    """Find the id of a PR comment whose body starts with ``marker``.

    Paginates through issue comments (100 per page) until a match is
    found or all pages are exhausted.

    Args:
        repo: GitHub ``owner/name`` slug.
        pr_number: Pull-request number.
        token: GitHub token with ``pull-requests: read``.
        marker: Hidden HTML-comment marker that identifies the sticky
            comment (matched after stripping leading whitespace).

    Returns:
        The integer comment id, or ``None`` if no comment matches.

    Raises:
        RuntimeError: If a GitHub API call fails.
    """
    page = 1
    while True:
        url = (
            f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
            f"?per_page=100&page={page}"
        )
        comments = _gh_request("GET", url, token) or []
        for c in comments:
            body = (c.get("body") or "").lstrip()
            if body.startswith(marker):
                return c["id"]
        if len(comments) < 100:
            return None
        page += 1


def post_or_update_comment(
    repo: str,
    pr_number: int,
    token: str,
    body: str,
    marker: str = COMMENT_MARKER,
) -> str:
    """Update the existing sticky PR comment if present; otherwise create one.

    Looks up an existing comment by ``marker``; if found, ``PATCH``es it
    in place; otherwise ``POST``s a new one.

    Args:
        repo: GitHub ``owner/name`` slug.
        pr_number: Pull-request number.
        token: GitHub token with ``pull-requests: write``.
        body: Full Markdown body to post (must already include
            ``marker`` somewhere near the top for future lookups).
        marker: Hidden HTML-comment marker that identifies the sticky
            comment.

    Returns:
        The created/updated comment's ``html_url``, or ``""`` if the
        API response lacked one.

    Raises:
        RuntimeError: If a GitHub API call fails.
    """
    existing_id = find_existing_comment_id(repo, pr_number, token, marker)
    payload = {"body": body}
    if existing_id is not None:
        resp = _gh_request(
            "PATCH",
            f"{GITHUB_API}/repos/{repo}/issues/comments/{existing_id}",
            token,
            payload=payload,
        )
        action = "Updated"
    else:
        resp = _gh_request(
            "POST",
            f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
            token,
            payload=payload,
        )
        action = "Created"
    url = (resp or {}).get("html_url", "")
    print(f"{action} PR comment: {url}")
    return url


def _cmd_base_ref(args: argparse.Namespace, token: str) -> int:
    print(resolve_base_ref(args.repo, args.pr, token))
    return 0


def _cmd_post(args: argparse.Namespace, token: str) -> int:
    with open(args.body_file) as f:
        body = f.read()
    if not body.strip():
        print("Empty body; nothing to post.")
        return 0
    post_or_update_comment(args.repo, args.pr, token, body)
    return 0


def _add_common_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--repo", required=True, help="owner/name")
    sp.add_argument("--pr", required=True, type=int, help="PR number")


def main() -> int:
    """Dispatch to the requested subcommand.

    Reads ``GITHUB_TOKEN`` from the environment (the GitHub convention);
    there is no ``--token`` CLI flag so configuration comes from a
    single source.

    Returns:
        ``0`` on success, ``1`` if a GitHub API call failed, or ``2``
        if ``GITHUB_TOKEN`` is not set in the environment.

    Raises:
        SystemExit: Indirectly via ``argparse`` if argument parsing
            fails.
    """
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_base = sub.add_parser("base-ref", help="Print the PR's target branch.")
    _add_common_args(sp_base)
    sp_base.set_defaults(func=_cmd_base_ref)

    sp_post = sub.add_parser(
        "post", help="Post or update a sticky PR comment."
    )
    _add_common_args(sp_post)
    sp_post.add_argument(
        "--body-file",
        required=True,
        help="File whose contents are the comment body.",
    )
    sp_post.set_defaults(func=_cmd_post)

    args = p.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN env var must be set.", file=sys.stderr)
        return 2
    try:
        return args.func(args, token)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
