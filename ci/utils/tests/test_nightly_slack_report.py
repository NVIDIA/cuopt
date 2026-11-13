# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the nightly Slack report helpers.

Run with: python -m pytest ci/utils/tests
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aggregate_nightly as agg  # noqa: E402
import generate_slack_payloads as slack  # noqa: E402
import nightly_report as report  # noqa: E402

BUILD = "/tmp/conda-bld-output/bld/rattler-build_x/work/cpp/tests/"


def test_clean_message_drops_build_path_and_keeps_location():
    msg = (
        f"{BUILD}mip/cb_test.cu:123\n"
        "Expected equality of these values:\n  a\n    Which is: 1\n  b"
    )
    out = slack.clean_message(msg)
    assert "rattler-build" not in out
    assert out.endswith("@ cb_test.cu:123")
    assert out.startswith("Expected equality of these values:")


def test_clean_message_pytest_uses_first_e_line():
    msg = "Traceback...\nE   AssertionError: assert 1 == 2\nE  + where 1 = f()"
    assert slack.clean_message(msg) == "AssertionError: assert 1 == 2"


def test_clean_message_empty():
    assert slack.clean_message("") == ""


def _entry(name, msg, matrix="m"):
    return {"name": name, "message": msg, "matrix_label": matrix}


def test_same_reason_failures_collapse_into_one_line():
    entries = [
        _entry(f"t{i}", f"{BUILD}a.cu:{i}\nExpected: {i} vs 9")
        for i in range(5)
    ]
    lines, hidden = slack.issue_lines(entries, ":repeat:")
    assert hidden == 0
    assert len(lines) == 1
    assert "*5 tests*" in lines[0]
    assert "+2 more" in lines[0]


def test_below_group_threshold_stays_individual():
    entries = [
        _entry(f"t{i}", f"{BUILD}a.cu:{i}\nExpected: {i} vs 9")
        for i in range(2)
    ]
    lines, _ = slack.issue_lines(entries, ":repeat:")
    assert len(lines) == 2


def test_duplicate_name_and_matrix_is_listed_once():
    entries = [_entry("t", "x"), _entry("t", "x")]
    lines, _ = slack.issue_lines(entries, ":x:")
    assert len(lines) == 1


def test_empty_reason_never_groups():
    entries = [_entry(f"t{i}", "") for i in range(5)]
    lines, _ = slack.issue_lines(entries, ":x:")
    assert len(lines) == 5


def test_lines_over_the_limit_are_counted():
    entries = [_entry(f"t{i}", "") for i in range(15)]
    lines, hidden = slack.issue_lines(entries, ":x:", limit=10)
    assert len(lines) == 10
    assert hidden == 5


def test_failure_night_window_prunes_old_dates():
    rec = {}
    assert report._record_failure_night(rec, "2026-11-01") == 1
    assert report._record_failure_night(rec, "2026-11-05") == 2
    assert report._record_failure_night(rec, "2026-11-05") == 2  # same night
    assert (
        report._record_failure_night(rec, "2026-11-30") == 1
    )  # others aged out


def test_last_green_sha_picks_latest_green_night(tmp_path, monkeypatch):
    index = {
        "dates": {
            "2026-11-10/main": {
                "date": "2026-11-10",
                "branch": "main",
                "sha": "old",
                "job_summary": {"failed": 0},
            },
            "2026-11-11/main": {
                "date": "2026-11-11",
                "branch": "main",
                "sha": "bad",
                "job_summary": {"failed": 2},
            },
            "2026-11-12/main": {
                "date": "2026-11-12",
                "branch": "main",
                "sha": "good",
                "job_summary": {"failed": 0},
            },
            "2026-11-12/other": {
                "date": "2026-11-12",
                "branch": "other",
                "sha": "x",
                "job_summary": {"failed": 0},
            },
            "2026-11-13/main": {
                "date": "2026-11-13",
                "branch": "main",
                "sha": "today",
                "job_summary": {"failed": 0},
            },
            "2026-11-09/main": {
                "date": "2026-11-09",
                "branch": "main",
                "job_summary": {"failed": 0},
            },
        }
    }

    def fake_download(uri, local):
        Path(local).write_text(json.dumps(index))
        return True

    monkeypatch.setattr(agg, "s3_download", fake_download)
    assert (
        agg.find_last_green_sha(
            "s3://x/index.json", "2026-11-13", "main", tmp_path
        )
        == "good"
    )


def test_last_green_sha_empty_without_index(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "s3_download", lambda uri, local: False)
    assert (
        agg.find_last_green_sha("s3://x", "2026-11-13", "main", tmp_path) == ""
    )
    assert agg.find_last_green_sha("", "2026-11-13", "main", tmp_path) == ""


# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------

import generate_build_payload as build  # noqa: E402


def _job(name, conclusion, url=""):
    return {"name": name, "conclusion": conclusion, "html_url": url}


def _texts(payload):
    out = []
    for b in payload["blocks"]:
        if "text" in b:
            out.append(b["text"]["text"])
        for el in b.get("elements", []):
            out.append(el["text"])
    return "\n".join(out)


def test_green_build_is_one_line_and_counts_skipped():
    jobs = {
        "jobs": [_job(f"wheel-build / {i}", "success") for i in range(4)]
        + [_job("release-github", "skipped")]
        + [_job("build-summary", "success"), _job("compute-matrix", "success")]
    }
    text = _texts(build.build_payload(jobs, "main", "d", "u"))
    assert "All 4 build jobs passed (1 skipped)" in text
    assert "wheel-build" not in text  # passing groups are not listed


def test_failed_build_lists_only_the_failed_group_with_log_link():
    jobs = {
        "jobs": [
            _job("cpp-build / a", "success"),
            _job("wheel-build / b", "failure", "http://logs/b"),
            _job("wheel-build / c", "success"),
        ]
    }
    text = _texts(build.build_payload(jobs, "main", "d", "u"))
    assert "1 build job(s) failed, 2 passed" in text
    assert "*wheel-build* — 1/2 failed" in text
    assert "http://logs/b" in text
    assert "cpp-build" not in text


def test_cancelled_build_is_not_reported_as_passed():
    jobs = {"jobs": [_job("a / x", "success"), _job("b / y", "cancelled")]}
    text = _texts(build.build_payload(jobs, "main", "d", "u"))
    assert "Build incomplete" in text
    assert "All" not in text


def test_commit_link_uses_repo_url():
    payload = build.build_payload(
        {"jobs": []}, "main", "d", "u", "a" * 40, "https://github.com/o/r"
    )
    assert "https://github.com/o/r/commit/" + "a" * 40 in _texts(payload)
