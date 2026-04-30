#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run developer-skill agent tests: send prompts to an agent (Claude CLI) with the
cuOpt developer skill as context, then validate that the response follows the
skill (required phrases present, forbidden phrases absent).

Usage (from repo root):
  python ci/utils/run_dev_skill_agent_tests.py              # live: main test set (pass@1)
  python ci/utils/run_dev_skill_agent_tests.py --replay D    # replay: validate saved responses
  python ci/utils/run_dev_skill_agent_tests.py --pass-at 5   # pass@5: run each request 5x, pass if any passes
  python ci/utils/run_dev_skill_agent_tests.py --runtimes-file out/runtimes.json  # write median runtimes to JSON
  python ci/utils/run_dev_skill_agent_tests.py --report out   # write results to out/YYYY-MM-DD_HH-MM-SS/
  python ci/utils/run_dev_skill_agent_tests.py --dataset      # main + issue-style (SWE-bench-like) set

Requires: Claude CLI installed and authenticated (`claude auth login`) for live runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime

# Phrases that indicate a following forbidden term is in a "don't do this" context (negation-aware check).
# Includes "wrong"/"incorrect" so code examples like "// WRONG ... new int[]" don't trigger.
_NEGATION_PATTERN = re.compile(
    r"\b(don'?t|do not|avoid|never|no\s|not\s|prohibit|won'?t|shouldn'?t|must not|cannot|can'?t|"
    r"refuse|refusing|prohibited|disallow|against|wrong|incorrect|❌)\b",
    re.IGNORECASE,
)
# Max chars before a forbidden phrase to look for negation.
_NEGATION_LOOKBACK = 100


def repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))


def load_config(root: str, path: str) -> dict:
    """Load a test config JSON. path can be absolute or relative to root."""
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def config_suite_name(path: str) -> str:
    """Return a short name for save/replay subdir (e.g. dev_skill_agent_tests_issue_style)."""
    return os.path.splitext(os.path.basename(path))[0]


def run_claude(root: str, skill_path: str | None, prompt: str, timeout: int) -> tuple[str, float, dict]:
    """Run Claude CLI with optional skill as system context.

    Returns (response_text, elapsed_seconds, metadata) where metadata
    contains token counts and turn/step information when available.
    When skill_path is None the skill file is omitted (baseline / no-skill mode).
    """
    cmd = [
        "claude",
        "-p",
        "--no-session-persistence",
        "--output-format", "json",
    ]
    if skill_path is not None:
        abs_skill = os.path.join(root, skill_path)
        if not os.path.isfile(abs_skill):
            raise FileNotFoundError(f"Skill file not found: {abs_skill}")
        cmd += ["--append-system-prompt-file", abs_skill]
    cmd.append(prompt)
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    metadata: dict = {}
    try:
        data = json.loads(result.stdout)
        response_text = data.get("result", "")
        usage = data.get("usage", {})
        metadata = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "num_turns": data.get("num_turns", 0),
            "cost_usd": data.get("total_cost_usd", 0.0),
            "duration_ms": data.get("duration_ms", 0),
            "duration_api_ms": data.get("duration_api_ms", 0),
        }
    except (json.JSONDecodeError, TypeError):
        response_text = (result.stdout or "") + (result.stderr or "")
    return (response_text, elapsed, metadata)


def _phrase_in_response(text_lower: str, item: str | list[str]) -> bool:
    """Return True if the required phrase(s) appear. item can be a string or list of alternatives (any one)."""
    if isinstance(item, list):
        return any(p.lower() in text_lower for p in item)
    return item.lower() in text_lower


def _forbidden_phrase_violation(text: str, text_lower: str, phrase: str) -> bool:
    """
    Return True if phrase appears in text in a way that violates the rule (i.e. not only in a negated context).
    E.g. "do not run that" contains "run that" but in a negated context, so no violation.
    """
    phrase_lower = phrase.lower()
    if phrase_lower not in text_lower:
        return False
    start = 0
    while True:
        i = text_lower.find(phrase_lower, start)
        if i == -1:
            break
        window = text_lower[max(0, i - _NEGATION_LOOKBACK) : i]
        if not _NEGATION_PATTERN.search(window):
            return True  # found an occurrence not preceded by negation
        start = i + 1
    return False


def check_response(
    response: str,
    must_include: list[str] | list[str | list[str]],
    must_not_include: list[str],
) -> tuple[bool, list[str]]:
    """Validate response. Return (passed, list of failure reasons).

    - must_include: each entry can be a string (must appear) or a list of strings (any one must appear).
    - must_not_include: phrase must not appear in a non-negated context (e.g. 'don't use X' is allowed).
    """
    failures = []
    lower = response.lower()
    for item in must_include:
        if _phrase_in_response(lower, item):
            continue
        if isinstance(item, list):
            failures.append(f"Response must include one of: {[repr(p) for p in item]}")
        else:
            failures.append(f"Response must include: {item!r}")
    for phrase in must_not_include:
        if _forbidden_phrase_violation(response, lower, phrase):
            failures.append(f"Response must NOT include: {phrase!r}")
    return (len(failures) == 0, failures)


# Default folder for report and runtimes when --report / --runtimes-file are not specified (relative to repo root)
DEFAULT_RESULTS_DIR = "out/dev_skill_agent_tests"


def claude_available() -> bool:
    """Return True if Claude CLI is installed and authenticated."""
    try:
        r = subprocess.run(
            ["claude", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run developer skill agent tests (Claude CLI + validation)")
    parser.add_argument(
        "--replay",
        metavar="DIR",
        help="Replay mode: validate saved responses from DIR (one file per test id, or per suite/test_id). No CLI call.",
    )
    parser.add_argument(
        "--tests-file",
        metavar="PATH",
        action="append",
        dest="tests_files",
        help="Additional test config JSON (same schema). Can be repeated. Default: ci/utils/dev_skill_agent_tests.json only.",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Also run issue-style (SWE-bench-like) tests from ci/utils/dev_skill_agent_tests_issue_style.json.",
    )
    parser.add_argument(
        "--pass-at",
        type=int,
        metavar="K",
        default=1,
        help="pass@K: run each request K times; pass if at least one response passes (default: 1). Only applies to live runs; replay is always pass@1.",
    )
    parser.add_argument(
        "--report",
        metavar="DIR",
        nargs="?",
        const=DEFAULT_RESULTS_DIR,
        default=None,
        help=f"Write results.csv and report.md to DIR/YYYY-MM-DD_HH-MM-SS/. Omit DIR to use {DEFAULT_RESULTS_DIR}. By default report is written; use --no-report to disable.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write report or runtimes to disk.",
    )
    parser.add_argument(
        "--runtimes-file",
        metavar="PATH",
        nargs="?",
        const=None,
        default=None,
        help=f"Write runtimes JSON. Dir or omitted => PATH/YYYY-MM-DD_HH-MM-SS/runtimes.json (.json file => write there). Default: same as report dir ({DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        nargs="?",
        const=DEFAULT_RESULTS_DIR,
        default=None,
        help=f"Save each response for replay. DIR defaults to {DEFAULT_RESULTS_DIR}. Writes to DIR/YYYY-MM-DD_HH-MM-SS/.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full response on failure")
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Baseline mode: run prompts without injecting the skill file (omits --append-system-prompt-file).",
    )
    args = parser.parse_args()

    # Apply default results dir when report/runtimes not explicitly set and not disabled
    if not getattr(args, "no_report", False):
        args.report = args.report if args.report is not None else DEFAULT_RESULTS_DIR
        args.runtimes_file = args.runtimes_file if args.runtimes_file is not None else DEFAULT_RESULTS_DIR
    else:
        args.report = None
        args.runtimes_file = None

    pass_at = max(1, args.pass_at)

    # Date-time folder for this run when writing report/save/runtimes
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_results_dir: str | None = None  # actual path used for --report/--save (with timestamp)

    root = repo_root()
    if not args.replay and not claude_available():
        print("Claude CLI is not available or not authenticated. Run: claude auth login", file=sys.stderr)
        print("Use --replay DIR to validate saved responses without the CLI.", file=sys.stderr)
        return 2
    os.chdir(root)

    # Build list of (suite_name, config) to run
    default_path = "ci/utils/dev_skill_agent_tests.json"
    configs_to_run: list[tuple[str, dict]] = []
    if args.tests_files:
        for p in args.tests_files:
            cfg = load_config(root, p)
            configs_to_run.append((config_suite_name(p), cfg))
    else:
        configs_to_run.append((config_suite_name(default_path), load_config(root, default_path)))
    if args.dataset:
        issue_path = "ci/utils/dev_skill_agent_tests_issue_style.json"
        if not any(s == config_suite_name(issue_path) for s, _ in configs_to_run):
            configs_to_run.append((config_suite_name(issue_path), load_config(root, issue_path)))

    # Resolve timestamped output dirs for report, save, runtimes (YYYY-MM-DD_HH-MM-SS)
    save_dir_actual: str | None = None
    runtimes_path_actual: str | None = None
    if args.report:
        report_base = os.path.join(root, args.report) if not os.path.isabs(args.report) else args.report
        run_results_dir = os.path.join(report_base, run_ts)
        os.makedirs(run_results_dir, exist_ok=True)
        if args.report == DEFAULT_RESULTS_DIR:
            print(f"Run results (report) will be written to: {run_results_dir} (default: {DEFAULT_RESULTS_DIR})")
        else:
            print(f"Run results (report) will be written to: {run_results_dir}")
    if args.save:
        save_base = os.path.join(root, args.save) if not os.path.isabs(args.save) else args.save
        save_dir_actual = os.path.join(save_base, run_ts)
        if run_results_dir is None:
            run_results_dir = save_dir_actual
        os.makedirs(save_dir_actual, exist_ok=True)
        if save_dir_actual != run_results_dir:
            print(f"Saved responses will be written to: {save_dir_actual}")
    if args.runtimes_file:
        rft_base = os.path.join(root, args.runtimes_file) if not os.path.isabs(args.runtimes_file) else args.runtimes_file
        if not args.runtimes_file.strip().endswith(".json"):
            runtimes_dir_actual = os.path.join(rft_base, run_ts)
            os.makedirs(runtimes_dir_actual, exist_ok=True)
            runtimes_path_actual = os.path.join(runtimes_dir_actual, "runtimes.json")
            if run_results_dir is None:
                run_results_dir = runtimes_dir_actual
            print(f"Runtimes will be written to: {runtimes_path_actual}")
        else:
            runtimes_path_actual = rft_base
            os.makedirs(os.path.dirname(runtimes_path_actual) or ".", exist_ok=True)

    passed = 0
    failed = 0
    skipped = 0
    runtimes_report: dict[str, dict] = {}  # label -> { runtimes: [], median: float, passed: bool, tokens/turns }
    report_rows: list[dict] = []  # for --report: { label, test_id, prompt, passed, median_seconds, tokens, turns, ... }

    no_skill = getattr(args, "no_skill", False)

    for suite_name, config in configs_to_run:
        skill_file = None if no_skill else config["skill_file"]
        timeout = config.get("timeout_seconds", 120)
        tests = config["tests"]
        default_inc = config.get("default_assertions", {}).get("must_include", [])
        default_not = config.get("default_assertions", {}).get("must_not_include", [])

        for test in tests:
            test_id = test["id"]
            prompt = test["prompt"]
            must_include = test.get("must_include", default_inc)
            must_not_include = test.get("must_not_include", default_not)
            # Replay/save path: if single suite, DIR/<id>.txt; else DIR/<suite>/<id>.txt
            if len(configs_to_run) == 1:
                replay_file = f"{test_id}.txt"
            else:
                replay_file = os.path.join(suite_name, f"{test_id}.txt")

            label = f"{suite_name}/{test_id}" if len(configs_to_run) > 1 else test_id
            runtimes: list[float] = []
            responses: list[str] = []
            metadata_list: list[dict] = []
            test_passed = False

            if args.replay:
                replay_path = os.path.join(args.replay, replay_file)
                if not os.path.isfile(replay_path):
                    print(f"SKIP {label}: no replay file {replay_path}")
                    skipped += 1
                    if args.report:
                        report_rows.append({"label": label, "test_id": test_id, "prompt": prompt, "passed": "SKIP", "median_seconds": "", "median_input_tokens": "", "median_output_tokens": "", "median_num_turns": "", "failure_reasons": [], "response_preview": ""})
                    continue
                t0 = time.perf_counter()
                with open(replay_path, encoding="utf-8") as f:
                    response = f.read()
                runtimes.append(time.perf_counter() - t0)
                responses.append(response)
            else:
                # Live: run pass_at times (pass@1 or pass@5, etc.)
                for attempt in range(pass_at):
                    try:
                        response, elapsed, meta = run_claude(root, skill_file, prompt, timeout)
                        runtimes.append(elapsed)
                        responses.append(response)
                        metadata_list.append(meta)
                    except Exception as e:
                        print(f"FAIL {label} (attempt {attempt + 1}/{pass_at}): {e}")
                        runtimes.clear()
                        responses.clear()
                        metadata_list.clear()
                        failed += 1
                        if args.report:
                            report_rows.append({"label": label, "test_id": test_id, "prompt": prompt, "passed": "FAIL", "median_seconds": "", "median_input_tokens": "", "median_output_tokens": "", "median_num_turns": "", "failure_reasons": [str(e)], "response_preview": ""})
                        break
                else:
                    if args.save and responses and save_dir_actual:
                        save_path = os.path.join(save_dir_actual, replay_file)
                        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(responses[0])

            if not responses:
                continue

            for response in responses:
                ok, failures_list = check_response(response, must_include, must_not_include)
                if ok:
                    test_passed = True
                    break

            failures_list: list[str] = []
            if not test_passed and responses:
                _, failures_list = check_response(responses[0], must_include, must_not_include)

            # Aggregate token/step metadata across attempts
            def _median_meta(key: str) -> int:
                vals = [m.get(key, 0) for m in metadata_list if m]
                return round(statistics.median(vals)) if vals else 0

            med_in_tok = _median_meta("input_tokens")
            med_out_tok = _median_meta("output_tokens")
            med_turns = _median_meta("num_turns")

            if test_passed:
                print(f"PASS {label}" + (f" (pass@{pass_at})" if pass_at > 1 else ""))
                passed += 1
                if args.report and runtimes:
                    report_rows.append({"label": label, "test_id": test_id, "prompt": prompt, "passed": "PASS", "median_seconds": round(statistics.median(runtimes), 3), "median_input_tokens": med_in_tok, "median_output_tokens": med_out_tok, "median_num_turns": med_turns, "failure_reasons": [], "response_preview": ""})
            else:
                print(f"FAIL {label}" + (f" (0/{pass_at} passed)" if pass_at > 1 else ""))
                if pass_at == 1 and responses:
                    for f in failures_list:
                        print(f"  - {f}")
                    if args.verbose and responses:
                        print("  Response (first 1500 chars):")
                        print(responses[0][:1500])
                failed += 1
                if args.report:
                    preview = (responses[0][:800] + "..." if len(responses[0]) > 800 else responses[0]) if responses else ""
                    report_rows.append({"label": label, "test_id": test_id, "prompt": prompt, "passed": "FAIL", "median_seconds": round(statistics.median(runtimes), 3) if runtimes else "", "median_input_tokens": med_in_tok, "median_output_tokens": med_out_tok, "median_num_turns": med_turns, "failure_reasons": failures_list, "response_preview": preview})

            if runtimes:
                median_sec = statistics.median(runtimes)
                runtimes_report[label] = {
                    "runtimes_seconds": runtimes,
                    "median_seconds": round(median_sec, 3),
                    "passed": test_passed,
                    "input_tokens": [m.get("input_tokens", 0) for m in metadata_list],
                    "output_tokens": [m.get("output_tokens", 0) for m in metadata_list],
                    "num_turns": [m.get("num_turns", 0) for m in metadata_list],
                    "median_input_tokens": med_in_tok,
                    "median_output_tokens": med_out_tok,
                    "median_num_turns": med_turns,
                    "cost_usd": [m.get("cost_usd", 0.0) for m in metadata_list],
                }
                if pass_at > 1:
                    print(f"  median runtime: {median_sec:.2f}s  tokens(in/out): {med_in_tok}/{med_out_tok}  turns: {med_turns}")

    # Summary
    total_run = passed + failed
    exit_code = 0 if failed == 0 else 1
    print(f"\nResult: {passed} passed, {failed} failed" + (f", {skipped} skipped" if skipped else ""))
    if pass_at > 1 and total_run:
        print(f"pass@{pass_at}: {passed}/{total_run} tests passed")

    # Median runtime table
    if runtimes_report:
        print("\nMedian runtime per request (seconds), tokens, turns:")
        for label, data in sorted(runtimes_report.items()):
            status = "PASS" if data["passed"] else "FAIL"
            tok_str = f"  tokens(in/out): {data.get('median_input_tokens', 0)}/{data.get('median_output_tokens', 0)}" if data.get("median_input_tokens") else ""
            turn_str = f"  turns: {data.get('median_num_turns', 0)}" if data.get("median_num_turns") else ""
            print(f"  {label}: {data['median_seconds']:.2f}s{tok_str}{turn_str}  [{status}]")

    # Overall median runtime (median of per-test median runtimes)
    median_runtime_overall: float | None = None
    median_input_tokens_overall: int | None = None
    median_output_tokens_overall: int | None = None
    median_num_turns_overall: int | None = None
    total_cost_usd: float = 0.0
    if runtimes_report:
        per_test_medians = [d["median_seconds"] for d in runtimes_report.values()]
        median_runtime_overall = round(statistics.median(per_test_medians), 3)
        in_toks = [d["median_input_tokens"] for d in runtimes_report.values() if d.get("median_input_tokens")]
        out_toks = [d["median_output_tokens"] for d in runtimes_report.values() if d.get("median_output_tokens")]
        turns = [d["median_num_turns"] for d in runtimes_report.values() if d.get("median_num_turns")]
        if in_toks:
            median_input_tokens_overall = round(statistics.median(in_toks))
        if out_toks:
            median_output_tokens_overall = round(statistics.median(out_toks))
        if turns:
            median_num_turns_overall = round(statistics.median(turns))
        for d in runtimes_report.values():
            total_cost_usd += sum(d.get("cost_usd", []))

    # Shareable status block at end of run
    status_summary = {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total_run": total_run,
        "total_tests": passed + failed + skipped,
        "pass_at": pass_at,
        "replay": args.replay is not None,
        "no_skill": no_skill,
        "exit_code": exit_code,
        "median_runtime_seconds": median_runtime_overall,
        "median_input_tokens": median_input_tokens_overall,
        "median_output_tokens": median_output_tokens_overall,
        "median_num_turns": median_num_turns_overall,
        "total_cost_usd": round(total_cost_usd, 4),
    }
    _rt = f"{median_runtime_overall:.2f}s" if median_runtime_overall is not None else "n/a"
    _it = str(median_input_tokens_overall) if median_input_tokens_overall is not None else "n/a"
    _ot = str(median_output_tokens_overall) if median_output_tokens_overall is not None else "n/a"
    _nt = str(median_num_turns_overall) if median_num_turns_overall is not None else "n/a"
    _cost = f"${total_cost_usd:.4f}" if total_cost_usd > 0 else "n/a"

    print("\n" + "=" * 60)
    print("STATUS (shareable)")
    print("=" * 60)
    print(f"  passed:               {passed}")
    print(f"  failed:               {failed}")
    print(f"  skipped:              {skipped}")
    print(f"  total run:            {total_run}")
    print(f"  pass@{pass_at}:               {passed}/{total_run}")
    print(f"  median_runtime:       {_rt}")
    print(f"  median_input_tokens:  {_it}")
    print(f"  median_output_tokens: {_ot}")
    print(f"  median_num_turns:     {_nt}")
    print(f"  total_cost_usd:       {_cost}")
    print(f"  no_skill:             {no_skill}")
    print(f"  replay:               {status_summary['replay']}")
    print(f"  exit_code:            {exit_code}")
    print("=" * 60)

    if args.runtimes_file and runtimes_report:
        out_path = runtimes_path_actual if runtimes_path_actual else (os.path.join(root, args.runtimes_file) if not os.path.isabs(args.runtimes_file) else args.runtimes_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = {
            "pass_at": pass_at,
            "replay": args.replay is not None,
            "status": status_summary,
            "tests": runtimes_report,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nRuntimes and status written to {out_path}")

    # CSV + Markdown report (in timestamped folder)
    if args.report and report_rows and run_results_dir:
        report_dir = run_results_dir
        csv_path = os.path.join(report_dir, "results.csv")
        md_path = os.path.join(report_dir, "report.md")

        # CSV: test_id, label, prompt, passed, median_seconds, input_tokens, output_tokens, num_turns, failure_reasons
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test_id", "label", "prompt", "passed", "median_seconds", "median_input_tokens", "median_output_tokens", "median_num_turns", "failure_reasons"])
            for r in report_rows:
                reasons = " | ".join(r["failure_reasons"]) if r["failure_reasons"] else ""
                med = r["median_seconds"] if r["median_seconds"] != "" else ""
                in_tok = r.get("median_input_tokens", "")
                out_tok = r.get("median_output_tokens", "")
                turns = r.get("median_num_turns", "")
                w.writerow([r["test_id"], r["label"], r["prompt"], r["passed"], med, in_tok, out_tok, turns, reasons])

        # Markdown report
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Dev skill agent test report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}  \n")
            f.write(f"**pass@:** {pass_at}  **replay:** {status_summary['replay']}  **no_skill:** {no_skill}  \n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Passed:** {passed}  \n")
            f.write(f"- **Failed:** {failed}  \n")
            f.write(f"- **Skipped:** {skipped}  \n")
            f.write(f"- **Exit code:** {exit_code}  \n")
            if median_runtime_overall is not None:
                f.write(f"- **Median runtime (overall):** {median_runtime_overall:.2f}s  \n")
            if median_input_tokens_overall is not None:
                f.write(f"- **Median input tokens (overall):** {median_input_tokens_overall}  \n")
            if median_output_tokens_overall is not None:
                f.write(f"- **Median output tokens (overall):** {median_output_tokens_overall}  \n")
            if median_num_turns_overall is not None:
                f.write(f"- **Median turns/steps (overall):** {median_num_turns_overall}  \n")
            if total_cost_usd > 0:
                f.write(f"- **Total cost:** ${total_cost_usd:.4f}  \n")
            f.write("\n## Results\n\n")
            f.write("| test_id | prompt | median_seconds | input_tokens | output_tokens | turns | pass/fail |\n")
            f.write("|---------|--------|----------------|-------------|--------------|-------|----------|\n")
            for r in report_rows:
                prompt_short = (r["prompt"][:60] + "…") if len(r["prompt"]) > 60 else r["prompt"]
                prompt_short = prompt_short.replace("|", "\\|").replace("\n", " ")
                med = r["median_seconds"] if r["median_seconds"] != "" else "—"
                in_tok = r.get("median_input_tokens", "—") or "—"
                out_tok = r.get("median_output_tokens", "—") or "—"
                turns = r.get("median_num_turns", "—") or "—"
                f.write(f"| {r['test_id']} | {prompt_short} | {med} | {in_tok} | {out_tok} | {turns} | {r['passed']} |\n")
            failed_rows = [r for r in report_rows if r["passed"] == "FAIL"]
            if failed_rows:
                f.write("\n## Failed tests (details)\n\n")
                for r in failed_rows:
                    f.write(f"### {r['label']}\n\n")
                    f.write("**Prompt:**  \n")
                    f.write(f"> {r['prompt']}\n\n")
                    f.write("**Failure reasons:**  \n")
                    for reason in r["failure_reasons"]:
                        f.write(f"- {reason}\n")
                    f.write("\n")
                    if r.get("response_preview"):
                        f.write("**Response preview:**  \n\n")
                        f.write("```\n")
                        f.write(r["response_preview"].replace("```", "` ` `"))
                        f.write("\n```\n\n")
            f.write("---\n")
            f.write(f"*Report written to {report_dir}*\n")

        print(f"\nReport written to {report_dir}: results.csv, report.md")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
