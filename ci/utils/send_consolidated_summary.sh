#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Send a consolidated Slack notification for the entire nightly run.
# Reads the aggregated JSON produced by aggregate_nightly.py and sends:
#   - Main message: Header + status summary + test totals + failed CI jobs
#   - Thread replies: matrix details, failure details, links, HTML report
#
# If SLACK_BOT_TOKEN is available, posts via chat.postMessage (enables
# threading). Falls back to webhook (no threading) otherwise.
#
# Required environment variables:
#   SLACK_WEBHOOK_URL       - Slack incoming webhook URL (fallback)
#   CONSOLIDATED_SUMMARY    - Path to consolidated_summary.json
#
# Optional environment variables:
#   CONSOLIDATED_HTML           - Path to consolidated HTML file to upload
#   SLACK_BOT_TOKEN             - Slack Bot Token (xoxb-*) for threading + file uploads
#   SLACK_CHANNEL_ID            - Slack channel ID (required with bot token)
#   PRESIGNED_REPORT_URL        - Presigned URL for consolidated HTML report
#   PRESIGNED_DASHBOARD_URL     - Presigned URL for dashboard

set -euo pipefail

CONSOLIDATED_SUMMARY="${CONSOLIDATED_SUMMARY:?CONSOLIDATED_SUMMARY must point to consolidated_summary.json}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL is required}"
CONSOLIDATED_HTML="${CONSOLIDATED_HTML:-}"
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
PRESIGNED_REPORT_URL="${PRESIGNED_REPORT_URL:-}"
PRESIGNED_DASHBOARD_URL="${PRESIGNED_DASHBOARD_URL:-}"

if [ ! -f "${CONSOLIDATED_SUMMARY}" ]; then
    echo "ERROR: Summary file not found: ${CONSOLIDATED_SUMMARY}" >&2
    exit 1
fi

# Generate Slack payloads — one JSON object per line.
# Line 1 = main message, lines 2+ = thread replies.
PAYLOADS=$(python3 - "${CONSOLIDATED_SUMMARY}" "${PRESIGNED_REPORT_URL}" "${PRESIGNED_DASHBOARD_URL}" <<'PYEOF'
import json, sys

summary_path = sys.argv[1]
presigned_report_url = sys.argv[2] if len(sys.argv) > 2 else ""
presigned_dashboard_url = sys.argv[3] if len(sys.argv) > 3 else ""

with open(summary_path) as f:
    d = json.load(f)

branch = d.get("branch", "main")
date = d.get("date", "unknown")
github_run_url = d.get("github_run_url", "")
jobs = d.get("job_summary", {})
totals = d.get("test_totals", {})
grid = d.get("matrix_grid", [])
has_new = d.get("has_new_failures", False)
failed_ci_jobs = d.get("failed_ci_jobs", [])
untracked_failed = d.get("untracked_failed_ci_jobs", [])
workflow_jobs = d.get("workflow_jobs", [])

total_jobs = jobs.get("total", 0)
failed_jobs = jobs.get("failed", 0)
flaky_jobs = jobs.get("flaky", 0)
passed_jobs = jobs.get("passed", 0)

total_ci_jobs = len(workflow_jobs)
failed_ci_count = len(failed_ci_jobs)
passed_ci_count = sum(1 for j in workflow_jobs if j["conclusion"] == "success")

status_icons = {
    "passed": ":white_check_mark:",
    "failed-new": ":rotating_light:",
    "failed-recurring": ":x:",
    "flaky": ":warning:",
    "no-results": ":grey_question:",
}

def make_payload(blocks):
    return json.dumps({
        "username": "cuOpt Nightly Bot",
        "icon_emoji": ":robot_face:",
        "blocks": blocks,
    })


# ══════════════════════════════════════════════════════════════════════
# MAIN MESSAGE (line 1) — posted to channel, becomes thread parent
# ══════════════════════════════════════════════════════════════════════
blocks = []

# Identify which workflows have failures (from both CI jobs and matrix grid)
failing_workflows = set()
for j in failed_ci_jobs:
    prefix = j["name"].split(" / ")[0] if " / " in j["name"] else j["name"]
    failing_workflows.add(prefix)
for g in grid:
    if g["status"].startswith("failed"):
        failing_workflows.add(g["test_type"])
flaky_workflows = set()
for g in grid:
    if g["status"] == "flaky":
        flaky_workflows.add(g["test_type"])

has_failures = len(failing_workflows) > 0
untracked_count = len(untracked_failed)

if has_failures and (has_new or untracked_count > 0):
    emoji = ":rotating_light:"
    text = f"{len(failing_workflows)} workflow(s) with NEW failures"
    mention = "<@rgsl888prabhu> "
elif has_failures:
    emoji = ":x:"
    text = f"Recurring failures in {len(failing_workflows)} workflow(s)"
    mention = ""
elif flaky_workflows:
    emoji = ":large_yellow_circle:"
    text = "All jobs passed but flaky tests detected"
    mention = ""
else:
    emoji = ":white_check_mark:"
    text = f"All {total_jobs} matrix jobs passed"
    if total_ci_jobs > 0:
        text += f", all {passed_ci_count} CI jobs succeeded"
    mention = ""

stats_parts = []
if totals.get("failed", 0) > 0:
    stats_parts.append(f":x: {totals['failed']} failed")
if totals.get("flaky", 0) > 0:
    stats_parts.append(f":warning: {totals['flaky']} flaky")
if not stats_parts:
    stats_parts.append(f":white_check_mark: {totals.get('total', 0)} tests passed")
stats = "  |  ".join(stats_parts)

blocks.append({
    "type": "header",
    "text": {
        "type": "plain_text",
        "text": f"cuOpt Nightly Tests \u2014 {branch} \u2014 {date}",
        "emoji": True,
    },
})
blocks.append({
    "type": "section",
    "text": {
        "type": "mrkdwn",
        "text": f"{mention}{emoji} *{text}*\n\n{stats}",
    },
})

# Per-workflow failure summary using CI job counts from GitHub API
# Build a lookup: workflow prefix -> (failed, total) from workflow_jobs
wf_counts = {}
for j in workflow_jobs:
    prefix = j["name"].split(" / ")[0] if " / " in j["name"] else j["name"]
    wf_counts.setdefault(prefix, {"failed": 0, "total": 0})
    wf_counts[prefix]["total"] += 1
    if j["conclusion"] == "failure":
        wf_counts[prefix]["failed"] += 1

# Build a lookup: workflow prefix -> list of failing matrix_labels from grid
wf_failing_labels = {}
for g in grid:
    if g["status"].startswith("failed"):
        wf_failing_labels.setdefault(g["test_type"], []).append(g["matrix_label"])

if failing_workflows:
    lines = []
    for wf in sorted(failing_workflows):
        counts = wf_counts.get(wf, {})
        f_count = counts.get("failed", 0)
        t_count = counts.get("total", 0)
        # Append failing matrix labels (up to 3, then "+N more")
        labels = wf_failing_labels.get(wf, [])
        label_suffix = ""
        if labels:
            shown = labels[:3]
            label_suffix = " (" + ", ".join(shown)
            if len(labels) > 3:
                label_suffix += f", +{len(labels) - 3} more"
            label_suffix += ")"
        if t_count > 0:
            lines.append(f":x:  *{wf}* — {f_count}/{t_count} failed{label_suffix}")
        else:
            lines.append(f":x:  *{wf}* — failed{label_suffix}")
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(lines)},
    })

# Links in main message
link_parts = []
if github_run_url:
    link_parts.append(f"<{github_run_url}|:github: GitHub Actions>")
if presigned_report_url:
    link_parts.append(f"<{presigned_report_url}|:bar_chart: Full Report>")
if presigned_dashboard_url:
    link_parts.append(f"<{presigned_dashboard_url}|:chart_with_upwards_trend: Dashboard>")
if link_parts:
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "  |  ".join(link_parts)}],
    })

print(make_payload(blocks))


# ══════════════════════════════════════════════════════════════════════
# THREAD REPLIES (lines 2+) — posted as replies to main message
# ══════════════════════════════════════════════════════════════════════

# ── Thread 1: Failing and flaky tests (grouped by workflow) ───────────
# Build per-workflow test issue lists
new_failures = d.get("new_failures", [])
recurring = d.get("recurring_failures", [])
flaky = d.get("flaky_tests", [])
resolved = d.get("resolved_tests", [])

# Collect all test issues by test_type (workflow)
issues_by_wf = {}
for f_entry in new_failures:
    tt = f_entry.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["new"].append(f_entry)
for f_entry in recurring:
    tt = f_entry.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["recurring"].append(f_entry)
for f_entry in flaky:
    tt = f_entry.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["flaky"].append(f_entry)
for r in resolved:
    tt = r.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["resolved"].append(r)

if issues_by_wf:
    for wf_name, issues in sorted(issues_by_wf.items()):
        wf_blocks = []
        wf_text = f"*{wf_name}*\n"

        # New failures first (most urgent, show more error context)
        for f_entry in issues["new"][:10]:
            msg = f_entry.get("message", "")[:150].replace("\n", " ")
            matrix = f_entry.get("matrix_label", "")
            wf_text += f":new:  `{f_entry['name']}` ({matrix}) — {msg}\n"

        # Flaky (actionable — tests that are unstable)
        for f_entry in issues["flaky"][:10]:
            matrix = f_entry.get("matrix_label", "")
            wf_text += f":warning:  `{f_entry['name']}` ({matrix})\n"

        # Recurring failures (known issues)
        for f_entry in issues["recurring"][:10]:
            matrix = f_entry.get("matrix_label", "")
            first = f_entry.get("first_seen", "?")
            wf_text += f":repeat:  `{f_entry['name']}` ({matrix}) — since {first}\n"

        # Resolved
        for r in issues["resolved"][:5]:
            matrix = r.get("matrix_label", "")
            count = r.get("failure_count", "?")
            wf_text += f":white_check_mark:  `{r['name']}` ({matrix}) — was failing {count}x\n"

        # Truncation notes
        for category, label, limit in [("new", "new failures", 10), ("recurring", "recurring", 10),
                                        ("flaky", "flaky", 10), ("resolved", "resolved", 5)]:
            if len(issues[category]) > limit:
                wf_text += f"_...+{len(issues[category]) - limit} more {label}_\n"

        # Per-job log links: find workflow_jobs matching this workflow prefix
        job_urls = [j["url"] for j in workflow_jobs
                    if j.get("url") and j["name"].split(" / ")[0] == wf_name
                    and j["conclusion"] == "failure"]
        if not job_urls:
            # Also try matching by test_type prefix for tracked jobs
            job_urls = [j["url"] for j in workflow_jobs
                        if j.get("url") and j["name"].startswith(wf_name)
                        and j["conclusion"] == "failure"]
        if job_urls:
            wf_text += f"<{job_urls[0]}|:link: View Logs>\n"

        # Chunk if needed
        while wf_text:
            chunk = wf_text[:2900]
            wf_blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": chunk.rstrip()},
            })
            wf_text = wf_text[2900:]

        print(make_payload(wf_blocks))

PYEOF
)

# ── Send messages ─────────────────────────────────────────────────────
echo "Sending consolidated Slack notification..."

THREAD_TS=""
FIRST=true

while IFS= read -r payload; do
    if [ "${FIRST}" = true ] && [ -n "${SLACK_BOT_TOKEN}" ] && [ -n "${SLACK_CHANNEL_ID}" ]; then
        # Post main message via chat.postMessage to get thread_ts
        BOT_PAYLOAD=$(python3 -c "
import json, sys
p = json.loads(sys.argv[1])
p['channel'] = sys.argv[2]
print(json.dumps(p))
" "${payload}" "${SLACK_CHANNEL_ID}")

        RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            --data "${BOT_PAYLOAD}" \
            "https://slack.com/api/chat.postMessage")

        THREAD_TS=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ts',''))" 2>/dev/null || echo "")
        OK=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")

        if [ "${OK}" != "True" ]; then
            echo "WARNING: chat.postMessage failed: ${RESPONSE}" >&2
            # Fall back to webhook for this and remaining messages
            THREAD_TS=""
            curl -s -X POST -H 'Content-type: application/json' --data "${payload}" "${SLACK_WEBHOOK_URL}" || true
        else
            echo "Main message posted (ts=${THREAD_TS})"
        fi
        FIRST=false
    elif [ -n "${THREAD_TS}" ] && [ -n "${SLACK_BOT_TOKEN}" ] && [ -n "${SLACK_CHANNEL_ID}" ]; then
        # Post thread reply via chat.postMessage
        THREAD_PAYLOAD=$(python3 -c "
import json, sys
p = json.loads(sys.argv[1])
p['channel'] = sys.argv[2]
p['thread_ts'] = sys.argv[3]
print(json.dumps(p))
" "${payload}" "${SLACK_CHANNEL_ID}" "${THREAD_TS}")

        RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            --data "${THREAD_PAYLOAD}" \
            "https://slack.com/api/chat.postMessage")

        OK=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")
        if [ "${OK}" != "True" ]; then
            echo "WARNING: Thread reply failed: ${RESPONSE}" >&2
        fi
    else
        # Fallback: webhook (no threading)
        response=$(curl -s -X POST \
            -H 'Content-type: application/json' \
            --data "${payload}" \
            "${SLACK_WEBHOOK_URL}")
        if [ "${response}" != "ok" ]; then
            echo "WARNING: Slack webhook returned: ${response}" >&2
        fi
        FIRST=false
    fi
done <<< "${PAYLOADS}"
echo "Consolidated Slack notification sent."

# ── Upload HTML report as file in thread ──────────────────────────────
if [ -n "${SLACK_BOT_TOKEN}" ] && [ -n "${SLACK_CHANNEL_ID}" ] && [ -n "${CONSOLIDATED_HTML}" ] && [ -f "${CONSOLIDATED_HTML}" ]; then
    echo "Uploading HTML report to Slack..."

    REPORT_DATE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('date','report'))" "${CONSOLIDATED_SUMMARY}" 2>/dev/null || echo "report")
    REPORT_BRANCH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('branch','main'))" "${CONSOLIDATED_SUMMARY}" 2>/dev/null || echo "main")
    UPLOAD_FILENAME="cuopt-nightly-${REPORT_BRANCH}-${REPORT_DATE}.html"
    FILE_SIZE=$(stat --format=%s "${CONSOLIDATED_HTML}")
    UPLOAD_TITLE="cuOpt Nightly Report — ${REPORT_BRANCH} — ${REPORT_DATE}"

    # Step 1: Get an upload URL from Slack
    URL_RESPONSE=$(curl -s -X POST \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        --data-urlencode "filename=${UPLOAD_FILENAME}" \
        --data-urlencode "length=${FILE_SIZE}" \
        "https://slack.com/api/files.getUploadURLExternal")

    UPLOAD_URL=$(echo "${URL_RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('upload_url',''))" 2>/dev/null)
    FILE_ID=$(echo "${URL_RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('file_id',''))" 2>/dev/null)

    if [ -z "${UPLOAD_URL}" ] || [ -z "${FILE_ID}" ]; then
        echo "WARNING: Slack file upload failed at getUploadURLExternal. Response: ${URL_RESPONSE}" >&2
    else
        # Step 2: Upload the file content to the presigned URL
        curl -s -X POST \
            -F "file=@${CONSOLIDATED_HTML}" \
            "${UPLOAD_URL}"

        # Step 3: Complete the upload and share to channel (in thread if available)
        COMPLETE_PAYLOAD=$(python3 -c "
import json, sys
payload = {
    'files': [{'id': sys.argv[1], 'title': sys.argv[2]}],
    'channel_id': sys.argv[3],
    'initial_comment': 'Full nightly test report \u2014 download and open in a browser for interactive details.',
}
thread_ts = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else ''
if thread_ts:
    payload['thread_ts'] = thread_ts
print(json.dumps(payload))
" "${FILE_ID}" "${UPLOAD_TITLE}" "${SLACK_CHANNEL_ID}" "${THREAD_TS}")

        COMPLETE_RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            --data "${COMPLETE_PAYLOAD}" \
            "https://slack.com/api/files.completeUploadExternal")

        if echo "${COMPLETE_RESPONSE}" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('ok') else 1)" 2>/dev/null; then
            echo "HTML report uploaded to Slack."
        else
            echo "WARNING: Slack file upload failed at completeUploadExternal. Response: ${COMPLETE_RESPONSE}" >&2
        fi
    fi
else
    if [ -n "${SLACK_BOT_TOKEN}" ] && [ -z "${SLACK_CHANNEL_ID}" ]; then
        echo "WARNING: SLACK_BOT_TOKEN set but SLACK_CHANNEL_ID missing, skipping file upload." >&2
    fi
fi
