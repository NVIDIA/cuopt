#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run developer-skill agent tests: send prompts through Claude CLI with the
# cuOpt developer skill as context and validate responses (follow skill vs diverge).
#
# From repo root:
#   ./ci/run_dev_skill_agent_tests.sh           # live (requires: claude auth login)
#   ./ci/run_dev_skill_agent_tests.sh --replay ci/utils/dev_skill_responses  # validate saved responses
#   ./ci/run_dev_skill_agent_tests.sh --pass-at 5   # pass@5: run each request 5x, pass if any passes
#   ./ci/run_dev_skill_agent_tests.sh --runtimes-file out/runtimes.json  # write median runtimes
#   ./ci/run_dev_skill_agent_tests.sh --report out  # write results.csv + report.md to out/YYYY-MM-DD_HH-MM-SS/
#   ./ci/run_dev_skill_agent_tests.sh --save ci/utils/dev_skill_responses    # run once, save for replay

set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

exec python3 ci/utils/run_dev_skill_agent_tests.py "$@"
