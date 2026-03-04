#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prints "true" if routing-related files changed vs origin/main, else "false".
# Only used when running under the PR workflow (refs/heads/pull-request/*); other
# workflows (nightly, manual test.yaml) run all tests and do not call this script.
# Must be run from the repository root.

set -euo pipefail

ROUTING_PATTERNS=(
    'python/cuopt/cuopt/routing/'
    'python/cuopt/cuopt/tests/routing/'
    'cpp/src/routing/'
    'cpp/include/cuopt/routing/'
    'cpp/tests/routing/'
)

# If explicitly set by caller, respect it
if [[ -n "${CUOPT_RUN_ROUTING_TESTS:-}" ]]; then
    if [[ "${CUOPT_RUN_ROUTING_TESTS}" == "false" ]]; then
        echo "false"
        exit 0
    fi
    echo "true"
    exit 0
fi

# Not in a git repo or no merge base: run routing tests to be safe
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "true"
    exit 0
fi

MB=""
if git fetch origin main --depth=500 2>/dev/null; then
    MB=$(git merge-base HEAD origin/main 2>/dev/null) || true
fi
if [[ -z "${MB:-}" ]]; then
    echo "true"
    exit 0
fi

CHANGED=$(git diff --name-only "${MB}...HEAD" 2>/dev/null) || { echo "true"; exit 0; }
for pat in "${ROUTING_PATTERNS[@]}"; do
    if echo "${CHANGED}" | grep -qE "^${pat}"; then
        echo "true"
        exit 0
    fi
done
echo "false"
