#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Self-test for the cuOpt.Headings Vale rule.
#
# Vale skips an entire heading when an `exceptions` entry matches it, so a
# careless entry can silently stop the rule from checking anything. This
# asserts the rule still flags what it should, and still accepts what it
# should, independently of what the real docs happen to contain.

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../vale/tests" && pwd)"
CONFIG="${TESTS_DIR}/.vale.ini"
STATUS=0

# Each heading in must-flag.md is a violation; every one must be reported.
missing=$(python3 - "$TESTS_DIR" "$CONFIG" <<'PY'
import re
import subprocess
import sys

tests_dir, config = sys.argv[1], sys.argv[2]
path = f"{tests_dir}/must-flag.md"

expected = [m.group(1) for m in re.finditer(r"^## (.+)$", open(path).read(), re.M)]
proc = subprocess.run(
    ["vale", f"--config={config}", "--output=line", path],
    capture_output=True, text=True,
)
reported = proc.stdout
print("\n".join(h for h in expected if h not in reported))
PY
)

if [ -n "$missing" ]; then
    echo "ERROR: the Headings rule no longer flags these violations:"
    echo "$missing" | sed 's/^/  - /'
    echo "An over-broad 'exceptions' entry in Headings.yml is the usual cause."
    STATUS=1
fi

# Nothing in must-pass.md may be reported.
if ! output=$(vale --config="$CONFIG" --output=line "${TESTS_DIR}/must-pass.md" 2>&1); then
    echo "ERROR: the Headings rule flags headings that are already correct:"
    echo "$output" | sed 's/^/  /'
    STATUS=1
fi

if [ "$STATUS" -eq 0 ]; then
    echo "Headings rule self-test passed."
fi

exit "$STATUS"
