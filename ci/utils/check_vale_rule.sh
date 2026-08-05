#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Self-test for the cuOpt.Headings Vale rule.
#
# Vale skips an entire heading when an `exceptions` entry matches it, so a
# careless entry can silently stop the rule from checking anything. This
# asserts the rule still flags what it should and still accepts what it
# should, independently of what the real docs happen to contain.
#
# Add a case below when you add an exception.

set -euo pipefail

# Headings that violate title case. Every one must be reported.
MUST_FLAG=(
    "Process model"
    "cuOpt process model"
    "gRPC job states"
    "Using MPS or LP file directly"
)

# Correct headings. None may be reported.
MUST_PASS=(
    "Where to Find Examples"
    "Connect and Solve"
    "How mTLS Works"
    "Start the Server with TLS"
    "Job States"
)

STYLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../vale/styles" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

cat > "${WORK_DIR}/.vale.ini" <<EOF
StylesPath = ${STYLES_DIR}
MinAlertLevel = error

[*.md]
BasedOnStyles = cuOpt
EOF

write_doc() {
    local path=$1
    shift
    printf '# Fixture\n\n' > "${path}"
    printf '## %s\n\n' "$@" >> "${path}"
}

write_doc "${WORK_DIR}/must-flag.md" "${MUST_FLAG[@]}"
write_doc "${WORK_DIR}/must-pass.md" "${MUST_PASS[@]}"

STATUS=0

reported=$(vale --config="${WORK_DIR}/.vale.ini" --output=line \
    "${WORK_DIR}/must-flag.md" 2>&1 || true)
for heading in "${MUST_FLAG[@]}"; do
    if [[ "${reported}" != *"${heading}"* ]]; then
        if [ "${STATUS}" -eq 0 ]; then
            echo "ERROR: the Headings rule no longer flags these violations:"
        fi
        echo "  - ${heading}"
        STATUS=1
    fi
done
if [ "${STATUS}" -ne 0 ]; then
    echo "An over-broad 'exceptions' entry in Headings.yml is the usual cause."
fi

if ! output=$(vale --config="${WORK_DIR}/.vale.ini" --output=line \
    "${WORK_DIR}/must-pass.md" 2>&1); then
    echo "ERROR: the Headings rule flags headings that are already correct:"
    echo "${output}" | sed 's/^/  /'
    STATUS=1
fi

if [ "${STATUS}" -eq 0 ]; then
    echo "Headings rule self-test passed."
fi

exit "${STATUS}"
