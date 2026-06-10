#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verify that committed codegen output matches what generate_conversions.py produces.
# Fails if a developer edited field_registry.yaml without re-running ./build.sh codegen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODEGEN_DIR="${REPO_DIR}/cpp/src/grpc/codegen"
GENERATED_DIR="${CODEGEN_DIR}/generated"

TMPDIR=$(mktemp -d)
trap 'rm -rf ${TMPDIR}' EXIT

echo "Running code generator into temp directory..."
python "${CODEGEN_DIR}/generate_conversions.py" \
    --registry "${CODEGEN_DIR}/field_registry.yaml" \
    --output-dir "${TMPDIR}"

echo "Comparing generated output with committed files..."

FAILED=0

for f in "${TMPDIR}"/*; do
    fname=$(basename "$f")
    committed="${GENERATED_DIR}/${fname}"
    if [ ! -f "${committed}" ]; then
        echo "MISSING: ${committed} (new generated file not committed)"
        FAILED=1
        continue
    fi
    if ! diff -q "$f" "${committed}" > /dev/null 2>&1; then
        echo "MISMATCH: cpp/src/grpc/codegen/generated/${fname}"
        diff -u "${committed}" "$f" | head -30
        FAILED=1
    fi
done

for committed in "${GENERATED_DIR}"/*; do
    [ -f "${committed}" ] || continue
    fname=$(basename "${committed}")
    if [ ! -f "${TMPDIR}/${fname}" ]; then
        echo "ORPHANED: cpp/src/grpc/codegen/generated/${fname} (no longer generated)"
        FAILED=1
    fi
done

if [ ${FAILED} -ne 0 ]; then
    echo ""
    echo "ERROR: Committed generated files are out of sync with field_registry.yaml."
    echo "Run './build.sh codegen' and commit the results."
    exit 1
fi

echo "OK: All generated files match field_registry.yaml."
