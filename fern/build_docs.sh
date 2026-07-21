#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build Fern docs: generate MDX, validate, then preview or publish.
# Usage:
#   fern/build_docs.sh              # local preview (http://localhost:3000)
#   fern/build_docs.sh --check          # validate only (fern check), no server
#   fern/build_docs.sh --publish-docs   # publish to Fern cloud (production)
#   fern/build_docs.sh --preview        # CI PR preview (fern generate --docs --preview)
#
# Prerequisites: node, npm, jq, and a conda environment with Python + numpydoc.
# Run from the repo root.

set -e

REPODIR=$(cd "$(dirname "$0")/.."; pwd)
PUBLISH=0
PREVIEW=0
CHECK=0
for arg in "$@"; do
    [[ "$arg" == "--publish-docs" ]] && PUBLISH=1
    [[ "$arg" == "--preview" ]] && PREVIEW=1
    [[ "$arg" == "--check" ]] && CHECK=1
done

if ! command -v node &>/dev/null || ! command -v npm &>/dev/null; then
    echo "ERROR: Node.js (with npm) is required for the Fern CLI."
    echo "       Install from: https://nodejs.org/  (or: conda install nodejs)"
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required to read the Fern version pin."
    echo "       Install: sudo apt-get install jq"
    exit 1
fi

# Install Fern CLI at the version pinned in fern/fern.config.json
FERN_VERSION=$(jq -r .version "${REPODIR}/fern/fern.config.json")
if ! fern --version 2>/dev/null | grep -q "${FERN_VERSION}"; then
    echo "Installing fern-api@${FERN_VERSION}..."
    npm install -g "fern-api@${FERN_VERSION}"
fi

# Regenerate dynamic API reference pages
PY=${PYTHON:-python3}
if command -v "${PY}" &>/dev/null; then
    "${PY}" "${REPODIR}/fern/generate_api_docs.py"
else
    echo "  [WARN] Python not found; skipping API doc generation."
fi

echo "Running fern check..."
fern check

if [[ "${CHECK}" -eq 1 ]]; then
    echo "Check-only mode; fern check already passed above."
elif [[ "${PUBLISH}" -eq 1 ]]; then
    if [[ -z "${FERN_TOKEN:-}" ]]; then
        echo "ERROR: FERN_TOKEN environment variable is not set."
        exit 1
    fi
    echo "Publishing to Fern cloud..."
    fern generate --docs
    echo "Docs published to https://nvidia-cuopt.docs.buildwithfern.com"
elif [[ "${PREVIEW}" -eq 1 ]]; then
    if [[ -z "${FERN_TOKEN:-}" ]]; then
        echo "FERN_TOKEN not set; skipping PR preview publish (fern check already passed above)."
    else
        echo "Publishing Fern PR preview..."
        fern generate --docs --preview
    fi
else
    echo ""
    echo "Starting local preview at http://localhost:3000 (Ctrl+C to stop)..."
    fern docs dev
fi
