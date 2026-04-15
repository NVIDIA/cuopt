#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SONAR_SCANNER_VERSION="6.2.1.4610"

if [ -z "${SONAR_TOKEN:-}" ]; then
  echo "ERROR: SONAR_TOKEN environment variable is not set"
  exit 1
fi

BRANCH="${1:-main}"

# Install sonar-scanner CLI
echo "Installing sonar-scanner ${SONAR_SCANNER_VERSION}..."
SONAR_SCANNER_DIR="/tmp/sonar-scanner"
mkdir -p "${SONAR_SCANNER_DIR}"
curl -sSLo /tmp/sonar-scanner.zip \
  "https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}-linux-x64.zip"
unzip -q /tmp/sonar-scanner.zip -d "${SONAR_SCANNER_DIR}"
SONAR_SCANNER_BIN="${SONAR_SCANNER_DIR}/sonar-scanner-${SONAR_SCANNER_VERSION}-linux-x64/bin/sonar-scanner"
rm /tmp/sonar-scanner.zip

echo "Running SonarQube analysis for branch: ${BRANCH}"

"${SONAR_SCANNER_BIN}" \
  -Dsonar.branch.name="${BRANCH}"

echo "SonarQube analysis completed successfully"
