#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sync skills/plugin version from repo root VERSION file.
# Run from repo root: ./ci/utils/sync_skills_version.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

VERSION_FILE="${REPO_ROOT}/VERSION"
if [[ ! -f "${VERSION_FILE}" ]]; then
  echo "ERROR: VERSION file not found at ${VERSION_FILE}"
  exit 1
fi

RELEASE_VERSION=$(tr -d ' \n\r' < "${VERSION_FILE}")
if [[ -z "${RELEASE_VERSION}" ]]; then
  echo "ERROR: VERSION file is empty"
  exit 1
fi

echo "Syncing skills version to ${RELEASE_VERSION} (from VERSION)..."

# .cursor-plugin/plugin.json and gemini-extension.json: top-level "version"
for f in .cursor-plugin/plugin.json gemini-extension.json; do
  if [[ -f "$f" ]]; then
    sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${RELEASE_VERSION}\"/" "$f"
    echo "  updated $f"
  fi
done

# .claude-plugin/marketplace.json: metadata.version
if [[ -f ".claude-plugin/marketplace.json" ]]; then
  sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${RELEASE_VERSION}\"/" .claude-plugin/marketplace.json
  echo "  updated .claude-plugin/marketplace.json"
fi

# skills/*/SKILL.md: add or update version in YAML frontmatter (after name:)
SKILLS_DIR="skills"
for skill_md in "${SKILLS_DIR}"/*/SKILL.md; do
  [[ -f "$skill_md" ]] || continue
  if grep -q '^version:' "$skill_md" 2>/dev/null; then
    sed -i "s/^version:.*/version: \"${RELEASE_VERSION}\"/" "$skill_md"
  else
    sed -i "/^name:/a version: \"${RELEASE_VERSION}\"" "$skill_md"
  fi
  echo "  updated $skill_md"
done

echo "Done. Skills version is now ${RELEASE_VERSION}."
