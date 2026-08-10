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

# skills/*/skill-card.md: update the version on the line after "## Skill Version(s):"
for skill_card in "${SKILLS_DIR}"/*/skill-card.md; do
  [[ -f "$skill_card" ]] || continue
  # Scope the substitution to the line immediately following the header.
  sed -i '/^## Skill Version(s): <br>$/{n; s|^[0-9][0-9]\.[0-9][0-9]\.[0-9][0-9]\( (source:.*) <br>\)$|'"${RELEASE_VERSION}"'\1|}' "$skill_card"
  # Verify the replacement actually landed; fail loudly if the version line is missing or malformed.
  if ! grep -A1 "^## Skill Version(s):" "$skill_card" | grep -q "^${RELEASE_VERSION}"; then
    echo "ERROR: $skill_card: version line under '## Skill Version(s):' is missing or malformed; manual repair needed"
    exit 1
  fi
  echo "  updated $skill_card"
done

echo "Done. Skills version is now ${RELEASE_VERSION}."
