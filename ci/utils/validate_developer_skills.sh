#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate developer skill content: required sections and key commands.
# Run from repo root: ./ci/utils/validate_developer_skills.sh
# Use this to ensure developer SKILL.md files stay in sync with the repo workflow.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SKILLS_DIR="skills"
ERRORS=0

# Developer skill names (directories under skills/)
DEV_SKILLS=("cuopt-developer" "cuopt-installation-developer")

# cuopt-developer: required section headings (must appear in SKILL.md)
CUOPT_DEV_SECTIONS=(
  "Build & Test"
  "Run Tests"
  "Before You Commit"
  "Safety Rules"
)

# Key phrases that should appear (workflow commands / critical guidance)
CUOPT_DEV_PHRASES=(
  "./build.sh"
  "ctest"
  "pytest"
  "check_style.sh"
  "commit -s"
  "DCO"
)

# cuopt-installation-developer: required concepts (build from source, tests)
INSTALL_DEV_PHRASES=(
  "build"
  "test"
  "CONTRIBUTING"
)

check_skill_file() {
  local skill_name="$1"
  local skill_md="${SKILLS_DIR}/${skill_name}/SKILL.md"
  if [[ ! -f "$skill_md" ]]; then
    echo "SKIP: ${skill_name} (no SKILL.md)"
    return 0
  fi
  local content
  content=$(cat "$skill_md")
  local failed=0

  if [[ "$skill_name" == "cuopt-developer" ]]; then
    for section in "${CUOPT_DEV_SECTIONS[@]}"; do
      if ! echo "$content" | grep -q "$section"; then
        echo "ERROR: ${skill_name}/SKILL.md missing section or heading: ${section}"
        ERRORS=$((ERRORS + 1))
        failed=1
      fi
    done
    for phrase in "${CUOPT_DEV_PHRASES[@]}"; do
      if ! echo "$content" | grep -qF "$phrase"; then
        echo "ERROR: ${skill_name}/SKILL.md missing required phrase: ${phrase}"
        ERRORS=$((ERRORS + 1))
        failed=1
      fi
    done
  fi

  if [[ "$skill_name" == "cuopt-installation-developer" ]]; then
    for phrase in "${INSTALL_DEV_PHRASES[@]}"; do
      if ! echo "$content" | grep -qi "$phrase"; then
        echo "ERROR: ${skill_name}/SKILL.md missing required concept: ${phrase}"
        ERRORS=$((ERRORS + 1))
        failed=1
      fi
    done
  fi

  if [[ $failed -eq 0 ]]; then
    echo "PASS: ${skill_name}"
  fi
}

echo "Validating developer skills in $SKILLS_DIR..."
for name in "${DEV_SKILLS[@]}"; do
  check_skill_file "$name"
done

if [[ $ERRORS -gt 0 ]]; then
  echo "Validation failed with $ERRORS error(s)."
  exit 1
fi
echo "All developer skill validations passed."
exit 0
