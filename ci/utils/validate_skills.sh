#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate cuOpt agent skills and plugin manifests.
# Run from repo root: ./ci/utils/validate_skills.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SKILLS_DIR="skills"
CLAUDE_MARKETPLACE=".claude-plugin/marketplace.json"
AGENTS_MD="agents/AGENTS.md"
ERRORS=0

echo "Validating skills in $SKILLS_DIR..."

for dir in "$SKILLS_DIR"/*/; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir")
  skill_md="${dir}SKILL.md"
  if [ ! -f "$skill_md" ]; then
    echo "ERROR: $name missing SKILL.md"
    ERRORS=$((ERRORS + 1))
    continue
  fi
  if ! grep -q '^name:' "$skill_md" || ! grep -q '^description:' "$skill_md"; then
    echo "ERROR: $name/SKILL.md missing frontmatter (name: or description:)"
    ERRORS=$((ERRORS + 1))
  fi
done

if [ -f "$CLAUDE_MARKETPLACE" ]; then
  echo "Validating $CLAUDE_MARKETPLACE..."
  while IFS= read -r line; do
    path=$(echo "$line" | sed -n 's/.*"source"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    [ -z "$path" ] && continue
    path="${path#./}"
    if [ -n "$path" ] && [ ! -f "$path/SKILL.md" ]; then
      echo "ERROR: marketplace.json source missing SKILL.md: $path"
      ERRORS=$((ERRORS + 1))
    fi
  done < <(grep '"source"' "$CLAUDE_MARKETPLACE" || true)
  for dir in "$SKILLS_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    if ! grep -q "\"name\": \"$name\"" "$CLAUDE_MARKETPLACE"; then
      echo "ERROR: skill $name not listed in $CLAUDE_MARKETPLACE"
      ERRORS=$((ERRORS + 1))
    fi
  done
fi

if [ -f "$AGENTS_MD" ]; then
  echo "Validating $AGENTS_MD references..."
  for dir in "$SKILLS_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    if ! grep -q "$name" "$AGENTS_MD"; then
      echo "ERROR: agents/AGENTS.md does not reference skill: $name"
      ERRORS=$((ERRORS + 1))
    fi
  done
fi

if [ $ERRORS -gt 0 ]; then
  echo "Validation failed with $ERRORS error(s)."
  exit 1
fi
echo "All validations passed."
