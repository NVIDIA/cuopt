#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

## Usage
# Primary interface: ./ci/release/update-version.sh --run-context=main|release <new_version>
# Fallback: Environment variable support for automation needs
# NOTE: Must be run from the root of the repository
#
# CLI args take precedence when both are provided
# If neither RUN_CONTEXT nor --run-context is provided, defaults to main
#
# Examples:
#   ./ci/release/update-version.sh --run-context=main 25.12.00
#   ./ci/release/update-version.sh --run-context=release 25.12.00
#   RAPIDS_RUN_CONTEXT=main ./ci/release/update-version.sh 25.12.00

# Verify we're running from the repository root
if [[ ! -f "VERSION" ]] || [[ ! -f "ci/release/update-version.sh" ]] || [[ ! -d "python" ]]; then
    echo "Error: This script must be run from the root of the cuopt repository"
    echo ""
    echo "Usage:"
    echo "  cd /path/to/cuopt"
    echo "  ./ci/release/update-version.sh --run-context=main|release <new_version>"
    echo ""
    echo "Example:"
    echo "  ./ci/release/update-version.sh --run-context=main 25.12.00"
    exit 1
fi

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-context=*)
      CLI_RUN_CONTEXT="${1#*=}"
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Determine RUN_CONTEXT with precedence: CLI > Environment > Default
if [[ -n "${CLI_RUN_CONTEXT:-}" ]]; then
    RUN_CONTEXT="${CLI_RUN_CONTEXT}"
    echo "Using run-context from CLI: ${RUN_CONTEXT}"
elif [[ -n "${RAPIDS_RUN_CONTEXT:-}" ]]; then
    RUN_CONTEXT="${RAPIDS_RUN_CONTEXT}"
    echo "Using RUN_CONTEXT from environment: ${RUN_CONTEXT}"
else
    RUN_CONTEXT="main"
    echo "Using default run-context: ${RUN_CONTEXT}"
fi

# Validate RUN_CONTEXT
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context '${RUN_CONTEXT}'. Must be 'main' or 'release'"
    exit 1
fi

NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
DOCKER_TAG=$(echo "$NEXT_FULL_TAG" | sed -E 's/^([0-9]{2})\.0*([0-9]+)\.0*([0-9]+).*/\1.\2.\3/')

# Need to distutils-normalize the versions for some use cases (RAPIDS)
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")

# Determine branch name based on context
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > ./VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

DEPENDENCIES=(
  cudf
  cuopt
  cuopt-mps-parser
  cuopt-server
  cuopt-sh-client
  libcuopt
  libcuopt-tests
  libraft-headers
  librmm
  pylibraft
  rmm
)

for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}\(\[.*\]\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
  for FILE in docs/cuopt/source/*/quick-start.rst README.md; do
    sed_runner "/${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==[0-9]\+\.[0-9]\+\.\*/==${NEXT_SHORT_TAG_PEP440}.\*/g" "${FILE}"
    sed_runner "/${DEP}=/ s/=[0-9]\+\.[0-9]\+\(\.[0-9]\+\)\?[^ ]*/=${NEXT_SHORT_TAG}.*/g" "${FILE}"
    sed_runner "/${DEP}:/ s/:[0-9]\{2\}\.[0-9]\{1,2\}\.[0-9]\+\(-cuda[0-9]\+\.[0-9]\+-\)\(py[0-9]\+\)/:${DOCKER_TAG}\1\2/g" "${FILE}"
  done
done

# Update README.md version badge
sed_runner 's/badge\/version-[0-9]\+\.[0-9]\+\.[0-9]\+-blue/badge\/version-'${NEXT_FULL_TAG}'-blue/g' README.md

# Update nightly
sed_runner 's/'"cuopt_version: \"[0-9][0-9].[0-9][0-9]\""'/'"cuopt_version: \"${NEXT_SHORT_TAG}\""'/g' .github/workflows/nightly.yaml

# Update Helm chart files
sed_runner 's/\(tag: "\)[0-9][0-9]\.[0-9]\+\.[0-9]\+\(-cuda12\.9-py3\.12"\)/\1'${DOCKER_TAG}'\2/g' helmchart/cuopt-server/values.yaml
sed_runner 's/\(appVersion: \)[0-9][0-9]\.[0-9]\+\.[0-9]\+/\1'${DOCKER_TAG}'/g' helmchart/cuopt-server/Chart.yaml
sed_runner 's/\(version: \)[0-9][0-9]\.[0-9]\+\.[0-9]\+/\1'${DOCKER_TAG}'/g' helmchart/cuopt-server/Chart.yaml

# CI files - context-aware branch references and version updates
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  # CI image tags of the form {rapids_version}-{something}
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# Documentation references - context-aware
if [[ "${RUN_CONTEXT}" == "main" ]]; then
  # In main context, keep external documentation on main (no changes needed)
  :
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
  # In release context, use release branch for external documentation links (word boundaries to avoid partial matches)
  sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" docs/cuopt/source/faq.rst
  sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" docs/cuopt/source/cuopt-python/routing/routing-example.ipynb
fi
