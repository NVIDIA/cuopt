#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Gathers per-classifier JARs into one Maven-repository-layout tree, which is the form a
# publishing workflow consumes.
#
# Input:  one directory per classifier, as build_cuopt_java_jar.sh writes them, each holding
#           cuopt-<version>-<classifier>.jar
#           cuopt-<version>.pom
# Output: com/nvidia/cuopt/cuopt/<version>/ holding every classifier JAR, the sources and
#         javadoc JARs, and the POM named cuopt-<version>.pom.
#
# The POM must be named after the artifact rather than left as pom.xml, and the sources and
# javadoc JARs are required by Maven Central, so a bundle missing either is rejected late.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=java/cuopt/ci/argparse.sh
source "${SCRIPT_DIR}/argparse.sh"

GROUP_PATH="com/nvidia/cuopt"
ARTIFACT_ID="cuopt"
JARS_DIR=""
OUTPUT_DIR=""
EXTRA_JARS_DIR=""

print_help() {
  cat << 'EOF'
Usage: assemble_maven_repo.sh --jars-dir <path> --output-dir <path> [--extra-jars-dir <path>]

REQUIRED:
    -j, --jars-dir         Parent directory holding one subdirectory per classifier.
    -o, --output-dir       Directory to receive the Maven-repository layout. Must not exist
                           or must be empty, so a stale artifact cannot be published.

OPTIONS:
    -e, --extra-jars-dir   Directory holding the sources and javadoc JARs, normally
                           java/cuopt/target.
    -h, --help             Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h | --help) print_help; exit 0 ;;
    -j | --jars-dir) require_value "$1" "${2:-}"; JARS_DIR=$2; shift 2 ;;
    -o | --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR=$2; shift 2 ;;
    -e | --extra-jars-dir) require_value "$1" "${2:-}"; EXTRA_JARS_DIR=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; print_help >&2; exit 2 ;;
  esac
done

require_arg --jars-dir "${JARS_DIR}"
require_arg --output-dir "${OUTPUT_DIR}"

if [[ ! -d "${JARS_DIR}" ]]; then
  echo "jars directory not found: ${JARS_DIR}" >&2
  exit 1
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
  echo "output directory ${OUTPUT_DIR} is not empty; remove it before re-running" >&2
  exit 1
fi

# The version is read from a JAR name rather than the POM, so the layout can only ever describe
# artifacts that are actually present.
first_jar="$(find "${JARS_DIR}" -name "${ARTIFACT_ID}-*-*.jar" \
  ! -name '*-sources.jar' ! -name '*-javadoc.jar' -print -quit)"
if [[ -z "${first_jar}" ]]; then
  echo "no ${ARTIFACT_ID}-*.jar found under ${JARS_DIR}" >&2
  exit 1
fi
VERSION="$(basename "${first_jar}" | sed -E "s/^${ARTIFACT_ID}-([0-9][^-]*)-.*\.jar$/\1/")"
if [[ -z "${VERSION}" || "${VERSION}" == "$(basename "${first_jar}")" ]]; then
  echo "could not read a version from $(basename "${first_jar}")" >&2
  exit 1
fi

TARGET="${OUTPUT_DIR}/${GROUP_PATH}/${ARTIFACT_ID}/${VERSION}"
mkdir -p "${TARGET}"
echo "Assembling ${GROUP_PATH}/${ARTIFACT_ID}/${VERSION}"

classifiers=0
while IFS= read -r jar; do
  cp "${jar}" "${TARGET}/"
  echo "  $(basename "${jar}")"
  classifiers=$((classifiers + 1))
done < <(find "${JARS_DIR}" -name "${ARTIFACT_ID}-${VERSION}-*.jar" ! -name '*-sources.jar' ! -name '*-javadoc.jar' | sort)

if [[ "${classifiers}" -eq 0 ]]; then
  echo "no classifier JARs found for version ${VERSION}" >&2
  exit 1
fi

pom="$(find "${JARS_DIR}" -name "${ARTIFACT_ID}-${VERSION}.pom" -print -quit)"
if [[ -z "${pom}" ]]; then
  echo "no ${ARTIFACT_ID}-${VERSION}.pom found under ${JARS_DIR}" >&2
  exit 1
fi
cp "${pom}" "${TARGET}/${ARTIFACT_ID}-${VERSION}.pom"
echo "  ${ARTIFACT_ID}-${VERSION}.pom"

for kind in sources javadoc; do
  extra=""
  if [[ -n "${EXTRA_JARS_DIR}" ]]; then
    extra="$(find "${EXTRA_JARS_DIR}" -name "${ARTIFACT_ID}-${VERSION}-${kind}.jar" -print -quit)"
  fi
  if [[ -z "${extra}" ]]; then
    extra="$(find "${JARS_DIR}" -name "${ARTIFACT_ID}-${VERSION}-${kind}.jar" -print -quit)"
  fi
  if [[ -z "${extra}" ]]; then
    echo "WARNING: no ${kind} JAR found; Maven Central requires one before release" >&2
    continue
  fi
  cp "${extra}" "${TARGET}/"
  echo "  ${ARTIFACT_ID}-${VERSION}-${kind}.jar"
done

echo
echo "Maven repository layout at ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -type f | sed "s|^${OUTPUT_DIR}/|  |" | sort
