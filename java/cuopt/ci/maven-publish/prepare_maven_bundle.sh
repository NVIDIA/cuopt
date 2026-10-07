#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Host wrapper that prepares a signed Maven repository tree in a container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/argparse.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/maven_utils.sh"

INPUT_DIR=""
OUTPUT_DIR=""
IMAGE="maven:3-eclipse-temurin-17"

print_help() {
  cat << EOF

Usage: prepare_maven_bundle.sh --input <path> --output <path>

Copies a Maven repository directory into a clean output tree and GPG-signs
every publishable file. The input tree is mounted read-only and is not changed.
Both release and -SNAPSHOT versions are accepted. The destination check
(Maven Central vs Sonatype snapshots) happens in the publish scripts.

REQUIRED:
    -i, --input                Maven repository directory to sign.
    -o, --output               Empty directory for the signed Maven tree.

ENVIRONMENT VARIABLES:
    GPG_PRIVATE_KEY            Armored GPG private key (required).
    GPG_PASSPHRASE             Passphrase for the GPG private key (required).
    GITHUB_OUTPUT              If set, receives GROUP_ID, ARTIFACT_ID, VERSION.

EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        print_help
        exit 0
        ;;
      -i|--input)
        require_value "$1" "${2:-}"
        INPUT_DIR=$2
        shift 2
        ;;
      -o|--output)
        require_value "$1" "${2:-}"
        OUTPUT_DIR=$2
        shift 2
        ;;
      *)
        echo "Error: Unknown argument $1"
        print_help
        exit 1
        ;;
    esac
  done
}

parse_args "$@"

require_arg --input "${INPUT_DIR}"
require_arg --output "${OUTPUT_DIR}"

if [[ ! -d ${INPUT_DIR} ]]; then
  fatal "--input '${INPUT_DIR}' does not exist or is not a directory"
fi

: "${GPG_PRIVATE_KEY:?must be set}"
: "${GPG_PASSPHRASE:?must be set}"

INPUT_DIR="$(cd "${INPUT_DIR}" && pwd)"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

if [[ ${INPUT_DIR} == "${OUTPUT_DIR}" ]]; then
  fatal "--input and --output must be different directories"
fi
if [[ -n $(find "${OUTPUT_DIR}" ! -path "${OUTPUT_DIR}" -print -quit) ]]; then
  fatal "--output '${OUTPUT_DIR}' must be empty"
fi

METADATA_SCRATCH="$(mktemp -d)"
trap 'rm -rf "${METADATA_SCRATCH}"' EXIT

echo "Prepare signed Maven bundle"
echo "  image:      ${IMAGE}"
echo "  input dir:  ${INPUT_DIR}"
echo "  output dir: ${OUTPUT_DIR}"

DOCKER_ARGS=(
  --rm
  --volume "${INPUT_DIR}:/input:ro"
  --volume "${OUTPUT_DIR}:/bundle"
  --volume "${METADATA_SCRATCH}:/metadata"
  --workdir /bundle
  --env GPG_PRIVATE_KEY="${GPG_PRIVATE_KEY}"
  --env GPG_PASSPHRASE="${GPG_PASSPHRASE}"
  --env HOST_UID="$(id -u)"
  --env HOST_GID="$(id -g)"
  --volume "${SCRIPT_DIR}:/scripts:ro"
)

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" \
  bash /scripts/prepare_maven_bundle_in_container.sh

METADATA_FILE="${METADATA_SCRATCH}/bundle_metadata.env"
if [[ ! -f ${METADATA_FILE} ]]; then
  fatal "worker did not produce ${METADATA_FILE}"
fi

GROUP_ID=$(sed -n 's/^GROUP_ID=//p' "${METADATA_FILE}")
ARTIFACT_ID=$(sed -n 's/^ARTIFACT_ID=//p' "${METADATA_FILE}")
VERSION=$(sed -n 's/^VERSION=//p' "${METADATA_FILE}")
require_maven_coordinates "${GROUP_ID}" "${ARTIFACT_ID}" "${VERSION}"

{
  echo "GROUP_ID=${GROUP_ID}"
  echo "ARTIFACT_ID=${ARTIFACT_ID}"
  echo "VERSION=${VERSION}"
} | tee -a "${GITHUB_OUTPUT:-/dev/null}"

echo "Signed bundle prepared for ${GROUP_ID}:${ARTIFACT_ID}:${VERSION}"
