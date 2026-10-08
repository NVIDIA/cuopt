#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Deploys the Maven repository tree under --input to the Sonatype snapshot
# repository via 'mvn deploy-file' inside a container, then downloads what
# Nexus stored into a retained ZIP at --output-bundle. VERSION must end with
# '-SNAPSHOT'.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/argparse.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/maven_utils.sh"

INPUT_DIR=""
GROUP_ID=""
ARTIFACT_ID=""
VERSION=""
OUTPUT_BUNDLE=""
IMAGE="maven:3-eclipse-temurin-17"

print_help() {
  cat << EOF

Usage: maven_snapshot_publish.sh --input <path> --group-id <g> \\
                                 --artifact-id <a> --version <v> \\
                                 --output-bundle <path>

Deploys the Maven repository tree under --input to the Sonatype snapshot
repository via 'mvn deploy-file' inside a container, then downloads what
Nexus stored into a retained ZIP at --output-bundle. VERSION must end with
'-SNAPSHOT'.

REQUIRED:
    -i, --input                    Maven repository directory to deploy.
    -g, --group-id                 Maven groupId, e.g. ai.rapids.
    -a, --artifact-id              Maven artifactId, e.g. cudf.
    -v, --version                  Snapshot version, e.g. 1.2.3-SNAPSHOT.
    -o, --output-bundle            Path for the retained snapshot ZIP.

OPTIONS:
    -h, --help                     Show this help message.

ENVIRONMENT VARIABLES:
    MAVEN_DEPLOY_USERNAME          Publisher Portal user token username.
    MAVEN_DEPLOY_TOKEN             Publisher Portal user token password.

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
      -g|--group-id)
        require_value "$1" "${2:-}"
        GROUP_ID=$2
        shift 2
        ;;
      -a|--artifact-id)
        require_value "$1" "${2:-}"
        ARTIFACT_ID=$2
        shift 2
        ;;
      -v|--version)
        require_value "$1" "${2:-}"
        VERSION=$2
        shift 2
        ;;
      -o|--output-bundle)
        require_value "$1" "${2:-}"
        OUTPUT_BUNDLE=$2
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
require_arg --group-id "${GROUP_ID}"
require_arg --artifact-id "${ARTIFACT_ID}"
require_arg --version "${VERSION}"
require_arg --output-bundle "${OUTPUT_BUNDLE}"

require_maven_coordinates "${GROUP_ID}" "${ARTIFACT_ID}" "${VERSION}"
require_snapshot_version "${VERSION}"

if [[ ! -d ${INPUT_DIR} ]]; then
  fatal "--input '${INPUT_DIR}' does not exist or is not a directory"
fi
: "${MAVEN_DEPLOY_USERNAME:?must be set}"
: "${MAVEN_DEPLOY_TOKEN:?must be set}"

require_cmds docker

INPUT_DIR="$(cd "${INPUT_DIR}" && pwd)"
OUTPUT_BUNDLE_PARENT="$(dirname "${OUTPUT_BUNDLE}")"
mkdir -p "${OUTPUT_BUNDLE_PARENT}"
OUTPUT_BUNDLE_PARENT="$(cd "${OUTPUT_BUNDLE_PARENT}" && pwd)"
OUTPUT_BUNDLE="${OUTPUT_BUNDLE_PARENT}/$(basename "${OUTPUT_BUNDLE}")"
if [[ -e ${OUTPUT_BUNDLE} ]]; then
  fatal "--output-bundle '${OUTPUT_BUNDLE}' already exists"
fi

BUNDLE_SCRATCH="$(mktemp -d)"
trap 'rm -rf "${BUNDLE_SCRATCH}"' EXIT

echo "Sonatype snapshot deploy: ${GROUP_ID}:${ARTIFACT_ID}:${VERSION}"

export MAVEN_DEPLOY_USERNAME MAVEN_DEPLOY_TOKEN \
       GROUP_ID ARTIFACT_ID VERSION

docker run \
  --rm \
  --volume "${INPUT_DIR}:/input:ro" \
  --volume "${BUNDLE_SCRATCH}:/bundle" \
  --volume "${SCRIPT_DIR}:/scripts:ro" \
  --workdir /bundle \
  --env MAVEN_DEPLOY_USERNAME --env MAVEN_DEPLOY_TOKEN \
  --env GROUP_ID --env ARTIFACT_ID --env VERSION \
  --env HOST_UID="$(id -u)" --env HOST_GID="$(id -g)" \
  "${IMAGE}" \
  bash /scripts/maven_snapshot_publish_in_container.sh

BUNDLE_IN_SCRATCH="${BUNDLE_SCRATCH}/${ARTIFACT_ID}-${VERSION}.zip"
if [[ ! -f ${BUNDLE_IN_SCRATCH} ]]; then
  fatal "container did not produce ${BUNDLE_IN_SCRATCH}"
fi
mv "${BUNDLE_IN_SCRATCH}" "${OUTPUT_BUNDLE}"
