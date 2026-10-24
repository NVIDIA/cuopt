#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Creates a deployment bundle from a signed Maven repository tree, uploads it to
# the Central Publisher Portal with the Publisher API, and polls until the
# deployment passes validation.
# Never calls the publish endpoint: publication always requires a human click.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/argparse.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/maven_utils.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/maven_central_publish_steps.sh"

CENTRAL_PORTAL_URL="https://central.sonatype.com"
# Used by wait_for_validated in the sourced step library.
# shellcheck disable=SC2034
POLL_INTERVAL_SEC=15
# shellcheck disable=SC2034
POLL_TIMEOUT_SEC=900

INPUT_DIR=""
GROUP_ID=""
ARTIFACT_ID=""
VERSION=""
OUTPUT_BUNDLE=""
AUTO_DROP="true"

print_help() {
  cat << EOF

Usage: maven_central_publish.sh --input <path> --group-id <g> \\
                                --artifact-id <a> --version <v> \\
                                --output-bundle <path> [OPTIONS]

Copies a signed Maven repository tree into scratch space, generates .md5 and
.sha1 sidecars, and creates a retained ZIP. It uploads the ZIP to the Central
Publisher Portal with the Publisher API as a USER_MANAGED deployment, so it is
never published without a human click.

REQUIRED:
    -i, --input                    Signed Maven repository directory.
    -g, --group-id                 Maven groupId, e.g. ai.rapids.
    -a, --artifact-id              Maven artifactId, e.g. cudf.
    -v, --version                  Release version, e.g. 1.2.3.
    -o, --output-bundle            Path for the retained deployment ZIP.

OPTIONS:
    --auto-drop <true|false>       Drop after VALIDATED (default: true).
    -h, --help                     Show this help message.

ENVIRONMENT VARIABLES:
    MAVEN_DEPLOY_USERNAME          Publisher Portal user token username.
    MAVEN_DEPLOY_TOKEN             Publisher Portal user token password.
    GITHUB_OUTPUT                  If set, receives DEPLOYMENT_ID.

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
      --auto-drop)
        require_value "$1" "${2:-}"
        AUTO_DROP=$2
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

if [[ ${AUTO_DROP} != "true" && ${AUTO_DROP} != "false" ]]; then
  fatal "--auto-drop must be 'true' or 'false' (got '${AUTO_DROP}')"
fi
require_maven_coordinates "${GROUP_ID}" "${ARTIFACT_ID}" "${VERSION}"
require_release_version "${VERSION}"
if [[ ! -d ${INPUT_DIR} ]]; then
  fatal "--input '${INPUT_DIR}' does not exist or is not a directory"
fi
: "${MAVEN_DEPLOY_USERNAME:?must be set}"
: "${MAVEN_DEPLOY_TOKEN:?must be set}"

require_cmds base64 curl jq zip md5sum sha1sum

INPUT_DIR="$(cd "${INPUT_DIR}" && pwd)"
OUTPUT_BUNDLE_PARENT="$(dirname "${OUTPUT_BUNDLE}")"
mkdir -p "${OUTPUT_BUNDLE_PARENT}"
OUTPUT_BUNDLE_PARENT="$(cd "${OUTPUT_BUNDLE_PARENT}" && pwd)"
OUTPUT_BUNDLE="${OUTPUT_BUNDLE_PARENT}/$(basename "${OUTPUT_BUNDLE}")"
if [[ -e ${OUTPUT_BUNDLE} ]]; then
  fatal "--output-bundle '${OUTPUT_BUNDLE}' already exists"
fi
echo "Maven Central upload"
echo "  coordinates:   ${GROUP_ID}:${ARTIFACT_ID}:${VERSION}"
echo "  auto-drop:     ${AUTO_DROP}"
echo "  input dir:     ${INPUT_DIR}"
echo "  output bundle: ${OUTPUT_BUNDLE}"
echo "  portal:        ${CENTRAL_PORTAL_URL}"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

BUNDLE_DIR="${WORK_DIR}/bundle"
BUNDLE_ARTIFACT_DIR="${BUNDLE_DIR}/$(maven_group_path "${GROUP_ID}")/${ARTIFACT_ID}/${VERSION}"
mkdir -p "${BUNDLE_DIR}"

copy_bundle "${INPUT_DIR}" "${BUNDLE_DIR}"
require_bundle_contents "${BUNDLE_ARTIFACT_DIR}"
generate_bundle_checksums "${BUNDLE_ARTIFACT_DIR}"

BUNDLE_ZIP="${WORK_DIR}/${ARTIFACT_ID}-${VERSION}.zip"
create_bundle_zip "${BUNDLE_ZIP}" "${BUNDLE_DIR}"
mv "${BUNDLE_ZIP}" "${OUTPUT_BUNDLE}"

PUBLISHING_TYPE="USER_MANAGED"
echo "Uploading bundle to the Publisher Portal"
DEPLOYMENT_ID=$(portal_upload_bundle "${OUTPUT_BUNDLE}" "${GROUP_ID}:${ARTIFACT_ID}:${VERSION}" "${PUBLISHING_TYPE}")
if [[ ! ${DEPLOYMENT_ID} =~ ^[A-Za-z0-9-]+$ ]]; then
  fatal "Publisher Portal returned an invalid deployment id"
fi
echo "  Portal deployment id: ${DEPLOYMENT_ID}"
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  echo "DEPLOYMENT_ID=${DEPLOYMENT_ID}" >> "${GITHUB_OUTPUT}"
fi

wait_for_validated "${DEPLOYMENT_ID}"

if [[ ${AUTO_DROP} == "true" ]]; then
  drop_deployment "${DEPLOYMENT_ID}"
else
  cat <<EOF
Deployment left in PENDING state.
  Deployment id:  ${DEPLOYMENT_ID}
  Portal UI:      ${CENTRAL_PORTAL_URL}/publishing/deployments
A human must review the deployment and click Publish to release it to
Maven Central.
EOF
fi
