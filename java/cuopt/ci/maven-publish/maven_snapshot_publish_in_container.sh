#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container worker for maven_snapshot_publish.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/maven_utils.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/prepare_maven_bundle_steps.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/maven_snapshot_publish_steps.sh"

: "${MAVEN_DEPLOY_USERNAME:?must be set}"
: "${MAVEN_DEPLOY_TOKEN:?must be set}"
: "${GROUP_ID:?must be set}"
: "${ARTIFACT_ID:?must be set}"
: "${VERSION:?must be set}"
: "${HOST_UID:?must be set}"
: "${HOST_GID:?must be set}"

INPUT_DIR=/input
BUNDLE_DIR=/bundle
SNAPSHOT_REPOSITORY_URL="https://central.sonatype.com/repository/maven-snapshots"

trap 'chown -R "${HOST_UID}:${HOST_GID}" "${BUNDLE_DIR}" 2>/dev/null || true' EXIT

install_container_deps
require_cmds curl mvn zip

WORK_DIR="$(mktemp -d)"
DEPLOY_DIR="${WORK_DIR}/deploy"
DOWNLOAD_DIR="${WORK_DIR}/downloaded"
SETTINGS_FILE="${WORK_DIR}/settings.xml"
DEPLOY_LOG="${WORK_DIR}/deploy.log"
mkdir -p "${DEPLOY_DIR}" "${DOWNLOAD_DIR}"

stage_snapshot_deploy_inputs "${INPUT_DIR}" "${DEPLOY_DIR}"

GROUP_PATH="$(maven_group_path "${GROUP_ID}")"
ARTIFACT_DIR="${DEPLOY_DIR}/${GROUP_PATH}/${ARTIFACT_ID}/${VERSION}"
if [[ ! -d ${ARTIFACT_DIR} ]]; then
  fatal "expected artifact dir ${ARTIFACT_DIR} not present in input"
fi

write_maven_settings "${SETTINGS_FILE}"

deploy_snapshot_with_mvn \
  "${ARTIFACT_DIR}" "${ARTIFACT_ID}" "${VERSION}" \
  "${SNAPSHOT_REPOSITORY_URL}" "${SETTINGS_FILE}" "${DEPLOY_LOG}"

# Scope at the artifactId directory so the parent maven-metadata.xml is
# fetched alongside everything under the SNAPSHOT version directory.
download_deployed_tree \
  "${DEPLOY_LOG}" "${SNAPSHOT_REPOSITORY_URL}" \
  "${GROUP_PATH}/${ARTIFACT_ID}" "${DOWNLOAD_DIR}"

BUNDLE_ZIP="${BUNDLE_DIR}/${ARTIFACT_ID}-${VERSION}.zip"
(cd "${DOWNLOAD_DIR}" && zip -qr "${BUNDLE_ZIP}" .)

echo "Snapshot deploy complete: ${SNAPSHOT_REPOSITORY_URL}/${GROUP_PATH}/${ARTIFACT_ID}/${VERSION}/"
