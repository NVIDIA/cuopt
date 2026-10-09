#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container worker for prepare_maven_bundle.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/maven_utils.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/prepare_maven_bundle_steps.sh"

INPUT_DIR=/input
BUNDLE_DIR=/bundle
METADATA_DIR=/metadata

: "${GPG_PRIVATE_KEY:?must be set}"
: "${GPG_PASSPHRASE:?must be set}"
: "${HOST_UID:?must be set}"
: "${HOST_GID:?must be set}"

trap 'chown -R "${HOST_UID}:${HOST_GID}" "${BUNDLE_DIR}" "${METADATA_DIR}" 2>/dev/null || true' EXIT

install_container_deps
cp -a "${INPUT_DIR}/." "${BUNDLE_DIR}/"
remove_generated_sidecars "${BUNDLE_DIR}"

POM_FILE=""
discover_single_pom "${BUNDLE_DIR}" POM_FILE
ARTIFACT_DIR="$(dirname "${POM_FILE}")"

GROUP_ID=""
ARTIFACT_ID=""
VERSION=""
read_pom_coordinates "${POM_FILE}" GROUP_ID ARTIFACT_ID VERSION
require_maven_coordinates "${GROUP_ID}" "${ARTIFACT_ID}" "${VERSION}"

EXPECTED_ARTIFACT_DIR="${BUNDLE_DIR}/$(maven_group_path "${GROUP_ID}")/${ARTIFACT_ID}/${VERSION}"
if [[ ${ARTIFACT_DIR} != "${EXPECTED_ARTIFACT_DIR}" ]]; then
  fatal "POM location ${ARTIFACT_DIR} does not match expected Maven repository layout ${EXPECTED_ARTIFACT_DIR}"
fi

GPG_KEY_ID=""
import_gpg_signing_key GPG_KEY_ID
echo "GPG signing key: ${GPG_KEY_ID}"

# Populated through a nameref in list_artifacts_to_sign.
# shellcheck disable=SC2034
ARTIFACTS=()
echo "Signing artifacts under ${ARTIFACT_DIR}"
list_artifacts_to_sign "${ARTIFACT_DIR}" ARTIFACTS
sign_artifacts_in_place ARTIFACTS "${GPG_KEY_ID}"

write_bundle_metadata "${METADATA_DIR}/bundle_metadata.env" \
  "${GROUP_ID}" "${ARTIFACT_ID}" "${VERSION}"

echo "Signing complete for ${GROUP_ID}:${ARTIFACT_ID}:${VERSION}"
