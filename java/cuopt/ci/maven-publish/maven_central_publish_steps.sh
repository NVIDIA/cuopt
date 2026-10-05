#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sonatype HTTP wrappers and local bundle helpers for maven_central_publish.sh.

_portal_auth() {
  printf '%s:%s' "${MAVEN_DEPLOY_USERNAME}" "${MAVEN_DEPLOY_TOKEN}" \
    | base64 | tr -d '\n'
}

# _sonatype_error_log METHOD URL STATUS BODY_FILE
_sonatype_error_log() {
  echo "Sonatype request failed"
  echo "  method: $1"
  echo "  URL:    $2"
  echo "  status: $3"
  echo "  body:   $(cat "$4")"
}

# _sonatype_call METHOD URL
# Bearer auth, no request body. Prints response body on 2xx; logs failure and
# returns 1 on non-2xx.
_sonatype_call() {
  local method=$1 url=$2
  local body_file status
  body_file=$(mktemp)
  status=$(curl -sS --retry 3 --retry-delay 2 \
    -H "Authorization: Bearer $(_portal_auth)" \
    -X "${method}" -o "${body_file}" -w '%{http_code}' "${url}") \
    || status="transport-error"
  if [[ ${status} != 2* ]]; then
    _sonatype_error_log "${method}" "${url}" "${status}" "${body_file}" >&2
    rm -f "${body_file}"
    return 1
  fi
  cat "${body_file}"
  rm -f "${body_file}"
}

# _sonatype_upload FILE URL
# Basic auth, PUT FILE. Logs failure and returns 1 on non-2xx.
_sonatype_upload() {
  local upload_file=$1 url=$2
  local body_file status
  body_file=$(mktemp)
  status=$(curl -sS --retry 3 --retry-delay 2 \
    --user "${MAVEN_DEPLOY_USERNAME}:${MAVEN_DEPLOY_TOKEN}" \
    --upload-file "${upload_file}" \
    -o "${body_file}" -w '%{http_code}' "${url}") \
    || status="transport-error"
  if [[ ${status} != 2* ]]; then
    _sonatype_error_log PUT "${url}" "${status}" "${body_file}" >&2
    rm -f "${body_file}"
    return 1
  fi
  rm -f "${body_file}"
}

staging_search_repositories() {
  local state=${1:-}
  local url
  url="${OSSRH_STAGING_API_URL}/manual/search/repositories?ip=client&profile_id=$(printf %s "${PROFILE_ID}" | jq -sRr @uri)"
  if [[ -n ${state} ]]; then
    url+="&state=$(printf %s "${state}" | jq -sRr @uri)"
  fi
  _sonatype_call GET "${url}"
}

staging_upload_file() {
  local file_path=$1 repository_path=$2
  local encoded_path
  encoded_path=$(jq -rn --arg path "${repository_path}" \
    '$path | split("/") | map(@uri) | join("/")')
  _sonatype_upload "${file_path}" \
    "${OSSRH_STAGING_API_URL}/service/local/staging/deploy/maven2/${encoded_path}"
}

staging_handoff_repository() {
  local repository_key=$1
  _sonatype_call POST \
    "${OSSRH_STAGING_API_URL}/manual/upload/repository/${repository_key}?publishing_type=portal_api" \
    >/dev/null
}

staging_drop_repository() {
  local repository_key=$1
  _sonatype_call DELETE \
    "${OSSRH_STAGING_API_URL}/manual/drop/repository/${repository_key}" \
    >/dev/null
}

portal_get_status() {
  local deployment_id=$1
  _sonatype_call POST \
    "${CENTRAL_PORTAL_URL}/api/v1/publisher/status?id=${deployment_id}" \
    || echo '{}'
}

portal_drop_deployment() {
  local deployment_id=$1
  _sonatype_call DELETE \
    "${CENTRAL_PORTAL_URL}/api/v1/publisher/deployment/${deployment_id}" \
    >/dev/null
}

copy_bundle() {
  local source_dir=$1 destination_dir=$2
  echo "Copying signed Maven tree into bundle scratch space"
  cp -a "${source_dir}/." "${destination_dir}/"
}

require_bundle_contents() {
  local artifact_dir=$1
  local pattern desc entry
  local expected=(
    "${ARTIFACT_ID}-${VERSION}.pom|the POM"
    "${ARTIFACT_ID}-${VERSION}.pom.asc|the POM signature"
    "${ARTIFACT_ID}-${VERSION}.jar|the unclassified primary jar"
    "${ARTIFACT_ID}-${VERSION}.jar.asc|the unclassified primary jar signature"
    "${ARTIFACT_ID}-${VERSION}-sources.jar|the sources jar"
    "${ARTIFACT_ID}-${VERSION}-sources.jar.asc|the sources jar signature"
    "${ARTIFACT_ID}-${VERSION}-javadoc.jar|the javadoc jar"
    "${ARTIFACT_ID}-${VERSION}-javadoc.jar.asc|the javadoc jar signature"
  )
  for entry in "${expected[@]}"; do
    IFS='|' read -r pattern desc <<< "${entry}"
    if [[ ! -f ${artifact_dir}/${pattern} ]]; then
      fatal "bundle is missing ${desc} (${pattern})"
    fi
  done
}

generate_bundle_checksums() {
  local root=$1 file
  find "${root}" -type f \( \
    -name '*.md5' -o -name '*.sha1' -o -name '*.sha256' -o -name '*.sha512' \
  \) -delete
  while IFS= read -r file; do
    md5sum "${file}" | awk '{ print $1 }' > "${file}.md5"
    sha1sum "${file}" | awk '{ print $1 }' > "${file}.sha1"
  done < <(find "${root}" -type f ! -name '*.md5' ! -name '*.sha1' \
    ! -name '*.sha256' ! -name '*.sha512' | sort)
}

create_bundle_zip() {
  local zip_path=$1 bundle_dir=$2
  echo "Zipping bundle at ${zip_path}"
  (cd "${bundle_dir}" && zip -qr "${zip_path}" .)
}

require_no_open_staging_repository() {
  local response count
  response=$(staging_search_repositories open)
  count=$(jq -er '.repositories | length' <<< "${response}") \
    || fatal "OSSRH staging search returned an invalid response"
  if (( count != 0 )); then
    jq . <<< "${response}" >&2 || true
    fatal "found ${count} open OSSRH staging repository/repositories for ${PROFILE_ID} from this IP; drop or hand them off before retrying"
  fi
}

upload_tree_to_staging() {
  local root=$1 file repository_path count=0
  echo "Uploading Maven repository files to the OSSRH staging service"
  while IFS= read -r -d '' file; do
    repository_path=${file#"${root}/"}
    staging_upload_file "${file}" "${repository_path}"
    # Consumed by the EXIT cleanup trap in maven_central_publish.sh.
    # shellcheck disable=SC2034
    STAGING_FILE_UPLOADED="true"
    count=$((count + 1))
  done < <(find "${root}" -type f -print0)
  echo "  uploaded files: ${count}"
}

find_open_staging_repository() {
  local response count
  response=$(staging_search_repositories open)
  count=$(jq -er '.repositories | length' <<< "${response}") \
    || fatal "OSSRH staging search returned an invalid response"
  if (( count != 1 )); then
    jq . <<< "${response}" >&2 || true
    fatal "expected exactly one open OSSRH staging repository after upload, found ${count}"
  fi
  STAGING_REPOSITORY_KEY=$(jq -er '.repositories[0].key | select(type == "string" and length > 0)' \
    <<< "${response}") || fatal "OSSRH staging response did not contain a repository key"
  # Keys are composite paths of the form <profileId>/<clientIp>/<groupId>--default-repository,
  # so slashes are expected. Just guard against characters that would break URL path use.
  if [[ ! ${STAGING_REPOSITORY_KEY} =~ ^[A-Za-z0-9._/-]+$ ]]; then
    fatal "OSSRH staging service returned an invalid repository key: ${STAGING_REPOSITORY_KEY}"
  fi
  echo "  staging repository: ${STAGING_REPOSITORY_KEY}"
}

wait_for_portal_handoff() {
  local repository_key=$1
  local response state elapsed=0
  echo "Polling OSSRH staging handoff (interval=${POLL_INTERVAL_SEC}s, timeout=${POLL_TIMEOUT_SEC}s)"
  while (( elapsed < POLL_TIMEOUT_SEC )); do
    response=$(staging_search_repositories)
    state=$(jq -r --arg key "${repository_key}" \
      '.repositories[]? | select(.key == $key) | .state // "UNKNOWN"' \
      <<< "${response}")
    DEPLOYMENT_ID=$(jq -r --arg key "${repository_key}" \
      '.repositories[]? | select(.key == $key) | .portal_deployment_id // empty' \
      <<< "${response}")
    echo "  [${elapsed}s] staging state=${state:-UNKNOWN}"
    if [[ -n ${DEPLOYMENT_ID} ]]; then
      if [[ ! ${DEPLOYMENT_ID} =~ ^[A-Za-z0-9-]+$ ]]; then
        fatal "OSSRH staging service returned an invalid Portal deployment id"
      fi
      echo "  Portal deployment id: ${DEPLOYMENT_ID}"
      return 0
    fi
    sleep "${POLL_INTERVAL_SEC}"
    elapsed=$((elapsed + POLL_INTERVAL_SEC))
  done
  fatal "timed out waiting for OSSRH staging repository '${repository_key}' to reach the Portal"
}

wait_for_validated() {
  local deployment_id=$1
  local response state elapsed=0
  echo "Polling status (interval=${POLL_INTERVAL_SEC}s, timeout=${POLL_TIMEOUT_SEC}s)"
  while (( elapsed < POLL_TIMEOUT_SEC )); do
    response=$(portal_get_status "${deployment_id}")
    state=$(jq -r '.deploymentState // "UNKNOWN"' <<< "${response}")
    echo "  [${elapsed}s] state=${state}"
    case "${state}" in
      VALIDATED) return 0 ;;
      FAILED|VALIDATION_FAILED)
        jq . <<< "${response}" >&2 || true
        fatal "Publisher Portal reported terminal state ${state}"
        ;;
      PUBLISHED)
        fatal "unexpected PUBLISHED state (USER_MANAGED deployments should never auto-publish)"
        ;;
    esac
    sleep "${POLL_INTERVAL_SEC}"
    elapsed=$((elapsed + POLL_INTERVAL_SEC))
  done
  fatal "timed out waiting for VALIDATED, last state was '${state}'"
}

drop_deployment() {
  local deployment_id=$1
  echo "Dropping deployment ${deployment_id} (--auto-drop true)"
  portal_drop_deployment "${deployment_id}"
  echo "Deployment dropped. Nothing was published to Maven Central."
}
