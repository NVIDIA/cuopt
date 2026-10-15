#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Step functions used by maven_snapshot_publish_in_container.sh.

# stage_snapshot_deploy_inputs INPUT_DIR DEPLOY_DIR
# Copy INPUT_DIR into DEPLOY_DIR, stripping any sidecars that mvn regenerates.
stage_snapshot_deploy_inputs() {
  local input_dir=$1 deploy_dir=$2
  cp -a "${input_dir}/." "${deploy_dir}/"
  find "${deploy_dir}" -type f \( \
    -name '*.asc' -o -name '*.md5' -o \
    -name '*.sha1' -o -name '*.sha256' -o -name '*.sha512' -o \
    -name 'maven-metadata.xml*' \
  \) -delete
}

# write_maven_settings OUT_PATH
# Wire the ossrh server-id to MAVEN_DEPLOY_USERNAME / MAVEN_DEPLOY_TOKEN.
write_maven_settings() {
  local out=$1
  cat > "${out}" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0">
  <servers>
    <server>
      <id>ossrh</id>
      <username>${env.MAVEN_DEPLOY_USERNAME}</username>
      <password>${env.MAVEN_DEPLOY_TOKEN}</password>
    </server>
  </servers>
</settings>
EOF
}

# deploy_snapshot_with_mvn ARTIFACT_DIR ARTIFACT_ID VERSION \
#                          REPO_URL SETTINGS_FILE LOG_PATH
# Deploys every jar under ARTIFACT_DIR via mvn deploy-file.
# The main jar is <artifact_id>-<version>.jar. sources.jar and javadoc.jar
# use mvn's -Dsources / -Djavadoc. Any other jar is attached with a
# classifier taken from its filename. mvn output goes to LOG_PATH so the
# caller can enumerate Nexus-assigned URLs.
deploy_snapshot_with_mvn() {
  local artifact_dir=$1 artifact_id=$2 version=$3
  local repo_url=$4 settings_file=$5 log_path=$6

  local main_jar="${artifact_dir}/${artifact_id}-${version}.jar"
  local pom_file="${artifact_dir}/${artifact_id}-${version}.pom"
  local sources_jar="${artifact_dir}/${artifact_id}-${version}-sources.jar"
  local javadoc_jar="${artifact_dir}/${artifact_id}-${version}-javadoc.jar"

  if [[ ! -f ${main_jar} ]]; then
    fatal "missing primary jar: ${main_jar}"
  fi
  if [[ ! -f ${pom_file} ]]; then
    fatal "missing POM: ${pom_file}"
  fi

  # Collect classifier jars: every jar under artifact_dir except main /
  # sources / javadoc, which mvn takes separately below.
  local -a files=() types=() classifiers=()
  local jar classifier
  for jar in "${artifact_dir}"/*.jar; do
    case ${jar} in
      "${main_jar}"|"${sources_jar}"|"${javadoc_jar}") continue ;;
    esac
    classifier=$(basename "${jar}" .jar)
    classifier=${classifier#"${artifact_id}-${version}-"}
    files+=("${jar}")
    types+=(jar)
    classifiers+=("${classifier}")
  done

  local -a mvn_args=(
    -B
    -Dmaven.wagon.http.retryHandler.count=3
    -DretryFailedDeploymentCount=3
    -s "${settings_file}"
    org.apache.maven.plugins:maven-deploy-plugin:3.1.4:deploy-file
    -Durl="${repo_url}"
    -DrepositoryId=ossrh
    -Dfile="${main_jar}"
    -DpomFile="${pom_file}"
  )
  if [[ -f ${sources_jar} ]]; then
    mvn_args+=(-Dsources="${sources_jar}")
  fi
  if [[ -f ${javadoc_jar} ]]; then
    mvn_args+=(-Djavadoc="${javadoc_jar}")
  fi
  if (( ${#files[@]} > 0 )); then
    mvn_args+=(
      -Dfiles="$(IFS=,; echo "${files[*]}")"
      -Dtypes="$(IFS=,; echo "${types[*]}")"
      -Dclassifiers="$(IFS=,; echo "${classifiers[*]}")"
    )
  fi

  # tee so parse_uploaded_urls can enumerate Nexus URLs from LOG_PATH.
  set +e
  mvn "${mvn_args[@]}" 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e
  if (( rc != 0 )); then
    fatal "mvn deploy-file failed (exit ${rc})"
  fi
}

# parse_uploaded_urls LOG_PATH REPO_URL ARTIFACT_BASE_PATH
# Print every URL Nexus acknowledged via 'Uploaded to ossrh: <URL>' whose path
# starts with <REPO_URL>/<ARTIFACT_BASE_PATH>/. Deduplicated.
parse_uploaded_urls() {
  local log_path=$1 repo_url=$2 artifact_base_path=$3
  local prefix="${repo_url}/${artifact_base_path}/"
  awk -v p="${prefix}" '
    /Uploaded to ossrh:/ {
      for (i = 1; i <= NF; i++) if (index($i, p) == 1) print $i
    }
  ' "${log_path}" | sort -u
}

# download_deployed_tree LOG_PATH REPO_URL ARTIFACT_BASE_PATH BUNDLE_DIR
download_deployed_tree() {
  local log_path=$1 repo_url=$2 artifact_base_path=$3 bundle_dir=$4
  local -a urls
  mapfile -t urls < <(parse_uploaded_urls "${log_path}" "${repo_url}" "${artifact_base_path}")
  if (( ${#urls[@]} == 0 )); then
    fatal "mvn deploy log has no 'Uploaded to ossrh:' URLs under ${repo_url}/${artifact_base_path}/"
  fi

  echo "Downloading deployed tree (${#urls[@]} file(s))"
  local url rel dest
  for url in "${urls[@]}"; do
    rel="${url#"${repo_url}/"}"
    dest="${bundle_dir}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    _download_with_retry "${url}" "${dest}"
  done
}

# _download_with_retry URL DEST -- covers the brief PUT->GET propagation lag.
_download_with_retry() {
  local url=$1 dest=$2
  local attempt status=""
  for attempt in 1 2 3 4 5; do
    status=$(curl -sSL -o "${dest}" -w '%{http_code}' "${url}") \
      || status="transport-error"
    if [[ ${status} == 2* ]]; then
      return 0
    fi
    echo "  [attempt ${attempt}] ${url} -> ${status}, retrying" >&2
    sleep $((attempt * 2))
  done
  fatal "failed to download ${url} after 5 attempts (last status: ${status})"
}
