#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Step functions used by prepare_maven_bundle_in_container.sh.

install_container_deps() {
  if command -v gpg >/dev/null && command -v zip >/dev/null; then
    return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -qq -y --no-install-recommends gnupg zip
}

remove_generated_sidecars() {
  local root=$1
  find "${root}" -type f \( \
    -name '*.asc' -o -name '*.md5' -o -name '*.sha1' -o \
    -name '*.sha256' -o -name '*.sha512' \
  \) -delete
}

discover_single_pom() {
  local input_dir=$1
  local -n out_pom_var=$2
  local -a candidates
  mapfile -t candidates < <(find "${input_dir}" -type f -name '*.pom' | sort)
  # The single-candidate assignment is observed through the caller's nameref.
  # shellcheck disable=SC2034
  case ${#candidates[@]} in
    0) fatal "no *.pom found anywhere under ${input_dir}" ;;
    1) out_pom_var=${candidates[0]} ;;
    *)
      {
        echo "Error: expected exactly one POM under ${input_dir}, found:"
        printf '  %s\n' "${candidates[@]}"
      } >&2
      exit 1
      ;;
  esac
}

read_pom_coordinates() {
  local pom_file=$1
  local -n out_group_var=$2
  local -n out_artifact_var=$3
  local -n out_version_var=$4
  out_group_var=$(mvn -q -B -f "${pom_file}" help:evaluate -Dexpression=project.groupId -DforceStdout)
  out_artifact_var=$(mvn -q -B -f "${pom_file}" help:evaluate -Dexpression=project.artifactId -DforceStdout)
  out_version_var=$(mvn -q -B -f "${pom_file}" help:evaluate -Dexpression=project.version -DforceStdout)
  if [[ -z ${out_group_var} || -z ${out_artifact_var} || -z ${out_version_var} ]]; then
    fatal "failed to read groupId/artifactId/version from ${pom_file}"
  fi
}

import_gpg_signing_key() {
  local -n out_key_id=$1
  export GNUPGHOME
  GNUPGHOME="$(mktemp -d)"
  chmod 700 "${GNUPGHOME}"
  echo "${GPG_PRIVATE_KEY}" | gpg --batch --yes --pinentry-mode loopback \
    --passphrase "${GPG_PASSPHRASE}" --import
  out_key_id=$(gpg --list-secret-keys --keyid-format=long --with-colons \
    | awk -F: '$1=="sec" { print $5; exit }')
  if [[ -z ${out_key_id} ]]; then
    fatal "no GPG secret key was imported"
  fi
}

list_artifacts_to_sign() {
  local artifact_dir=$1
  local -n out_files=$2
  mapfile -t out_files < <(find "${artifact_dir}" -maxdepth 1 -type f \
    ! -name '*.asc' ! -name '*.md5' ! -name '*.sha1' ! -name '*.sha256' \
    ! -name '*.sha512' | sort)
  if (( ${#out_files[@]} == 0 )); then
    fatal "no artifacts found under ${artifact_dir} to sign"
  fi
}

sign_artifacts_in_place() {
  local -n files_ref=$1
  local key_id=$2
  local file
  for file in "${files_ref[@]}"; do
    gpg --batch --yes --pinentry-mode loopback \
      --passphrase "${GPG_PASSPHRASE}" \
      --local-user "${key_id}" \
      --armor --detach-sign \
      --output "${file}.asc" \
      "${file}"
  done
}

write_bundle_metadata() {
  local out_path=$1 group_id=$2 artifact_id=$3 version=$4
  {
    echo "GROUP_ID=${group_id}"
    echo "ARTIFACT_ID=${artifact_id}"
    echo "VERSION=${version}"
  } > "${out_path}"
}
