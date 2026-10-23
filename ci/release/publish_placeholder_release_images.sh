#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

## Usage
# ./ci/release/publish_placeholder_release_images.sh
#
# Publishes PLACEHOLDER release-tagged per-arch images AND multiarch
# manifests to NGC nvstaging only (nvcr.io/nvstaging/nvaie/cuopt), copied
# from the most recently pushed NIGHTLY images. This does NOT rebuild
# anything from source — it copies existing nightly image content
# (remotely, via `docker buildx imagetools create`, no local pull/push of
# image layers needed) onto real release-tagged image refs, so that a
# release tag resolves to actual images (not just a manifest aliasing
# nightly tags) before the real, vetted release images are built and
# published under the same tag. It never touches Docker Hub.
#
# The release tag is derived from the repo's VERSION file (same scheme
# ci/docker/create_multiarch_manifest.sh uses for real releases: dotted
# <major>.<minor>.<patch> with leading zeros in each segment stripped).
# The source nightly tag is the same version with a trailing 'a' (the
# nightly IMAGE_TAG_PREFIX convention from build_test_publish_images.yaml).
#
# By default this script does NOT touch "latest" - "latest" is a floating
# tag downstream consumers treat as the current vetted release, and these
# are placeholder images copied from nightly. Set PUSH_LATEST=true to also
# point "latest" at this placeholder content; only do this deliberately.
#
# Requires: docker CLI (with buildx) logged in to nvcr.io with push access
# to nvstaging/nvaie/cuopt (docker login nvcr.io).
#
# Env overrides:
#   CUDA_VERS    space-separated list of full CUDA versions (default: "12.9.0 13.3.0")
#   RELEASE_TAG  override the derived release tag (default: derived from VERSION)
#   NIGHTLY_TAG  override the derived source nightly tag (default: "${RELEASE_TAG}a")
#   PUSH_LATEST  set to "true" to also push "latest" manifests (default: "false")
#   DRY_RUN      set to "true" to only check image existence, skip all pushes

set -euo pipefail

if [[ ! -f "VERSION" ]] || [[ ! -f "ci/release/publish_placeholder_release_images.sh" ]]; then
    echo "Error: This script must be run from the root of the cuopt repository" >&2
    exit 1
fi

NGC_REPO="nvcr.io/nvstaging/nvaie/cuopt"

# rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
# Strip leading zeros from each dotted segment, e.g. 26.10.00 -> 26.10.0
BASE_VER=$(sed -E 's/\.0+([0-9])/\.\1/g' VERSION | tr -d '[:space:]')

RELEASE_TAG="${RELEASE_TAG:-${BASE_VER}}"
NIGHTLY_TAG="${NIGHTLY_TAG:-${RELEASE_TAG}a}"
CUDA_VERS="${CUDA_VERS:-12.9.0 13.3.0}"
DRY_RUN="${DRY_RUN:-false}"
PUSH_LATEST="${PUSH_LATEST:-false}"

echo "=== Placeholder release manifest publish ==="
echo "NGC repo:    ${NGC_REPO}"
echo "Source tag:  ${NIGHTLY_TAG} (nightly)"
echo "Target tag:  ${RELEASE_TAG} (placeholder release)"
echo "CUDA_VERS:   ${CUDA_VERS}"
echo "PUSH_LATEST: ${PUSH_LATEST}"
echo "DRY_RUN:     ${DRY_RUN}"
if [[ "${PUSH_LATEST}" == "true" ]]; then
    echo "WARNING: PUSH_LATEST=true - 'latest' in nvstaging will point at placeholder (nightly-derived) content."
fi
echo "=============================================="

check_image_exists() {
    local image=$1
    if docker manifest inspect "$image" >/dev/null 2>&1; then
        echo "  found: $image"
        return 0
    fi
    echo "  MISSING: $image"
    return 1
}

# Copies a nightly per-arch image onto a real release-tagged image ref,
# entirely registry-side (no local pull of image layers).
copy_release_image() {
    local src_image=$1
    local dest_image=$2

    check_image_exists "$src_image" || return 1

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [dry-run] would copy ${src_image} -> ${dest_image}"
        return 0
    fi

    docker buildx imagetools create --tag "$dest_image" "$src_image"
    echo "  copied: ${dest_image}"
}

# Builds a multiarch manifest list from release-tagged per-arch images
# (which must already exist, e.g. via copy_release_image above).
create_release_manifest() {
    local manifest_name=$1
    local amd64_image=$2
    local arm64_image=$3

    echo "--- Manifest: ${manifest_name} ---"

    # In a dry run the per-arch release images may not exist yet
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [dry-run] would create + push ${manifest_name}"
        return 0
    fi

    check_image_exists "$amd64_image" || return 1
    check_image_exists "$arm64_image" || return 1

    # imagetools create (not `docker manifest create`) because the per-arch
    # refs above are themselves OCI indexes (imagetools always wraps single
    # copies in one) - the legacy manifest tool rejects list-of-list input.
    docker buildx imagetools create --tag "$manifest_name" "$amd64_image" "$arm64_image"

    echo "  pushed: ${manifest_name}"
}

# Copies nightly per-arch images to release-tagged refs, then builds the
# release manifest list from those release-tagged refs (mirrors the real
# release flow: build_images.yaml pushes per-arch images first, then
# create_multiarch_manifest.sh builds the list from them).
publish_release_variant() {
    local manifest_tag=$1   # e.g. "${RELEASE_TAG}-cu12"
    local nightly_tag=$2    # e.g. "${NIGHTLY_TAG}-cu12"

    local release_amd64="${NGC_REPO}:${manifest_tag}-amd64"
    local release_arm64="${NGC_REPO}:${manifest_tag}-arm64"

    copy_release_image "${NGC_REPO}:${nightly_tag}-amd64" "$release_amd64" || return 1
    copy_release_image "${NGC_REPO}:${nightly_tag}-arm64" "$release_arm64" || return 1

    create_release_manifest "${NGC_REPO}:${manifest_tag}" "$release_amd64" "$release_arm64"
}

publish_variant_and_latest() {
    local alias=$1          # e.g. "cu13" or "cu13-ubi10"

    publish_release_variant "${RELEASE_TAG}-${alias}" "${NIGHTLY_TAG}-${alias}"

    if [[ "${PUSH_LATEST}" == "true" ]]; then
        create_release_manifest \
            "${NGC_REPO}:latest-${alias}" \
            "${NGC_REPO}:${RELEASE_TAG}-${alias}-amd64" \
            "${NGC_REPO}:${RELEASE_TAG}-${alias}-arm64"
    fi
}

for CUDA_VER in ${CUDA_VERS}; do
    CUDA_MAJOR="${CUDA_VER%%.*}"

    publish_variant_and_latest "cu${CUDA_MAJOR}"

    # UBI10 base images are only published for CUDA 13.x and later
    if [[ "${CUDA_MAJOR}" == "13" ]]; then
        publish_variant_and_latest "cu${CUDA_MAJOR}-ubi10"
    fi
done

if [[ "${PUSH_LATEST}" == "true" ]]; then
    echo "=== Done. 'latest' was pushed (PUSH_LATEST=true) - points at placeholder content. ==="
else
    echo "=== Done. Reminder: 'latest' was NOT touched - only real release builds should move it. ==="
fi
