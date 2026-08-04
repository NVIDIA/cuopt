#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Promote nightly per-arch images to release tags on NGC staging and create
# multi-arch manifests.  No rebuild is performed — images are pulled, retagged,
# pushed, and stitched into manifests.
#
# Required environment variables:
#   NIGHTLY_TAG_PREFIX  — nightly IMAGE_TAG_PREFIX, e.g. 26.8.0a
#   RELEASE_VERSION     — release version to tag as,   e.g. 26.8.0
#   CUDA_SHORT          — trimmed CUDA version,         e.g. 12.9
#   PYTHON_SHORT        — trimmed Python version,       e.g. 3.14

set -euo pipefail

REGISTRY="nvcr.io/nvstaging/nvaie/cuopt"
CUDA_MAJOR="${CUDA_SHORT%%.*}"

promote_image() {
    local src=$1
    local dst=$2
    echo "Promoting: $src -> $dst"
    docker pull "$src"
    docker tag  "$src" "$dst"
    docker push "$dst"
    echo "✓ Promoted: $dst"
}

create_manifest() {
    local manifest_name=$1
    local amd64_image=$2
    local arm64_image=$3

    echo "Creating manifest: $manifest_name"
    docker manifest create --amend "$manifest_name" "$amd64_image" "$arm64_image"
    docker manifest annotate "$manifest_name" "$arm64_image" --arch arm64
    docker manifest annotate "$manifest_name" "$amd64_image" --arch amd64
    docker manifest push "$manifest_name"
    echo "✓ Pushed manifest: $manifest_name"
}

# ── cuda+py per-arch images ────────────────────────────────────────────────────

NIGHTLY_AMD64="${REGISTRY}:${NIGHTLY_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-amd64"
NIGHTLY_ARM64="${REGISTRY}:${NIGHTLY_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-arm64"
RELEASE_AMD64="${REGISTRY}:${RELEASE_VERSION}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-amd64"
RELEASE_ARM64="${REGISTRY}:${RELEASE_VERSION}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-arm64"

echo "=== Promoting cuda+py per-arch images ==="
promote_image "$NIGHTLY_AMD64" "$RELEASE_AMD64"
promote_image "$NIGHTLY_ARM64" "$RELEASE_ARM64"

# ── cuda+py and cu<major> manifests ───────────────────────────────────────────

echo "=== Creating release manifests ==="
create_manifest \
    "${REGISTRY}:${RELEASE_VERSION}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}" \
    "$RELEASE_AMD64" "$RELEASE_ARM64"

create_manifest \
    "${REGISTRY}:${RELEASE_VERSION}-cu${CUDA_MAJOR}" \
    "$RELEASE_AMD64" "$RELEASE_ARM64"

echo "=== Creating latest manifests ==="
create_manifest \
    "${REGISTRY}:latest-cuda${CUDA_SHORT}-py${PYTHON_SHORT}" \
    "$RELEASE_AMD64" "$RELEASE_ARM64"

create_manifest \
    "${REGISTRY}:latest-cu${CUDA_MAJOR}" \
    "$RELEASE_AMD64" "$RELEASE_ARM64"

# ── UBI10 (CUDA 13+ only) ─────────────────────────────────────────────────────

if [[ "${CUDA_MAJOR}" == "13" ]]; then
    NIGHTLY_UBI10_AMD64="${REGISTRY}:${NIGHTLY_TAG_PREFIX}-cuda${CUDA_SHORT}-ubi10-amd64"
    NIGHTLY_UBI10_ARM64="${REGISTRY}:${NIGHTLY_TAG_PREFIX}-cuda${CUDA_SHORT}-ubi10-arm64"
    RELEASE_UBI10_AMD64="${REGISTRY}:${RELEASE_VERSION}-cuda${CUDA_SHORT}-ubi10-amd64"
    RELEASE_UBI10_ARM64="${REGISTRY}:${RELEASE_VERSION}-cuda${CUDA_SHORT}-ubi10-arm64"

    echo "=== Promoting UBI10 per-arch images ==="
    promote_image "$NIGHTLY_UBI10_AMD64" "$RELEASE_UBI10_AMD64"
    promote_image "$NIGHTLY_UBI10_ARM64" "$RELEASE_UBI10_ARM64"

    echo "=== Creating UBI10 release manifests ==="
    create_manifest \
        "${REGISTRY}:${RELEASE_VERSION}-cuda${CUDA_SHORT}-ubi10" \
        "$RELEASE_UBI10_AMD64" "$RELEASE_UBI10_ARM64"

    create_manifest \
        "${REGISTRY}:${RELEASE_VERSION}-cu${CUDA_MAJOR}-ubi10" \
        "$RELEASE_UBI10_AMD64" "$RELEASE_UBI10_ARM64"

    create_manifest \
        "${REGISTRY}:latest-cuda${CUDA_SHORT}-ubi10" \
        "$RELEASE_UBI10_AMD64" "$RELEASE_UBI10_ARM64"

    create_manifest \
        "${REGISTRY}:latest-cu${CUDA_MAJOR}-ubi10" \
        "$RELEASE_UBI10_AMD64" "$RELEASE_UBI10_ARM64"
else
    echo "Skipping UBI10 (CUDA_MAJOR='${CUDA_MAJOR}' — UBI10 requires CUDA 13+)"
fi

echo "=== Promotion to release complete ==="
