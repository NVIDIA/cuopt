#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Function to check if a Docker image exists in the registry
check_image_exists() {
    local image=$1
    echo "Checking if image exists: $image"

    # Try to pull the image manifest to check if it exists
    if docker manifest inspect "$image" >/dev/null 2>&1; then
        echo "✓ Image exists: $image"
        return 0
    else
        echo "✗ Image does not exist: $image"
        return 1
    fi
}

# Function to create manifest with error checking
create_manifest() {
    local manifest_name=$1
    local amd64_image=$2
    local arm64_image=$3

    echo "Creating manifest: $manifest_name"

    # Check if both architecture images exist
    if ! check_image_exists "$amd64_image"; then
        echo "Error: AMD64 image not found: $amd64_image"
        return 1
    fi

    if ! check_image_exists "$arm64_image"; then
        echo "Error: ARM64 image not found: $arm64_image"
        return 1
    fi

    echo "Creating multi-arch manifest..."
    docker manifest rm "$manifest_name" >/dev/null 2>&1 || true
    docker manifest create "$manifest_name" "$amd64_image" "$arm64_image"

    # Annotate with architecture information
    echo "Annotating ARM64 architecture..."
    docker manifest annotate "$manifest_name" "$arm64_image" --arch arm64

    echo "Annotating AMD64 architecture..."
    docker manifest annotate "$manifest_name" "$amd64_image" --arch amd64

    # Push the manifest
    echo "Pushing manifest: $manifest_name"
    docker manifest push "$manifest_name"

    echo "✓ Successfully created and pushed manifest: $manifest_name"
}

CUDA_MAJOR="${CUDA_VER%%.*}"

IS_RELEASE=false
if [[ "${IMAGE_TAG_PREFIX}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then IS_RELEASE=true; fi

create_short_manifests() {
    local short=$1 repo
    for repo in nvidia/cuopt nvcr.io/nvstaging/nvaie/cuopt; do
        echo "=== Creating ${short} manifests in ${repo} ==="
        create_manifest \
            "${repo}:${IMAGE_TAG_PREFIX}-${short}" \
            "${repo}:${IMAGE_TAG_PREFIX}-${short}-amd64" \
            "${repo}:${IMAGE_TAG_PREFIX}-${short}-arm64"
        if [[ "${IS_RELEASE}" == "true" ]]; then
            create_manifest \
                "${repo}:latest-${short}" \
                "${repo}:${IMAGE_TAG_PREFIX}-${short}-amd64" \
                "${repo}:${IMAGE_TAG_PREFIX}-${short}-arm64"
        fi
    done
}

create_short_manifests "cu${CUDA_MAJOR}"

# UBI10 CUDA base images are only published for CUDA 13.x and later
if [[ "${CUDA_MAJOR}" == "13" ]]; then
    create_short_manifests "cu${CUDA_MAJOR}-ubi10"
else
    echo "Skipping UBI10 manifests (CUDA_MAJOR='${CUDA_MAJOR}' — UBI10 base images require CUDA 13+)"
fi

echo "=== Multi-architecture manifest creation completed ==="
