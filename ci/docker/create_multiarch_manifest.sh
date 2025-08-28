#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    # Create the manifest
    echo "Creating multi-arch manifest..."
    docker manifest create --amend "$manifest_name" "$amd64_image" "$arm64_image"

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

# Create manifest for dockerhub and nvstaging
echo "=== Creating Docker Hub manifests ==="
create_manifest \
    "nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}" \
    "nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-amd64" \
    "nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-arm64"

echo "=== Creating NVCR staging manifests ==="
create_manifest \
    "nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}" \
    "nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-amd64" \
    "nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-arm64"

# Only create latest manifests for release builds
if [[ "${BUILD_TYPE}" == "release" ]]; then
    echo "=== Creating latest manifests for release build ==="

    echo "Creating Docker Hub latest manifest..."
    create_manifest \
        "nvidia/cuopt:latest-cuda${CUDA_SHORT}-py${PYTHON_SHORT}" \
        "nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-amd64" \
        "nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-arm64"

    echo "Creating NVCR staging latest manifest..."
    create_manifest \
        "nvcr.io/nvstaging/nvaie/cuopt:latest-cuda${CUDA_SHORT}-py${PYTHON_SHORT}" \
        "nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-amd64" \
        "nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_SHORT}-py${PYTHON_SHORT}-arm64"
else
    echo "Skipping latest manifest creation (BUILD_TYPE=${BUILD_TYPE}, not 'release')"
fi

echo "=== Multi-architecture manifest creation completed ==="
