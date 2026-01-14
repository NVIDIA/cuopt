#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install Protobuf development libraries
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "rocky" ]]; then
        echo "Detected Rocky Linux. Installing Protobuf via dnf..."
        # Enable PowerTools (Rocky 8) or CRB (Rocky 9) repository for protobuf-devel
        if [[ "${VERSION_ID%%.*}" == "8" ]]; then
            dnf config-manager --set-enabled powertools || dnf config-manager --set-enabled PowerTools || true
        elif [[ "${VERSION_ID%%.*}" == "9" ]]; then
            dnf config-manager --set-enabled crb || true
        fi
        dnf install -y protobuf-devel protobuf-compiler
    elif [[ "$ID" == "ubuntu" ]]; then
        echo "Detected Ubuntu. Installing Protobuf via apt..."
        apt-get update
        apt-get install -y libprotobuf-dev protobuf-compiler
    else
        echo "Unknown OS: $ID. Please install Protobuf development libraries manually."
        exit 1
    fi
else
    echo "/etc/os-release not found. Cannot determine OS. Please install Protobuf development libraries manually."
    exit 1
fi
