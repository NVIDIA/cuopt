#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install Protobuf development libraries
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "rocky" ]]; then
        echo "Detected Rocky Linux. Installing Protobuf via dnf..."
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
