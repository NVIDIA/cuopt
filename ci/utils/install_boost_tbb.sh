#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install Boost and TBB
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "rocky" ]]; then
        echo "Detected Rocky Linux. Installing Boost and TBB via dnf..."
        dnf install -y epel-release
        dnf install -y boost1.78-devel tbb-devel
        if [[ "$(uname -m)" == "x86_64" ]]; then
            dnf install -y gcc-toolset-14-libquadmath-devel
        fi
    elif [[ "$ID" == "ubuntu" ]]; then
        echo "Detected Ubuntu. Installing Boost and TBB via apt..."
        apt-get update
        apt-get install -y libboost-dev libtbb-dev
    else
        echo "Unknown OS: $ID. Please install Boost development libraries manually."
        exit 1
    fi
else
    echo "/etc/os-release not found. Cannot determine OS. Please install Boost development libraries manually."
    exit 1
fi
