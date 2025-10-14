#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
