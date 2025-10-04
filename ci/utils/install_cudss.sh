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

# Clean metadata & install cudss
if command -v dnf &> /dev/null; then
    # Adding static library just to please CMAKE requirements
    if [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -ge 13 ] && [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -lt 14 ]; then
        dnf -y install libcudss0-static-cuda-13 libcudss0-devel-cuda-13 libcudss0-cuda-13
    else
        dnf -y install libcudss0-static-cuda-12 libcudss0-devel-cuda-12 libcudss0-cuda-12
    fi
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y libcudss-devel
else
    echo "Neither dnf nor apt-get found. Cannot install cudss dependencies."
    exit 1
fi
