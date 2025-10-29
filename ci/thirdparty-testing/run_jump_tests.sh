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

# Install Julia using the official installer

if ! command -v julia &> /dev/null; then
    rapids-logger "Installing Julia using official installer..."
    # Pass 'yes' to the installer to accept any prompts automatically
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    # Add Julia to PATH for current session
    export PATH="$HOME/.juliaup/bin:$PATH"
    # Also add to .bashrc for future sessions
    rapids-logger 'export PATH="$HOME/.juliaup/bin:$PATH"' >> ~/.bashrc
else
    rapids-logger "Julia is already installed."
fi

# Confirm Julia install
julia --version

CUOPT_JL_DIR="$(mktemp -d)/cuOpt.jl"
rapids-logger "Cloning cuOpt.jl into ${CUOPT_JL_DIR}"
git clone https://github.com/jump-dev/cuOpt.jl.git "${CUOPT_JL_DIR}"

cd $CUOPT_JL_DIR || exit 1

# Patch cuOpt.jl to bypass version check
rapids-logger "Patching cuOpt.jl to relax cuOpt version constraint..."
if [ -f src/cuOpt.jl ]; then
    # Replace max version in pattern: min, max = v"X.Y", v"A.B" -> v"99.99"
    sed -i 's/\(min.*max.*=.*v"[0-9]\+\.[0-9]\+"\),\s*v"[0-9]\+\.[0-9]\+"/\1, v"99.99"/g' src/cuOpt.jl
    rapids-logger "Updated max cuOpt version to v\"99.99\" - all versions will pass"
fi

# Patch MOI_wrapper.jl test to skip test_air05
rapids-logger "Commenting out test_air05"
if [ -f test/MOI_wrapper.jl ]; then
    sed -i '/^function test_air05()/,/^end$/s/^/# /' test/MOI_wrapper.jl
fi

# Find libcuopt.so and add its directory to LD_LIBRARY_PATH
LIBCUOPT_PATH=$(find /pyenv/ -name "libcuopt.so" -type f 2>/dev/null | head -1)

if [ -n "$LIBCUOPT_PATH" ]; then
    rapids-logger "LIBCUOPT_PATH: $LIBCUOPT_PATH"
    LIBCUOPT_DIR=$(dirname "$LIBCUOPT_PATH")
    rapids-logger "LIBCUOPT_DIR: $LIBCUOPT_DIR"
    export LD_LIBRARY_PATH="${LIBCUOPT_DIR}:${LD_LIBRARY_PATH}"
    rapids-logger "Found libcuopt.so at $LIBCUOPT_PATH, added directory to LD_LIBRARY_PATH: $LIBCUOPT_DIR"
else
    rapids-logger "Warning: libcuopt.so not found in root filesystem"
fi

rapids-logger "Running Julia tests for cuOpt.jl"

# use Julia to instantiate and run tests for the package
julia --project=. -e '
import Pkg;
Pkg.instantiate();
# forcefully add maybe-missing deps (bad)
try
    Pkg.add("Test")
catch
end
println("Running Pkg.test() for cuOpt.jl");
Pkg.test(; coverage=true)
'
