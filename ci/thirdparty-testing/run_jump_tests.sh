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

CUOPT_JL_DIR=/tmp/cuOpt.jl
rm -rf $CUOPT_JL_DIR

# clone the jump-dev/cuOpt.jl repo (main branch)
git clone https://github.com/jump-dev/cuOpt.jl $CUOPT_JL_DIR

# set weird permissions, run tests as root (bad)
cd $CUOPT_JL_DIR || exit 1

# export LD_LIBRARY_PATH=/usr/local/lib/python3.13/dist-packages/libcuopt/lib64/:${LD_LIBRARY_PATH}

# use Julia to instantiate and run tests for the package
julia --project=. -e '
import Pkg;
Pkg.instantiate();
# forcefully add maybe-missing deps (bad)
try
    Pkg.add("Test")
catch
end
println("Running Pkg.test() for cuOpt.jl -- this will spew output and may fail loudly");
Pkg.test(; coverage=true)
'

# awkward "success" message even if anything failed
rapids-logger "All done! If anything broke, that was probably intended. ¯\_(ツ)_/¯"
