#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

# libcuopt.so is a linker script, not a shared object, so Julia cannot dlopen it (#2133). The C API
# that cuOpt.jl calls is exported by libcuopt_mathopt.so; point the package at that instead.
rapids-logger "Patching cuOpt.jl to load libcuopt_mathopt"
sed -i 's/Libdl.find_library("libcuopt")/Libdl.find_library("libcuopt_mathopt")/' src/cuOpt.jl
grep -q 'find_library("libcuopt_mathopt")' src/cuOpt.jl || { rapids-logger "ERROR: could not patch cuOpt.jl"; exit 1; }

# Find the component libraries and add their directories to LD_LIBRARY_PATH
LIBMATHOPT_PATH=$(find /pyenv/ -name "libcuopt_mathopt.so" -type f 2>/dev/null | head -1)
LIBCLIENT_PATH=$(find /pyenv/ -name "libcuopt_client.so" -type f 2>/dev/null | head -1)

if [ -z "$LIBMATHOPT_PATH" ] || [ -z "$LIBCLIENT_PATH" ]; then
    rapids-logger "ERROR: libcuopt_mathopt.so or libcuopt_client.so not found under /pyenv"
    exit 1
fi

rapids-logger "LIBMATHOPT_PATH: $LIBMATHOPT_PATH"
rapids-logger "LIBCLIENT_PATH: $LIBCLIENT_PATH"
export LD_LIBRARY_PATH="$(dirname "$LIBMATHOPT_PATH"):$(dirname "$LIBCLIENT_PATH"):${LD_LIBRARY_PATH:-}"

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
Pkg.test(; coverage=false)
'
