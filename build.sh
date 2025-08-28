#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)
LIBCUOPT_BUILD_DIR=${LIBCUOPT_BUILD_DIR:=${REPODIR}/cpp/build}
LIBMPS_PARSER_BUILD_DIR=${LIBMPS_PARSER_BUILD_DIR:=${REPODIR}/cpp/libmps_parser/build}

VALIDARGS="clean libcuopt libmps_parser cuopt_mps_parser cuopt cuopt_server cuopt_sh_client docs deb -a -b -g -v -l= --verbose-pdlp  [--cmake-args=\\\"<args>\\\"] [--cache-tool=<tool>] -n --allgpuarch --ci-only-arch --show_depr_warn -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   libcuopt         - build the cuopt C++ code
   libmps_parser    - build the libmps_parser C++ code
   cuopt_mps_parser - build the cuopt_mps_parser python package
   cuopt            - build the cuopt Python package
   cuopt_server     - build the cuopt_server Python package
   cuopt_sh_client  - build cuopt self host client
   docs             - build the docs
   deb              - build deb package (requires libcuopt to be built first)
 and <flag> is:
   -v               - verbose build mode
   -g               - build for debug
   -a               - Enable assertion (by default in debug mode)
   -b               - Build with benchmark settings
   -n               - no install step
   -l=              - log level. Options are: TRACE | DEBUG | INFO | WARN | ERROR | CRITICAL | OFF. Default=INFO
   --verbose-pdlp   - verbose mode for pdlp solver
   --cache-tool=<tool> - pass the build cache tool (eg: ccache, sccache, distcc) that will be used
                      to speedup the build process.
   --cmake-args=\\\"<args>\\\"   - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   --allgpuarch     - build for all supported GPU architectures
   --ci-only-arch   - build for volta and ampere only
   --show_depr_warn - show cmake deprecation warnings
   -h               - print this text

 default action (no args) is to build and install 'libcuopt' then 'cuopt' then 'docs' targets

 libcuopt build dir is: ${LIBCUOPT_BUILD_DIR}

 Set env var LIBCUOPT_BUILD_DIR to override libcuopt build dir.
"
CUOPT_MPS_PARSER_BUILD_DIR=${REPODIR}/python/cuopt/cuopt/linear_programming/build
PY_LIBCUOPT_BUILD_DIR=${REPODIR}/python/libcuopt/build
CUOPT_BUILD_DIR=${REPODIR}/python/cuopt/build
CUOPT_SERVER_BUILD_DIR=${REPODIR}/python/cuopt_server/build
CUOPT_SH_CLIENT_BUILD_DIR=${REPODIR}/python/cuopt_self_hosted/build
DOCS_BUILD_DIR=${REPODIR}/docs/cuopt/build
BUILD_DIRS="${LIBCUOPT_BUILD_DIR} ${LIBMPS_PARSER_BUILD_DIR} ${CUOPT_BUILD_DIR} ${CUOPT_SERVER_BUILD_DIR} ${CUOPT_SERVICE_CLIENT_BUILD_DIR} ${CUOPT_SH_CLIENT_BUILD_DIR} ${CUOPT_MPS_PARSER_BUILD_DIR} ${PY_LIBCUOPT_BUILD_DIR} ${DOCS_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
DEFINE_ASSERT=False
DEFINE_PDLP_VERBOSE_MODE=False
INSTALL_TARGET=install
BUILD_DISABLE_DEPRECATION_WARNING=ON
BUILD_ALL_GPU_ARCH=0
BUILD_CI_ONLY=0
CACHE_ARGS=()
PYTHON_ARGS_FOR_INSTALL=("-m" "pip" "install" "--no-build-isolation" "--no-deps")
LOGGING_ACTIVE_LEVEL="INFO"
FETCH_RAPIDS=ON

# Set defaults for vars that may not have been defined externally
#  FIXME: if PREFIX is not set, check CONDA_PREFIX, but there is no fallback
#  from there!
INSTALL_PREFIX=${PREFIX:=${CONDA_PREFIX}}
BUILD_ABI=${BUILD_ABI:=ON}

export CMAKE_GENERATOR=Ninja

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function buildAll {
    (( NUMARGS == 0 )) || ! (echo " ${ARGS} " | grep -q " [^-]\+ ")
}

function cacheTool {
    # Check for multiple cache options
    if [[ $(echo "$ARGS" | { grep -Eo "\-\-cache\-tool" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cache-tool options were provided, please provide only one: ${ARGS}"
        exit 1
    fi
    # Check for cache tool option
    if [[ -n $(echo "$ARGS" | { grep -E "\-\-cache\-tool" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        CACHE_TOOL=$(echo "$ARGS" | sed -e 's/.*--cache-tool=//' -e 's/ .*//')
        if [[ -n ${CACHE_TOOL} ]]; then
            # Remove the full CACHE_TOOL argument from list of args so that it passes validArgs function
            ARGS=${ARGS//--cache-tool=$CACHE_TOOL/}
            CACHE_ARGS=("-DCMAKE_CUDA_COMPILER_LAUNCHER=${CACHE_TOOL}" "-DCMAKE_C_COMPILER_LAUNCHER=${CACHE_TOOL}" "-DCMAKE_CXX_COMPILER_LAUNCHER=${CACHE_TOOL}")
        fi
    fi
}

function loggingArgs {
    if [[ $(echo "$ARGS" | { grep -Eo "\-l" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple -l logging options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    LOG_LEVEL_LIST=("TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")

    # Check for logging option
    if [[ -n $(echo "$ARGS" | { grep -E "\-l" || true; } ) ]]; then
        LOGGING_ARGS=$(echo "$ARGS" | { grep -Eo "\-l=\S+" || true; })
        if [[ -n ${LOGGING_ARGS} ]]; then
            # Remove the full log argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$LOGGING_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            LOGGING_ARGS=$(echo "$LOGGING_ARGS" | sed -e 's/^"//' -e 's/"$//' | cut -c4- | grep -Eo "\S+" | tr '[:lower:]' '[:upper:]')
            if [[ "${LOG_LEVEL_LIST[*]}" =~ $LOGGING_ARGS ]]; then
                LOGGING_ACTIVE_LEVEL=$LOGGING_ARGS
            else
                echo "Invalid logging arg $LOGGING_ARGS, expected any of ${LOG_LEVEL_LIST[*]}"
                exit 1
            fi
        fi
    fi
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo "$ARGS" | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo "$ARGS" | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo "$ARGS" | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo "$EXTRA_CMAKE_ARGS" | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi

    read -ra EXTRA_CMAKE_ARGS <<< "$EXTRA_CMAKE_ARGS"
}


if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    cacheTool
    cmakeArgs
    loggingArgs
    for a in ${ARGS}; do
        if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG="-v"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
    DEFINE_ASSERT=true
fi
if hasArg -a; then
    DEFINE_ASSERT=true
fi
if hasArg -b; then
    DEFINE_BENCHMARK=true
fi
if hasArg --verbose-pdlp; then
    DEFINE_PDLP_VERBOSE_MODE=true
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --ci-only-arch; then
    BUILD_CI_ONLY=1
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNING=OFF
fi

function contains_string {
    local search_string="$1"
    shift
    local array=("$@")

    for element in "${array[@]}"; do
        if [[ "$element" == *"$search_string"* ]]; then
            return 0
        fi
    done

    return 1
}

# Append `-DFIND_CUOPT_CPP=ON` to CMAKE_ARGS unless a user specified the option.
if ! contains_string "DFIND_CUOPT_CPP" "${EXTRA_CMAKE_ARGS[@]}"; then
    EXTRA_CMAKE_ARGS+=("-DFIND_CUOPT_CPP=ON")
fi

if ! contains_string "DFIND_MPS_PARSER_CPP" "${EXTRA_CMAKE_ARGS[@]}"; then
    EXTRA_CMAKE_ARGS+=("-DFIND_MPS_PARSER_CPP=ON")
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done

    # Cleaning up python artifacts
    find "${REPODIR}"/python/ | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild$)"  | xargs rm -rf

fi

if [ ${BUILD_CI_ONLY} -eq 1 ] && [ ${BUILD_ALL_GPU_ARCH} -eq 1 ]; then
    echo "Options --ci-only-arch and --allgpuarch can not be used simultaneously"
    exit 1
fi

if  [ ${BUILD_ALL_GPU_ARCH} -eq 1 ]; then
    CUOPT_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
    echo "Building for *ALL* supported GPU architectures..."
else
    if [ ${BUILD_CI_ONLY} -eq 1 ]; then
        CUOPT_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for RAPIDS supported architectures..."
    else
        CUOPT_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    fi
fi

################################################################################
# Configure, build, and install libmps_parser
if buildAll || hasArg libmps_parser; then
    mkdir -p "${LIBMPS_PARSER_BUILD_DIR}"
    cd "${LIBMPS_PARSER_BUILD_DIR}"
    cmake -DDEFINE_ASSERT=${DEFINE_ASSERT} \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          "${CACHE_ARGS[@]}" \
          "${REPODIR}"/cpp/libmps_parser/

    if hasArg -n; then
        cmake --build "${LIBMPS_PARSER_BUILD_DIR}" ${VERBOSE_FLAG}
    else
        cmake --build "${LIBMPS_PARSER_BUILD_DIR}" --target ${INSTALL_TARGET} ${VERBOSE_FLAG}
    fi
fi

################################################################################
# Configure, build, and install libcuopt
if buildAll || hasArg libcuopt; then
    mkdir -p "${LIBCUOPT_BUILD_DIR}"
    cd "${LIBCUOPT_BUILD_DIR}"
    cmake -DDEFINE_ASSERT=${DEFINE_ASSERT} \
          -DDEFINE_BENCHMARK="${DEFINE_BENCHMARK}" \
          -DDEFINE_PDLP_VERBOSE_MODE=${DEFINE_PDLP_VERBOSE_MODE} \
          -DLIBCUOPT_LOGGING_LEVEL="${LOGGING_ACTIVE_LEVEL}" \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          -DCMAKE_CUDA_ARCHITECTURES=${CUOPT_CMAKE_CUDA_ARCHITECTURES} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DFETCH_RAPIDS=${FETCH_RAPIDS} \
          "${EXTRA_CMAKE_ARGS[@]}" \
          "${REPODIR}"/cpp
    if hasArg -n; then
        cmake --build "${LIBCUOPT_BUILD_DIR}" ${VERBOSE_FLAG}
    else
        cmake --build "${LIBCUOPT_BUILD_DIR}" --target ${INSTALL_TARGET} ${VERBOSE_FLAG} -j"${PARALLEL_LEVEL}"
    fi
fi

################################################################################
# Build deb package
if hasArg deb; then
    # Check if libcuopt has been built
    if [ ! -d "${LIBCUOPT_BUILD_DIR}" ]; then
        echo "Error: libcuopt must be built before creating deb package. Run with 'libcuopt' target first."
        exit 1
    fi

    echo "Building deb package..."
    cd "${LIBCUOPT_BUILD_DIR}"
    cpack -G DEB
    echo "Deb package created in ${LIBCUOPT_BUILD_DIR}"
fi


# Build and install the cuopt Python package
if buildAll || hasArg cuopt; then
    cd "${REPODIR}"/python/cuopt

    # $EXTRA_CMAKE_ARGS gets concatenated into a string with [*] and then we find/replace spaces with semi-colons
    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${LIBCUOPT_BUILD_DIR};-DCMAKE_CUDA_ARCHITECTURES=${CUOPT_CMAKE_CUDA_ARCHITECTURES};${EXTRA_CMAKE_ARGS[*]// /;}" \
        python "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi

# Build and install the cuopt MPS parser Python package
if buildAll || hasArg cuopt_mps_parser; then
    cd "${REPODIR}"/python/cuopt/cuopt/linear_programming

    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${LIBCUOPT_BUILD_DIR};-DCMAKE_CUDA_ARCHITECTURES=${CUOPT_CMAKE_CUDA_ARCHITECTURES};${EXTRA_CMAKE_ARGS[*]// /;}" \
        python "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi

# Build and install the cuopt_server Python package
if buildAll || hasArg cuopt_server; then
    cd "${REPODIR}"/python/cuopt_server
    python "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi

# Build and install the cuopt_sh_client Python package
if buildAll || hasArg cuopt_sh_client; then
    cd "${REPODIR}"/python/cuopt_self_hosted/
    python "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi

# Build the docs
if buildAll || hasArg docs; then
    cd "${REPODIR}"/cpp/doxygen
    doxygen Doxyfile

    cd "${REPODIR}"/docs/cuopt
    make clean
    make html linkcheck
fi
