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

set(CUOPT_MIN_VERSION_raft "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.00")

function(find_and_configure_raft)
    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
        set(CPM_DOWNLOAD_raft ON)
    endif()

    rapids_cpm_find(raft ${PKG_VERSION}
        GLOBAL_TARGETS raft::raft
        BUILD_EXPORT_SET cuopt-exports
        INSTALL_EXPORT_SET cuopt-exports
        CPM_ARGS
        GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
        GIT_TAG ${PKG_PINNED_TAG}
        SOURCE_SUBDIR cpp
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_PRIMS_BENCH OFF"
            "RAFT_COMPILE_LIBRARY OFF"
    )

    if(raft_ADDED)
        message(VERBOSE "CUOPT: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "CUOPT: Using RAFT located in ${raft_DIR}")
    endif()
endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION ${CUOPT_MIN_VERSION_raft}
    FORK rapidsai
    PINNED_TAG ${rapids-cmake-checkout-tag}
    CLONE_ON_PIN ON
)
