# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Header-only JSON parser, used by the routing gRPC test driver to read problem
# files. Uses rapids_cpm_find so an existing install that ships a CMake config
# package (conda, system) is reused, and the pinned source is fetched via CPM
# otherwise. Going through CPM also populates the CPM cache, so any later use of
# nlohmann_json in the project resolves without repeating the search/download.
#
# NOTE: nlohmann_json is not part of rapids-cmake's version file, so the version
# is pinned here rather than by rapids_cpm_<pkg>. The pin matches cuVS
# (rapidsai/cuvs cpp/cmake/thirdparty/get_nlohmann_json.cmake) and the
# nlohmann_json conda package, so the conda copy is reused rather than fetched.
#
# Test-only: not added to cuopt's BUILD/INSTALL export sets, since nothing that
# is installed links it.
function(find_and_configure_nlohmann_json)
    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    rapids_cpm_find(nlohmann_json ${PKG_VERSION}
            GLOBAL_TARGETS nlohmann_json::nlohmann_json
            CPM_ARGS
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL
            OPTIONS
            "JSON_BuildTests OFF"
            "JSON_Install OFF"
    )
endfunction()

find_and_configure_nlohmann_json(VERSION 3.12.0 PINNED_TAG v3.12.0)
