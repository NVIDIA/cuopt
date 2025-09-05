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

# FindTBB.cmake - Find TBB (Threading Building Blocks) library
#
# This module defines the following variables:
#   TBB_FOUND        - True if TBB is found
#   TBB_INCLUDE_DIRS - TBB include directories
#   TBB_LIBRARIES    - TBB libraries
#   TBB::tbb         - Imported target for TBB

# Try pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_TBB QUIET tbb)
endif()

find_path(TBB_INCLUDE_DIR
  NAMES tbb/tbb.h
  PATHS
    ${PC_TBB_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    /opt/intel/tbb/include
    /opt/intel/oneapi/tbb/latest/include
)

find_library(TBB_LIBRARY
  NAMES tbb
  PATHS
    ${PC_TBB_LIBRARY_DIRS}
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/intel/tbb/lib
    /opt/intel/oneapi/tbb/latest/lib
)

find_library(TBB_MALLOC_LIBRARY
  NAMES tbbmalloc
  PATHS
    ${PC_TBB_LIBRARY_DIRS}
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/intel/tbb/lib
    /opt/intel/oneapi/tbb/latest/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB
  REQUIRED_VARS TBB_INCLUDE_DIR TBB_LIBRARY
)

if(TBB_FOUND AND NOT TARGET TBB::tbb)
  add_library(TBB::tbb UNKNOWN IMPORTED)
  set_target_properties(TBB::tbb PROPERTIES
    IMPORTED_LOCATION "${TBB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}"
  )

  if(TBB_MALLOC_LIBRARY)
    set_target_properties(TBB::tbb PROPERTIES
      INTERFACE_LINK_LIBRARIES "${TBB_MALLOC_LIBRARY}"
    )
  endif()

  # Add compile definitions from pkg-config if available
  if(PC_TBB_CFLAGS_OTHER)
    set_target_properties(TBB::tbb PROPERTIES
      INTERFACE_COMPILE_OPTIONS "${PC_TBB_CFLAGS_OTHER}"
    )
  endif()
endif()

mark_as_advanced(TBB_INCLUDE_DIR TBB_LIBRARY TBB_MALLOC_LIBRARY)
