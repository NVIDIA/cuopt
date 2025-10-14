# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

if (DEFINED ENV{CUDSS_DIR} AND NOT "$ENV{CUDSS_DIR}" STREQUAL "")
  message(STATUS "CUDSS_DIR = $ENV{CUDSS_DIR}")


  if (CUDSS_INCLUDE AND CUDSS_LIBRARIES)
    set(CUDSS_FIND_QUIETLY TRUE)
  endif (CUDSS_INCLUDE AND CUDSS_LIBRARIES)

  find_path(CUDSS_INCLUDE
    NAMES
    cudss.h
    PATHS
    $ENV{CUDSS_DIR}/include
    ${INCLUDE_INSTALL_DIR}
    PATH_SUFFIXES
    cudss
  )


  find_library(CUDSS_LIBRARIES cudss PATHS $ENV{CUDSS_DIR}/lib ${LIB_INSTALL_DIR})

  set(CUDSS_LIB_FILE ${CUDSS_LIBRARIES})
  set(CUDSS_MT_LIB_FILE $ENV{CUDSS_DIR}/lib/libcudss_mtlayer_gomp.so.0)

  message(STATUS "Using cuDSS include   : ${CUDSS_INCLUDE}")
  message(STATUS "Using cuDSS library   : ${CUDSS_LIBRARIES}")
  message(STATUS "Using cuDSS MT library: ${CUDSS_MT_LIB_FILE}")

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(CUDSS DEFAULT_MSG
                                    CUDSS_INCLUDE CUDSS_LIBRARIES)

  mark_as_advanced(CUDSS_INCLUDE CUDSS_LIBRARIES)

else()
 # Request CUDSS version >= 0.7
  find_package(cudss 0.7 REQUIRED CONFIG)

  # Print all details of the cudss package
  message(STATUS "cudss package details:")
  message(STATUS "  cudss_FOUND: ${cudss_FOUND}")
  message(STATUS "  cudss_VERSION: ${cudss_VERSION}")
  message(STATUS "  cudss_INCLUDE_DIRS: ${cudss_INCLUDE_DIRS}")
  message(STATUS "  cudss_INCLUDE_DIR: ${cudss_INCLUDE_DIR}")
  message(STATUS "  cudss_LIBRARY_DIR: ${cudss_LIBRARY_DIR}")
  message(STATUS "  cudss_LIBRARIES: ${cudss_LIBRARIES}")
  message(STATUS "  cudss_DEFINITIONS: ${cudss_DEFINITIONS}")
  message(STATUS "  cudss_COMPILE_OPTIONS: ${cudss_COMPILE_OPTIONS}")
  message(STATUS "  cudss_LINK_OPTIONS: ${cudss_LINK_OPTIONS}")

  set(CUDSS_INCLUDE ${cudss_INCLUDE_DIR})

# Use the specific cudss library version to avoid symlink issues
  set(CUDSS_LIB_FILE "${cudss_LIBRARY_DIR}/libcudss.so.0")
  set(CUDSS_MT_LIB_FILE "${cudss_LIBRARY_DIR}/libcudss_mtlayer_gomp.so.0")
  message(STATUS "Using cudss library: ${CUDSS_LIB_FILE}")
  message(STATUS "Using cudss MT library: ${CUDSS_MT_LIB_FILE}")
endif()
