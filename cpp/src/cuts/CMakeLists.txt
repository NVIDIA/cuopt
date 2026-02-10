# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(CUTS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/cuts.cpp
  )

# Uncomment to enable debug info
#set_source_files_properties(${CUTS_SRC_FILES} DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTIES COMPILE_OPTIONS "-g1")

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${CUTS_SRC_FILES} PARENT_SCOPE)
