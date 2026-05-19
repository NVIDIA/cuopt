# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(PARSERS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/data_model_view.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_data_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/cython_mps_parser.cpp
)

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES} ${PARSERS_SRC_FILES} PARENT_SCOPE)
