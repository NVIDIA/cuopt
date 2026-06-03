# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(MPS_FAST_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/experimental_mps_fast/fast_parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/experimental_mps_fast/fast_parser_adapter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/experimental_mps_fast/file_reader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/experimental_mps_fast/lz4_file_reader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/experimental_mps_fast/mps_section_scanner.cpp
)

set(PARSERS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/data_model_view.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/file_to_string.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lp_parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_data_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/cython_parser.cpp
  ${MPS_FAST_SRC_FILES}
)

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES} ${PARSERS_SRC_FILES} PARENT_SCOPE)
set(MPS_FAST_SRC_FILES ${MPS_FAST_SRC_FILES} PARENT_SCOPE)
