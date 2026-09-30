# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

if(NOT DEFINED PSLP_SOURCE_DIR OR NOT IS_DIRECTORY "${PSLP_SOURCE_DIR}")
    message(FATAL_ERROR "PSLP_SOURCE_DIR must name an existing PSLP source directory")
endif()

if(NOT DEFINED PSLP_PATCH_FILE OR NOT EXISTS "${PSLP_PATCH_FILE}")
    message(FATAL_ERROR "PSLP_PATCH_FILE must name an existing patch file")
endif()

find_package(Git REQUIRED)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --check -- "${PSLP_PATCH_FILE}"
    WORKING_DIRECTORY "${PSLP_SOURCE_DIR}"
    RESULT_VARIABLE patch_check_result
    ERROR_VARIABLE patch_check_error
)

if(patch_check_result STREQUAL "0")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply -- "${PSLP_PATCH_FILE}"
        WORKING_DIRECTORY "${PSLP_SOURCE_DIR}"
        RESULT_VARIABLE patch_result
        ERROR_VARIABLE patch_error
    )
    if(NOT patch_result STREQUAL "0")
        message(FATAL_ERROR "Failed to apply PSLP patch: ${patch_error}")
    endif()
    message(STATUS "Applied PSLP patch: ${PSLP_PATCH_FILE}")
else()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --reverse --check -- "${PSLP_PATCH_FILE}"
        WORKING_DIRECTORY "${PSLP_SOURCE_DIR}"
        RESULT_VARIABLE reverse_check_result
        ERROR_VARIABLE reverse_check_error
    )
    if(NOT reverse_check_result STREQUAL "0")
        message(FATAL_ERROR
            "PSLP patch neither applies cleanly nor is already applied.\n"
            "Forward check: ${patch_check_error}\n"
            "Reverse check: ${reverse_check_error}"
        )
    endif()
    message(STATUS "PSLP patch already applied: ${PSLP_PATCH_FILE}")
endif()
