/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>

#include <cstdint>

namespace cuopt::linear_programming {

// Forward declarations
template <typename i_t, typename f_t>
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

/**
 * @brief Map pdlp_solver_settings_t to protobuf PDLPSolverSettings message.
 *
 * Populates a protobuf message using the generated protobuf C++ API.
 * Does not perform serialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::PDLPSolverSettings* pb_settings);

/**
 * @brief Map protobuf PDLPSolverSettings message to pdlp_solver_settings_t.
 *
 * Reads from a protobuf message using the generated protobuf C++ API.
 * Does not perform deserialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_proto_to_pdlp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                                pdlp_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Map mip_solver_settings_t to protobuf MIPSolverSettings message.
 *
 * Populates a protobuf message using the generated protobuf C++ API.
 * Does not perform serialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_mip_settings_to_proto(const mip_solver_settings_t<i_t, f_t>& settings,
                               cuopt::remote::MIPSolverSettings* pb_settings);

/**
 * @brief Map protobuf MIPSolverSettings message to mip_solver_settings_t.
 *
 * Reads from a protobuf message using the generated protobuf C++ API.
 * Does not perform deserialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                               mip_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::linear_programming
